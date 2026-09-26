#!/bin/bash
set -euo pipefail

# ---- Self-heal a retained-script rec bundle if it's missing ----------------
# Model selection itself happens in the binary (resolve_model: OCR_MODEL,
# with OCR_LANG as a deprecated alias). The Dockerfile bakes every bundle at
# build time via fetch_release_models.sh (flat /app/models/{det,rec,cls}.onnx
# + keys.txt for the v6 default tiers; /app/models/rec/<script>/ for the
# retained PP-OCRv5 recognizers). So in a normal deployment the `[[ ! -f ]]`
# branch below never fires — it exists as a self-heal path in case
# /app/models/rec gets mounted over with an empty volume.
#
# Only the nested PP-OCRv5 scripts are fetchable here; the v6 tiers
# (medium/small/tiny) are flat siblings baked into the image. An OCR_MODEL of
# medium/small/tiny needs no per-script fetch, so it falls through untouched.
RETAINED_SCRIPTS="arabic eslav greek korean thai"

SELECTED_SCRIPT="${OCR_MODEL:-${OCR_LANG:-}}"
if [[ -n "${SELECTED_SCRIPT}" ]] && grep -qw "${SELECTED_SCRIPT}" <<<"${RETAINED_SCRIPTS}"; then
  REC_ONNX="/app/models/rec/${SELECTED_SCRIPT}/rec.onnx"
  if [[ ! -f "${REC_ONNX}" ]]; then
    echo "[entrypoint] ${SELECTED_SCRIPT} rec bundle missing, fetching…"
    bash /app/scripts/download_models.sh --lang "${SELECTED_SCRIPT}"
    # chown only when we own uid 0 — the Dockerfile installs the ocr user
    # and today the base image runs as root, but this stays correct if that
    # ever changes.
    if [[ $EUID -eq 0 ]]; then
      chown -R ocr:ocr /app/models
    fi
  else
    echo "[entrypoint] ${SELECTED_SCRIPT} rec bundle already present, skipping download"
  fi
fi

# Render nginx config from template — substitutes ${MAX_BODY_MB} so the
# proxy and the C++ servers (which both read MAX_BODY_MB at startup) agree
# on the body cap (default 100 to match historical behaviour), and the
# proxy timeouts derived from IDLE_CONNECTION_TIMEOUT_S below.
export MAX_BODY_MB="${MAX_BODY_MB:-100}"
# Validate up front: matches the C++ env_int(..., 1, 102400) range so the
# nginx config rendered here and the Drogon/gRPC limits inside the
# server agree on the same accepted values.
#  - reject leading zeros / "0"     (nginx interprets `0m` as unlimited)
#  - reject empty / non-numeric     (nginx fails 90s into startup with a confusing parse error)
#  - reject anything > 102400 MB    (matches env_int upper bound)
if ! [[ "$MAX_BODY_MB" =~ ^[1-9][0-9]*$ ]] || (( MAX_BODY_MB > 102400 )); then
  echo "[entrypoint] FATAL: MAX_BODY_MB must be a positive integer in [1, 102400] (got: '$MAX_BODY_MB')" >&2
  exit 1
fi

# The server closes a connection silent for IDLE_CONNECTION_TIMEOUT_S seconds
# (0 = never), a request still being processed included, so that value is how
# long one request may run. nginx has to wait as long for the response: its
# proxy timeouts get the value less a second (so an overrun answers 504 Backend
# timeout rather than the 502 of a dropped connection), 24 h -- the server's
# upper bound -- for 0. Its idle upstream keep-alive connections close at half
# the value, at most nginx's default 60 s, so it never reuses one the server is
# about to drop. Same range check as the server.
export IDLE_CONNECTION_TIMEOUT_S="${IDLE_CONNECTION_TIMEOUT_S:-120}"
if ! [[ "$IDLE_CONNECTION_TIMEOUT_S" =~ ^(0|[1-9][0-9]*)$ ]] || (( IDLE_CONNECTION_TIMEOUT_S > 86400 )); then
  echo "[entrypoint] FATAL: IDLE_CONNECTION_TIMEOUT_S must be an integer in [0, 86400] (got: '$IDLE_CONNECTION_TIMEOUT_S')" >&2
  exit 1
fi
if (( IDLE_CONNECTION_TIMEOUT_S == 0 )); then
  export PROXY_TIMEOUT_S=86400 UPSTREAM_KEEPALIVE_S=60
else
  export PROXY_TIMEOUT_S=$(( IDLE_CONNECTION_TIMEOUT_S > 1 ? IDLE_CONNECTION_TIMEOUT_S - 1 : 1 ))
  UPSTREAM_KEEPALIVE_S=$(( IDLE_CONNECTION_TIMEOUT_S / 2 ))
  export UPSTREAM_KEEPALIVE_S=$(( UPSTREAM_KEEPALIVE_S < 1 ? 1 : (UPSTREAM_KEEPALIVE_S > 60 ? 60 : UPSTREAM_KEEPALIVE_S) ))
fi

# ---- Preflight: TRT engine cache must be writable -------------------------
# Mirrors get_engine_cache_dir() in src/engine/onnx_to_trt.cpp:
#   $TRT_ENGINE_CACHE → $HOME/.cache/turbo-ocr → /tmp/turbo-ocr-engines
# A read-only volume mount here makes the first request crash with no clear
# signal; fail fast at startup with an actionable message instead.
if [[ -n "${TRT_ENGINE_CACHE:-}" ]]; then
  TRT_CACHE_DIR="${TRT_ENGINE_CACHE}"
else
  # Mirror src/engine/onnx_to_trt.cpp::get_engine_cache_dir(), but
  # resolve $HOME the way the BINARY will see it after gosu drops to
  # ocr — the entrypoint itself is running as root with HOME=/root,
  # which would point at a path the binary will never touch and that
  # ocr can't write to.
  OCR_HOME=$(gosu ocr bash -c 'printf %s "${HOME:-}"' 2>/dev/null || true)
  if [[ -n "$OCR_HOME" ]]; then
    TRT_CACHE_DIR="${OCR_HOME}/.cache/turbo-ocr"
  else
    TRT_CACHE_DIR="/tmp/turbo-ocr-engines"
  fi
fi
mkdir -p "${TRT_CACHE_DIR}" 2>/dev/null || true

# We're still root here (gosu drops privileges further down). Bind-mounted
# host volumes inherit the host uid/gid (commonly 1000), but the ocr user
# inside the container is uid 1001 — so without this chown the binary's
# std::ofstream silently fails with EACCES when writing engine files. The
# only externally visible symptom is "[TRT] Failed to build engine from:
# <onnx>" with no diagnostic, even though TRT actually built the engine.
# Cheap when the dir is already correctly owned, critical when it isn't.
if [[ $EUID -eq 0 ]]; then
  chown -R ocr:ocr "${TRT_CACHE_DIR}" 2>/dev/null || true
fi

# Probe writability AS THE ocr USER — the previous probe ran as root and
# always passed even when the actual binary couldn't write a byte.
TRT_CACHE_SENTINEL="${TRT_CACHE_DIR}/.entrypoint_writecheck.$$"
if ! gosu ocr bash -c ": > '${TRT_CACHE_SENTINEL}'" 2>/dev/null; then
  echo "[entrypoint] FATAL: TRT engine cache directory '${TRT_CACHE_DIR}' is not writable by the ocr user." >&2
  echo "[entrypoint]        Even after chown -R ocr:ocr, the directory cannot be written. Likely a read-only mount." >&2
  exit 1
fi
rm -f "${TRT_CACHE_SENTINEL}"

NGINX_CONF=/tmp/nginx.conf
envsubst '${MAX_BODY_MB} ${PROXY_TIMEOUT_S} ${UPSTREAM_KEEPALIVE_S}' < /app/docker/nginx.conf.template > "$NGINX_CONF"

# Start nginx reverse proxy (absorbs connection storms, keep-alive to Drogon)
nginx -c "$NGINX_CONF"

# TRT's engine builder dlopen()s libnvinfer_builder_resource*.so by bare name
# at first-startup engine build. The nvcr base ldconfig's the core TRT libs,
# but the builder-resource lib is resolved via the loader search path — so the
# TRT lib dir must be on LD_LIBRARY_PATH or the build aborts with an opaque
# "Failed to build engine" and no diagnostic. Prepend whichever TRT lib dirs
# exist; absent on the CPU-only image, where this is a harmless no-op.
for _trt_lib in /usr/local/tensorrt/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu; do
  if [[ -d "$_trt_lib" ]]; then
    LD_LIBRARY_PATH="${_trt_lib}:${LD_LIBRARY_PATH:-}"
  fi
done
export LD_LIBRARY_PATH

# Drop to non-root user and run the OCR server.
# TRT engines are auto-built from ONNX on first startup (cached by TRT version + model hash).
#
# nginx runs as a detached daemon and so never sees the container's stop
# signal. We can't `exec` the server (that would replace this shell and a trap
# could never fire), so run it in the background and relay the signal: the
# trap forwards SIGTERM/SIGINT to the server so it drives its own HTTP/gRPC
# drain first; we then `nginx -s quit` to drain the proxy AFTER the backend is
# down so no in-flight proxied request is cut. gosu execs the binary, so $! is
# the server's own PID — it stays the effective signal target for k8s.
gosu ocr "$@" &
server_pid=$!

trap 'kill -s TERM "$server_pid" 2>/dev/null || true' TERM
trap 'kill -s INT  "$server_pid" 2>/dev/null || true' INT

# `wait` is interrupted (returns 128+signo) when a trapped signal arrives; loop
# until the server has actually exited, then keep its real status for k8s.
# `|| true` so `set -e` doesn't abort on the server's non-zero exit.
status=0
while true; do
  wait "$server_pid" && { status=0; break; } || status=$?
  (( status > 128 )) && continue   # signal-interrupted wait; the server is still draining
  break                            # server exited with $status
done

nginx -s quit 2>/dev/null || true
exit "$status"
