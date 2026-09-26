"""Integration tests for ?words=1 (per-line word boxes)."""

import base64
import io

import pytest
import requests
from PIL import Image, ImageDraw

from conftest import _get_font

WORDS = ["Invoice", "total:", "1,250.00", "EUR", "(paid)."]


def _ink_box(word, xy, font, size):
    """Bounding box of the pixels one word actually inks (not its advance)."""
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).text(xy, word, fill=255, font=font)
    return mask.point(lambda v: 255 if v >= 128 else 0).getbbox()


def _render_line():
    """A line whose words sit at known places; returns (png, jpeg, ink boxes)."""
    font = _get_font(40)
    img = Image.new("RGB", (1100, 140), "white")
    draw = ImageDraw.Draw(img)
    x, y = 40, 45
    boxes = {}
    for w in WORDS:
        draw.text((x, y), w, fill="black", font=font)
        l, t, r, b = _ink_box(w, (x, y), font, img.size)
        boxes[w] = (l, t, r - 1, b - 1)
        x = draw.textbbox((x, y), w, font=font)[2] + 22
    png, jpg = io.BytesIO(), io.BytesIO()
    img.save(png, format="PNG")
    img.save(jpg, format="JPEG", quality=95)
    return png.getvalue(), jpg.getvalue(), boxes


def _aabb(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def _check_words(data, ink):
    results = data["results"]
    assert results, "no text recognized"
    seen = 0
    for item in results:
        words = item.get("words")
        assert words, f"line {item['text']!r} has no words"
        assert " ".join(w["text"] for w in words) == " ".join(item["text"].split())
        lx0, ly0, lx1, ly1 = _aabb(item["bounding_box"])
        for w in words:
            assert 0.0 < w["confidence"] <= 1.0
            x0, y0, x1, y1 = _aabb(w["bounding_box"])
            # Inside its line box, or past it by at most a third of the line
            # height where the box clips a glyph (a descender).
            m = (ly1 - ly0) // 3 + 1
            assert lx0 - m <= x0 < x1 <= lx1 + m and ly0 - m <= y0 < y1 <= ly1 + m
            if w["text"] in ink:
                seen += 1
                gx0, gy0, gx1, gy1 = ink[w["text"]]
                # Never cuts the word's ink (anti-aliased rim aside), hugs it.
                assert x0 <= gx0 + 1 and x1 >= gx1 - 1, (w, ink[w["text"]])
                assert gx0 - x0 <= 4 and x1 - gx1 <= 4, (w, ink[w["text"]])
    assert seen >= 3, f"too few words read back exactly: {results}"


def test_words_raw_png(server_url):
    png, _, ink = _render_line()
    r = requests.post(f"{server_url}/ocr/raw?words=1", data=png, timeout=30)
    assert r.status_code == 200, r.text
    _check_words(r.json(), ink)


def test_words_raw_jpeg(server_url):
    """JPEG takes the device-decode path on GPU builds."""
    _, jpg, ink = _render_line()
    r = requests.post(f"{server_url}/ocr/raw?words=1", data=jpg, timeout=30)
    assert r.status_code == 200, r.text
    _check_words(r.json(), ink)


def test_words_base64_route(server_url):
    png, _, ink = _render_line()
    body = {"image": base64.b64encode(png).decode("ascii")}
    r = requests.post(f"{server_url}/ocr?words=1", json=body, timeout=30)
    assert r.status_code == 200, r.text
    _check_words(r.json(), ink)


def test_no_words_without_the_flag(server_url):
    png, _, _ = _render_line()
    r = requests.post(f"{server_url}/ocr/raw", data=png, timeout=30)
    assert r.status_code == 200
    assert r.json()["results"]
    assert all("words" not in item for item in r.json()["results"])


def test_words_with_text_off_is_rejected(server_url):
    png, _, _ = _render_line()
    r = requests.post(f"{server_url}/ocr/raw?words=1&text=0&layout=1", data=png,
                      timeout=30)
    assert r.status_code == 400


def test_words_pdf_text_layer(server_url):
    """Text-layer lines (mode=geometric) take their words from the PDF."""
    reportlab = pytest.importorskip("reportlab")
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica", 18)
    c.drawString(72, 700, "Native text layer words.")
    c.save()
    r = requests.post(f"{server_url}/ocr/pdf?mode=geometric&words=1",
                      data=buf.getvalue(), timeout=60)
    assert r.status_code == 200, r.text
    page = r.json()["pages"][0]
    lines = [it for it in page["results"] if "Native" in it["text"]]
    assert lines, page
    words = lines[0]["words"]
    assert [w["text"] for w in words] == ["Native", "text", "layer", "words."]
    lx0, ly0, lx1, ly1 = _aabb(lines[0]["bounding_box"])
    prev_x1 = None
    for w in words:
        x0, y0, x1, y1 = _aabb(w["bounding_box"])
        assert lx0 - 1 <= x0 < x1 <= lx1 + 1
        if prev_x1 is not None:
            assert x0 > prev_x1  # left to right, no overlap
        prev_x1 = x1


def test_words_batch_matches_single(server_url):
    """/ocr/batch runs the batched recognizer; its words match /ocr/raw's."""
    png, jpg, ink = _render_line()
    b64 = base64.b64encode(png).decode("ascii")
    r = requests.post(f"{server_url}/ocr/batch?words=1",
                      json={"images": [b64, b64, b64]}, timeout=60)
    assert r.status_code == 200, r.text
    batch = r.json()["batch_results"]
    assert len(batch) == 3
    single = requests.post(f"{server_url}/ocr/raw?words=1", data=png,
                           timeout=30).json()
    want = [[(w["text"], w["bounding_box"]) for w in it["words"]]
            for it in single["results"]]
    for page in batch:
        _check_words(page, ink)
        got = [[(w["text"], w["bounding_box"]) for w in it["words"]]
               for it in page["results"]]
        assert got == want


def _glyphless_pdf(text, x_pt=72, baseline_pt=700, size_pt=20):
    """A one-page PDF whose only text is an invisible layer in a glyphless
    font -- empty glyphs, FontBBox [0 0 0 0] -- the way OCR tools write the
    text layer of a scanned page."""
    procs = " ".join(f"/g{c} 9 0 R" for c in range(32, 127))
    diffs = " ".join(f"/g{c}" for c in range(32, 127))
    widths = " ".join("500" for _ in range(32, 127))
    content = (f"BT /F1 {size_pt} Tf 3 Tr {x_pt} {baseline_pt} Td "
               f"({text}) Tj ET").encode()
    cmap = (b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap "
            b"/CMapName /A def 1 begincodespacerange <00> <FF> endcodespacerange "
            b"1 beginbfrange <20> <7E> <0020> endbfrange endcmap "
            b"CMapName currentdict /CMap defineresource pop end end")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        (f"<< /Type /Font /Subtype /Type3 /FontBBox [0 0 0 0] "
         f"/FontMatrix [0.001 0 0 0.001 0 0] /CharProcs 6 0 R /Encoding 7 0 R "
         f"/FirstChar 32 /LastChar 126 /Widths [{widths}] /ToUnicode 8 0 R "
         f"/Resources << >> >>").encode(),
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        f"<< {procs} >>".encode(),
        f"<< /Type /Encoding /Differences [32 {diffs}] >>".encode(),
        b"<< /Length %d >>\nstream\n" % len(cmap) + cmap + b"\nendstream",
        b"<< /Length 8 >>\nstream\n500 0 d0\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def test_words_pdf_glyphless_text_layer(server_url):
    """An invisible OCR text layer (glyphless font) still gives lines and
    words their real height, where the text sits."""
    dpi = 100
    pdf = _glyphless_pdf("Invisible text layer words")
    r = requests.post(f"{server_url}/ocr/pdf?mode=geometric&words=1&dpi={dpi}",
                      data=pdf, timeout=60)
    assert r.status_code == 200, r.text
    page = r.json()["pages"][0]
    assert [it["text"] for it in page["results"]] == ["Invisible text layer words"]
    line = page["results"][0]
    scale = page["dpi"] / 72.0
    baseline = (792 - 700) * scale
    lx0, ly0, lx1, ly1 = _aabb(line["bounding_box"])
    # About the font size tall, around the baseline -- not a flat line.
    assert 0.6 * 20 * scale <= ly1 - ly0 <= 1.4 * 20 * scale, line
    assert ly0 < baseline <= ly1 + 2, line
    words = line["words"]
    assert [w["text"] for w in words] == ["Invisible", "text", "layer", "words"]
    for w in words:
        x0, y0, x1, y1 = _aabb(w["bounding_box"])
        assert y1 - y0 >= 0.6 * 20 * scale, w
        assert lx0 - 1 <= x0 < x1 <= lx1 + 1, w
