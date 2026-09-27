// Under ThreadSanitizer, OpenCV runs its own loops on the calling thread.
// OpenCV parallelizes on TBB, whose synchronization TSan cannot see, so every
// buffer a TBB worker allocates and the caller frees reads as a race. Races
// between our own threads stay fully checked.
#include "catch_amalgamated.hpp"

#include <opencv2/core/utility.hpp>

#if defined(__SANITIZE_THREAD__)
#define TURBO_OCR_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define TURBO_OCR_TSAN 1
#endif
#endif

#ifdef TURBO_OCR_TSAN
namespace {
struct SerialOpenCv : Catch::EventListenerBase {
  using EventListenerBase::EventListenerBase;
  void testRunStarting(const Catch::TestRunInfo &) override {
    cv::setNumThreads(0);
  }
};
} // namespace
CATCH_REGISTER_LISTENER(SerialOpenCv)
#endif
