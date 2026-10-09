#pragma once

#include <cstddef>
#include <cstdlib>

#if defined(__AVX512F__)
#define KILN_CPU_ALIGNMENT 64
#elif defined(__AVX2__) || defined(__AVX__)
#define KILN_CPU_ALIGNMENT 32
#elif defined(__SSE__) || defined(__ARM_NEON) || defined(__aarch64__)
#define KILN_CPU_ALIGNMENT 16
#else
#define KILN_CPU_ALIGNMENT sizeof(void *)
#endif

#define KILN_ALIGN_UP(bytes) (((bytes) + KILN_CPU_ALIGNMENT - 1) & ~(KILN_CPU_ALIGNMENT - 1))

#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace kiln {

inline constexpr std::size_t kCpuAlignment = KILN_CPU_ALIGNMENT;

static_assert((kCpuAlignment & (kCpuAlignment - 1)) == 0,
    "KILN_CPU_ALIGNMENT must be a power of two");

namespace detail {
inline void *aligned_alloc_impl(std::size_t alignment, std::size_t size) {
#ifdef _MSC_VER
  return _aligned_malloc(size, alignment);
#else
  void *p = nullptr;
  if (::posix_memalign(&p, alignment, size) != 0) {
    return nullptr;
  }
  return p;
#endif
}

inline void aligned_free_impl(void *p) noexcept {
#ifdef _MSC_VER
  _aligned_free(p);
#else
  std::free(p);
#endif
}
}  // namespace detail

}  // namespace kiln

#define ALIGNED_ALLOC(alignment, size) ::kiln::detail::aligned_alloc_impl(alignment, size)
#define ALIGNED_FREE(ptr) ::kiln::detail::aligned_free_impl(ptr)
