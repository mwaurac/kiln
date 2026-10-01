#include <core/Allocator.h>

#include <cstddef>
#include <cstdlib>
#include <stdexcept>

#include "core/Device.h"

#if defined(__AVX512F__)
#define CPU_ALIGNMENT 64
#elif defined(__AVX2__) || defined(__AVX__)
#define CPU_ALIGNMENT 32
#elif defined(__SSE__) || defined(__ARM_NEON) || defined(__aarch64__)
#define CPU_ALIGNMENT 16
#else
// Fallback
#define CPU_ALIGNMENT sizeof(void *)
#endif

#define ALIGN_UP(bytes) (((bytes) + CPU_ALIGNMENT - 1) & ~(CPU_ALIGNMENT - 1))

#ifdef __MSC_VER
#include <malloc.h>
#endif

void *AlignedAlloc(std::size_t alignment, std::size_t size) {
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

void AlignedFree(void *p) noexcept {
#ifdef _MSC_VER
  _aligned_free(p);
#else
  std::free(p);
#endif
}

namespace kiln {
Buffer::Buffer(std::size_t bytes, Device device) : size_(bytes), device_(device) {
  if (device != Device::CPU) {
    throw std::runtime_error("Buffer: only CPU device is implemented");
  }
  if (bytes == 0) {
    ptr_ = nullptr;
    return;
  }

  std::size_t alloc_size = ALIGN_UP(bytes);
  ptr_ = AlignedAlloc(CPU_ALIGNMENT, alloc_size);
  if (!ptr_) {
    throw std::bad_alloc();
  }
}

Buffer::~Buffer() {
  if (ptr_) {
    AlignedFree(ptr_);
  }
}

std::size_t Buffer::size() const {
  return size_;
}

const void *Buffer::base() const {
  return ptr_;
}

Device Buffer::device() const {
  return device_;
}
}  // namespace kiln
