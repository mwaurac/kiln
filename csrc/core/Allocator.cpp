#include <common/Error.h>
#include <common/Export.h>
#include <core/Allocator.h>
#include <core/Device.h>

#include <cstddef>
#include <stdexcept>

namespace kiln {
Buffer::Buffer(std::size_t bytes, Device device) : size_(bytes), device_(device) {
  KILN_CHECK(device == Device::CPU, "Buffer: only CPU device is implemented");
  if (bytes == 0) {
    ptr_ = nullptr;
    return;
  }

  std::size_t alloc_size = KILN_ALIGN_UP(bytes);
  ptr_ = ALIGNED_ALLOC(kiln::kCpuAlignment, alloc_size);
  if (!ptr_) {
    throw std::bad_alloc();
  }
}

Buffer::~Buffer() {
  if (ptr_) {
    ALIGNED_FREE(ptr_);
  }
}

std::size_t Buffer::size() const {
  return size_;
}

void *Buffer::base() const {
  return ptr_;
}

Device Buffer::device() const {
  return device_;
}
}  // namespace kiln
