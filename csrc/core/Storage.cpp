#include <core/Device.h>
#include <core/Dtype.h>
#include <core/Storage.h>

#include <cstddef>
#include <memory>

namespace kiln {
Storage::Storage(std::size_t numel, DType dtype, Device device) : dtype_(dtype) {
  buf_ = std::make_shared<Buffer>(nbytes(numel), device);
}

std::size_t Storage::nbytes(std::size_t numel) const {
  return numel * dtype_size(dtype_);
}

const void *Storage::data() const {
  return buf_->base();
}

Device Storage::device() const {
  return buf_->device();
}

}  // namespace kiln
