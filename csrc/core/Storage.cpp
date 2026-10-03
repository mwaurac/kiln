#include <core/Device.h>
#include <core/Dtype.h>
#include <core/Storage.h>

#include <cstddef>
#include <memory>

namespace kiln {
Storage::Storage(std::size_t numel, DType dtype, Device device) : numel_(numel), dtype_(dtype) {
  buf_ = std::make_shared<Buffer>(nbytes(), device);
}

std::size_t Storage::nbytes() const {
  return numel_ * dtype_size(dtype_);
}

const void *Storage::data() const {
  return buf_->base();
}
void *Storage::data() {
  return buf_->base();
}
Device Storage::device() const {
  return buf_->device();
}

DType Storage::dtype() const {
  return dtype_;
}

}  // namespace kiln
