#include <common/Error.h>
#include <core/Device.h>
#include <core/Dtype.h>
#include <core/Storage.h>

#include <cstddef>
#include <memory>

namespace kiln {
Storage::Storage(std::size_t numel, DType dtype, Device device)
    : numel_(numel), dtype_(dtype), device_(device) {
  buf_ = std::make_shared<Buffer>(nbytes(), device);
}

Storage::Storage(void *data,
    std::size_t numel,
    DType dtype,
    Device device,
    std::shared_ptr<void> owner)
    : borrowed_(data), owner_(std::move(owner)), numel_(numel), dtype_(dtype), device_(device) {
  KILN_CHECK(device == Device::CPU, "Storage: only CPU device is implemented");
  KILN_CHECK(owner_ != nullptr, "Storage: borrowed memory requires a lifetime owner");
  KILN_CHECK(borrowed_ != nullptr || nbytes() == 0, "Storage: borrowed data pointer is null");
}

std::size_t Storage::nbytes() const {
  return kiln::nbytes(numel_, dtype_);
}

const void *Storage::data() const {
  return buf_ ? buf_->base() : borrowed_;
}
void *Storage::data() {
  return buf_ ? buf_->base() : borrowed_;
}
Device Storage::device() const {
  return device_;
}

DType Storage::dtype() const {
  return dtype_;
}

}  // namespace kiln
