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
  if (device != Device::CPU) {
    throw std::runtime_error("Storage: only CPU device is implemented");
  }
  if (!owner_) {
    throw std::runtime_error("Storage: borrowed memory requires a lifetime owner");
  }
  if (nbytes() > 0 && borrowed_ == nullptr) {
    throw std::runtime_error("Storage: borrowed data pointer is null");
  }
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
