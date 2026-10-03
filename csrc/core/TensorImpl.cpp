#include <core/Device.h>
#include <core/Dtype.h>
#include <core/TensorImpl.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace kiln {

TensorImpl::TensorImpl(Shape &shape, DType dtype, Device device) : shape_(shape) {
  storage_ = std::make_shared<Storage>(numel(), dtype, device);
  compute_strides();
}

const std::size_t TensorImpl::numel() const {
  std::size_t numel = 1;
  for (uint64_t dim : shape_) {
    numel *= dim;
  }

  return numel;
}

void TensorImpl::compute_strides() {
  strides_.resize(shape_.size());

  std::size_t stride = 1;

  for (std::size_t i = shape_.size(); i-- > 0;) {
    strides_[i] = stride;
    stride *= shape_[i];
  }
}

const std::shared_ptr<Storage> &TensorImpl::storage() const {
  return storage_;
}

const int64_t TensorImpl::offset() const {
  return offset_;
}

const Shape TensorImpl::shape() const {
  return shape_;
}

const Strides TensorImpl::strides() const {
  return strides_;
}

DType TensorImpl::dtype() const {
  return storage_->dtype();
}

Device TensorImpl::device() const {
  return storage_->device();
}
}  // namespace kiln
