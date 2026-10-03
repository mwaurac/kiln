#include <core/Device.h>
#include <core/Dtype.h>
#include <core/Tensor.h>
#include <core/TensorImpl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace kiln {
Shape Tensor::shape() const {
  return impl_->shape();
}

Strides Tensor::strides() const {
  return impl_->strides();
}

DType Tensor::dtype() const {
  return impl_->dtype();
}

Device Tensor::device() const {
  return impl_->device();
}

const std::size_t Tensor::numel() const {
  return impl_->numel();
}

Tensor Tensor::empty(Shape &shape, DType dtype, Device device) {
  return Tensor(shape, dtype, device);
}

Tensor Tensor::zeros(Shape &shape, DType dtype, Device device) {
  Tensor t(shape, dtype, device);

  if (t.impl_->storage()->data() != nullptr) {
    std::memset(t.impl_->storage()->data(), 0, t.impl_->storage()->nbytes());
  }
  return t;
}

Tensor Tensor::ones(Shape &shape, DType dtype, Device device) {
  Tensor t(shape, dtype, device);

  auto *data = t.impl_->storage()->data();
  auto numel = t.impl_->numel();

  switch (dtype) {
    case kiln::DType::F32:
      std::fill(static_cast<float *>(data), static_cast<float *>(data) + numel, 1.0f);
      break;
    case F16:
      // TODO: unimplemented
      break;
    case BF16:
      break;
    case Q8_0:
      break;
  }

  return t;
}

Tensor Tensor::reshape(const std::vector<int64_t> &dims) const {}

Tensor Tensor::view(const std::vector<int64_t> &dims) const {}

Tensor Tensor::transpose() const {}

}  // namespace kiln
