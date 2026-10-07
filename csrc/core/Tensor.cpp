#include <core/Device.h>
#include <core/Dtype.h>
#include <core/Tensor.h>
#include <core/TensorImpl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
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

std::size_t Tensor::nbytes() const {
  return impl_->nbytes();
}

std::size_t Tensor::itemsize() const {
  return impl_->itemsize();
}

bool Tensor::is_contiguous() const {
  return impl_->is_contiguous();
}

void *Tensor::data() {
  return impl_->data();
}

const void *Tensor::data() const {
  return impl_->data();
}

float *Tensor::data_f32() {
  return impl_->data_f32();
}

const float *Tensor::data_f32() const {
  return impl_->data_f32();
}

std::uint16_t *Tensor::data_f16() {
  return impl_->data_f16();
}

const std::uint16_t *Tensor::data_f16() const {
  return impl_->data_f16();
}

std::uint16_t *Tensor::data_bf16() {
  return impl_->data_bf16();
}

const std::uint16_t *Tensor::data_bf16() const {
  return impl_->data_bf16();
}

block_q8_0 *Tensor::data_q8_0() {
  return impl_->data_q8_0();
}

const block_q8_0 *Tensor::data_q8_0() const {
  return impl_->data_q8_0();
}

Tensor Tensor::empty(const Shape &shape, DType dtype, Device device) {
  return Tensor(shape, dtype, device);
}

Tensor Tensor::zeros(const Shape &shape, DType dtype, Device device) {
  Tensor t(shape, dtype, device);

  if (t.numel() == 0) {
    return t;
  }
  std::memset(t.data(), 0, t.nbytes());
  return t;
}

Tensor Tensor::ones(const Shape &shape, DType dtype, Device device) {
  Tensor t(shape, dtype, device);

  auto numel = t.numel();
  if (numel == 0) {
    return t;
  }

  switch (dtype) {
    case kiln::DType::F32:
      std::fill(t.data_f32(), t.data_f32() + numel, 1.0f);
      break;
    case kiln::DType::F16:
      std::fill(t.data_f16(), t.data_f16() + numel, std::uint16_t(0x3C00));
      break;
    case kiln::DType::BF16:
      std::fill(t.data_bf16(), t.data_bf16() + numel, std::uint16_t(0x3F80));
      break;
    default:
      throw std::runtime_error(
          "ones: unsupported dtype " + dtype_name(dtype) + " (only f32/f16/bf16 are supported)");
  }

  return t;
}

Tensor Tensor::lazy(const Shape &shape, DType dtype, Device device) {
  return Tensor(std::make_shared<TensorImpl>(shape, dtype, device, /*allocate=*/false));
}

bool Tensor::has_storage() const {
  return impl_->has_storage();
}

void Tensor::allocate() {
  impl_->allocate();
}

Tensor Tensor::reshape(const std::vector<int64_t> &dims) const {
  return Tensor(std::make_shared<TensorImpl>(impl_->reshape(dims)));
}

Tensor Tensor::view(const std::vector<int64_t> &dims) const {
  return Tensor(std::make_shared<TensorImpl>(impl_->view(dims)));
}

Tensor Tensor::transpose() const {
  return Tensor(std::make_shared<TensorImpl>(impl_->transpose()));
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
  return Tensor(std::make_shared<TensorImpl>(impl_->transpose(dim0, dim1)));
}

Tensor Tensor::contiguous() const {
  return Tensor(std::make_shared<TensorImpl>(impl_->contiguous()));
}

std::string Tensor::print_tensor() const {
  std::string s = "kiln.Tensor(shape=[";

  for (size_t i = 0; i < shape().size(); ++i) {
    s += std::to_string(shape()[i]);
    if (i + 1 < shape().size()) {
      s += ", ";
    }
  }
  s += std::string("], dtype=") + dtype_name(dtype()) + ", device=" + device_name(device()) + ")";
  return s;
}

}  // namespace kiln
