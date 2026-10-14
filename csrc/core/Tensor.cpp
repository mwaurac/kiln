#include <common/Error.h>
#include <core/Device.h>
#include <core/Dtype.h>
#include <core/Graph.h>
#include <core/Layout.h>
#include <core/Tensor.h>
#include <core/TensorImpl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
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

namespace {
const TensorImpl *cimpl(const Tensor &t) {
  return static_cast<const TensorImpl *>(t.impl().get());
}
}  // namespace

void *Tensor::data() {
  return impl_->data();
}

const void *Tensor::data() const {
  return cimpl(*this)->data();
}

float *Tensor::data_f32() {
  return impl_->data_f32();
}

const float *Tensor::data_f32() const {
  return cimpl(*this)->data_f32();
}

std::uint16_t *Tensor::data_f16() {
  return impl_->data_f16();
}

const std::uint16_t *Tensor::data_f16() const {
  return cimpl(*this)->data_f16();
}

std::uint16_t *Tensor::data_bf16() {
  return impl_->data_bf16();
}

const std::uint16_t *Tensor::data_bf16() const {
  return cimpl(*this)->data_bf16();
}

block_q8_0 *Tensor::data_q8_0() {
  return impl_->data_q8_0();
}

const block_q8_0 *Tensor::data_q8_0() const {
  return cimpl(*this)->data_q8_0();
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
      KILN_ERROR("ones: unsupported dtype ",
          dtype_name(dtype),
          " (only f32/f16/bf16 are supported)");
  }

  return t;
}

Tensor Tensor::lazy(const Shape &shape, DType dtype, Device device) {
  return Tensor(std::make_shared<TensorImpl>(shape, dtype, device, /*allocate=*/false));
}

Tensor Tensor::from_blob(void *data,
    const Shape &shape,
    const Strides &strides,
    DType dtype,
    Device device,
    std::shared_ptr<void> owner) {
  Layout layout(shape, strides);
  auto storage = std::make_shared<Storage>(data, layout.numel(), dtype, device, std::move(owner));
  TensorImpl impl(storage, layout, /*offset=*/0, dtype, device);
  KILN_CHECK(!dtype_is_quantized(dtype) || impl.is_contiguous(),
      "from_blob: non-contiguous views of block-quantized dtype ",
      dtype_name(dtype),
      " are not supported");
  return Tensor(std::make_shared<TensorImpl>(std::move(impl)));
}

Tensor Tensor::from_blob(const void *data,
    const Shape &shape,
    const Strides &strides,
    DType dtype,
    Device device,
    std::shared_ptr<void> owner) {
  Layout layout(shape, strides);
  auto storage = std::make_shared<Storage>(data, layout.numel(), dtype, device, std::move(owner));
  TensorImpl impl(storage, layout, /*offset=*/0, dtype, device);
  KILN_CHECK(!dtype_is_quantized(dtype) || impl.is_contiguous(),
      "from_blob: non-contiguous views of block-quantized dtype ",
      dtype_name(dtype),
      " are not supported");
  return Tensor(std::make_shared<TensorImpl>(std::move(impl)));
}

bool Tensor::has_storage() const {
  return impl_->has_storage();
}

bool Tensor::is_borrowed() const {
  return impl_->is_borrowed();
}

bool Tensor::is_readonly() const {
  return impl_->is_readonly();
}

void Tensor::allocate() {
  impl_->allocate();
}

Tensor Tensor::copy_(const Tensor &src) {
  KILN_CHECK(src.has_storage(), "copy_: source has no storage (lazy output); call execute() first");
  KILN_CHECK(dtype() == src.dtype(),
      "copy_: dtype mismatch: destination is ",
      dtype_name(dtype()),
      ", source is ",
      dtype_name(src.dtype()));
  KILN_CHECK(shape() == src.shape(),
      "copy_: shape mismatch: destination is ",
      shape_string(shape()),
      ", source is ",
      shape_string(src.shape()));
  if (impl_.get() == src.impl_.get()) {
    return *this;
  }
  if (!has_storage()) {
    allocate();
  }
  KILN_CHECK(!is_readonly(), "copy_: destination is read-only borrowed memory");
  const std::size_t n = numel();
  if (n == 0) {
    return *this;
  }
  if (dtype_is_quantized(dtype())) {
    KILN_CHECK(is_contiguous() && src.is_contiguous(),
        "copy_: strided copy of block-quantized dtype ",
        dtype_name(dtype()),
        " is not supported");
    std::memmove(data(), src.data(), nbytes());
    return *this;
  }
  if (is_contiguous() && src.is_contiguous()) {
    std::memmove(data(), src.data(), nbytes());
    return *this;
  }
  const std::size_t esize = itemsize();
  const Layout &dlayout = impl_->layout();
  const Layout &slayout = src.impl_->layout();
  const Shape &dims = shape();
  const std::size_t rank = dims.size();
  std::vector<std::size_t> idx(rank, 0);
  auto *dst = static_cast<char *>(data());
  const auto *base = static_cast<const char *>(src.data());
  std::size_t dst_elem = 0;
  std::size_t src_elem = 0;
  for (std::size_t i = 0; i < n; ++i) {
    std::memcpy(dst + dst_elem * esize, base + src_elem * esize, esize);
    for (std::size_t dd = rank; dd-- > 0;) {
      dst_elem += dlayout.stride(dd);
      src_elem += slayout.stride(dd);
      if (++idx[dd] < dims[dd]) {
        break;
      }
      dst_elem -= dlayout.stride(dd) * dims[dd];
      src_elem -= slayout.stride(dd) * dims[dd];
      idx[dd] = 0;
    }
  }
  return *this;
}

Tensor Tensor::reshape(const std::vector<int64_t> &dims) const {
  if (!is_contiguous()) {
    return contiguous().view(dims);
  }
  return view(dims);
}

Tensor Tensor::view(const std::vector<int64_t> &dims) const {
  Tensor out(std::make_shared<TensorImpl>(impl_->view(dims)));
  out.impl_->set_base(impl_);
  return out;
}

Tensor Tensor::transpose() const {
  Tensor out(std::make_shared<TensorImpl>(impl_->transpose()));
  out.impl_->set_base(impl_);
  return out;
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
  Tensor out(std::make_shared<TensorImpl>(impl_->transpose(dim0, dim1)));
  out.impl_->set_base(impl_);
  return out;
}

Tensor Tensor::contiguous() const {
  if (is_contiguous()) {
    return *this;
  }
  if (has_storage()) {
    return Tensor(std::make_shared<TensorImpl>(impl_->contiguous()));
  }

  Tensor out = Tensor::lazy(shape(), dtype(), device());
  std::shared_ptr<Graph> graph = Graph::resolve_output_graph({this});
  (graph ? *graph : default_graph()).add_node("contiguous", Op::COPY, *this, out);
  return out;
}

std::string Tensor::print_tensor() const {
  return std::string("kiln.Tensor(shape=") + shape_string(shape()) +
         ", dtype=" + dtype_name(dtype()) + ", device=" + device_name(device()) + ")";
}

}  // namespace kiln
