#include <core/Device.h>
#include <core/Dtype.h>
#include <core/Layout.h>
#include <core/TensorImpl.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kiln {

TensorImpl::TensorImpl(const Shape &shape, DType dtype, Device device, bool allocate)
    : offset_(0), layout_(Layout::contiguous(shape)), dtype_(dtype), device_(device) {
  if (allocate) {
    storage_ = std::make_shared<Storage>(numel(), dtype_, device_);
  }
}

TensorImpl::TensorImpl(std::shared_ptr<Storage> storage,
    const Layout &layout,
    std::int64_t offset,
    DType dtype,
    Device device)
    : storage_(std::move(storage)), offset_(offset), layout_(layout), dtype_(dtype), device_(device) {
}

TensorImpl::TensorImpl(std::shared_ptr<Storage> storage,
    const Shape &shape,
    const Strides &strides,
    std::int64_t offset,
    DType dtype,
    Device device)
    : storage_(std::move(storage)),
      offset_(offset),
      layout_(Layout(shape, strides)),
      dtype_(dtype),
      device_(device) {}

const std::size_t TensorImpl::numel() const {
  return layout_.numel();
}

const std::shared_ptr<Storage> &TensorImpl::storage() const {
  return storage_;
}

void TensorImpl::allocate() {
  if (!storage_) {
    storage_ = std::make_shared<Storage>(numel(), dtype_, device_);
  }
}

const int64_t TensorImpl::offset() const {
  return offset_;
}

std::size_t TensorImpl::nbytes() const {
  return kiln::nbytes(numel(), dtype_);
}

std::size_t TensorImpl::itemsize() const {
  return dtype_itemsize(dtype_);
}

void *TensorImpl::data() {
  if (!storage_) {
    throw std::runtime_error("TensorImpl::data: tensor has no storage (lazy tensor)");
  }
  if (storage_->data() == nullptr) {
    if (nbytes() == 0) {
      return nullptr;
    }
    throw std::runtime_error("TensorImpl::data: storage data is null");
  }
  auto *base = static_cast<char *>(storage_->data());
  return base + kiln::nbytes(static_cast<std::size_t>(offset_), dtype_);
}

const void *TensorImpl::data() const {
  if (!storage_) {
    throw std::runtime_error("TensorImpl::data: tensor has no storage (lazy tensor)");
  }
  if (storage_->data() == nullptr) {
    if (nbytes() == 0) {
      return nullptr;
    }
    throw std::runtime_error("TensorImpl::data: storage data is null");
  }
  const auto *base = static_cast<const char *>(storage_->data());
  return base + kiln::nbytes(static_cast<std::size_t>(offset_), dtype_);
}

namespace {
void require_dtype(DType actual, DType expected, const char *what) {
  if (actual != expected) {
    throw std::runtime_error(std::string(what) + ": dtype mismatch: tensor is " +
                             dtype_name(actual) + ", accessor requires " + dtype_name(expected));
  }
}
}  // namespace

float *TensorImpl::data_f32() {
  require_dtype(dtype_, DType::F32, "TensorImpl::data_f32");
  return static_cast<float *>(data());
}
const float *TensorImpl::data_f32() const {
  require_dtype(dtype_, DType::F32, "TensorImpl::data_f32");
  return static_cast<const float *>(data());
}
std::uint16_t *TensorImpl::data_f16() {
  require_dtype(dtype_, DType::F16, "TensorImpl::data_f16");
  return static_cast<std::uint16_t *>(data());
}
const std::uint16_t *TensorImpl::data_f16() const {
  require_dtype(dtype_, DType::F16, "TensorImpl::data_f16");
  return static_cast<const std::uint16_t *>(data());
}
std::uint16_t *TensorImpl::data_bf16() {
  require_dtype(dtype_, DType::BF16, "TensorImpl::data_bf16");
  return static_cast<std::uint16_t *>(data());
}
const std::uint16_t *TensorImpl::data_bf16() const {
  require_dtype(dtype_, DType::BF16, "TensorImpl::data_bf16");
  return static_cast<const std::uint16_t *>(data());
}
block_q8_0 *TensorImpl::data_q8_0() {
  require_dtype(dtype_, DType::Q8_0, "TensorImpl::data_q8_0");
  return static_cast<block_q8_0 *>(data());
}
const block_q8_0 *TensorImpl::data_q8_0() const {
  require_dtype(dtype_, DType::Q8_0, "TensorImpl::data_q8_0");
  return static_cast<const block_q8_0 *>(data());
}

const Shape TensorImpl::shape() const {
  return layout_.shape_vec();
}

const Strides TensorImpl::strides() const {
  return layout_.strides_vec();
}

bool TensorImpl::is_contiguous() const {
  return layout_.is_contiguous();
}

DType TensorImpl::dtype() const {
  return dtype_;
}

Device TensorImpl::device() const {
  return device_;
}

TensorImpl TensorImpl::view(const std::vector<int64_t> &dims) const {
  return TensorImpl(storage(), layout_.view(dims), offset(), dtype_, device_);
}

TensorImpl copy_contiguous(const TensorImpl &src) {
  const Layout &layout = src.layout();
  if (!src.has_storage()) {
    return TensorImpl(layout.shape_vec(), src.dtype(), src.device(), false);
  }
  TensorImpl out(layout.shape_vec(), src.dtype(), src.device());

  const size_t n = src.numel();
  auto *dst = static_cast<char *>(out.storage()->data());
  auto *base = static_cast<const char *>(src.storage()->data()) +
               kiln::nbytes(static_cast<size_t>(src.offset()), src.dtype());

  if (n == 0) {
    return out;
  }

  if (src.is_contiguous()) {
    std::memcpy(dst, base, kiln::nbytes(n, src.dtype()));
    return out;
  }

  if (dtype_is_quantized(src.dtype())) {
    throw std::runtime_error("copy_contiguous: strided copy of block-quantized dtype " +
                             dtype_name(src.dtype()) + " is not supported");
  }
  const size_t esize = dtype_itemsize(src.dtype());

  const Shape shape = src.shape();
  const size_t rank = shape.size();
  std::vector<size_t> idx(rank, 0);

  size_t src_elem = 0;

  for (size_t i = 0; i < n; ++i) {
    std::memcpy(dst + i * esize, base + src_elem * esize, esize);

    for (size_t dd = rank; dd-- > 0;) {
      src_elem += layout.stride(dd);
      if (++idx[dd] < shape[dd]) {
        break;
      }
      src_elem -= layout.stride(dd) * shape[dd];
      idx[dd] = 0;
    }
  }
  return out;
}

TensorImpl TensorImpl::reshape(const std::vector<int64_t> &dims) const {
  if (is_contiguous()) {
    return view(dims);
  }

  return contiguous().view(dims);
}

TensorImpl TensorImpl::transpose() const {
  return TensorImpl(storage(), layout_.transpose(), offset(), dtype_, device_);
}

TensorImpl TensorImpl::transpose(int64_t dim0, int64_t dim1) const {
  return TensorImpl(storage(), layout_.transpose(dim0, dim1), offset(), dtype_, device_);
}

TensorImpl TensorImpl::contiguous() const {
  if (is_contiguous())
    return *this;
  return copy_contiguous(*this);
}
}  // namespace kiln
