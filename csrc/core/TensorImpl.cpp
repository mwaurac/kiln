#include <core/Device.h>
#include <core/Dtype.h>
#include <core/TensorImpl.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kiln {

TensorImpl::TensorImpl(Shape &shape, DType dtype, Device device) : shape_(shape), offset_(0) {
  storage_ = std::make_shared<Storage>(numel(), dtype, device);
  compute_strides();
}

TensorImpl::TensorImpl(std::shared_ptr<Storage> storage,
    const Shape &shape,
    const Strides &strides,
    std::int64_t offset)
    : storage_(std::move(storage)), shape_(shape), strides_(strides), offset_(offset) {}

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

bool TensorImpl::is_contiguous() const {
  if (shape_.empty()) {
    return true;
  }
  std::uint64_t expected = 1;
  for (std::size_t i = shape_.size(); i-- > 0;) {
    if (shape_[i] == 1) {
      continue;
    }
    if (strides_[i] != expected) {
      return false;
    }
    expected *= shape_[i];
  }
  return true;
}

DType TensorImpl::dtype() const {
  return storage_->dtype();
}

Device TensorImpl::device() const {
  return storage_->device();
}

[[noreturn]] void bad_reshape(const char *msg) {
  throw std::runtime_error(std::string("view/reshape: ") + msg);
}

static uint64_t infer_dim(const int64_t numel, int64_t known_product) {
  if (known_product == 0) {
    if (numel != 0) {
      bad_reshape("cannot infer dimension for non-zero numel with zero in shape");
    }
    return 0;
  } else {
    if (numel % known_product != 0) {
      bad_reshape("cannot infer dimension (-1): numel is not divisible");
    }
    return static_cast<uint64_t>(numel / known_product);
  }
}

static Shape resolve_shape(const int64_t numel, const std::vector<int64_t> &dims) {
  Shape new_shape;
  new_shape.reserve(dims.size());

  std::optional<size_t> infer_pos;
  std::size_t known_product = 1;

  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] == -1) {
      if (infer_pos) {
        bad_reshape("only one dimension can be inferred (-1)");
      }
      infer_pos = i;
      new_shape.push_back(0);  // placeholder, filled below
    } else if (dims[i] < -1) {
      bad_reshape("invalid negative dimension (only -1 is allowed)");
    } else {
      new_shape.push_back(static_cast<uint64_t>(dims[i]));
      known_product *= static_cast<uint64_t>(dims[i]);
    }
  }

  if (infer_pos) {
    new_shape[*infer_pos] = infer_dim(numel, known_product);
  }

  return new_shape;
}

TensorImpl TensorImpl::view(const std::vector<int64_t> &dims) const {
  const auto new_shape = resolve_shape(numel(), dims);

  std::size_t new_numel = 1;
  for (uint64_t d : new_shape) {
    new_numel *= d;
  }
  if (new_numel != numel()) {
    bad_reshape("numel mismatch");
  }

  if (!is_contiguous()) {
    bad_reshape("tensor is non-contiguous, use reshape instead");
  }

  Strides new_strides(new_shape.size());
  std::size_t stride = 1;
  for (std::size_t i = new_shape.size(); i-- > 0;) {
    new_strides[i] = stride;
    stride *= new_shape[i];
  }

  return TensorImpl(storage(), new_shape, new_strides, offset());
}

TensorImpl copy_contiguous(const TensorImpl &src) {
  Shape shape = src.shape();
  TensorImpl out(shape, src.dtype(), src.device());

  const size_t esize = dtype_size(src.dtype());
  const size_t n = src.numel();
  auto *dst = static_cast<char *>(out.storage()->data());
  auto *base =
      static_cast<const char *>(src.storage()->data()) + static_cast<size_t>(src.offset()) * esize;

  if (n == 0) {
    return out;
  }

  if (src.is_contiguous()) {
    std::memcpy(dst, base, n * esize);
    return out;
  }

  const Strides strides = src.strides();
  const size_t rank = shape.size();
  std::vector<size_t> idx(rank, 0);

  size_t src_elem = 0;

  for (size_t i = 0; i < n; ++i) {
    std::memcpy(dst + i * esize, base + src_elem * esize, esize);

    for (size_t dd = rank; dd-- > 0;) {
      src_elem += strides[dd];
      if (++idx[dd] < shape[dd]) {
        break;
      }
      src_elem -= strides[dd] * shape[dd];
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
  const Shape src_shape = shape();
  const Strides src_strides = strides();
  const std::size_t rank = src_shape.size();
  Shape new_shape(rank);
  Strides new_strides(rank);
  for (std::size_t i = 0; i < rank; ++i) {
    new_shape[i] = src_shape[rank - 1 - i];
    new_strides[i] = src_strides[rank - 1 - i];
  }

  return TensorImpl(storage(), new_shape, new_strides, offset());
}

TensorImpl TensorImpl::transpose(int64_t dim0, int64_t dim1) const {
  const std::size_t rank = shape_.size();
  auto normalize = [rank](int64_t d, const char *name) -> std::size_t {
    if (d < 0) {
      d += static_cast<int64_t>(rank);
    }
    if (d < 0 || d >= static_cast<int64_t>(rank)) {
      throw std::runtime_error("transpose: " + std::string(name) + " (" +
          std::to_string(d < 0 ? d - static_cast<int64_t>(rank) : d) +
          ") out of range for rank-" + std::to_string(rank) + " tensor");
    }
    return static_cast<std::size_t>(d);
  };
  const std::size_t d0 = normalize(dim0, "dim0");
  const std::size_t d1 = normalize(dim1, "dim1");

  Shape new_shape = shape_;
  Strides new_strides = strides_;
  std::swap(new_shape[d0], new_shape[d1]);
  std::swap(new_strides[d0], new_strides[d1]);

  return TensorImpl(storage_, new_shape, new_strides, offset_);
}

TensorImpl TensorImpl::contiguous() const {
  if (is_contiguous())
    return *this;
  return copy_contiguous(*this);
}
}  // namespace kiln
