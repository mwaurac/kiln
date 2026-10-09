#include <common/Error.h>
#include <core/Layout.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace kiln {
namespace {
void require_rank(std::size_t rank, const char *what) {
  KILN_CHECK(rank <= Layout::kMaxDims,
      what,
      ": rank (",
      rank,
      ") exceeds max dims (",
      Layout::kMaxDims,
      ")");
}

[[noreturn]] void bad_reshape(const char *msg) {
  KILN_ERROR("view/reshape: ", msg);
}

std::uint64_t infer_dim(std::size_t numel, std::size_t known_product) {
  if (known_product == 0) {
    if (numel != 0) {
      bad_reshape("cannot infer dimension for non-zero numel with zero in shape");
    }
    return 0;
  }
  if (numel % known_product != 0) {
    bad_reshape("cannot infer dimension (-1): numel is not divisible");
  }
  return static_cast<std::uint64_t>(numel / known_product);
}
}  // namespace

Layout::Layout() : rank_(0) {
  shape_.fill(0);
  strides_.fill(0);
}

Layout::Layout(const Shape &shape, const Strides &strides) {
  require_rank(shape.size(), "Layout");
  KILN_CHECK(strides.size() == shape.size(),
      "Layout: strides rank (",
      strides.size(),
      ") does not match shape rank (",
      shape.size(),
      ")");
  rank_ = shape.size();
  shape_.fill(0);
  strides_.fill(0);
  for (std::size_t i = 0; i < rank_; ++i) {
    shape_[i] = shape[i];
    strides_[i] = strides[i];
  }
}

Strides Layout::contiguous_strides(const Shape &shape) {
  require_rank(shape.size(), "Layout");
  Strides strides(shape.size());
  std::size_t stride = 1;
  for (std::size_t i = shape.size(); i-- > 0;) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

Layout Layout::contiguous(const Shape &shape) {
  return Layout(shape, contiguous_strides(shape));
}

std::uint64_t Layout::shape(std::size_t dim) const {
  KILN_CHECK(dim < rank_,
      "Layout::shape: dim (",
      dim,
      ") out of range for rank-",
      rank_,
      " layout");
  return shape_[dim];
}

std::uint64_t Layout::stride(std::size_t dim) const {
  KILN_CHECK(dim < rank_,
      "Layout::stride: dim (",
      dim,
      ") out of range for rank-",
      rank_,
      " layout");
  return strides_[dim];
}

Shape Layout::shape_vec() const {
  Shape out;
  out.reserve(rank_);
  for (std::size_t i = 0; i < rank_; ++i) {
    out.push_back(shape_[i]);
  }
  return out;
}

Strides Layout::strides_vec() const {
  Strides out;
  out.reserve(rank_);
  for (std::size_t i = 0; i < rank_; ++i) {
    out.push_back(strides_[i]);
  }
  return out;
}

std::size_t Layout::numel() const {
  std::size_t n = 1;
  for (std::size_t i = 0; i < rank_; ++i) {
    n *= shape_[i];
  }
  return n;
}

bool Layout::is_contiguous() const {
  if (rank_ == 0) {
    return true;
  }
  std::uint64_t expected = 1;
  for (std::size_t i = rank_; i-- > 0;) {
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

Layout Layout::transpose() const {
  Shape new_shape(rank_);
  Strides new_strides(rank_);
  for (std::size_t i = 0; i < rank_; ++i) {
    new_shape[i] = shape_[rank_ - 1 - i];
    new_strides[i] = strides_[rank_ - 1 - i];
  }
  return Layout(new_shape, new_strides);
}

Layout Layout::transpose(std::int64_t dim0, std::int64_t dim1) const {
  auto normalize = [this](std::int64_t d, const char *name) -> std::size_t {
    std::int64_t rank = static_cast<std::int64_t>(rank_);
    std::int64_t orig = d;
    if (d < 0) {
      d += rank;
    }
    if (d < 0 || d >= rank) {
      KILN_ERROR("transpose: ", name, " (", orig, ") out of range for rank-", rank_, " tensor");
    }
    return static_cast<std::size_t>(d);
  };
  const std::size_t d0 = normalize(dim0, "dim0");
  const std::size_t d1 = normalize(dim1, "dim1");

  Shape new_shape = shape_vec();
  Strides new_strides = strides_vec();
  std::swap(new_shape[d0], new_shape[d1]);
  std::swap(new_strides[d0], new_strides[d1]);
  return Layout(new_shape, new_strides);
}

Shape Layout::resolve_shape(std::size_t numel, const std::vector<std::int64_t> &dims) {
  require_rank(dims.size(), "view/reshape");
  Shape new_shape;
  new_shape.reserve(dims.size());

  std::optional<std::size_t> infer_pos;
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
      new_shape.push_back(static_cast<std::uint64_t>(dims[i]));
      known_product *= static_cast<std::uint64_t>(dims[i]);
    }
  }

  if (infer_pos) {
    new_shape[*infer_pos] = infer_dim(numel, known_product);
  }

  return new_shape;
}

Layout Layout::view(const std::vector<std::int64_t> &dims) const {
  const std::size_t n = numel();
  const Shape new_shape = resolve_shape(n, dims);

  std::size_t new_numel = 1;
  for (std::uint64_t d : new_shape) {
    new_numel *= d;
  }
  if (new_numel != n) {
    bad_reshape("numel mismatch");
  }

  if (!is_contiguous()) {
    bad_reshape("tensor is non-contiguous, use reshape instead");
  }

  return Layout::contiguous(new_shape);
}

bool Layout::operator==(const Layout &other) const {
  if (rank_ != other.rank_) {
    return false;
  }
  for (std::size_t i = 0; i < rank_; ++i) {
    if (shape_[i] != other.shape_[i] || strides_[i] != other.strides_[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace kiln
