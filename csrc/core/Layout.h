#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kiln {

using Shape = std::vector<std::uint64_t>;
using Strides = std::vector<std::uint64_t>;

class Layout {
 public:
  static constexpr std::size_t kMaxDims = 6;

  Layout();
  Layout(const Shape &shape, const Strides &strides);

  static Layout contiguous(const Shape &shape);

  std::size_t rank() const {
    return rank_;
  }
  std::uint64_t shape(std::size_t dim) const;
  std::uint64_t stride(std::size_t dim) const;

  Shape shape_vec() const;
  Strides strides_vec() const;

  std::size_t numel() const;
  bool is_contiguous() const;

  Layout transpose() const;
  Layout transpose(std::int64_t dim0, std::int64_t dim1) const;
  Layout view(const std::vector<std::int64_t> &dims) const;

  static Shape resolve_shape(std::size_t numel, const std::vector<std::int64_t> &dims);

  bool operator==(const Layout &other) const;
  bool operator!=(const Layout &other) const {
    return !(*this == other);
  }

 private:
  std::size_t rank_ = 0;
  std::array<std::uint64_t, kMaxDims> shape_{};
  std::array<std::uint64_t, kMaxDims> strides_{};

  static Strides contiguous_strides(const Shape &shape);
};

}  // namespace kiln
