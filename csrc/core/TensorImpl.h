#pragma once

#include <core/Device.h>
#include <core/Dtype.h>
#include <core/Storage.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace kiln {

class Storage;

using Strides = std::vector<uint64_t>;
using Shape = std::vector<uint64_t>;

class TensorImpl {
 private:
  std::shared_ptr<Storage> storage_;
  std::int64_t offset_;

  Strides strides_;
  Shape shape_;
  DType dtype_;
  Device device_;

  void compute_strides();

 public:
  TensorImpl(const Shape &shape, DType dtype, Device device = Device::CPU, bool allocate = true);
  TensorImpl(std::shared_ptr<Storage> storage,
      const Shape &shape,
      const Strides &strides,
      std::int64_t offset,
      DType dtype,
      Device device);

  bool has_storage() const { return storage_ != nullptr; }
  void allocate();

  const std::shared_ptr<Storage> &storage() const;
  const std::int64_t offset() const;
  DType dtype() const;
  Device device() const;
  const Shape shape() const;
  const Strides strides() const;
  const std::size_t numel() const;
  bool is_contiguous() const;

  TensorImpl view(const std::vector<int64_t> &dims) const;
  TensorImpl reshape(const std::vector<int64_t> &dims) const;
  TensorImpl transpose() const;
  TensorImpl transpose(int64_t dim0, int64_t dim1) const;
  TensorImpl contiguous() const;
};
}  // namespace kiln
