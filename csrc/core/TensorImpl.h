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

  void compute_strides();

 public:
  TensorImpl(Shape &shape, DType dtype, Device device = Device::CPU);
  TensorImpl(std::shared_ptr<Storage> storage,
      const Shape &shape,
      Strides strides,
      std::size_t offset);

  const std::shared_ptr<Storage> &storage() const;
  const std::int64_t offset() const;
  DType dtype() const;
  Device device() const;
  const Shape shape() const;
  const Strides strides() const;
  const std::size_t numel() const;
};
}  // namespace kiln
