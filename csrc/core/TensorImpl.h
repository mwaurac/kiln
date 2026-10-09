#pragma once

#include <core/Device.h>
#include <core/Dtype.h>
#include <core/Layout.h>
#include <core/Storage.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace kiln {

class Storage;

class TensorImpl {
 private:
  std::shared_ptr<Storage> storage_;
  std::int64_t offset_;

  Layout layout_;
  DType dtype_;
  Device device_;

 public:
  TensorImpl(const Shape &shape, DType dtype, Device device = Device::CPU, bool allocate = true);
  TensorImpl(std::shared_ptr<Storage> storage,
      const Layout &layout,
      std::int64_t offset,
      DType dtype,
      Device device);
  TensorImpl(std::shared_ptr<Storage> storage,
      const Shape &shape,
      const Strides &strides,
      std::int64_t offset,
      DType dtype,
      Device device);

  bool has_storage() const {
    return storage_ != nullptr;
  }
  void allocate();

  const std::shared_ptr<Storage> &storage() const;
  const std::int64_t offset() const;
  DType dtype() const;
  Device device() const;
  const Shape shape() const;
  const Strides strides() const;
  const Layout &layout() const {
    return layout_;
  }
  const std::size_t numel() const;
  std::size_t nbytes() const;
  std::size_t itemsize() const;
  bool is_contiguous() const;

  void *data();
  const void *data() const;

  float *data_f32();
  const float *data_f32() const;
  std::uint16_t *data_f16();
  const std::uint16_t *data_f16() const;
  std::uint16_t *data_bf16();
  const std::uint16_t *data_bf16() const;
  block_q8_0 *data_q8_0();
  const block_q8_0 *data_q8_0() const;

  TensorImpl view(const std::vector<int64_t> &dims) const;
  TensorImpl reshape(const std::vector<int64_t> &dims) const;
  TensorImpl transpose() const;
  TensorImpl transpose(int64_t dim0, int64_t dim1) const;
  TensorImpl contiguous() const;
};
}  // namespace kiln
