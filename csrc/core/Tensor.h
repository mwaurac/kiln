#pragma once

#include <core/Device.h>
#include <core/Dtype.h>
#include <core/TensorImpl.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace kiln {
class Tensor {
 public:
  Tensor(const Shape &shape, DType dtype, Device device)
      : impl_(std::make_shared<TensorImpl>(shape, dtype, device)) {}
  Tensor() = delete;

  Tensor(const Tensor &) = default;
  Tensor &operator=(const Tensor &) = default;
  Tensor(Tensor &&) noexcept = default;
  Tensor &operator=(Tensor &&) noexcept = default;

  Shape shape() const;
  Strides strides() const;
  DType dtype() const;
  Device device() const;
  const std::size_t numel() const;
  std::size_t nbytes() const;
  std::size_t itemsize() const;
  bool is_contiguous() const;

  static Tensor empty(const Shape &shape, DType dtype = DType::F32, Device device = Device::CPU);
  static Tensor zeros(const Shape &shape, DType dtype, Device device);
  static Tensor ones(const Shape &shape, DType dtype, Device device);
  static Tensor lazy(const Shape &shape, DType dtype = DType::F32, Device device = Device::CPU);

  static Tensor from_blob(void *data,
      const Shape &shape,
      const Strides &strides,
      DType dtype,
      Device device,
      std::shared_ptr<void> owner);

  bool has_storage() const;
  void allocate();

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

  Tensor reshape(const std::vector<int64_t> &dims) const;
  Tensor view(const std::vector<int64_t> &dims) const;
  Tensor transpose() const;
  Tensor transpose(int64_t dim0, int64_t dim1) const;
  Tensor contiguous() const;

  std::string print_tensor() const;

  const std::shared_ptr<TensorImpl> &impl() const {
    return impl_;
  }

 private:
  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
  std::shared_ptr<TensorImpl> impl_;
};
}  // namespace kiln
