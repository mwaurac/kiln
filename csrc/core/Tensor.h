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
  Tensor(Shape &shape, DType dtype, Device device)
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

  static Tensor empty(Shape &shape, DType dtype = DType::F32, Device device = Device::CPU);
  static Tensor zeros(Shape &shape, DType dtype, Device device);
  static Tensor ones(Shape &shape, DType dtype, Device device);

  Tensor reshape(const std::vector<int64_t> &dims) const;
  Tensor view(const std::vector<int64_t> &dims) const;
  Tensor transpose() const;
  Tensor transpose(int64_t dim0, int64_t dim1) const;
  Tensor contiguous() const;

  std::string print_tensor() const;

 private:
  explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
  std::shared_ptr<TensorImpl> impl_;
};
}  // namespace kiln
