#include <core/Device.h>
#include <core/Dtype.h>
#include <core/TensorImpl.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace kiln {
class Tensor {
 public:
  Tensor(Shape &shape, DType dtype, Device device)
      : impl_(std::make_shared<TensorImpl>(shape, dtype, device)) {}
  Tensor() = delete;

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

 private:
  std::shared_ptr<TensorImpl> impl_;
};
}  // namespace kiln
