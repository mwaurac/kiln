#include <core/TensorImpl.h>

#include <memory>
#include <vector>

#include "core/Device.h"
#include "core/Dtype.h"

namespace kiln {
class Tensor {
 public:
  Tensor() = delete;

  Tensor empty(Shape &shape, DType dtype = DType::F32, Device device = Device::CPU);
  Tensor zeros(Shape &shape, DType dtype, Device device);
  Tensor ones(Shape &shape, DType dtype, Device device);

  Tensor reshape(const std::vector<int64_t> &dims) const;
  Tensor view(const std::vector<int64_t> &dims) const;
  Tensor transpose() const;

 private:
  std::shared_ptr<TensorImpl> impl_;
};
}  // namespace kiln
