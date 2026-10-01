#include <core/Tensor.h>

namespace kiln {
Tensor Tensor::empty(Shape &shape, DType dtype, Device device) {}

Tensor Tensor::zeros(Shape &shape, DType dtype, Device device) {}

Tensor Tensor::ones(Shape &shape, DType dtype, Device device) {}

Tensor Tensor::reshape(const std::vector<int64_t> &dims) const {}

Tensor Tensor::view(const std::vector<int64_t> &dims) const {}

Tensor Tensor::transpose() const {}

}  // namespace kiln
