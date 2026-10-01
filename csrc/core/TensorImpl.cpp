#include <core/Device.h>
#include <core/Dtype.h>
#include <core/TensorImpl.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace kiln {

TensorImpl::TensorImpl(Shape &shape, DType dtype, Device device) : shape_(shape) {
  std::size_t numel = 1;
  for (uint64_t dim : shape_) {
    numel *= dim;
  }
  storage_ = std::make_shared<Storage>(numel, dtype, device);
}
}  // namespace kiln
