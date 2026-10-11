#pragma once

#include <core/Device.h>
#include <core/Dtype.h>
#include <core/Tensor.h>

#include <cstddef>
#include <vector>

namespace kiln_test {

inline kiln::Tensor filled_f32(const kiln::Shape &shape) {
  kiln::Tensor t(shape, kiln::DType::F32, kiln::Device::CPU);
  float *p = t.data_f32();
  for (std::size_t i = 0; i < t.numel(); ++i) {
    p[i] = static_cast<float>(i);
  }
  return t;
}

inline std::vector<float> as_vec(const kiln::Tensor &t) {
  const float *p = t.data_f32();
  return std::vector<float>(p, p + t.numel());
}

}  // namespace kiln_test
