#pragma once

#include <core/Tensor.h>

#include <cstdint>

namespace kiln::kernels {

void add_f32(const Tensor &a, const Tensor &b, Tensor &out);
void mul_f32(const Tensor &a, const Tensor &b, Tensor &out);
void matmul_f32(const Tensor &a, const Tensor &b, Tensor &out);
void relu_f32(const Tensor &x, Tensor &out);
void silu_f32(const Tensor &x, Tensor &out);
void softmax_f32(const Tensor &x, Tensor &out, std::int64_t axis);

}  // namespace kiln::kernels
