#pragma once

#include <core/Tensor.h>

#include <cstdint>

namespace kiln {

enum Op : std::uint8_t {
  NONE = 0,
  ADD,
  SUBTRACT,
  MATMUL,
  RELU,
  SOFTMAX,
  COPY,
  MUL,
  SILU,
  OPS_COUNT,
};

static const char *OP_NAME[OPS_COUNT] = {
    "NONE",
    "ADD",
    "SUBTRACT",
    "MATMUL",
    "RELU",
    "SOFTMAX",
    "COPY",
    "MUL",
    "SILU",
};

Tensor add(const Tensor &a, const Tensor &b);
Tensor mul(const Tensor &a, const Tensor &b);
Tensor matmul(const Tensor &a, const Tensor &b);
Tensor relu(const Tensor &x);
Tensor silu(const Tensor &x);
Tensor softmax(const Tensor &x, int64_t axis = -1);
}  // namespace kiln
