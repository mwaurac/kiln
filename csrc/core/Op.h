#pragma once

#include <cstdint>

namespace kiln {

enum Op : std::uint8_t {
  NONE = 0,
  ADD,
  SUBTRACT,
  MATMUL,
  RELU,
  SOFTMAX,
  OPS_COUNT,
};

static const char *OP_NAME[OPS_COUNT] = {
    "NONE",
    "ADD",
    "SUBTRACT",
    "MATMUL",
    "RELU",
    "SOFTMAX",
};
}  // namespace kiln
