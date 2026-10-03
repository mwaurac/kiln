#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace kiln {

enum DType : uint8_t {
  F32 = 0,
  F16 = 1,
  BF16 = 2,
  // Quantized formats
  Q8_0 = 3,
  DTYPE_COUNT = 4,
};

struct block_q8_0 {
  uint16_t d;     // delta (fp16 bits)
  int8_t qs[32];  // quants
};

struct DTypeInfo {
  const char *name;
  size_t block_size;
  size_t bytes_per_block;
  bool is_quantized;
};

std::size_t dtype_size(DType dtype);
std::string dtype_name(DType dtype);
}  // namespace kiln
