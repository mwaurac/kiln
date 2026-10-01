#include <core/Dtype.h>

#include <cstddef>

namespace kiln {

const DTypeInfo type_info[DTYPE_COUNT] = {
    {
        .name = "f32",
        .block_size = 1,
        .bytes_per_block = sizeof(float),
        .is_quantized = false,
    },
    {
        .name = "f16",
        .block_size = 1,
        .bytes_per_block = sizeof(uint16_t),
        .is_quantized = false,
    },
    {
        .name = "bf16",
        .block_size = 1,
        .bytes_per_block = sizeof(uint16_t),
        .is_quantized = false,
    },
    {
        .name = "q8_0",
        .block_size = 32,
        .bytes_per_block = sizeof(block_q8_0),
        .is_quantized = true,
    },
};

std::size_t dtype_size(DType dtype) {
  return type_info[dtype].bytes_per_block;
}

}  // namespace kiln
