#include <core/Dtype.h>

#include <cstddef>
#include <stdexcept>
#include <string>

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

std::size_t dtype_itemsize(DType dtype) {
  if (dtype >= DType::DTYPE_COUNT) {
    throw std::runtime_error("dtype_itemsize: unknown dtype");
  }
  const DTypeInfo &info = type_info[dtype];
  if (info.is_quantized) {
    throw std::runtime_error(std::string("dtype_itemsize: ") + info.name + " is block-quantized (" +
                             std::to_string(info.block_size) + " elements per " +
                             std::to_string(info.bytes_per_block) + "-byte block) " +
                             "and has no scalar itemsize; use nbytes(numel, dtype)");
  }
  return info.bytes_per_block;
}

std::size_t nbytes(std::size_t numel, DType dtype) {
  if (dtype >= DType::DTYPE_COUNT) {
    throw std::runtime_error("nbytes: unknown dtype");
  }
  const DTypeInfo &info = type_info[dtype];
  if (!info.is_quantized) {
    return numel * info.bytes_per_block;
  }
  if (numel % info.block_size != 0) {
    throw std::runtime_error("nbytes: quantized dtype " + std::string(info.name) +
                             " requires numel to be a multiple of block size " +
                             std::to_string(info.block_size) + ", got " + std::to_string(numel));
  }
  return (numel / info.block_size) * info.bytes_per_block;
}

bool dtype_is_quantized(DType dtype) {
  if (dtype >= DType::DTYPE_COUNT) {
    throw std::runtime_error("dtype_is_quantized: unknown dtype");
  }
  return type_info[dtype].is_quantized;
}

std::string dtype_name(DType dtype) {
  return type_info[dtype].name;
}
}  // namespace kiln
