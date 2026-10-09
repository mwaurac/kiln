#include <common/Error.h>
#include <core/Dtype.h>

#include <cstddef>
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
  KILN_CHECK(dtype < DType::DTYPE_COUNT, "dtype_itemsize: unknown dtype");
  const DTypeInfo &info = type_info[dtype];
  KILN_CHECK(!info.is_quantized,
      "dtype_itemsize: ",
      info.name,
      " is block-quantized (",
      info.block_size,
      " elements per ",
      info.bytes_per_block,
      "-byte block) ",
      "and has no scalar itemsize; use nbytes(numel, dtype)");
  return info.bytes_per_block;
}

std::size_t nbytes(std::size_t numel, DType dtype) {
  KILN_CHECK(dtype < DType::DTYPE_COUNT, "nbytes: unknown dtype");
  const DTypeInfo &info = type_info[dtype];
  if (!info.is_quantized) {
    return numel * info.bytes_per_block;
  }
  KILN_CHECK(numel % info.block_size == 0,
      "nbytes: quantized dtype ",
      info.name,
      " requires numel to be a multiple of block size ",
      info.block_size,
      ", got ",
      numel);
  return (numel / info.block_size) * info.bytes_per_block;
}

bool dtype_is_quantized(DType dtype) {
  KILN_CHECK(dtype < DType::DTYPE_COUNT, "dtype_is_quantized: unknown dtype");
  return type_info[dtype].is_quantized;
}

std::string dtype_name(DType dtype) {
  return type_info[dtype].name;
}
}  // namespace kiln
