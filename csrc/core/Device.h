#pragma once

#include <cstdint>
#include <string>

namespace kiln {
enum class Device : uint8_t {
  CPU = 0,
  CUDA = 1,
};

const std::string device_name(Device device);
}  // namespace kiln
