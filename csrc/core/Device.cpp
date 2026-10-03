#include <core/Device.h>

#include <stdexcept>
#include <string>

namespace kiln {

const std::string device_name(Device d) {
  switch (d) {
    case Device::CPU:
      return "cpu";
    case Device::CUDA:
      return "cuda";
    default:
      throw std::invalid_argument("Unsupported device");
  }
}
}  // namespace kiln
