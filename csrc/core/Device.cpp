#include <common/Error.h>
#include <core/Device.h>

#include <string>

namespace kiln {

const std::string device_name(Device d) {
  switch (d) {
    case Device::CPU:
      return "cpu";
    case Device::CUDA:
      return "cuda";
    default:
      KILN_ERROR("Unsupported device");
  }
}
}  // namespace kiln
