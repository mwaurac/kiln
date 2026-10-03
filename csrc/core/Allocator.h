#pragma once
#include <core/Device.h>

#include <cstddef>

namespace kiln {

class Buffer {
 public:
  Buffer(std::size_t bytes, Device device);
  ~Buffer();

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  std::size_t size() const;
  void *base() const;
  Device device() const;

 private:
  void *ptr_ = nullptr;
  std::size_t size_ = 0;
  Device device_ = Device::CPU;
};
}  // namespace kiln
