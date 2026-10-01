#pragma once

#include <core/Allocator.h>
#include <core/Device.h>

#include <cstddef>
#include <memory>

#include "core/Dtype.h"

namespace kiln {
class Storage {
 public:
  Storage(std::size_t numel, DType dtype, Device device);
  Storage(const Storage &) = delete;
  Storage &operator=(const Storage &) = delete;
  ~Storage() = default;

  std::size_t nbytes(std::size_t numel) const;
  const void *data() const;

  Device device() const;

 private:
  std::shared_ptr<Buffer> buf_;
  DType dtype_;
};
}  // namespace kiln
