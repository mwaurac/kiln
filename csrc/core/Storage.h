#pragma once

#include <core/Allocator.h>
#include <core/Device.h>

#include <cstddef>
#include <memory>

#include "core/Dtype.h"

namespace kiln {
class Storage {
 public:
  // Owned, uninitialized storage
  Storage(std::size_t numel, DType dtype, Device device);
  // Borrowed storage
  Storage(void *data, std::size_t numel, DType dtype, Device device, std::shared_ptr<void> owner);
  Storage(const Storage &) = delete;
  Storage &operator=(const Storage &) = delete;
  ~Storage() = default;

  std::size_t nbytes() const;
  const void *data() const;
  void *data();

  Device device() const;
  DType dtype() const;

 private:
  std::shared_ptr<Buffer> buf_;
  void *borrowed_ = nullptr;
  std::shared_ptr<void> owner_;  // lifetime keeper
  std::size_t numel_;
  DType dtype_;
  Device device_;
};
}  // namespace kiln
