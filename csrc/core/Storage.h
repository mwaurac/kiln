#pragma once

#include <core/Allocator.h>
#include <core/Device.h>

#include <cstddef>
#include <memory>

#include "core/Dtype.h"

namespace kiln {
class Storage {
 public:
  enum class Kind { Owned, Borrowed, Arena };

  // Owned, uninitialized storage
  Storage(std::size_t numel, DType dtype, Device device);
  // Borrowed mutable storage
  Storage(void *data, std::size_t numel, DType dtype, Device device, std::shared_ptr<void> owner);
  // Borrowed read-only storage
  Storage(const void *data,
      std::size_t numel,
      DType dtype,
      Device device,
      std::shared_ptr<void> owner);
  Storage(const Storage &) = delete;
  Storage &operator=(const Storage &) = delete;
  ~Storage() = default;

  std::size_t nbytes() const;
  const void *data() const;
  void *data();

  Device device() const;
  DType dtype() const;

  Kind kind() const;
  bool is_owned() const;
  bool is_borrowed() const;
  bool is_readonly() const;

 private:
  std::shared_ptr<Buffer> buf_;
  const void *borrowed_ = nullptr;
  std::shared_ptr<void> owner_;  // lifetime keeper
  bool readonly_ = false;
  std::size_t numel_;
  DType dtype_;
  Device device_;
};
}  // namespace kiln
