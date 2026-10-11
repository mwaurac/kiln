#include <core/Device.h>
#include <core/Dtype.h>
#include <core/Storage.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

std::shared_ptr<void> owner() {
  return std::make_shared<int>(0);
}

}  // namespace

TEST(Storage, OwnedReportsSizeAndKind) {
  kiln::Storage s(6, kiln::DType::F32, kiln::Device::CPU);
  EXPECT_EQ(s.nbytes(), 6 * 4);
  EXPECT_EQ(s.dtype(), kiln::DType::F32);
  EXPECT_EQ(s.device(), kiln::Device::CPU);
  EXPECT_EQ(s.kind(), kiln::Storage::Kind::Owned);
  EXPECT_TRUE(s.is_owned());
  EXPECT_FALSE(s.is_borrowed());
  EXPECT_FALSE(s.is_readonly());
  EXPECT_NE(s.data(), nullptr);
}

TEST(Storage, OwnedQuantizedBlockMath) {
  EXPECT_EQ(kiln::Storage(32, kiln::DType::Q8_0, kiln::Device::CPU).nbytes(), 34);
  EXPECT_EQ(kiln::Storage(0, kiln::DType::Q8_0, kiln::Device::CPU).nbytes(), 0);
  EXPECT_THROW(kiln::Storage(3, kiln::DType::Q8_0, kiln::Device::CPU), std::runtime_error);
}

TEST(Storage, BorrowedMutableRoundtrip) {
  std::vector<float> v{1.0f, 2.0f};
  kiln::Storage s(v.data(), 2, kiln::DType::F32, kiln::Device::CPU, owner());
  EXPECT_EQ(s.kind(), kiln::Storage::Kind::Borrowed);
  EXPECT_TRUE(s.is_borrowed());
  EXPECT_FALSE(s.is_owned());
  EXPECT_FALSE(s.is_readonly());
  EXPECT_EQ(s.data(), static_cast<const void *>(v.data()));
  // Mutable access writes through.
  *static_cast<float *>(s.data()) = 9.0f;
  EXPECT_EQ(v[0], 9.0f);
}

TEST(Storage, BorrowedReadonlyAllowsReadsRejectsWrites) {
  static const float kVals[2] = {1.0f, 2.0f};
  kiln::Storage s(static_cast<const void *>(kVals),
      2,
      kiln::DType::F32,
      kiln::Device::CPU,
      owner());
  EXPECT_TRUE(s.is_borrowed());
  EXPECT_TRUE(s.is_readonly());
  const kiln::Storage &cs = s;  // const access reads; mutable access throws
  EXPECT_EQ(static_cast<const float *>(cs.data())[1], 2.0f);
  EXPECT_THROW(static_cast<void>(s.data()), std::runtime_error);
}

TEST(Storage, BorrowedRequiresOwnerAndPointer) {
  float v = 0.0f;
  EXPECT_THROW(kiln::Storage(&v, 1, kiln::DType::F32, kiln::Device::CPU, nullptr),
      std::runtime_error);
  EXPECT_THROW(
      kiln::Storage(static_cast<void *>(nullptr), 2, kiln::DType::F32, kiln::Device::CPU, owner()),
      std::runtime_error);
  // Null pointer is fine when there are no bytes to point at.
  EXPECT_NO_THROW(
      kiln::Storage(static_cast<void *>(nullptr), 0, kiln::DType::F32, kiln::Device::CPU, owner()));
}

TEST(Storage, DtypeAndDeviceAccessors) {
  std::uint16_t raw[4] = {};
  kiln::Storage s(raw, 4, kiln::DType::F16, kiln::Device::CPU, owner());
  EXPECT_EQ(s.dtype(), kiln::DType::F16);
  EXPECT_EQ(s.device(), kiln::Device::CPU);
  EXPECT_EQ(s.nbytes(), 8);
}
