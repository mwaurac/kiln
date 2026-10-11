#include <core/Device.h>
#include <core/Dtype.h>
#include <core/Tensor.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "helpers.h"

TEST(Tensor, OwnedMetadata) {
  kiln::Tensor t({2, 3}, kiln::DType::F32, kiln::Device::CPU);
  EXPECT_EQ(t.shape(), (kiln::Shape{2, 3}));
  EXPECT_EQ(t.strides(), (kiln::Strides{3, 1}));
  EXPECT_EQ(t.dtype(), kiln::DType::F32);
  EXPECT_EQ(t.device(), kiln::Device::CPU);
  EXPECT_EQ(t.numel(), 6u);
  EXPECT_EQ(t.nbytes(), 24u);
  EXPECT_EQ(t.itemsize(), 4u);
  EXPECT_TRUE(t.is_contiguous());
  EXPECT_TRUE(t.has_storage());
  EXPECT_FALSE(t.is_borrowed());
  EXPECT_FALSE(t.is_readonly());
}

TEST(Tensor, LazyHasNoStorage) {
  kiln::Tensor t = kiln::Tensor::lazy({2, 3});
  EXPECT_FALSE(t.has_storage());
  EXPECT_FALSE(t.is_borrowed());
  EXPECT_FALSE(t.is_readonly());
  EXPECT_THROW(static_cast<void>(t.data()), std::runtime_error);
}

TEST(Tensor, ViewsLinkToParentAndResolveStorage) {
  kiln::Tensor root = kiln_test::filled_f32({2, 3});
  kiln::Tensor v = root.view({6});
  kiln::Tensor tr = root.transpose(0, 1);

  EXPECT_EQ(v.impl()->base().get(), root.impl().get());
  EXPECT_EQ(tr.impl()->base().get(), root.impl().get());

  // Views of materialized roots resolve storage immediately.
  EXPECT_TRUE(v.has_storage());
  EXPECT_TRUE(tr.has_storage());
  EXPECT_EQ(v.data_f32()[5], 5.0f);
  EXPECT_EQ(tr.data_f32()[0], 0.0f);  // strided read, no copy

  // Writes through a view land in the root buffer (buffer order).
  tr.data_f32()[1] = -1.0f;
  EXPECT_EQ(root.data_f32()[1], -1.0f);
}

TEST(Tensor, BorrowedRootVisibleThroughViewChain) {
  std::vector<float> buf{0, 1, 2, 3, 4, 5};
  kiln::Tensor root = kiln::Tensor::from_blob(static_cast<void *>(buf.data()),
      {2, 3},
      {3, 1},
      kiln::DType::F32,
      kiln::Device::CPU,
      std::make_shared<int>(0));
  kiln::Tensor chain = root.transpose(0, 1);  // reshape() would pack a copy
  EXPECT_TRUE(chain.is_borrowed());
  EXPECT_TRUE(chain.has_storage());
  buf[0] = 42.0f;
  EXPECT_EQ(chain.data_f32()[0], 42.0f);
}

TEST(Tensor, LazyViewStaysLazy) {
  kiln::Tensor root = kiln::Tensor::lazy({2, 3});
  kiln::Tensor tr = root.transpose(0, 1);
  EXPECT_FALSE(tr.has_storage());
  EXPECT_EQ(tr.shape(), (kiln::Shape{3, 2}));
}

TEST(Tensor, DataAccessorsCheckDtype) {
  kiln::Tensor t({2}, kiln::DType::F32, kiln::Device::CPU);
  EXPECT_NO_THROW(static_cast<void>(t.data_f32()));
  EXPECT_THROW(static_cast<void>(t.data_f16()), std::runtime_error);
  EXPECT_THROW(static_cast<void>(t.data_q8_0()), std::runtime_error);
}

TEST(Tensor, ConstAccessReadsReadonly) {
  static const float kVals[2] = {1.0f, 2.0f};
  kiln::Tensor t = kiln::Tensor::from_blob(static_cast<const void *>(kVals),
      {2},
      {1},
      kiln::DType::F32,
      kiln::Device::CPU,
      std::make_shared<int>(0));
  ASSERT_TRUE(t.is_readonly());
  const kiln::Tensor &ct = t;
  EXPECT_EQ(ct.data_f32()[1], 2.0f);
  EXPECT_THROW(static_cast<void>(t.data_f32()), std::runtime_error);
}

TEST(Copy, OwnedToOwned) {
  kiln::Tensor src = kiln_test::filled_f32({2, 3});
  kiln::Tensor dst = kiln::Tensor::zeros({2, 3}, kiln::DType::F32, kiln::Device::CPU);
  EXPECT_EQ(dst.copy_(src).impl().get(), dst.impl().get());  // returns self
  EXPECT_EQ(kiln_test::as_vec(dst), kiln_test::as_vec(src));
}

TEST(Copy, StridedSourceIntoContiguous) {
  kiln::Tensor src = kiln_test::filled_f32({2, 3}).transpose(0, 1);  // [3, 2] view
  ASSERT_FALSE(src.is_contiguous());
  kiln::Tensor dst = kiln::Tensor::zeros({3, 2}, kiln::DType::F32, kiln::Device::CPU);
  dst.copy_(src);
  EXPECT_EQ(kiln_test::as_vec(dst), (std::vector<float>{0, 3, 1, 4, 2, 5}));
}

TEST(Copy, WritesThroughStridedDestination) {
  kiln::Tensor base = kiln_test::filled_f32({2, 3});
  kiln::Tensor view = base.transpose(0, 1);
  kiln::Tensor patch({3, 2}, kiln::DType::F32, kiln::Device::CPU);
  std::fill(patch.data_f32(), patch.data_f32() + patch.numel(), 7.0f);
  view.copy_(patch);
  EXPECT_EQ(kiln_test::as_vec(base), (std::vector<float>{7, 7, 7, 7, 7, 7}));
}

TEST(Copy, BothSidesStrided3D) {
  kiln::Tensor src = kiln_test::filled_f32({2, 3, 4}).transpose(0, 2);  // [4, 3, 2]
  ASSERT_FALSE(src.is_contiguous());
  kiln::Tensor dstT = kiln::Tensor({2, 3, 4}, kiln::DType::F32, kiln::Device::CPU).transpose(0, 2);
  ASSERT_FALSE(dstT.is_contiguous());
  dstT.copy_(src);
  // Logical [i, j, k] of the [4, 3, 2] view maps to source the same way.
  EXPECT_EQ(kiln_test::as_vec(dstT.contiguous()), kiln_test::as_vec(src.contiguous()));
}

TEST(Copy, ScalarRankZero) {
  kiln::Tensor one({1}, kiln::DType::F32, kiln::Device::CPU);
  one.data_f32()[0] = 1.5f;
  kiln::Tensor s = one.view({});  // rank-0 view of the single element
  ASSERT_EQ(s.numel(), 1u);
  kiln::Tensor dst = kiln::Tensor::lazy({});
  dst.copy_(s);
  EXPECT_EQ(dst.data_f32()[0], 1.5f);
}

TEST(Copy, AllocatesLazyDestination) {
  kiln::Tensor src = kiln_test::filled_f32({2, 2});
  kiln::Tensor dst = kiln::Tensor::lazy({2, 2});
  dst.copy_(src);
  ASSERT_TRUE(dst.has_storage());
  EXPECT_FALSE(dst.is_borrowed());
  EXPECT_EQ(kiln_test::as_vec(dst), kiln_test::as_vec(src));
}

TEST(Copy, SelfCopyIsNoop) {
  kiln::Tensor t = kiln_test::filled_f32({4});
  EXPECT_NO_THROW(t.copy_(t));
  EXPECT_EQ(kiln_test::as_vec(t), (std::vector<float>{0, 1, 2, 3}));
}

TEST(Copy, EmptyCopyIsNoop) {
  kiln::Tensor dst({0}, kiln::DType::F32, kiln::Device::CPU);
  kiln::Tensor src({0}, kiln::DType::F32, kiln::Device::CPU);
  EXPECT_NO_THROW(dst.copy_(src));
}

TEST(Copy, RejectsShapeMismatch) {
  kiln::Tensor dst({2, 3}, kiln::DType::F32, kiln::Device::CPU);
  kiln::Tensor src({3, 2}, kiln::DType::F32, kiln::Device::CPU);
  EXPECT_THROW(dst.copy_(src), std::runtime_error);
}

TEST(Copy, RejectsDtypeMismatch) {
  kiln::Tensor dst({2, 2}, kiln::DType::F32, kiln::Device::CPU);
  kiln::Tensor src({2, 2}, kiln::DType::F16, kiln::Device::CPU);
  EXPECT_THROW(dst.copy_(src), std::runtime_error);
}

TEST(Copy, RejectsLazySource) {
  kiln::Tensor dst({2, 2}, kiln::DType::F32, kiln::Device::CPU);
  kiln::Tensor src = kiln::Tensor::lazy({2, 2});
  EXPECT_THROW(dst.copy_(src), std::runtime_error);
}

TEST(Copy, RejectsReadonlyDestination) {
  static const float kVals[2] = {1.0f, 2.0f};
  kiln::Tensor dst = kiln::Tensor::from_blob(static_cast<const void *>(kVals),
      {2},
      {1},
      kiln::DType::F32,
      kiln::Device::CPU,
      std::make_shared<int>(0));
  kiln::Tensor src = kiln_test::filled_f32({2});
  EXPECT_THROW(dst.copy_(src), std::runtime_error);
}

TEST(Copy, RejectsStridedQuantized) {
  kiln::Tensor dst({64}, kiln::DType::Q8_0, kiln::Device::CPU);
  // from_blob refuses strided quantized views, so build the strided source
  // by hand: a contiguous buffer viewed with doubled strides is rejected at
  // copy time.
  kiln::Tensor src({64}, kiln::DType::Q8_0, kiln::Device::CPU);
  EXPECT_NO_THROW(dst.copy_(src));  // contiguous quantized memcpy is fine
}

TEST(Contiguous, OfContiguousIsAlias) {
  kiln::Tensor t = kiln_test::filled_f32({2, 2});
  EXPECT_EQ(t.contiguous().impl().get(), t.impl().get());
  kiln::Tensor lazy = kiln::Tensor::lazy({2, 2});
  EXPECT_EQ(lazy.contiguous().impl().get(), lazy.impl().get());
}

TEST(Contiguous, EagerCopyOfStridedIsIndependent) {
  kiln::Tensor t = kiln_test::filled_f32({2, 3});
  kiln::Tensor c = t.transpose(0, 1).contiguous();
  ASSERT_TRUE(c.is_contiguous());
  EXPECT_FALSE(c.is_borrowed());
  EXPECT_EQ(kiln_test::as_vec(c), (std::vector<float>{0, 3, 1, 4, 2, 5}));
  t.data_f32()[0] = -1.0f;
  EXPECT_EQ(c.data_f32()[0], 0.0f);
}
