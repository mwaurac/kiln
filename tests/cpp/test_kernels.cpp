// M2 reference kernels: F32 correctness with strided inputs, row-bias
// broadcast, and rerun-with-new-inputs behavior.

#include <core/Device.h>
#include <core/Dtype.h>
#include <core/Executor.h>
#include <core/Graph.h>
#include <core/Op.h>
#include <core/Tensor.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "helpers.h"

namespace {

kiln::Tensor like(const std::vector<float> &vals, const kiln::Shape &shape) {
  kiln::Tensor t(shape, kiln::DType::F32, kiln::Device::CPU);
  EXPECT_EQ(t.numel(), vals.size());
  float *p = t.data_f32();
  for (std::size_t i = 0; i < vals.size(); ++i) {
    p[i] = vals[i];
  }
  return t;
}

}  // namespace

TEST(Kernels, AddSameShape) {
  kiln::Tensor y = kiln::add(like({1, 2, 3, 4}, {2, 2}), like({5, 6, 7, 8}, {2, 2}));
  kiln::execute(y);
  EXPECT_EQ(kiln_test::as_vec(y), (std::vector<float>{6, 8, 10, 12}));
}

TEST(Kernels, AddRowBiasBroadcast) {
  // [2, 3] + [3], bias is the second operand.
  kiln::Tensor y = kiln::add(like({1, 2, 3, 4, 5, 6}, {2, 3}), like({10, 20, 30}, {3}));
  ASSERT_EQ(y.shape(), (kiln::Shape{2, 3}));
  kiln::execute(y);
  EXPECT_EQ(kiln_test::as_vec(y), (std::vector<float>{11, 22, 33, 14, 25, 36}));
}

TEST(Kernels, AddRowBiasBroadcastReversed) {
  kiln::Tensor y = kiln::add(like({10, 20, 30}, {3}), like({1, 2, 3, 4, 5, 6}, {2, 3}));
  kiln::execute(y);
  EXPECT_EQ(kiln_test::as_vec(y), (std::vector<float>{11, 22, 33, 14, 25, 36}));
}

TEST(Kernels, AddRejectsNonBroadcastable) {
  EXPECT_THROW(kiln::add(like({1, 2, 3, 4}, {2, 2}), like({1, 2, 3}, {3})), std::runtime_error);
}

TEST(Kernels, AddReadsStridedInputs) {
  kiln::Tensor a = like({0, 1, 2, 3, 4, 5}, {2, 3}).transpose(0, 1);  // [3, 2]
  kiln::Tensor b = like({0, 10, 20, 30, 40, 50}, {2, 3}).transpose(0, 1);
  kiln::Tensor y = kiln::add(a, b);
  kiln::execute(y);
  EXPECT_EQ(kiln_test::as_vec(y), (std::vector<float>{0, 33, 11, 44, 22, 55}));
}

TEST(Kernels, MatmulOuterDimsWithTransposedRhs) {
  // x [2, 3] @ w.T where w is [4, 3] owned -> [2, 4].
  kiln::Tensor x = like({1, 2, 3, 4, 5, 6}, {2, 3});
  kiln::Tensor w = like({1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}, {4, 3});
  kiln::Tensor y = kiln::matmul(x, w.transpose(0, 1));
  ASSERT_EQ(y.shape(), (kiln::Shape{2, 4}));
  kiln::execute(y);
  // Row 0 dot cols of w.T == rows of w: [1,2,3].[1,0,0]=1, .[0,1,0]=2, ...
  EXPECT_EQ(kiln_test::as_vec(y), (std::vector<float>{1, 2, 3, 6, 4, 5, 6, 15}));
}

TEST(Kernels, MulReluSiluParity) {
  kiln::Tensor m = kiln::mul(like({2, 3}, {2}), like({4, 5}, {2}));
  kiln::execute(m);
  EXPECT_EQ(kiln_test::as_vec(m), (std::vector<float>{8, 15}));

  kiln::Tensor r = kiln::relu(like({-1, 0, 2}, {3}));
  kiln::execute(r);
  EXPECT_EQ(kiln_test::as_vec(r), (std::vector<float>{0, 0, 2}));

  kiln::Tensor s = kiln::silu(like({0}, {1}));
  kiln::execute(s);
  EXPECT_NEAR(kiln_test::as_vec(s)[0], 0.0f, 1e-6f);
}

TEST(Kernels, SoftmaxSumsToOnePerRow) {
  kiln::Tensor s = kiln::softmax(like({1, 2, 3, 4}, {2, 2}), -1);
  kiln::execute(s);
  auto v = kiln_test::as_vec(s);
  ASSERT_EQ(v.size(), 4u);
  EXPECT_NEAR(v[0] + v[1], 1.0f, 1e-6f);
  EXPECT_NEAR(v[2] + v[3], 1.0f, 1e-6f);
  // Larger input dominates: softmax([1,2]) = [e/(e+e^2), e^2/(e+e^2)].
  const float e1 = std::exp(1.0f), e2 = std::exp(2.0f);
  EXPECT_NEAR(v[0], e1 / (e1 + e2), 1e-6f);
}

TEST(Kernels, SoftmaxAxisZero) {
  kiln::Tensor s = kiln::softmax(like({1, 3, 2, 4}, {2, 2}), 0);
  kiln::execute(s);
  auto v = kiln_test::as_vec(s);
  EXPECT_NEAR(v[0] + v[2], 1.0f, 1e-6f);
  EXPECT_NEAR(v[1] + v[3], 1.0f, 1e-6f);
}

TEST(Kernels, NonF32ExecutionRejected) {
  kiln::Tensor q({2}, kiln::DType::F16, kiln::Device::CPU);
  kiln::Tensor y = kiln::relu(q);
  EXPECT_THROW(kiln::execute(y), std::runtime_error);
}

TEST(Kernels, RerunPicksUpChangedInputs) {
  kiln::Tensor x = like({1, 1, 1, 1}, {2, 2});
  kiln::Tensor w = like({1, 0, 0, 1}, {2, 2});
  kiln::Tensor y = kiln::add(kiln::relu(x), w);
  kiln::execute(y);
  EXPECT_EQ(kiln_test::as_vec(y), (std::vector<float>{2, 1, 1, 2}));
  x.copy_(like({2, 2, 2, 2}, {2, 2}));
  kiln::execute(y);
  EXPECT_EQ(kiln_test::as_vec(y), (std::vector<float>{3, 2, 2, 3}));
}
