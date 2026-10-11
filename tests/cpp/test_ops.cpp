// M2 tasks 1-2 (C++ side): lazy builders validate and record; kernels are
// M2 task 3, so execution is *not* covered here beyond "not implemented".
// Python covers dunders, capture ergonomics, and error propagation.

#include <core/Device.h>
#include <core/Dtype.h>
#include <core/Graph.h>
#include <core/Op.h>
#include <core/Tensor.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <stdexcept>

#include "helpers.h"

namespace {
kiln::Tensor owned(const kiln::Shape &shape, kiln::DType dtype = kiln::DType::F32) {
  return kiln::Tensor(shape, dtype, kiln::Device::CPU);
}
}  // namespace

TEST(Ops, AddInfersShapeAndStaysLazy) {
  kiln::Tensor y = kiln::add(owned({2, 3}), owned({2, 3}));
  EXPECT_EQ(y.shape(), (kiln::Shape{2, 3}));
  EXPECT_FALSE(y.has_storage());
}

TEST(Ops, MulMatchesAddRules) {
  kiln::Tensor y = kiln::mul(owned({4}), owned({4}));
  EXPECT_EQ(y.shape(), (kiln::Shape{4}));
  EXPECT_FALSE(y.has_storage());
}

TEST(Ops, MatmulInfersOuterDims) {
  kiln::Tensor y = kiln::matmul(owned({4, 8}), owned({8, 16}));
  EXPECT_EQ(y.shape(), (kiln::Shape{4, 16}));
  EXPECT_FALSE(y.has_storage());
}

TEST(Ops, UnaryOpsPassShapeThrough) {
  for (auto op : {kiln::relu, kiln::silu}) {
    kiln::Tensor y = op(owned({2, 3}));
    EXPECT_EQ(y.shape(), (kiln::Shape{2, 3}));
    EXPECT_FALSE(y.has_storage());
  }
  kiln::Tensor s = kiln::silu(owned({}));
  EXPECT_EQ(s.shape(), (kiln::Shape{}));
}

TEST(Ops, SoftmaxNormalizesAxis) {
  auto &g = kiln::default_graph();
  kiln::Tensor a = kiln::softmax(owned({2, 3}), -1);
  EXPECT_EQ(g.find(a)->axis_, 1);
  kiln::Tensor b = kiln::softmax(owned({2, 3}), 0);
  EXPECT_EQ(g.find(b)->axis_, 0);
  kiln::Tensor c = kiln::softmax(owned({2, 3}));  // default -1
  EXPECT_EQ(g.find(c)->axis_, 1);
}

TEST(Ops, BuildersRecordIntoCurrentGraph) {
  auto g = std::make_shared<kiln::Graph>();
  const std::size_t before = g->size();
  kiln::Tensor y = kiln::Tensor::lazy({2});
  {
    kiln::CaptureGuard cap(g);
    y = kiln::add(owned({2}), owned({2}));
  }
  EXPECT_EQ(g->size(), before + 1);
  EXPECT_NE(g->find(y), nullptr);
  EXPECT_EQ(y.impl()->graph(), g);  // result bound to its graph
}

TEST(Ops, BuildersRejectMismatchedShapes) {
  EXPECT_THROW(kiln::add(owned({2, 3}), owned({3, 2})), std::runtime_error);
  EXPECT_THROW(kiln::mul(owned({6}), owned({2, 3})), std::runtime_error);
}

TEST(Ops, BuildersRejectMismatchedDtypes) {
  EXPECT_THROW(kiln::add(owned({2}, kiln::DType::F32), owned({2}, kiln::DType::F16)),
      std::runtime_error);
}

TEST(Ops, BuildersRejectQuantizedDtypes) {
  EXPECT_THROW(kiln::add(owned({32}, kiln::DType::Q8_0), owned({32}, kiln::DType::Q8_0)),
      std::runtime_error);
  EXPECT_THROW(kiln::relu(owned({32}, kiln::DType::Q8_0)), std::runtime_error);
}

TEST(Ops, MatmulRejectsRankAndInnerMismatch) {
  EXPECT_THROW(kiln::matmul(owned({2, 3, 4}), owned({4, 5})), std::runtime_error);
  EXPECT_THROW(kiln::matmul(owned({4, 8}), owned({7, 16})), std::runtime_error);
}

TEST(Ops, SoftmaxRejectsBadAxisAndRank) {
  EXPECT_THROW(kiln::softmax(owned({2, 3}), 2), std::runtime_error);
  EXPECT_THROW(kiln::softmax(owned({2, 3}), -3), std::runtime_error);
  EXPECT_THROW(kiln::softmax(owned({})), std::runtime_error);
}

TEST(Capture, NestedInnermostWins) {
  auto outer = std::make_shared<kiln::Graph>();
  auto inner = std::make_shared<kiln::Graph>();
  EXPECT_EQ(kiln::Graph::current(), nullptr);
  {
    kiln::CaptureGuard c1(outer);
    EXPECT_EQ(kiln::Graph::current(), outer);
    {
      kiln::CaptureGuard c2(inner);
      EXPECT_EQ(kiln::Graph::current(), inner);
      kiln::add(owned({1}), owned({1}));
    }
    EXPECT_EQ(kiln::Graph::current(), outer);
    EXPECT_EQ(inner->size(), 1u);
    EXPECT_EQ(outer->size(), 0u);
  }
  EXPECT_EQ(kiln::Graph::current(), nullptr);
}

TEST(Capture, MixingGraphsThrows) {
  auto g1 = std::make_shared<kiln::Graph>();
  auto g2 = std::make_shared<kiln::Graph>();
  kiln::Tensor y1 = kiln::Tensor::lazy({2});
  {
    kiln::CaptureGuard c1(g1);
    y1 = kiln::add(owned({2}), owned({2}));
  }
  {
    kiln::CaptureGuard c2(g2);
    EXPECT_THROW(kiln::add(y1, owned({2})), std::runtime_error);  // y1 lives in g1
    EXPECT_THROW(kiln::mul(owned({2}), y1), std::runtime_error);
  }
  EXPECT_EQ(g2->size(), 0u);  // failed builds record nothing
}

TEST(Capture, ContinuesInputGraphOutsideCapture) {
  auto g = std::make_shared<kiln::Graph>();
  kiln::Tensor y = kiln::Tensor::lazy({3});
  {
    kiln::CaptureGuard cap(g);
    y = kiln::relu(owned({3}));
  }
  const std::size_t before = g->size();
  kiln::Tensor z = kiln::add(y, owned({3}));  // no capture: inherits y's graph
  EXPECT_EQ(z.impl()->graph(), g);
  EXPECT_EQ(g->size(), before + 1);
}

TEST(Capture, MaterializedInputsDefaultToDefaultGraph) {
  kiln::Tensor y = kiln::add(owned({2}), owned({2}));
  EXPECT_EQ(y.impl()->graph(), kiln::default_graph_ptr());
  EXPECT_NE(kiln::default_graph().find(y), nullptr);
}
