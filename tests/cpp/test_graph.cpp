#include <core/Dtype.h>
#include <core/Executor.h>
#include <core/Graph.h>
#include <core/Tensor.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "helpers.h"

TEST(Graph, FindReturnsProducerOnly) {
  kiln::Tensor src = kiln_test::filled_f32({2, 2});
  kiln::Tensor out = kiln::Tensor::lazy({2, 2});
  auto &g = kiln::default_graph();
  kiln::Node *n = g.add_node("copy", kiln::Op::COPY, src, out);
  EXPECT_EQ(g.find(out), n);
  EXPECT_EQ(g.find(kiln_test::filled_f32({2, 2})), nullptr);
}

TEST(Graph, FindChasesViewsToProducer) {
  kiln::Tensor src = kiln_test::filled_f32({2, 3});
  kiln::Tensor mid = kiln::Tensor::lazy({2, 3});
  auto &g = kiln::default_graph();
  kiln::Node *n = g.add_node("copy", kiln::Op::COPY, src, mid);
  kiln::Tensor v = mid.transpose(0, 1);
  EXPECT_EQ(g.find(v), n);  // view chain resolves to mid's producer
  EXPECT_EQ(g.find(mid), n);
}

TEST(Graph, ScheduleRunsProducerBeforeConsumer) {
  kiln::Tensor src = kiln_test::filled_f32({4});
  kiln::Tensor a = kiln::Tensor::lazy({4});
  kiln::Tensor b = kiln::Tensor::lazy({4});
  auto &g = kiln::default_graph();
  kiln::Node *n1 = g.add_node("n1", kiln::Op::COPY, src, a);
  kiln::Node *n2 = g.add_node("n2", kiln::Op::COPY, a, b);
  kiln::Executor ex;
  ex.schedule(g, b);
  ASSERT_EQ(ex.order().size(), 2u);
  EXPECT_EQ(ex.order()[0], n1);
  EXPECT_EQ(ex.order()[1], n2);
}

TEST(Graph, ScheduleExcludesUnrelatedNodes) {
  kiln::Tensor src = kiln_test::filled_f32({4});
  kiln::Tensor a = kiln::Tensor::lazy({4});
  kiln::Tensor other = kiln::Tensor::lazy({4});
  auto &g = kiln::default_graph();
  g.add_node("wanted", kiln::Op::COPY, src, a);
  g.add_node("unrelated", kiln::Op::COPY, src, other);
  kiln::Executor ex;
  ex.schedule(g, a);
  ASSERT_EQ(ex.order().size(), 1u);
  EXPECT_EQ(ex.order()[0]->out_.impl().get(), a.impl().get());
  EXPECT_FALSE(other.has_storage());  // untouched by the run below
  ex.run();
  EXPECT_TRUE(a.has_storage());
  EXPECT_FALSE(other.has_storage());
}

TEST(Executor, NoneNodeOnlyAllocates) {
  kiln::Tensor out = kiln::Tensor::lazy({3});
  auto &g = kiln::default_graph();
  kiln::Tensor dummy = kiln::Tensor::lazy({3});
  g.add_node("alloc", kiln::Op::NONE, dummy, out);
  kiln::execute(out);
  EXPECT_TRUE(out.has_storage());
}

TEST(Executor, ExecuteWithoutProducerIsNoop) {
  kiln::Tensor t = kiln_test::filled_f32({2});
  EXPECT_NO_THROW(kiln::execute(t));
  EXPECT_EQ(kiln_test::as_vec(t), (std::vector<float>{0, 1}));
}

TEST(Executor, CopyWithUnexecutedSourceThrows) {
  kiln::Tensor src = kiln::Tensor::lazy({2});  // no producer, never runs
  kiln::Tensor out = kiln::Tensor::lazy({2});
  auto &g = kiln::default_graph();
  g.add_node("copy", kiln::Op::COPY, src, out);
  EXPECT_THROW(kiln::execute(out), std::runtime_error);
}

TEST(Contiguous, DeferredThroughViewExecutes) {
  kiln::Tensor src = kiln_test::filled_f32({2, 3});  // [[0,1,2],[3,4,5]]
  auto &g = kiln::default_graph();
  const std::size_t before = g.size();

  kiln::Tensor mid = kiln::Tensor::lazy({2, 3});
  g.add_node("copy", kiln::Op::COPY, src, mid);
  kiln::Tensor midT = mid.transpose(0, 1);  // lazy strided view, shape [3, 2]
  ASSERT_FALSE(midT.has_storage());

  kiln::Tensor c = midT.contiguous();  // must record a node, not detach
  EXPECT_EQ(g.size(), before + 2);
  EXPECT_FALSE(c.has_storage());
  EXPECT_EQ(c.shape(), (kiln::Shape{3, 2}));

  kiln::execute(c);  // schedules through the view back to the COPY producer
  ASSERT_TRUE(c.has_storage());
  EXPECT_TRUE(c.is_contiguous());
  EXPECT_EQ(kiln_test::as_vec(c), (std::vector<float>{0, 3, 1, 4, 2, 5}));
}

TEST(Contiguous, LazyReshapeDefersThenExecutes) {
  kiln::Tensor src = kiln_test::filled_f32({2, 3});
  auto &g = kiln::default_graph();
  const std::size_t before = g.size();

  kiln::Tensor mid = kiln::Tensor::lazy({2, 3});
  g.add_node("copy", kiln::Op::COPY, src, mid);
  kiln::Tensor flat = mid.transpose(0, 1).reshape({6});
  EXPECT_EQ(g.size(), before + 2);  // COPY produce + deferred contiguous
  EXPECT_FALSE(flat.has_storage());

  kiln::execute(flat);
  EXPECT_EQ(kiln_test::as_vec(flat), (std::vector<float>{0, 3, 1, 4, 2, 5}));
}
