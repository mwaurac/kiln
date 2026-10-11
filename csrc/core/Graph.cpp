#include <common/Error.h>
#include <core/Graph.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace kiln {
namespace {
thread_local std::vector<std::shared_ptr<Graph>> capture_stack;
}  // namespace

Node *Graph::add_node(std::string name, Op op, Tensor src1, Tensor src2, Tensor out) {
  auto n = std::make_unique<Node>(static_cast<std::int32_t>(nodes_.size()),
      std::move(name),
      op,
      std::move(src1),
      std::move(src2),
      std::move(out));

  Node *ptr = n.get();
  nodes_.push_back(std::move(n));

  const auto &impl = ptr->out_.impl();
  if (impl) {
    producer_[impl.get()] = ptr;
    try {
      impl->set_graph(shared_from_this());
    } catch (const std::bad_weak_ptr &) {
      KILN_ERROR(
          "Graph::add_node: graph must be held in a std::shared_ptr "
          "(or use default_graph()) so outputs can reference it");
    }
  }
  return ptr;
}

Node *Graph::add_node(std::string name, Op op, Tensor src, Tensor out) {
  static Tensor nil = Tensor::lazy({0});
  return add_node(std::move(name), op, std::move(src), nil, std::move(out));
}

Node *Graph::find(const Tensor &t) const {
  auto impl = t.impl();
  while (impl) {
    auto it = producer_.find(impl.get());
    if (it != producer_.end()) {
      return it->second;
    }
    impl = impl->base();
  }
  return nullptr;
}

std::shared_ptr<Graph> Graph::current() {
  if (capture_stack.empty()) {
    return nullptr;
  }
  return capture_stack.back();
}

void Graph::push_current(std::shared_ptr<Graph> graph) {
  capture_stack.push_back(std::move(graph));
}

void Graph::pop_current(const Graph *graph) {
  KILN_CHECK(!capture_stack.empty(), "Graph: capture stack underflow (unbalanced guard)");
  KILN_CHECK(capture_stack.back().get() == graph, "Graph: captures exited out of order");
  capture_stack.pop_back();
}

std::shared_ptr<Graph> Graph::resolve_output_graph(std::initializer_list<const Tensor *> inputs) {
  std::shared_ptr<Graph> resolved = current();
  for (const Tensor *t : inputs) {
    std::shared_ptr<Graph> owner = t->impl()->graph();
    if (!owner) {
      continue;
    }
    if (!resolved) {
      resolved = owner;  // inherit the input's graph outside any capture
      continue;
    }
    KILN_CHECK(owner == resolved,
        "graph mismatch: op inputs belong to different capture graphs; "
        "build all inputs of an op inside the same `with` block");
  }
  return resolved;  // null means the default graph
}

void Graph::begin_capture() {
  captures_.push_back(std::make_unique<CaptureGuard>(shared_from_this()));
}

void Graph::end_capture() {
  KILN_CHECK(!captures_.empty(), "Graph: end_capture without an active `with` block");
  captures_.pop_back();
}

CaptureGuard::CaptureGuard(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {
  Graph::push_current(graph_);
}

CaptureGuard::~CaptureGuard() {
  Graph::pop_current(graph_.get());
}

Graph &default_graph() {
  return *default_graph_ptr();
}

std::shared_ptr<Graph> default_graph_ptr() {
  static std::shared_ptr<Graph> instance = std::make_shared<Graph>();
  return instance;
}
}  // namespace kiln
