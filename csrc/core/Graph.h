#pragma once

#include <core/Op.h>
#include <core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kiln {

struct Node {
  int32_t id_ = -1;
  std::string name_;
  Op op_ = Op::NONE;
  Tensor src1_;
  Tensor src2_;
  Tensor out_;
  // Op payload, e.g. the normalized axis for SOFTMAX. -1 when unused.
  std::int64_t axis_ = -1;

  Node(int32_t id, std::string name, Op op, Tensor src1, Tensor src2, Tensor out)
      : id_(id),
        name_(std::move(name)),
        op_(op),
        src1_(std::move(src1)),
        src2_(std::move(src2)),
        out_(std::move(out)) {}
};

struct CaptureGuard {
  explicit CaptureGuard(std::shared_ptr<Graph> graph);
  ~CaptureGuard();
  CaptureGuard(const CaptureGuard &) = delete;
  CaptureGuard &operator=(const CaptureGuard &) = delete;

 private:
  std::shared_ptr<Graph> graph_;
};

class Graph : public std::enable_shared_from_this<Graph> {
 public:
  Node *add_node(std::string name, Op op, Tensor src1, Tensor src2, Tensor out);
  Node *add_node(std::string name, Op op, Tensor src, Tensor out);
  Node *find(const Tensor &t) const;

  static std::shared_ptr<Graph> current();
  static std::shared_ptr<Graph> resolve_output_graph(std::initializer_list<const Tensor *> inputs);

  void begin_capture();
  void end_capture();

  std::size_t size() const {
    return nodes_.size();
  }
  const std::vector<std::unique_ptr<Node>> &nodes() const {
    return nodes_;
  }

 private:
  friend struct CaptureGuard;
  static void push_current(std::shared_ptr<Graph> graph);
  static void pop_current(const Graph *graph);

  std::vector<std::unique_ptr<Node>> nodes_;
  std::vector<std::unique_ptr<CaptureGuard>> captures_;
  mutable std::unordered_map<const TensorImpl *, Node *> producer_;
};

Graph &default_graph();
std::shared_ptr<Graph> default_graph_ptr();
}  // namespace kiln
