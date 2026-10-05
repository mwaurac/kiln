#pragma once

#include <core/Op.h>
#include <core/Tensor.h>

#include <cstddef>
#include <cstdint>
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

  Node(int32_t id, std::string name, Op op, Tensor src1, Tensor src2, Tensor out)
      : id_(id),
        name_(std::move(name)),
        op_(op),
        src1_(std::move(src1)),
        src2_(std::move(src2)),
        out_(std::move(out)) {}
};

class Graph {
 public:
  Node *add_node(std::string name, Op op, Tensor src1, Tensor src2, Tensor out);
  Node *find(const Tensor &t) const;

  std::size_t size() const { return nodes_.size(); }
  const std::vector<std::unique_ptr<Node>> &nodes() const { return nodes_; }

 private:
  std::vector<std::unique_ptr<Node>> nodes_;
  mutable std::unordered_map<const TensorImpl *, Node *> producer_;
};

Graph &default_graph();
}  // namespace kiln
