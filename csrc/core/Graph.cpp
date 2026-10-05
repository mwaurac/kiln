#include <core/Graph.h>

#include <string>
#include <utility>

namespace kiln {
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
  }
  return ptr;
}

Node *Graph::find(const Tensor &t) const {
  const auto &impl = t.impl();
  if (!impl) {
    return nullptr;
  }
  auto it = producer_.find(impl.get());
  return it == producer_.end() ? nullptr : it->second;
}

Graph &default_graph() {
  static Graph g;
  return g;
}
}  // namespace kiln
