#pragma once

#include <core/Graph.h>
#include <core/Tensor.h>

#include <unordered_set>
#include <vector>

namespace kiln {
class Executor {
 private:
  std::vector<Node *> order_;
  void visit(const Graph &g, Node *n, std::unordered_set<Node *> &seen);

 public:
  const std::vector<Node *> &order() const;

  void schedule(const Graph &g, const Tensor &output);
  void execute(Node *n);
  void run();
};

void execute(const Tensor &output);
}  // namespace kiln
