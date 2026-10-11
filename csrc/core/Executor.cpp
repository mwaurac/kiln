#include <common/Error.h>
#include <core/Executor.h>

#include <string>
#include <unordered_set>
#include <vector>

namespace kiln {
const std::vector<Node *> &Executor::order() const {
  return order_;
}

void Executor::visit(const Graph &g, Node *n, std::unordered_set<Node *> &seen) {
  if (n == nullptr || !seen.insert(n).second) {
    return;
  }

  const Tensor *inputs[] = {&n->src1_, &n->src2_};
  for (const Tensor *in : inputs) {
    if (Node *p = g.find(*in)) {
      visit(g, p, seen);
    }
  }

  order_.push_back(n);
}

void Executor::schedule(const Graph &g, const Tensor &output) {
  order_.clear();

  std::unordered_set<Node *> seen;
  if (Node *n = g.find(output)) {
    visit(g, n, seen);
  }
}

void Executor::execute(Node *n) {
  KILN_CHECK(n != nullptr, "Executor::execute: null node");

  switch (n->op_) {
    case Op::NONE:
      n->out_.allocate();
      break;
    case Op::COPY:
      if (!n->src1_.has_storage()) {
        KILN_ERROR("Executor::execute: COPY source has no storage; ",
            "its producer is missing or never ran");
      }
      n->out_.copy_(n->src1_);
      break;
    case Op::ADD:
    case Op::SUBTRACT:
    case Op::MATMUL:
    case Op::RELU:
    case Op::SOFTMAX:
      KILN_ERROR("Executor::execute: op not implemented: ", OP_NAME[n->op_]);
    default:
      KILN_ERROR("Executor::execute: unknown op");
  }
}

void Executor::run() {
  for (Node *n : order_) {
    execute(n);
  }
}

void execute(const Tensor &output) {
  Executor ex;
  ex.schedule(default_graph(), output);
  ex.run();
}
}  // namespace kiln
