#include <common/Error.h>
#include <core/Dtype.h>
#include <core/Graph.h>
#include <core/Layout.h>
#include <core/Op.h>

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>

namespace kiln {
namespace {
void require_same_shape(const char *op, const Tensor &a, const Tensor &b) {
  KILN_CHECK(a.shape() == b.shape(),
      op,
      ": shape mismatch: ",
      shape_string(a.shape()),
      " vs ",
      shape_string(b.shape()),
      " (broadcasting is not supported yet)");
}

void require_same_dtype(const char *op, const Tensor &a, const Tensor &b) {
  KILN_CHECK(a.dtype() == b.dtype(),
      op,
      ": dtype mismatch: ",
      dtype_name(a.dtype()),
      " vs ",
      dtype_name(b.dtype()));
}

void require_same_device(const char *op, const Tensor &a, const Tensor &b) {
  KILN_CHECK(a.device() == b.device(),
      op,
      ": device mismatch: ",
      device_name(a.device()),
      " vs ",
      device_name(b.device()));
}

void require_compute_dtype(const char *op, DType dtype) {
  KILN_CHECK(!dtype_is_quantized(dtype),
      op,
      ": quantized dtype ",
      dtype_name(dtype),
      " is not supported by compute ops");
}

Graph &target_graph(const std::shared_ptr<Graph> &graph) {
  return graph ? *graph : default_graph();
}

Tensor lazy_out(const Shape &shape,
    DType dtype,
    Device device,
    std::initializer_list<const Tensor *> inputs,
    const char *name,
    Op op,
    const Tensor *second = nullptr) {
  std::shared_ptr<Graph> graph = Graph::resolve_output_graph(inputs);
  Tensor out = Tensor::lazy(shape, dtype, device);
  const Tensor &first = **inputs.begin();
  if (second) {
    target_graph(graph).add_node(name, op, first, *second, out);
  } else {
    target_graph(graph).add_node(name, op, first, out);
  }
  return out;
}
}  // namespace

Tensor add(const Tensor &a, const Tensor &b) {
  require_same_shape("add", a, b);
  require_same_dtype("add", a, b);
  require_same_device("add", a, b);
  require_compute_dtype("add", a.dtype());
  return lazy_out(a.shape(), a.dtype(), a.device(), {&a, &b}, "add", Op::ADD, &b);
}

Tensor mul(const Tensor &a, const Tensor &b) {
  require_same_shape("mul", a, b);
  require_same_dtype("mul", a, b);
  require_same_device("mul", a, b);
  require_compute_dtype("mul", a.dtype());
  return lazy_out(a.shape(), a.dtype(), a.device(), {&a, &b}, "mul", Op::MUL, &b);
}

Tensor matmul(const Tensor &a, const Tensor &b) {
  KILN_CHECK(a.shape().size() == 2 && b.shape().size() == 2,
      "matmul: only 2D tensors are supported, got shapes ",
      shape_string(a.shape()),
      " and ",
      shape_string(b.shape()));
  KILN_CHECK(a.shape()[1] == b.shape()[0],
      "matmul: inner dimensions must match, got shapes ",
      shape_string(a.shape()),
      " and ",
      shape_string(b.shape()));
  require_same_dtype("matmul", a, b);
  require_same_device("matmul", a, b);
  require_compute_dtype("matmul", a.dtype());
  return lazy_out({a.shape()[0], b.shape()[1]},
      a.dtype(),
      a.device(),
      {&a, &b},
      "matmul",
      Op::MATMUL,
      &b);
}

Tensor relu(const Tensor &x) {
  require_compute_dtype("relu", x.dtype());
  return lazy_out(x.shape(), x.dtype(), x.device(), {&x}, "relu", Op::RELU);
}

Tensor silu(const Tensor &x) {
  require_compute_dtype("silu", x.dtype());
  return lazy_out(x.shape(), x.dtype(), x.device(), {&x}, "silu", Op::SILU);
}

Tensor softmax(const Tensor &x, int64_t axis) {
  require_compute_dtype("softmax", x.dtype());
  const auto rank = static_cast<int64_t>(x.shape().size());
  KILN_CHECK(rank >= 1, "softmax: expected rank >= 1, got shape ", shape_string(x.shape()));
  if (axis < 0) {
    axis += rank;
  }
  KILN_CHECK(axis >= 0 && axis < rank,
      "softmax: axis (",
      axis,
      ") out of range for shape ",
      shape_string(x.shape()));
  std::shared_ptr<Graph> graph = Graph::resolve_output_graph({&x});
  Tensor out = Tensor::lazy(x.shape(), x.dtype(), x.device());
  Node *node = target_graph(graph).add_node("softmax", Op::SOFTMAX, x, out);
  node->axis_ = axis;
  return out;
}
}  // namespace kiln
