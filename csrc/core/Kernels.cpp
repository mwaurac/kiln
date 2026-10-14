#include <common/Error.h>
#include <core/Dtype.h>
#include <core/Kernels.h>
#include <core/Layout.h>
#include <core/Tensor.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace kiln::kernels {
namespace {

void require_f32(const Tensor &t, const char *op) {
  KILN_CHECK(t.dtype() == DType::F32,
      op,
      ": reference kernels only support f32, got dtype ",
      dtype_name(t.dtype()));
}

void require_has_storage(const Tensor &t, const char *op, const char *which) {
  KILN_CHECK(t.has_storage(),
      op,
      ": ",
      which,
      " has no storage (lazy output); its producer is missing or never ran; "
      "call execute() on the producer graph first");
}

const float *cread_f32(const Tensor &t) {
  return static_cast<const float *>(
             static_cast<const Storage *>(t.impl()->storage().get())->data()) +
         static_cast<std::size_t>(t.impl()->offset());
}

// Element offset odometer for one strided operand.
struct StrideCursor {
  Strides strides;
  Shape shape;
  std::vector<std::size_t> idx;
  std::size_t offset = 0;

  StrideCursor(const Tensor &t) : strides(t.strides()), shape(t.shape()), idx(shape.size(), 0) {}

  void advance() {
    for (std::size_t d = shape.size(); d-- > 0;) {
      offset += strides[d];
      if (++idx[d] < shape[d]) {
        return;
      }
      offset -= strides[d] * shape[d];
      idx[d] = 0;
    }
  }
};

void elementwise_binary(const char *op,
    const Tensor &a,
    const Tensor &b,
    Tensor &out,
    float (*fn)(float, float)) {
  require_f32(a, op);
  require_f32(b, op);
  require_f32(out, op);
  require_has_storage(a, op, "input a");
  require_has_storage(b, op, "input b");
  const std::size_t n = out.numel();
  if (n == 0) {
    return;
  }
  KILN_CHECK(a.shape() == b.shape() && a.shape() == out.shape(), op, ": internal shape mismatch");
  const float *pa = cread_f32(a);
  const float *pb = cread_f32(b);
  float *po = static_cast<float *>(out.impl()->storage()->data()) +
              static_cast<std::size_t>(out.impl()->offset());
  StrideCursor ca(a), cb(b);
  // Output is contiguous by construction; linear write is valid.
  for (std::size_t i = 0; i < n; ++i) {
    po[i] = fn(pa[ca.offset], pb[cb.offset]);
    ca.advance();
    cb.advance();
  }
}

float add_fn(float x, float y) {
  return x + y;
}
float mul_fn(float x, float y) {
  return x * y;
}

void elementwise_unary(const char *op, const Tensor &x, Tensor &out, float (*fn)(float)) {
  require_f32(x, op);
  require_f32(out, op);
  require_has_storage(x, op, "input");
  const std::size_t n = out.numel();
  if (n == 0) {
    return;
  }
  KILN_CHECK(x.shape() == out.shape(), op, ": internal shape mismatch");
  const float *px = cread_f32(x);
  float *po = static_cast<float *>(out.impl()->storage()->data()) +
              static_cast<std::size_t>(out.impl()->offset());
  StrideCursor cx(x);
  for (std::size_t i = 0; i < n; ++i) {
    po[i] = fn(px[cx.offset]);
    cx.advance();
  }
}

float relu_fn(float x) {
  return x > 0.0f ? x : 0.0f;
}
float silu_fn(float x) {
  return x / (1.0f + std::expf(-x));
}

}  // namespace

void add_f32(const Tensor &a, const Tensor &b, Tensor &out) {
  // Row-bias broadcast [M,N] + [N] (either order); otherwise identical shapes.
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const Shape &so = out.shape();
  require_f32(a, "add");
  require_f32(b, "add");
  require_f32(out, "add");
  require_has_storage(a, "add", "input a");
  require_has_storage(b, "add", "input b");
  if (sa == sb) {
    elementwise_binary("add", a, b, out, add_fn);
    return;
  }
  const Tensor *mat = nullptr;
  const Tensor *vec = nullptr;
  if (sa.size() == 2 && sb.size() == 1 && sb[0] == sa[1] && so == sa) {
    mat = &a;
    vec = &b;
  } else if (sb.size() == 2 && sa.size() == 1 && sa[0] == sb[1] && so == sb) {
    mat = &b;
    vec = &a;
  } else {
    KILN_ERROR("add: shape mismatch: ",
        shape_string(sa),
        " vs ",
        shape_string(sb),
        " (only identical shapes or [M, N] + [N] row-bias broadcast are supported)");
  }
  const std::size_t m = mat->shape()[0];
  const std::size_t n = mat->shape()[1];
  if (m == 0 || n == 0) {
    return;
  }
  const float *pm = cread_f32(*mat);
  const float *pv = cread_f32(*vec);
  float *po = static_cast<float *>(out.impl()->storage()->data()) +
              static_cast<std::size_t>(out.impl()->offset());
  const Strides sm = mat->strides();
  const Strides sv = vec->strides();
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      po[i * n + j] = pm[i * sm[0] + j * sm[1]] + pv[j * sv[0]];
    }
  }
}

void mul_f32(const Tensor &a, const Tensor &b, Tensor &out) {
  elementwise_binary("mul", a, b, out, mul_fn);
}

void matmul_f32(const Tensor &a, const Tensor &b, Tensor &out) {
  require_f32(a, "matmul");
  require_f32(b, "matmul");
  require_f32(out, "matmul");
  require_has_storage(a, "matmul", "input a");
  require_has_storage(b, "matmul", "input b");
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const Shape &so = out.shape();
  KILN_CHECK(sa.size() == 2 && sb.size() == 2 && so.size() == 2,
      "matmul: only 2D tensors are supported, got shapes ",
      shape_string(sa),
      " and ",
      shape_string(sb));
  const std::size_t m = sa[0];
  const std::size_t k = sa[1];
  const std::size_t kb = sb[0];
  const std::size_t n = sb[1];
  KILN_CHECK(k == kb,
      "matmul: inner dimensions must match, got shapes ",
      shape_string(sa),
      " and ",
      shape_string(sb));
  KILN_CHECK(so[0] == m && so[1] == n,
      "matmul: output shape ",
      shape_string(so),
      " does not match outer dims [",
      m,
      ", ",
      n,
      "]");
  if (m == 0 || n == 0) {
    return;
  }
  const float *pa = cread_f32(a);
  const float *pb = cread_f32(b);
  float *po = static_cast<float *>(out.impl()->storage()->data()) +
              static_cast<std::size_t>(out.impl()->offset());
  const Strides stra = a.strides();
  const Strides strb = b.strides();
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (std::size_t p = 0; p < k; ++p) {
        acc += pa[i * stra[0] + p * stra[1]] * pb[p * strb[0] + j * strb[1]];
      }
      po[i * n + j] = acc;
    }
  }
}

void relu_f32(const Tensor &x, Tensor &out) {
  elementwise_unary("relu", x, out, relu_fn);
}

void silu_f32(const Tensor &x, Tensor &out) {
  elementwise_unary("silu", x, out, silu_fn);
}

void softmax_f32(const Tensor &x, Tensor &out, std::int64_t axis) {
  require_f32(x, "softmax");
  require_f32(out, "softmax");
  require_has_storage(x, "softmax", "input");
  KILN_CHECK(x.shape() == out.shape(), "softmax: internal shape mismatch");
  const Shape shape = x.shape();
  const std::size_t rank = shape.size();
  KILN_CHECK(rank >= 1, "softmax: expected rank >= 1, got shape ", shape_string(shape));
  std::int64_t ax = axis;
  if (ax < 0) {
    ax += static_cast<std::int64_t>(rank);
  }
  KILN_CHECK(ax >= 0 && ax < static_cast<std::int64_t>(rank),
      "softmax: axis (",
      axis,
      ") out of range for shape ",
      shape_string(shape));
  const std::size_t dax = shape[static_cast<std::size_t>(ax)];
  const std::size_t total = x.numel();
  if (total == 0 || dax == 0) {
    return;
  }
  const float *px = cread_f32(x);
  float *po = static_cast<float *>(out.impl()->storage()->data()) +
              static_cast<std::size_t>(out.impl()->offset());
  const Strides sin = x.strides();
  const Strides sout = Layout::contiguous(shape).strides_vec();
  const std::size_t stride_in_ax = sin[static_cast<std::size_t>(ax)];
  const std::size_t stride_out_ax = sout[static_cast<std::size_t>(ax)];

  // Iterate over all slices orthogonal to `ax` with an odometer that skips ax.
  std::vector<std::size_t> idx(rank, 0);
  std::size_t in_base = 0;
  std::size_t out_base = 0;
  const std::size_t slices = total / dax;
  for (std::size_t s = 0; s < slices; ++s) {
    float m = -std::numeric_limits<float>::infinity();
    for (std::size_t j = 0; j < dax; ++j) {
      const float v = px[in_base + j * stride_in_ax];
      if (v > m) {
        m = v;
      }
    }
    float sum = 0.0f;
    for (std::size_t j = 0; j < dax; ++j) {
      const float e = std::expf(px[in_base + j * stride_in_ax] - m);
      po[out_base + j * stride_out_ax] = e;
      sum += e;
    }
    for (std::size_t j = 0; j < dax; ++j) {
      po[out_base + j * stride_out_ax] /= sum;
    }
    // Advance to next slice (skip the reduction axis).
    for (std::size_t d = rank; d-- > 0;) {
      if (d == static_cast<std::size_t>(ax)) {
        continue;
      }
      in_base += sin[d];
      out_base += sout[d];
      if (++idx[d] < shape[d]) {
        break;
      }
      in_base -= sin[d] * shape[d];
      out_base -= sout[d] * shape[d];
      idx[d] = 0;
    }
  }
}

}  // namespace kiln::kernels
