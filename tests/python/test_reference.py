"""M2 reference kernels: F32 execution parity, broadcast, and graph behavior."""

import numpy as np
import pytest

import kiln


def _rng():
    return np.random.default_rng(0)


def test_m2_acceptance_silu_mlp_with_row_bias():
    rng = _rng()
    x = kiln.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
    w = kiln.from_numpy(rng.standard_normal((16, 8)).astype(np.float32))
    b = kiln.from_numpy(rng.standard_normal((16,)).astype(np.float32))
    y = kiln.silu(x @ w.mT + b)
    assert not y.has_storage
    kiln.execute(y)
    z = x.numpy() @ w.numpy().T + b.numpy()
    np.testing.assert_allclose(y.numpy(), z / (1.0 + np.exp(-z)), rtol=1e-5, atol=1e-6)


def test_add_row_bias_broadcast_both_orders():
    rng = _rng()
    mat = rng.standard_normal((3, 5)).astype(np.float32)
    vec = rng.standard_normal((5,)).astype(np.float32)
    for y, ref in (
        (kiln.from_numpy(mat) + kiln.from_numpy(vec), mat + vec),
        (kiln.from_numpy(vec) + kiln.from_numpy(mat), mat + vec),
    ):
        assert y.shape == [3, 5]
        kiln.execute(y)
        np.testing.assert_allclose(y.numpy(), ref, rtol=1e-6, atol=1e-7)


def test_add_rejects_non_broadcastable_shapes():
    with pytest.raises(RuntimeError, match=r"only identical shapes or \[M, N\]"):
        kiln.zeros([4, 16]) + kiln.zeros([8])
    with pytest.raises(RuntimeError, match=r"only identical shapes or \[M, N\]"):
        kiln.zeros([4, 16]) + kiln.zeros([4])


def test_matmul_supports_transposed_operand():
    rng = _rng()
    a = rng.standard_normal((8, 4)).astype(np.float32)
    b = rng.standard_normal((8, 16)).astype(np.float32)
    ka, kb = kiln.from_numpy(a), kiln.from_numpy(b)
    y = kiln.matmul(ka.mT, kb)  # [4, 8] @ [8, 16], lhs is a strided view
    assert y.shape == [4, 16]
    kiln.execute(y)
    np.testing.assert_allclose(y.numpy(), a.T @ b, rtol=1e-5, atol=1e-6)


def test_matmul_inner_mismatch_names_shapes():
    with pytest.raises(RuntimeError, match=r"inner dimensions.*\[4, 8\].*\[7, 16\]"):
        kiln.zeros([4, 8]) @ kiln.zeros([7, 16])


def test_elementwise_parity_with_strided_inputs():
    rng = _rng()
    p = rng.standard_normal((2, 3)).astype(np.float32)
    q = rng.standard_normal((2, 3)).astype(np.float32)
    kp, kq = kiln.from_numpy(p), kiln.from_numpy(q)
    cases = [
        (kp.transpose(0, 1) + kq.transpose(0, 1), p.T + q.T),
        (kp.transpose(0, 1) * kq.transpose(0, 1), p.T * q.T),
        (kiln.relu(kp), np.maximum(p, 0)),
        (kiln.silu(kp), p / (1.0 + np.exp(-p))),
    ]
    for y, ref in cases:
        kiln.execute(y)
        np.testing.assert_allclose(y.numpy(), ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_softmax_parity_per_axis(axis):
    rng = _rng()
    v = rng.standard_normal((2, 3)).astype(np.float32)
    s = kiln.softmax(kiln.from_numpy(v), axis=axis)
    kiln.execute(s)
    e = np.exp(v - v.max(axis=axis, keepdims=True))
    np.testing.assert_allclose(s.numpy(), e / e.sum(axis=axis, keepdims=True), rtol=1e-5, atol=1e-6)


def test_topological_order_shared_subgraph_and_rerun():
    x = kiln.from_numpy(np.ones((2, 2), dtype=np.float32))
    w = kiln.from_numpy(np.eye(2, dtype=np.float32))
    h = kiln.relu(x)  # shared producer
    y1, y2 = h + w, h * w
    kiln.execute(y1)
    kiln.execute(y2)  # shared subgraph executes for each requested output
    np.testing.assert_array_equal(y1.numpy(), np.full((2, 2), 1.0) + np.eye(2))
    np.testing.assert_array_equal(y2.numpy(), np.eye(2))
    # Changing inputs across runs changes outputs without retracing.
    x.copy_(kiln.from_numpy(np.full((2, 2), 2.0, dtype=np.float32)))
    kiln.execute(y1)
    np.testing.assert_array_equal(y1.numpy(), np.full((2, 2), 2.0) + np.eye(2))


def test_unexecuted_output_raises():
    y = kiln.relu(kiln.from_numpy(np.ones((2,), dtype=np.float32)))
    assert not y.has_storage
    with pytest.raises(RuntimeError, match="call execute"):
        y.numpy()


def test_non_f32_execution_rejected():
    with pytest.raises(RuntimeError, match="only support f32"):
        kiln.execute(kiln.relu(kiln.zeros([2], dtype=kiln.f16)))
