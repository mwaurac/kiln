import numpy as np

import kiln


def test_add_mul_dunders_build_lazy():
    a = kiln.zeros([2, 3])
    b = kiln.zeros([2, 3])
    for y in (a + b, kiln.add(a, b), a * b, kiln.mul(a, b)):
        assert y.shape == [2, 3]
        assert not y.has_storage


def test_matmul_infers_outer_dims():
    x = kiln.from_numpy(np.random.randn(4, 8).astype(np.float32))
    w = kiln.from_numpy(np.random.randn(16, 8).astype(np.float32))
    y = x @ w.mT
    assert y.shape == [4, 16]
    assert not y.has_storage
    assert kiln.matmul(x, w.transpose(0, 1)).shape == [4, 16]


def test_unary_ops_pass_shape_through():
    t = kiln.zeros([2, 3])
    for y in (kiln.relu(t), kiln.silu(t), kiln.softmax(t), kiln.softmax(t, axis=0)):
        assert y.shape == [2, 3]
        assert not y.has_storage
    s = kiln.silu(kiln.zeros([]))
    assert s.shape == []


def test_capture_records_into_given_graph():
    g = kiln.Graph()
    assert len(g) == 0
    with g:
        y = kiln.relu(kiln.zeros([2]))
    assert len(g) == 1
    assert not y.has_storage


def test_captures_are_independent():
    g1, g2 = kiln.Graph(), kiln.Graph()
    with g1:
        kiln.relu(kiln.zeros([2]))
        kiln.relu(kiln.zeros([2]))
    with g2:
        kiln.silu(kiln.zeros([2]))
    assert len(g1) == 2
    assert len(g2) == 1


def test_building_continues_input_graph_outside_capture():
    g = kiln.Graph()
    with g:
        y = kiln.relu(kiln.zeros([3]))
    assert len(g) == 1
    z = y + kiln.zeros([3])  # no capture: inherits y's graph
    assert len(g) == 2
    assert not z.has_storage


def test_default_path_needs_no_capture():
    y = kiln.relu(kiln.zeros([2, 2]))
    assert y.shape == [2, 2]
    assert not y.has_storage
