import pytest

import kiln


def test_view():
    t = kiln.zeros([2, 3])
    v = t.view([3, 2])
    assert v.shape == [3, 2]
    assert v.strides == [2, 1]
    assert v.numel() == 6

    # Higher-rank split
    t = kiln.zeros([2, 4])
    v = t.view([2, 2, -1])
    assert v.shape == [2, 2, 2]
    assert v.strides == [4, 2, 1]

    # A single -1 infers
    t = kiln.zeros([2, 3])
    assert t.view([-1]).shape == [6]
    assert t.view([3, -1]).shape == [3, 2]
    assert t.view([2, 3]).shape == [2, 3]

    # numel mismatch
    with pytest.raises(RuntimeError):
        t.view([5])
    with pytest.raises(RuntimeError):
        t.view([4, 2])
    with pytest.raises(RuntimeError):
        t.view([3, -1]) if False else t.view([3, 3])  # 9 != 6, sanity check

    # Invalid dims ie two -1 or < -1
    with pytest.raises(RuntimeError):
        t.view([-1, -1])
    with pytest.raises(RuntimeError):
        t.view([2, -1, -1])
    with pytest.raises(RuntimeError):
        t.view([-2])

    # Non-contiguous input
    tr = t.transpose(0, 1)
    assert tr.shape == [3, 2]
    assert tr.strides == [1, 3]
    with pytest.raises(RuntimeError):
        tr.view([6])


def test_reshape():
    t = kiln.zeros([2, 3])
    r = t.reshape([3, 2])
    assert r.shape == [3, 2]
    assert r.strides == [2, 1]

    assert t.reshape([-1]).shape == [6]
    assert t.reshape([3, -1]).shape == [3, 2]

    # Non-contiguous path
    tr = t.transpose(0, 1)
    r = tr.reshape([6])
    assert r.shape == [6]
    assert r.strides == [1]

    with pytest.raises(RuntimeError):
        t.reshape([5])
    with pytest.raises(RuntimeError):
        t.reshape([-1, -1])


def test_transpose():
    # 2D swap: transpose(0, 1)
    t = kiln.zeros([2, 3])
    tr = t.transpose(0, 1)
    assert tr.shape == [3, 2]
    assert tr.strides == [1, 3]
    assert tr.numel() == 6

    # Partial 3D swap: only the two named axes move
    t = kiln.zeros([2, 3, 4])
    tr = t.transpose(0, 2)
    assert tr.shape == [4, 3, 2]
    assert tr.strides == [1, 4, 12]

    # Swapping twice restores original
    back = tr.transpose(0, 2)
    assert back.shape == [2, 3, 4]
    assert back.strides == [12, 4, 1]

    # Negative dims count from the back (-1 is the last axis).
    t = kiln.zeros([2, 3])
    tr = t.transpose(-2, -1)
    assert tr.shape == [3, 2]
    assert tr.strides == [1, 3]

    # Same dim is a no-op
    t = kiln.zeros([2, 3])
    tr = t.transpose(1, 1)
    assert tr.shape == [2, 3]
    assert tr.strides == [3, 1]

    # OOR dims are rejected
    with pytest.raises(RuntimeError):
        t.transpose(0, 2)
    with pytest.raises(RuntimeError):
        t.transpose(0, -3)

    t = kiln.zeros([2, 3])
    assert t.T.shape == [3, 2]
    assert t.T.strides == [1, 3]

    assert t.mT.shape == [3, 2]
    assert t.mT.strides == [1, 3]
    t3 = kiln.zeros([2, 3, 4])
    assert t3.mT.shape == [2, 4, 3]
    assert t3.mT.strides == [12, 1, 4]
    # .mT needs at least 2 dims
    with pytest.raises(RuntimeError):
        kiln.zeros([3]).mT


def test_contiguous():
    t = kiln.zeros([2, 3])
    c = t.contiguous()
    assert c.shape == [2, 3]
    assert c.strides == [3, 1]

    tr = t.transpose(0, 1).contiguous()
    assert tr.shape == [3, 2]
    assert tr.strides == [2, 1]
