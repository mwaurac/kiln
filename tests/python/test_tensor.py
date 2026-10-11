import struct

import numpy as np

import kiln


# --- factories -----------------------------------------------------------


def test_zeros_factory():
    t = kiln.zeros([2, 3])
    assert t.shape == [2, 3]
    assert t.strides == [3, 1]
    assert t.numel() == 6
    assert t.dtype == kiln.float32
    assert t.device == kiln.Device.CPU
    assert t.has_storage
    assert not t.is_borrowed
    assert not t.is_readonly
    np.testing.assert_array_equal(t.numpy(), np.zeros((2, 3), dtype=np.float32))


def test_ones_values_f32_f16_bf16():
    np.testing.assert_array_equal(kiln.ones([3]).numpy(), np.ones(3, dtype=np.float32))
    assert kiln.ones([2], dtype=kiln.f16).data() == struct.pack("2H", 0x3C00, 0x3C00)
    assert kiln.ones([2], dtype=kiln.bf16).data() == struct.pack("2H", 0x3F80, 0x3F80)


def test_constructor_defaults():
    t = kiln.Tensor([2, 3])
    assert t.dtype == kiln.float32
    assert t.device == kiln.Device.CPU
    assert kiln.Tensor([2], dtype=kiln.f16).dtype == kiln.f16
    assert kiln.empty([4], dtype=kiln.bf16).dtype == kiln.bf16


def test_scalar_tensor():
    t = kiln.zeros([], dtype=kiln.float32)
    assert t.shape == []
    assert t.strides == []
    assert t.numel() == 1
    assert t.nbytes == 4
    assert t.itemsize == 4
    assert t.is_contiguous()
    assert t.has_storage
    assert t.data() == b"\x00" * 4
    assert struct.unpack("f", kiln.ones([]).data()) == (1.0,)
    assert kiln.ones([], dtype=kiln.f16).data() == struct.pack("H", 0x3C00)
    assert kiln.ones([], dtype=kiln.bf16).data() == struct.pack("H", 0x3F80)


def test_empty_tensors():
    t = kiln.empty([0])
    assert t.shape == [0]
    assert t.numel() == 0
    assert t.nbytes == 0
    assert t.data() == b""
    assert t.has_storage
    assert t.is_contiguous()

    z = kiln.zeros([2, 0, 3])
    assert z.shape == [2, 0, 3]
    assert z.numpy().shape == (2, 0, 3)

    q = kiln.empty([0], dtype=kiln.q8_0)  # zero blocks is well-defined
    assert q.numel() == 0
    assert q.nbytes == 0


def test_repr():
    assert repr(kiln.zeros([2, 3])) == "kiln.Tensor(shape=[2, 3], dtype=f32, device=cpu)"
    assert repr(kiln.zeros([], dtype=kiln.f16)) == "kiln.Tensor(shape=[], dtype=f16, device=cpu)"


# --- views ---------------------------------------------------------------


def test_view_metadata():
    t = kiln.zeros([2, 3])
    v = t.view([3, 2])
    assert v.shape == [3, 2]
    assert v.strides == [2, 1]
    assert v.numel() == 6
    assert v.is_borrowed == t.is_borrowed

    assert kiln.zeros([2, 4]).view([2, 2, -1]).shape == [2, 2, 2]
    assert t.view([-1]).shape == [6]
    assert t.view([3, -1]).shape == [3, 2]


def test_view_shares_storage():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = kiln.from_numpy(a, copy=False)
    v = t.view([6])
    r = t.reshape([3, 2])
    assert v.is_borrowed and r.is_borrowed
    a[0, 0] = -5.0
    assert v.numpy()[0] == -5.0
    assert r.numpy()[0, 0] == -5.0


def test_reshape_metadata():
    t = kiln.zeros([2, 3])
    r = t.reshape([3, 2])
    assert r.shape == [3, 2]
    assert r.strides == [2, 1]
    assert t.reshape([-1]).shape == [6]

    # Non-contiguous reshape packs a copy first.
    tr = t.transpose(0, 1)
    r = tr.reshape([6])
    assert r.shape == [6]
    assert r.strides == [1]
    assert r.is_contiguous()


def test_reshape_of_transposed_values():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    flat = kiln.from_numpy(a).transpose(0, 1).reshape([6])
    np.testing.assert_array_equal(flat.numpy(), a.T.reshape(6))


def test_transpose_metadata():
    t = kiln.zeros([2, 3])
    tr = t.transpose(0, 1)
    assert tr.shape == [3, 2]
    assert tr.strides == [1, 3]
    assert tr.numel() == 6
    assert not tr.is_contiguous()

    t3 = kiln.zeros([2, 3, 4])
    tr = t3.transpose(0, 2)
    assert tr.shape == [4, 3, 2]
    assert tr.strides == [1, 4, 12]
    back = tr.transpose(0, 2)
    assert back.shape == [2, 3, 4]
    assert back.strides == [12, 4, 1]

    # Negative dims count from the back; same dim is a no-op.
    assert t.transpose(-2, -1).shape == [3, 2]
    assert t.transpose(1, 1).strides == [3, 1]

    assert t.T.shape == [3, 2]
    assert t.T.strides == [1, 3]
    assert t.mT.shape == [3, 2]
    assert kiln.zeros([2, 3, 4]).mT.shape == [2, 4, 3]


def test_scalar_views():
    t = kiln.zeros([])
    assert t.view([]).shape == []
    assert t.reshape([]).shape == []
    assert t.view([1]).shape == [1]
    assert t.T.shape == []


def test_empty_views():
    t = kiln.zeros([0])
    assert t.view([0]).shape == [0]
    assert t.view([-1]).shape == [0]
    assert t.view([2, 0]).shape == [2, 0]
    assert t.reshape([0, 1]).shape == [0, 1]


# --- contiguous / copy_ --------------------------------------------------


def test_contiguous_of_contiguous_is_alias():
    a = np.ones((2, 2), dtype=np.float32)
    t = kiln.from_numpy(a, copy=False)
    c = t.contiguous()
    assert c.is_borrowed  # same storage, no copy
    a[0, 0] = 42.0
    assert c.numpy()[0, 0] == 42.0


def test_contiguous_of_strided_is_owned_snapshot():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = kiln.from_numpy(a, copy=False)
    snap = t.transpose(0, 1).contiguous()
    assert snap.is_contiguous()
    assert not snap.is_borrowed
    np.testing.assert_array_equal(snap.numpy(), a.T)
    a[1, 2] = -1.0
    assert snap.numpy()[2, 1] != -1.0


def test_copy_basic():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    dst = kiln.zeros([2, 3])
    out = dst.copy_(kiln.from_numpy(a))
    np.testing.assert_array_equal(dst.numpy(), a)
    np.testing.assert_array_equal(out.numpy(), a)  # copy_ returns self


def test_copy_strided_source():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    src = kiln.from_numpy(a, copy=False).transpose(0, 1)
    dst = kiln.zeros([3, 2])
    dst.copy_(src)
    np.testing.assert_array_equal(dst.numpy(), a.T)


def test_copy_writes_through_strided_destination():
    base = kiln.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3), copy=False)
    view = base.transpose(0, 1)
    patch = kiln.from_numpy(np.full((3, 2), 7.0, dtype=np.float32))
    view.copy_(patch)
    np.testing.assert_array_equal(base.numpy(), np.full((2, 3), 7.0, dtype=np.float32))


def test_copy_self_and_empty_are_noops():
    t = kiln.from_numpy(np.arange(4, dtype=np.float32))
    t.copy_(t)
    np.testing.assert_array_equal(t.numpy(), np.arange(4, dtype=np.float32))

    dst = kiln.from_numpy(np.empty((0,), dtype=np.float32))
    dst.copy_(kiln.from_numpy(np.empty((0,), dtype=np.float32)))
    assert dst.numel() == 0


def test_copy_large_strided_roundtrip():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((16, 24)).astype(np.float32)
    src = kiln.from_numpy(a, copy=False).transpose(0, 1)
    dst = kiln.empty([24, 16])
    dst.copy_(src)
    np.testing.assert_array_equal(dst.numpy(), a.T)


# --- bytes / execute -----------------------------------------------------


def test_data_requires_contiguous_materialized():
    t = kiln.from_numpy(np.arange(4, dtype=np.float32))
    assert t.data() == np.arange(4, dtype=np.float32).tobytes()


def test_execute_is_noop_for_materialized():
    t = kiln.ones([2, 2])
    kiln.execute(t)
    np.testing.assert_array_equal(t.numpy(), np.ones((2, 2), dtype=np.float32))
