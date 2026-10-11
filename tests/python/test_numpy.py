import gc

import numpy as np

import kiln


def test_round_trip_contiguous():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = kiln.from_numpy(a)
    assert t.shape == [2, 3]
    assert t.strides == [3, 1]
    assert t.dtype == kiln.float32
    assert not t.is_borrowed  # default copy=True owns its storage
    np.testing.assert_array_equal(t.numpy(), a)


def test_from_numpy_default_is_owned_copy():
    a = np.ones((2, 2), dtype=np.float32)
    t = kiln.from_numpy(a)
    a[0, 0] = 42.0
    # Tensor keeps the original values; it does not track the array.
    np.testing.assert_array_equal(t.numpy(), np.ones((2, 2), dtype=np.float32))


def test_from_numpy_copy_false_shares_memory_both_ways():
    a = np.ones((2, 2), dtype=np.float32)
    t = kiln.from_numpy(a, copy=False)
    assert t.is_borrowed
    # numpy -> tensor
    a[0, 0] = 42.0
    np.testing.assert_array_equal(t.numpy(), a)
    # a second tensor over the same array observes the same memory
    t2 = kiln.from_numpy(a, copy=False)
    a[1, 1] = 7.0
    np.testing.assert_array_equal(t2.numpy(), a)
    np.testing.assert_array_equal(t.numpy(), a)


def test_from_numpy_copy_false_keeps_array_alive():
    t = kiln.from_numpy(np.ones(4, dtype=np.float32), copy=False)
    gc.collect()
    np.testing.assert_array_equal(t.numpy(), np.ones(4, dtype=np.float32))


def test_numpy_returns_owned_copy():
    t = kiln.ones([2, 2])
    a = t.numpy()
    assert isinstance(a, np.ndarray)
    assert a.dtype == np.float32
    assert a.flags["C_CONTIGUOUS"]
    a *= 0
    # Tensor still all ones.
    assert t.data() == b"\x00\x00\x80?" * 4


def test_from_numpy_copy_false_transposed_view_shares_with_strides():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert not a.T.flags["C_CONTIGUOUS"]
    t = kiln.from_numpy(a.T, copy=False)
    assert t.shape == [3, 2]
    assert t.strides == [1, 3]
    np.testing.assert_array_equal(t.numpy(), a.T)
    a[1, 2] = -1.0
    assert t.numpy()[2, 1] == -1.0


def test_from_numpy_default_copies_non_contiguous():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = kiln.from_numpy(a.T)  # copy=True packs the view
    assert t.shape == [3, 2]
    assert t.is_contiguous()
    np.testing.assert_array_equal(t.numpy(), a.T)
    a[1, 2] = -1.0
    assert t.numpy()[2, 1] != -1.0


def test_from_numpy_strided_and_fortran():
    a = np.arange(12, dtype=np.float32)
    t = kiln.from_numpy(a[::2], copy=False)
    assert t.shape == [6]
    np.testing.assert_array_equal(t.numpy(), a[::2])

    b = np.arange(6, dtype=np.float32).reshape(2, 3)
    f = np.asfortranarray(b)
    assert not f.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(kiln.from_numpy(f, copy=False).numpy(), f)
    np.testing.assert_array_equal(kiln.from_numpy(f).numpy(), f)


def test_from_numpy_reversed_slice_copies_by_default():
    b = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(kiln.from_numpy(b[:, ::-1]).numpy(), b[:, ::-1])
    np.testing.assert_array_equal(
        kiln.from_numpy(np.ascontiguousarray(b[:, ::-1])).numpy(), b[:, ::-1]
    )


def test_from_numpy_read_only_copies_by_default():
    a = np.ones((2, 2), dtype=np.float32)
    a.flags.writeable = False
    np.testing.assert_array_equal(kiln.from_numpy(a).numpy(), a)


def test_numpy_of_transposed_tensor_packs_copy():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = kiln.from_numpy(a)
    tr = t.transpose(0, 1)
    assert not tr.is_contiguous()
    np.testing.assert_array_equal(tr.contiguous().numpy(), a.T)
    np.testing.assert_array_equal(tr.numpy(), a.T)


def test_scalar_and_empty():
    s = kiln.from_numpy(np.asarray(np.float32(1.5)))
    assert s.shape == []
    assert s.numel() == 1
    assert s.numpy().shape == ()
    assert float(s.numpy()) == 1.5

    e = kiln.from_numpy(np.empty((0,), dtype=np.float32))
    assert e.numel() == 0
    assert e.numpy().shape == (0,)
    z = kiln.from_numpy(np.empty((2, 0, 3), dtype=np.float32))
    assert z.shape == [2, 0, 3]
    assert z.numpy().shape == (2, 0, 3)
