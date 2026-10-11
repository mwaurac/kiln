import numpy as np
import pytest

import kiln


# --- factories -----------------------------------------------------------


def test_q8_0_scalar_shape_rejected():
    # 1 element is not a whole number of 32-element blocks.
    with pytest.raises(RuntimeError):
        kiln.empty([], dtype=kiln.q8_0)


def test_ones_rejects_quantized_dtype():
    with pytest.raises(RuntimeError, match="unsupported dtype"):
        kiln.ones([32], dtype=kiln.q8_0)


def test_rank_above_max_dims_rejected():
    with pytest.raises(RuntimeError, match="exceeds max dims"):
        kiln.zeros([1] * 7)


# --- views ---------------------------------------------------------------


def test_view_rejects_numel_mismatch():
    t = kiln.zeros([2, 3])
    with pytest.raises(RuntimeError):
        t.view([5])
    with pytest.raises(RuntimeError):
        t.view([4, 2])
    with pytest.raises(RuntimeError):
        t.view([3, 3])


def test_view_rejects_bad_infer_dims():
    t = kiln.zeros([2, 3])
    with pytest.raises(RuntimeError):
        t.view([-1, -1])
    with pytest.raises(RuntimeError):
        t.view([2, -1, -1])
    with pytest.raises(RuntimeError):
        t.view([-2])


def test_view_rejects_non_contiguous():
    tr = kiln.zeros([2, 3]).transpose(0, 1)
    with pytest.raises(RuntimeError):
        tr.view([6])


def test_view_empty_rejects_wrong_numel():
    with pytest.raises(RuntimeError):
        kiln.zeros([0]).view([3])


def test_reshape_rejects_bad_shapes():
    t = kiln.zeros([2, 3])
    with pytest.raises(RuntimeError):
        t.reshape([5])
    with pytest.raises(RuntimeError):
        t.reshape([-1, -1])


def test_transpose_rejects_out_of_range_dims():
    t = kiln.zeros([2, 3])
    with pytest.raises(RuntimeError):
        t.transpose(0, 2)
    with pytest.raises(RuntimeError):
        t.transpose(0, -3)
    with pytest.raises(RuntimeError):
        kiln.zeros([]).transpose(0, 1)


def test_mt_needs_at_least_2_dims():
    with pytest.raises(RuntimeError):
        kiln.zeros([3]).mT
    with pytest.raises(RuntimeError):
        kiln.zeros([]).mT


# --- data / numpy --------------------------------------------------------


def test_data_rejects_non_contiguous():
    tr = kiln.zeros([2, 3]).transpose(0, 1)
    with pytest.raises(RuntimeError, match="non-contiguous"):
        tr.data()


def test_numpy_rejects_non_f32():
    with pytest.raises(RuntimeError, match="only float32"):
        kiln.zeros([2, 2], dtype=kiln.f16).numpy()
    with pytest.raises(RuntimeError, match="only float32"):
        kiln.zeros([2, 2], dtype=kiln.bf16).numpy()
    with pytest.raises(RuntimeError, match="only float32"):
        kiln.from_buffer(bytearray(8), [4], dtype=kiln.f16).numpy()


# --- from_numpy ----------------------------------------------------------


def test_from_numpy_rejects_non_f32():
    with pytest.raises(RuntimeError, match="only native float32"):
        kiln.from_numpy(np.ones((2, 2), dtype=np.float64))
    with pytest.raises(RuntimeError, match="only native float32"):
        kiln.from_numpy(np.ones((2, 2), dtype=np.int32))
    with pytest.raises(RuntimeError, match="only native float32"):
        kiln.from_numpy(np.ones((2, 2), dtype=np.float16))


def test_from_numpy_copy_false_rejects_negative_strides():
    b = np.arange(6, dtype=np.float32).reshape(2, 3)
    with pytest.raises(RuntimeError, match="negative strides"):
        kiln.from_numpy(b[:, ::-1], copy=False)


def test_from_numpy_copy_false_rejects_read_only():
    a = np.ones((2, 2), dtype=np.float32)
    a.flags.writeable = False
    with pytest.raises(RuntimeError, match="writeable"):
        kiln.from_numpy(a, copy=False)


# --- from_buffer ---------------------------------------------------------


def test_from_buffer_rejects_too_small():
    with pytest.raises(RuntimeError, match="exceeds"):
        kiln.from_buffer(bytearray(4), [2])


def test_from_buffer_rejects_rank_mismatch():
    with pytest.raises(RuntimeError, match="rank"):
        kiln.from_buffer(bytearray(8), [2], strides=[1, 1])


def test_from_buffer_rejects_bad_byte_offset():
    raw = bytearray(16)
    with pytest.raises(RuntimeError, match="non-negative"):
        kiln.from_buffer(raw, [2], byte_offset=-4)
    with pytest.raises(RuntimeError, match="multiple"):
        kiln.from_buffer(raw, [2], byte_offset=2)
    with pytest.raises(RuntimeError, match="multiple"):
        kiln.from_buffer(raw, [2], dtype=kiln.f16, byte_offset=1)
    with pytest.raises(RuntimeError, match="exceeds"):
        kiln.from_buffer(raw, [2], byte_offset=12)
    with pytest.raises(RuntimeError, match="exceeds"):
        kiln.from_buffer(raw, [1], byte_offset=16)


def test_from_buffer_rejects_bad_q8_0():
    with pytest.raises(RuntimeError, match="multiple of block size"):
        kiln.from_buffer(bytearray(34), [3], dtype=kiln.q8_0)
    with pytest.raises(RuntimeError, match="block-quantized"):
        kiln.from_buffer(bytearray(68), [64], dtype=kiln.q8_0, strides=[2])
    with pytest.raises(RuntimeError, match="multiple"):
        kiln.from_buffer(bytearray(68), [32], dtype=kiln.q8_0, byte_offset=2)


# --- copy_ ---------------------------------------------------------------


def test_copy_rejects_shape_mismatch():
    dst = kiln.zeros([2, 3])
    with pytest.raises(RuntimeError, match=r"shape mismatch.*\[2, 3\].*\[3, 2\]"):
        dst.copy_(kiln.zeros([3, 2]))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        dst.copy_(kiln.zeros([6]))


def test_copy_rejects_dtype_mismatch():
    with pytest.raises(RuntimeError, match=r"dtype mismatch.*f32.*f16"):
        kiln.zeros([2, 2]).copy_(kiln.zeros([2, 2], dtype=kiln.f16))


def test_copy_rejects_readonly_destination():
    dst = kiln.from_buffer(b"\x00" * 8, [2])
    with pytest.raises(RuntimeError, match="read-only"):
        dst.copy_(kiln.ones([2]))
