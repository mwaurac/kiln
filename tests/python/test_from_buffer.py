import gc
import mmap
import struct

import numpy as np

import kiln


def test_from_buffer_shares_memory():
    raw = bytearray(8)
    t = kiln.from_buffer(raw, [2])
    assert t.shape == [2]
    assert t.dtype == kiln.float32
    assert t.is_borrowed
    assert not t.is_readonly
    struct.pack_into("2f", raw, 0, 1.5, 2.5)
    np.testing.assert_array_equal(t.numpy(), [1.5, 2.5])


def test_from_buffer_keeps_owner_alive():
    t = kiln.from_buffer(bytearray(b"\x00\x00\x80?\x00\x00\x00\x40"), [2])
    gc.collect()
    np.testing.assert_array_equal(t.numpy(), [1.0, 2.0])


def test_from_buffer_explicit_strides():
    raw = bytearray(16)
    struct.pack_into("4f", raw, 0, 1.0, 2.0, 3.0, 4.0)
    t = kiln.from_buffer(raw, [2, 2], strides=[2, 1])
    assert t.strides == [2, 1]
    np.testing.assert_array_equal(t.numpy(), [[1.0, 2.0], [3.0, 4.0]])


def test_from_buffer_byte_offset():
    raw = bytearray(struct.pack("4f", 1.0, 2.0, 3.0, 4.0))
    t = kiln.from_buffer(raw, [2], byte_offset=8)
    assert t.is_borrowed
    assert t.nbytes == 8
    np.testing.assert_array_equal(t.numpy(), [3.0, 4.0])


def test_from_buffer_read_only_bytes():
    t = kiln.from_buffer(b"\x00\x00\x80?\x00\x00\x00\x40", [2])
    assert t.is_borrowed
    assert t.is_readonly
    np.testing.assert_array_equal(t.numpy(), [1.0, 2.0])
    # Read-only borrowed bytes are still readable as a copy.
    assert t.data() == struct.pack("2f", 1.0, 2.0)


def test_from_buffer_f16_bytes():
    raw = bytearray(struct.pack("4H", 0x3C00, 0x4000, 0x4200, 0x0000))  # 1,2,3,0
    t = kiln.from_buffer(raw, [4], dtype=kiln.f16)
    assert t.dtype == kiln.f16
    assert t.nbytes == 8
    assert t.itemsize == 2
    assert t.is_borrowed
    assert t.data() == bytes(raw)


def test_from_buffer_bf16_bytes_and_strides():
    raw = bytearray(struct.pack("4H", 0x3F80, 0x4000, 0x4040, 0x4080))  # 1,2,3,4
    t = kiln.from_buffer(raw, [2, 2], dtype=kiln.bf16, strides=[2, 1])
    assert t.dtype == kiln.bf16
    assert t.shape == [2, 2]
    assert t.strides == [2, 1]
    assert t.nbytes == 8
    assert t.data() == bytes(raw)


def test_from_buffer_f16_with_offset_and_strides():
    pad = struct.pack("2H", 0x0000, 0x0000)
    body = struct.pack("4H", 0x3C00, 0x4000, 0x4200, 0x4400)
    raw = bytearray(pad + body)
    t = kiln.from_buffer(raw, [2, 2], dtype=kiln.f16, strides=[2, 1], byte_offset=4)
    assert t.data() == body
    assert t.is_borrowed


def test_from_buffer_q8_0_bytes():
    raw = bytearray(34)  # one block: fp16 scale + 32 int8
    t = kiln.from_buffer(raw, [32], dtype=kiln.q8_0)
    assert t.nbytes == 34
    assert t.data() == bytes(raw)


def test_borrowed_mmap_stays_alive():
    m = mmap.mmap(-1, 16)
    m.write(struct.pack("4f", 1.0, 2.0, 3.0, 4.0))
    t = kiln.from_buffer(m, [4])
    assert t.is_borrowed
    del m
    gc.collect()
    # The tensor retained the mapping.
    np.testing.assert_array_equal(t.numpy(), [1.0, 2.0, 3.0, 4.0])


def test_readonly_mmap_is_readonly_tensor():
    with mmap.mmap(-1, 8, access=mmap.ACCESS_READ) as m:
        t = kiln.from_buffer(m, [2])
        assert t.is_borrowed
        assert t.is_readonly
        np.testing.assert_array_equal(t.numpy(), [0.0, 0.0])
    gc.collect()


def test_from_buffer_offset_into_mmap():
    m = mmap.mmap(-1, 16)
    m.write(struct.pack("4f", 1.0, 2.0, 3.0, 4.0))
    t = kiln.from_buffer(m, [2], byte_offset=8)
    del m
    gc.collect()
    np.testing.assert_array_equal(t.numpy(), [3.0, 4.0])


def test_is_borrowed_diagnostic():
    assert not kiln.ones([2]).is_borrowed
    assert not kiln.from_numpy(np.ones(2, dtype=np.float32)).is_borrowed
    assert kiln.from_numpy(np.ones(2, dtype=np.float32), copy=False).is_borrowed
    assert kiln.from_buffer(bytearray(8), [2]).is_borrowed
