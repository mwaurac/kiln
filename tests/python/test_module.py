import kiln


def test_version():
    assert kiln.__version__ == "0.1.0"


def test_top_level_exports():
    for name in [
        "Tensor",
        "DType",
        "Device",
        "execute",
        "empty",
        "zeros",
        "ones",
        "nbytes",
        "itemsize",
        "from_numpy",
        "from_buffer",
        "float32",
        "f16",
        "bf16",
        "q8_0",
    ]:
        assert hasattr(kiln, name), name
    assert set(kiln.__all__) >= {
        "Tensor",
        "execute",
        "from_numpy",
        "from_buffer",
    }


def test_dtype_device_aliases():
    assert kiln.float32 == kiln.DType.F32
    assert kiln.f16 == kiln.DType.F16
    assert kiln.bf16 == kiln.DType.BF16
    assert kiln.q8_0 == kiln.DType.Q8_0
    assert kiln.Device.CPU != kiln.Device.CUDA
