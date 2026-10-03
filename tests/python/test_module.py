import pytest

import kiln
from kiln import DType
from kiln import Device


def test_version():
    assert kiln.__version__ == "0.1.0"


def test_zero_factory():
    t = kiln.zeros([2, 3])
    print(t)
    assert t.shape == [2, 3]
    assert t.numel() == 6
    assert t.dtype == DType.F32
    assert t.device == Device.CPU
    assert t.strides == [3, 1]
