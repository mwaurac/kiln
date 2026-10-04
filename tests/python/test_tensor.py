import pytest

import kiln


def test_transpose():
    t = kiln.zeros([3, 4])
    t1 = t.transpose()
    assert t1.shape == [4, 3]
    t1.strides == [3, 1]


def test_view():
    t = kiln.zeros([2, 4])
    t_v = t.view([2, 2, -1])
    assert t_v.strides == [4, 2, 1]
    assert t_v.shape == [2, 2, 2]

    with pytest.raises(Exception):
        t.view([3, -1])
    with pytest.raises(Exception):
        t.view([2, -1, -1])
