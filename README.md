# kiln

Kiln's goal is a Python-authored, C++-executed inference runtime: write
familiar tensor code in Python, compile once, and run native CPU kernels
with stable memory.

## Install

```bash
pip install kiln-rt
```

## Use

```python
import numpy as np
import kiln

a = np.arange(6, dtype=np.float32).reshape(2, 3)
t = kiln.from_numpy(a)
print(t.shape, t.strides, t.nbytes)
```

```python
c = t.transpose(0, 1).contiguous()
np.testing.assert_array_equal(c.numpy(), a.T)
```

```python
z = kiln.zeros([2, 3])
o = kiln.ones([2, 3])
```

