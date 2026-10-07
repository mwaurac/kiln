# Working with tensors


## Creating tensors

```python
import kiln

t = kiln.Tensor([2, 3])
t = kiln.Tensor([2, 3], dtype=kiln.f16)
e = kiln.empty([2, 3])
z = kiln.zeros([2, 3])
o = kiln.ones([2, 3])
s = kiln.zeros([])
```

`empty` leaves memory uninitialized. `zeros` works for every dtype.
`ones` supports `float32`, `f16`, and `bf16`.

## Inspecting tensors

```python
t = kiln.zeros([2, 3])
t.shape
t.strides
t.dtype
t.device
t.numel()
t.nbytes
t.itemsize
t.has_storage
t.is_contiguous()
```

## Reshaping

`view`, `transpose`, `.T`, and `.mT` return new views over the same
memory.

```python
t = kiln.zeros([2, 3])
t.view([3, 2])
t.view([-1])
t.transpose(0, 1)
t.T
```

```python
kiln.zeros([2, 3, 4]).mT
```

`view` needs a contiguous tensor. `reshape` behaves like `view`,
except on a non-contiguous tensor it returns a packed contiguous copy
first.

```python
t = kiln.zeros([2, 3])
c = t.transpose(0, 1).contiguous()
```

## Reading raw bytes

```python
kiln.zeros([2, 2]).data()
kiln.empty([0]).data()
```

## Running execute

```python
kiln.execute(t)
```

For tensors created with the APIs above, `execute` is a no-op.
