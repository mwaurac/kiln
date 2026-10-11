# Tests

## Layout

`tests/python/` mirrors the package: `nn/` will hold `kiln.nn` tests when that
module exists.

- `test_tensor.py` — the `Tensor` object (factories, metadata, views,
  `contiguous`, `copy_`, bytes). Happy paths only.
- `test_numpy.py` — NumPy interop (`from_numpy` / `numpy`).
- `test_from_buffer.py` — buffer interop (`from_buffer`: sharing, offsets,
  dtypes, mmap lifetime).
- `test_errors.py` — every `RuntimeError` reachable through the Python API,
  grouped by call.
- `test_module.py` — package surface (version, exports, aliases).

`tests/cpp/` proves core logic, including paths Python cannot construct yet
(no lazy factory, no raw graph access):

- `test_storage.cpp` — `Storage` ownership (owned / borrowed / read-only).
- `test_tensor.cpp` — metadata, view chains, `copy_`, `contiguous`, accessors.
- `test_graph.cpp` — nodes, producer lookup through views, scheduling,
  `COPY` execution, deferred materialization.

## Split rule

C++ owns kernel/logic correctness once, with raw fixtures. Python owns
binding glue (arg parsing, defaults), NumPy/buffer-protocol interop, GC
lifetime, and error propagation through the API. Do not re-prove kernel math
in Python with identical scenarios; do not test buffer-protocol or GC
behavior in C++.
