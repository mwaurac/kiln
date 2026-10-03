from typing import Sequence

class DType:
    F32: int
    F16: int
    BF16: int
    Q8_0: int

class Device:
    CPU: int
    CUDA: int

float32: DType
f16: DType
bf16: DType
q8_0: DType

class Tensor:
    def reshape(self, dims: Sequence[int]) -> "Tensor": ...
    def view(self, dims: Sequence[int]) -> "Tensor": ...
    def transpose(self) -> "Tensor": ...

def empty(
    shape: Sequence[int],
    dtype: DType,
    device: Device,
) -> Tensor: ...
def zeros(
    shape: Sequence[int],
    dtype: DType,
    device: Device,
) -> Tensor: ...
def ones(
    shape: Sequence[int],
    dtype: DType,
    device: Device,
) -> Tensor: ...

