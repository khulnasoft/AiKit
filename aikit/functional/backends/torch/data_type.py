# global
from typing import Optional, Union, Sequence, List
import numpy as np
import torch

# local
import aikit
from aikit.func_wrapper import with_unsupported_dtypes
from aikit.functional.aikit.data_type import _handle_nestable_dtype_info
from . import backend_version

aikit_dtype_dict = {
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
    torch.bool: "bool",
}

native_dtype_dict = {
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "bool": torch.bool,
}


class Finfo:
    def __init__(self, torch_finfo: torch.finfo):
        self._torch_finfo = torch_finfo

    def __repr__(self):
        return repr(self._torch_finfo)

    @property
    def bits(self):
        return self._torch_finfo.bits

    @property
    def eps(self):
        return self._torch_finfo.eps

    @property
    def max(self):
        return self._torch_finfo.max

    @property
    def min(self):
        return self._torch_finfo.min

    @property
    def smallest_normal(self):
        return self._torch_finfo.tiny


# Array API Standard #
# -------------------#


@with_unsupported_dtypes({"2.1.2 and below": ("bfloat16", "float16")}, backend_version)
def astype(
    x: torch.Tensor,
    dtype: torch.dtype,
    /,
    *,
    copy: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = aikit.as_native_dtype(dtype)
    if x.dtype == dtype:
        return x.clone() if copy else x
    return x.to(dtype)


def broadcast_arrays(*arrays: torch.Tensor) -> List[torch.Tensor]:
    try:
        return list(torch.broadcast_tensors(*arrays))
    except RuntimeError as e:
        raise aikit.utils.exceptions.AikitBroadcastShapeError(e) from e


def broadcast_to(
    x: torch.Tensor,
    /,
    shape: Union[aikit.NativeShape, Sequence[int]],
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    aikit.utils.assertions.check_shapes_broadcastable(x.shape, shape)
    if x.ndim > len(shape):
        return torch.broadcast_to(x.reshape(-1), shape)
    return torch.broadcast_to(x, shape)


@_handle_nestable_dtype_info
def finfo(type: Union[torch.dtype, str, torch.Tensor, np.ndarray], /) -> Finfo:
    if isinstance(type, (torch.Tensor, np.ndarray)):
        type = type.dtype
    return Finfo(torch.finfo(aikit.as_native_dtype(type)))


@_handle_nestable_dtype_info
def iinfo(type: Union[torch.dtype, str, torch.Tensor, np.ndarray], /) -> torch.iinfo:
    if isinstance(type, (torch.Tensor, np.ndarray)):
        type = type.dtype
    return torch.iinfo(aikit.as_native_dtype(type))


def result_type(*arrays_and_dtypes: Union[torch.tensor, torch.dtype]) -> aikit.Dtype:
    input = []
    for val in arrays_and_dtypes:
        torch_val = as_native_dtype(val)
        if isinstance(torch_val, torch.dtype):
            torch_val = torch.tensor(1, dtype=torch_val)
        input.append(torch_val)

    result = torch.tensor(1, dtype=torch.result_type(input[0], input[1]))

    for i in range(2, len(input)):
        result = torch.tensor(1, dtype=torch.result_type(result, input[i]))
    return as_aikit_dtype(result.dtype)


# Extra #
# ------#


def as_aikit_dtype(
    dtype_in: Union[torch.dtype, str, int, float, complex, bool, np.dtype],
    /,
) -> aikit.Dtype:
    if dtype_in is int:
        return aikit.default_int_dtype()
    if dtype_in is float:
        return aikit.default_float_dtype()
    if dtype_in is complex:
        return aikit.default_complex_dtype()
    if dtype_in is bool:
        return aikit.Dtype("bool")
    if isinstance(dtype_in, np.dtype):
        dtype_in = dtype_in.name
    if isinstance(dtype_in, str):
        if dtype_in in native_dtype_dict:
            dtype_str = dtype_in
        else:
            raise aikit.utils.exceptions.AikitException(
                "Cannot convert to aikit dtype."
                f" {dtype_in} is not supported by PyTorch backend."
            )
    else:
        dtype_str = aikit_dtype_dict[dtype_in]

    if "uint" in dtype_str:
        return aikit.UintDtype(dtype_str)
    elif "int" in dtype_str:
        return aikit.IntDtype(dtype_str)
    elif "float" in dtype_str:
        return aikit.FloatDtype(dtype_str)
    elif "complex" in dtype_str:
        return aikit.ComplexDtype(dtype_str)
    elif "bool" in dtype_str:
        return aikit.Dtype("bool")
    else:
        raise aikit.utils.exceptions.AikitException(
            f"Cannot recognize {dtype_str} as a valid Dtype."
        )


@with_unsupported_dtypes({"2.1.2 and below": ("uint16",)}, backend_version)
def as_native_dtype(
    dtype_in: Union[torch.dtype, str, bool, int, float, np.dtype]
) -> torch.dtype:
    if dtype_in is int:
        return aikit.default_int_dtype(as_native=True)
    if dtype_in is float:
        return aikit.default_float_dtype(as_native=True)
    if dtype_in is complex:
        return aikit.default_complex_dtype(as_native=True)
    if dtype_in is bool:
        return torch.bool
    if isinstance(dtype_in, np.dtype):
        dtype_in = dtype_in.name
    if not isinstance(dtype_in, str):
        return dtype_in
    if dtype_in in native_dtype_dict:
        return native_dtype_dict[aikit.Dtype(dtype_in)]
    else:
        raise aikit.utils.exceptions.AikitException(
            f"Cannot convert to PyTorch dtype. {dtype_in} is not supported by PyTorch."
        )


def dtype(x: Union[torch.tensor, np.ndarray], *, as_native: bool = False) -> aikit.Dtype:
    if as_native:
        return aikit.as_native_dtype(x.dtype)
    return as_aikit_dtype(x.dtype)


def dtype_bits(dtype_in: Union[torch.dtype, str, np.dtype], /) -> int:
    dtype_str = as_aikit_dtype(dtype_in)
    if "bool" in dtype_str:
        return 1
    return int(
        dtype_str.replace("torch.", "")
        .replace("uint", "")
        .replace("int", "")
        .replace("bfloat", "")
        .replace("float", "")
        .replace("complex", "")
    )


def is_native_dtype(dtype_in: Union[torch.dtype, str], /) -> bool:
    if not aikit.is_hashable_dtype(dtype_in):
        return False
    return bool(dtype_in in aikit_dtype_dict and isinstance(dtype_in, torch.dtype))
