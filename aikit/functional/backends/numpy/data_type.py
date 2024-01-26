# global
from typing import Optional, Union, Sequence, List

import numpy as np

# local
import aikit
from aikit.func_wrapper import with_unsupported_dtypes
from aikit.functional.aikit.data_type import _handle_nestable_dtype_info
from . import backend_version

aikit_dtype_dict = {
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
    np.dtype("uint64"): "uint64",
    # np.dtype("bfloat16"): "bfloat16",
    np.dtype("float16"): "float16",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("complex64"): "complex64",
    np.dtype("complex128"): "complex128",
    np.dtype("bool"): "bool",
    np.int8: "int8",
    np.int16: "int16",
    np.int32: "int32",
    np.int64: "int64",
    np.uint8: "uint8",
    np.uint16: "uint16",
    np.uint32: "uint32",
    np.uint64: "uint64",
    np.float16: "float16",
    np.float32: "float32",
    np.float64: "float64",
    np.complex64: "complex64",
    np.complex128: "complex128",
    np.bool_: "bool",
}

native_dtype_dict = {
    "int8": np.dtype("int8"),
    "int16": np.dtype("int16"),
    "int32": np.dtype("int32"),
    "int64": np.dtype("int64"),
    "uint8": np.dtype("uint8"),
    "uint16": np.dtype("uint16"),
    "uint32": np.dtype("uint32"),
    "uint64": np.dtype("uint64"),
    "float16": np.dtype("float16"),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "complex64": np.dtype("complex64"),
    "complex128": np.dtype("complex128"),
    "bool": np.dtype("bool"),
}

char_rep_dtype_dict = {
    "?": "bool",
    "i": int,
    "i1": "int8",
    "i2": "int16",
    "i4": "int32",
    "i8": "int64",
    "f": float,
    "f2": "float16",
    "f4": "float32",
    "f8": "float64",
    "c": complex,
    "c8": "complex64",
    "c16": "complex128",
    "u": "uint32",
    "u1": "uint8",
    "u2": "uint16",
    "u4": "uint32",
    "u8": "uint64",
}


class Finfo:
    def __init__(self, np_finfo: np.finfo):
        self._np_finfo = np_finfo

    def __repr__(self):
        return repr(self._np_finfo)

    @property
    def bits(self):
        return self._np_finfo.bits

    @property
    def eps(self):
        return float(self._np_finfo.eps)

    @property
    def max(self):
        return float(self._np_finfo.max)

    @property
    def min(self):
        return float(self._np_finfo.min)

    @property
    def smallest_normal(self):
        return float(self._np_finfo.tiny)


# Array API Standard #
# -------------------#


def astype(
    x: np.ndarray,
    dtype: np.dtype,
    /,
    *,
    copy: bool = True,
    out: Optional[aikit.Array] = None,
) -> np.ndarray:
    dtype = aikit.as_native_dtype(dtype)
    if x.dtype == dtype:
        return np.copy(x) if copy else x
    return x.astype(dtype)


def broadcast_arrays(*arrays: np.ndarray) -> List[np.ndarray]:
    try:
        return np.broadcast_arrays(*arrays)
    except ValueError as e:
        raise aikit.utils.exceptions.AikitBroadcastShapeError(e) from e


@with_unsupported_dtypes({"1.26.3 and below": ("complex",)}, backend_version)
def broadcast_to(
    x: np.ndarray,
    /,
    shape: Union[aikit.NativeShape, Sequence[int]],
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    aikit.utils.assertions.check_shapes_broadcastable(x.shape, shape)
    if x.ndim > len(shape):
        return np.broadcast_to(x.reshape([-1]), shape)
    return np.broadcast_to(x, shape)


@_handle_nestable_dtype_info
def finfo(type: Union[np.dtype, str, np.ndarray], /) -> Finfo:
    if isinstance(type, np.ndarray):
        type = type.dtype
    return Finfo(np.finfo(aikit.as_native_dtype(type)))


@_handle_nestable_dtype_info
def iinfo(type: Union[np.dtype, str, np.ndarray], /) -> np.iinfo:
    if isinstance(type, np.ndarray):
        type = type.dtype
    return np.iinfo(aikit.as_native_dtype(type))


def result_type(*arrays_and_dtypes: Union[np.ndarray, np.dtype]) -> aikit.Dtype:
    if len(arrays_and_dtypes) <= 1:
        return np.result_type(arrays_and_dtypes)
    result = np.result_type(arrays_and_dtypes[0], arrays_and_dtypes[1])
    for i in range(2, len(arrays_and_dtypes)):
        result = np.result_type(result, arrays_and_dtypes[i])
    return as_aikit_dtype(result)


# Extra #
# ------#


def as_aikit_dtype(
    dtype_in: Union[np.dtype, str, int, float, complex, bool],
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

    if isinstance(dtype_in, str):
        if dtype_in in char_rep_dtype_dict:
            return as_aikit_dtype(char_rep_dtype_dict[dtype_in])
        if dtype_in in native_dtype_dict:
            dtype_str = dtype_in
        else:
            raise aikit.utils.exceptions.AikitException(
                "Cannot convert to aikit dtype."
                f" {dtype_in} is not supported by NumPy backend."
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


@with_unsupported_dtypes({"1.26.3 and below": ("bfloat16",)}, backend_version)
def as_native_dtype(dtype_in: Union[np.dtype, str, bool, int, float], /) -> np.dtype:
    if dtype_in is int:
        return aikit.default_int_dtype(as_native=True)
    if dtype_in is float:
        return aikit.default_float_dtype(as_native=True)
    if dtype_in is complex:
        return aikit.default_complex_dtype(as_native=True)
    if dtype_in is bool:
        return np.dtype("bool")
    if not isinstance(dtype_in, str):
        return dtype_in
    if dtype_in in char_rep_dtype_dict:
        return as_native_dtype(char_rep_dtype_dict[dtype_in])
    if dtype_in in native_dtype_dict.values():
        return native_dtype_dict[aikit.Dtype(dtype_in)]
    else:
        raise aikit.utils.exceptions.AikitException(
            f"Cannot convert to numpy dtype. {dtype_in} is not supported by NumPy."
        )


def dtype(x: np.ndarray, *, as_native: bool = False) -> aikit.Dtype:
    if as_native:
        return aikit.to_native(x).dtype
    return as_aikit_dtype(x.dtype)


def dtype_bits(dtype_in: Union[np.dtype, str], /) -> int:
    dtype_str = as_aikit_dtype(dtype_in)
    if "bool" in dtype_str:
        return 1
    return int(
        dtype_str.replace("uint", "")
        .replace("int", "")
        .replace("bfloat", "")
        .replace("float", "")
        .replace("complex", "")
    )


def is_native_dtype(dtype_in: Union[np.dtype, str], /) -> bool:
    if not aikit.is_hashable_dtype(dtype_in):
        return False
    return dtype_in in aikit_dtype_dict
