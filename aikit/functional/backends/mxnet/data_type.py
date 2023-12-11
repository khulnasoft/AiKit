import mxnet as mx
from typing import Optional, Union, Sequence, List
import numpy as np
import aikit
from aikit.functional.aikit.data_type import _handle_nestable_dtype_info
from aikit.utils.exceptions import AikitNotImplementedException

aikit_dtype_dict = {
    np.dtype("int8"): "int8",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("uint8"): "uint8",
    np.dtype("float16"): "float16",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("bool"): "bool",
    np.int8: "int8",
    np.int32: "int32",
    np.int64: "int64",
    np.uint8: "uint8",
    np.float16: "float16",
    np.float32: "float32",
    np.float64: "float64",
    np.bool_: "bool",
}
native_dtype_dict = {
    "int8": np.int8,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "bool": np.bool_,
}

char_rep_dtype_dict = {
    "?": "bool",
    "i": int,
    "i1": "int8",
    "i4": "int32",
    "i8": "int64",
    "f": float,
    "f2": "float16",
    "f4": "float32",
    "f8": "float64",
    "u1": "uint8",
}


class Finfo:
    def __init__(self, mx_finfo: mx.np.finfo):
        self._mx_finfo = mx_finfo

    def __repr__(self):
        return repr(self._mx_finfo)

    @property
    def bits(self):
        return self._mx_finfo.bits

    @property
    def eps(self):
        return float(self._mx_finfo.eps)

    @property
    def max(self):
        return float(self._mx_finfo.max)

    @property
    def min(self):
        return float(self._mx_finfo.min)

    @property
    def smallest_normal(self):
        return float(self._mx_finfo.tiny)


class Bfloat16Finfo:
    def __init__(self, mx_finfo: mx.np.finfo):
        self._mx_finfo = mx_finfo

    def __repr__(self):
        return repr(self._mx_finfo)


def astype(
    x: Union[(None, mx.ndarray.NDArray)],
    dtype: Union[(None, str)],
    /,
    *,
    copy: bool = True,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    dtype = aikit.as_native_dtype(dtype)
    if x.dtype == dtype:
        return mx.nd.copy(x) if copy else x
    return x.astype(dtype)


def broadcast_arrays(
    *arrays: Union[(None, mx.ndarray.NDArray)]
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise AikitNotImplementedException()


def broadcast_to(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    shape: Union[(aikit.NativeShape, Sequence[int])],
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


@_handle_nestable_dtype_info
def finfo(type: Union[str, mx.ndarray.NDArray, np.dtype], /) -> Finfo:
    if isinstance(type, mx.ndarray.NDArray):
        type = type.dtype
    return Finfo(mx.np.finfo(aikit.as_native_dtype(type)))


@_handle_nestable_dtype_info
def iinfo(type: Union[str, mx.ndarray.NDArray, np.dtype], /) -> np.iinfo:
    # using np.iinfo as mx use np dtypes and mxnet iinfo not provided
    if isinstance(type, mx.ndarray.NDArray):
        type = type.asnumpy().dtype
    return np.iinfo(aikit.as_native_dtype(type))


def result_type(*arrays_and_dtypes: Union[(None, mx.ndarray.NDArray)]) -> aikit.Dtype:
    raise AikitNotImplementedException()


def as_aikit_dtype(
    dtype_in: Union[(str, int, float, complex, bool, np.dtype)], /
) -> aikit.Dtype:
    if dtype_in is int:
        return aikit.default_int_dtype()
    if dtype_in is float:
        return aikit.default_float_dtype()
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
                f" {dtype_in} is not supported by MXNet backend."
            )
    else:
        dtype_str = aikit_dtype_dict[dtype_in]

    if "int" in dtype_str:
        return aikit.IntDtype(dtype_str)
    elif "float" in dtype_str:
        return aikit.FloatDtype(dtype_str)
    elif "bool" in dtype_str:
        return aikit.Dtype("bool")
    else:
        raise aikit.utils.exceptions.AikitException(
            f"Cannot recognize {dtype_str} as a valid Dtype."
        )


def as_native_dtype(dtype_in: Union[(None, str, bool, int, float, np.dtype)]) -> None:
    if dtype_in is int:
        return aikit.default_int_dtype(as_native=True)
    if dtype_in is float:
        return aikit.default_float_dtype(as_native=True)
    if dtype_in is bool:
        return np.dtype("bool")
    if not isinstance(dtype_in, str):
        return dtype_in
    if dtype_in in char_rep_dtype_dict:
        return as_native_dtype(char_rep_dtype_dict[dtype_in])
    if dtype_in in native_dtype_dict:
        return native_dtype_dict[aikit.Dtype(dtype_in)]
    else:
        raise aikit.utils.exceptions.AikitException(
            f"Cannot convert to MXNet dtype. {dtype_in} is not supported by MXNet."
        )


def dtype(
    x: Union[(None, mx.ndarray.NDArray, np.ndarray)], *, as_native: bool = False
) -> aikit.Dtype:
    if as_native:
        return aikit.as_native_dtype(x.dtype)
    return as_aikit_dtype(x.dtype)


def dtype_bits(dtype_in: Union[(None, str, np.dtype)], /) -> int:
    raise AikitNotImplementedException()


def is_native_dtype(dtype_in: Union[(None, str)], /) -> bool:
    raise AikitNotImplementedException()
