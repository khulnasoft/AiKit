# global
import aikit
from aikit.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
import aikit.functional.frontends.paddle as paddle_frontend
from aikit.functional.frontends.paddle.func_wrapper import (
    to_aikit_arrays_and_back,
)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def arange(start, end=None, step=1, dtype=None, name=None):
    return aikit.arange(start, end, step=step, dtype=dtype)


@with_supported_dtypes(
    {"2.5.2 and below": ("float16", "float32", "float64", "int32", "int64", "bool")},
    "paddle",
)
@to_aikit_arrays_and_back
def assign(x, output=None):
    if len(aikit.shape(x)) == 0:
        x = aikit.reshape(aikit.Array(x), (1,))
        if aikit.exists(output):
            output = aikit.reshape(aikit.Array(output), (1,))
    else:
        x = aikit.reshape(x, aikit.shape(x))
    ret = aikit.copy_array(x, to_aikit_array=False, out=output)
    return ret


@with_unsupported_dtypes(
    {"2.5.2 and below": ("bfloat16", "uint16", "uint32", "uint64")}, "paddle"
)
@to_aikit_arrays_and_back
def clone(x):
    return aikit.copy_array(x)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64")},
    "paddle",
)
@to_aikit_arrays_and_back
def complex(real, imag, name=None):
    assert real.dtype == imag.dtype, (
        "(InvalidArgument) The type of data we are trying to retrieve does not match"
        " the type of data currently contained in the container."
    )
    complex_dtype = "complex64" if real.dtype == "float32" else "complex128"
    imag_cmplx = aikit.astype(imag, complex_dtype) * 1j
    complex_array = real + imag_cmplx
    return complex_array


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def diag(x, offset=0, padding_value=0, name=None):
    if len(x.shape) == 1:
        padding_value = aikit.astype(padding_value, aikit.dtype(x))
        ret = aikit.diagflat(x, offset=offset, padding_value=padding_value)
        if len(ret.shape) != 2:
            ret = aikit.reshape(ret, (1, 1))
    else:
        ret = aikit.diag(x, k=offset)
    return ret


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def diagflat(x, offset=0, name=None):
    arr = aikit.diagflat(x, offset=offset)
    return arr


@to_aikit_arrays_and_back
def empty(shape, dtype=None):
    return aikit.empty(shape=shape, dtype=dtype)


@to_aikit_arrays_and_back
def empty_like(x, dtype=None, name=None):
    return aikit.empty_like(x, dtype=dtype)


@to_aikit_arrays_and_back
def eye(num_rows, num_columns=None, dtype=None, name=None):
    return aikit.eye(num_rows, num_columns, dtype=dtype)


@to_aikit_arrays_and_back
def full(shape, fill_value, /, *, dtype=None, name=None):
    dtype = "float32" if dtype is None else dtype
    return aikit.full(shape, fill_value, dtype=dtype)


@to_aikit_arrays_and_back
def full_like(x, fill_value, /, *, dtype=None, name=None):
    dtype = x.dtype if dtype is None else dtype
    return aikit.full_like(x, fill_value, dtype=dtype)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def linspace(start, stop, num, dtype=None, name=None):
    return aikit.linspace(start, stop, num=num, dtype=dtype)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def logspace(start, stop, num, base=10.0, dtype=None, name=None):
    return aikit.logspace(start, stop, num=num, base=base, dtype=dtype)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def meshgrid(*args, **kwargs):
    return aikit.meshgrid(*args, indexing="ij")


@with_unsupported_dtypes({"2.5.2 and below": "int8"}, "paddle")
@to_aikit_arrays_and_back
def ones(shape, /, *, dtype=None, name=None):
    dtype = "float32" if dtype is None else dtype
    return aikit.ones(shape, dtype=dtype)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("uint8", "int8", "complex64", "complex128")}, "paddle"
)
@to_aikit_arrays_and_back
def ones_like(x, /, *, dtype=None, name=None):
    dtype = x.dtype if dtype is None else dtype
    return aikit.ones_like(x, dtype=dtype)


@to_aikit_arrays_and_back
def to_tensor(data, /, *, dtype=None, place=None, stop_gradient=True):
    array = aikit.array(data, dtype=dtype, device=place)
    return paddle_frontend.Tensor(array, dtype=dtype, place=place)


@with_unsupported_dtypes(
    {
        "2.5.2 and below": (
            "uint8",
            "int8",
            "int16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
def tril(x, diagonal=0, name=None):
    return aikit.tril(x, k=diagonal)


@with_supported_dtypes({"2.5.2 and below": ("int32", "int64")}, "paddle")
@to_aikit_arrays_and_back
def tril_indices(row, col, offset=0, dtype="int64"):
    arr = aikit.tril_indices(row, col, offset)
    arr = aikit.astype(arr, dtype)
    return arr


@with_unsupported_dtypes(
    {
        "2.5.2 and below": (
            "uint8",
            "int8",
            "int16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
def triu(x, diagonal=0, name=None):
    return aikit.triu(x, k=diagonal)


@with_supported_dtypes({"2.5.2 and below": ("int32", "int64")}, "paddle")
@to_aikit_arrays_and_back
def triu_indices(row, col=None, offset=0, dtype="int64"):
    arr = aikit.triu_indices(row, col, offset)
    if not aikit.to_scalar(aikit.shape(arr[0], as_array=True)):
        return arr
    arr = aikit.astype(arr, dtype)
    return arr


@with_unsupported_dtypes({"2.5.2 and below": "int8"}, "paddle")
@to_aikit_arrays_and_back
def zeros(shape, /, *, dtype=None, name=None):
    dtype = "float32" if dtype is None else dtype
    return aikit.zeros(shape, dtype=dtype)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("uint8", "int8", "complex64", "complex128")}, "paddle"
)
@to_aikit_arrays_and_back
def zeros_like(x, /, *, dtype=None, name=None):
    dtype = x.dtype if dtype is None else dtype
    return aikit.zeros_like(x, dtype=dtype)
