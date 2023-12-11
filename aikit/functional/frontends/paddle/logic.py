# global
import aikit
import aikit.functional.frontends.paddle as paddle
from aikit.func_wrapper import (
    with_unsupported_dtypes,
    handle_out_argument,
    with_supported_dtypes,
)
from aikit.functional.frontends.paddle.func_wrapper import (
    to_aikit_arrays_and_back,
)


@with_supported_dtypes(
    {
        "2.5.2 and below": (
            "float32",
            "float64",
            "bool",
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
@handle_out_argument
def allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None):
    ret = aikit.allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return paddle.to_tensor([ret])


@with_supported_dtypes(
    {
        "2.5.2 and below": (
            "bool",
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
@handle_out_argument
def bitwise_and(x, y, /, *, name=None, out=None):
    return aikit.bitwise_and(x, y, out=out)


@with_supported_dtypes(
    {
        "2.5.2 and below": (
            "bool",
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
@handle_out_argument
def bitwise_not(x, out=None, name=None):
    return aikit.bitwise_invert(x, out=out)


@with_supported_dtypes(
    {
        "2.5.2 and below": (
            "bool",
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
@handle_out_argument
def bitwise_or(x, y, name=None, out=None):
    return aikit.bitwise_or(x, y, out=out)


@with_supported_dtypes(
    {
        "2.5.2 and below": (
            "bool",
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
@handle_out_argument
def bitwise_xor(x, y, /, *, name=None, out=None):
    return aikit.bitwise_xor(x, y, out=out)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("uint8", "int8", "int16", "complex64", "complex128")},
    "paddle",
)
@to_aikit_arrays_and_back
def equal(x, y, /, *, name=None):
    return aikit.equal(x, y)


@with_unsupported_dtypes(
    {
        "2.5.2 and below": (
            "uint8",
            "int8",
            "int16",
            "float16",
            "complex64",
            "complex128",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
def equal_all(x, y, /, *, name=None):
    return paddle.to_tensor([aikit.array_equal(x, y)])


@with_unsupported_dtypes(
    {"2.5.2 and below": ("bool", "uint8", "int8", "int16", "complex64", "complex128")},
    "paddle",
)
@to_aikit_arrays_and_back
def greater_equal(x, y, /, *, name=None):
    return aikit.greater_equal(x, y)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("bool", "uint8", "int8", "int16", "complex64", "complex128")},
    "paddle",
)
@to_aikit_arrays_and_back
def greater_than(x, y, /, *, name=None):
    return aikit.greater(x, y)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("uint8", "int8", "int16", "complex64", "complex128")},
    "paddle",
)
@to_aikit_arrays_and_back
def is_empty(x, name=None):
    return aikit.is_empty(x)


@to_aikit_arrays_and_back
def is_tensor(x):
    return aikit.is_array(x)


@with_supported_dtypes(
    {
        "2.5.2 and below": (
            "float32",
            "float64",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
def isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None):
    return aikit.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("bool", "uint8", "int8", "int16", "complex64", "complex128")},
    "paddle",
)
@to_aikit_arrays_and_back
def less_equal(x, y, /, *, name=None):
    return aikit.less_equal(x, y)


@with_supported_dtypes(
    {"2.5.2 and below": ("bool", "float16", "float32", "float64", "int32", "int64")},
    "paddle",
)
@to_aikit_arrays_and_back
def less_than(x, y, /, *, name=None):
    return aikit.astype(aikit.less(x, y), aikit.bool)


@with_supported_dtypes(
    {
        "2.5.2 and below": (
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "float32",
            "float64",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
@handle_out_argument
def logical_and(x, y, /, *, name=None, out=None):
    return aikit.logical_and(x, y, out=out)


@with_supported_dtypes(
    {
        "2.5.2 and below": (
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "float32",
            "float64",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
@handle_out_argument
def logical_not(x, /, *, name=None, out=None):
    return aikit.logical_not(x, out=out)


@with_supported_dtypes(
    {
        "2.5.2 and below": (
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "float32",
            "float64",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
@handle_out_argument
def logical_or(x, y, /, *, name=None, out=None):
    return aikit.logical_or(x, y, out=out)


@with_supported_dtypes(
    {
        "2.5.2 and below": (
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "float32",
            "float64",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
@handle_out_argument
def logical_xor(x, y, /, *, name=None, out=None):
    return aikit.logical_xor(x, y, out=out)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("uint8", "int8", "int16", "complex64", "complex128")},
    "paddle",
)
@to_aikit_arrays_and_back
def not_equal(x, y, /, *, name=None):
    if aikit.is_float_dtype(x):
        diff = aikit.abs(aikit.subtract(x, y))
        res = aikit.not_equal(x, y)
        return aikit.where(diff < 1e-8, False, res)
    return aikit.not_equal(x, y)
