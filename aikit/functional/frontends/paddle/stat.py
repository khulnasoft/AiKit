# global
import aikit
from aikit.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from aikit.functional.frontends.paddle.func_wrapper import (
    to_aikit_arrays_and_back,
)


@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def mean(input, axis=None, keepdim=False, out=None):
    ret = aikit.mean(input, axis=axis, keepdims=keepdim, out=out)
    return ret


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")},
    "paddle",
)
@to_aikit_arrays_and_back
def median(x, axis=None, keepdim=False, name=None):
    x = (
        aikit.astype(x, aikit.float64)
        if aikit.dtype(x) == "float64"
        else aikit.astype(x, aikit.float32)
    )
    return aikit.median(x, axis=axis, keepdims=keepdim)


@with_supported_dtypes(
    {"2.5.0 and below": ("float16", "float32", "float64", "int32", "int64")},
    "paddle",
)
@to_aikit_arrays_and_back
def nanmedian(x, axis=None, keepdim=True, name=None):
    return aikit.nanmedian(x, axis=axis, keepdims=keepdim)


@with_supported_dtypes(
    {"2.6.0 and below": ("bool", "float16", "float32", "float64", "int32", "int64")},
    "paddle",
)
@to_aikit_arrays_and_back
def numel(x, name=None):
    prod = aikit.prod(x.size, dtype=aikit.int64)
    try:
        length = len(x)
    except (ValueError, TypeError):
        length = 1  # if 0 dimensional tensor with 1 element
    return aikit.array(prod if prod > 0 else aikit.array(length, dtype=aikit.int64))


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "uint16")},
    "paddle",
)
@to_aikit_arrays_and_back
def std(x, axis=None, unbiased=True, keepdim=False, name=None):
    x = (
        aikit.astype(x, aikit.float64)
        if aikit.dtype(x) == "float64"
        else aikit.astype(x, aikit.float32)
    )
    return aikit.std(x, axis=axis, correction=int(unbiased), keepdims=keepdim)


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def var(x, axis=None, unbiased=True, keepdim=False, name=None):
    if unbiased:
        correction = 1
    else:
        correction = 0
    return aikit.var(x, axis=axis, correction=correction, keepdims=keepdim)
