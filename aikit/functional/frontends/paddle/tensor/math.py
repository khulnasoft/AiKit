# local
from ..math import *  # noqa: F401
import aikit
from aikit.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from aikit.functional.frontends.paddle.func_wrapper import to_aikit_arrays_and_back

# NOTE:
# Only inplace functions are to be added in this file.
# Please add non-inplace counterparts to `/frontends/paddle/math.py`.


@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def ceil_(x, name=None):
    return aikit.ceil(x, out=x)


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def clip_(x, min=None, max=None, name=None):
    aikit.utils.assertions.check_all_or_any_fn(
        min,
        max,
        fn=aikit.exists,
        type="any",
        limit=[1, 2],
        message="at most one of min or max can be None",
    )
    if min is None:
        min = aikit.min(x)
    if max is None:
        max = aikit.max(x)
    res = aikit.clip(x, min, max)
    if res.dtype != x.dtype:
        res = aikit.astype(res, x.dtype)
    return res


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def exp_(x, name=None):
    return aikit.inplace_update(x, exp(x))


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def floor_(x, name=None):
    return aikit.inplace_update(x, floor(x))


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def lerp_(x, y, weight, name=None):
    return aikit.inplace_update(x, lerp(x, y, weight))


@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def reciprocal_(x, name=None):
    return aikit.inplace_update(x, reciprocal(x))


@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def round_(x, name=None):
    return aikit.inplace_update(x, round(x))


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
def rsqrt_(x, name=None):
    return aikit.inplace_update(x, reciprocal(sqrt(x)))


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def sqrt_(x, name=None):
    return aikit.inplace_update(x, sqrt(x))


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def subtract_(x, y, name=None):
    return aikit.inplace_update(x, subtract(x, y))


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def tanh_(x, name=None):
    return aikit.inplace_update(x, tanh(x))
