# global
import aikit
from aikit.func_wrapper import (
    with_unsupported_dtypes,
    with_supported_dtypes,
    with_supported_device_and_dtypes,
)
from aikit.functional.frontends.paddle.func_wrapper import to_aikit_arrays_and_back


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def abs(x, name=None):
    return aikit.abs(x)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def acos(x, name=None):
    return aikit.acos(x)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def acosh(x, name=None):
    return aikit.acosh(x)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("bool", "unsigned", "int8", "float16", "bfloat16")}, "paddle"
)
@to_aikit_arrays_and_back
def add(x, y, name=None):
    return aikit.add(x, y)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("bool", "unsigned", "int8", "float16", "bfloat16")}, "paddle"
)
@to_aikit_arrays_and_back
def add_(x, y, name=None):
    return aikit.inplace_update(x, add(x, y))


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def addmm(input, x, y, beta=1.0, alpha=1.0, name=None):
    value = alpha * aikit.matmul(x, y) + (beta * input)
    return value


@with_supported_dtypes({"2.5.0 and below": "bool"}, "paddle")
@to_aikit_arrays_and_back
def all(x, axis, keepdim=False, name=None):
    return aikit.all(x, axis=axis, keepdims=keepdim)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def amax(x, axis=None, keepdims=False):
    if axis is None:
        return aikit.max(x)
    if isinstance(axis, int):
        axis = [axis]
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] += x.ndim
    for i in axis:
        if i < 0 or i >= x.ndim:
            raise ValueError(f"axis {i} is out of range [-0:{x.ndim}]")
    return aikit.max(x, axis=axis, keepdims=keepdims)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def amin(x, axis=None, keepdim=False, name=None):
    return aikit.min(x, axis=axis, keepdims=keepdim)


@with_supported_dtypes(
    {"2.5.2 and below": ("complex64", "complex128", "float32", "float64")},
    "paddle",
)
@to_aikit_arrays_and_back
def angle(x, name=None):
    return aikit.angle(x)


@with_supported_dtypes({"2.5.0 and below": "bool"}, "paddle")
@to_aikit_arrays_and_back
def any(x, axis=None, keepdim=False, name=None):
    return aikit.any(x, axis=axis, keepdims=keepdim)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def asin(x, name=None):
    return aikit.asin(x)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def asinh(x, name=None):
    return aikit.asinh(x)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def atan(x, name=None):
    return aikit.atan(x)


@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def atan2(x, y, name=None):
    return aikit.atan2(x, y)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def atanh(x, name=None):
    return aikit.atanh(x)


@with_supported_dtypes({"2.5.2 and below": ("int32", "int64")}, "paddle")
@to_aikit_arrays_and_back
def broadcast_shape(x_shape, y_shape):
    return aikit.broadcast_shapes(x_shape, y_shape)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def ceil(x, name=None):
    return aikit.ceil(x)


@with_unsupported_dtypes({"2.4.2 and below": ("int16", "float16")}, "paddle")
@to_aikit_arrays_and_back
def conj(x, name=None):
    return aikit.conj(x)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def cos(x, name=None):
    return aikit.cos(x)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def cosh(x, name=None):
    return aikit.cosh(x)


@with_supported_dtypes(
    {"2.5.2 and below": ("int32", "int64", "float16", "float32", "float64", "bool")},
    "paddle",
)
@to_aikit_arrays_and_back
def count_nonzero(x, axis=None, keepdim=False, name=None):
    return aikit.astype(aikit.count_nonzero(x, axis=axis, keepdims=keepdim), aikit.int64)


@with_supported_dtypes(
    {
        "2.5.2 and below": (
            "int32",
            "int64",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
def cumprod(x, dim=None, dtype=None, name=None):
    return aikit.cumprod(x, axis=dim, dtype=dtype)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def cumsum(x, axis=None, dtype=None, name=None):
    return aikit.cumsum(x, axis=axis, dtype=dtype)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def deg2rad(x, name=None):
    return aikit.deg2rad(x)


@with_supported_dtypes(
    {
        "2.5.2 and below": (
            "int32",
            "int64",
            "float64",
            "complex128",
            "float32",
            "complex64",
            "bool",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
def diagonal(x, offset=0, axis1=0, axis2=1, name=None):
    return aikit.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def diff(x, n=1, axis=-1, prepend=None, append=None, name=None):
    return aikit.diff(x, n=n, axis=axis, prepend=prepend, append=append)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def digamma(x, name=None):
    digamma_fun = aikit.digamma
    return aikit.array(digamma_fun(x), dtype=x.dtype)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def divide(x, y, name=None):
    return aikit.divide(x, y)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def erf(x, name=None):
    return aikit.erf(x)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def exp(x, name=None):
    return aikit.exp(x)


@with_supported_dtypes({"2.5.2 and below": ("float16", "float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def expm1(x, name=None):
    return aikit.expm1(x)


@with_supported_dtypes(
    {"2.5.2 and below": ("bfloat16", "float32", "float64")}, "paddle"
)
@to_aikit_arrays_and_back
def floor(x, name=None):
    return aikit.floor(x)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def floor_divide(x, y, name=None):
    return aikit.floor_divide(x, y)


@with_supported_device_and_dtypes(
    {
        "2.5.2 and below": {
            "cpu": ("float32", "float64", "int32", "int64"),
            "gpu": ("float16", "float32", "float64", "int32", "int64"),
        }
    },
    "paddle",
)
@to_aikit_arrays_and_back
def floor_mod(x, y, name=None):
    return aikit.remainder(x, y)


@with_unsupported_dtypes({"2.5.2 and below": "bfloat16"}, "paddle")
@to_aikit_arrays_and_back
def fmax(x, y, name=None):
    return aikit.fmax(x, y)


@with_unsupported_dtypes({"2.5.2 and below": "bfloat16"}, "paddle")
@to_aikit_arrays_and_back
def fmin(x, y, name=None):
    return aikit.fmin(x, y)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def frac(x, name=None):
    y = aikit.trunc(x)
    return aikit.subtract(x, y)


@with_supported_dtypes({"2.5.2 and below": ("int32", "int64")}, "paddle")
@to_aikit_arrays_and_back
def gcd(x, y, name=None):
    return aikit.gcd(x, y)


@with_supported_dtypes(
    {"2.5.2 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def heaviside(x, y, name=None):
    return aikit.heaviside(x, y)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def inner(x, y, name=None):
    result = aikit.inner(x, y)
    if (x.shape == () and y.shape == (1,)) or (x.shape == (1,) and y.shape == ()):
        result = result.reshape((1,))
    elif x.shape == (1,) and y.shape == (1,):
        result = result.reshape((1,))
    return result


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def inverse(x, name=None):
    return aikit.inv(x)


@with_supported_dtypes(
    {"2.5.2 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def isfinite(x, name=None):
    return aikit.isfinite(x)


@with_supported_dtypes(
    {"2.5.2 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def isinf(x, name=None):
    return aikit.isinf(x)


@with_supported_dtypes(
    {"2.5.2 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def isnan(x, name=None):
    return aikit.isnan(x)


@with_supported_dtypes(
    {"2.5.2 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def kron(x, y, name=None):
    return aikit.kron(x, y)


@with_supported_dtypes({"2.5.2 and below": ("int32", "int64")}, "paddle")
@to_aikit_arrays_and_back
def lcm(x, y, name=None):
    return aikit.lcm(x, y)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def lerp(x, y, weight, name=None):
    return aikit.lerp(x, y, weight)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def lgamma(x, name=None):
    return aikit.lgamma(x)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def log(x, name=None):
    return aikit.log(x)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def log10(x, name=None):
    return aikit.log10(x)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def log1p(x, name=None):
    return aikit.log1p(x)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def log2(x, name=None):
    return aikit.log2(x)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def logit(x, eps=None, name=None):
    return aikit.logit(x, eps=eps)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def max(x, axis=None, keepdim=False, name=None):
    return aikit.max(x, axis=axis, keepdims=keepdim)


# maximum
@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def maximum(x, y, name=None):
    return aikit.maximum(x, y)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def min(x, axis=None, keepdim=False, name=None):
    return aikit.min(x, axis=axis, keepdims=keepdim)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def minimum(x, y, name=None):
    return aikit.minimum(x, y)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def mm(input, mat2, name=None):
    return aikit.matmul(input, mat2)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def multiply(x, y, name=None):
    return aikit.multiply(x, y)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def nanmean(x, axis=None, keepdims=False):
    return aikit.nanmean(x, axis=axis, keepdims=keepdims)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def nansum(x, axis=None, dtype=None, name=None):
    return aikit.nansum(x, axis=axis, dtype=dtype)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int8", "int16", "int32", "int64")},
    "paddle",
)
@to_aikit_arrays_and_back
def neg(x, name=None):
    return aikit.negative(x)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def outer(x, y, name=None):
    return aikit.outer(x, y)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def pow(x, y, name=None):
    return aikit.pow(x, y)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def prod(x, axis=None, keepdim=False, dtype=None, name=None):
    return aikit.prod(x, axis=axis, keepdims=keepdim, dtype=dtype)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def rad2deg(x, name=None):
    return aikit.rad2deg(x)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def reciprocal(x, name=None):
    return aikit.reciprocal(x)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def remainder(x, y, name=None):
    return aikit.remainder(x, y)


@with_supported_device_and_dtypes(
    {
        "2.5.2 and below": {
            "cpu": ("float32", "float64"),
            "gpu": ("float16", "float32", "float64"),
        }
    },
    "paddle",
)
@to_aikit_arrays_and_back
def remainder_(x, y, name=None):
    return aikit.inplace_update(x, remainder(x, y))


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def round(x, name=None):
    sign = aikit.sign(x)
    x = sign * aikit.floor(aikit.abs(x) + 0.5)
    return x


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def rsqrt(x, name=None):
    return 1 / aikit.sqrt(x)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def sgn(x, name=None):
    return aikit.sign(x, np_variant=True)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def sign(x, name=None):
    return aikit.sign(x, np_variant=False)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def sin(x, name=None):
    return aikit.sin(x)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def sinh(x, name=None):
    return aikit.sinh(x)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def sqrt(x, name=None):
    return aikit.sqrt(x)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def square(x, name=None):
    return aikit.square(x)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def stanh(x, scale_a=0.67, scale_b=1.7159, name=None):
    # TODO this function will be simplified as soon as the aikit.stanh(x,a,b) is added
    exp_ax = aikit.exp(aikit.multiply(scale_a, x))
    exp_minus_ax = aikit.exp(aikit.multiply(-scale_a, x))
    numerator = aikit.subtract(exp_ax, exp_minus_ax)
    denominator = aikit.add(exp_ax, exp_minus_ax)
    ret = aikit.multiply(scale_b, aikit.divide(numerator, denominator))
    return ret


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def subtract(x, y, name=None):
    return aikit.subtract(x, y)


@with_supported_dtypes(
    {
        "2.5.2 and below": (
            "float64",
            "int64",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
def sum(x, axis=None, dtype=None, keepdim=False, name=None):
    return aikit.sum(
        x,
        axis=axis,
        keepdims=keepdim,
        dtype=dtype,
    )


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int6")}, "paddle"
)
@to_aikit_arrays_and_back
def take(
    x,
    index,
    mode="raise",
    name=None,
):
    if mode not in ["raise", "wrap", "clip"]:
        raise ValueError(
            f"'mode' in 'take' should be 'raise', 'wrap', 'clip', but received {mode}."
        )
    x = aikit.reshape(x, (-1,))
    if mode == "clip":
        index = aikit.clip(index, 0, x.shape[-1] - 1)
    elif mode == "wrap":
        index = aikit.where(index < 0, index % x.shape[-1], index)
        index = aikit.where(index >= x.shape[-1], index % x.shape[-1], index)
    return aikit.gather(x, index, axis=0)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def tan(x, name=None):
    return aikit.tan(x)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def tanh(x, name=None):
    return aikit.tanh(x)


@with_supported_dtypes(
    {"2.5.2 and below": ("int32", "int64", "float32", "float64")}, "paddle"
)
@to_aikit_arrays_and_back
def trace(x, offset=0, axis1=0, axis2=1, name=None):
    return aikit.trace(x, offset=offset, axis1=axis1, axis2=axis2)


@with_supported_dtypes(
    {"2.4.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def trunc(x, name=None):
    return aikit.trunc(x)


mod = remainder
