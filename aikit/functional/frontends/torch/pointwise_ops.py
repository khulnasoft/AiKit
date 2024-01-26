# global
import aikit
from aikit.func_wrapper import (
    with_unsupported_dtypes,
    with_supported_dtypes,
)
import aikit.functional.frontends.torch as torch_frontend
from aikit.functional.frontends.torch.func_wrapper import (
    to_aikit_arrays_and_back,
)


@to_aikit_arrays_and_back
def abs(input, *, out=None):
    return aikit.abs(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def acos(input, *, out=None):
    return aikit.acos(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def acosh(input, *, out=None):
    return aikit.acosh(input, out=out)


@with_supported_dtypes(
    {"1.12.0 and below": ("float32", "float64", "int32", "int64")}, "jax"
)
@to_aikit_arrays_and_back
def add(input, other, *, alpha=1, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.add(input, other, alpha=alpha, out=out)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.2 and below": ("float16", "bfloat16")}, "torch")
def addcdiv(input, tensor1, tensor2, *, value=1, out=None):
    return aikit.add(input, aikit.multiply(value, aikit.divide(tensor1, tensor2)), out=out)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.2 and below": ("float16", "bfloat16")}, "torch")
def addcmul(input, tensor1, tensor2, *, value=1, out=None):
    return aikit.add(input, aikit.multiply(value, aikit.multiply(tensor1, tensor2)), out=out)


@to_aikit_arrays_and_back
def angle(input, *, out=None):
    return aikit.angle(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def asin(input, *, out=None):
    return aikit.asin(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def asinh(input, *, out=None):
    return aikit.asinh(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def atan(input, *, out=None):
    return aikit.atan(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def atan2(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.atan2(input, other, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def atanh(input, *, out=None):
    return aikit.atanh(input, out=out)


@to_aikit_arrays_and_back
def bitwise_and(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.bitwise_and(input, other, out=out)


@to_aikit_arrays_and_back
def bitwise_left_shift(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.bitwise_left_shift(input, other, out=out)


@to_aikit_arrays_and_back
def bitwise_not(input, *, out=None):
    return aikit.bitwise_invert(input, out=out)


@to_aikit_arrays_and_back
def bitwise_or(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.bitwise_or(input, other, out=out)


@to_aikit_arrays_and_back
def bitwise_right_shift(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.bitwise_right_shift(input, other, out=out)


@to_aikit_arrays_and_back
def bitwise_xor(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.bitwise_xor(input, other, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def ceil(input, *, out=None):
    return aikit.ceil(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16", "complex")}, "torch")
@to_aikit_arrays_and_back
def clamp(input, min=None, max=None, *, out=None):
    aikit.utils.assertions.check_all_or_any_fn(
        min,
        max,
        fn=aikit.exists,
        type="any",
        limit=[1, 2],
        message="at most one of min or max can be None",
    )
    if min is None:
        return aikit.minimum(input, max, out=out)
    if max is None:
        return aikit.maximum(input, min, out=out)
    return aikit.clip(input, min, max, out=out)


@to_aikit_arrays_and_back
def conj_physical(input, *, out=None):
    return aikit.conj(input, out=out)


@with_unsupported_dtypes({"1.12.0 and below": ("float16",)}, "jax")
@to_aikit_arrays_and_back
def copysign(input, other, *, out=None):
    return aikit.copysign(input, other, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def cos(input, *, out=None):
    return aikit.cos(input, out=out)


@to_aikit_arrays_and_back
def cosh(input, *, out=None):
    return aikit.cosh(input, out=out)


@to_aikit_arrays_and_back
def deg2rad(input, *, out=None):
    return aikit.array(input * aikit.pi / 180, out=out)


@to_aikit_arrays_and_back
def div(input, other, *, rounding_mode=None, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    if rounding_mode is not None:
        promoted = input.dtype
        if rounding_mode == "trunc":
            return aikit.astype(aikit.trunc_divide(input, other, out=out), promoted)
        else:
            return aikit.astype(aikit.floor_divide(input, other, out=out), promoted)
    else:
        return aikit.divide(input, other, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16", "complex")}, "torch")
@to_aikit_arrays_and_back
def erf(input, *, out=None):
    return aikit.erf(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16", "complex")}, "torch")
@to_aikit_arrays_and_back
def erfc(input, *, out=None):
    return 1.0 - aikit.erf(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def exp(input, *, out=None):
    return aikit.exp(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def exp2(input, out=None):
    return aikit.exp2(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def expm1(input, out=None):
    return aikit.expm1(input, out=out)


@to_aikit_arrays_and_back
def flipud(input):
    return aikit.flipud(input)


@with_unsupported_dtypes({"1.12.0 and below": ("bfloat16", "float16")}, "jax")
@to_aikit_arrays_and_back
def float_power(input, exponent, *, out=None):
    input, exponent = torch_frontend.promote_types_of_torch_inputs(input, exponent)
    return aikit.float_power(input, exponent, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def floor(input, *, out=None):
    return aikit.floor(input, out=out)


@to_aikit_arrays_and_back
def floor_divide(input, other, *, out=None):
    return aikit.floor_divide(input, other, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def fmod(x1, x2, out=None):
    return aikit.fmod(x1, x2, out=out)


@to_aikit_arrays_and_back
def frac(input, *, out=None):
    return input - aikit.sign(input) * aikit.floor(aikit.abs(input))


@with_unsupported_dtypes({"2.1.2 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def frexp(input, *, out=None):
    return aikit.frexp(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("bfloat16",)}, "torch")
@to_aikit_arrays_and_back
def gradient(input, *, spacing=1, dim=None, edge_order=1):
    return aikit.gradient(input, spacing=spacing, edge_order=edge_order, axis=dim)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def hypot(input, other, *, out=None):
    return aikit.hypot(input, other, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def i0(input, *, out=None):
    return aikit.i0(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("bfloat16",)}, "torch")
@to_aikit_arrays_and_back
def igamma(input, other, *, out=None):
    return aikit.igamma(input, x=other, out=out)


@to_aikit_arrays_and_back
def imag(input):
    return aikit.imag(input)


@with_supported_dtypes({"2.1.2 and below": ("float16", "float32", "float64")}, "torch")
@to_aikit_arrays_and_back
def ldexp(input, other, *, out=None):
    value = aikit.pow(2, other, out=out)
    value = aikit.multiply(input, value, out=out)
    return value


@with_unsupported_dtypes({"2.1.2 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def lerp(input, end, weight, *, out=None):
    return aikit.lerp(input, end, weight, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def lgamma(input, *, out=None):
    return aikit.lgamma(input, out=out)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
def log(input, *, out=None):
    return aikit.log(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def log10(input, *, out=None):
    return aikit.log10(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def log1p(input, *, out=None):
    return aikit.log1p(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def log2(input, *, out=None):
    return aikit.log2(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def logaddexp(x1, x2, out=None):
    return aikit.logaddexp(x1, x2, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def logaddexp2(x1, x2, out=None):
    return aikit.logaddexp2(x1, x2, out=out)


@to_aikit_arrays_and_back
def logical_and(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.logical_and(input, other, out=out)


@to_aikit_arrays_and_back
def logical_not(input, *, out=None):
    return aikit.logical_not(input, out=out)


@to_aikit_arrays_and_back
def logical_or(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.logical_or(input, other, out=out)


@to_aikit_arrays_and_back
def logical_xor(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.logical_xor(input, other, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def logit(input, eps=None, *, out=None):
    return aikit.logit(input, eps=eps, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("bfloat16",)}, "torch")
@to_aikit_arrays_and_back
def masked_fill(input, mask, value):
    return aikit.where(mask, value, input, out=input)


@to_aikit_arrays_and_back
def mul(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.multiply(input, other, out=out)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
def mvlgamma(input, p, *, out=None):
    aikit.assertions.check_greater(
        p, 1, allow_equal=True, message="p has to be greater than or equal to 1"
    )
    c = 0.25 * p * (p - 1) * aikit.log(aikit.pi, out=out)
    b = 0.5 * aikit.arange((1 - p), 1, 1, dtype=input.dtype, device=input.device, out=out)
    return (
        aikit.sum(
            aikit.lgamma(aikit.expand_dims(input, axis=-1) + b, out=out), axis=-1, out=out
        )
        + c
    )


@with_unsupported_dtypes({"2.1.2 and below": ("bfloat16",)}, "tensorflow")
@to_aikit_arrays_and_back
def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None):
    return aikit.nan_to_num(input, nan=nan, posinf=posinf, neginf=neginf, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("bool",)}, "torch")
@to_aikit_arrays_and_back
def negative(input, *, out=None):
    return aikit.negative(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("bfloat16", "float16")}, "torch")
@to_aikit_arrays_and_back
def nextafter(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.nextafter(input, other, out=out)


@to_aikit_arrays_and_back
def positive(input, *, out=None):
    return aikit.positive(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("bool",)}, "torch")
@to_aikit_arrays_and_back
def pow(input, exponent, *, out=None):
    if not aikit.is_array(exponent):
        if (
            any(dtype in str(input.dtype) for dtype in ["int8", "int16"])
            and isinstance(exponent, int)
        ) or ("float16" in str(input.dtype) and isinstance(exponent, float)):
            exponent = aikit.array(exponent, dtype=input.dtype)
        else:
            exponent = torch_frontend.as_tensor(exponent).aikit_array
    input, exponent = torch_frontend.promote_types_of_torch_inputs(input, exponent)
    ret_dtype = input.dtype
    if not aikit.is_int_dtype(exponent) and aikit.is_int_dtype(ret_dtype):
        ret_dtype = exponent.dtype
    ret = aikit.pow(input, exponent)
    if aikit.any(input == 0) and aikit.is_int_dtype(exponent):
        ret = aikit.where(aikit.bitwise_and(input == 0, exponent < 0), 0, ret, out=out)
    return ret.astype(ret_dtype)


@to_aikit_arrays_and_back
def rad2deg(input, *, out=None):
    return aikit.rad2deg(input, out=out)


@to_aikit_arrays_and_back
def real(input):
    return aikit.real(input)


@to_aikit_arrays_and_back
def reciprocal(input, *, out=None):
    return aikit.reciprocal(input)


@to_aikit_arrays_and_back
def remainder(input, other, *, out=None):
    if aikit.is_array(input) and aikit.isscalar(other):
        other = aikit.full(input.shape, other)
    return aikit.remainder(input, other, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("bfloat16",)}, "torch")
@to_aikit_arrays_and_back
def round(input, *, decimals=0, out=None):
    m = aikit.full(input.shape, 10.0**decimals)
    upscale = aikit.multiply(input, m)
    rounded = aikit.round(upscale)
    return aikit.divide(rounded, m, out=out).astype(input.dtype)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def rsqrt(input, *, out=None):
    return aikit.reciprocal(aikit.sqrt(input), out=out)


@to_aikit_arrays_and_back
def sgn(input, *, out=None):
    if aikit.is_complex_dtype(input.dtype):
        input_abs = aikit.abs(input, out=out)
        # TODO wrap this in Where function after solve it's errors
        if input_abs == 0:
            return 0
        else:
            return aikit.divide(input, input_abs, out=out)
    else:
        return aikit.sign(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def sigmoid(input, *, out=None):
    return aikit.sigmoid(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("complex",)}, "torch")
@to_aikit_arrays_and_back
def sign(input, *, out=None):
    return aikit.sign(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("complex",)}, "torch")
@to_aikit_arrays_and_back
def signbit(input, *, out=None):
    return aikit.signbit(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def sin(input, *, out=None):
    return aikit.sin(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def sinc(input, *, out=None):
    return aikit.sinc(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def sinh(input, *, out=None):
    return aikit.sinh(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def sqrt(input, *, out=None):
    return aikit.sqrt(input, out=out)


@to_aikit_arrays_and_back
def square(input, *, out=None):
    return aikit.square(input, out=out)


@to_aikit_arrays_and_back
def subtract(input, other, *, alpha=1, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.subtract(input, other * alpha, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def tan(input, *, out=None):
    return aikit.tan(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def tanh(input, *, out=None):
    return aikit.tanh(input, out=out)


@to_aikit_arrays_and_back
def true_divide(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.divide(input, other, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def trunc(input, *, out=None):
    return aikit.trunc(input, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("bfloat16",)}, "tensorflow")
@to_aikit_arrays_and_back
def xlogy(input, other, *, out=None):
    return aikit.xlogy(input, other, out=out)


absolute = abs
arccos = acos
arccosh = acosh
arcsin = asin
arcsinh = asinh
arctan = atan
arctan2 = atan2
arctanh = atanh
clip = clamp
divide = div
fix = trunc
multiply = mul
sub = subtract
