# local
import aikit
from aikit.func_wrapper import with_supported_dtypes
from aikit.functional.frontends.paddle.func_wrapper import to_aikit_arrays_and_back
from aikit.functional.frontends.paddle.tensor.math import tanh as paddle_tanh


tanh = paddle_tanh


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def celu(
    x,
    /,
    *,
    alpha=1.0,
    name=None,
):
    return aikit.celu(x, alpha=alpha)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def elu(
    x,
    /,
    *,
    alpha=1.0,
    name=None,
):
    return aikit.elu(x, alpha=alpha)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def gelu(x, approximate=False, name=None):
    return aikit.gelu(x, approximate=approximate)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def glu(x, axis=-1, name=None):
    size = x.shape[axis]
    aikit.utils.assertions.check_equal(
        size % 2, 0, message="axis size must be divisible by 2", as_array=False
    )
    a, b = aikit.split(x, num_or_size_splits=2, axis=axis)
    return aikit.multiply(a, aikit.sigmoid(b))


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def gumbel_softmax(x, temperature=1.0, hard=False, axis=-1, name=None):
    gumbel_noice = -aikit.log(-aikit.log(aikit.random_uniform(aikit.shape(x) + 1e-20) + 1e-20))
    gumbel_logits = (x + gumbel_noice) / temperature
    y_soft = aikit.softmax(gumbel_logits, axis=axis)

    if hard:
        y_hard = aikit.one_hot(aikit.argmax(y_soft, axis=axis), aikit.shape(y_soft)[axis])
        return y_hard
    else:
        return y_soft


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def hardshrink(x, threshold=0.5, name=None):
    mask = aikit.logical_or(aikit.greater(x, threshold), aikit.less(x, -threshold))
    return aikit.where(mask, x, 0.0)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def hardsigmoid(x, slope=0.1666667, offset=0.5, name=None):
    ret = aikit.minimum(aikit.maximum(aikit.add(aikit.multiply(x, slope), offset), 0), 1)
    return ret


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def hardswish(x, name=None):
    relu6_val = aikit.relu6(aikit.add(x, 3))
    ret = aikit.multiply(x, aikit.divide(relu6_val, 6))
    return ret


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def hardtanh(
    x,
    /,
    *,
    min=-1.0,
    max=1.0,
    name=None,
):
    less = aikit.where(aikit.less(x, min), min, x)
    ret = aikit.where(aikit.greater(x, max), max, less).astype(x.dtype)
    return ret


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def leaky_relu(x, negative_slope=0.01, name=None):
    return aikit.leaky_relu(x)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def log_sigmoid(x, name=None):
    return -aikit.softplus(-x)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def log_softmax(x, axis=-1, dtype=None, name=None):
    x = aikit.astype(x, dtype) if dtype else x
    ret = aikit.log_softmax(x, axis=axis)
    ret = aikit.astype(ret, dtype) if dtype else ret
    return ret


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def mish(x, name=None):
    return aikit.mish(x)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def prelu(x, weight, data_format="NCHW", name=None):
    return aikit.add(aikit.maximum(0, x), aikit.multiply(weight, aikit.minimum(0, x)))


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def relu(x, name=None):
    return aikit.relu(x)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def relu6(x, name=None):
    return aikit.relu6(x)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def relu_(x, name=None):
    ret = aikit.relu(x)
    aikit.inplace_update(x, ret)
    return x


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def rrelu(
    x,
    /,
    *,
    lower=0.125,
    upper=0.3333333333333333,
    training=False,
    name=None,
):
    if lower < 0 or lower > 1:
        raise ValueError(
            "The lower value must be no less than zero or greater than one. Received:"
            f" {lower}."
        )

    if upper < lower:
        raise ValueError(
            "The upper value must be greater than lower value. Received: lower"
            f" {lower}, upper {upper}."
        )

    if upper > 1:
        raise ValueError(
            f"The upper value must be no greater than one. Received: {upper}."
        )

    is_test = not training
    if is_test:
        add = lower + upper
        ret = add * x * 0.5
        out = aikit.where(x >= 0, x, ret)
        return out.astype(x.dtype)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def selu(
    x,
    /,
    *,
    alpha=1.6732632423543772848170429916717,
    scale=1.0507009873554804934193349852946,
    name=None,
):
    if scale <= 1.0:
        raise ValueError(f"The scale must be greater than 1.0. Received: {scale}.")

    if alpha < 0:
        raise ValueError(f"The alpha must be no less than zero. Received: {alpha}.")

    ret = aikit.where(x > 0, x, alpha * aikit.expm1(x))
    arr = scale * ret
    return aikit.astype(arr, x.dtype)


def silu(x, name=None):
    return aikit.silu(x)


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def softmax_(x, axis=-1, dtype=None, name=None):
    ret = aikit.softmax(x, axis=axis)
    aikit.inplace_update(x, ret)
    return x


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def softplus(x, beta=1, threshold=20, name=None):
    return aikit.softplus(x, beta=beta, threshold=threshold)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def softshrink(
    x,
    /,
    *,
    threshold=0.5,
    name=None,
):
    low = aikit.where(aikit.less(x, -threshold), aikit.add(x, threshold), 0)
    up = aikit.where(aikit.greater(x, threshold), aikit.subtract(x, threshold), 0)
    add = aikit.add(low, up)
    return aikit.astype(add, x.dtype)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def softsign(
    x,
    /,
    *,
    name=None,
):
    return aikit.divide(x, aikit.add(1, aikit.abs(x)))


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def swish(x, name=None):
    return aikit.multiply(x, aikit.sigmoid(x))


@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def tanh_(x, name=None):
    ret = aikit.tanh(x)
    aikit.inplace_update(x, ret)
    return x
    # else:
    # ToDo implement a correctly after fixing aikit.random_uniform
    # a = aikit.random_normal(low=lower, high=upper)
    # ret = aikit.where(x >= 0, x, aikit.multiply(a, x))
    # return ret.astype(x.dtype)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def tanhshrink(
    x,
    /,
    *,
    name=None,
):
    return aikit.subtract(x, aikit.tanh(x))
