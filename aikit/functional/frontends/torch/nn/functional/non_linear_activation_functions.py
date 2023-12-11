# local
import aikit
from aikit.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from aikit.functional.frontends.torch.func_wrapper import to_aikit_arrays_and_back


@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.1.1 and below": (
            "complex",
            "float16",
        )
    },
    "torch",
)
def celu(input, alpha=1.0, inplace=False):
    return aikit.celu(input, alpha=alpha)


def celu_(input, alpha=1.0):
    return celu(input, alpha=alpha, inplace=True)


@to_aikit_arrays_and_back
def elu(input, alpha=1.0, inplace=False):
    prod = aikit.multiply(
        alpha,
        aikit.subtract(aikit.exp(input), 1),
    )
    return aikit.where(aikit.greater(input, 0), input, prod)


def elu_(input, alpha=1.0):
    return elu(input, alpha=alpha, inplace=True)


@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.1.1 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def gelu(input, *, approximate="none"):
    if approximate == "none":
        return aikit.gelu(input, approximate=False)
    elif approximate == "tanh":
        return aikit.gelu(input, approximate=True)
    else:
        raise aikit.utils.exceptions.AikitException(
            "`approximate` argument must be either 'none' or 'tanh'."
        )


@to_aikit_arrays_and_back
def glu(input, dim=-1):
    a, b = aikit.split(input, num_or_size_splits=2, axis=dim)
    return aikit.multiply(a, aikit.sigmoid(b))


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    gumbels = -aikit.empty_like(logits).exponential().log()
    gumbels = (logits + gumbels) / tau
    y_soft = aikit.softmax(gumbels, axis=dim)

    if hard:
        indices = y_soft.max(axis=dim, keepdims=True)[1]
        y_hard = aikit.zeros_like(logits)
        updates = aikit.ones_like(indices)
        y_hard = aikit.scatter_nd(indices, updates, reduction="replace", out=y_hard)

        ret = y_hard - y_soft.stop_gradient(preserve_type=True) + y_soft
    else:
        ret = y_soft

    return ret


@to_aikit_arrays_and_back
def hardshrink(input, lambd=0.5):
    mask = aikit.logical_or(aikit.greater(input, lambd), aikit.less(input, -lambd))
    return aikit.where(mask, input, 0.0)


@to_aikit_arrays_and_back
def hardsigmoid(input, inplace=False):
    return aikit.divide(aikit.minimum(aikit.maximum(aikit.add(input, 3), 0), 6), 6)


@to_aikit_arrays_and_back
def hardswish(input, inplace=False):
    relu6_val = aikit.relu6(aikit.add(input, 3))
    return aikit.multiply(input, aikit.divide(relu6_val, 6))


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    less = aikit.where(aikit.less(input, min_val), min_val, input)
    return aikit.where(aikit.greater(input, max_val), max_val, less).astype(input.dtype)


@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def hardtanh_(input, min_val=-1.0, max_val=1.0):
    return hardtanh(input, min_val=min_val, max_val=max_val, inplace=True)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def leaky_relu(input, negative_slope=0.01, inplace=False):
    return aikit.leaky_relu(input, alpha=negative_slope)


@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def leaky_relu_(input, negative_slope=0.01):
    return leaky_relu(input, negative_slope=negative_slope, inplace=True)


@to_aikit_arrays_and_back
@with_supported_dtypes({"2.1.1 and below": ("float",)}, "torch")
def local_response_norm(input, size, alpha=0.0001, beta=0.75, k=1.0):
    non_batched = input.ndim == 3
    if non_batched:
        input = aikit.expand_dims(input, axis=2)
    ret = aikit.local_response_norm(
        input, size, bias=k, alpha=alpha, beta=beta, average=True, data_format="NCHW"
    )
    if non_batched:
        ret = aikit.squeeze(ret, axis=2)
    return ret


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    if dtype:
        input = aikit.astype(aikit.array(input), aikit.as_aikit_dtype(dtype))
    if dim is None:
        dim = -1
    return aikit.log_softmax(input, axis=dim)


@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.1.1 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def logsigmoid(input):
    return aikit.logsigmoid(input)


@to_aikit_arrays_and_back
def mish(input, inplace=False):
    return aikit.multiply(
        input,
        aikit.tanh(aikit.softplus(input)),
    )


@to_aikit_arrays_and_back
def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
    abs_square = aikit.pow(aikit.abs(input), p)
    sum_ = aikit.sum(abs_square, axis=dim, keepdims=True)
    pnorm_res = aikit.pow(sum_, 1.0 / p)
    max_ = aikit.maximum(pnorm_res, eps)
    return aikit.divide(input, max_, out=out)


@to_aikit_arrays_and_back
def prelu(input, weight):
    return aikit.add(aikit.maximum(0, input), aikit.multiply(weight, aikit.minimum(0, input)))


@to_aikit_arrays_and_back
def relu(input, inplace=False):
    return aikit.relu(input)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("complex",)}, "torch")
def relu6(input, inplace=False):
    return aikit.relu6(input)


@to_aikit_arrays_and_back
def relu_(input):
    return relu(input, inplace=True)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False):
    if training:
        # alpha = aikit.random_uniform(low=lower, high=upper)
        # ToDo implement alpha correctly after fixing aikit.random_uniform
        pass
    else:
        alpha = (lower + upper) / 2
    return aikit.subtract(
        aikit.relu(input), aikit.multiply(alpha, aikit.relu(aikit.negative(input)))
    )


@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def rrelu_(input, lower=1.0 / 8, upper=1.0 / 3, training=False):
    return rrelu(input, lower=lower, upper=upper, training=training, inplace=True)


@to_aikit_arrays_and_back
@with_supported_dtypes({"2.1.1 and below": ("float32", "float64")}, "torch")
def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    return aikit.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=scale,
        mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )


@to_aikit_arrays_and_back
def selu(input, inplace=False):
    return aikit.selu(input)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def sigmoid(input):
    return aikit.sigmoid(input)


@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def silu(input, inplace=False):
    return aikit.multiply(input, aikit.sigmoid(input))


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def softmax(input, dim=None, _stacklevel=3, dtype=None):
    if dtype:
        input = aikit.astype(aikit.array(input), aikit.as_aikit_dtype(dtype))
    return aikit.softmax(input, axis=dim)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def softmin(input, dim=None, dtype=None):
    if dtype:
        input = aikit.astype(aikit.array(input), aikit.as_aikit_dtype(dtype))
    return aikit.softmax(-input, axis=dim)


@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.1.1 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def softplus(input, beta=1, threshold=20):
    return aikit.softplus(input, beta=beta, threshold=threshold)


@to_aikit_arrays_and_back
def softshrink(input, lambd=0.5):
    low = aikit.where(aikit.less(input, -lambd), aikit.add(input, lambd), 0)
    up = aikit.where(aikit.greater(input, lambd), aikit.subtract(input, lambd), 0)
    return aikit.add(low, up)


@to_aikit_arrays_and_back
def softsign(input):
    return aikit.divide(input, aikit.add(1, aikit.abs(input)))


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def tanh(input):
    return aikit.tanh(input)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def tanhshrink(input):
    return aikit.subtract(input, aikit.tanh(input))


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def threshold(input, threshold, value, inplace=False):
    return aikit.where(aikit.greater(input, threshold), input, value)


@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def threshold_(input, threshold, value):
    return threshold(input, threshold, value, inplace=True)
