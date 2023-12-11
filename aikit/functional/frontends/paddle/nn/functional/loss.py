# local
import aikit
from aikit.func_wrapper import with_supported_dtypes
import aikit.functional.frontends.paddle as paddle
from aikit.utils.exceptions import handle_exceptions
from aikit.functional.frontends.paddle.func_wrapper import (
    inputs_to_aikit_arrays,
    to_aikit_arrays_and_back,
)


# --- Helpers --- #
# --------------- #


# helpers
def _get_reduction_func(reduction):
    if reduction == "none":

        def ret(x):
            return x

    elif reduction == "mean":
        ret = aikit.mean
    elif reduction == "sum":
        ret = aikit.sum
    else:
        raise aikit.utils.exceptions.AikitException(
            f"{reduction} is not a valid value for reduction"
        )
    return ret


def _pairwise_distance(x1, x2, *, p=2.0, eps=1e-06, keepdim=False):
    x1, x2 = paddle.promote_types_of_paddle_inputs(x1, x2)
    x1_dim = len(x1.shape)
    x2_dim = len(x2.shape)
    if x1_dim > x2_dim:
        output_dim = x1_dim
    else:
        output_dim = x2_dim

    return aikit.vector_norm(x1 - x2 + eps, ord=p, axis=output_dim - 1, keepdims=keepdim)


# --- Main --- #
# ------------ #


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def binary_cross_entropy(input, label, weight=None, reduction="mean", name=None):
    reduction = _get_reduction_func(reduction)
    result = aikit.binary_cross_entropy(label, input, epsilon=0.0, reduction="none")
    if weight is not None:
        result = aikit.multiply(weight, result)
    result = reduction(result)
    return result


@with_supported_dtypes(
    {"2.5.2 and below": ("float32",)},
    "paddle",
)
@inputs_to_aikit_arrays
def binary_cross_entropy_with_logits(
    logit,
    label,
    weight=None,
    reduction="mean",
    pos_weight=None,
    name=None,
):
    ret = aikit.binary_cross_entropy(
        label, logit, from_logits=True, reduction="none", pos_weight=pos_weight
    )
    reduction = _get_reduction_func(reduction)
    if weight is not None:
        ret = aikit.multiply(weight, ret)
    ret = reduction(ret).astype(label.dtype)
    return paddle.to_tensor(aikit.atleast_1d(ret))


@handle_exceptions
@to_aikit_arrays_and_back
@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
def cosine_embedding_loss(
    input1, input2, label, margin=0.0, reduction="mean", name=None
):
    if len(label.shape) != 1:
        raise ValueError("1D target tensor expected, multi-target not supported")

    if input1.shape != input2.shape:
        raise ValueError(
            "the shape of input tensor 1 should be equal to input tensor 2, but found"
            " inputs with different sizes"
        )

    if len(input1.shape) > 2:
        raise ValueError(
            "1D target tensor expects 1D or 2D input tensors, but found inputs with"
            " different sizes"
        )

    prod_sum = (input1 * input2).sum(axis=-1)
    mag_square1 = aikit.square(input1).sum(axis=-1) + 1e-11
    mag_square2 = aikit.square(input2).sum(axis=-1) + 1e-11
    denom = aikit.sqrt(mag_square1 * mag_square2)
    cos = prod_sum / denom
    zeros = aikit.zeros_like(cos)
    pos = 1 - cos
    neg = aikit.clip(cos - margin, 0, 0)
    out_pos = aikit.where(label == 1, pos, zeros)
    out_neg = aikit.where(label == -1, neg, zeros)
    out = out_pos + out_neg

    if reduction == "none":
        pass
    if reduction == "mean":
        out = aikit.mean(out)
    elif reduction == "sum":
        out = aikit.sum(out)

    return out


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def dice_loss(input, label, epsilon=0.00001, name=None):
    aikit.assertions.check_true(
        len(input.shape) >= 2,
        message="The rank of input should be greater than or equal to 2.",
    )
    aikit.assertions.check_true(
        len(input.shape) == len(label.shape),
        message=str(
            "The rank of input and label should be equal, "
            "but received input: %d, label: %d." % (len(input.shape), len(label.shape))
        ),
    )
    aikit.assertions.check_true(
        label.shape[-1] == 1,
        message=str(
            "The last dimension of label should be 1, but received %d."
            % label.shape[-1]
        ),
    )
    aikit.assertions.check_true(
        tuple(input.shape[:-1]) == tuple(label.shape[:-1]),
        message="All dimensions should be equal except the last one.",
    )
    aikit.assertions.check_true(
        input.size > 0 and label.size > 0,
        message="Any dimension of input and label cannot be equal to 0.",
    )
    label = aikit.squeeze(label, axis=-1)
    label = aikit.one_hot(label, input.shape[-1])
    reduce_dim = list(range(1, len(input.shape)))
    intersect = aikit.multiply(input, label)
    inse = aikit.sum(intersect, axis=reduce_dim)
    dice_denominator = aikit.sum(input, axis=reduce_dim) + aikit.sum(label, axis=reduce_dim)
    dice_score = 1 - inse * 2 / (dice_denominator + epsilon)
    return aikit.mean(dice_score)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32",)},
    "paddle",
)
@to_aikit_arrays_and_back
def hinge_embedding_loss(input, label, margin=1.0, reduction="mean"):
    if reduction not in ["sum", "mean", "none"]:
        raise ValueError(
            "'reduction' in 'hinge_embedding_loss' should be 'sum', 'mean' or 'none',"
            f" but received {reduction}."
        )

    zero_ = aikit.zeros([1], dtype=input.dtype)
    loss = aikit.where(label == 1.0, input, zero_) + aikit.where(
        label == -1.0, aikit.functional.aikit.activations.relu(margin - input), zero_
    )

    if reduction == "mean":
        return aikit.mean(loss)
    elif reduction == "sum":
        return aikit.sum(loss)
    elif reduction == "none":
        return loss


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def kl_div(
    input,
    label,
    reduction="mean",
    name=None,
):
    if input.shape != label.shape:
        raise ValueError(
            "the shape of input tensor should be equal to target tensor, but found"
            " inputs with different sizes"
        )

    out = label * (aikit.log(label) - input)

    size = aikit.shape(input)
    if len(size) < 1:
        size = [1]

    if reduction == "mean":
        out = aikit.mean(out)
    elif reduction == "batchmean":
        out = aikit.sum(out) / size[0]
    elif reduction == "sum":
        out = aikit.sum(out)
    else:
        pass
    return out.astype(label.dtype)


@inputs_to_aikit_arrays
def l1_loss(
    input,
    label,
    reduction="mean",
    name=None,
):
    sum_diff = aikit.abs(input - label)
    reduction = _get_reduction_func(reduction)
    out = reduction(sum_diff)
    if out.shape == ():
        out = out.expand_dims()
    return paddle.to_tensor(out)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32",)},
    "paddle",
)
@to_aikit_arrays_and_back
def log_loss(input, label, epsilon=0.0001, name=None):
    out = -label * aikit.log(input + epsilon) - (
        (1 - label) * aikit.log(1 - input + epsilon)
    )
    return out


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def margin_ranking_loss(input, other, label, margin=0.0, reduction="mean", name=None):
    reduction = _get_reduction_func(reduction)

    out = aikit.subtract(input, other)
    neg_label = aikit.negative(label)
    out = aikit.multiply(neg_label, out)

    if margin != 0.0:
        margin_var = aikit.full([1], margin, dtype=out.dtype)
        out = aikit.add(out, margin_var)

    out = aikit.where(out < 0, 0, out)
    out = reduction(out).astype(input.dtype)
    out = aikit.atleast_1d(out)

    return out


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@inputs_to_aikit_arrays
def mse_loss(input, label, reduction="mean", name=None):
    reduction = _get_reduction_func(reduction)
    ret = aikit.square(input - label)
    ret = reduction(ret)

    return paddle.to_tensor(ret)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def multi_label_soft_margin_loss(
    input, label, weight=None, reduction="mean", name=None
):
    reduction = _get_reduction_func(reduction)
    loss = -(
        label * aikit.log(aikit.sigmoid(input))
        + (1 - label) * aikit.log(1 - aikit.sigmoid(input))
    )

    if weight is not None:
        loss = aikit.multiply(weight, loss)
    loss = aikit.mean(loss, axis=-1)
    ret = reduction(loss).astype(input.dtype)
    return ret


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def nll_loss(
    input,
    label,
    weight=None,
    ignore_index=-100,
    reduction="mean",
):
    """Refer
    https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss for
    more on NLL(Negative log likelihood) Loss."""
    if weight is None:
        weight = aikit.ones(aikit.shape(input[0]))
    input = aikit.log(input)
    loss = aikit.zeros(aikit.shape(label))
    den = 0
    for i in range(0, aikit.shape(loss)[0]):
        den = den + weight[label[i]]
        loss[i] = -weight[label[i]] * input[i][label[i]]
    output = 0.0
    if reduction == "sum":
        output = aikit.sum(loss)
        if ignore_index >= 0 and ignore_index < aikit.shape(input)[1]:
            output = output - loss[ignore_index]
        return output
    num = aikit.sum(loss)
    output = num / den
    if ignore_index >= 0 and ignore_index < aikit.shape(input)[1]:
        output = output - loss[ignore_index] / den
    return output


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def sigmoid_focal_loss(
    logit,
    label,
    normalizer=None,
    alpha=0.25,
    gamma=2.0,
    reduction="sum",
    name=None,
):
    if reduction not in ["sum", "mean", "none"]:
        raise ValueError(
            "The value of 'reduction' in sigmoid_focal_loss should be 'sum', 'mean' or"
            f" 'none', but received {reduction}, which is not allowed."
        )

    if normalizer is not None and normalizer.ndim > 1:
        raise ValueError(
            "Expected zero or one dimension of normalizer in sigmoid_focal_loss but"
            f" got {normalizer.ndim}."
        )

    if not isinstance(logit, aikit.Array):
        logit = aikit.array(logit)

    if not isinstance(label, aikit.Array):
        label = aikit.array(label)

    pred = aikit.sigmoid(logit)
    loss = -(
        label * alpha * aikit.pow((1 - pred), gamma) * aikit.log(pred)
        + (1 - label) * (1 - alpha) * aikit.pow(pred, gamma) * aikit.log(1 - pred)
    )

    if normalizer is not None:
        loss /= normalizer

    if reduction == "sum":
        return aikit.sum(loss)
    elif reduction == "mean":
        return aikit.mean(loss)

    return loss


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def smooth_l1_loss(
    input,
    label,
    reduction="mean",
    delta=1.0,
    name=None,
):
    sum_diff = aikit.abs(input - label).astype(label.dtype)
    condition = sum_diff <= delta
    out = aikit.where(
        condition,
        0.5 * aikit.pow(aikit.abs(input - label), 2).astype(label.dtype),
        (delta * aikit.abs(aikit.abs(input - label))).astype(label.dtype)
        - (0.5 * aikit.pow(delta, 2)).astype(label.dtype),
    )
    if reduction == "none":
        pass
    elif reduction == "mean":
        out = aikit.mean(out)
    elif reduction == "sum":
        out = aikit.sum(out)
    return out.astype(label.dtype)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64")},
    "paddle",
)
@inputs_to_aikit_arrays
def softmax_with_cross_entropy(
    logits,
    label,
    soft_label=False,
    ignore_index=-100,
    numeric_stable_mode=True,
    return_softmax=False,
    axis=-1,
):
    input_dims = len(list(logits.shape))
    if input_dims == 0:
        raise ValueError("The dimension of input should be larger than zero!")
    label_dims = len(list(label.shape))
    if input_dims - 1 != label_dims and input_dims != label_dims:
        raise ValueError(
            "Expected nput_dims - 1 = label_dims or input_dims == label_dims          "
            f"   (got nput_dims{input_dims}, label_dims{label_dims})"
        )
    logits = aikit.array(logits)
    label = aikit.array(label)
    if input_dims - 1 == label_dims:
        label = aikit.expand_dims(label, axis=axis)
    if numeric_stable_mode:
        max_logits = aikit.max(logits, axis=axis, keepdims=True)
        log_max_sum_logits = aikit.log(
            aikit.sum(aikit.exp(aikit.subtract(logits, max_logits)), axis=axis, keepdims=True)
        )
        softmax = aikit.exp(
            aikit.subtract(aikit.subtract(logits, max_logits), log_max_sum_logits)
        )
    else:
        softmax = aikit.softmax(logits, axis=axis)

    if soft_label:
        loss = -aikit.sum(
            aikit.multiply(
                label,
                aikit.subtract(
                    logits, aikit.log(aikit.sum(aikit.exp(logits), axis=axis, keepdims=True))
                ),
            ),
            axis=axis,
            keepdims=True,
        )
    else:
        mask = aikit.not_equal(label.astype("float64"), float(ignore_index))
        loss = aikit.add(
            -aikit.take_along_axis(logits, label, axis),
            aikit.log(aikit.sum(aikit.exp(logits), axis=axis, keepdims=True)),
        )
        loss = aikit.multiply(loss, mask)
    if return_softmax:
        return paddle.to_tensor(loss), paddle.to_tensor(softmax)
    return paddle.to_tensor(loss)


@with_supported_dtypes({"2.5.2 and below": ("float32",)}, "paddle")
@to_aikit_arrays_and_back
def square_error_cost(input, label):
    return aikit.square(aikit.subtract(input, label))


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def triplet_margin_loss(
    input,
    positive,
    negative,
    margin=1.0,
    p=2.0,
    eps=1e-06,
    swap=False,
    reduction="mean",
):
    reduction = _get_reduction_func(reduction)

    a_dim = input.ndim
    p_dim = positive.ndim
    n_dim = negative.ndim

    aikit.assertions.check_true(
        a_dim == p_dim and p_dim == n_dim,
        lambda: (
            "The input, positive, and negative tensors are expected to have "
            f"the same number of dimensions, but got: input {a_dim}D, "
            f"positive {p_dim}D, and negative {n_dim}D inputs"
        ),
    )

    dist_positive = _pairwise_distance(input, positive, p=p, eps=eps)
    dist_negative = _pairwise_distance(input, negative, p=p, eps=eps)
    if swap:
        dist_swap = _pairwise_distance(positive, negative, p=p, eps=eps)
        dist_negative = aikit.minimum(dist_negative, dist_swap)
    loss = aikit.maximum(
        dist_positive - dist_negative + aikit.array(margin), aikit.array(0.0)
    )

    loss = reduction(loss).astype(input.dtype)
    return loss
