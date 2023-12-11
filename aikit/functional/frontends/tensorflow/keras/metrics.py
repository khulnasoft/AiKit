import aikit
from aikit.functional.frontends.tensorflow.func_wrapper import to_aikit_arrays_and_back


# --- Helpers --- #
# --------------- #


def _binary_matches(y_true, y_pred, threshold=0.5):
    threshold = aikit.astype(aikit.array(threshold), y_pred.dtype)
    y_pred = aikit.astype(aikit.greater(y_pred, threshold), y_pred.dtype)
    return aikit.astype(
        aikit.equal(y_true, y_pred), aikit.default_float_dtype(as_native=True)
    )


def _cond_convert_labels(y_true):
    are_zeros = aikit.equal(y_true, 0.0)
    are_ones = aikit.equal(y_true, 1.0)
    is_binary = aikit.all(aikit.logical_or(are_zeros, are_ones))
    # convert [0, 1] labels to [-1, 1]
    if is_binary:
        return 2.0 * y_true - 1
    return y_true


@to_aikit_arrays_and_back
def _sparse_categorical_matches(y_true, y_pred):
    reshape = False
    y_true = aikit.array(y_true)
    y_pred = aikit.array(y_pred)
    y_true_org_shape = aikit.shape(y_true)
    y_true_rank = y_true.ndim
    y_pred_rank = y_pred.ndim
    # y_true shape to (num_samples,)
    if (
        (y_true_rank is not None)
        and (y_pred_rank is not None)
        and (len(aikit.shape(y_true)) == len(aikit.shape(y_pred)))
    ):
        y_true = aikit.squeeze(y_true, axis=-1)
        reshape = True
    y_pred = aikit.argmax(y_pred, axis=-1)
    # cast prediction type to be the same as ground truth
    y_pred = aikit.astype(y_pred, y_true.dtype, copy=False)
    matches = aikit.astype(aikit.equal(y_true, y_pred), aikit.float32)
    if reshape:
        matches = aikit.reshape(matches, shape=y_true_org_shape)
    return matches


@to_aikit_arrays_and_back
def _sparse_top_k_categorical_matches(y_true, y_pred, k=5):
    # Temporary composition
    def _in_top_k(targets, predictions, topk):
        # Sanity check
        aikit.utils.assertions.check_equal(
            targets.ndim,
            1,
            message="targets must be 1-dimensional",
            as_array=False,
        )
        aikit.utils.assertions.check_equal(
            predictions.ndim,
            2,
            message="predictions must be 2-dimensional",
            as_array=False,
        )
        targets_batch = aikit.shape(targets)[0]
        pred_batch = aikit.shape(predictions)[0]
        aikit.utils.assertions.check_equal(
            targets_batch,
            pred_batch,
            message=(
                f"first dim of predictions: {pred_batch} must match targets length:"
                f" {targets_batch}"
            ),
            as_array=False,
        )

        # return array of top k values from the input
        def _top_k(input, topk):
            x = aikit.array(input)
            sort = aikit.argsort(x, descending=True)
            topk = min(x.shape[-1], topk)

            # Safety check for equal values
            result = []
            for ind, li in enumerate(sort):
                temp = [x[ind, _] for _ in li[:topk]]
                result.append(temp)

            return aikit.array(result)

        top_k = _top_k(predictions, topk)

        labels = aikit.shape(predictions)[1]
        # float comparison?
        return aikit.array(
            [
                (
                    0 <= res < labels
                    and aikit.min(top_k[ind] - predictions[ind, res]) <= 1e-9
                )
                for ind, res in enumerate(targets)
            ]
        )

    reshape = False
    y_true = aikit.array(y_true)
    y_pred = aikit.array(y_pred)
    y_true_org_shape = aikit.shape(y_true)
    y_true_rank = y_true.ndim
    y_pred_rank = y_pred.ndim

    # y_pred shape to (batch_size, num_samples), y_true shape to (num_samples,)
    if (y_true_rank is not None) and (y_pred_rank is not None):
        if y_pred_rank > 2:
            y_pred = aikit.reshape(y_pred, shape=[-1, y_pred.shape[-1]])
        if y_true_rank > 1:
            reshape = True
            y_true = aikit.reshape(y_true, shape=[-1])

    matches = aikit.astype(
        _in_top_k(targets=aikit.astype(y_true, aikit.int32), predictions=y_pred, topk=k),
        aikit.float32,
    )

    # return to original shape
    if reshape:
        return aikit.reshape(matches, shape=y_true_org_shape)
    return matches


# --- Main --- #
# ------------ #


@to_aikit_arrays_and_back
def binary_accuracy(y_true, y_pred, threshold=0.5):
    return aikit.mean(_binary_matches(y_true, y_pred, threshold), axis=-1)


@to_aikit_arrays_and_back
def binary_crossentropy(
    y_true, y_pred, from_logits: bool = False, label_smoothing: float = 0.0
):
    y_pred = aikit.asarray(y_pred)
    y_true = aikit.asarray(y_true, dtype=y_pred.dtype)
    label_smoothing = aikit.asarray(label_smoothing, dtype=y_pred.dtype)
    y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if from_logits:
        zeros = aikit.zeros_like(y_pred, dtype=y_pred.dtype)
        cond = y_pred >= zeros
        relu_logits = aikit.where(cond, y_pred, zeros)
        neg_abs_logits = aikit.where(cond, -y_pred, y_pred)
        bce = aikit.add(relu_logits - y_pred * y_true, aikit.log1p(aikit.exp(neg_abs_logits)))
    else:
        epsilon_ = 1e-7
        y_pred = aikit.clip(y_pred, epsilon_, 1.0 - epsilon_)
        bce = y_true * aikit.log(y_pred + epsilon_)
        bce += (1 - y_true) * aikit.log(1 - y_pred + epsilon_)
        bce = -bce
    return aikit.mean(bce, axis=-1).astype(y_pred.dtype)


@to_aikit_arrays_and_back
def binary_focal_crossentropy(
    y_true, y_pred, gamma=2.0, from_logits=False, label_smoothing=0.0, axis=-1
):
    y_pred = aikit.asarray(y_pred)
    y_true = aikit.asarray(y_true, dtype=y_pred.dtype)
    label_smoothing = aikit.asarray(label_smoothing, dtype=y_pred.dtype)
    gamma = aikit.asarray(gamma, dtype=y_pred.dtype)

    if label_smoothing > 0.0:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if from_logits:
        sigmoidal = aikit.sigmoid(y_pred)
    else:
        sigmoidal = y_pred

    p_t = (y_true * sigmoidal) + ((1 - y_true) * (1 - sigmoidal))
    focal_factor = aikit.pow(1.0 - p_t, gamma)

    if from_logits:
        zeros = aikit.zeros_like(y_pred, dtype=y_pred.dtype)
        cond = y_pred >= zeros
        relu_logits = aikit.where(cond, y_pred, zeros)
        neg_abs_logits = aikit.where(cond, -y_pred, y_pred)
        bce = aikit.add(relu_logits - y_pred * y_true, aikit.log1p(aikit.exp(neg_abs_logits)))
    else:
        epsilon_ = 1e-7
        y_pred = aikit.clip(y_pred, epsilon_, 1.0 - epsilon_)
        bce = y_true * aikit.log(y_pred + epsilon_)
        bce += (1 - y_true) * aikit.log(1 - y_pred + epsilon_)
        bce = -bce
    bfce = focal_factor * bce
    return aikit.mean(bfce, axis=aikit.to_scalar(axis))


@to_aikit_arrays_and_back
def categorical_accuracy(y_true, y_pred):
    return _sparse_categorical_matches(aikit.argmax(y_true, axis=-1), y_pred)


@to_aikit_arrays_and_back
def categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0.0):
    if from_logits:
        y_pred = aikit.softmax(y_pred)
    return aikit.mean(aikit.categorical_cross_entropy(y_true, y_pred, label_smoothing))


@to_aikit_arrays_and_back
def cosine_similarity(y_true, y_pred):
    y_pred = aikit.asarray(y_pred)
    y_true = aikit.asarray(y_true)

    if len(y_pred.shape) == len(y_pred.shape) and len(y_true.shape) == 2:
        numerator = aikit.sum(y_true * y_pred, axis=1)
    else:
        numerator = aikit.vecdot(y_true, y_pred)
    denominator = aikit.matrix_norm(y_true) * aikit.matrix_norm(y_pred)
    return numerator / denominator


@to_aikit_arrays_and_back
def hinge(y_true, y_pred):
    y_true = aikit.astype(aikit.array(y_true), y_pred.dtype, copy=False)
    y_true = _cond_convert_labels(y_true)
    return aikit.mean(aikit.maximum(1.0 - y_true * y_pred, 0.0), axis=-1)


@to_aikit_arrays_and_back
def kl_divergence(y_true, y_pred):
    # clip to range but avoid div-0
    y_true = aikit.clip(y_true, 1e-7, 1)
    y_pred = aikit.clip(y_pred, 1e-7, 1)
    return aikit.sum(y_true * aikit.log(y_true / y_pred), axis=-1).astype(y_true.dtype)


@to_aikit_arrays_and_back
def log_cosh(y_true, y_pred):
    y_true = aikit.astype(y_true, y_pred.dtype)
    diff = y_pred - y_true
    log_val = aikit.astype(aikit.log(2.0), diff.dtype)
    return aikit.mean(diff + aikit.softplus(-2.0 * diff) - log_val, axis=-1)


@to_aikit_arrays_and_back
def mean_absolute_error(y_true, y_pred):
    return aikit.mean(aikit.abs(y_true - y_pred), axis=-1)


@to_aikit_arrays_and_back
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = aikit.astype(y_true, y_pred.dtype, copy=False)

    diff = aikit.abs((y_true - y_pred) / aikit.maximum(aikit.abs(y_true), 1e-7))
    return 100.0 * aikit.mean(diff, axis=-1)


@to_aikit_arrays_and_back
def mean_squared_error(y_true, y_pred):
    return aikit.mean(aikit.square(aikit.subtract(y_true, y_pred)), axis=-1)


@to_aikit_arrays_and_back
def mean_squared_logarithmic_error(y_true, y_pred):
    y_true = aikit.astype(y_true, y_pred.dtype)
    first_log = aikit.log(aikit.maximum(y_pred, 1e-7) + 1.0)
    second_log = aikit.log(aikit.maximum(y_true, 1e-7) + 1.0)
    return aikit.mean(aikit.square(aikit.subtract(first_log, second_log)), axis=-1)


@to_aikit_arrays_and_back
def poisson(y_true, y_pred):
    y_true = aikit.astype(y_true, y_pred.dtype, copy=False)
    return aikit.mean(y_pred - y_true * aikit.log(y_pred + 1e-7), axis=-1)


@to_aikit_arrays_and_back
def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1):
    if from_logits:
        y_pred = aikit.softmax(y_pred)
    return aikit.sparse_cross_entropy(y_true, y_pred, axis=axis)


@to_aikit_arrays_and_back
def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    return _sparse_top_k_categorical_matches(y_true, y_pred, k)


@to_aikit_arrays_and_back
def squared_hinge(y_true, y_pred):
    y_true = aikit.astype(aikit.array(y_true), y_pred.dtype)
    y_true = _cond_convert_labels(y_true)
    return aikit.mean(aikit.square(aikit.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1)


kld = kl_divergence
kullback_leibler_divergence = kl_divergence
logcosh = log_cosh
mae = mean_absolute_error
mape = mean_absolute_percentage_error
mse = mean_squared_error
msle = mean_squared_logarithmic_error
