# global
import aikit
from aikit import (
    with_supported_dtypes,
    with_unsupported_dtypes,
    with_supported_device_and_dtypes,
)
from aikit.functional.frontends.tensorflow import check_tensorflow_casting
from aikit.functional.frontends.tensorflow.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_tf_dtype,
    to_aikit_dtype,
)


# --- Helpers --- #
# --------------- #


def _chbevl(x, coef, N):
    """Evaluates the series.

            N-1
             - '
      y  =   >   coef[i] T (x/2)
             -            i
            i=0

    of Chebyshev polynomials Ti at argument x/2.

    Coefficients are stored in reverse order, i.e. the zero
    order term is last in the array.  Note N is the number of
    coefficients, not the order.

    If coefficients are for the interval a to b, x must
    have been transformed to x -> 2(2x - b - a)/(b-a) before
    entering the routine.  This maps x from (a, b) to (-1, 1),
    over which the Chebyshev polynomials are defined.

    If the coefficients are for the inverted interval, in
    which (a, b) is mapped to (1/b, 1/a), the transformation
    required is x -> 2(2ab/x - b - a)/(b-a).  If b is infinity,
    this becomes x -> 4a/x - 1.
    """
    b0 = coef[0:1]
    b1 = aikit.zeros_like(x)
    i = N - 1
    p = 1

    while i > 0:
        b2 = b1
        b1 = b0
        with aikit.PreciseMode(True):
            b0 = x * b1 - b2 + coef[p : p + 1]
        p += 1
        i -= 1

    return 0.5 * (b0 - b2)


def _get_chebyshev_coefficients_for_exp_i1():
    """Chebyshev coefficients for exp(-x) I1(x) / x in the interval [0,8].

    lim(x->0){ exp(-x) I1(x) / x } = 1/2.

    Returns list of 29 float elements
    -------
    """
    return aikit.array(
        [
            2.77791411276104639959e-18,
            -2.11142121435816608115e-17,
            1.55363195773620046921e-16,
            -1.10559694773538630805e-15,
            7.60068429473540693410e-15,
            -5.04218550472791168711e-14,
            3.22379336594557470981e-13,
            -1.98397439776494371520e-12,
            1.17361862988909016308e-11,
            -6.66348972350202774223e-11,
            3.62559028155211703701e-10,
            -1.88724975172282928790e-9,
            9.38153738649577178388e-9,
            -4.44505912879632808065e-8,
            2.00329475355213526229e-7,
            -8.56872026469545474066e-7,
            3.47025130813767847674e-6,
            -1.32731636560394358279e-5,
            4.78156510755005422638e-5,
            -1.61760815825896745588e-4,
            5.12285956168575772895e-4,
            -1.51357245063125314899e-3,
            4.15642294431288815669e-3,
            -1.05640848946261981558e-2,
            2.47264490306265168283e-2,
            -5.29459812080949914269e-2,
            1.02643658689847095384e-1,
            -1.76416518357834055153e-1,
            2.52587186443633654823e-1,
        ]
    )


def _get_chebyshev_coefficients_for_exp_sqrt_i1():
    """Chebyshev coefficients for exp(-x) sqrt(x) I1(x) in the inverted
    interval [8,infinity].

    lim(x->inf){ exp(-x) sqrt(x) I1(x) } = 1/sqrt(2pi).

    Returns a list of 25 elements containing float
    -------
    """
    return aikit.array(
        [
            7.51729631084210481353e-18,
            4.41434832307170791151e-18,
            -4.65030536848935832153e-17,
            -3.20952592199342395980e-17,
            2.96262899764595013876e-16,
            3.30820231092092828324e-16,
            -1.88035477551078244854e-15,
            -3.81440307243700780478e-15,
            1.04202769841288027642e-14,
            4.27244001671195135429e-14,
            -2.10154184277266431302e-14,
            -4.08355111109219731823e-13,
            -7.19855177624590851209e-13,
            2.03562854414708950722e-12,
            1.41258074366137813316e-11,
            3.25260358301548823856e-11,
            -1.89749581235054123450e-11,
            -5.58974346219658380687e-10,
            -3.83538038596423702205e-9,
            -2.63146884688951950684e-8,
            -2.51223623787020892529e-7,
            -3.88256480887769039346e-6,
            -1.10588938762623716291e-4,
            -9.76109749136146840777e-3,
            7.78576235018280120474e-1,
        ]
    )


# --- Main --- #
# ------------ #


@with_unsupported_dtypes(
    {
        "1.2.0": ("float16", "complex64", "complex128"),
        "1.8.0 and below": ("float16",),
        "2.15.0 and below": ("int8", "int16", "uint8", "uint16", "uint32", "uint64"),
    },
    "tensorflow",
)
@to_aikit_arrays_and_back
def abs(x, name=None):
    dtype = aikit.dtype(x)
    if dtype in ["complex64", "complex128"]:
        return aikit.sqrt(aikit.square(aikit.real(x)) + aikit.square(aikit.imag(x)))
    return aikit.abs(x)


@to_aikit_arrays_and_back
def accumulate_n(inputs, shape=None, tensor_dtype=None, name=None):
    return aikit.sum(inputs, axis=0)


@to_aikit_arrays_and_back
def acos(x, name="acos"):
    return aikit.acos(x)


@to_aikit_arrays_and_back
def acosh(x, name="acosh"):
    return aikit.acosh(x)


@to_aikit_arrays_and_back
def add(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return aikit.add(x, y)


@to_aikit_arrays_and_back
def add_n(inputs, name=None):
    inputs = aikit.array(inputs)
    return aikit.sum(inputs, dtype=inputs.dtype, axis=0)


@to_aikit_arrays_and_back
def angle(input, name=None):
    return aikit.angle(input)


@to_aikit_arrays_and_back
def argmax(input, axis, output_type=None, name=None):
    output_type = to_aikit_dtype(output_type)
    if output_type in ["uint16", "int16", "int32", "int64"]:
        return aikit.astype(aikit.argmax(input, axis=axis), output_type)
    else:
        return aikit.astype(aikit.argmax(input, axis=axis), "int64")


@to_aikit_arrays_and_back
def argmin(input, axis=None, output_type="int64", name=None):
    output_type = to_aikit_dtype(output_type)
    if output_type in ["int32", "int64"]:
        return aikit.astype(aikit.argmin(input, axis=axis), output_type)
    else:
        return aikit.astype(aikit.argmin(input, axis=axis), "int64")


@to_aikit_arrays_and_back
def asin(x, name=None):
    return aikit.asin(x)


@to_aikit_arrays_and_back
def asinh(x, name="asinh"):
    return aikit.asinh(x)


@to_aikit_arrays_and_back
def atan(x, name=None):
    return aikit.atan(x)


@to_aikit_arrays_and_back
def atan2(y, x, name=None):
    return aikit.atan2(y, x)


@to_aikit_arrays_and_back
def atanh(x, name="atanh"):
    return aikit.atanh(x)


@with_supported_dtypes(
    {"2.15.0 and below": ("float16", "float32", "float64")}, "tensorflow"
)
@to_aikit_arrays_and_back
def bessel_i1(x, name=None):
    z = aikit.abs(x)
    result = aikit.zeros_like(z)

    mask1 = z <= 8.0

    if aikit.any(mask1) > 0:
        y = (z[mask1] / aikit.array([2.0])) - aikit.array([2.0])
        result[mask1] = (
            _chbevl(y, _get_chebyshev_coefficients_for_exp_i1(), 29)
            * z[mask1]
            * aikit.exp(z[mask1])
        )

    mask2 = ~mask1
    if aikit.any(mask2) > 0:
        result[mask2] = (
            aikit.exp(z[mask2])
            * _chbevl(
                aikit.array([32.0]) / z[mask2] - aikit.array([2.0]),
                _get_chebyshev_coefficients_for_exp_sqrt_i1(),
                25,
            )
            / aikit.sqrt(z[mask2])
        )

    result[x < 0.0] = -result[x < 0.0]

    return result


@with_supported_dtypes(
    {"2.15.0 and below": ("int32",)},
    "tensorflow",
)
@to_aikit_arrays_and_back
def bincount(
    arr,
    weights=None,
    minlength=None,
    maxlength=None,
    dtype=aikit.int32,
    name=None,
    axis=None,
    binary_output=False,
):
    return aikit.bincount(arr, weights=weights, minlength=minlength)


@to_aikit_arrays_and_back
def ceil(x, name=None):
    return aikit.ceil(x)


@handle_tf_dtype
@to_aikit_arrays_and_back
def confusion_matrix(
    labels, predictions, num_classes=None, weights=None, dtype=aikit.int32, name=None
):
    labels = aikit.astype(
        aikit.squeeze(aikit.array(labels), axis=None), aikit.int64, copy=False
    )
    predictions = aikit.astype(
        aikit.squeeze(aikit.array(predictions), axis=None), aikit.int64, copy=False
    )
    # failsafe for (1,) array will be squeeze to 0-dim
    labels = aikit.expand_dims(labels, axis=-1) if labels.ndim == 0 else labels
    predictions = (
        aikit.expand_dims(predictions, axis=-1) if predictions.ndim == 0 else predictions
    )

    # Sanity check (potential optimization)
    aikit.utils.assertions.check_greater(
        labels, 0, allow_equal=True, message="labels contains negative values"
    )
    aikit.utils.assertions.check_greater(
        predictions, 0, allow_equal=True, message="predictions contains negative values"
    )

    if num_classes is None:
        num_classes = max(aikit.max(labels), aikit.max(predictions)) + 1
    else:
        num_classes_int64 = aikit.astype(aikit.array(num_classes), aikit.int64, copy=False)
        aikit.utils.assertions.check_less(
            labels, num_classes_int64, message="labels out of bound"
        )
        aikit.utils.assertions.check_less(
            predictions, num_classes_int64, message="predictions out of bound"
        )

    if weights is not None:
        weights = aikit.array(weights)
        aikit.utils.assertions.check_equal(
            aikit.shape(predictions),
            aikit.shape(weights),
            message="weights shape do not match predictions",
            as_array=False,
        )
        weights = aikit.astype(weights, dtype, copy=False)

    shape = aikit.stack([num_classes, num_classes])
    indices = aikit.stack([labels, predictions], axis=1)
    values = aikit.ones_like(predictions, dtype=dtype) if weights is None else weights
    return aikit.scatter_nd(indices, values, shape=shape)


@to_aikit_arrays_and_back
def conj(x, name=None):
    return aikit.conj(x)


@to_aikit_arrays_and_back
def cos(x, name=None):
    return aikit.cos(x)


@to_aikit_arrays_and_back
def cosh(x, name=None):
    return aikit.cosh(x)


@handle_tf_dtype
@to_aikit_arrays_and_back
def count_nonzero(input, axis=None, keepdims=None, dtype=aikit.int64, name=None):
    x = aikit.array(input)
    if keepdims is None:
        keepdims = False
    zero = aikit.zeros(aikit.shape(x), dtype=x.dtype)
    return aikit.astype(
        aikit.sum(
            aikit.astype(aikit.not_equal(x, zero), aikit.int64),
            axis=axis,
            keepdims=keepdims,
        ),
        dtype,
        copy=False,
    )


@to_aikit_arrays_and_back
def cumprod(x, axis, exclusive=False, reverse=False, name=None):
    return aikit.astype(
        aikit.cumprod(x, axis=axis, exclusive=exclusive, reverse=reverse), x.dtype
    )


@to_aikit_arrays_and_back
def cumsum(x, axis, exclusive=False, reverse=False, name=None):
    return aikit.astype(
        aikit.cumsum(x, axis=axis, exclusive=exclusive, reverse=reverse), x.dtype
    )


@to_aikit_arrays_and_back
def digamma(x, name=None):
    return aikit.digamma(x)


@to_aikit_arrays_and_back
def divide(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return aikit.divide(x, y)


@to_aikit_arrays_and_back
def divide_no_nan(x, y, name="divide_no_nan"):
    x, y = check_tensorflow_casting(x, y)
    return aikit.where(
        y == 0,
        aikit.array(0.0, dtype=aikit.promote_types(x.dtype, y.dtype)),
        x / y,
    )


@to_aikit_arrays_and_back
def equal(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return aikit.equal(x, y)


@to_aikit_arrays_and_back
def erfcinv(x, name="erfcinv"):
    return 1 / (1 - aikit.erf(x))


@to_aikit_arrays_and_back
def exp(x, name=None):
    return aikit.exp(x)


@to_aikit_arrays_and_back
def expm1(x, name=None):
    return aikit.expm1(x)


@to_aikit_arrays_and_back
def floor(x, name=None):
    return aikit.floor(x)


@to_aikit_arrays_and_back
def floordiv(x, y, name=None):
    return aikit.floor_divide(x, y)


@to_aikit_arrays_and_back
def floormod(x, y, name=None):
    return aikit.remainder(x, y)


@to_aikit_arrays_and_back
def greater(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return aikit.greater(x, y)


@to_aikit_arrays_and_back
def greater_equal(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return aikit.greater_equal(x, y)


@with_supported_device_and_dtypes(
    {
        "2.15.0 and below": {
            "cpu": ("float32", "float64"),
            "gpu": ("bfloat16", "float16", "float32", "float64"),
        }
    },
    "tensorflow",
)
@to_aikit_arrays_and_back
def igamma(a, x, name=None):
    return aikit.igamma(a, x=x)


@with_supported_dtypes(
    {"2.15.0 and below": ("float16", "float32", "float64", "complex64", "complex128")},
    "tensorflow",
)
@to_aikit_arrays_and_back
def imag(input, name=None):
    return aikit.imag(input)


@to_aikit_arrays_and_back
def in_top_k(target, pred, k, name=None):
    top_k = aikit.top_k(target, k)
    return aikit.array([val in top_k.values for val in target])


@with_supported_dtypes(
    {
        "2.15.0 and below": ("int32", "int64"),
    },
    "tensorflow",
)
@to_aikit_arrays_and_back
def invert_permutation(x, name=None):
    return aikit.invert_permutation(x)


@with_supported_dtypes(
    {
        "2.15.0 and below": ("bfloat16", "half", "float32", "float64"),
    },
    "tensorflow",
)
@to_aikit_arrays_and_back
def is_finite(x, name=None):
    return aikit.isfinite(x)


@to_aikit_arrays_and_back
def is_inf(x, name=None):
    return aikit.isinf(x)


@to_aikit_arrays_and_back
def is_nan(x, name=None):
    return aikit.isnan(x)


@to_aikit_arrays_and_back
def is_non_decreasing(x, name="is_non_decreasing"):
    if aikit.array(x).size < 2:
        return aikit.array(True)
    if aikit.array(x).size == 2:
        return aikit.array([x[0] <= x[1]])
    return aikit.all(aikit.less_equal(x, aikit.roll(x, -1)))


@to_aikit_arrays_and_back
def is_strictly_increasing(x, name="is_strictly_increasing"):
    if aikit.array(x).size < 2:
        return aikit.array(True)
    if aikit.array(x).size == 2:
        return aikit.array(x[0] < x[1])
    return aikit.all(aikit.less(x, aikit.roll(x, -1)))


@to_aikit_arrays_and_back
@with_supported_dtypes({"2.15.0 and below": ("float32", "float64")}, "tensorflow")
def l2_normalize(x, axis=None, epsilon=1e-12, name=None):
    square_sum = aikit.sum(aikit.square(x), axis=axis, keepdims=True)
    x_inv_norm = aikit.reciprocal(aikit.sqrt(aikit.maximum(square_sum, epsilon)))
    return aikit.multiply(x, x_inv_norm)


@to_aikit_arrays_and_back
def less(x, y, name="None"):
    x, y = check_tensorflow_casting(x, y)
    return aikit.less(x, y)


@to_aikit_arrays_and_back
def less_equal(x, y, name="LessEqual"):
    x, y = check_tensorflow_casting(x, y)
    return aikit.less_equal(x, y)


# lgamma
@to_aikit_arrays_and_back
@with_supported_dtypes({"2.15.0 and below": ("float32", "float64")}, "tensorflow")
def lgamma(x, name=None):
    return aikit.lgamma(x)


@to_aikit_arrays_and_back
def log(x, name=None):
    return aikit.log(x)


@to_aikit_arrays_and_back
def log1p(x, name=None):
    return aikit.log1p(x)


@to_aikit_arrays_and_back
def log_sigmoid(x, name=None):
    return -aikit.softplus(-x)


@to_aikit_arrays_and_back
def log_softmax(logits, axis=None):
    if axis is None:
        axis = -1
    return aikit.log_softmax(logits, axis=axis)


@to_aikit_arrays_and_back
def logical_and(x, y, name="LogicalAnd"):
    return aikit.logical_and(x, y)


@to_aikit_arrays_and_back
def logical_not(x, name="logical_not"):
    return aikit.logical_not(x)


@to_aikit_arrays_and_back
def logical_or(x, y, name="logical_or"):
    return aikit.logical_or(x, y)


@to_aikit_arrays_and_back
def logical_xor(x, y, name="LogicalXor"):
    return aikit.logical_xor(x, y)


@to_aikit_arrays_and_back
def maximum(x, y, name=None):
    return aikit.maximum(x, y)


@to_aikit_arrays_and_back
def minimum(x, y, name=None):
    return aikit.minimum(x, y)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.6.0 and below": ("bfloat16",)}, "paddle")
def mod(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return aikit.remainder(x, y)


@to_aikit_arrays_and_back
def multiply(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return aikit.multiply(x, y)


@to_aikit_arrays_and_back
def multiply_no_nan(x, y, name="multiply_no_nan"):
    x, y = check_tensorflow_casting(x, y)
    return aikit.where(
        y == 0,
        aikit.array(0.0, dtype=aikit.promote_types(x.dtype, y.dtype)),
        x * y,
    )


@to_aikit_arrays_and_back
def negative(x, name=None):
    return aikit.negative(x)


@to_aikit_arrays_and_back
def nextafter(x1, x2, name=None):
    return aikit.nextafter(x1, x2)


@to_aikit_arrays_and_back
def not_equal(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return aikit.not_equal(x, y)


@to_aikit_arrays_and_back
def polyval(coeffs, x, name=None):
    aikit.utils.assertions.check_isinstance(coeffs, list)
    x = aikit.array(x)
    if len(coeffs) < 1:
        return aikit.zeros_like(x, dtype=x.dtype)
    coeffs = [aikit.array(_) for _ in coeffs]
    p = coeffs[0]
    for c in coeffs[1:]:
        p = c + p * x
    return p


@to_aikit_arrays_and_back
def pow(x, y, name="pow"):
    x, y = check_tensorflow_casting(x, y)
    return aikit.pow(x, y)


@to_aikit_arrays_and_back
def real(input, name=None):
    return aikit.real(input)


@to_aikit_arrays_and_back
def reciprocal(x, name="reciprocal"):
    return aikit.reciprocal(x)


@to_aikit_arrays_and_back
def reciprocal_no_nan(x, name="reciprocal_no_nan"):
    return aikit.where(
        x == 0,
        aikit.array(0.0, dtype=x.dtype),
        aikit.ones_like(x, dtype=x.dtype) / x,
    )


@to_aikit_arrays_and_back
def reduce_all(input_tensor, axis=None, keepdims=False, name="reduce_all"):
    return aikit.all(input_tensor, axis=axis, keepdims=keepdims)


@to_aikit_arrays_and_back
def reduce_any(input_tensor, axis=None, keepdims=False, name="reduce_any"):
    return aikit.any(input_tensor, axis=axis, keepdims=keepdims)


@to_aikit_arrays_and_back
def reduce_euclidean_norm(
    input_tensor, axis=None, keepdims=False, name="reduce_euclidean_norm"
):
    return aikit.vector_norm(
        input_tensor, axis=axis, keepdims=keepdims, ord=2
    )  # ord = '2' is the euclidean norm


@to_aikit_arrays_and_back
def reduce_logsumexp(input_tensor, axis=None, keepdims=False, name="reduce_logsumexp"):
    # stable logsumexp trick
    max_input_tensor = aikit.max(input_tensor, axis=axis, keepdims=False)
    return (
        aikit.log(
            aikit.sum(
                aikit.exp(input_tensor - max_input_tensor),
                axis=axis,
                keepdims=keepdims,
            )
        )
        + max_input_tensor
    ).astype(input_tensor.dtype)


@to_aikit_arrays_and_back
def reduce_max(input_tensor, axis=None, keepdims=False, name="reduce_max"):
    return aikit.max(input_tensor, axis=axis, keepdims=keepdims)


@to_aikit_arrays_and_back
def reduce_mean(input_tensor, axis=None, keepdims=False, name="reduce_mean"):
    if aikit.exists(axis):
        axis = aikit.to_list(axis)
    return aikit.mean(input_tensor, axis=axis, keepdims=keepdims)


@to_aikit_arrays_and_back
def reduce_min(input_tensor, axis=None, keepdims=False, name="reduce_min"):
    return aikit.min(input_tensor, axis=axis, keepdims=keepdims)


@to_aikit_arrays_and_back
def reduce_prod(input_tensor, axis=None, keepdims=False, name="reduce_prod"):
    return aikit.prod(input_tensor, axis=axis, keepdims=keepdims).astype(
        input_tensor.dtype
    )


@to_aikit_arrays_and_back
def reduce_std(input_tensor, axis=None, keepdims=False, name="reduce_std"):
    return aikit.std(input_tensor, axis=axis, keepdims=keepdims)


@to_aikit_arrays_and_back
def reduce_sum(input_tensor, axis=None, keepdims=False, name="reduce_sum"):
    input_tensor = aikit.array(input_tensor)
    return aikit.sum(input_tensor, axis=axis, keepdims=keepdims).astype(
        input_tensor.dtype
    )


@to_aikit_arrays_and_back
def reduce_variance(input_tensor, axis=None, keepdims=False, name="reduce_variance"):
    return aikit.var(input_tensor, axis=axis, keepdims=keepdims)


@with_supported_device_and_dtypes(
    {
        "2.15.0 and below": {
            "cpu": ("float32", "float64"),
            "gpu": ("bfloat16", "float16", "float32", "float64"),
        }
    },
    "tensorflow",
)
@to_aikit_arrays_and_back
def rint(x, name=None):
    return aikit.round(x)


@to_aikit_arrays_and_back
def round(x, name=None):
    return aikit.round(x)


@to_aikit_arrays_and_back
def rsqrt(x, name=None):
    return aikit.reciprocal(aikit.sqrt(x))


@to_aikit_arrays_and_back
def scalar_mul(scalar, x, name="scalar_mul"):
    scalar, x = check_tensorflow_casting(scalar, x)
    return aikit.multiply(x, scalar).astype(x.dtype)


@with_unsupported_dtypes(
    {"2.15.0 and below": ("float16", "bool", "int16", "int8")},
    "tensorflow",
)
@to_aikit_arrays_and_back
def segment_sum(data, segment_ids, name="segment_sum"):
    data = aikit.array(data)
    segment_ids = aikit.array(segment_ids)
    aikit.utils.assertions.check_equal(
        list(segment_ids.shape), [list(data.shape)[0]], as_array=False
    )
    sum_array = aikit.zeros(
        tuple([int(segment_ids[-1] + 1)] + (list(data.shape))[1:]), dtype=data.dtype
    )
    for i in range((segment_ids).shape[0]):
        sum_array[segment_ids[i]] = sum_array[segment_ids[i]] + data[i]
    return sum_array


@to_aikit_arrays_and_back
def sigmoid(x, name=None):
    return aikit.sigmoid(x)


@with_supported_dtypes(
    {
        "2.15.0 and below": (
            "bfloat16",
            "float16",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    },
    "tensorflow",
)
@to_aikit_arrays_and_back
def sin(x, name=None):
    return aikit.sin(x)


@to_aikit_arrays_and_back
def sinh(x, name=None):
    return aikit.sinh(x)


@to_aikit_arrays_and_back
def softmax(logits, axis=None, name=None):
    return aikit.softmax(logits, axis=axis)


@to_aikit_arrays_and_back
def softplus(features, name=None):
    return aikit.softplus(features)


@with_supported_dtypes(
    {"2.15.0 and below": ("bfloat32", "float32", "float64")}, "tensorflow"
)
@to_aikit_arrays_and_back
def softsign(features, name=None):
    return aikit.divide(features, aikit.abs(features) + 1)


@to_aikit_arrays_and_back
def sqrt(x, name=None):
    return aikit.sqrt(x)


@to_aikit_arrays_and_back
def square(x, name=None):
    return aikit.square(x)


@with_supported_dtypes(
    {
        "2.15.0 and below": (
            "bfloat16",
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
            "complex64",
            "complex128",
        )
    },
    "tensorflow",
)
@to_aikit_arrays_and_back
def squared_difference(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    res = aikit.square(aikit.subtract(x, y))
    if isinstance(res, complex):
        res = res.real - res.imag * 1j  # Changing the sign of the imaginary part
        return res
    return res


@to_aikit_arrays_and_back
def subtract(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return aikit.subtract(x, y)


@to_aikit_arrays_and_back
def tan(x, name=None):
    return aikit.tan(x)


@with_supported_dtypes(
    {"2.15.0 and below": ("float16", "float32", "float64", "complex64", "complex128")},
    "tensorflow",
)
@to_aikit_arrays_and_back
def tanh(x, name=None):
    return aikit.tanh(x)


@to_aikit_arrays_and_back
def top_k(input, k=1, sorted=True, name=None):
    return aikit.top_k(input, k, sorted=sorted)


@to_aikit_arrays_and_back
def truediv(x, y, name="truediv"):
    x, y = check_tensorflow_casting(x, y)
    x_dtype = aikit.dtype(x)
    if x_dtype in ["int8", "uint8", "int16", "uint16"]:
        return aikit.divide(aikit.astype(x, aikit.float32), aikit.astype(y, aikit.float32))
    elif x_dtype in ["int32", "uint32", "int64", "uint64"]:
        return aikit.divide(aikit.astype(x, aikit.float64), aikit.astype(y, aikit.float64))
    return aikit.divide(x, y)


@to_aikit_arrays_and_back
def unsorted_segment_mean(
    data, segment_ids, num_segments, name="unsorted_segment_mean"
):
    aikit.utils.assertions.check_equal(
        list(segment_ids.shape), [list(data.shape)[0]], as_array=False
    )
    x = aikit.zeros(tuple([num_segments] + (list(data.shape))[1:]))
    count = aikit.zeros((num_segments,))
    for i in range((segment_ids).shape[0]):
        x[segment_ids[i]] = x[segment_ids[i]] + data[i]
        count[segment_ids[i]] += 1
    for j in range(num_segments):
        x[j] = aikit.divide(x[j], count[j])
    return x


@to_aikit_arrays_and_back
def unsorted_segment_min(data, segment_ids, num_segments, name="unsorted_segment_min"):
    data = aikit.array(data)
    segment_ids = aikit.array(segment_ids)

    aikit.utils.assertions.check_equal(
        list(segment_ids.shape), [list(data.shape)[0]], as_array=False
    )
    min_array = aikit.zeros(
        tuple([num_segments.item()] + (list(data.shape))[1:]), dtype=aikit.int32
    )
    for i in range((segment_ids).shape[0]):
        min_array[segment_ids[i]] = aikit.minimum(min_array[segment_ids[i]], data[i])
    return min_array


@to_aikit_arrays_and_back
def unsorted_segment_sqrt_n(
    data, segment_ids, num_segments, name="unsorted_segement_sqrt_n"
):
    aikit.utils.assertions.check_equal(
        list(segment_ids.shape), [list(data.shape)[0]], as_array=False
    )
    x = aikit.zeros(tuple([num_segments] + (list(data.shape))[1:]))
    count = aikit.zeros((num_segments,))
    for i in range((segment_ids).shape[0]):
        x[segment_ids[i]] = x[segment_ids[i]] + data[i]
        count[segment_ids[i]] += 1
    for j in range(num_segments):
        x[j] = aikit.divide(x[j], aikit.sqrt(count[j]))
    return x


@to_aikit_arrays_and_back
def unsorted_segment_sum(data, segment_ids, num_segments, name="unsorted_segment_sum"):
    data = aikit.array(data)
    segment_ids = aikit.array(segment_ids)
    aikit.utils.assertions.check_equal(
        list(segment_ids.shape), [list(data.shape)[0]], as_array=False
    )
    sum_array = aikit.zeros(
        tuple([num_segments.item()] + (list(data.shape))[1:]), dtype=aikit.int32
    )
    for i in range((segment_ids).shape[0]):
        sum_array[segment_ids[i]] = sum_array[segment_ids[i]] + data[i]
    return sum_array


@with_supported_dtypes(
    {"2.15.0 and below": ("float32", "float64", "complex64", "complex128")},
    "tensorflow",
)
@to_aikit_arrays_and_back
def xdaikit(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    if (x == 0).all():
        return 0.0
    return aikit.divide(x, y)


@to_aikit_arrays_and_back
@with_supported_dtypes({"2.15.0 and below": ("float32", "float64")}, "tensorflow")
def xlog1py(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return x * aikit.log1p(y)


@to_aikit_arrays_and_back
def xlogy(x, y, name=None):
    return aikit.xlogy(x, y)


@to_aikit_arrays_and_back
def zero_fraction(value, name="zero_fraction"):
    zero = aikit.zeros(tuple(value.shape), dtype=aikit.float32)
    x = aikit.array(value, dtype=aikit.float32)
    count_zero = aikit.sum(aikit.equal(x, zero))
    count_nonzero = aikit.sum(aikit.not_equal(x, zero))
    return aikit.divide(count_zero, aikit.add(count_zero, count_nonzero))


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {
        "2.15.0 and below": ("float32", "float64"),
    },
    "tensorflow",
)
def zeta(x, q, name=None):
    return aikit.zeta(x, q)
