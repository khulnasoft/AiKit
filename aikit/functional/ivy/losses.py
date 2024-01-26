"""Collection of Aikit loss functions."""

# local
import aikit
from typing import Optional, Union
from aikit.func_wrapper import (
    handle_array_function,
    handle_nestable,
    handle_array_like_without_promotion,
    inputs_to_aikit_arrays,
)
from aikit.utils.exceptions import handle_exceptions


# Helpers #
# ------- #


def _reduce_loss(red, loss, axis, out):
    if red == "sum":
        return aikit.negative(aikit.sum(loss, axis=axis), out=out)
    elif red == "mean":
        return aikit.negative(aikit.mean(loss, axis=axis), out=out)
    else:
        return aikit.negative(loss, out=out)


# Extra #
# ------#


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_aikit_arrays
@handle_array_function
def cross_entropy(
    true: Union[aikit.Array, aikit.NativeArray],
    pred: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    axis: int = -1,
    epsilon: float = 1e-7,
    reduction: str = "mean",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Compute cross-entropy between predicted and true discrete distributions.

    Parameters
    ----------
    true
        input array containing true labels.
    pred
        input array containing the predicted labels.
    axis
        the axis along which to compute the cross-entropy. If axis is ``-1``,
        the cross-entropy will be computed along the last dimension. Default: ``-1``.
    epsilon
        a float in [0.0, 1.0] specifying the amount of smoothing when calculating
        the loss. If epsilon is ``0``, no smoothing will be applied. Default: ``1e-7``.
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        The cross-entropy loss between the given distributions

    Examples
    --------
    >>> x = aikit.array([0, 0, 1, 0])
    >>> y = aikit.array([0.25, 0.25, 0.25, 0.25])
    >>> print(aikit.cross_entropy(x, y))
    aikit.array(1.3862944)

    >>> z = aikit.array([0.1, 0.1, 0.7, 0.1])
    >>> print(aikit.cross_entropy(x, z))
    aikit.array(0.35667497)
    """
    aikit.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])
    pred = aikit.clip(pred, epsilon, 1 - epsilon)
    log_pred = aikit.log(pred)
    return _reduce_loss(reduction, log_pred * true, axis, out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_aikit_arrays
@handle_array_function
def binary_cross_entropy(
    true: Union[aikit.Array, aikit.NativeArray],
    pred: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    from_logits: bool = False,
    epsilon: float = 0.0,
    reduction: str = "mean",
    pos_weight: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
    axis: Optional[int] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Compute the binary cross entropy loss.

    Parameters
    ----------
    true
        input array containing true labels.
    pred
        input array containing Predicted labels.
    from_logits
        Whether `pred` is expected to be a logits tensor. By
        default, we assume that `pred` encodes a probability distribution.
    epsilon
        a float in [0.0, 1.0] specifying the amount of smoothing when calculating the
        loss. If epsilon is ``0``, no smoothing will be applied. Default: ``0``.
    reduction
        ``'none'``: No reduction will be applied to the output.
        ``'mean'``: The output will be averaged.
        ``'sum'``: The output will be summed. Default: ``'none'``.
    pos_weight
        a weight for positive examples. Must be an array with length equal to the number
        of classes.
    axis
        Axis along which to compute crossentropy.
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        The binary cross entropy between the given distributions.


    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([0, 1, 0, 0])
    >>> y = aikit.array([0.2, 0.8, 0.3, 0.8])
    >>> z = aikit.binary_cross_entropy(x, y)
    >>> print(z)
    aikit.array([0.223,0.223,0.357,1.61])

    >>> x = aikit.array([[0, 1, 1, 0]])
    >>> y = aikit.array([[2.6, 6.2, 3.7, 5.3]])
    >>> z = aikit.binary_cross_entropy(x, y, reduction='mean')
    >>> print(z)
    aikit.array(7.6666193)

    >>> x = aikit.array([[0, 1, 1, 0]])
    >>> y = aikit.array([[2.6, 6.2, 3.7, 5.3]])
    >>> pos_weight = aikit.array([1, 2, 3, 4])
    >>> z = aikit.binary_cross_entropy(x, y, pos_weight=pos_weight, from_logits=True)
    aikit.array([[2.67164493e+00, 4.05471958e-03, 7.32684899e-02, 5.30496836e+00]])

    >>> x = aikit.array([[0, 1, 1, 0]])
    >>> y = aikit.array([[2.6, 6.2, 3.7, 5.3]])
    >>> pos_weight = aikit.array([1, 2, 3, 4])
    >>> z = aikit.binary_cross_entropy(x, y, pos_weight=pos_weight, from_logits=True, reduction='sum', axis=1)
    aikit.array([8.05393649])

    >>> x = aikit.array([[0, 1, 1, 0]])
    >>> y = aikit.array([[2.6, 6.2, 3.7, 5.3]])
    >>> z = aikit.binary_cross_entropy(x, y, reduction='none', epsilon=0.5)
    aikit.array([[11.49992943,  3.83330965,  3.83330965, 11.49992943]])

    >>> x = aikit.array([[0, 1, 0, 0]])
    >>> y = aikit.array([[0.6, 0.2, 0.7, 0.3]])
    >>> z = aikit.binary_cross_entropy(x, y, epsilon=1e-3)
    >>> print(z)
    aikit.array([[0.916,1.61,1.2,0.357]])

    With :class:`aikit.NativeArray` input:

    >>> x = aikit.native_array([0, 1, 0, 1])
    >>> y = aikit.native_array([0.2, 0.7, 0.2, 0.6])
    >>> z = aikit.binary_cross_entropy(x, y)
    >>> print(z)
    aikit.array([0.223,0.357,0.223,0.511])

    With a mix of :class:`aikit.Array` and :class:`aikit.NativeArray` inputs:

    >>> x = aikit.array([0, 0, 1, 1])
    >>> y = aikit.native_array([0.1, 0.2, 0.8, 0.6])
    >>> z = aikit.binary_cross_entropy(x, y)
    >>> print(z)
    aikit.array([0.105,0.223,0.223,0.511])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([1, 0, 0]),b=aikit.array([0, 0, 1]))
    >>> y = aikit.Container(a=aikit.array([0.6, 0.2, 0.3]),b=aikit.array([0.8, 0.2, 0.2]))
    >>> z = aikit.binary_cross_entropy(x, y)
    >>> print(z)
    {a:aikit.array([0.511,0.223,0.357]),b:aikit.array([1.61,0.223,1.61])}

    With a mix of :class:`aikit.Array` and :class:`aikit.Container` inputs:

    >>> x = aikit.array([1 , 1, 0])
    >>> y = aikit.Container(a=aikit.array([0.7, 0.8, 0.2]))
    >>> z = aikit.binary_cross_entropy(x, y)
    >>> print(z)
    {
       a: aikit.array([0.357, 0.223, 0.223])
    }

    Instance Method Examples
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Using :class:`aikit.Array` instance method:

    >>> x = aikit.array([1, 0, 0, 0])
    >>> y = aikit.array([0.8, 0.2, 0.2, 0.2])
    >>> z = aikit.binary_cross_entropy(x, y)
    >>> print(z)
    aikit.array([0.223, 0.223, 0.223, 0.223])
    """  # noqa: E501
    aikit.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])

    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon should be a float in [0, 1]")

    if not from_logits and pos_weight is not None:
        raise ValueError("pos_weight is only allowed when from_logits is set to True")

    true = true.astype(pred.dtype)

    epsilon = aikit.asarray(epsilon, dtype=pred.dtype)

    true = true * (1.0 - epsilon) + 0.5 * epsilon

    if from_logits:
        if pos_weight is not None:
            num_classes = pred.shape[0] if len(pred.shape) == 1 else pred.shape[1]
            if pos_weight.shape[0] != num_classes:
                raise ValueError(
                    "pos_weight must have the same size as the number of classes in"
                    " pred at non-singleton dimension 1"
                )
            epsilon_ = 1e-7
            pred = aikit.sigmoid(pred)
            pred = aikit.clip(pred, epsilon_, 1 - epsilon_)
            loss = -(
                true * -aikit.log(pred) * pos_weight + (1 - true) * -aikit.log(1 - pred)
            )
        else:
            zeros = aikit.zeros_like(pred, dtype=pred.dtype)
            cond = pred >= zeros
            relu_logits = aikit.where(cond, pred, zeros)
            neg_abs_logits = aikit.where(cond, -pred, pred)
            loss = (
                aikit.add(relu_logits - pred * true, aikit.log1p(aikit.exp(neg_abs_logits)))
                * -1
            )
    else:
        epsilon_ = 1e-7
        pred = aikit.clip(pred, epsilon_, 1 - epsilon_)
        loss = true * aikit.log(pred + epsilon_) + (1 - true) * aikit.log(
            1 - pred + epsilon_
        )

    return _reduce_loss(reduction, loss, axis, out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_aikit_arrays
@handle_array_function
def sparse_cross_entropy(
    true: Union[aikit.Array, aikit.NativeArray],
    pred: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    axis: int = -1,
    epsilon: float = 1e-7,
    reduction: str = "mean",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Compute sparse cross entropy between logits and labels.

    Parameters
    ----------
    true
     input array containing the true labels as logits.
    pred
     input array containing the predicted labels as logits.
    axis
     the axis along which to compute the cross-entropy. If axis is ``-1``, the
     cross-entropy will be computed along the last dimension. Default: ``-1``.
    epsilon
     a float in [0.0, 1.0] specifying the amount of smoothing when calculating the
     loss. If epsilon is ``0``, no smoothing will be applied. Default: ``1e-7``.
    out
     optional output array, for writing the result to. It must have a shape
     that the inputs broadcast to.

    Returns
    -------
    ret
        The sparse cross-entropy loss between the given distributions

    Examples
    --------
    With :class:`aikit.Array` input:

    >> x = aikit.array([2])
    >> y = aikit.array([0.1, 0.1, 0.7, 0.1])
    >> print(aikit.sparse_cross_entropy(x, y))
    aikit.array([0.35667494])

    >>> x = aikit.array([3])
    >>> y = aikit.array([0.1, 0.1, 0.7, 0.1])
    >>> print(aikit.cross_entropy(x, y))
    aikit.array(21.79329094)

    >>> x = aikit.array([2,3])
    >>> y = aikit.array([0.1, 0.1])
    >>> print(aikit.cross_entropy(x, y))
    aikit.array(11.512926)

    With :class:`aikit.NativeArray` input:

    >>> x = aikit.native_array([4])
    >>> y = aikit.native_array([0.1, 0.2, 0.1, 0.1, 0.5])
    >>> print(aikit.sparse_cross_entropy(x, y))
    aikit.array([0.693])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([4]))
    >>> y = aikit.Container(a=aikit.array([0.1, 0.2, 0.1, 0.1, 0.5]))
    >>> print(aikit.sparse_cross_entropy(x, y))
    {
        a: aikit.array([0.693])
    }

    With a mix of :class:`aikit.Array` and :class:`aikit.NativeArray` inputs:

    >>> x = aikit.array([0])
    >>> y = aikit.native_array([0.1, 0.2, 0.6, 0.1])
    >>> print(aikit.sparse_cross_entropy(x,y))
    aikit.array([2.3])

    With a mix of :class:`aikit.Array` and :class:`aikit.Container` inputs:

    >>> x = aikit.array([0])
    >>> y = aikit.Container(a=aikit.array([0.1, 0.2, 0.6, 0.1]))
    >>> print(aikit.sparse_cross_entropy(x,y))
    {
        a: aikit.array([2.3])
    }

    Instance Method Examples
    ~~~~~~~~~~~~~~~~~~~~~~~~
    With :class:`aikit.Array` input:

    >>> x = aikit.array([2])
    >>> y = aikit.array([0.1, 0.1, 0.7, 0.1])
    >>> print(x.sparse_cross_entropy(y))
    aikit.array([0.357])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([2]))
    >>> y = aikit.Container(a=aikit.array([0.1, 0.1, 0.7, 0.1]))
    >>> print(x.sparse_cross_entropy(y))
    {
        a: aikit.array([0.357])
    }
    """
    aikit.utils.assertions.check_elem_in_list(reduction, ["none", "sum", "mean"])
    true = aikit.one_hot(true, pred.shape[axis])
    return aikit.cross_entropy(
        true, pred, axis=axis, epsilon=epsilon, reduction=reduction, out=out
    )
