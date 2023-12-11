# global
from typing import Union, Optional

# local
import aikit
from aikit.func_wrapper import (
    handle_nestable,
    inputs_to_aikit_arrays,
    handle_array_like_without_promotion,
    handle_array_function,
    to_native_arrays_and_back,
)
from aikit.utils.exceptions import handle_exceptions


# log_poisson_loss
@handle_exceptions
@handle_nestable
@inputs_to_aikit_arrays
@handle_array_function
def log_poisson_loss(
    true: Union[aikit.Array, aikit.NativeArray],
    pred: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    compute_full_loss: bool = False,
    axis: int = -1,
    reduction: str = "none",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Compute the log-likelihood loss between the prediction and the target
    under the assumption that the target has a Poisson distribution. Caveat: By
    default, this is not the exact loss, but the loss minus a constant term
    [log(z!)]. That has no effect for optimization, but does not play well with
    relative loss comparisons. To compute an approximation of the log factorial
    term, specify ``compute_full_loss=True`` to enable Stirling's
    Approximation.

    Parameters
    ----------
    true
        input array containing true labels.
    pred
        input array containing Predicted labels.
    compute_full_loss
        whether to compute the full loss. If false, a constant term is dropped
        in favor of more efficient optimization. Default: ``False``.
    axis
        the axis along which to compute the log-likelihood loss. If axis is ``-1``,
        the log-likelihood loss will be computed along the last dimension.
        Default: ``-1``.
    reduction
        ``'none'``: No reduction will be applied to the output.
        ``'mean'``: The output will be averaged.
        ``'sum'``: The output will be summed. Default: ``'none'``.
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        The binary log-likelihood loss between the given distributions.


    Examples
    --------
    >>> x = aikit.array([0, 0, 1, 0])
    >>> y = aikit.array([0.25, 0.25, 0.25, 0.25])
    >>> print(aikit.log_poisson_loss(x, y))
    aikit.array([1.28402555, 1.28402555, 1.03402555, 1.28402555])

    >>> z = aikit.array([0.1, 0.1, 0.7, 0.1])
    >>> print(aikit.log_poisson_loss(x, z, reduction='mean'))
    aikit.array(1.1573164)
    """
    try:
        assert true.shape == pred.shape
    except ValueError as e:
        raise ValueError(
            "`pred` and `true` must have the same shape, received "
            f"({pred.shape} vs {true.shape})."
        ) from e

    loss = aikit.exp(pred) - pred * true
    if compute_full_loss:
        stirling_approx = (
            (true * aikit.log(true)) - true + (0.5 * aikit.log(2 * aikit.pi * true))
        )
        cond = aikit.logical_and(true >= 0.0, true <= 1.0)
        loss += aikit.where(cond, aikit.zeros_like(loss), stirling_approx)
    if reduction == "sum":
        return aikit.sum(loss, axis=axis, out=out)
    elif reduction == "mean":
        return aikit.mean(loss, axis=axis, out=out)
    else:
        return aikit.inplace_update(out, loss) if out is not None else loss


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_aikit_arrays
@handle_array_function
def l1_loss(
    input: Union[aikit.Array, aikit.NativeArray],
    target: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    reduction: Optional[str] = "mean",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """
    Compute L1 loss (Mean Absolute Error - MAE) between targeticted and input values.

    Parameters
    ----------
    input : Union[aikit.Array, aikit.NativeArray]
        Input array containing input values.
    target : Union[aikit.Array, aikit.NativeArray]
        Input array containing targeted values.
    reduction : str, optional
        Reduction method for the output loss. Options:
        "none" (no reduction), "mean" (mean of losses),
        "sum" (sum of losses). Default: "mean".
    out : Optional[aikit.Array], optional
        Optional output array for writing the result to.
        It must have a shape that the inputs broadcast to.


    Returns
    -------
    aikit.Array
        The L1 loss (MAE) between the given input and targeticted values.


    Examples
    --------
    >>> x = aikit.array([1.0, 2.0, 3.0])
    >>> y = aikit.array([0.5, 2.5, 2.0])
    >>> print(aikit.l1_loss(x, y))
    aikit.array(0.6)
    >>> a = aikit.array([[1.0, 2.0], [3.0, 4.0]])
    >>> b = aikit.array([[0.5, 1.5], [2.5, 3.5]])
    >>> print(aikit.l1_loss(a, b))
    aikit.array(0.5)
    """
    loss = aikit.abs(target - input)

    if reduction == "sum":
        return aikit.sum(loss, out=out)
    elif reduction == "mean":
        return aikit.mean(loss, out=out)
    else:
        return aikit.inplace_update(out, loss) if out is not None else loss


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_aikit_arrays
@handle_array_function
def huber_loss(
    true: Union[aikit.Array, aikit.NativeArray],
    pred: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    delta: Optional[float] = 1.0,
    reduction: Optional[str] = "mean",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Compute the Huber loss (smooth L1 loss) between true and predicted
    values.

    Parameters
    ----------
    true: array_like
        The true (ground truth) values.
    pred : array_like
        The predicted values by the model.
    delta : float, optional
        The threshold parameter that determines the point where the loss transitions fro
        -m
        squared error to absolute error. Default is 1.0.
    reduction : str, optional
        The type of reduction to apply to the loss. Possible values are "mean" (default)
        and "sum".
    out : array_like, optional
        Optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret : array_like
        The Huber loss between the true and predicted values.

    Examples
    --------
    >>> true = aikit.array([2, 4, 7, 1])
    >>> pred = aikit.array([2.5, 3.5, 8, 0.8])
    >>> huber_loss(true, pred, delta=1.0)
    aikit.array([0.125, 0.125, 0.5  , 0.125])

    >>> huber_loss(true, pred, delta=2.0)
    aikit.array([0.125, 0.125, 0.5  , 0.2  ])

    >>> huber_loss(true, pred, delta=0.5)
    aikit.array([0.25 , 0.25 , 0.   , 0.125])
    """
    abs_diff = aikit.abs(true - pred)
    quadratic_loss = 0.5 * (abs_diff**2)
    linear_loss = delta * (abs_diff - 0.5 * delta)
    loss = aikit.where(abs_diff <= delta, quadratic_loss, linear_loss)

    if reduction == "sum":
        return aikit.sum(loss, out=out)
    elif reduction == "mean":
        return aikit.mean(loss, out=out)
    else:
        return aikit.inplace_update(out, loss) if out is not None else loss


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_aikit_arrays
@handle_array_function
def smooth_l1_loss(
    input: Union[aikit.Array, aikit.NativeArray],
    target: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    beta: Optional[float] = 1.0,
    reduction: Optional[str] = "mean",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Compute the smooth L1 loss between two input tensors.

    Parameters
    ----------
    input : array_like
        First input tensor.
    target : array_like
        Second input tensor.
    beta : float, optional
        The smooth parameter. Default is 1.0.
    reduction : str, optional
        Specifies the type of reduction to apply to the output.
        Should be one of 'none', 'sum', or 'mean'. Default is 'mean'.
    out : array, optional
        Optional output array, for writing the result to.
        It must have a shape that the inputs broadcast to.

    Returns
    -------
    ret : array
        The smooth_l1_loss between the two input tensors.

    Examples
    --------
    >>> input = aikit.array([1.0, 2.0, 3.0])
    >>> target = aikit.array([2.5, 1.8, 3.2])
    >>> aikit.smooth_l1_loss(x, y, beta=1.0)
    aikit.array(0.3467)
    >>> input = aikit.array([1.0, 2.0, 3.0])
    >>> target = aikit.array([6.0, 2.0, 3.0])
    >>> aikit.smooth_l1_loss(x, y, beta=1.0)
    aikit.array(1.5)
    >>> input = aikit.array([2.0, 3.0, 5.0, 7.0])
    >>> target = aikit.array([2.5, 3.5, 5.5, 6.5])
    >>> loss = aikit.smooth_l1_loss(input, target, beta=1.5, reduction='sum')
    aikit.array(0.5)
    >>> input = aikit.array([0.8, 1.2, 2.5, 3.7])
    >>> target = aikit.array([0.9, 1.0, 2.3, 3.6])
    >>> loss = aikit.smooth_l1_loss(input, target, beta=0.5, reduction='none')
    aikit.array([0.0133, 0.0250, 0.0056, 0.0025])
    >>> input = aikit.array([2.0, 3.0, 5.0, 7.0])
    >>> target = aikit.array([2.5, 3.5, 5.5, 6.5])
    >>> loss = aikit.smooth_l1_loss(input, target, beta=0.2, reduction='mean')
    aikit.array(0.025)

    With :class:`aikit.NativeArray` input:

    >>> x = aikit.native_array([1.5, 2.2, 3.7])
    >>> y = aikit.native_array([2.1, 1.9, 3.5])
    >>> print(aikit.smooth_l1_loss(x, y, beta=0.5))
    aikit.array(0.0675)

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([1.0, 2.0, 3.0]))
    >>> y = aikit.Container(a=aikit.array([2.5, 1.8, 3.2]))
    >>> print(aikit.smooth_l1_loss(x, y, beta=1.0))
    {
        a: aikit.array(0.3467)
    }

    With a mix of :class:`aikit.Array` and :class:`aikit.NativeArray` inputs:

    >>> x = aikit.array([1.0, 2.0, 3.0])
    >>> y = aikit.native_array([6.0, 2.0, 3.0])
    >>> print(aikit.smooth_l1_loss(x, y, beta=0.5))
    aikit.array(1.5)

    With a mix of :class:`aikit.Array` and :class:`aikit.Container` inputs:

    >>> x = aikit.array([1.0, 2.0, 3.0])
    >>> y = aikit.Container(a=aikit.array([6.0, 2.0, 3.0]))
    >>> print(aikit.smooth_l1_loss(x, y, beta=1.0))
    {
        a: aikit.array(1.5)
    }

    Instance Method Examples
    ~~~~~~~~~~~~~~~~~~~~~~~~
    With :class:`aikit.Array` input:

    >>> x = aikit.array([1.0, 2.0, 3.0])
    >>> y = aikit.array([2.5, 1.8, 3.2])
    >>> print(x.smooth_l1_loss(y, beta=1.0))
    aikit.array(0.3467)

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([1.0, 2.0, 3.0]))
    >>> y = aikit.Container(a=aikit.array([2.5, 1.8, 3.2]))
    >>> print(x.smooth_l1_loss(y, beta=1.0))
    {
        a: aikit.array(0.3467)
    }
    """
    if beta < 1e-5:
        # if beta == 0,  will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = aikit.abs(input - target)
    else:
        n = aikit.abs(input - target)
        cond = n < beta
        loss = aikit.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)

    if reduction == "mean":
        return aikit.mean(loss, out=out)
    elif reduction == "sum":
        return aikit.sum(loss, out=out)
    elif reduction == "none":
        return aikit.inplace_update(out, loss) if out is not None else loss


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_aikit_arrays
@handle_array_function
def soft_margin_loss(
    input: Union[aikit.Array, aikit.NativeArray],
    target: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    reduction: Optional[str] = "mean",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Compute the soft-margin hinge loss between predicted scores and true
    binary labels.

    Parameters
    ----------
    input : array_like
        True binary labels, of shape (batch_size,).
    target : array_like
        Predicted scores, of shape (batch_size,).
    reduction : {'mean', 'sum', 'none'}, optional
        Type of reduction to apply to the output. Default is 'mean'.
    out : array_like, optional
        Optional output array, for writing the result to.
        It must have a shape that the inputs broadcast to.

    Returns
    -------
    ret : array
        The soft-margin hinge loss between the predicted scores
        and true binary labels.

    Examples
    --------
    >>> input = aikit.array([1, 0, 1, 0])
    >>> target = aikit.array([0.8, 0.2, -0.6, 1.5])
    >>> aikit.soft_margin_loss(input, target)
    aikit.array(0.6987)

    >>> input = aikit.array([1, 1, 0, 0])
    >>> target = aikit.array([0.8, 0.7, 0.2, 0.1])
    >>> aikit.soft_margin_loss(input, target, reduction='sum')
    aikit.array(2.1606)

    >>> input = aikit.array([1, 1, 0, 0])
    >>> target = aikit.array([0.8, 0.7, 0.2, 0.1])
    >>> aikit.soft_margin_loss(input, target, reduction='none')
    aikit.array([0.3711, 0.4032, 0.6931, 0.6931])
    """
    loss = aikit.sum(aikit.log1p(aikit.exp(-input * target))) / input.size

    if reduction == "sum":
        return aikit.sum(loss, out=out)
    elif reduction == "mean":
        return aikit.mean(loss, out=out)
    else:
        return aikit.inplace_update(out, loss) if out is not None else loss


@handle_exceptions
@handle_nestable
@inputs_to_aikit_arrays
@handle_array_function
def kl_div(
    input: Union[aikit.Array, aikit.NativeArray],
    target: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    reduction: Optional[str] = "mean",
    log_target=False,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Compute the Kullback-Leibler divergence loss between two input tensors
    (conventionally, probability distributions).

    Parameters
    ----------
    input : array_like
        Tensor of arbitrary shape in log-probabilities
    target : array_like
        Tensor of the same shape as input. See log_target for
        the target’s interpretation
    reduction : {'mean', 'sum', 'batchmean', 'none'}, optional
        Type of reduction to apply to the output. Default is 'mean'.
    log_target : bool
        A flag indicating whether target is passed in the log space.
        It is recommended to pass certain distributions (like softmax)
        in the log space to avoid numerical issues caused by explicit log.
        Default: False

    Returns
    -------
    ret : array
        The Kullback-Leibler divergence loss between the two input tensors.

    Examples
    --------
    >>> input = aikit.array([[0.2, 0.8], [0.5, 0.5]])
    >>> target = aikit.array([[0.6, 0.4], [0.3, 0.7]])
    >>> aikit.kl_div(input, target)
    aikit.array(-0.555969)

    >>> input = aikit.array([[0.2, 0.8], [0.5, 0.5]])
    >>> target = aikit.array([[0.6, 0.4], [0.3, 0.7]])
    >>> aikit.kl_div(input, target, reduction='sum')
    aikit.array(-2.223876)

    >>> input = aikit.array([[0.2, 0.8], [0.5, 0.5]])
    >>> target = aikit.array([[0.6, 0.4], [0.3, 0.7]])
    >>> aikit.kl_div(input, target, reduction='batchmean')
    aikit.array(-1.111938)

    >>> input = aikit.array([0.2, 0.8], [0.5, 0.5])
    >>> target = aikit.array([0.6, 0.4], [0.3, 0.7])
    >>> aikit.kl_div(input, target, reduction='none')
    aikit.array([[-0.42649534, -0.68651628],
                [-0.51119184, -0.59967244]])
    """
    if not log_target:  # default
        loss_pointwise = target * (aikit.log(target) - input)
    else:
        loss_pointwise = aikit.exp(target) * (target - input)

    if reduction == "mean":  # default
        loss = aikit.mean(loss_pointwise)
    elif reduction == "batchmean":  # mathematically correct
        loss = aikit.sum(loss_pointwise) / input.shape[0]
    elif reduction == "sum":
        loss = aikit.sum(loss_pointwise)
    else:  # reduction == "none"
        loss = loss_pointwise
    return aikit.inplace_update(out, loss) if out is not None else loss


kl_div.mixed_backend_wrappers = {
    "to_add": (
        "handle_backend_invalid",
        "inputs_to_native_arrays",
        "outputs_to_aikit_arrays",
        "handle_out_argument",
    ),
    "to_skip": ("inputs_to_aikit_arrays",),
}


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
def poisson_nll_loss(
    input: Union[aikit.Array, aikit.NativeArray],
    target: Union[aikit.Array, aikit.NativeArray],
    *,
    log_input: bool = True,
    full: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> aikit.Array:
    r"""Compute the Poisson Negative Log Likelihood Loss.

    This function calculates the negative log likelihood loss
    between the `input` and `target`under the assumption that
    the target follows a Poisson distribution. By default, the loss
    is not the exact loss, but the loss minus a constant term [log(z!)].
    This omission does not affect optimization but can be significant for
    relative loss comparisons. The Stirling's Approximation is used to
    approximate the log factorial term when `full` is set to True.

    Parameters
    ----------
    input
        Expectation of the underlying Poisson distribution.
    target
        Random sample from the Poisson distribution described by the input.
    log_input
        If `True`, the loss is computed as
        :math:`exp(input) - target * input`. If `False`, the loss is computed as
        :math:`input - target * log(input + eps)`. Default is `True`.
    full
        Whether to compute the full loss, i.e., to add the Stirling approximation term
        :math:`target * log(target) - target + 0.5 * log(2 * pi * target)`.
        Default is `False`.
    eps
        Small value to prevent evaluation of `log(0)` when `log_input` is `False`.
        Default is 1e-8.
    reduction
        Specifies the reduction applied to the output.
        Options are 'none', 'mean', or 'sum'.
        'none': no reduction will be applied.
        'mean': the output will be averaged.
        'sum': the output will be summed.
        Default is 'mean'.

    Returns
    -------
    ret
        An array of the same shape as `input` representing
        the Poisson Negative Log Likelihood Loss.

    Raises
    ------
    ValueError
        If the `input` and `target` tensors do not have the same shape.

    Examples
    --------
    >>> input_tensor = aikit.array([1, 2, 3, 4], dtype=aikit.float64)
    >>> target_tensor = aikit.array([2, 2, 2, 2], dtype=aikit.float64)
    >>> loss = poisson_nll_loss(input_tensor, target_tensor, log_input=False)
    >>> print(loss)
    aikit.array(0.91097307)
    """
    return aikit.current_backend().poisson_nll_loss(
        input,
        target,
        log_input=log_input,
        full=full,
        eps=eps,
        reduction=reduction,
    )
