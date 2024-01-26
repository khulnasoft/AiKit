"""Collection of Ivy activation functions."""

from typing import Union, Optional, Callable, Literal

# local
import aikit
from aikit.utils.backend import current_backend
from aikit.func_wrapper import (
    handle_array_function,
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    handle_array_like_without_promotion,
    handle_device,
    handle_complex_input,
    handle_backend_invalid,
)
from aikit.utils.exceptions import handle_exceptions


def _gelu_jax_like(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    fn_original: Optional[Callable] = None,
    approximate: bool = False,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    # We don't have the exact implementation
    # cuz the erf function doesn't work on complex numbers
    return fn_original(x, approximate=True, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
@handle_complex_input
def gelu(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    approximate: bool = False,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the Gaussian error linear unit (GELU) activation function.

    Parameters
    ----------
    x
        Input array.
    approximate
        Whether to approximate, default is ``True``. An approximation is always used if
        the input array is complex.
    complex_mode
        optional specifier for how to handle complex data types. See
        ``aikit.func_wrapper.handle_complex_input`` for more detail.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with gelu applied element-wise.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([-1.2, -0.6, 1.5])
    >>> y = aikit.gelu(x)
    >>> y
    aikit.array([-0.138, -0.165, 1.4])

    With :class:`aikit.NativeArray` input:

    >>> x = aikit.native_array([-1.3, 3.8, 2.1])
    >>> y = aikit.gelu(x)
    >>> y
    aikit.array([-0.126, 3.8, 2.06])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([1., 2.]), b=aikit.array([-0.9, -1.]))
    >>> y = aikit.gelu(x)
    >>> y
    {
        a: aikit.array([0.841, 1.95]),
        b: aikit.array([-0.166, -0.159])
    }
    """
    return current_backend(x).gelu(x, approximate=approximate, out=out)


gelu.jax_like = _gelu_jax_like


def _leaky_relu_jax_like(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    fn_original: Optional[Callable] = None,
    alpha: float = 0.2,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    return aikit.where(
        (
            aikit.logical_or(
                aikit.real(x) < 0, aikit.logical_and(aikit.real(x) == 0, aikit.imag(x) < 0)
            )
        ),
        aikit.astype(x * alpha, x.dtype),
        x,
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
@handle_complex_input
def leaky_relu(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    alpha: float = 0.2,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the leaky rectified linear unit function element-wise.

    If the input is complex, then by default each element is scaled by `alpha` if
    either its real part is strictly negative or if its real part is zero and its
    imaginary part is negative. This behaviour can be changed by specifying a different
    `complex_mode`.

    Parameters
    ----------
    x
        Input array.
    alpha
        Negative slope for ReLU.
    complex_mode
        optional specifier for how to handle complex data types. See
        ``aikit.func_wrapper.handle_complex_input`` for more detail.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with leaky relu applied element-wise.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([0.39, -0.85])
    >>> y = aikit.leaky_relu(x)
    >>> print(y)
    aikit.array([ 0.39, -0.17])

    >>> x = aikit.array([1.5, 0.7, -2.4])
    >>> y = aikit.zeros(3)
    >>> aikit.leaky_relu(x, out=y)
    >>> print(y)
    aikit.array([ 1.5 ,  0.7 , -0.48])

    >>> x = aikit.array([[1.1, 2.2, 3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> aikit.leaky_relu(x, out=x)
    >>> print(x)
    aikit.array([[ 1.1 ,  2.2 ,  3.3 ],
       [-0.88, -1.1 , -1.32]])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([0.0, -1.2]), b=aikit.array([0.4, -0.2]))
    >>> x = aikit.leaky_relu(x, out=x)
    >>> print(x)
    {
        a: aikit.array([0., -0.24000001]),
        b: aikit.array([0.40000001, -0.04])
    }
    """
    return current_backend(x).leaky_relu(x, alpha=alpha, out=out)


leaky_relu.jax_like = _leaky_relu_jax_like


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def log_softmax(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the log_softmax function element-wise.

    Parameters
    ----------
    x
        Input array.
    axis
        The dimension log_softmax would be performed on. The default is ``None``.
    complex_mode
        optional specifier for how to handle complex data types. See
        ``aikit.func_wrapper.handle_complex_input`` for more detail.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The output array with log_softmax applied element-wise to input.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([-1.0, -0.98])
    >>> y = aikit.log_softmax(x)
    >>> print(y)
    aikit.array([-0.703, -0.683])

    >>> x = aikit.array([1.0, 2.0, 3.0])
    >>> y = aikit.log_softmax(x)
    >>> print(y)
    aikit.array([-2.41, -1.41, -0.408])

    With :class:`aikit.NativeArray` input:

    >>> x = aikit.native_array([1.5, 0.5, 1.0])
    >>> y = aikit.log_softmax(x)
    >>> print(y)
    aikit.array([-0.68, -1.68, -1.18])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([1.5, 0.5, 1.0]))
    >>> y = aikit.log_softmax(x)
    >>> print(y)
    {
        a: aikit.array([-0.68, -1.68, -1.18])
    }

    >>> x = aikit.Container(a=aikit.array([1.0, 2.0]), b=aikit.array([0.4, -0.2]))
    >>> y = aikit.log_softmax(x)
    >>> print(y)
    {
        a: aikit.array([-1.31, -0.313]),
        b: aikit.array([-0.437, -1.04])
    }
    """
    return current_backend(x).log_softmax(x, axis=axis, out=out)


def _relu_jax_like(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    fn_original=None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    return aikit.where(
        (
            aikit.logical_or(
                aikit.real(x) < 0, aikit.logical_and(aikit.real(x) == 0, aikit.imag(x) < 0)
            )
        ),
        aikit.array(0.0, dtype=x.dtype),
        x,
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
@handle_complex_input
def relu(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the rectified linear unit function element-wise.

    If the input is complex, then by default each element is set to zero  if
    either its real part is strictly negative or if its real part is zero and its
    imaginary part is negative. This behaviour can be changed by specifying a different
    `complex_mode`.

    Parameters
    ----------
    x
        input array
    complex_mode
        optional specifier for how to handle complex data types. See
        ``aikit.func_wrapper.handle_complex_input`` for more detail.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the rectified linear unit activation of each element in
        ``x``.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([-1., 0., 1.])
    >>> y = aikit.relu(x)
    >>> print(y)
    aikit.array([0., 0., 1.])

    >>> x = aikit.array([1.5, 0.7, -2.4])
    >>> y = aikit.zeros(3)
    >>> aikit.relu(x, out = y)
    >>> print(y)
    aikit.array([1.5, 0.7, 0.])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([1.0, -1.2]), b=aikit.array([0.4, -0.2]))
    >>> x = aikit.relu(x, out=x)
    >>> print(x)
    {
        a: aikit.array([1., 0.]),
        b: aikit.array([0.40000001, 0.])
    }
    """
    return current_backend(x).relu(x, out=out)


relu.jax_like = _relu_jax_like


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
@handle_complex_input
def sigmoid(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the sigmoid function element-wise.

    Parameters
    ----------
    x
        input array.
    complex_mode
        optional specifier for how to handle complex data types. See
        ``aikit.func_wrapper.handle_complex_input`` for more detail.
    out
        optional output array, for writing the result to. It must have a shape that the
        input broadcast to.
        default: None

    Returns
    -------
    ret
        an array containing the sigmoid activation of each element in ``x``.
        sigmoid activation of x is defined as 1/(1+exp(-x)).

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([-1.0, 1.0, 2.0])
    >>> y = aikit.sigmoid(x)
    >>> print(y)
    aikit.array([0.2689414 , 0.7310586 , 0.88079703])

    >>> x = aikit.array([-1.0, 1.0, 2.0])
    >>> y = aikit.zeros(3)
    >>> aikit.sigmoid(x, out=y)
    >>> print(y)
    aikit.array([0.2689414 , 0.7310586 , 0.88079703])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([0.]),
    ...                   b=aikit.Container(c=aikit.array([1.]),
    ...                                   d=aikit.array([2.])))
    >>> y = aikit.sigmoid(x)
    >>> print(y)
    {
        a: aikit.array([0.5]),
        b: {
            c: aikit.array([0.7310586]),
            d: aikit.array([0.88079703])
        }
    }

    >>> x = aikit.Container(a=aikit.array([0.]),
    ...                   b=aikit.Container(c=aikit.array([1.]),
    ...                                   d=aikit.array([2.])))
    >>> y = aikit.Container(a=aikit.array([0.]),
    ...                   b=aikit.Container(c=aikit.array([0.]),
    ...                                   d=aikit.array([0.])))
    >>> aikit.sigmoid(x, out=y)
    >>> print(y)
    {
        a: aikit.array([0.5]),
        b: {
            c: aikit.array([0.7310586]),
            d: aikit.array([0.88079703])
        }
    }
    """
    return current_backend(x).sigmoid(x, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
@handle_complex_input
def softmax(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the softmax function element-wise.

    Parameters
    ----------
    x
        Input array.
    axis
        The dimension softmax would be performed on. The default is ``None``.
    complex_mode
        optional specifier for how to handle complex data types. See
        ``aikit.func_wrapper.handle_complex_input`` for more detail.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with softmax applied element-wise.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([1.0, 0, 1.0])
    >>> y = aikit.softmax(x)
    >>> print(y)
    aikit.array([0.422, 0.155, 0.422])

    >>> x = aikit.array([[1.1, 2.2, 3.3],
    ...                [4.4, 5.5, 6.6]])
    >>> y = aikit.softmax(x, axis = 1)
    >>> print(y)
    aikit.array([[0.0768, 0.231 , 0.693 ],
               [0.0768, 0.231 , 0.693 ]])
    """
    return current_backend(x).softmax(x, axis=axis, out=out)


def _wrap_between(y, a):
    """Wrap y between [-a, a]"""
    a = aikit.array(a, dtype=y.dtype)
    a2 = aikit.array(2 * a, dtype=y.dtype)
    zero = aikit.array(0, dtype=y.dtype)
    rem = aikit.remainder(aikit.add(y, a), a2)
    rem = aikit.where(rem < zero, rem + a2, rem) - a
    return rem


def _softplus_jax_like(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    fn_original=None,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[aikit.Array] = None,
):
    if beta is not None:
        x_beta = aikit.multiply(x, aikit.array(beta, dtype=x.dtype))
    else:
        x_beta = x
    amax = aikit.relu(x_beta)
    res = aikit.subtract(x_beta, aikit.multiply(amax, aikit.array(2, dtype=x.dtype)))
    res = aikit.add(amax, aikit.log(aikit.add(1, aikit.exp(res))))
    res = aikit.real(res) + _wrap_between(aikit.imag(res), aikit.pi).astype(
        x.dtype
    ) * aikit.astype(1j, x.dtype)
    if beta is not None:
        res = aikit.divide(res, aikit.array(beta, dtype=x.dtype))
    if threshold is not None:
        res = aikit.where(
            aikit.real(x_beta) < threshold,
            res,
            x,
        ).astype(x.dtype)
    return res


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
@handle_complex_input
def softplus(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the softplus function element-wise.

    If the input is complex, then by default we apply the softplus operation
    `log(1+ exp(x))` to  each element
    If threshold is set we check if either its real part is strictly negative or
    if its real part is zero and its imaginary part is negative then we apply
    `input×β > threshold`.

    Parameters
    ----------
    x
        input array.
    beta
        The beta value for the softplus formation. Default: ``None``.
    threshold
        values above this revert to a linear function
        If the input is complex, only its real part is considered. Default: ``None``
    complex_mode
        optional specifier for how to handle complex data types. See
        ``aikit.func_wrapper.handle_complex_input`` for more detail.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the softplus activation of each element in ``x``.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([-0.3461, -0.6491])
    >>> y = aikit.softplus(x)
    >>> print(y)
    aikit.array([0.535,0.42])

    >>> x = aikit.array([-0.3461, -0.6491])
    >>> y = aikit.softplus(x, beta=0.5)
    >>> print(y)
    aikit.array([1.22, 1.09])

    >>> x = aikit.array([1., 2., 3.])
    >>> y = aikit.softplus(x, threshold=2)
    >>> print(y)
    aikit.array([1.31, 2.13, 3.  ])
    """
    return current_backend(x).softplus(x, beta=beta, threshold=threshold, out=out)


softplus.jax_like = _softplus_jax_like


# Softsign
@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def softsign(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the softsign function element-wise.

    Parameters
    ----------
    x
        Input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with softsign applied element-wise.

    Examples
    --------
    With :class:`aikit.Array` input:
    >>> x = aikit.array([1.0, 2.0, 3.0])
    >>> y = aikit.softsign(x)
    >>> print(y)
    aikit.array([0.5, 0.66666667, 0.75])
    """
    return current_backend(x).softsign(x, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
@handle_complex_input
def mish(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the mish activation function element-wise.

    Parameters
    ----------
    x
        input array
    complex_mode
        optional specifier for how to handle complex data types. See
        ``aikit.func_wrapper.handle_complex_input`` for more detail.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the mish activation of each element in
        ``x``.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([-1., 0., 1.])
    >>> y = aikit.mish(x)
    >>> print(y)
    aikit.array([-0.30340147,  0.        ,  0.86509842])

    >>> x = aikit.array([1.5, 0.7, -2.4])
    >>> y = aikit.zeros(3)
    >>> aikit.mish(x, out = y)
    >>> print(y)
    aikit.array([ 1.40337825,  0.56114835, -0.20788449])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([1.0, -1.2]), b=aikit.array([0.4, -0.2]))
    >>> x = aikit.mish(x)
    >>> print(x)
    {
        a: aikit.array([0.86509842, -0.30883577]),
        b: aikit.array([0.28903052, -0.10714479])
    }
    """
    return current_backend(x).mish(x, out=out)


def _hardswish_jax_like(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    fn_original=None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    def hard_sigmoid(x):
        return aikit.relu6(x + 3.0) / 6

    return aikit.multiply(x, hard_sigmoid(x).astype(x.dtype))


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_complex_input
def hardswish(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the hardswish activation function element-wise.

    Parameters
    ----------
    x
        input array
    complex_mode
        optional specifier for how to handle complex data types. See
        ``aikit.func_wrapper.handle_complex_input`` for more detail.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the hardswish activation of each element in ``x``.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([0., 0., 4.])
    >>> y = aikit.hardswish(x)
    >>> y
    aikit.array([0., 0., 4.])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([-3., 4., 5.]), b=aikit.array([0., 5.]))
    >>> x = aikit.hardswish(x, out=x)
    >>> x
    {
        a: aikit.array([-0.,  4.,  5.]),
        b: aikit.array([0., 5.])
    }
    """
    return current_backend(x).hardswish(x, out=out)


hardswish.jax_like = _hardswish_jax_like
