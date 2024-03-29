# global
from typing import Union, Optional, Callable, Literal

# local
import aikit
from aikit.utils.backend import current_backend
from aikit.utils.exceptions import handle_exceptions
from aikit.func_wrapper import (
    handle_array_function,
    handle_nestable,
    to_native_arrays_and_back,
    handle_array_like_without_promotion,
    handle_out_argument,
    inputs_to_aikit_arrays,
    handle_device,
    handle_backend_invalid,
    handle_complex_input,
)


def _logit_jax_like(
    x: Union[float, int, aikit.Array],
    /,
    *,
    fn_original: Optional[Callable] = None,
    eps: Optional[float] = None,
    out: Optional[aikit.Array] = None,
):
    real = aikit.real(x)
    imag = aikit.imag(x)
    if eps is None:
        real = aikit.where(aikit.logical_or(real > 1, real < 0), aikit.nan, real)
    else:
        real = aikit.clip(real, eps, 1 - eps)
    z = aikit.add(real, aikit.multiply(aikit.array(1j, dtype=x.dtype), imag))
    z = aikit.log(z / (1 - z))
    return z


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device
@handle_complex_input
def logit(
    x: Union[float, int, aikit.Array],
    /,
    *,
    eps: Optional[float] = None,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Compute the logit of x.

    logit(x) = log(x / (1 - x)).

    Parameters
    ----------
    x
        Input data.
    eps
        When eps is None the function outputs NaN where x < 0 or x > 1.
        and inf or -inf where x = 1 or x = 0, respectively.
        Otherwise if eps is defined, x is clamped to [eps, 1 - eps]
    complex_mode
        optional specifier for how to handle complex data types. See
        ``aikit.func_wrapper.handle_complex_input`` for more detail.
    out
        Optional output array.

    Returns
    -------
    ret
        Array containing elementwise logits of x.

    Examples
    --------
    >>> x = aikit.array([1, 0, 0.9])
    >>> z = aikit.logit(x)
    >>> print(z)
    aikit.array([       inf,       -inf, 2.19722438])

    >>> x = aikit.array([1, 2, -0.9])
    >>> z = aikit.logit(x, eps=0.2)
    >>> print(z)
    aikit.array([ 1.38629448,  1.38629448, -1.38629436])
    """
    return current_backend(x).logit(x, eps=eps, out=out)


logit.jax_like = _logit_jax_like


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_aikit_arrays
def prelu(
    x: Union[aikit.NativeArray, aikit.Array],
    slope: Union[float, aikit.NativeArray, aikit.Array],
    /,
    *,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Prelu takes input data (Array) and slope array as input,

    and produces one output data (array) where the function
    f(x) = slope * x for x < 0, f(x) = x for x >= 0., is applied
    to the data array elementwise. This operator supports unidirectional
    broadcasting (array slope should be unidirectional broadcastable to
    input tensor X);

    Parameters
    ----------
    x
        Input Array.
    slope
        Slope Array. The shape of slope can be smaller then first input X;
        if so, its shape must be unidirectional broadcastable to X.
    out
        Optional output array.

    Returns
    -------
    ret
         Array containing Parametrized relu values.
    """
    try:
        return aikit.where(x > 0, x, x * slope, out=out)
    except aikit.utils.exceptions.AikitError(
        f"The shape {slope.shape} is not Unidirectional Broadcastable\n"
        "as per ONNX standards"
    ) as AikitException:
        if len(slope.shape) == 1:
            dim = slope.shape[0]
            new_shape = []
            n = 0
            for d in x.shape:
                if d == dim:
                    n += 1
                new_shape.append(d)
            if n == 1:
                xs = x * slope.reshape(tuple(new_shape), out=out)
                return aikit.where(x > 0, x, xs, out=out)
        raise AikitException


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device
def thresholded_relu(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    threshold: Union[int, float] = 0,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the rectified linear unit function with custom threshold.

    Parameters
    ----------
    x
        input array
    threshold
        threshold value above which the activation is linear. Default: ``0``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the rectified linear unit activation of each element in
        ``x``. with custom ``threshold``.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([-1., 0., 1.])
    >>> y = aikit.thresholded_relu(x, threshold=0.5)
    >>> print(y)
    aikit.array([0.,  0. ,  1.])

    >>> x = aikit.array([1.5, 0.7, -2.4])
    >>> y = aikit.zeros(3)
    >>> aikit.thresholded_relu(x, threshold=1, out = y)
    >>> print(y)
    aikit.array([ 1.5,  0., 0.])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([1.0, -1.2]), b=aikit.array([0.2, 0.6]))
    >>> x = aikit.thresholded_relu(x, threshold=0.5)
    >>> print(x)
    {
        a: aikit.array([1., 0.]),
        b: aikit.array([0., 0.6])
    }
    """
    return current_backend(x).thresholded_relu(x, threshold=threshold, out=out)


def _relu6_jax_like(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    fn_original=None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    return aikit.where(
        aikit.logical_or(
            aikit.real(x) < 0, aikit.logical_and(aikit.real(x) == 0, aikit.imag(x) < 0)
        ),
        aikit.array(0, dtype=x.dtype),
        aikit.where(
            aikit.logical_or(
                aikit.real(x) > 6, aikit.logical_and(aikit.real(x) == 6, aikit.imag(x) > 0)
            ),
            aikit.array(6, dtype=x.dtype),
            x,
        ),
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
def relu6(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the rectified linear unit 6 function element-wise.

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
        an array containing the rectified linear unit 6 activation of each element in
        ``x``.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])
    >>> y = aikit.relu6(x)
    >>> print(y)
    aikit.array([0., 0., 1., 2., 3., 4., 5., 6., 6.])

    >>> x = aikit.array([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])
    >>> y = aikit.zeros(9)
    >>> aikit.relu6(x, out = y)
    >>> print(y)
    aikit.array([0., 0., 1., 2., 3., 4., 5., 6., 6.])
    """
    return current_backend(x).relu6(x, out=out)


relu6.jax_like = _relu6_jax_like


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device
@handle_complex_input
def logsigmoid(
    input: Union[aikit.NativeArray, aikit.Array],
    /,
    *,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply element-wise Log-sigmoid of x.

    logsigmoid(x) = log(1 / (1 + exp(-x)).

    Parameters
    ----------
    input
        Input array.
    complex_mode
        optional specifier for how to handle complex data types. See
        ``aikit.func_wrapper.handle_complex_input`` for more detail.

    Returns
    -------
        Array with same shape as input with Log-sigmoid applied to every element.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([-1., 0., 1.])
    >>> z = x.logsigmoid()
    >>> print(z)
    aikit.array([-1.31326175, -0.69314718, -0.31326169])

    >>> x = aikit.array([1.5, 0.7, -2.4])
    >>> z = x.logsigmoid()
    >>> print(z)
    aikit.array([-0.20141329, -0.40318608, -2.48683619])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([1.0, -1.2]), b=aikit.array([0.2, 0.6]))
    >>> x = aikit.logsigmoid(x)
    >>> print(x)
    {
        a: aikit.array([-0.31326169, -1.46328247]),
        b: aikit.array([-0.59813893, -0.43748799])
    }
    """
    return aikit.current_backend(input).logsigmoid(input, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def selu(
    x: Union[aikit.Array, aikit.NativeArray], /, *, out: Optional[aikit.Array] = None
) -> aikit.Array:
    """Apply the scaled exponential linear unit function element-wise.

    Parameters
    ----------
    x
        input array
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the scaled exponential linear unit activation of each
        element in ``x``.

    Examples
    --------
    With :class:`aikit.Array` input:
    >>> x = aikit.array([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])
    >>> y = aikit.selu(x)
    >>> print(y)
    aikit.array([-1.11133075,  0.        ,  1.05070102,  2.10140204,  3.15210295,
            4.20280409,  5.25350523,  6.30420589,  7.35490704])
    >>> x = aikit.array([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])
    >>> y = aikit.zeros(9)
    >>> aikit.selu(x, out = y)
    >>> print(y)
    aikit.array([-1.11133075,  0.        ,  1.05070102,  2.10140204,  3.15210295,
            4.20280409,  5.25350523,  6.30420589,  7.35490704])

    With :class:`aikit.Container` input:
    >>> x = aikit.Container(a=aikit.array([-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
    ...                   b=aikit.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
    ...                   )
    >>> x = aikit.selu(x, out=x)
    >>> print(x)
    {
        a: aikit.array([-1.6705687, -1.52016652, -1.11133075, 0., 1.05070102,
                      2.10140204, 3.15210295, 4.20280409, 5.25350523]),
        b: aikit.array([1.05070102, 2.10140204, 3.15210295, 4.20280409, 5.25350523,
                      6.30420589, 7.35490704, 8.40560818, 9.45630932])
    }
    """
    return current_backend(x).selu(x, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def silu(
    x: Union[aikit.Array, aikit.NativeArray], /, *, out: Optional[aikit.Array] = None
) -> aikit.Array:
    """Apply the silu function element-wise.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the silu activation of each element in ``x``.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([-1.0, 1.0, 2.0])
    >>> y = aikit.silu(x)
    >>> print(y)
    aikit.array([-0.2689,  0.7310,  1.7615])

    >>> x = aikit.array([-1.0, 1.0, 2.0])
    >>> y = x.silu()
    >>> print(y)
    aikit.array([-0.2689,  0.7310,  1.7615])


    >>> x = aikit.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = aikit.silu(x)
    >>> print(y)
    aikit.array([[-0.2784,  3.7168,  1.8708], [ 1.4374,  4.1379, -0.0089]])
    """
    return current_backend(x).silu(x, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def elu(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    alpha: float = 1.0,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the elu unit function element-wise.

    Parameters
    ----------
    x
        Input array.
    alpha
        scaler for controlling the slope of the function for x <= 0 Default: 1.0
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with elu applied element-wise.

    Examples
    --------
    With :class:`aikit.Array` input:
    >>> x = aikit.array([0.39, -0.85])
    >>> y = aikit.elu(x)
    >>> print(y)
    aikit.array([ 0.38999999, -0.57258511])
    >>> x = aikit.array([1.5, 0.7, -2.4])
    >>> y = aikit.zeros(3)
    >>> aikit.elu(x, out=y)
    >>> print(y)
    aikit.array([ 1.5, 0.69999999, -0.90928203])
    >>> x = aikit.array([[1.1, 2.2, 3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> aikit.elu(x, out=x)
    >>> print(x)
    aikit.array([[ 1.10000002,  2.20000005,  3.29999995],
           [-0.98772264, -0.99591321, -0.99863964]])
    With :class:`aikit.Container` input:
    >>> x = aikit.Container(a=aikit.array([0.0, -1.2]), b=aikit.array([0.4, -0.2]))
    >>> x = aikit.elu(x, out=x)
    >>> print(x)
    {
        a: aikit.array([0., -0.69880581]),
        b: aikit.array([0.40000001, -0.18126924])
    }
    """
    return current_backend(x).elu(x, alpha=alpha, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def hardtanh(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    max_val: float = 1,
    min_val: float = -1,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the hardtanh unit function element-wise.

    Parameters
    ----------
    x
        Input array.
    min_val
        minimum value of the linear region range. Default: -1.
    max_val
        maximum value of the linear region range. Default: 1.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with elu applied element-wise.

    Examples
    --------
    With :class:`aikit.Array` input:
    >>> x = aikit.array([0.39, -0.85])
    >>> y = aikit.hardtanh(x)
    >>> print(y)
    aikit.array([ 0.39, -0.85])
    >>> x = aikit.array([1.5, 0.7, -2.4])
    >>> y = aikit.zeros(3)
    >>> aikit.hardtanh(x, out=y)
    >>> print(y)
    aikit.array([ 1., 0.7, -1.])
    >>> x = aikit.array([[1.1, 2.2, 3.3],
    ...                [-0.4, 0.5, -6.6]])
    >>> aikit.hardtanh(x, out=x)
    >>> print(x)
    aikit.array([[ 1.,  1., 1.],
           [-0.4, 0.5, -1.]])
    With :class:`aikit.Container` input:
    >>> x = aikit.Container(a=aikit.array([0.0, -1.2]), b=aikit.array([0.4, -0.2]))
    >>> x = aikit.hardtanhx, out=x)
    >>> print(x)
    {
        a: aikit.array([0., -1.]),
        b: aikit.array([0.4, -0.2])
    }
    """
    return current_backend(x).hardtanh(x, max_val=max_val, min_val=min_val, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def tanhshrink(
    x: Union[aikit.Array, aikit.NativeArray], /, *, out: Optional[aikit.Array] = None
) -> aikit.Array:
    """Apply the tanhshrink function element-wise.

    Parameters
    ----------
    x
        input array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the tanhshrink activation of each element in ``x``.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([-1.0, 1.0, 2.0])
    >>> y = aikit.tanhshrink(x)
    >>> print(y)
    aikit.array([-0.23840582,  0.23840582,  1.03597236])

    >>> x = aikit.array([-1.0, 1.0, 2.0])
    >>> y = x.tanhshrink()
    >>> print(y)
    aikit.array([-0.23840582,  0.23840582,  1.03597236])


    >>> x = aikit.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = aikit.tanhshrink(x)
    >>> print(y)
    aikit.array([[-0.43827677,  2.80100036,  1.12954807],
                [ 0.76459098,  3.20044947, -5.60000372]])
    """
    return current_backend(x).tanhshrink(x, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def softshrink(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    lambd: float = 0.5,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the softshrink function element-wise.

    Parameters
    ----------
    x
        input array.
    lambd
        the value of the lower bound of the linear region range.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
         an array containing the softshrink activation of each element in ``x``.

    Examples
    --------
    With :class:`aikit.Array` input:
    >>> x = aikit.array([-1.0, 1.0, 2.0])
    >>> y = aikit.softshrink(x)
    >>> print(y)
    aikit.array([-0.5,  0.5,  1.5])

    >>> x = aikit.array([-1.0, 1.0, 2.0])
    >>> y = x.softshrink()
    >>> print(y)
    aikit.array([-0.5,  0.5,  1.5])


    >>> x = aikit.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = aikit.softshrink(x)
    >>> print(y)
    aikit.array([[-0.79999995,  3.29999995,  1.59999991],
       [ 1.20000005,  3.69999981, -6.0999999 ]])
    """
    return current_backend(x).softshrink(x, lambd=lambd, out=out)


def _celu_jax_like(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    fn_original: Optional[Callable] = None,
    alpha: float = 1.0,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    # implementation of max(0, x) for complex numbers
    complex_max = aikit.where(
        (
            aikit.logical_or(
                aikit.real(x) < 0, aikit.logical_and(aikit.real(x) == 0, aikit.imag(x) < 0)
            )
        ),
        aikit.astype(0.0, x.dtype),
        x,
    )

    # implementation of min(0, x) for complex numbers
    complex_min = aikit.where(
        (
            aikit.logical_or(
                aikit.real(x) < 0, aikit.logical_and(aikit.real(x) == 0, aikit.imag(x) < 0)
            )
        ),
        x,
        aikit.astype(0.0, x.dtype),
    )
    return complex_max + alpha * aikit.expm1(complex_min / alpha)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_device
def threshold(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    threshold: float,
    value: float,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the threshold function element-wise.

    Parameters
    ----------
    x
        input array.
    threshold
        The value to threshold at.
    value
        The value to replace with.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the threshold activation of each element in ``x``.

    Examples
    --------
    With :class:`aikit.Array` input:
    >>> x = aikit.array([-1.0, 1.0, 2.0])
    >>> y = aikit.threshold(x,value=0.0, threshold=1.5)
    >>> print(y)
    aikit.array([0., 0., 2.])

    >>> x = aikit.array([-1.0, 1.0, 2.0])
    >>> x.threshold(value=0.0, threshold=1.5)
    >>> print(y)
    aikit.array([0., 0., 2.])


    >>> x = aikit.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = aikit.threshold(x, value=0.0, threshold=1.5)
    >>> print(y)
    aikit.array([[0.        , 3.79999995, 2.0999999 ],
            [1.70000005, 4.19999981, 0.        ]])
    """
    return current_backend(x).threshold(x, threshold=threshold, value=value, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
@handle_complex_input
def celu(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    alpha: float = 1.0,
    complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the Continuously Differentiable Exponential Linear Unit (CELU)
    activation function to each element of the input.

    Parameters
    ----------
    x
        Input array.
    alpha
        The alpha value (negative slope) for the CELU formulation. Default is ``1.0``
    complex_mode
        optional specifier for how to handle complex data types. See
        ``aikit.func_wrapper.handle_complex_input`` for more detail.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The input array with celu applied element-wise.


    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([0.39, -0.85])
    >>> y = aikit.celu(x)
    >>> y
    aikit.array([ 0.39, -0.57])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([0.39, -0.85]), b=aikit.array([1., -0.2]))
    >>> y = aikit.celu(x)
    >>> y
    {
        a: aikit.array([0.38999999, -0.57]),
        b: aikit.array([1., -0.18])
    }
    """
    return current_backend(x).celu(x, alpha=alpha, out=out)


celu.jax_like = _celu_jax_like


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def scaled_tanh(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    alpha: float = 1.7159,
    beta: float = 0.67,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Compute the scaled hyperbolic tangent (tanh) activation.

    The scaled tanh activation function is defined as:
    out = alpha * tanh(beta * x)


    Parameters
    ----------
    x
        input array.
    alpha
        The scaling parameter for the output.
        Determines the amplitude of the tanh function.
        Default: 1.7159
    beta
        The scaling parameter for the input.
        Determines the slope of the tanh function.
        Default: 0.67
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
         The input array after applying the scaled tanh activation.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([22.])
    >>> y = aikit.scaled_tanh(x)
    >>> y
    aikit.array([1.71589994]))

    >>> x = aikit.array([4.0, 7.0])
    >>> y = aikit.scaled_tanh(x, alpha=1.2, beta=5)
    >>> y
    aikit.array([1.20000005, 1.20000005])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([1.2, -1.2]), b=aikit.array([4.4, -2.2]))
    >>> y = aikit.scaled_tanh(x)
    >>> y
    {
        a: aikit.array([1.14324772, -1.14324772]),
        b: aikit.array([1.70648694, -1.54488957])
    }
    >>> x = aikit.Container(a=aikit.array([1.2]), b=aikit.array([4.4]))
    >>> y = aikit.scaled_tanh(x, alpha=0.2, beta=0.5)
    >>> y
    {
    a: aikit.array([0.10740992]),
    b: aikit.array([0.19514863])
    }
    """
    return current_backend(x).scaled_tanh(x, alpha=alpha, beta=beta, out=out)


stanh = scaled_tanh


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def hardshrink(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    lambd: float = 0.5,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Apply the hardshrink function element-wise.

    Parameters
    ----------
    x
        input array.
    lambd
        the value for the Hardshrink formulation.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
         an array containing the hardshrink activation of each element in ``x``.

    Examples
    --------
    With :class:`aikit.Array` input:
    >>> x = aikit.array([-1.0, 1.0, 2.0])
    >>> y = aikit.hardshrink(x)
    >>> print(y)
    aikit.array([-1.,  1.,  2.])
    >>> x = aikit.array([-1.0, 1.0, 2.0])
    >>> y = x.hardshrink()
    >>> print(y)
    aikit.array([-0.5,  0.5,  1.5])
    >>> x = aikit.array([[-1.3, 3.8, 2.1], [1.7, 4.2, -6.6]])
    >>> y = aikit.hardshrink(x)
    >>> print(y)
    aikit.array([[-1.29999995,  3.79999995,  2.0999999 ],
       [ 1.70000005,  4.19999981, -6.5999999 ]])
    """
    return current_backend(x).hardshrink(x, lambd=lambd, out=out)
