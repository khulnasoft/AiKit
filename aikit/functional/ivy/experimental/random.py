# local
from typing import Optional, Union, Sequence
import aikit
from aikit.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    inputs_to_native_shapes,
    handle_nestable,
    infer_dtype,
    handle_device,
    handle_backend_invalid,
)
from aikit.utils.exceptions import handle_exceptions


# dirichlet
@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device
def dirichlet(
    alpha: Union[aikit.Array, aikit.NativeArray, float, Sequence[float]],
    /,
    *,
    size: Optional[Union[aikit.Shape, aikit.NativeShape]] = None,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    seed: Optional[int] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Draw size samples of dimension k from a Dirichlet distribution. A
    Dirichlet- distributed random variable can be seen as a multivariate
    generalization of a Beta distribution. The Dirichlet distribution is a
    conjugate prior of a multinomial distribution in Bayesian inference.

    Parameters
    ----------
    alpha
        Sequence of floats of length k
    size
        optional int or tuple of ints, Output shape. If the given shape is,
        e.g., (m, n), then m * n * k samples are drawn. Default is None,
        in which case a vector of length k is returned.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data
        type will be the default floating-point data type. Default ``None``
    seed
        A python integer. Used to create a random seed distribution
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        The drawn samples, of shape (size, k).

    Examples
    --------
    >>> alpha = [1.0, 2.0, 3.0]
    >>> aikit.dirichlet(alpha)
    aikit.array([0.10598304, 0.21537054, 0.67864642])

    >>> alpha = [1.0, 2.0, 3.0]
    >>> aikit.dirichlet(alpha, size = (2,3))
    aikit.array([[[0.48006698, 0.07472073, 0.44521229],
        [0.55479872, 0.05426367, 0.39093761],
        [0.19531053, 0.51675832, 0.28793114]],

       [[0.12315625, 0.29823365, 0.5786101 ],
        [0.15564976, 0.50542368, 0.33892656],
        [0.1325352 , 0.44439589, 0.42306891]]])
    """
    return aikit.current_backend().dirichlet(
        alpha,
        size=size,
        dtype=dtype,
        seed=seed,
        out=out,
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@inputs_to_native_shapes
@to_native_arrays_and_back
@handle_device
def beta(
    a: Union[float, aikit.NativeArray, aikit.Array],
    b: Union[float, aikit.NativeArray, aikit.Array],
    /,
    *,
    shape: Optional[Union[aikit.Shape, aikit.NativeShape]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    seed: Optional[int] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return an array filled with random values sampled from a beta
    distribution.

    Parameters
    ----------
    a
        Alpha parameter of the beta distribution.
    b
        Beta parameter of the beta distribution.
    shape
        If the given shape is, e.g ``(m, n, k)``, then ``m * n * k`` samples are drawn
        Can only be specified when ``mean`` and ``std`` are numeric values, else
        exception will be raised.
        Default is ``None``, where a single value is returned.
    device
        device on which to create the array. 'cuda:0',
        'cuda:1', 'cpu' etc. (Default value = None).
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data
        type will be the default floating point data type. Default ``None``
    seed
        A python integer. Used to create a random seed distribution
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        Returns an array with the given shape filled with random values sampled from
        a beta distribution.
    """
    return aikit.current_backend().beta(
        a, b, shape=shape, device=device, dtype=dtype, seed=seed, out=out
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@inputs_to_native_shapes
@to_native_arrays_and_back
@handle_device
def gamma(
    alpha: Union[float, aikit.NativeArray, aikit.Array],
    beta: Union[float, aikit.NativeArray, aikit.Array],
    /,
    *,
    shape: Optional[Union[float, aikit.NativeArray, aikit.Array]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    seed: Optional[int] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return an array filled with random values sampled from a gamma
    distribution.

    Parameters
    ----------
    alpha
        Alpha parameter of the gamma distribution.
    beta
        Beta parameter of the gamma distribution.
    shape
        Shape parameter of the gamma distribution.
    device
        device on which to create the array. 'cuda:0',
        'cuda:1', 'cpu' etc. (Default value = None).
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data
        type will be the default floating point data type. Default ``None``
    seed
        A python integer. Used to create a random seed distribution
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        Returns an array filled with random values sampled from a gamma distribution.
    """
    return aikit.current_backend().gamma(
        alpha, beta, shape=shape, device=device, dtype=dtype, seed=seed, out=out
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@inputs_to_native_shapes
@to_native_arrays_and_back
@infer_dtype
@handle_device
def poisson(
    lam: Union[float, aikit.Array, aikit.NativeArray],
    *,
    shape: Optional[Union[aikit.Shape, aikit.NativeShape]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    seed: Optional[int] = None,
    fill_value: Optional[Union[int, float]] = 0,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Draws samples from a poisson distribution.

    Parameters
    ----------
    lam
        Rate parameter(s) describing the poisson distribution(s) to sample.
        It must have a shape that is broadcastable to the requested shape.
    shape
        If the given shape is, e.g '(m, n, k)', then 'm * n * k' samples are drawn.
        (Default value = 'None', where 'aikit.shape(lam)' samples are drawn)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        (Default value = None).
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data
        type will be the default floating-point data type. Default ``None``
    seed
        A python integer. Used to create a random seed distribution.
    fill_value
        if lam is negative, fill the output array with this value
        on that specific dimension.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
     ret
        Drawn samples from the poisson distribution

    Examples
    --------
    >>> lam = [1.0, 2.0, 3.0]
    >>> aikit.poisson(lam)
    aikit.array([1., 4., 4.])

    >>> lam = [1.0, 2.0, 3.0]
    >>> aikit.poisson(lam, shape = (2,3))
    aikit.array([[0., 2., 2.],
               [1., 2., 3.]])
    """
    return aikit.current_backend(lam).poisson(
        lam,
        shape=shape,
        device=device,
        dtype=dtype,
        seed=seed,
        fill_value=fill_value,
        out=out,
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@inputs_to_native_shapes
@to_native_arrays_and_back
@infer_dtype
@handle_device
def bernoulli(
    probs: Union[float, aikit.Array, aikit.NativeArray],
    *,
    logits: Optional[Union[float, aikit.Array, aikit.NativeArray]] = None,
    shape: Optional[Union[aikit.Shape, aikit.NativeShape]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    seed: Optional[int] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Draws samples from Bernoulli distribution parameterized by probs or
    logits (but not both)

    Parameters
    ----------
    logits
        An N-D Array representing the log-odds of a 1 event.
        Each entry in the Array parameterizes an independent Bernoulli
        distribution where the probability of an event is sigmoid
        (logits). Only one of logits or probs should be passed in.
    probs
        An N-D Array representing the probability of a 1 event.
        Each entry in the Array parameterizes an independent Bernoulli
        distribution. Only one of logits or probs should be passed in
    shape
        If the given shape is, e.g '(m, n, k)', then 'm * n * k' samples are drawn.
        (Default value = 'None', where 'aikit.shape(logits)' samples are drawn)
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        (Default value = None).
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data
        type will be the default floating-point data type. Default ``None``
    seed
        A python integer. Used to create a random seed distribution
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Drawn samples from the Bernoulli distribution
    """
    return aikit.current_backend(probs).bernoulli(
        probs,
        logits=logits,
        shape=shape,
        device=device,
        dtype=dtype,
        seed=seed,
        out=out,
    )
