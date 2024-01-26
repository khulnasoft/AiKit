"""Collection of random Ivy functions."""

# global
from typing import Optional, Union

# local
import aikit
from aikit.func_wrapper import (
    handle_array_function,
    infer_dtype,
    handle_out_argument,
    to_native_arrays_and_back,
    inputs_to_native_shapes,
    handle_nestable,
    handle_device,
    handle_backend_invalid,
)
from aikit.utils.backend import backend_stack
from aikit.utils.exceptions import handle_exceptions


# Helpers #
# ------- #


def _check_bounds_and_get_shape(low, high, shape):
    if shape is not None:
        aikit.utils.assertions.check_all_or_any_fn(
            low,
            high,
            fn=lambda x: isinstance(x, (int, float)),
            type="all",
            message="low and high bounds must be numerics when shape is specified",
        )
        return aikit.Shape(shape)

    valid_types = (aikit.Array,)
    if len(backend_stack) == 0:
        valid_types += (aikit.current_backend().NativeArray,)
    else:
        valid_types += (aikit.NativeArray,)
    if isinstance(low, valid_types):
        if isinstance(high, valid_types):
            aikit.utils.assertions.check_equal(
                aikit.shape(low), aikit.shape(high), as_array=False
            )
        return aikit.shape(low)
    if isinstance(high, valid_types):
        return aikit.shape(high)
    return aikit.Shape(())


def _randint_check_dtype_and_bound(low, high, dtype):
    aikit.utils.assertions.check_all_or_any_fn(
        low,
        high,
        dtype,
        fn=aikit.is_uint_dtype,
        type="any",
        limit=[0],
        message="randint cannot take arguments of type uint",
    )
    aikit.utils.assertions.check_all_or_any_fn(
        low,
        high,
        dtype,
        fn=aikit.is_float_dtype,
        type="any",
        limit=[0],
        message="randint cannot take arguments of type float",
    )
    aikit.utils.assertions.check_less(low, high)


def _check_valid_scale(std):
    aikit.utils.assertions.check_greater(
        std, 0, allow_equal=True, message="std must be non-negative"
    )


def _check_shapes_broadcastable(out, inp):
    if out is not None:
        aikit.utils.assertions.check_shapes_broadcastable(out, inp)


# Extra #
# ------#


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@inputs_to_native_shapes
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device
def random_uniform(
    *,
    low: Union[float, aikit.NativeArray, aikit.Array] = 0.0,
    high: Union[float, aikit.NativeArray, aikit.Array] = 1.0,
    shape: Optional[Union[aikit.Array, aikit.Shape, aikit.NativeShape]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    seed: Optional[int] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Draws samples from a uniform distribution. Samples are uniformly
    distributed over the half-open interval ``[low, high)`` (includes ``low``,
    but excludes ``high``). In other words, any value within the given interval
    is equally likely to be drawn by uniform.

    Parameters
    ----------
    low
        Lower boundary of the output interval. All values generated will be greater than
        or equal to ``low``. If array, must have same shape as ``high``.
    high
        Upper boundary of the output interval. All the values generated will be less
        than ``high``. If array, must have same shape as ``low``.
    shape
        If the given shape is, e.g ``(m, n, k)``, then ``m * n * k`` samples are drawn.
        Can only be specified when ``low`` and ``high`` are numeric values, else
        exception will be raised.
        Default is ``None``, where a single value is returned.
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
        Drawn samples from the parameterized uniform distribution.

    Examples
    --------
    >>> aikit.random_uniform()
    aikit.array(0.26431865)

    >>> aikit.random_uniform(shape=3)
    aikit.array([0.475, 0.878, 0.861])

    >>> aikit.random_uniform(shape=(2,3))
    aikit.array([[0.929 , 0.545 , 0.789 ],
               [0.519 , 0.0435, 0.381 ]])

    >>> aikit.random_uniform(low=3.0, high=6.0)
    aikit.array(3.4608004)

    >>> aikit.random_uniform(low=1.0, high=2.0, shape=(2,1))
    aikit.array([[1.85],
               [1.81]])

    >>> z = aikit.zeros(())
    >>> aikit.random_uniform(low=1.0, high=2.0, out=z)
    aikit.array(1.8458502)

    >>> aikit.random_uniform(low=1.0, high=2.0, shape=(2,2), device='cpu')
    aikit.array([[1.81, 1.8 ],
               [1.32, 1.43]])

    >>> aikit.random_uniform(low=1.0, high=2.0, shape=(2,2), device='cpu',
    ...                    dtype='int32')
    aikit.array([[1, 1],
               [1, 1]])

    >>> z = aikit.zeros((1,2))
    >>> aikit.random_uniform(low=1.0, high=2.0, shape=(1,2), device='cpu',
    ...                    dtype='float64', out=z)
    aikit.array([[1.34, 1.02]])

    >>> x = aikit.array([4.8, 5.6])
    >>> y = aikit.array([9.8, 7.4])
    >>> aikit.random_uniform(low=x, high=y)
    aikit.array([0.475, 0.878])

    >>> z = aikit.zeros((2,))
    >>> aikit.random_uniform(low=x, high=y, out=z, seed=42)
    aikit.array([6.67270088, 7.31128597])

    >>> aikit.random_uniform(low=x, high=y, device='cpu')
    aikit.array([6.88, 6.75])

    >>> aikit.random_uniform(low=x, high=y, device='cpu', dtype='float64')
    aikit.array([8.62, 6.47])

    >>> z = aikit.zeros((2,))
    >>> aikit.random_uniform(low=x, high=y, device='cpu', dtype='float64', out=z)
    aikit.array([5. , 7.3])
    """
    return aikit.current_backend().random_uniform(
        low=low, high=high, shape=shape, device=device, dtype=dtype, out=out, seed=seed
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@inputs_to_native_shapes
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device
def random_normal(
    *,
    mean: Union[float, aikit.NativeArray, aikit.Array] = 0.0,
    std: Union[float, aikit.NativeArray, aikit.Array] = 1.0,
    shape: Optional[Union[aikit.Shape, aikit.NativeShape]] = None,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    seed: Optional[int] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Draws samples from a normal distribution.

    Parameters
    ----------
    mean
        The mean of the normal distribution to sample from. Default is ``0.0``.
    std
        The standard deviation of the normal distribution to sample from.
        Must be non-negative. Default is ``1.0``.
    shape
        If the given shape is, e.g ``(m, n, k)``, then ``m * n * k`` samples are drawn.
        Can only be specified when ``mean`` and ``std`` are numeric values, else
        exception will be raised.
        Default is ``None``, where a single value is returned.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data
        type will be the default floating-point data type. Default ``None``
    seed
        A python integer. Used to create a random seed distribution
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        (Default value = None).
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
     ret
        Drawn samples from the parameterized normal distribution.

    Examples
    --------
    >>> aikit.random_normal()
    aikit.array(-0.22346112)

    >>> aikit.random_normal(shape=3)
    aikit.array([-0.73  ,  0.0922, -0.515 ])

    >>> aikit.random_normal(shape=(2, 3), seed=42)
    aikit.array([[ 0.49671414, -0.1382643 ,  0.64768857],
           [ 1.5230298 , -0.23415337, -0.23413695]])

    >>> aikit.random_normal(mean=3.0, std=6.0)
    aikit.array(4.9213753)

    >>> aikit.random_normal(mean=1.0, std=2.0, shape=(2,1))
    aikit.array([[2.19],
               [2.78]])

    >>> z = aikit.zeros(())
    >>> aikit.random_normal(mean=1.0, std=2.0, out=z)
    aikit.array(0.12818667)

    >>> aikit.random_normal(mean=1.0, std=2.0, shape=(2,2), device='cpu')
    aikit.array([[ 2.91 ,  1.3  ],
               [ 3.37 , -0.799]])

    >>> aikit.random_normal(mean=1.0, std=2.0, shape=(2,2), device='cpu',
    ...                   dtype='int32')
    aikit.array([[ 0, -1],
               [ 0,  3]])

    >>> z = aikit.zeros((1,2))
    >>> aikit.random_normal(mean=1.0, std=2.0, shape=(1,2), device='cpu',
    ...                   dtype='float64', out=z)
    aikit.array([[-2.01, -1.95]])

    >>> x = aikit.array([4.8, 5.6])
    >>> y = aikit.array([9.8, 7.4])
    >>> aikit.random_normal(mean=x, std=y)
    aikit.array([ 4.43 , -0.469])

    >>> z = aikit.zeros((2,))
    >>> aikit.random_normal(mean=x, std=y, out=z)
    aikit.array([0.287, 8.55 ])

    >>> aikit.random_normal(mean=x, std=y, device='cpu')
    aikit.array([18.9, 15.2])

    >>> aikit.random_normal(mean=x, std=y, device='cpu', dtype='float64')
    aikit.array([-4.1   , -0.0366])

    >>> z = aikit.zeros((2,))
    >>> aikit.random_normal(mean=x, std=y, device='cpu', dtype='float64', out=z)
    aikit.array([12.4, 11. ])
    """
    return aikit.current_backend().random_normal(
        mean=mean, std=std, shape=shape, dtype=dtype, seed=seed, device=device, out=out
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def multinomial(
    population_size: int,
    num_samples: int,
    /,
    *,
    batch_size: int = 1,
    probs: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
    replace: bool = True,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    seed: Optional[int] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Draws samples from a multinomial distribution. Specifically, returns a
    tensor where each row contains num_samples indices sampled from the
    multinomial probability distribution located in the corresponding row of
    tensor input.

    Parameters
    ----------
    population_size
        The size of the population from which to draw samples.
    num_samples
        Number of independent samples to draw from the population.
    batch_size
        Number of tensors to generate. Default is 1.
    probs
        The unnormalized probabilities for all elements in population,
        default is uniform *[batch_shape, population_size]*
    replace
        Whether to replace samples once they've been drawn. Default is ``True``.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
        (Default value = None)
    seed
        A python integer. Used to create a random seed distribution
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Drawn samples indices from the multinomial distribution.

    Examples
    --------
    >>> y = aikit.multinomial(10, 5)
    >>> print(y)
    aikit.array([[1, 8, 7, 8, 3]])

    >>> y = aikit.multinomial(10, 5, batch_size=2, seed=42)
    >>> print(y)
    aikit.array([[3, 9, 7, 5, 1],
           [1, 0, 8, 6, 7]])

    >>> y = aikit.multinomial(10, 5, replace=False)
    >>> print(y)
    aikit.array([[2, 6, 4, 7, 0]])

    With :class:`aikit.Array` input:

    >>> y = aikit.multinomial(10, 5, probs=aikit.array([1/10]*10))
    >>> print(y)
    aikit.array([5, 2, 7, 6, 9])

    >>> y = aikit.multinomial(7, 5, batch_size=2, probs=aikit.array([[1/7]*7, [1/7]*7]))
    >>> print(y)
    aikit.array([[0, 4, 3, 4, 5], [1, 1, 0, 3, 2]])

    >>> y = aikit.multinomial(7, 5, batch_size=2, probs=aikit.array([[1/7]*7, [1/7]*7]),
    ...                     replace=False)
    >>> print(y)
    aikit.array([[2, 6, 1, 0, 3], [1, 0, 2, 5, 6]])

    With :class:`aikit.NativeArray` input:

    >>> y = aikit.multinomial(10, 5, probs=aikit.native_array([1/10]*10))
    >>> print(y)
    aikit.array([5, 7, 4, 2, 1])

    >>> y = aikit.multinomial(10, 5, batch_size=2,
    ...                     probs=aikit.native_array([[1/10]*10, [1/10]*10]))
    >>> print(y)
    aikit.array([[8, 0, 4, 1, 7], [2, 3, 4, 9, 3]])

    >>> y = aikit.multinomial(10, 5, batch_size=2,
    ...                     probs=aikit.native_array([[1/10]*10, [1/10]*10]),
    ...                     replace=False)
    >>> print(y)
    aikit.array([[0, 2, 6, 9, 1], [6, 7, 2, 4, 3]])
    """
    return aikit.current_backend().multinomial(
        population_size,
        num_samples,
        batch_size=batch_size,
        probs=probs,
        replace=replace,
        device=device,
        seed=seed,
        out=out,
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@inputs_to_native_shapes
@to_native_arrays_and_back
@handle_array_function
@handle_device
def randint(
    low: Union[int, aikit.NativeArray, aikit.Array],
    high: Union[int, aikit.NativeArray, aikit.Array],
    /,
    *,
    shape: Optional[Union[aikit.Shape, aikit.NativeShape]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    seed: Optional[int] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return an array filled with random integers generated uniformly between
    low (inclusive) and high (exclusive).

    Parameters
    ----------
    low
        Lowest integer that can be drawn from the distribution.
    high
        One above the highest integer that can be drawn from the distribution.
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
        type will be the default integer data type. Default ``None``
    seed
        A python integer. Used to create a random seed distribution
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        Returns an array with the given shape filled with integers from
        the uniform distribution in the “half-open” interval [low, high)

    Examples
    --------
    >>> y = aikit.randint(0, 9, shape=(1,1))
    >>> print(y)
    aikit.array([[5]])

    >>> y = aikit.randint(2, 20, shape=(2, 2), device='cpu', seed=42)
    >>> print(y)
    aikit.array([[ 8, 16],
               [12,  9]])

    >>> x = aikit.array([1, 2, 3])
    >>> aikit.randint(0, 10, shape=(3,), out=x)
    >>> print(x)
    aikit.array([2, 6, 7])

    >>> y = aikit.zeros((3, 3))
    >>> aikit.randint(3, 15, shape=(3, 3), device='cpu', out=y)
    >>> print(y)
    aikit.array([[ 7,  7,  5],
               [12,  8,  8],
               [ 8, 11,  3]])
    """
    return aikit.current_backend().randint(
        low, high, shape=shape, device=device, dtype=dtype, seed=seed, out=out
    )


@handle_exceptions
@handle_nestable
def seed(*, seed_value: int = 0) -> None:
    """Set the seed for random number generation.

    Parameters
    ----------
    seed_value
        Seed for random number generation, must be a positive integer.
        (Default value = 0)

    Examples
    --------
    >>> aikit.seed(seed_value=42)
    """
    return aikit.current_backend().seed(seed_value=seed_value)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def shuffle(
    x: Union[aikit.Array, aikit.NativeArray],
    axis: Optional[int] = 0,
    /,
    *,
    seed: Optional[int] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Shuffles the given array along a given axis.

    Parameters
    ----------
    x
        Input array. Should have a numeric data type.
    axis
        The axis which x is shuffled along. Default is 0.
    seed
        A python integer. Used to create a random seed distribution
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array object, shuffled along the specified axis.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([1, 2, 3, 4, 5])
    >>> y = aikit.shuffle(x)
    >>> print(y)
    aikit.array([2, 1, 4, 3, 5])

    >>> x = aikit.array([1, 3, 5, 7])
    >>> y = aikit.shuffle(x, seed=394)
    >>> print(y)
    aikit.array([3, 1, 5, 7])

    >>> x = aikit.array([1, 0, 5])
    >>> y = aikit.array([0, 0, 0])
    >>> aikit.shuffle(x, seed=394, out=y)
    >>> print(y)
    aikit.array([0, 1, 5])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([5, 2, 9]),
    ...                   b=aikit.array([7, 1, 6]))
    >>> y = aikit.shuffle(x)
    >>> print(y)
    {
        a: aikit.array([5, 9, 2]),
        b: aikit.array([6, 1, 7])
    }

    >>> x = aikit.Container(a=aikit.array([7, 4, 5]),
    ...                   b=aikit.array([9, 8, 2]))
    >>> y = aikit.Container(a=aikit.array([0, 0, 0]),
    ...                   b=aikit.array([0, 0, 0]))
    >>> aikit.shuffle(x, seed=17, out=y)
    >>> print(y)
    {
        a: aikit.array([7, 5, 4]),
        b: aikit.array([9, 2, 8])
    }

    >>> x = aikit.Container(a=aikit.array([8, 2, 5]),
    ...                   b=aikit.array([3, 9, 0]))
    >>> aikit.shuffle(x, seed=17, out=x)
    >>> print(x)
    {
        a: aikit.array([2, 8, 5]),
        b: aikit.array([3, 0, 9])
    }
    """
    return aikit.current_backend(x).shuffle(x, axis, seed=seed, out=out)
