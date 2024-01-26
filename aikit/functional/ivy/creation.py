# global
from __future__ import annotations
import functools
from numbers import Number
from typing import (
    Union,
    Tuple,
    Optional,
    List,
    Sequence,
    Callable,
    Protocol,
    TypeVar,
    Iterable,
)
import numpy as np

# local
import aikit
from aikit import to_aikit
from aikit.utils.exceptions import handle_exceptions
from aikit.utils.backend import current_backend
from aikit.func_wrapper import (
    handle_array_function,
    infer_dtype,
    handle_out_argument,
    outputs_to_aikit_arrays,
    inputs_to_native_arrays,
    inputs_to_native_shapes,
    to_native_arrays_and_back,
    handle_nestable,
    handle_array_like_without_promotion,
    handle_device,
    handle_backend_invalid,
    temp_asarray_wrapper,
)

# Helpers #
# --------#


def _asarray_handle_nestable(fn: Callable) -> Callable:
    fn_name = fn.__name__

    @functools.wraps(fn)
    def _asarray_handle_nestable_wrapper(*args, **kwargs):
        """Call `fn` with the *nestable* property of the function correctly
        handled. This means mapping the function to the container leaves if any
        containers are passed in the input.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with the nestable property handled correctly.
        """
        # This decorator should only be applied to aikit.asarray, so we know where
        # the container must be if there is one.
        cont_fn = getattr(aikit.Container, f"static_{fn_name}")
        if isinstance(args[0], aikit.Container):
            return cont_fn(*args, **kwargs)

        # if the passed arguments does not contain a container, the function using
        # the passed arguments, returning an aikit or a native array.
        return fn(*args, **kwargs)

    _asarray_handle_nestable_wrapper.handle_nestable = True
    return _asarray_handle_nestable_wrapper


def _aikit_to_native(x):
    # checks the first element of the leaf list and
    # converts it to a native array if it is an aikit array
    # assumes that either all elements in a leaf list are aikit arrays
    # or none of them are
    if isinstance(x, (list, tuple)) and len(x) != 0 and isinstance(x[0], (list, tuple)):
        for i, item in enumerate(x):
            x = list(x) if isinstance(x, tuple) else x
            x[i] = _aikit_to_native(item)
    elif (isinstance(x, (list, tuple)) and len(x) > 0) and aikit.is_aikit_array(x[0]):
        x = aikit.to_native(x, nested=True)
    elif aikit.is_aikit_array(x):
        x = aikit.to_native(x)
    return x


def _shape_to_native(x: Iterable) -> Tuple[int]:
    # checks the first element of the leaf list and
    # converts it to a native array if it is an aikit array

    # This function is to be used with the nested_map function
    # it was a lambda function before but was replaced with the defined function below
    def nested_map_shape_fn(x: Iterable) -> List:
        return x.shape if isinstance(x, aikit.Shape) else x

    if isinstance(x, (list, tuple)) and len(x) != 0 and isinstance(x[0], (list, tuple)):
        for i, item in enumerate(x):
            x = list(x) if isinstance(x, tuple) else x
            x[i] = _shape_to_native(item)

    else:
        if (isinstance(x, (list, tuple)) and len(x) > 0) and (
            isinstance(x[0], aikit.Shape) and aikit.array_mode
        ):
            x = aikit.nested_map(x, nested_map_shape_fn)
        elif isinstance(x, aikit.Shape) and aikit.array_mode:
            x = x.shape

    return x


def _flatten_nest(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from _flatten_nest(x)
        else:
            yield x


def _remove_np_bfloat16(obj):
    # unlike other frameworks, torch and paddle do not support creating tensors
    # from numpy arrays that have bfloat16 dtype using any extension because
    # bfloat16 in not supported natively by numpy (as of version <=1.25)
    if isinstance(obj, np.ndarray) and obj.dtype.name == "bfloat16":
        return obj.tolist()
    return obj


def _asarray_to_native_arrays_and_back(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _asarray_to_native_arrays_and_back_wrapper(*args, dtype=None, **kwargs):
        """Wrap `fn` so that input arrays are all converted to
        `aikit.NativeArray` instances and return arrays are all converted to
        `aikit.Array` instances.

        This wrapper is specifically for the backend implementations of
        asarray.

        It assumes either all the elements in a leaf list are aikit arrays
        or none of them are. It checks the first element of all the leaf
        list. If it is an aikit array, it converts all the elements in the
        leaf list to native otherwise it skips that leaf list.
        """
        new_arg = _aikit_to_native(args[0])
        new_args = (new_arg,) + args[1:]
        if dtype is not None:
            dtype = aikit.default_dtype(dtype=dtype, as_native=True)
        return to_aikit(fn(*new_args, dtype=dtype, **kwargs))

    return _asarray_to_native_arrays_and_back_wrapper


def _asarray_infer_dtype(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _asarray_infer_dtype_wrapper(*args, dtype=None, **kwargs):
        """Determine the correct `dtype`, and then calls the function with the
        `dtype` passed explicitly. This wrapper is specifically for the backend
        implementations of asarray.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        dtype
            The dtype for the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with `dtype` passed explicitly.
        """

        def _infer_dtype(obj):
            if isinstance(obj, aikit.NativeShape):
                obj = list(obj)
            if hasattr(obj, "dtype"):
                return obj.dtype.name if isinstance(obj, np.ndarray) else obj.dtype
            else:
                return aikit.default_dtype(item=obj)

        if not aikit.exists(dtype):
            arr = args[0]
            # get default dtypes for all elements
            dtype_list = [aikit.nested_map(lambda x: _infer_dtype(x), arr, shallow=False)]
            # flatten the nested structure
            dtype_list = _flatten_nest(dtype_list)
            # keep unique dtypes
            dtype_list = list(set(dtype_list))
            if len(dtype_list) != 0:  # handle the case of empty input
                # promote all dtypes to a single dtype
                dtype = dtype_list[0]
                # we disable precise mode to avoid wider than necessary casting
                # that might result from the mixing of int32 and float32
                with aikit.PreciseMode(False):
                    for dt in dtype_list[1:]:
                        dtype = aikit.promote_types(dtype, dt)
            else:
                dtype = aikit.default_float_dtype()
            dtype = aikit.as_native_dtype(dtype)
        # call the function with dtype provided explicitly
        return fn(*args, dtype=dtype, **kwargs)

    _asarray_infer_dtype_wrapper.infer_dtype = True
    return _asarray_infer_dtype_wrapper


def _asarray_infer_device(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _asarray_infer_device_wrapper(*args, device=None, **kwargs):
        """Determine the correct `device`, and then calls the function with the
        `device` passed explicitly. This wrapper is specifically for the
        backend implementations of asarray.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        device
            The device for the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, with `device` passed explicitly.
        """
        if isinstance(args[0], list):
            return fn(
                *args, device=aikit.default_device(device, as_native=True), **kwargs
            )

        # find the first array argument, if required
        arr = None if aikit.exists(device) else args[0]
        # infer the correct device
        device = aikit.default_device(device, item=arr, as_native=True)
        # call the function with device provided explicitly
        return fn(*args, device=device, **kwargs)

    _asarray_infer_device_wrapper.infer_device = True
    return _asarray_infer_device_wrapper


def _asarray_inputs_to_native_shapes(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_native_shapes(*args, **kwargs):
        new_arg = _shape_to_native(args[0])
        new_args = (new_arg,) + args[1:]
        return fn(*new_args, **kwargs)

    _inputs_to_native_shapes.inputs_to_native_shapes = True
    return _inputs_to_native_shapes


# Type hints #
# -----------#

SupportsBufferProtocol = TypeVar("SupportsBufferProtocol")
_T_co = TypeVar("_T_co", covariant=True)


class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> Union[_T_co, NestedSequence[_T_co]]: ...

    def __len__(self, /) -> int: ...


# Array API Standard #
# -------------------#


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@outputs_to_aikit_arrays
@handle_array_function
@handle_device
def arange(
    start: Number,
    /,
    stop: Optional[Number] = None,
    step: Number = 1,
    *,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return evenly spaced values within a given interval, with the spacing
    being specified.

    Values are generated within the half-open interval [start, stop) (in other words,
    the interval including start but excluding stop). For integer arguments the function
    is equivalent to the Python built-in range function, but returns an array in the
    chosen ml_framework rather than a list.

    See :math:`linspace` for a certain number of evenly spaced values in an interval.

    Parameters
    ----------
    start
        if stop is specified, the start of interval (inclusive); otherwise, the end of
        the interval (exclusive). If stop is not specified, the default starting value
        is 0.
    stop
        the end of the interval. Default: ``None``.
    step
        the distance between two adjacent elements (out[i+1] - out[i]). Must not be 0;
        may be negative, this results in an empty array if stop >= start. Default: 1.
    dtype
        output array data type. If dtype is None, the output array data type must be
        inferred from start, stop and step. If those are all integers, the output array
        dtype must be the default integer dtype; if one or more have type float, then
        the output array dtype must be the default floating-point data type. Default:
        None.
    device
        device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        a one-dimensional array containing evenly spaced values. The length of the
        output array must be ceil((stop-start)/step) if stop - start and step have the
        same sign, and length 0 otherwise.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.arange.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    >>> stop = 5
    >>> x = aikit.arange(stop)
    >>> print(x)
    aikit.array([0, 1, 2, 3, 4])

    >>> start = 1
    >>> stop = 5
    >>> x = aikit.arange(start, stop)
    >>> print(x)
    aikit.array([1, 2, 3, 4])

    >>> start = 1
    >>> stop = 10
    >>> step = 2
    >>> x = aikit.arange(start, stop, step)
    >>> print(x)
    aikit.array([1, 3, 5, 7, 9])

    >>> start = 1
    >>> stop = 10
    >>> step = 2
    >>> dtype = "float64"
    >>> device = "cpu"
    >>> x = aikit.arange(start, stop, step, dtype=dtype, device=device)
    >>> print(x, x.dtype, x.device)
    aikit.array([1., 3., 5., 7., 9.]) float64 cpu
    """
    return current_backend().arange(
        start, stop, step, dtype=dtype, device=device, out=out
    )


@temp_asarray_wrapper
@handle_backend_invalid
@handle_array_like_without_promotion
@handle_out_argument
@handle_array_function
@handle_device
def asarray(
    obj: Union[
        aikit.Array,
        aikit.NativeArray,
        aikit.Shape,
        aikit.NativeShape,
        bool,
        int,
        float,
        NestedSequence,
        SupportsBufferProtocol,
        np.ndarray,
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Convert the input to an array.

    Parameters
    ----------
    obj
        input data, in any form that can be converted to an array. This includes lists,
        lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
    copy
        boolean, indicating whether or not to copy the input. Default: ``None``.
    dtype
       output array data type. If ``dtype`` is ``None``, the output array data type must
       be the default floating-point data type. Default  ``None``.
    device
       device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array interpretation of x.

    Examples
    --------
    With list of lists as input:

    >>> aikit.asarray([[1,2],[3,4]])
    aikit.array([[1, 2],
               [3, 4]])

    With tuple of lists as input:

    >>> aikit.asarray(([1.4,5.6,5.5],[3.1,9.1,7.5]))
    aikit.array([[1.39999998, 5.5999999 , 5.5       ],
               [3.0999999 , 9.10000038, 7.5       ]])

    With ndarray as input:

    >>> x = aikit.np.ndarray(shape=(2,2), order='C')
    >>> aikit.asarray(x)
    aikit.array([[6.90786433e-310, 6.90786433e-310],
               [6.90786433e-310, 6.90786433e-310]])

    With :class:`aikit.Container` as input:

    >>> x = aikit.Container(a = [(1,2),(3,4),(5,6)], b = ((1,2,3),(4,5,6)))
    >>> aikit.asarray(x)
    {
        a: aikit.array([[1, 2],[3, 4], [5, 6]]),
        b: aikit.array([[1, 2, 3],
                   [4, 5, 6]])
    }

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.asarray.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.
    """
    return current_backend().asarray(
        obj, copy=copy, dtype=dtype, device=device, out=out
    )


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_shapes
@outputs_to_aikit_arrays
@handle_array_function
@infer_dtype
@handle_device
def zeros(
    shape: Union[aikit.Shape, aikit.NativeShape],
    *,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return a new array having a specified ``shape`` and filled with zeros.

    Parameters
    ----------
    shape
       output array shape.
    dtype
       output array data type. If ``dtype`` is ``None``, the output array data type must
       be the default floating-point data type. Default  ``None``.
    device
       device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing zeros.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.zeros.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.NativeShape` input:
    >>> shape = (3, 5)
    >>> x = aikit.zeros(shape)
    >>> print(x)
    aikit.array([[0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.]])

    >>> x = aikit.zeros(5)
    >>> print(x)
    aikit.array([0., 0., 0., 0., 0.])
    """
    return current_backend().zeros(shape, dtype=dtype, device=device, out=out)


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_shapes
@outputs_to_aikit_arrays
@handle_array_function
@infer_dtype
@handle_device
def ones(
    shape: Union[aikit.Shape, aikit.NativeShape],
    *,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return a new array having a specified ``shape`` and filled with ones.

    .. note::

        An output array having a complex floating-point data type must contain complex
        numbers having a real component equal to one and an imaginary component equal to
        zero (i.e., ``1 + 0j``).

    Parameters
    ----------
    shape
        output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be the default floating-point data type. Default  ``None``.
    device
        device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing ones.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.ones.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Shape` input:

    >>> shape = (2,2)
    >>> x = aikit.ones(shape)
    >>> print(x)
    aikit.array([[1., 1.],
           [1., 1.]])

    With :class:`aikit.Dtype` input:

    >>> shape = (3,2)
    >>> d_type = aikit.int64
    >>> y = aikit.ones(shape, dtype=d_type)
    >>> print(y)
    aikit.array([[1, 1],
           [1, 1],
           [1, 1]])

    With :class:`aikit.Device` input:

    >>> shape = (3,2)
    >>> y = aikit.ones(shape, device="cpu")
    >>> print(y)
    aikit.array([[1., 1.],
           [1., 1.],
           [1., 1.]])

    With :class:`aikit.Array` input:

    >>> shape = (1, 5, 2)
    >>> x = aikit.zeros(shape)
    >>> aikit.ones(shape, out=x)
    >>> print(x)
    aikit.array([[[1., 1.],
            [1., 1.],
            [1., 1.],
            [1., 1.],
            [1., 1.]]])
    """
    return current_backend().ones(shape, dtype=dtype, device=device, out=out)


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device
def full_like(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    fill_value: Number,
    *,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return a new array filled with ``fill_value`` and having the same
    ``shape`` as an input array ``x`` .

    Parameters
    ----------
    x
        input array from which to derive the output array shape.
    fill_value
        Scalar fill value
    dtype
        output array data type. If ``dtype`` is `None`, the output array data type must
        be inferred from ``x``. Default: ``None``.
    device
        device on which to place the created array. If ``device`` is ``None``, the
        output array device must be inferred from ``x``. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the same shape as ``x`` and where every element is equal to
        ``fill_value``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.full_like.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :code:`int` datatype:

    >>> x = aikit.array([1, 2, 3, 4, 5, 6])
    >>> fill_value = 1
    >>> y = aikit.full_like(x, fill_value)
    >>> print(y)
    aikit.array([1, 1, 1, 1, 1, 1])

    >>> fill_value = 0.000123
    >>> x = aikit.ones(5)
    >>> y = aikit.full_like(x, fill_value)
    >>> print(y)
    aikit.array([0.000123, 0.000123, 0.000123, 0.000123, 0.000123])

    With float datatype:

    >>> x = aikit.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> fill_value = 0.000123
    >>> y = aikit.full_like(x, fill_value)
    >>> print(y)
    aikit.array([0.000123, 0.000123, 0.000123, 0.000123, 0.000123, 0.000123])

    With :class:`aikit.NativeArray` input:

    >>> x = aikit.native_array([3.0, 8.0])
    >>> fill_value = 0.000123
    >>> y = aikit.full_like(x,fill_value)
    >>> print(y)
    aikit.array([0.000123, 0.000123])

    >>> x = aikit.native_array([[3., 8., 2.], [2., 8., 3.]])
    >>> y = aikit.full_like(x, fill_value)
    >>> print(y)
    aikit.array([[0.000123, 0.000123, 0.000123],
               [0.000123, 0.000123, 0.000123]])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([1.2, 2.2324, 3.234]),
    ...                   b=aikit.array([4.123, 5.23, 6.23]))
    >>> fill_value = 15.0
    >>> y = aikit.full_like(x, fill_value)
    >>> print(y)
    {
        a: aikit.array([15., 15., 15.]),
        b: aikit.array([15., 15., 15.])
    }
    """
    return current_backend(x).full_like(
        x, fill_value, dtype=dtype, device=device, out=out
    )


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device
def ones_like(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return a new array filled with ones and having the same shape as an
    input array ``x``.

    .. note::

        An output array having a complex floating-point data type must contain complex
        numbers having a real component equal to one and an imaginary component equal
        to zero (i.e., ``1 + 0j``).

    Parameters
    ----------
    x
        input array from which to derive the output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be inferred from ``x``. Default  ``None``.
    device
        device on which to place the created array. If device is ``None``, the output
        array device must be inferred from ``x``. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the same shape as ``x`` and filled with ``ones``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.ones_like.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([1, 2, 3, 4, 5, 6])
    >>> y = aikit.ones_like(x)
    >>> print(y)
    aikit.array([1, 1, 1, 1, 1, 1])

    >>> x = aikit.array([[0, 1, 2],[3, 4, 5]], dtype = aikit.float32)
    >>> y = aikit.ones_like(x)
    >>> print(y)
    aikit.array([[1., 1., 1.],
           [1., 1., 1.]])

    >>> x = aikit.array([3., 2., 1.])
    >>> y = aikit.zeros(3)
    >>> aikit.ones_like(x, out=y)
    >>> print(y)
    aikit.array([1., 1., 1.])

    With :class:`aikit.NativeArray` input:

    >>> x = aikit.native_array([[3, 8, 2],[2, 8, 3]])
    >>> y = aikit.ones_like(x)
    >>> print(y)
    aikit.array([[1, 1, 1],
           [1, 1, 1]])

    >>> x = aikit.native_array([3, 8, 2, 0, 0, 2])
    >>> y = aikit.ones_like(x, dtype=aikit.IntDtype('int32'), device=aikit.Device('cpu'))
    >>> print(y)
    aikit.array([1, 1, 1, 1, 1, 1])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([3, 2, 1]), b=aikit.array([8, 2, 3]))
    >>> y = aikit.ones_like(x)
    >>> print(y)
    {
        a: aikit.array([1, 1, 1]),
        b: aikit.array([1, 1, 1])
    }

    With :class:`aikit.Array` input:

    >>> x = aikit.array([2, 3, 8, 2, 1])
    >>> y = x.ones_like()
    >>> print(y)
    aikit.array([1, 1, 1, 1, 1])

    With :class:'aikit.Container' input:

    >>> x = aikit.Container(a=aikit.array([3., 8.]), b=aikit.array([2., 2.]))
    >>> y = x.ones_like()
    >>> print(y)
    {
        a: aikit.array([1., 1.]),
        b: aikit.array([1., 1.])
    }
    """
    return current_backend(x).ones_like(x, dtype=dtype, device=device, out=out)


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device
def zeros_like(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return a new array filled with zeros and having the same ``shape`` as an
    input array ``x``.

    Parameters
    ----------
    x
         input array from which to derive the output array shape.
    dtype
        output array data type. If ``dtype`` is ``None``, the output array data type
        must be inferred from ``x``. Default: ``None``.
    device
        device on which to place the created array. If ``device`` is ``None``, the
        output array device must be inferred from ``x``. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the same shape as ``x`` and filled with ``zeros``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.zeros_like.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([1, 2, 3, 4, 5, 6])
    >>> y = aikit.zeros_like(x)
    >>> print(y)
    aikit.array([0, 0, 0, 0, 0, 0])

    >>> x = aikit.array([[0, 1, 2],[3, 4, 5]], dtype = aikit.float32)
    >>> y = aikit.zeros_like(x)
    >>> print(y)
    aikit.array([[0., 0., 0.],
            [0., 0., 0.]])

    >>> x = aikit.array([3., 2., 1.])
    >>> y = aikit.ones(3)
    >>> aikit.zeros_like(x, out=y)
    >>> print(y)
    aikit.array([0., 0., 0.])

    With :class:`aikit.NativeArray` input:

    >>> x = aikit.native_array([[3, 8, 2],[2, 8, 3]])
    >>> y = aikit.zeros_like(x)
    >>> print(y)
    aikit.array([[0, 0, 0],[0, 0, 0]])


    >>> x = aikit.native_array([3, 8, 2, 0, 0, 2])
    >>> y = aikit.zeros_like(x, dtype=aikit.IntDtype('int32'), device=aikit.Device('cpu'))
    >>> print(y)
    aikit.array([0, 0, 0, 0, 0, 0])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([3, 2, 1]), b=aikit.array([8, 2, 3]))
    >>> y = aikit.zeros_like(x)
    >>> print(y)
    {
        a: aikit.array([0, 0, 0]),
        b: aikit.array([0, 0, 0])
    }


    With :class:`aikit.Array` input:

    >>> x = aikit.array([2, 3, 8, 2, 1])
    >>> y = x.zeros_like()
    >>> print(y)
    aikit.array([0, 0, 0, 0, 0])

    With :class:'aikit.Container' input:

    >>> x = aikit.Container(a=aikit.array([3., 8.]), b=aikit.array([2., 2.]))
    >>> y = x.zeros_like()
    >>> print(y)
    {
        a: aikit.array([0., 0.]),
        b: aikit.array([0., 0.])
    }
    """
    return current_backend(x).zeros_like(x, dtype=dtype, device=device, out=out)


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def tril(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    k: int = 0,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return the lower triangular part of a matrix (or a stack of matrices)
    ``x``.

    .. note::

        The main diagonal is defined as the set of indices ``{(i, i)}`` for ``i``
        on the interval ``[0, min(M, N) - 1]``.

    Parameters
    ----------
    x
        input array having shape (..., M, N) and whose innermost two dimensions form MxN
        matrices.
    k
        diagonal above which to zero elements. If k = 0, the diagonal is the main
        diagonal. If k < 0, the diagonal is below the main diagonal. If k > 0, the
        diagonal is above the main diagonal. Default: ``0``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the lower triangular part(s). The returned array must have
        the same shape and data type as x. All elements above the specified diagonal k
        must be zeroed. The returned array should be allocated on the same device as x.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.tril.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.
    """
    return current_backend(x).tril(x, k=k, out=out)


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def triu(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    k: int = 0,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return the upper triangular part of a matrix (or a stack of matrices)
    ``x``.

    .. note::

        The upper triangular part of the matrix is defined as the elements
        on and above the specified diagonal ``k``.

    Parameters
    ----------
    x
        input array having shape (..., M, N) and whose innermost two dimensions form MxN
        matrices.    *,
    k
        diagonal below which to zero elements. If k = 0, the diagonal is the main
        diagonal. If k < 0, the diagonal is below the main diagonal. If k > 0, the
        diagonal is above the main diagonal. Default: ``0``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the upper triangular part(s). The returned array must have
        the same shape and data type as x. All elements below the specified diagonal k
        must be zeroed. The returned array should be allocated on the same device as x.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.triu.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.
    """
    return current_backend(x).triu(x, k=k, out=out)


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_shapes
@outputs_to_aikit_arrays
@handle_array_function
@infer_dtype
@handle_device
def empty(
    shape: Union[aikit.Shape, aikit.NativeShape],
    *,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape
       output array shape.
    dtype
        output array data type. If dtype is None, the output array data type must be the
        default floating-point data type. Default: ``None``.
    device
        device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an uninitialized array having a specified shape


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.empty.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.
    """
    return current_backend().empty(shape, dtype=dtype, device=device, out=out)


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device
def empty_like(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return an uninitialized array with the same shape as an input array x.

    Parameters
    ----------
    x
        input array from which to derive the output array shape.
    dtype
        output array data type. If dtype is None, the output array data type must be
        inferred from x. Default: ``None``.
    device
        device on which to place the created array. If device is None, the output array
        device must be inferred from x. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the same shape as x and containing uninitialized data.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.empty_like.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.
    """
    return current_backend(x).empty_like(x, dtype=dtype, device=device, out=out)


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@outputs_to_aikit_arrays
@handle_array_function
@infer_dtype
@handle_device
def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return a two-dimensional array with ones on the k diagonal and zeros
    elsewhere.

    Parameters
    ----------
    n_rows
        number of rows in the output array.
    n_cols
        number of columns in the output array. If None, the default number of columns in
        the output array is equal to n_rows. Default: ``None``.
    k
        index of the diagonal. A positive value refers to an upper diagonal, a negative
        value to a lower diagonal, and 0 to the main diagonal. Default: ``0``.
    batch_shape
        optional input that determines returning identity array shape.
        Default: ``None``.
    dtype
        output array data type. If dtype is None, the output array data type must be the
        default floating-point data type. Default: ``None``.
    device
        the device on which to place the created array.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        device on which to place the created array. Default: ``None``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.eye.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances as a replacement to any of the arguments.

    Examples
    --------
    With :'n_rows' input:

    >>> x = aikit.eye(3)
    >>> print(x)
    aikit.array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])


    With :'n_cols' input:

    >>> x = aikit.eye(3,4)
    >>> print(x)
    aikit.array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.]])


    With :'k' input:

    >>> x = aikit.eye(3, k=1)
    >>> print(x)
    aikit.array([[0., 1., 0.],
               [0., 0., 1.],
               [0., 0., 0.]])


    With :'dtype' input:

    >>> x = aikit.eye(4, k=2, dtype=aikit.IntDtype('int32'))
    >>> print(x)
    aikit.array([[0, 0, 1, 0],
               [0, 0, 0, 1],
               [0, 0, 0, 0],
               [0, 0, 0, 0]])


    With :'batch_shape' input:

    >>> x = aikit.eye(2, 3, batch_shape=[3])
    >>> print(x)
    aikit.array([[[1., 0., 0.],
                [0., 1., 0.]],

                [[1., 0., 0.],
                [0., 1., 0.]],

                [[1., 0., 0.],
                [0., 1., 0.]]])


    With :'out' input:

    >>> y = aikit.ones((3, 3))
    >>> aikit.eye(3, out=y)
    >>> print(y)
    aikit.array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])


    With :'device' input:

    >>> x = aikit.eye(3, device=aikit.Device('cpu'))
    >>> print(x)
    aikit.array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
    """
    return current_backend().eye(
        n_rows,
        n_cols,
        k=k,
        batch_shape=batch_shape,
        dtype=dtype,
        device=device,
        out=out,
    )


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device
def linspace(
    start: Union[aikit.Array, aikit.NativeArray, float],
    stop: Union[aikit.Array, aikit.NativeArray, float],
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Generate a certain number of evenly-spaced values in an interval along a
    given axis.

    See :math:`arange` that allows to specify the step size of evenly spaced values in
    an interval.

    Parameters
    ----------
    start
        First entry in the range.
    stop
        Final entry in the range.
    num
        Number of values to generate.
    axis
        Axis along which the operation is performed.
    endpoint
        If True, stop is the last sample. Otherwise, it is not included.
    dtype
        output array data type.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Tensor of evenly-spaced values.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.linspace.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With float input:

    >>> x = aikit.linspace(1, 2, 3)
    >>> print(x)
    aikit.array([1. , 1.5, 2. ])

    >>> x = aikit.linspace(1, 2, 4, endpoint=False)
    >>> print(x)
    aikit.array([1., 1.25, 1.5 , 1.75])

    >>> x = aikit.linspace(1, 10, 4, dtype="int32")
    >>> print(x)
    aikit.array([ 1,  4,  7, 10])

    >>> x = aikit.linspace(1, 2, 4, device= "cpu")
    >>> print(x)
    aikit.array([1., 1.33333337, 1.66666663, 2.])

    >>> y = aikit.array([0,0,0,0])
    >>> aikit.linspace(1, 2, 4, out= y)
    >>> print(y)
    aikit.array([1, 1, 1, 2])

    With :class:`aikit.Array` input:

    >>> x = aikit.array([1,2])
    >>> y = aikit.array([4,5])
    >>> z = aikit.linspace(x, y, 4, axis = 0)
    >>> print(z)
    aikit.array([[1, 2],
               [2, 3],
               [3, 4],
               [4, 5]])
    """
    return current_backend(start).linspace(
        start,
        stop,
        num,
        axis=axis,
        endpoint=endpoint,
        dtype=dtype,
        device=device,
        out=out,
    )


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def meshgrid(
    *arrays: Union[aikit.Array, aikit.NativeArray],
    sparse: bool = False,
    indexing: str = "xy",
    out: Optional[aikit.Array] = None,
) -> List[aikit.Array]:
    """Return coordinate matrices from coordinate vectors.

    Parameters
    ----------
    arrays
        an arbitrary number of one-dimensional arrays representing grid coordinates.
        Each array should have the same numeric data type.
    sparse
        if True, a sparse grid is returned in order to conserve memory.
        Default: ``False``.
    indexing
        Cartesian ``'xy'`` or matrix ``'ij'`` indexing of output. If provided zero or
        one one-dimensional vector(s) (i.e., the zero- and one-dimensional cases,
        respectively), the ``indexing`` keyword has no effect and should be ignored.
        Default: ``'xy'``.

    Returns
    -------
    ret
        list of N arrays, where ``N`` is the number of provided one-dimensional input
        arrays. Each returned array must have rank ``N``. For ``N`` one-dimensional
        arrays having lengths ``Ni = len(xi)``,

        - if matrix indexing ``ij``, then each returned array must have the shape
          ``(N1, N2, N3, ..., Nn)``.
        - if Cartesian indexing ``xy``, then each returned array must have shape
          ``(N2, N1, N3, ..., Nn)``.

        Accordingly, for the two-dimensional case with input one-dimensional arrays of
        length ``M`` and ``N``, if matrix indexing ``ij``, then each returned array must
        have shape ``(M, N)``, and, if Cartesian indexing ``xy``, then each returned
        array must have shape ``(N, M)``.

        Similarly, for the three-dimensional case with input one-dimensional arrays of
        length ``M``, ``N``, and ``P``, if matrix indexing ``ij``, then each returned
        array must have shape ``(M, N, P)``, and, if Cartesian indexing ``xy``, then
        each returned array must have shape ``(N, M, P)``.

        Each returned array should have the same data type as the input arrays.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of
    the `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.meshgrid.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([1, 2])
    >>> y = aikit.array([3, 4])
    >>> xv, yv = aikit.meshgrid(x, y)
    >>> print(xv)
    aikit.array([[1, 2],
            [1, 2]])

    >>> print(yv)
    aikit.array([[3, 3],
            [4, 4]])

    >>> x = aikit.array([1, 2, 5])
    >>> y = aikit.array([4, 1])
    >>> xv, yv = aikit.meshgrid(x, y, indexing='ij')
    >>> print(xv)
    aikit.array([[1, 1],
            [2, 2],
            [5, 5]])

    >>> print(yv)
    aikit.array([[4, 1],
            [4, 1],
            [4, 1]])

    >>> x = aikit.array([1, 2, 3])
    >>> y = aikit.array([4, 5, 6])
    >>> xv, yv = aikit.meshgrid(x, y, sparse=True)
    >>> print(xv)
    aikit.array([[1, 2, 3]])

    >>> print(yv)
    aikit.array([[4], [5], [6]])

    With :class:`aikit.NativeArray` input:

    >>> x = aikit.native_array([1, 2])
    >>> y = aikit.native_array([3, 4])
    >>> xv, yv = aikit.meshgrid(x, y)
    >>> print(xv)
    aikit.array([[1, 2],
            [1, 2]])

    >>> print(yv)
    aikit.array([[3, 3],
            [4, 4]])
    """
    return current_backend().meshgrid(
        *arrays, sparse=sparse, indexing=indexing, out=out
    )


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_shapes
@inputs_to_native_arrays
@outputs_to_aikit_arrays
@handle_array_function
@handle_device
def full(
    shape: Union[aikit.Shape, aikit.NativeShape],
    fill_value: Union[float, bool],
    /,
    *,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return a new array having a specified ``shape`` and filled with
    ``fill_value``.

    Parameters
    ----------
    shape
        output array shape.
    fill_value
        fill value.
    dtype
        output array data type. If ``dtype`` is `None`, the output array data type must
        be inferred from ``fill_value``. If the fill value is an ``int``, the output
        array data type must be the default integer data type. If the fill value is a
        ``float``, the output array data type must be the default floating-point data
        type. If the fill value is a ``bool``, the output array must have boolean data
        type. Default: ``None``.
    device
        device on which to place the created array. Default: ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array where every element is equal to `fill_value`.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.full.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Shape` input:

    >>> shape = aikit.Shape((2,2))
    >>> fill_value = 8.6
    >>> x = aikit.full(shape, fill_value)
    >>> print(x)
    aikit.array([[8.6, 8.6],
               [8.6, 8.6]])

    With :class:`aikit.NativeShape` input:

    >>> shape = aikit.NativeShape((2, 2, 2))
    >>> fill_value = True
    >>> dtype = aikit.bool
    >>> device = aikit.Device('cpu')
    >>> x = aikit.full(shape, fill_value, dtype=dtype, device=device)
    >>> print(x)
    aikit.array([[[True,  True],
                [True,  True]],
               [[True,  True],
                [True,  True]]])

    With :class:`aikit.NativeDevice` input:

    >>> shape = aikit.NativeShape((1, 2))
    >>> fill_value = 0.68
    >>> dtype = aikit.float64
    >>> device = aikit.NativeDevice('cpu')
    >>> x = aikit.full(shape, fill_value, dtype=dtype, device=device)
    >>> print(x)
    aikit.array([[0.68, 0.68]])

    With :class:`aikit.Container` input:

    >>> shape = aikit.Container(a=aikit.NativeShape((2, 1)), b=aikit.Shape((2, 1, 2)))
    >>> fill_value = aikit.Container(a=0.99, b=False)
    >>> dtype = aikit.Container(a=aikit.float64, b=aikit.bool)
    >>> device = aikit.Container(a=aikit.NativeDevice('cpu'), b=aikit.Device('cpu'))
    >>> x = aikit.full(shape, fill_value, dtype=dtype, device=device)
    >>> print(x)
    {
        a: aikit.array([[0.99],
                      [0.99]]),
        b: aikit.array([[[False, False]],
                      [[False, False]]])
    }
    """
    return current_backend().full(
        shape, fill_value, dtype=dtype, device=device, out=out
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def to_dlpack(
    x: Union[aikit.Array, aikit.NativeArray], /, *, out: Optional[aikit.Array] = None
):
    """Return PyCapsule Object.

    Parameters
    ----------
    x  object
        input (array) object.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Return PyCapsule Object.

        .. admonition:: Note
           :class: note

           The returned array may be either a copy or a view. See
           :ref:`data-interchange` for details.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.from_dlpack.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.
    """
    return current_backend(x).to_dlpack(x, out=out)


@handle_backend_invalid
def from_dlpack(
    x: Union[aikit.Array, aikit.NativeArray], /, *, out: Optional[aikit.Array] = None
) -> aikit.Array:
    """Return a new array containing the data from another (array) object with
    a ``__dlpack__`` method or PyCapsule Object.

    Parameters
    ----------
    x  object
        input (array) object with a ``__dlpack__`` method or PyCapsule Object.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array containing the data in `x`.

        .. admonition:: Note
           :class: note

           The returned array may be either a copy or a view. See
           :ref:`data-interchange` for details.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.from_dlpack.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.
    """
    return current_backend(x).from_dlpack(x, out=out)


# Extra #
# ------#


array = asarray


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_arrays
@handle_array_function
@handle_device
def copy_array(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    to_aikit_array: bool = True,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Copy an array.

    Parameters
    ----------
    x
        array, input array containing elements to copy.
    to_aikit_array
        boolean, if True the returned array will be an aikit.Array object otherwise
        returns an aikit.NativeArray object (i.e. a torch.tensor, np.array, etc.,
        depending on the backend), defaults to True.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        a copy of the input array ``x``.

    Examples
    --------
    With one :class:`aikit.Array` input:

    >>> x = aikit.array([-1, 0, 1])
    >>> y = aikit.copy_array(x)
    >>> print(y)
    aikit.array([-1, 0, 1])

    >>> x = aikit.array([1, 0, 1, 1])
    >>> y = aikit.copy_array(x)
    >>> print(y)
    aikit.array([1, 0, 1, 1])

    >>> x = aikit.array([1, 0, 1, -1])
    >>> y = aikit.zeros((1, 4))
    >>> aikit.copy_array(x, out=y)
    >>> print(y)
    aikit.array([1, 0, 1, -1])

    >>> x = aikit.array([1, 0, 1, 1])
    >>> aikit.copy_array(x, out=x)
    >>> print(x)
    aikit.array([1, 0, 1, 1])

    With one :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([-1, 0, 1]))
    >>> y = aikit.copy_array(x)
    >>> print(y)
    {
        a: aikit.array([-1, 0, 1])
    }

    >>> x = aikit.Container(a=aikit.array([-1, 0, 1]),b=aikit.array([-1, 0, 1, 1, 1, 0]))
    >>> y = aikit.copy_array(x)
    >>> print(y)
    {
        a: aikit.array([-1, 0, 1]),
        b: aikit.array([-1, 0, 1, 1, 1, 0])
    }

    With one :class:`aikit.Container` static method:

    >>> x = aikit.Container(a=aikit.array([-1, 0, 1]),b=aikit.array([-1, 0, 1, 1, 1, 0]))
    >>> y = aikit.Container.static_copy_array(x)
    >>> print(y)
    {
        a: aikit.array([-1, 0, 1]),
        b: aikit.array([-1, 0, 1, 1, 1, 0])
    }

    With one :class:`aikit.Array` instance method:

    >>> x = aikit.array([-1, 0, 1])
    >>> y = x.copy_array()
    >>> print(y)
    aikit.array([-1, 0, 1])

    >>> x = aikit.array([1, 0, 1, 1])
    >>> y = x.copy_array()
    >>> print(y)
    aikit.array([1, 0, 1, 1])

    With :class:`aikit.Container` instance method:

    >>> x = aikit.Container(a=aikit.array([1, 0, 1]),b=aikit.array([-1, 0, 1, 1]))
    >>> y = x.copy_array()
    >>> print(y)
    {
        a: aikit.array([1, 0, 1]),
        b: aikit.array([-1, 0, 1, 1])
    }
    """
    return current_backend(x).copy_array(x, to_aikit_array=to_aikit_array, out=out)


@handle_backend_invalid
@handle_array_like_without_promotion
def native_array(
    x: Union[aikit.Array, aikit.NativeArray, List[Number], Tuple[Number], np.ndarray],
    /,
    *,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
) -> aikit.NativeArray:
    """Convert the input to a native array.

    Parameters
    ----------
    x
        input data, in any form that can be converted to an array. This includes lists,
        lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
    dtype
        datatype, optional. Datatype is inferred from the input data.
    device
        device on which to place the created array. Default: ``None``.

    Returns
    -------
    ret
        A native array interpretation of x.

    Examples
    --------
    With :class:`List[Number]` input:

    >>> x = [1, 2, 3]
    >>> x_native = aikit.native_array(x)
    >>> print(x_native)
    [1 2 3]

    With :class:`np.ndarray` input:
    >>> y = np.array([4, 5, 6])
    >>> y_native = aikit.native_array(y)
    >>> print(y_native)
    [4 5 6]

    With :class:`aikit.Array` input:
    >>> z = aikit.array([7, 8, 9])
    >>> z_native = aikit.native_array(z)
    >>> print(z_native)
    [7 8 9]
    """
    # ToDo: Make this more efficient,
    # ideally without first converting to aikit.Array with aikit.asarray and then
    # converting back to native with aikit.to_native

    return aikit.to_native(aikit.asarray(x, dtype=dtype, device=device))


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def one_hot(
    indices: Union[aikit.Array, aikit.NativeArray],
    depth: int,
    /,
    *,
    on_value: Optional[Number] = None,
    off_value: Optional[Number] = None,
    axis: Optional[int] = None,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    device: Union[aikit.Device, aikit.NativeDevice] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return a one-hot array. The locations represented by indices in the
    parameter indices take value on_value, while all other locations take value
    off_value.

    Parameters
    ----------
    indices
        Indices for where the ones should be scattered *[batch_shape, dim]*
    depth
        Scalar defining the depth of the one-hot dimension.
    on_value
        Scalar defining the value to fill in output when indices[j] == i.
        Default: ``1``.
    off_value
        Scalar defining the value to fill in output when indices[j] != i.
        Default: ``0``.
    axis
        Axis to scatter on. The default is ``-1``, a new inner-most axis is created.
    dtype
        The data type of the output tensor.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as x if
        None.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Tensor of zeros with the same shape and type as a, unless dtype provided which
        overrides.

    Examples
    --------
    With :class:`aikit.Array` inputs:

    >>> x = aikit.array([3, 1])
    >>> y = 5
    >>> z = x.one_hot(5)
    >>> print(z)
    aikit.array([[0., 0., 0., 1., 0.],
    ...    [0., 1., 0., 0., 0.]])

    >>> x = aikit.array([0])
    >>> y = 5
    >>> aikit.one_hot(x, y)
    aikit.array([[1., 0., 0., 0., 0.]])

    >>> x = aikit.array([0])
    >>> y = 5
    >>> aikit.one_hot(x, 5, out=z)
    aikit.array([[1., 0., 0., 0., 0.]])
    >>> print(z)
    aikit.array([[1., 0., 0., 0., 0.]])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([1, 2]), \
        b=aikit.array([3, 1]), c=aikit.array([2, 3]))
    >>> y = 5
    >>> z = x.one_hot(y)
    >>> print(z)
    {
        a: aikit.array([[0., 1., 0., 0., 0.],
                    [0., 0., 1., 0., 0.]]),
        b: aikit.array([[0., 0., 0., 1., 0.],
                    [0., 1., 0., 0., 0.]]),
        c: aikit.array([[0., 0., 1., 0., 0.],
                    [0., 0., 0., 1., 0.]])
    }

    >>> x = aikit.Container(a=aikit.array([2]), \
        b=aikit.array([]), c=aikit.native_array([4]))
    >>> y = 7
    >>> z = x.one_hot(y)
    >>> print(z)
    {
        a: aikit.array([[0., 0., 1., 0., 0., 0., 0.]]),
        b: aikit.array([], shape=(0, 7)),
        c: aikit.array([[0., 0., 0., 0., 1., 0., 0.]])
    }
    """
    return current_backend(indices).one_hot(
        indices,
        depth,
        on_value=on_value,
        off_value=off_value,
        axis=axis,
        dtype=dtype,
        device=device,
        out=out,
    )


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@infer_dtype
@handle_device
def logspace(
    start: Union[aikit.Array, aikit.NativeArray, float],
    stop: Union[aikit.Array, aikit.NativeArray, float],
    /,
    num: int,
    *,
    base: float = 10.0,
    axis: int = 0,
    endpoint: bool = True,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Generate a certain number of evenly-spaced values in log space, in an
    interval along a given axis.

    Parameters
    ----------
    start
        First value in the range in log space. base ** start is the starting value in
        the sequence. Can be an array or a float.
    stop
        Last value in the range in log space. base ** stop is the final value in the
        sequence. Can be an array or a float.
    num
        Number of values to generate.
    base
        The base of the log space. Default is 10.0
    axis
        Axis along which the operation is performed. Relevant only if start or stop are
        array-like. Default is 0.
    endpoint
        If True, stop is the last sample. Otherwise, it is not included. Default is
        True.
    dtype
        The data type of the output tensor. If None, the dtype of on_value is used or if
        that is None, the dtype of off_value is used, or if that is None, defaults to
        float32. Default is None.
    device
        device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Default is
        None.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to. Default is None.

    Returns
    -------
    ret
        Tensor of evenly-spaced values in log space.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With float input:

    >>> print(aikit.logspace(1, 2, 4))
    aikit.array([ 10., 21.5443469, 46.41588834, 100.])

    >>> print(aikit.logspace(1, 2, 4, endpoint=False))
    aikit.array([10., 17.7827941, 31.6227766, 56.23413252])

    >>> print(aikit.logspace(1, 2, 4, dtype= int))
    aikit.array([ 10.,  10.,  10., 100.])

    >>> out = aikit.array([0,0,0,0])
    >>> aikit.logspace(1, 2, 4, out = out)
    >>> print(out)
    aikit.array([ 10,  21,  46, 100])

    With :class:`aikit.Array` input:
    >>> x = aikit.array([1, 2])
    >>> y = aikit.array([4, 5])
    >>> print(aikit.logspace(x, y, 4))
    aikit.array([[1.e+01, 1.e+02],
               [1.e+02, 1.e+03],
               [1.e+03, 1.e+04],
               [1.e+04, 1.e+05])

    >>> x = aikit.array([1, 2])
    >>> y = aikit.array([4, 5])
    >>> print(aikit.logspace(x, y, 4, axis = 1))
    aikit.array([[[1.e+01, 1.e+02, 1.e+03, 1.e+04],
               [1.e+02, 1.e+03, 1.e+04, 1.e+05]]])

    >>> x = aikit.array([1, 2])
    >>> y = aikit.array([4])
    >>> print(aikit.logspace(x, y, 4))
    aikit.array([[   10.,   100.],
           [  100.,   100.],
           [ 1000.,  1000.],
           [10000., 10000.]])
    """
    result = base ** linspace(
        start,
        stop,
        num,
        endpoint=endpoint,
        axis=axis,
        dtype=dtype,
        device=device,
    )
    if aikit.exists(out):
        return aikit.inplace_update(out, result)
    return result


@handle_nestable
@outputs_to_aikit_arrays
def frombuffer(
    buffer: bytes,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    count: Optional[int] = -1,
    offset: Optional[int] = 0,
) -> aikit.Array:
    r"""Interpret a buffer as a 1-dimensional array.

    .. note::
        Note that either of the following must be true:
        1. count is a positive non-zero number, and the total number of bytes
        in the buffer is equal or greater than offset plus count times the size
        (in bytes) of dtype.
        2. count is negative, and the length (number of bytes) of the buffer
        subtracted by the offset is a multiple of the size (in bytes) of dtype.

    Parameters
    ----------
    buffer
        An object that exposes the buffer interface.
    dtype
        Data-type of the returned array; default: float.
    count
        Number of items to read. -1 means all data in the buffer.
    offset
        Start reading the buffer from this offset (in bytes); default: 0.

    Returns
    -------
    out
        1-dimensional array.

    Examples
    --------
    With :class:`bytes` inputs:

    >>> x = b'\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@'
    >>> y = aikit.frombuffer(x, dtype=aikit.float64)
    >>> print(y)
    aikit.array([1., 2.])

    >>> x = b'\x01\x02\x03\x04'
    >>> y = aikit.frombuffer(x, dtype='int8', count=-2, offset=1)
    >>> print(y)
    aikit.array([2, 3, 4])

    >>> x = b'\x00<\x00@\x00B\x00D\x00E'
    >>> y = aikit.frombuffer(x, dtype='float16', count=4, offset=2)
    >>> print(y)
    aikit.array([2., 3., 4., 5.])
    """
    return current_backend().frombuffer(
        buffer,
        dtype=dtype,
        count=count,
        offset=offset,
    )


@handle_exceptions
@handle_nestable
@outputs_to_aikit_arrays
@handle_device
def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: int = 0,
    /,
    *,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
) -> Tuple[aikit.Array]:
    """Return the indices of the upper triangular part of a row by col matrix
    in a 2-by-N shape (tuple of two N dimensional arrays), where the first row
    contains row coordinates of all indices and the second row contains column
    coordinates. Indices are ordered based on rows and then columns.  The upper
    triangular part of the matrix is defined as the elements on and above the
    diagonal.  The argument k controls which diagonal to consider. If k = 0,
    all elements on and above the main diagonal are retained. A positive value
    excludes just as many diagonals above the main diagonal, and similarly a
    negative value includes just as many diagonals below the main diagonal. The
    main diagonal are the set of indices {(i,i)} for i∈[0,min{n_rows,
    n_cols}−1].

    Notes
    -----
    Primary purpose of this function is to slice an array of shape (n,m). See
    https://numpy.org/doc/stable/reference/generated/numpy.triu_indices.html
    for examples

    Tensorflow does not support slicing 2-D tensor with tuple of tensor of indices

    Parameters
    ----------
    n_rows
       number of rows in the 2-d matrix.
    n_cols
       number of columns in the 2-d matrix. If None n_cols will be the same as n_rows
    k
       number of shifts from the main diagonal. k = 0 includes main diagonal,
       k > 0 moves upwards and k < 0 moves downwards
    device
       device on which to place the created array. Default: ``None``.

    Returns
    -------
    ret
        an 2xN shape, tuple of two N dimensional, where first subarray (i.e. ret[0])
        contains row coordinates of all indices and the second subarray (i.e ret[1])
        contains columns indices.

    Function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    >>> x = aikit.triu_indices(4,4,0)
    >>> print(x)
    (aikit.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]),
    aikit.array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]))

    >>> x = aikit.triu_indices(4,4,1)
    >>> print(x)
    (aikit.array([0, 0, 0, 1, 1, 2]),
    aikit.array([1, 2, 3, 2, 3, 3]))

    >>> x = aikit.triu_indices(4,4,-2)
    >>> print(x)
    (aikit.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]),
    aikit.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3]))

    >>> x = aikit.triu_indices(4,2,0)
    >>> print(x)
    (aikit.array([0, 0, 1]),
    aikit.array([0, 1, 1]))

    >>> x = aikit.triu_indices(2,4,0)
    >>> print(x)
    (aikit.array([0, 0, 0, 0, 1, 1, 1]),
    aikit.array([0, 1, 2, 3, 1, 2, 3]))

    >>> x = aikit.triu_indices(4,-4,0)
    >>> print(x)
    (aikit.array([]), aikit.array([]))

    >>> x = aikit.triu_indices(4,4,100)
    >>> print(x)
    (aikit.array([]), aikit.array([]))

    >>> x = aikit.triu_indices(2,4,-100)
    >>> print(x)
    (aikit.array([0, 0, 0, 0, 1, 1, 1, 1]), aikit.array([0, 1, 2, 3, 0, 1, 2, 3]))
    """
    return current_backend().triu_indices(n_rows, n_cols, k, device=device)
