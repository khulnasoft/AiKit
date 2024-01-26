# global
import ast
import logging
import inspect
import math
import functools
from numbers import Number
from typing import Union, Tuple, List, Optional, Callable, Iterable, Any
import numpy as np
import importlib

# local
import aikit
from aikit.utils.backend import current_backend
from aikit.func_wrapper import (
    handle_array_function,
    handle_out_argument,
    to_native_arrays_and_back,
    inputs_to_native_arrays,
    handle_nestable,
    handle_array_like_without_promotion,
    inputs_to_aikit_arrays,
    inputs_to_native_shapes,
    handle_device,
    handle_backend_invalid,
)
from aikit.utils.exceptions import handle_exceptions


# Helpers #
# --------#


def _is_valid_dtypes_attributes(fn: Callable) -> bool:
    if hasattr(fn, "supported_dtypes") and hasattr(fn, "unsupported_dtypes"):
        fn_supported_dtypes = fn.supported_dtypes
        fn_unsupported_dtypes = fn.unsupported_dtypes
        if isinstance(fn_supported_dtypes, dict):
            if isinstance(fn_unsupported_dtypes, dict):
                backend_str = aikit.current_backend_str()
                if (
                    backend_str in fn_supported_dtypes
                    and backend_str in fn_unsupported_dtypes
                ):
                    return False
        elif isinstance(fn_unsupported_dtypes, tuple):
            return False
    return True


def _handle_nestable_dtype_info(fn):
    @functools.wraps(fn)
    def _handle_nestable_dtype_info_wrapper(type):
        if isinstance(type, aikit.Container):
            type = type.cont_map(lambda x, kc: fn(x))
            type.__dict__["max"] = type.cont_map(lambda x, kc: x.max)
            type.__dict__["min"] = type.cont_map(lambda x, kc: x.min)
            return type
        return fn(type)

    return _handle_nestable_dtype_info_wrapper


# Unindent every line in the source such that
# class methods can be traced as normal methods
def _lstrip_lines(source: str) -> str:
    # Separate all lines
    source = source.split("\n")
    # Check amount of indent before first character
    indent = len(source[0]) - len(source[0].lstrip())
    # Remove same spaces from all lines
    for i in range(len(source)):
        source[i] = source[i][indent:]
    source = "\n".join(source)
    return source


# Get the list of function used the function
def _get_function_list(func):
    tree = ast.parse(_lstrip_lines(inspect.getsource(func)))
    names = {}
    # Extract all the call names
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            nodef = node.func
            if isinstance(nodef, ast.Name):
                names[nodef.id] = getattr(
                    func,
                    "__self__",
                    getattr(
                        importlib.import_module(func.__module__),
                        func.__qualname__.split(".")[0],
                        None,
                    ),
                )
            elif isinstance(nodef, ast.Attribute):
                if (
                    hasattr(nodef, "value")
                    and hasattr(nodef.value, "id")
                    and nodef.value.id not in ["aikit", "self"]
                    and "_frontend" not in nodef.value.id
                ):
                    continue
                names[ast.unparse(nodef)] = getattr(
                    func,
                    "__self__",
                    getattr(
                        importlib.import_module(func.__module__),
                        func.__qualname__.split(".")[0],
                        None,
                    ),
                )

    return names


# Get the reference of the functions from string
def _get_functions_from_string(func_names, module):
    ret = set()
    # We only care about the functions in the aikit or the same module
    for orig_func_name in func_names.keys():
        func_name = orig_func_name.split(".")[-1]
        if hasattr(aikit, func_name) and callable(getattr(aikit, func_name, None)):
            ret.add(getattr(aikit, func_name))
        elif hasattr(module, func_name) and callable(getattr(module, func_name, None)):
            ret.add(getattr(module, func_name))
        elif callable(getattr(func_names[orig_func_name], func_name, None)):
            ret.add(getattr(func_names[orig_func_name], func_name))
    return ret


# Get dtypes/device of nested functions, used for unsupported and supported dtypes
# IMPORTANT: a few caveats:
# 1. The base functions must be defined in aikit or the same module
# 2. If the dtypes/devices are set not in the base function, it will not be detected
# 3. Nested function cannot be parsed, due to be unable to get function reference
# 4. Functions need to be directly called, not assigned to a variable
def _nested_get(f, base_set, merge_fn, get_fn, wrapper=set):
    visited = set()
    to_visit = [f]
    out = base_set

    while to_visit:
        fn = to_visit.pop()
        if fn in visited:
            continue
        visited.add(fn)

        # if it's in the backend, we can get the dtypes directly
        # if it's in the front end, we need to recurse
        # if it's einops, we need to recurse
        if not getattr(fn, "__module__", None):
            continue
        is_frontend_fn = "frontend" in fn.__module__
        is_backend_fn = "backend" in fn.__module__ and not is_frontend_fn
        is_einops_fn = "einops" in fn.__name__
        if is_backend_fn:
            f_supported = get_fn(fn, False)
            if hasattr(fn, "partial_mixed_handler"):
                f_supported = merge_fn(
                    wrapper(f_supported["compositional"]),
                    wrapper(f_supported["primary"]),
                )
                logging.warning(
                    "This function includes the mixed partial function"
                    f" 'aikit.{fn.__name__}'. Please note that the returned data types"
                    " may not be exhaustive. Please check the dtypes of"
                    f" `aikit.{fn.__name__}` for more details"
                )
            out = merge_fn(wrapper(f_supported), out)
            continue
        elif is_frontend_fn or (hasattr(fn, "__name__") and is_einops_fn):
            f_supported = wrapper(get_fn(fn, False))
            out = merge_fn(f_supported, out)

        # skip if it's not a function

        if not (inspect.isfunction(fn) or inspect.ismethod(fn)):
            continue

        fl = _get_function_list(fn)
        res = list(_get_functions_from_string(fl, __import__(fn.__module__)))
        if is_frontend_fn:
            frontends = {
                "jax_frontend": "aikit.functional.frontends.jax",
                "np_frontend": "aikit.functional.frontends.numpy",
                "tf_frontend": "aikit.functional.frontends.tensorflow",
                "torch_frontend": "aikit.functional.frontends.torch",
                "paddle_frontend": "aikit.functional.frontends.paddle",
            }
            for key in fl:
                if "frontend" in key:
                    frontend_fn = fl[key]
                    for frontend in frontends:
                        if frontend in key:
                            key = key.replace(frontend, frontends[frontend])
                    if "(" in key:
                        key = key.split("(")[0]
                    frontend_module = ".".join(key.split(".")[:-1])
                    frontend_fl = {key: frontend_fn}
                    res += list(
                        _get_functions_from_string(
                            frontend_fl, importlib.import_module(frontend_module)
                        )
                    )
        to_visit.extend(set(res))

    return out


# allow passing "integer" if all integer dtypes are supported/unsupported for e.g.
def _expand_typesets(dtypes):
    typesets = {
        "valid": aikit.valid_dtypes,
        "numeric": aikit.valid_numeric_dtypes,
        "float": aikit.valid_float_dtypes,
        "integer": aikit.valid_int_dtypes,
        "unsigned": aikit.valid_uint_dtypes,
        "complex": aikit.valid_complex_dtypes,
    }
    dtypes = list(dtypes)
    typeset_list = []
    for i, dtype in reversed(list(enumerate(dtypes))):
        if dtype in typesets:
            typeset_list.extend(typesets[dtype])
            dtypes.pop(i)
    dtypes += typeset_list
    return dtypes


# Get the list of dtypes supported by the function
# by default returns the supported dtypes
def _get_dtypes(fn, complement=True):
    supported = set(aikit.valid_dtypes)

    # We only care about getting dtype info from the base function
    # if we do need to at some point use dtype information from the parent function
    # we can comment out the following condition
    is_frontend_fn = "frontend" in fn.__module__
    is_backend_fn = "backend" in fn.__module__ and not is_frontend_fn
    has_unsupported_dtypes_attr = hasattr(fn, "unsupported_dtypes")
    if not is_backend_fn and not is_frontend_fn and not has_unsupported_dtypes_attr:
        if complement:
            supported = set(aikit.all_dtypes).difference(supported)
        return supported

    # Their values are formatted like either
    # 1. fn.supported_dtypes = ("float16",)
    # Could also have the "all" value for the framework
    basic = [
        ("supported_dtypes", set.intersection, aikit.valid_dtypes),
        ("unsupported_dtypes", set.difference, aikit.invalid_dtypes),
    ]

    for key, merge_fn, base in basic:
        if hasattr(fn, key):
            dtypes = getattr(fn, key)
            # only einops allowed to be a dictionary
            if isinstance(dtypes, dict):
                dtypes = dtypes.get(aikit.current_backend_str(), base)
            aikit.utils.assertions.check_isinstance(dtypes, tuple)
            if not dtypes:
                dtypes = base
            dtypes = _expand_typesets(dtypes)
            supported = merge_fn(supported, set(dtypes))

    if complement:
        supported = set(aikit.all_dtypes).difference(supported)

    return tuple(supported)


# Array API Standard #
# -------------------#

Finfo = None
Iinfo = None


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def astype(
    x: Union[aikit.Array, aikit.NativeArray],
    dtype: Union[aikit.Dtype, aikit.NativeDtype],
    /,
    *,
    copy: bool = True,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Copy an array to a specified data type irrespective of :ref:`type-
    promotion` rules.

    .. note::
    Casting floating-point ``NaN`` and ``infinity`` values to integral data types
    is not specified and is implementation-dependent.

    .. note::
    When casting a boolean input array to a numeric data type, a value of ``True``
    must cast to a numeric value equal to ``1``, and a value of ``False`` must cast
    to a numeric value equal to ``0``.

    When casting a numeric input array to ``bool``, a value of ``0`` must cast to
    ``False``, and a non-zero value must cast to ``True``.

    Parameters
    ----------
    x
        array to cast.
    dtype
        desired data type.
    copy
        specifies whether to copy an array when the specified ``dtype`` matches
        the data type of the input array ``x``. If ``True``, a newly allocated
        array must always be returned. If ``False`` and the specified ``dtype``
        matches the data type of the input array, the input array must be returned;
        otherwise, a newly allocated must be returned. Default: ``True``.
    out
        optional output array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        an array having the specified data type. The returned array must have
        the same shape as ``x``.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([1, 2])
    >>> y = aikit.zeros_like(x)
    >>> y = aikit.astype(x, aikit.float64)
    >>> print(y)
    aikit.array([1., 2.])

    >>> x = aikit.array([3.141, 2.718, 1.618])
    >>> y = aikit.zeros_like(x)
    >>> aikit.astype(x, aikit.int32, out=y)
    >>> print(y)
    aikit.array([3., 2., 1.])

    >>> x = aikit.array([[-1, -2], [0, 2]])
    >>> aikit.astype(x, aikit.float64, out=x)
    >>> print(x)
    aikit.array([[-1., -2.],  [0.,  2.]])

    With :class:`aikit.NativeArray` input:

    >>> x = aikit.native_array([3.141, 2.718, 1.618])
    >>> y = aikit.astype(x, aikit.int32)
    >>> print(y)
    aikit.array([3, 2, 1])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([0,2,1]),b=aikit.array([1,0,0]))
    >>> print(aikit.astype(x, aikit.bool))
    {
        a: aikit.array([False, True, True]),
        b: aikit.array([True, False, False])
    }

    With :class:`aikit.Array` instance method:

    >>> x = aikit.array([[-1, -2], [0, 2]])
    >>> print(x.astype(aikit.float64))
    aikit.array([[-1., -2.],  [0.,  2.]])

    With :class:`aikit.Container` instance method:

    >>> x = aikit.Container(a=aikit.array([False,True,True]),
    ...                   b=aikit.array([3.14, 2.718, 1.618]))
    >>> print(x.astype(aikit.int32))
    {
        a: aikit.array([0, 1, 1]),
        b: aikit.array([3, 2, 1])
    }
    """
    return current_backend(x).astype(x, dtype, copy=copy, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@to_native_arrays_and_back
@handle_array_function
@handle_device
def broadcast_arrays(*arrays: Union[aikit.Array, aikit.NativeArray]) -> List[aikit.Array]:
    """Broadcasts one or more arrays against one another.

    Parameters
    ----------
    arrays
        an arbitrary number of arrays to-be broadcasted.

    Returns
    -------
    ret
        A list containing broadcasted arrays of type `aikit.Array`
        Each array must have the same shape, and each array must have the same
        dtype as its corresponding input array.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x1 = aikit.array([1, 2, 3])
    >>> x2 = aikit.array([4, 5, 6])
    >>> y = aikit.broadcast_arrays(x1, x2)
    >>> print(y)
    [aikit.array([1, 2, 3]), aikit.array([4, 5, 6])]

    With :class:`aikit.NativeArray` inputs:

    >>> x1 = aikit.native_array([0.3, 4.3])
    >>> x2 = aikit.native_array([3.1, 5])
    >>> x3 = aikit.native_array([2, 0])
    >>> y = aikit.broadcast_arrays(x1, x2, x3)
    [aikit.array([0.3, 4.3]), aikit.array([3.1, 5.]), aikit.array([2, 0])]

    With mixed :class:`aikit.Array` and :class:`aikit.NativeArray` inputs:

    >>> x1 = aikit.array([1, 2])
    >>> x2 = aikit.native_array([0.3, 4.3])
    >>> y = aikit.broadcast_arrays(x1, x2)
    >>> print(y)
    [aikit.array([1, 2]), aikit.array([0.3, 4.3])]

    With :class:`aikit.Container` inputs:

    >>> x1 = aikit.Container(a=aikit.array([3, 1]), b=aikit.zeros(2))
    >>> x2 = aikit.Container(a=aikit.array([4, 5]), b=aikit.array([2, -1]))
    >>> y = aikit.broadcast_arrays(x1, x2)
    >>> print(y)
    [{
        a: aikit.array([3, 1]),
        b: aikit.array([0., 0.])
    }, {
        a: aikit.array([4, 5]),
        b: aikit.array([2, -1])
    }]

    With mixed :class:`aikit.Array` and :class:`aikit.Container` inputs:

    >>> x1 = aikit.zeros(2)
    >>> x2 = aikit.Container(a=aikit.array([4, 5]), b=aikit.array([2, -1]))
    >>> y = aikit.broadcast_arrays(x1, x2)
    >>> print(y)
    [{
        a: aikit.array([0., 0.]),
        b: aikit.array([0., 0.])
    }, {
        a: aikit.array([4, 5]),
        b: aikit.array([2, -1])
    }]
    """
    return current_backend(arrays[0]).broadcast_arrays(*arrays)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_shapes
@to_native_arrays_and_back
@handle_array_function
@handle_device
def broadcast_to(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    shape: Tuple[int, ...],
    *,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Broadcasts an array to a specified shape.

    Parameters
    ----------
    x
        array to broadcast.
    shape
        array shape. Must be compatible with x (see Broadcasting). If
        the array is incompatible with the specified shape, the function
        should raise an exception.
    out
        optional output array, for writing the result to. It must have a
        shape that the inputs broadcast to.

    Returns
    -------
    ret
        an array having a specified shape. Must have the same data type as x.


    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([1, 2, 3])
    >>> y = aikit.broadcast_to(x, (3, 3))
    >>> print(y)
    aikit.array([[1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]])

    With :class:`aikit.NativeArray` input:

    >>> x = aikit.native_array([0.1 , 0.3])
    >>> y = aikit.broadcast_to(x, (3, 2))
    >>> print(y)
    aikit.array([[0.1, 0.3],
            [0.1, 0.3],
            [0.1, 0.3]])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([1, 2, 3]),
    ...                   b=aikit.array([4, 5, 6]))
    >>> y = aikit.broadcast_to(x, (3, 3))
    >>> print(y)
    {
        a: aikit.array([[1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3]]),
        b: aikit.array([[4, 5, 6],
                    [4, 5, 6],
                    [4, 5, 6]])
    }
    """
    return current_backend(x).broadcast_to(x, shape, out=out)


@handle_exceptions
@handle_nestable
@inputs_to_aikit_arrays
@handle_array_function
@handle_device
def can_cast(
    from_: Union[aikit.Dtype, aikit.Array, aikit.NativeArray],
    to: aikit.Dtype,
    /,
) -> bool:
    """Determine if one data type can be cast to another data type according to
    :ref:`type- promotion` rules.

    Parameters
    ----------
    from_
        input data type or array from which to cast.
    to
        desired data type.

    Returns
    -------
    ret
        ``True`` if the cast can occur according to :ref:`type-promotion` rules;
        otherwise, ``False``.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.can_cast.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Dtype` input:

    >>> print(aikit.can_cast(aikit.uint8, aikit.int32))
    True

    >>> print(aikit.can_cast(aikit.float64, 'int64'))
    False

    With :class:`aikit.Array` input:

    >>> x = aikit.array([1., 2., 3.])
    >>> print(aikit.can_cast(x, aikit.float64))
    True

    With :class:`aikit.NativeArray` input:

    >>> x = aikit.native_array([[-1, -1, -1],
    ...                       [1, 1, 1]],
    ...                       dtype='int16')
    >>> print(aikit.can_cast(x, 'uint8'))
    False

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),
    ...                   b=aikit.array([3, 4, 5]))
    >>> print(aikit.can_cast(x, 'int64'))
    {
        a: False,
        b: True
    }
    """
    if isinstance(from_, (aikit.Array, aikit.NativeArray)):
        from_ = from_.dtype
    dtype = aikit.promote_types(from_, to)
    return dtype == to


@handle_exceptions
@handle_backend_invalid
@inputs_to_native_arrays
@handle_device
def finfo(
    type: Union[aikit.Dtype, str, aikit.Array, aikit.NativeArray],
    /,
) -> Finfo:
    """Machine limits for floating-point data types.

    Parameters
    ----------
    type
        the kind of floating-point data-type about which to get information.

    Returns
    -------
    ret
        an object having the following attributes:

        - **bits**: *int*

        number of bits occupied by the floating-point data type.

        - **eps**: *float*

        difference between 1.0 and the next smallest representable floating-point
        number larger than 1.0 according to the IEEE-754 standard.

        - **max**: *float*

        largest representable number.

        - **min**: *float*

        smallest representable number.

        - **smallest_normal**: *float*

        smallest positive floating-point number with full precision.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.finfo.html>`_
    in the standard.

    Examples
    --------
    With :class:`aikit.Dtype` input:

    >>> y = aikit.finfo(aikit.float32)
    >>> print(y)
    finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)

    With :code:`str` input:

    >>> y = aikit.finfo('float32')
    >>> print(y)
    finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)

    With :class:`aikit.Array` input:

    >>> x = aikit.array([1.3,2.1,3.4], dtype=aikit.float64)
    >>> print(aikit.finfo(x))
    finfo(resolution=1e-15, min=-1.7976931348623157e+308, /
    max=1.7976931348623157e+308, dtype=float64)

    >>> x = aikit.array([0.7,8.4,3.14], dtype=aikit.float16)
    >>> print(aikit.finfo(x))
    finfo(resolution=0.001, min=-6.55040e+04, max=6.55040e+04, dtype=float16)

    With :class:`aikit.Container` input:

    >>> c = aikit.Container(x=aikit.array([-9.5,1.8,-8.9], dtype=aikit.float16),
    ...                   y=aikit.array([7.6,8.1,1.6], dtype=aikit.float64))
    >>> print(aikit.finfo(c))
    {
        x: finfo(resolution=0.001, min=-6.55040e+04, max=6.55040e+04, dtype=float16),
        y: finfo(resolution=1e-15, min=-1.7976931348623157e+308, /
            max=1.7976931348623157e+308, dtype=float64)
    }
    """
    return current_backend(None).finfo(type)


@handle_exceptions
@handle_backend_invalid
@inputs_to_native_arrays
@handle_device
def iinfo(
    type: Union[aikit.Dtype, str, aikit.Array, aikit.NativeArray],
    /,
) -> Iinfo:
    """Machine limits for integer data types.

    Parameters
    ----------
    type
        the kind of integer data-type about which to get information.

    Returns
    -------
    ret
        a class with that encapsules the following attributes:

        - **bits**: *int*

        number of bits occupied by the type.

        - **max**: *int*

        largest representable number.

        - **min**: *int*

        smallest representable number.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.iinfo.html>`_
    in the standard.

    Examples
    --------
    With :class:`aikit.Dtype` input:

    >>> aikit.iinfo(aikit.int32)
    iinfo(min=-2147483648, max=2147483647, dtype=int32)

    With :code:`str` input:

    >>> aikit.iinfo('int32')
    iinfo(min=-2147483648, max=2147483647, dtype=int32)

    With :class:`aikit.Array` input:

    >>> x = aikit.array([13,21,34], dtype=aikit.int8)
    >>> aikit.iinfo(x)
    iinfo(min=-128, max=127, dtype=int8)

    With :class:`aikit.NativeArray` input:

    >>> x = aikit.native_array([7,84,314], dtype=aikit.int64)
    >>> aikit.iinfo(x)
    iinfo(min=-9223372036854775808, max=9223372036854775807, dtype=int64)

    With :class:`aikit.Container` input:

    >>> c = aikit.Container(x=aikit.array([0,1800,89], dtype=aikit.uint16),
    ...                   y=aikit.array([76,81,16], dtype=aikit.uint32))
    >>> aikit.iinfo(c)
    {
        x: iinfo(min=0, max=65535, dtype=uint16),
        y: iinfo(min=0, max=4294967295, dtype=uint32)
    }
    """
    return current_backend(None).iinfo(type)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@inputs_to_native_arrays
@handle_device
def result_type(
    *arrays_and_dtypes: Union[aikit.Array, aikit.NativeArray, aikit.Dtype]
) -> aikit.Dtype:
    """Return the dtype that results from applying the type promotion rules
    (see :ref:`type-promotion`) to the arguments.

    .. note::
    If provided mixed dtypes (e.g., integer and floating-point), the returned dtype
    will be implementation-specific.

    Parameters
    ----------
    arrays_and_dtypes
        an arbitrary number of input arrays and/or dtypes.

    Returns
    -------
    ret
        the dtype resulting from an operation involving the input arrays and dtypes.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.result_type.html>`_
    in the standard.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([3, 4, 5])
    >>> y = aikit.array([3., 4., 5.])
    >>> d = aikit.result_type(x, y)
    >>> print(d)
    float32

    With :class:`aikit.Dtype` input:

    >>> d = aikit.result_type(aikit.uint8, aikit.uint64)
    >>> print(d)
    uint64

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a = aikit.array([3, 4, 5]))
    >>> d = x.a.dtype
    >>> print(d)
    int32

    >>> x = aikit.Container(a = aikit.array([3, 4, 5]))
    >>> d = aikit.result_type(x, aikit.float64)
    >>> print(d)
    {
        a: float64
    }
    """
    return current_backend(arrays_and_dtypes[0]).result_type(*arrays_and_dtypes)


# Extra #
# ------#

default_dtype_stack = []
default_float_dtype_stack = []
default_int_dtype_stack = []
default_uint_dtype_stack = []
default_complex_dtype_stack = []


class DefaultDtype:
    """Aikit's DefaultDtype class."""

    def __init__(self, dtype: aikit.Dtype):
        self._dtype = dtype

    def __enter__(self):
        set_default_dtype(self._dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_dtype()
        if self and (exc_type is not None):
            raise exc_val
        return self


class DefaultFloatDtype:
    """Aikit's DefaultFloatDtype class."""

    def __init__(self, float_dtype: aikit.Dtype):
        self._float_dtype = float_dtype

    def __enter__(self):
        set_default_float_dtype(self._float_dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_float_dtype()
        if self and (exc_type is not None):
            raise exc_val
        return self


class DefaultIntDtype:
    """Aikit's DefaultIntDtype class."""

    def __init__(self, int_dtype: aikit.Dtype):
        self._int_dtype = int_dtype

    def __enter__(self):
        set_default_int_dtype(self._int_dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_int_dtype()
        if self and (exc_type is not None):
            raise exc_val
        return self


class DefaultUintDtype:
    """Aikit's DefaultUintDtype class."""

    def __init__(self, uint_dtype: aikit.UintDtype):
        self._uint_dtype = uint_dtype

    def __enter__(self):
        set_default_uint_dtype(self._uint_dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_uint_dtype()
        if self and (exc_type is not None):
            raise exc_val
        return self


class DefaultComplexDtype:
    """Aikit's DefaultComplexDtype class."""

    def __init__(self, complex_dtype: aikit.Dtype):
        self._complex_dtype = complex_dtype

    def __enter__(self):
        set_default_complex_dtype(self._complex_dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_complex_dtype()
        if self and (exc_type is not None):
            raise exc_val
        return self


@handle_exceptions
def dtype_bits(dtype_in: Union[aikit.Dtype, aikit.NativeDtype, str], /) -> int:
    """Get the number of bits used for representing the input data type.

    Parameters
    ----------
    dtype_in
        The data type to determine the number of bits for.

    Returns
    -------
    ret
        The number of bits used to represent the data type.

    Examples
    --------
    With :class:`aikit.Dtype` inputs:

    >>> x = aikit.dtype_bits(aikit.float32)
    >>> print(x)
    32

    >>> x = aikit.dtype_bits('int64')
    >>> print(x)
    64

    With :class:`aikit.NativeDtype` inputs:

    >>> x = aikit.dtype_bits(aikit.native_bool)
    >>> print(x)
    1
    """
    return current_backend(dtype_in).dtype_bits(dtype_in)


@handle_exceptions
def is_hashable_dtype(dtype_in: Union[aikit.Dtype, aikit.NativeDtype], /) -> bool:
    """Check if the given data type is hashable or not.

    Parameters
    ----------
    dtype_in
        The data type to check.

    Returns
    -------
    ret
        True if data type is hashable else False
    """
    # Doing something like isinstance(dtype_in, collections.abc.Hashable)
    # fails where the `__hash__` method is overridden to simply raise an
    # exception.
    # [See `tensorflow.python.trackable.data_structures.ListWrapper`]
    try:
        hash(dtype_in)
        return True
    except TypeError:
        return False


@handle_exceptions
def as_aikit_dtype(dtype_in: Union[aikit.Dtype, str], /) -> aikit.Dtype:
    """Convert native data type to string representation.

    Parameters
    ----------
    dtype_in
        The data type to convert to string.

    Returns
    -------
    ret
        data type string 'float32'
    """
    return current_backend(None).as_aikit_dtype(dtype_in)


@handle_exceptions
def as_native_dtype(dtype_in: Union[aikit.Dtype, aikit.NativeDtype], /) -> aikit.NativeDtype:
    """Convert data type string representation to native data type.

    Parameters
    ----------
    dtype_in
        The data type string to convert to native data type.

    Returns
    -------
    ret
        data type e.g. aikit.float32.
    """
    return current_backend(None).as_native_dtype(dtype_in)


def _check_float64(input) -> bool:
    if aikit.is_array(input):
        return aikit.dtype(input) == "float64"
    if math.isfinite(input):
        m, e = math.frexp(input)
        return (abs(input) > 3.4028235e38) or (e < -126) or (e > 128)
    return False


def _check_complex128(input) -> bool:
    if aikit.is_array(input):
        return aikit.dtype(input) == "complex128"
    elif isinstance(input, np.ndarray):
        return str(input.dtype) == "complex128"
    if hasattr(input, "real") and hasattr(input, "imag"):
        return _check_float64(input.real) and _check_float64(input.imag)
    return False


@handle_exceptions
def closest_valid_dtype(type: Union[aikit.Dtype, str, None], /) -> Union[aikit.Dtype, str]:
    """Determine the closest valid datatype to the datatype passed as input.

    Parameters
    ----------
    type
        The data type for which to check the closest valid type for.

    Returns
    -------
    ret
        The closest valid data type as a native aikit.Dtype

    Examples
    --------
    With :class:`aikit.Dtype` input:

    >>> xType = aikit.float16
    >>> yType = aikit.closest_valid_dtype(xType)
    >>> print(yType)
    float16

    With :class:`aikit.NativeDtype` inputs:

    >>> xType = aikit.native_uint16
    >>> yType = aikit.closest_valid_dtype(xType)
    >>> print(yType)
    uint16

    With :code:`str` input:

    >>> xType = 'int32'
    >>> yType = aikit.closest_valid_dtype(xType)
    >>> print(yType)
    int32
    """
    return current_backend(type).closest_valid_dtype(type)


@handle_exceptions
@handle_nestable
@inputs_to_aikit_arrays
def default_float_dtype(
    *,
    input: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
    float_dtype: Optional[Union[aikit.FloatDtype, aikit.NativeDtype]] = None,
    as_native: bool = False,
) -> Union[aikit.Dtype, str, aikit.NativeDtype]:
    """
    Parameters
    ----------
    input
        Number or array for inferring the float dtype.
    float_dtype
        The float dtype to be returned.
    as_native
        Whether to return the float dtype as native dtype.

    Returns
    -------
        Return ``float_dtype`` as native or aikit dtype if provided, else
        if ``input`` is given, return its float dtype, otherwise return the
        global default float dtype.

    Examples
    --------
    >>> aikit.default_float_dtype()
    'float32'

    >>> aikit.set_default_float_dtype(aikit.FloatDtype("float64"))
    >>> aikit.default_float_dtype()
    'float64'

    >>> aikit.default_float_dtype(float_dtype=aikit.FloatDtype("float16"))
    'float16'

    >>> aikit.default_float_dtype(input=4294.967346)
    'float32'

    >>> x = aikit.array([9.8,8.9], dtype="float16")
    >>> aikit.default_float_dtype(input=x)
    'float16'
    """
    global default_float_dtype_stack
    if aikit.exists(float_dtype):
        if as_native is True:
            return aikit.as_native_dtype(float_dtype)
        return aikit.FloatDtype(aikit.as_aikit_dtype(float_dtype))
    as_native = aikit.default(as_native, False)
    if aikit.exists(input):
        if aikit.is_array(input):
            ret = aikit.dtype(input)
        elif isinstance(input, np.ndarray):
            ret = str(input.dtype)
        elif isinstance(input, (list, tuple, dict)):
            if aikit.nested_argwhere(
                input, lambda x: _check_float64(x), stop_after_n_found=1
            ):
                ret = aikit.float64
            else:
                if not default_float_dtype_stack:
                    def_dtype = default_dtype()
                    if aikit.is_float_dtype(def_dtype):
                        ret = def_dtype
                    else:
                        ret = "float32"
                else:
                    ret = default_float_dtype_stack[-1]
        elif isinstance(input, Number):
            if _check_float64(input):
                ret = aikit.float64
            else:
                if not default_float_dtype_stack:
                    def_dtype = default_dtype()
                    if aikit.is_float_dtype(def_dtype):
                        ret = def_dtype
                    else:
                        ret = "float32"
                else:
                    ret = default_float_dtype_stack[-1]
    else:
        if not default_float_dtype_stack:
            def_dtype = default_dtype()
            if aikit.is_float_dtype(def_dtype):
                ret = def_dtype
            else:
                ret = "float32"
        else:
            ret = default_float_dtype_stack[-1]
    if as_native:
        return aikit.as_native_dtype(ret)
    return aikit.FloatDtype(aikit.as_aikit_dtype(ret))


@handle_exceptions
def infer_default_dtype(
    dtype: Union[aikit.Dtype, aikit.NativeDtype, str], as_native: bool = False
) -> Union[aikit.Dtype, aikit.NativeDtype]:
    """Summary.

    Parameters
    ----------
    dtype

    as_native
        (Default value = False)

    Returns
    -------
        Return the default data type for the “kind” (integer or floating-point) of dtype

    Examples
    --------
    >>> aikit.set_default_int_dtype("int32")
    >>> aikit.infer_default_dtype("int8")
    'int8'

    >>> aikit.set_default_float_dtype("float64")
    >>> aikit.infer_default_dtype("float32")
    'float64'

    >>> aikit.set_default_uint_dtype("uint32")
    >>> x = aikit.array([0], dtype="uint64")
    >>> aikit.infer_default_dtype(x.dtype)
    'uint32'
    """
    if aikit.is_complex_dtype(dtype):
        default_dtype = aikit.default_complex_dtype(as_native=as_native)
    elif aikit.is_float_dtype(dtype):
        default_dtype = aikit.default_float_dtype(as_native=as_native)
    elif aikit.is_uint_dtype(dtype):
        default_dtype = aikit.default_uint_dtype(as_native=as_native)
    elif aikit.is_int_dtype(dtype):
        default_dtype = aikit.default_int_dtype(as_native=as_native)
    elif as_native:
        default_dtype = aikit.as_native_dtype("bool")
    else:
        default_dtype = aikit.as_aikit_dtype("bool")
    return default_dtype


@handle_exceptions
@inputs_to_aikit_arrays
def default_dtype(
    *,
    dtype: Optional[Union[aikit.Dtype, str]] = None,
    item: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
    as_native: bool = False,
) -> Union[aikit.Dtype, aikit.NativeDtype, str]:
    """
    Parameters
    ----------
    item
        Number or array for inferring the dtype.
    dtype
        The dtype to be returned.
    as_native
        Whether to return the dtype as native dtype.

    Returns
    -------
        Return ``dtype`` as native or aikit dtype if provided, else
        if ``item`` is given, return its dtype, otherwise return the
        global default dtype.

    Examples
    --------
    >>> aikit.default_dtype()
    'float32'

    >>> aikit.set_default_dtype(aikit.bool)
    >>> aikit.default_dtype()
    'bool'

    >>> aikit.set_default_dtype(aikit.int16)
    >>> aikit.default_dtype()
    'int16'

    >>> aikit.set_default_dtype(aikit.float64)
    >>> aikit.default_dtype()
    'float64'

    >>> aikit.default_dtype(dtype="int32")
    'int32'

    >>> aikit.default_dtype(dtype=aikit.float16)
    'float16'

    >>> aikit.default_dtype(item=53.234)
    'float64'

    >>> aikit.default_dtype(item=[1, 2, 3])
    'int32'

    >>> x = aikit.array([5.2, 9.7], dtype="complex128")
    >>> aikit.default_dtype(item=x)
    'complex128'
    """
    if aikit.exists(dtype):
        if as_native is True:
            return aikit.as_native_dtype(dtype)
        return aikit.as_aikit_dtype(dtype)
    as_native = aikit.default(as_native, False)
    if aikit.exists(item):
        if hasattr(item, "override_dtype_check"):
            return item.override_dtype_check()
        elif isinstance(item, (list, tuple, dict)) and len(item) == 0:
            pass
        elif aikit.is_complex_dtype(item):
            return aikit.default_complex_dtype(input=item, as_native=as_native)
        elif aikit.is_float_dtype(item):
            return aikit.default_float_dtype(input=item, as_native=as_native)
        elif aikit.is_uint_dtype(item):
            return aikit.default_int_dtype(input=item, as_native=as_native)
        elif aikit.is_int_dtype(item):
            return aikit.default_int_dtype(input=item, as_native=as_native)
        elif as_native:
            return aikit.as_native_dtype("bool")
        else:
            return "bool"
    global default_dtype_stack
    if not default_dtype_stack:
        global default_float_dtype_stack
        if default_float_dtype_stack:
            ret = default_float_dtype_stack[-1]
        else:
            ret = "float32"
    else:
        ret = default_dtype_stack[-1]
    if as_native:
        return aikit.as_native_dtype(ret)
    return aikit.as_aikit_dtype(ret)


@handle_exceptions
@inputs_to_aikit_arrays
def default_int_dtype(
    *,
    input: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
    int_dtype: Optional[Union[aikit.IntDtype, aikit.NativeDtype]] = None,
    as_native: bool = False,
) -> Union[aikit.IntDtype, aikit.NativeDtype]:
    """
    Parameters
    ----------
    input
        Number or array for inferring the int dtype.
    int_dtype
        The int dtype to be returned.
    as_native
        Whether to return the int dtype as native dtype.

    Returns
    -------
        Return ``int_dtype`` as native or aikit dtype if provided, else
        if ``input`` is given, return its int dtype, otherwise return the
        global default int dtype.

    Examples
    --------
    >>> aikit.set_default_int_dtype(aikit.intDtype("int16"))
    >>> aikit.default_int_dtype()
    'int16'

    >>> aikit.default_int_dtype(input=4294967346)
    'int64'

    >>> aikit.default_int_dtype(int_dtype=aikit.intDtype("int8"))
    'int8'

    >>> x = aikit.array([9,8], dtype="int32")
    >>> aikit.default_int_dtype(input=x)
    'int32'
    """
    global default_int_dtype_stack
    if aikit.exists(int_dtype):
        if as_native is True:
            return aikit.as_native_dtype(int_dtype)
        return aikit.IntDtype(aikit.as_aikit_dtype(int_dtype))
    as_native = aikit.default(as_native, False)
    if aikit.exists(input):
        if aikit.is_array(input):
            ret = aikit.dtype(input)
        elif isinstance(input, aikit.Shape):
            ret = aikit.default_int_dtype()
        elif isinstance(input, np.ndarray):
            ret = str(input.dtype)
        elif isinstance(input, (list, tuple, dict)):
            if aikit.nested_argwhere(
                input,
                lambda x: (
                    aikit.dtype(x) == "uint64"
                    if aikit.is_array(x)
                    else x > 9223372036854775807 and x != aikit.inf
                ),
                stop_after_n_found=1,
            ):
                ret = aikit.uint64
            elif aikit.nested_argwhere(
                input,
                lambda x: (
                    aikit.dtype(x) == "int64"
                    if aikit.is_array(x)
                    else x > 2147483647 and x != aikit.inf
                ),
                stop_after_n_found=1,
            ):
                ret = aikit.int64
            else:
                if not default_int_dtype_stack:
                    def_dtype = aikit.default_dtype()
                    if aikit.is_int_dtype(def_dtype):
                        ret = def_dtype
                    else:
                        ret = "int32"
                else:
                    ret = default_int_dtype_stack[-1]
        elif isinstance(input, Number):
            if (
                input > 9223372036854775807
                and input != aikit.inf
                and aikit.backend != "torch"
            ):
                ret = aikit.uint64
            elif input > 2147483647 and input != aikit.inf:
                ret = aikit.int64
            else:
                if not default_int_dtype_stack:
                    def_dtype = aikit.default_dtype()
                    if aikit.is_int_dtype(def_dtype):
                        ret = def_dtype
                    else:
                        ret = "int32"
                else:
                    ret = default_int_dtype_stack[-1]
    else:
        if not default_int_dtype_stack:
            def_dtype = aikit.default_dtype()
            if aikit.is_int_dtype(def_dtype):
                ret = def_dtype
            else:
                ret = "int32"
        else:
            ret = default_int_dtype_stack[-1]
    if as_native:
        return aikit.as_native_dtype(ret)
    return aikit.IntDtype(aikit.as_aikit_dtype(ret))


@handle_exceptions
@inputs_to_aikit_arrays
def default_uint_dtype(
    *,
    input: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
    uint_dtype: Optional[Union[aikit.UintDtype, aikit.NativeDtype]] = None,
    as_native: bool = False,
) -> Union[aikit.UintDtype, aikit.NativeDtype]:
    """
    Parameters
    ----------
    input
        Number or array for inferring the uint dtype.
    uint_dtype
        The uint dtype to be returned.
    as_native
        Whether to return the uint dtype as native dtype.

    Returns
    -------
        Return ``uint_dtype`` as native or aikit dtype if provided, else
        if ``input`` is given, return its uint dtype, otherwise return the
        global default uint dtype.

    Examples
    --------
    >>> aikit.set_default_uint_dtype(aikit.UintDtype("uint16"))
    >>> aikit.default_uint_dtype()
    'uint16'

    >>> aikit.default_uint_dtype(input=4294967346)
    'uint64'

    >>> aikit.default_uint_dtype(uint_dtype=aikit.UintDtype("uint8"))
    'uint8'

    >>> x = aikit.array([9,8], dtype="uint32")
    >>> aikit.default_uint_dtype(input=x)
    'uint32'
    """
    global default_uint_dtype_stack
    if aikit.exists(uint_dtype):
        if as_native is True:
            return aikit.as_native_dtype(uint_dtype)
        return aikit.UintDtype(aikit.as_aikit_dtype(uint_dtype))
    as_native = aikit.default(as_native, False)
    if aikit.exists(input):
        if aikit.is_array(input):
            ret = aikit.dtype(input)
        elif isinstance(input, np.ndarray):
            ret = input.dtype
        elif isinstance(input, (list, tuple, dict)):

            def is_native(x):
                return aikit.is_native_array(x)

            if aikit.nested_argwhere(
                input,
                lambda x: (
                    aikit.dtype(x) == "uint64"
                    if is_native(x)
                    else x > 9223372036854775807 and x != aikit.inf
                ),
                stop_after_n_found=1,
            ):
                ret = aikit.uint64
            else:
                if default_uint_dtype_stack:
                    ret = default_uint_dtype_stack[-1]
                else:
                    def_dtype = aikit.default_dtype()
                    if aikit.is_uint_dtype(def_dtype):
                        ret = def_dtype
                    else:
                        ret = "uint32"
        elif isinstance(input, Number):
            if input > 4294967295 and input != aikit.inf and aikit.backend != "torch":
                ret = aikit.uint64
            else:
                if default_uint_dtype_stack:
                    ret = default_uint_dtype_stack[-1]
                else:
                    def_dtype = aikit.default_dtype()
                    if aikit.is_uint_dtype(def_dtype):
                        ret = def_dtype
                    else:
                        ret = "uint32"
    else:
        if default_uint_dtype_stack:
            ret = default_uint_dtype_stack[-1]
        else:
            def_dtype = aikit.default_dtype()
            if aikit.is_uint_dtype(def_dtype):
                ret = def_dtype
            else:
                ret = "uint32"
    if as_native:
        return aikit.as_native_dtype(ret)
    return aikit.UintDtype(aikit.as_aikit_dtype(ret))


@handle_exceptions
@handle_nestable
@inputs_to_aikit_arrays
@handle_device
def default_complex_dtype(
    *,
    input: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
    complex_dtype: Optional[Union[aikit.ComplexDtype, aikit.NativeDtype]] = None,
    as_native: bool = False,
) -> Union[aikit.Dtype, str, aikit.NativeDtype]:
    """
    Parameters
    ----------
    input
        Number or array for inferring the complex dtype.
    complex_dtype
        The complex dtype to be returned.
    as_native
        Whether to return the complex dtype as native dtype.

    Returns
    -------
        Return ``complex_dtype`` as native or aikit dtype if provided, else
        if ``input`` is given, return its complex dtype, otherwise return the
        global default complex dtype.

    Examples
    --------
    >>> aikit.default_complex_dtype()
    'complex64'

    >>> aikit.set_default_complex_dtype(aikit.ComplexDtype("complex64"))
    >>> aikit.default_complex_dtype()
    'complex64'

    >>> aikit.default_complex_dtype(complex_dtype=aikit.ComplexDtype("complex128"))
    'complex128'

    >>> aikit.default_complex_dtype(input=4294.967346)
    'complex64'

    >>> x = aikit.array([9.8,8.9], dtype="complex128")
    >>> aikit.default_complex_dtype(input=x)
    'complex128'
    """
    global default_complex_dtype_stack
    if aikit.exists(complex_dtype):
        if as_native is True:
            return aikit.as_native_dtype(complex_dtype)
        return aikit.ComplexDtype(aikit.as_aikit_dtype(complex_dtype))
    as_native = aikit.default(as_native, False)
    if aikit.exists(input):
        if aikit.is_array(input):
            ret = aikit.dtype(input)
        elif isinstance(input, np.ndarray):
            ret = str(input.dtype)
        elif isinstance(input, (list, tuple, dict)):
            if aikit.nested_argwhere(
                input, lambda x: _check_complex128(x), stop_after_n_found=1
            ):
                ret = aikit.complex128
            else:
                if not default_complex_dtype_stack:
                    def_dtype = default_dtype()
                    if aikit.is_complex_dtype(def_dtype):
                        ret = def_dtype
                    else:
                        ret = "complex64"
                else:
                    ret = default_complex_dtype_stack[-1]
        elif isinstance(input, Number):
            if _check_complex128(input):
                ret = aikit.complex128
            else:
                if not default_complex_dtype_stack:
                    def_dtype = default_dtype()
                    if aikit.is_complex_dtype(def_dtype):
                        ret = def_dtype
                    else:
                        ret = "complex64"
                else:
                    ret = default_complex_dtype_stack[-1]
    else:
        if not default_complex_dtype_stack:
            def_dtype = default_dtype()
            if aikit.is_complex_dtype(def_dtype):
                ret = def_dtype
            else:
                ret = "complex64"
        else:
            ret = default_complex_dtype_stack[-1]
    if as_native:
        return aikit.as_native_dtype(ret)
    return aikit.ComplexDtype(aikit.as_aikit_dtype(ret))


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@inputs_to_native_arrays
@handle_device
def dtype(
    x: Union[aikit.Array, aikit.NativeArray], *, as_native: bool = False
) -> Union[aikit.Dtype, aikit.NativeDtype]:
    """Get the data type for input array x.

    Parameters
    ----------
    x
        Tensor for which to get the data type.
    as_native
        Whether or not to return the dtype in string format. Default is ``False``.

    Returns
    -------
    ret
        Data type of the array.

    Examples
    --------
    With :class:`aikit.Array` inputs:

    >>> x1 = aikit.array([1.0, 2.0, 3.5, 4.5, 5, 6])
    >>> y = aikit.dtype(x1)
    >>> print(y)
    float32

    With :class:`aikit.NativeArray` inputs:

    >>> x1 = aikit.native_array([1, 0, 1, -1, 0])
    >>> y = aikit.dtype(x1)
    >>> print(y)
    int32

    With :class:`aikit.Container` inputs:

    >>> x = aikit.Container(a=aikit.native_array([1.0, 2.0, -1.0, 4.0, 1.0]),
    ...                   b=aikit.native_array([1, 0, 0, 0, 1]))
    >>> y = aikit.dtype(x.a)
    >>> print(y)
    float32
    """
    return current_backend(x).dtype(x, as_native=as_native)


@handle_exceptions
@handle_nestable
def function_supported_dtypes(fn: Callable, recurse: bool = True) -> Union[Tuple, dict]:
    """Return the supported data types of the current backend's function. The
    function returns a dict containing the supported dtypes for the
    compositional and primary implementations in case of partial mixed
    functions.

    Parameters
    ----------
    fn
        The function to check for the supported dtype attribute
    recurse
        Whether to recurse into used aikit functions. Default is ``True``.

    Returns
    -------
    ret
        Tuple or dict containing the supported dtypes of the function

    Examples
    --------
    >>> print(aikit.function_supported_dtypes(aikit.acosh))
    ('bool', 'float64', 'int64', 'uint8', 'int8', 'float32', 'int32', 'int16', \
    'bfloat16')
    """
    aikit.utils.assertions.check_true(
        _is_valid_dtypes_attributes(fn),
        "supported_dtypes and unsupported_dtypes attributes cannot both exist "
        "in a particular backend",
    )
    if hasattr(fn, "partial_mixed_handler"):
        return {
            "compositional": function_supported_dtypes(fn.compos, recurse=recurse),
            "primary": _get_dtypes(fn, complement=False),
        }
    else:
        supported_dtypes = set(_get_dtypes(fn, complement=False))
        if recurse:
            supported_dtypes = _nested_get(
                fn, supported_dtypes, set.intersection, function_supported_dtypes
            )
    return (
        supported_dtypes
        if isinstance(supported_dtypes, dict)
        else tuple(supported_dtypes)
    )


@handle_exceptions
@handle_nestable
def function_unsupported_dtypes(
    fn: Callable, recurse: bool = True
) -> Union[Tuple, dict]:
    """Return the unsupported data types of the current backend's function. The
    function returns a dict containing the unsupported dtypes for the
    compositional and primary implementations in case of partial mixed
    functions.

    Parameters
    ----------
    fn
        The function to check for the unsupported dtype attribute
    recurse
        Whether to recurse into used aikit functions. Default is ``True``.

    Returns
    -------
    ret
        Tuple or dict containing the unsupported dtypes of the function

    Examples
    --------
    >>> aikit.set_backend('torch')
    >>> print(aikit.function_unsupported_dtypes(aikit.acosh))
    ('float16','uint16','uint32','uint64')
    """
    aikit.utils.assertions.check_true(
        _is_valid_dtypes_attributes(fn),
        "supported_dtypes and unsupported_dtypes attributes cannot both exist "
        "in a particular backend",
    )
    if hasattr(fn, "partial_mixed_handler"):
        return {
            "compositional": function_unsupported_dtypes(fn.compos, recurse=recurse),
            "primary": _get_dtypes(fn, complement=True),
        }
    else:
        unsupported_dtypes = set(_get_dtypes(fn, complement=True))
        if recurse:
            unsupported_dtypes = _nested_get(
                fn, unsupported_dtypes, set.union, function_unsupported_dtypes
            )

    return (
        unsupported_dtypes
        if isinstance(unsupported_dtypes, dict)
        else tuple(unsupported_dtypes)
    )


@handle_exceptions
def invalid_dtype(dtype_in: Union[aikit.Dtype, aikit.NativeDtype, str, None], /) -> bool:
    """Determine whether the provided data type is not support by the current
    framework.

    Parameters
    ----------
    dtype_in
        The data type for which to check for backend non-support

    Returns
    -------
    ret
        Boolean, whether the data-type string is un-supported.

    Examples
    --------
    >>> print(aikit.invalid_dtype(None))
    False

    >>> print(aikit.invalid_dtype("uint64"))
    False

    >>> print(aikit.invalid_dtype(aikit.float64))
    False

    >>> print(aikit.invalid_dtype(aikit.native_uint8))
    False
    """
    if dtype_in is None:
        return False
    return aikit.as_aikit_dtype(dtype_in) in aikit.invalid_dtypes


@handle_exceptions
@handle_nestable
@inputs_to_native_arrays
def is_bool_dtype(
    dtype_in: Union[aikit.Dtype, str, aikit.Array, aikit.NativeArray, Number],
    /,
) -> bool:
    """Determine whether the input data type is a bool data type.

    Parameters
    ----------
    dtype_in
        input data type to test.

    Returns
    -------
    ret
        "True" if the input data type is a bool, otherwise "False".

    Both the description and the type hints above assumes an array input for
    simplicity but this function is *nestable*, and therefore also accepts
    :class:`aikit.Container` instances in place of any of the arguments.
    """
    if aikit.is_array(dtype_in):
        dtype_in = aikit.dtype(dtype_in)
    elif isinstance(dtype_in, np.ndarray):
        return "bool" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, (bool, np.bool_)) and not isinstance(dtype_in, bool)
    elif isinstance(dtype_in, (list, tuple, dict)):
        return bool(
            aikit.nested_argwhere(
                dtype_in,
                lambda x: isinstance(x, (bool, np.bool_)) and x is not int,
            )
        )
    return "bool" in aikit.as_aikit_dtype(dtype_in)


@handle_exceptions
@handle_nestable
@inputs_to_native_arrays
def is_int_dtype(
    dtype_in: Union[aikit.Dtype, str, aikit.Array, aikit.NativeArray, Number],
    /,
) -> bool:
    """Determine whether the input data type is an int data type.

    Parameters
    ----------
    dtype_in
        input data type to test.

    Returns
    -------
    ret
        "True" if the input data type is an integer, otherwise "False".

    Both the description and the type hints above assumes an array input for
    simplicity but this function is *nestable*, and therefore also accepts
    :class:`aikit.Container` instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Dtype` input:

    >>> x = aikit.is_int_dtype(aikit.float64)
    >>> print(x)
    False

    With :class:`aikit.Array` input:

    >>> x = aikit.array([1., 2., 3.])
    >>> print(aikit.is_int_dtype(x), x.dtype)
    False float32

    With :class:`aikit.NativeArray` input:

    >>> x = aikit.native_array([[-1, -1, -1], [1, 1, 1]], dtype=aikit.int16)
    >>> print(aikit.is_int_dtype(x))
    True

    With :code:`Number` input:

    >>> x = 1
    >>> print(aikit.is_int_dtype(x))
    True

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),b=aikit.array([3, 4, 5]))
    >>> print(aikit.is_int_dtype(x))
    {
        a: False,
        b: True
    }
    """
    if aikit.is_array(dtype_in):
        dtype_in = aikit.dtype(dtype_in)
    elif isinstance(dtype_in, aikit.Shape):
        dtype_in = aikit.default_int_dtype()
    elif isinstance(dtype_in, np.ndarray):
        return "int" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, (int, np.integer)) and not isinstance(
            dtype_in, bool
        )
    elif isinstance(dtype_in, (list, tuple, dict)):

        def nested_fun(x):
            return (
                isinstance(x, (int, np.integer))
                or (aikit.is_array(x) and "int" in aikit.dtype(x))
            ) and x is not bool

        return bool(aikit.nested_argwhere(dtype_in, nested_fun))
    return "int" in aikit.as_aikit_dtype(dtype_in)


@handle_exceptions
def check_float(x: Any) -> bool:
    """Check if the input is a float or a float-like object.

    Parameters
    ----------
    x
        Input to check.

    Returns
    -------
    ret
        "True" if the input is a float or a float-like object, otherwise "False".
    """
    return isinstance(x, (int, float)) and x is not bool


@handle_exceptions
@handle_nestable
@inputs_to_native_arrays
def is_float_dtype(
    dtype_in: Union[aikit.Dtype, str, aikit.Array, aikit.NativeArray, Number],
    /,
) -> bool:
    """Determine whether the input data type is a float dtype.

    Parameters
    ----------
    dtype_in
        The array or data type to check

    Returns
    -------
    ret
        Whether or not the array or data type is of a floating point dtype

    Examples
    --------
    >>> x = aikit.is_float_dtype(aikit.float32)
    >>> print(x)
    True

    >>> arr = aikit.array([1.2, 3.2, 4.3], dtype=aikit.float32)
    >>> print(aikit.is_float_dtype(arr))
    True
    """
    if aikit.is_array(dtype_in):
        dtype_in = aikit.dtype(dtype_in)
    elif isinstance(dtype_in, aikit.Shape):
        dtype_in = aikit.default_int_dtype()
    elif isinstance(dtype_in, np.ndarray):
        return "float" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, (float, np.floating))
    elif isinstance(dtype_in, (list, tuple, dict)):
        return bool(
            aikit.nested_argwhere(
                dtype_in,
                lambda x: isinstance(x, (float, np.floating))
                or (aikit.is_array(x) and "float" in aikit.dtype(x)),
            )
        )
    return "float" in as_aikit_dtype(dtype_in)


@handle_exceptions
@handle_nestable
@inputs_to_native_arrays
def is_uint_dtype(
    dtype_in: Union[aikit.Dtype, str, aikit.Array, aikit.NativeArray, Number],
    /,
) -> bool:
    """Determine whether the input data type is a uint dtype.

    Parameters
    ----------
    dtype_in
        The array or data type to check

    Returns
    -------
    ret
        Whether or not the array or data type is of a uint dtype

    Examples
    --------
    >>> aikit.is_uint_dtype(aikit.UintDtype("uint16"))
    True

    >>> aikit.is_uint_dtype(aikit.Dtype("uint8"))
    True

    >>> aikit.is_uint_dtype(aikit.IntDtype("int64"))
    False
    """
    if aikit.is_array(dtype_in):
        dtype_in = aikit.dtype(dtype_in)
    elif isinstance(dtype_in, aikit.Shape):
        dtype_in = aikit.default_int_dtype()
    elif isinstance(dtype_in, np.ndarray):
        return "uint" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, np.unsignedinteger)
    elif isinstance(dtype_in, (list, tuple, dict)):
        return aikit.nested_argwhere(
            dtype_in,
            lambda x: isinstance(x, np.unsignedinteger)
            or (aikit.is_array(x) and "uint" in aikit.dtype(x)),
        )
    return "uint" in as_aikit_dtype(dtype_in)


@handle_exceptions
@handle_nestable
@inputs_to_aikit_arrays
def is_complex_dtype(
    dtype_in: Union[aikit.Dtype, str, aikit.Array, aikit.NativeArray, Number],
    /,
) -> bool:
    """Determine whether the input data type is a complex dtype.

    Parameters
    ----------
    dtype_in
        The array or data type to check

    Returns
    -------
    ret
        Whether or not the array or data type is of a complex dtype

    Examples
    --------
    >>> aikit.is_complex_dtype(aikit.ComplexDtype("complex64"))
    True

    >>> aikit.is_complex_dtype(aikit.Dtype("complex128"))
    True

    >>> aikit.is_complex_dtype(aikit.IntDtype("int64"))
    False
    """
    if aikit.is_array(dtype_in):
        dtype_in = aikit.dtype(dtype_in)
    elif isinstance(dtype_in, aikit.Shape):
        dtype_in = aikit.default_int_dtype()
    elif isinstance(dtype_in, np.ndarray):
        return "complex" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return isinstance(dtype_in, (complex, np.complexfloating))
    elif isinstance(dtype_in, (list, tuple, dict)):
        return aikit.nested_argwhere(
            dtype_in,
            lambda x: isinstance(x, (complex, np.complexfloating))
            or (aikit.is_array(x) and "complex" in aikit.dtype(x)),
        )
    return "complex" in as_aikit_dtype(dtype_in)


@handle_exceptions
def promote_types(
    type1: Union[aikit.Dtype, aikit.NativeDtype],
    type2: Union[aikit.Dtype, aikit.NativeDtype],
    /,
    *,
    array_api_promotion: bool = False,
) -> aikit.Dtype:
    """Promote the datatypes type1 and type2, returning the data type they
    promote to.

    Parameters
    ----------
    type1
        the first of the two types to promote
    type2
        the second of the two types to promote
    array_api_promotion
        whether to only use the array api promotion rules

    Returns
    -------
    ret
        The type that both input types promote to
    """
    # in case either is of none type
    if not (type1 and type2):
        return type1 if type1 else type2
    query = [aikit.as_aikit_dtype(type1), aikit.as_aikit_dtype(type2)]
    query = tuple(query)
    if query not in aikit.promotion_table:
        query = (query[1], query[0])

    def _promote(query):
        if array_api_promotion:
            return aikit.array_api_promotion_table[query]
        return aikit.promotion_table[query]

    return _promote(query)


@handle_exceptions
def set_default_dtype(dtype: Union[aikit.Dtype, aikit.NativeDtype, str], /):
    """Set the datatype `dtype` as default data type.

    Parameters
    ----------
    dtype
        the data_type to set as default data type

    Examples
    --------
    With :class:`aikit.Dtype` input:

    >>> aikit.set_default_dtype(aikit.bool)
    >>> aikit.default_dtype_stack
    ['bool']
    >>> aikit.unset_default_dtype()

    >>> aikit.set_default_dtype("float64")
    >>> aikit.default_dtype_stack
    ['float64']
    >>> aikit.unset_default_dtype()

    With :class:`aikit.NativeDtype` input:

    >>> aikit.set_default_dtype(aikit.native_uint64)
    >>> aikit.default_dtype_stack
    ['uint64']
    """
    dtype = aikit.as_aikit_dtype(dtype)
    aikit.utils.assertions._check_jax_x64_flag(dtype)
    global default_dtype_stack
    default_dtype_stack.append(dtype)


@handle_exceptions
def set_default_float_dtype(float_dtype: Union[aikit.Dtype, str], /):
    """Set the 'float_dtype' as the default data type.

    Parameters
    ----------
    float_dtype
        The float data type to be set as the default.

    Examples
    --------
    With :class: `aikit.Dtype` input:

    >>> aikit.set_default_float_dtype(aikit.floatDtype("float64"))
    >>> aikit.default_float_dtype()
    'float64'

    >>> aikit.set_default_float_dtype(aikit.floatDtype("float32"))
    >>> aikit.default_float_dtype()
    'float32'
    """
    float_dtype = aikit.FloatDtype(aikit.as_aikit_dtype(float_dtype))
    aikit.utils.assertions._check_jax_x64_flag(float_dtype)
    global default_float_dtype_stack
    default_float_dtype_stack.append(float_dtype)


@handle_exceptions
def set_default_int_dtype(int_dtype: Union[aikit.Dtype, str], /):
    """Set the 'int_dtype' as the default data type.

    Parameters
    ----------
    int_dtype
        The integer data type to be set as the default.

    Examples
    --------
    With :class: `aikit.Dtype` input:

    >>> aikit.set_default_int_dtype(aikit.intDtype("int64"))
    >>> aikit.default_int_dtype()
    'int64'

    >>> aikit.set_default_int_dtype(aikit.intDtype("int32"))
    >>> aikit.default_int_dtype()
    'int32'
    """
    int_dtype = aikit.IntDtype(aikit.as_aikit_dtype(int_dtype))
    aikit.utils.assertions._check_jax_x64_flag(int_dtype)
    global default_int_dtype_stack
    default_int_dtype_stack.append(int_dtype)


@handle_exceptions
def set_default_uint_dtype(uint_dtype: Union[aikit.Dtype, str], /):
    """Set the uint dtype to be default.

    Parameters
    ----------
    uint_dtype
        The uint dtype to be set as default.

    Examples
    --------
    >>> aikit.set_default_uint_dtype(aikit.UintDtype("uint8"))
    >>> aikit.default_uint_dtype()
    'uint8'

    >>> aikit.set_default_uint_dtype(aikit.UintDtype("uint64"))
    >>> aikit.default_uint_dtype()
    'uint64'
    """
    uint_dtype = aikit.UintDtype(aikit.as_aikit_dtype(uint_dtype))
    aikit.utils.assertions._check_jax_x64_flag(uint_dtype)
    global default_uint_dtype_stack
    default_uint_dtype_stack.append(uint_dtype)


@handle_exceptions
def set_default_complex_dtype(complex_dtype: Union[aikit.Dtype, str], /):
    """Set the 'complex_dtype' as the default data type.

    Parameters
    ----------
    complex_dtype
        The complex data type to be set as the default.

    Examples
    --------
    With :class: `aikit.Dtype` input:

    >>> aikit.set_default_complex_dtype(aikit.ComplexDtype("complex64"))
    >>> aikit.default_complex_dtype()
    'complex64'

    >>> aikit.set_default_float_dtype(aikit.ComplexDtype("complex128"))
    >>> aikit.default_complex_dtype()
    'complex128'
    """
    complex_dtype = aikit.ComplexDtype(aikit.as_aikit_dtype(complex_dtype))
    aikit.utils.assertions._check_jax_x64_flag(complex_dtype)
    global default_complex_dtype_stack
    default_complex_dtype_stack.append(complex_dtype)


@handle_exceptions
def type_promote_arrays(
    x1: Union[aikit.Array, aikit.NativeArray],
    x2: Union[aikit.Array, aikit.NativeArray],
    /,
) -> Tuple:
    """Type promote the input arrays, returning new arrays with the shared
    correct data type.

    Parameters
    ----------
    x1
        the first of the two arrays to type promote
    x2
        the second of the two arrays to type promote

    Returns
    -------
    ret1, ret2
        The input arrays after type promotion
    """
    new_type = aikit.promote_types(aikit.dtype(x1), aikit.dtype(x2))
    return aikit.astype(x1, new_type), aikit.astype(x2, new_type)


@handle_exceptions
def unset_default_dtype():
    """Reset the current default dtype to the previous state.

    Examples
    --------
    >>> aikit.set_default_dtype(aikit.int32)
    >>> aikit.set_default_dtype(aikit.bool)
    >>> aikit.default_dtype_stack
    ['int32', 'bool']

    >>> aikit.unset_default_dtype()
    >>> aikit.default_dtype_stack
    ['int32']

    >>> aikit.unset_default_dtype()
    >>> aikit.default_dtype_stack
    []
    """
    global default_dtype_stack
    if default_dtype_stack:
        default_dtype_stack.pop(-1)


@handle_exceptions
def unset_default_float_dtype():
    """Reset the current default float dtype to the previous state.

    Examples
    --------
    >>> aikit.set_default_float_dtype(aikit.float32)
    >>> aikit.set_default_float_dtype(aikit.float64)
    >>> aikit.default_float_dtype_stack
    ['float32','float64']

    >>> aikit.unset_default_float_dtype()
    >>> aikit.default_float_dtype_stack
    ['float32']
    """
    global default_float_dtype_stack
    if default_float_dtype_stack:
        default_float_dtype_stack.pop(-1)


@handle_exceptions
def unset_default_int_dtype():
    """Reset the current default int dtype to the previous state.

    Examples
    --------
    >>> aikit.set_default_int_dtype(aikit.intDtype("int16"))
    >>> aikit.default_int_dtype()
    'int16'

    >>> aikit.unset_default_int_dtype()
    >>> aikit.default_int_dtype()
    'int32'
    """
    global default_int_dtype_stack
    if default_int_dtype_stack:
        default_int_dtype_stack.pop(-1)


@handle_exceptions
def unset_default_uint_dtype():
    """Reset the current default uint dtype to the previous state.

    Examples
    --------
    >>> aikit.set_default_uint_dtype(aikit.UintDtype("uint8"))
    >>> aikit.default_uint_dtype()
    'uint8'

    >>> aikit.unset_default_uint_dtype()
    >>> aikit.default_uint_dtype()
    'uint32'
    """
    global default_uint_dtype_stack
    if default_uint_dtype_stack:
        default_uint_dtype_stack.pop(-1)


@handle_exceptions
def unset_default_complex_dtype():
    """Reset the current default complex dtype to the previous state.

    Examples
    --------
    >>> aikit.set_default_complex_dtype(aikit.complex64)
    >>> aikit.set_default_complex_dtype(aikit.complex128)
    >>> aikit.default_complex_dtype_stack
    ['complex64','complex128']

    >>> aikit.unset_default_complex_dtype()
    >>> aikit.default_complex_dtype_stack
    ['complex64']
    """
    global default_complex_dtype_stack
    if default_complex_dtype_stack:
        default_complex_dtype_stack.pop(-1)


@handle_exceptions
def valid_dtype(dtype_in: Union[aikit.Dtype, aikit.NativeDtype, str, None], /) -> bool:
    """Determine whether the provided data type is supported by the current
    framework.

    Parameters
    ----------
    dtype_in
        The data type for which to check for backend support

    Returns
    -------
    ret
        Boolean, whether or not the data-type string is supported.

    Examples
    --------
    >>> print(aikit.valid_dtype(None))
    True

    >>> print(aikit.valid_dtype(aikit.float64))
    True

    >>> print(aikit.valid_dtype('bool'))
    True

    >>> print(aikit.valid_dtype(aikit.native_float16))
    True
    """
    if dtype_in is None:
        return True
    return aikit.as_aikit_dtype(dtype_in) in aikit.valid_dtypes


@handle_exceptions
def promote_types_of_inputs(
    x1: Union[aikit.NativeArray, Number, Iterable[Number]],
    x2: Union[aikit.NativeArray, Number, Iterable[Number]],
    /,
    *,
    array_api_promotion: bool = False,
) -> Tuple[aikit.NativeArray, aikit.NativeArray]:
    """Promote the dtype of the given native array inputs to a common dtype
    based on type promotion rules.

    While passing float or integer values or any other non-array input
    to this function, it should be noted that the return will be an
    array-like object. Therefore, outputs from this function should be
    used as inputs only for those functions that expect an array-like or
    tensor-like objects, otherwise it might give unexpected results.
    """

    def _special_case(a1, a2):
        # check for float number and integer array case
        return isinstance(a1, float) and "int" in str(a2.dtype)

    def _get_target_dtype(scalar, arr):
        # identify a good dtype to give the scalar value,
        # based on it's own type and that of the arr value
        if _special_case(scalar, arr):
            return "float64"
        elif arr.dtype == bool and not isinstance(scalar, bool):
            return None  # let aikit infer a dtype
        elif isinstance(scalar, complex) and not aikit.is_complex_dtype(arr):
            return "complex128"
        else:
            return arr.dtype

    if hasattr(x1, "dtype") and not hasattr(x2, "dtype"):
        device = aikit.default_device(item=x1, as_native=True)
        x2 = aikit.asarray(x2, dtype=_get_target_dtype(x2, x1), device=device)
    elif hasattr(x2, "dtype") and not hasattr(x1, "dtype"):
        device = aikit.default_device(item=x2, as_native=True)
        x1 = aikit.asarray(x1, dtype=_get_target_dtype(x1, x2), device=device)
    elif not (hasattr(x1, "dtype") or hasattr(x2, "dtype")):
        x1 = aikit.asarray(x1)
        x2 = aikit.asarray(x2)

    if x1.dtype != x2.dtype:
        promoted = promote_types(
            x1.dtype, x2.dtype, array_api_promotion=array_api_promotion
        )
        x1 = aikit.astype(x1, promoted, copy=False)
        x2 = aikit.astype(x2, promoted, copy=False)

    aikit.utils.assertions._check_jax_x64_flag(x1.dtype)
    return aikit.to_native(x1), aikit.to_native(x2)


@handle_exceptions
def is_native_dtype(dtype_in: Union[aikit.Dtype, aikit.NativeDtype], /) -> bool:
    """Determine whether the input dtype is a Native dtype.

    Parameters
    ----------
    dtype_in
        Determine whether the input data type is a native data type object.

    Returns
    -------
    ret
        Boolean, whether or not dtype_in is a native data type.

    Examples
    --------
    >>> aikit.set_backend('numpy')
    >>> aikit.is_native_dtype(np.int32)
    True

    >>> aikit.set_backend('numpy')
    >>> aikit.is_native_array(aikit.float64)
    False
    """
    return current_backend(None).is_native_dtype(dtype_in)
