"""Collection of general Aikit functions."""

# global
import gc
import inspect
import math
from functools import wraps
from numbers import Number
from typing import (
    Callable,
    Any,
    Union,
    List,
    Tuple,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Literal,
)
import einops
import ml_dtypes  # noqa
import numpy as np

# local
import aikit
from aikit.utils.backend import current_backend, backend_stack
from aikit.functional.aikit.gradients import _is_variable
from aikit.utils.exceptions import handle_exceptions
from aikit.func_wrapper import (
    handle_array_function,
    inputs_to_aikit_arrays,
    inputs_to_native_arrays,
    to_native_arrays_and_back,
    inputs_to_native_shapes,
    outputs_to_aikit_shapes,
    outputs_to_aikit_arrays,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
    handle_view_indexing,
    handle_device,
    handle_partial_mixed_function,
    handle_backend_invalid,
)
from aikit.functional.aikit.device import dev

FN_CACHE = {}
INF = float("inf")

precise_mode_stack = []
queue_timeout_stack = []
array_mode_stack = []
shape_array_mode_stack = []
nestable_mode_stack = []
exception_trace_mode_stack = []
inplace_mode_stack = []
trace_mode_dict = {
    "frontend": "aikit/functional/frontends",
    "aikit": "aikit/",
    "full": "",
    "none": "",
}
show_func_wrapper_trace_mode_stack = []
min_denominator_stack = []
min_base_stack = []
tmp_dir_stack = []


# Extra #
# ------#


class PreciseMode:
    """Precise Mode Context Manager."""

    # noinspection PyShadowingNames
    def __init__(self, precise_mode: bool):
        self._precise_mode = precise_mode

    def __enter__(self):
        set_precise_mode(self._precise_mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_precise_mode()
        if self and (exc_type is not None):
            raise exc_val
        return self


aikit.precise_mode = precise_mode_stack[-1] if precise_mode_stack else True


@handle_exceptions
def set_precise_mode(mode: bool) -> None:
    """Set the mode of whether to use a promotion table that avoids any
    precision loss or a compute efficient table that avoids most wider-than-
    necessary promotions.

    Parameter
    ---------
    mode
        boolean whether to use high precision promotion table

    Examples
    --------
    >>> aikit.set_precise_mode(False)
    >>> aikit.precise_mode
    False

    >>> aikit.set_precise_mode(True)
    >>> aikit.precise_mode
    True
    """
    global precise_mode_stack
    aikit.utils.assertions.check_isinstance(mode, bool)
    precise_mode_stack.append(mode)
    aikit.__setattr__("precise_mode", mode, True)
    _update_promotion_table(precise=mode)


@handle_exceptions
def unset_precise_mode() -> None:
    """Reset the mode of whether to use a promotion table that avoids any
    precision loss or a compute efficient table that avoids most wider-than-
    necessary promotions.

    Examples
    --------
    >>> aikit.set_precise_mode(False)
    >>> aikit.precise_mode
    False

    >>> aikit.unset_precise_mode()
    >>> aikit.precise_mode
    True
    """
    global precise_mode_stack
    if precise_mode_stack:
        precise_mode_stack.pop(-1)
        mode = precise_mode_stack[-1] if precise_mode_stack else True
        aikit.__setattr__("precise_mode", mode, True)
        _update_promotion_table(precise=mode)


def _update_promotion_table(precise):
    """Update the current datatype promotion table."""
    if precise:
        aikit.promotion_table = {
            **aikit.array_api_promotion_table,
            **aikit.common_extra_promotion_table,
            **aikit.precise_extra_promotion_table,
        }

    else:
        aikit.promotion_table = {
            **aikit.array_api_promotion_table,
            **aikit.common_extra_promotion_table,
            **aikit.extra_promotion_table,
        }


class ArrayMode:
    """Array Mode Context Manager."""

    # noinspection PyShadowingNames
    def __init__(self, array_mode):
        self._array_mode = array_mode

    def __enter__(self):
        set_array_mode(self._array_mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_array_mode()
        if self and (exc_type is not None):
            raise exc_val
        return self


def get_referrers_recursive(
    item: object,
    *,
    depth: int = 0,
    max_depth: Optional[int] = None,
    seen_set: Optional[set] = None,
    local_set: Optional[set] = None,
) -> aikit.Container:
    """Recursively retrieve referrers for an object.

    This function recursively fetches referrers for the specified `item` up to a given
    `max_depth`.

    Parameters
    ----------
    item
        The object for which referrers should be retrieved.
    depth
        Current depth in the recursion. (default is 0)
    max_depth
        Maximum depth of recursion. If `None`, there's no depth limit. (default is None)
    seen_set
        Set of seen referrer IDs to prevent duplicates. (default is None)
    local_set
        Set of local referrer IDs to avoid redundancy. (default is None)

    Returns
    -------
    ret
        A container representing referrers and their sub-referrers, respecting the
        `max_depth`.

    Examples
    --------
    >>> import gc
    >>> def example_function():
    ...     obj = [1, 2, 3]
    ...     return get_referrers_recursive(obj, max_depth=2)
    >>> result = example_function()
    >>> print(result)
    Container(
        'ref_id_1': Container(
            'ref_id_2': 'tracked',
            'ref_id_3': 'tracked'
        )
    )
    """
    seen_set = aikit.default(seen_set, set())
    local_set = aikit.default(local_set, set())
    ret_cont = aikit.Container(
        repr=str(item).replace(" ", ""),
        alphabetical_keys=False,
        keyword_color_dict={"repr": "magenta"},
    )

    referrers = [
        ref
        for ref in gc.get_referrers(item)
        if not (
            isinstance(ref, dict)
            and min(k in ref for k in ["depth", "max_depth", "seen_set", "local_set"])
        )
    ]

    local_set.add(str(id(referrers)))
    for ref in referrers:
        ref_id = str(id(ref))
        if ref_id in local_set or hasattr(ref, "cell_contents"):
            continue
        seen = ref_id in seen_set
        seen_set.add(ref_id)

        def get_referrers_recursive_inner():
            return get_referrers_recursive(
                ref,
                depth=depth + 1,
                max_depth=max_depth,
                seen_set=seen_set,
                local_set=local_set,
            )

        this_repr = "tracked" if seen else str(ref).replace(" ", "")

        if not seen and (not max_depth or depth < max_depth):
            val = aikit.Container(
                repr=this_repr,
                alphabetical_keys=False,
                keyword_color_dict={"repr": "magenta"},
            )

            refs = get_referrers_recursive_inner()
            for k, v in refs.items():
                val[k] = v
        else:
            val = this_repr
        ret_cont[str(ref_id)] = val

    return ret_cont


@handle_exceptions
@handle_backend_invalid
def is_native_array(
    x: Union[aikit.Array, aikit.NativeArray], /, *, exclusive: bool = False
) -> bool:
    """Determine whether the input x is an :class:`aikit.NativeArray` instance.

    Parameters
    ----------
    x
        The input to check
    exclusive
        Whether to check if the data type is exclusively an array, rather than a
        variable or traced array.

    Returns
    -------
    ret
        Boolean, whether or not x is an :class:`aikit.NativeArray`.

    Examples
    --------
    >>> x = aikit.array([0, 1, 2])
    >>> aikit.is_native_array(x)
    False

    >>> x = aikit.native_array([9.1, -8.3, 2.8, 3.0])
    >>> aikit.is_native_array(x, exclusive=True)
    True
    """
    try:
        return current_backend(x).is_native_array(x, exclusive=exclusive)
    except ValueError:
        return False


@handle_exceptions
@handle_backend_invalid
def is_aikit_array(
    x: Union[aikit.Array, aikit.NativeArray], /, *, exclusive: Optional[bool] = False
) -> bool:
    """Determine whether the input x is a valid Aikit Array.

    Parameters
    ----------
    x
        The input to check
    exclusive
        Whether to check if the data type is exclusively an array, rather than a
        variable or traced array.

    Returns
    -------
    ret
        Boolean, whether or not x is a valid Aikit Array.

    Examples
    --------
    >>> x = aikit.array([0, 1, 2])
    >>> aikit.is_aikit_array(x)
    True

    >>> x = aikit.native_array([9.1, -8.3, 2.8, 3.0])
    >>> aikit.is_aikit_array(x, exclusive=True)
    False
    """
    return isinstance(x, aikit.Array) and aikit.is_native_array(x.data, exclusive=exclusive)


@handle_exceptions
@handle_backend_invalid
def is_array(x: Any, /, *, exclusive: bool = False) -> bool:
    """Determine whether the input x is either an Aikit Array or a Native Array.

    Parameters
    ----------
    x
        The input to check
    exclusive
        Whether to check if the data type is exclusively an array, rather than a
        variable or traced array.

    Returns
    -------
    ret
        Boolean, whether or not x is an array.

    Examples
    --------
    >>> x = aikit.array([0, 1, 2])
    >>> print(aikit.is_array(x))
    True

    >>> x = aikit.native_array([9.1, -8.3, 2.8, 3.0])
    >>> print(aikit.is_array(x, exclusive=True))
    True

    >>> x = [2, 3]
    >>> print(aikit.is_array(x))
    False
    """
    return aikit.is_aikit_array(x, exclusive=exclusive) or aikit.is_native_array(
        x, exclusive=exclusive
    )


@handle_exceptions
def is_aikit_container(x: Any, /) -> bool:
    """Determine whether the input x is an Aikit Container.

    Parameters
    ----------
    x
        The input to check

    Returns
    -------
    ret
        Boolean, whether or not x is an aikit container.

    Examples
    --------
    >>> x = aikit.Container()
    >>> print(aikit.is_aikit_container(x))
    True

    >>> x = [2, 3]
    >>> print(aikit.is_aikit_container(x))
    False
    """
    return isinstance(x, aikit.Container)


aikit.array_mode = array_mode_stack[-1] if array_mode_stack else True


@handle_exceptions
def set_array_mode(mode: bool) -> None:
    """Set the mode of whether to convert inputs to aikit.NativeArray, then
    convert outputs back to aikit.Array.

    It Stops the conversion of aikit.NativeArray to aikit.Array in the
    case when it is set to False.

    Parameter
    ---------
    mode
        boolean whether to perform aikit.Array conversions

    Examples
    --------
    >>> aikit.set_array_mode(False)
    >>> aikit.array_mode
    False

    >>> aikit.set_array_mode(True)
    >>> aikit.array_mode
    True
    """
    global array_mode_stack
    aikit.utils.assertions.check_isinstance(mode, bool)
    array_mode_stack.append(mode)
    aikit.__setattr__("array_mode", mode, True)


@handle_exceptions
def unset_array_mode() -> None:
    """Reset the mode of converting inputs to aikit.NativeArray, then converting
    outputs back to aikit.Array to the previous state.

    Examples
    --------
    >>> aikit.set_array_mode(False)
    >>> aikit.array_mode
    False

    >>> aikit.unset_shape_array_mode()
    >>> aikit.array_mode
    True
    """
    global array_mode_stack
    if array_mode_stack:
        array_mode_stack.pop(-1)
        mode = array_mode_stack[-1] if array_mode_stack else True
        aikit.__setattr__("array_mode", mode, True)


aikit.nestable_mode = nestable_mode_stack[-1] if nestable_mode_stack else True


@handle_exceptions
def set_nestable_mode(mode: bool) -> None:
    """Set the mode of whether to check if function inputs are aikit.Container.

    Parameter
    ---------
    mode
        boolean whether to check if function inputs are aikit.Container

    Examples
    --------
    >>> aikit.set_nestable_mode(False)
    >>> aikit.nestable_mode
    False

    >>> aikit.set_nestable_mode(True)
    >>> aikit.nestable_mode
    True
    """
    global nestable_mode_stack
    aikit.utils.assertions.check_isinstance(mode, bool)
    nestable_mode_stack.append(mode)
    aikit.__setattr__("nestable_mode", mode, True)


@handle_exceptions
def unset_nestable_mode() -> None:
    """Reset the mode of whether to check if function inputs are aikit.Container
    to the previous state.

    Examples
    --------
    >>> aikit.set_nestable_mode(False)
    >>> aikit.nestable_mode
    False

    >>> aikit.unset_nestable_mode()
    >>> aikit.nestable_mode
    True
    """
    global nestable_mode_stack
    if nestable_mode_stack:
        nestable_mode_stack.pop(-1)
        mode = nestable_mode_stack[-1] if nestable_mode_stack else True
        aikit.__setattr__("nestable_mode", mode, True)


aikit.exception_trace_mode = (
    exception_trace_mode_stack[-1] if exception_trace_mode_stack else "full"
)


@handle_exceptions
def set_exception_trace_mode(mode: Literal["aikit", "full", "frontend"]) -> None:
    """Set the mode of whether to show frontend-truncated exception stack
    traces, aikit- truncated exception stack traces or full exception stack
    traces.

    Parameter
    ---------
    mode
        str exception trace mode, one of `aikit`, `full` or `frontend`

    Examples
    --------
    >>> aikit.set_exception_trace_mode("aikit")
    >>> aikit.exception_trace_mode
    'aikit'

    >>> aikit.set_exception_trace_mode("full")
    >>> aikit.exception_trace_mode
    'full'
    """
    global exception_trace_mode_stack
    trace_modes = list(trace_mode_dict.keys())
    aikit.utils.assertions.check_elem_in_list(
        mode, trace_modes, False, f"trace mode must be one of {trace_modes}"
    )
    exception_trace_mode_stack.append(mode)
    aikit.__setattr__("exception_trace_mode", mode, True)


@handle_exceptions
def unset_exception_trace_mode() -> None:
    """Reset the trace mode to the previously set mode.

    Examples
    --------
    >>> aikit.set_exception_trace_mode("aikit")
    >>> aikit.exception_trace_mode
    'aikit'

    >>> aikit.unset_exception_trace_mode()
    >>> aikit.exception_trace_mode
    'full'
    """
    global exception_trace_mode_stack
    if exception_trace_mode_stack:
        exception_trace_mode_stack.pop(-1)
        mode = exception_trace_mode_stack[-1] if exception_trace_mode_stack else "full"
        aikit.__setattr__("exception_trace_mode", mode, True)


aikit.show_func_wrapper_trace_mode = (
    show_func_wrapper_trace_mode_stack[-1]
    if show_func_wrapper_trace_mode_stack
    else True
)


@handle_exceptions
def set_show_func_wrapper_trace_mode(mode: bool) -> None:
    """Set the mode of whether to show the full stack trace with function
    wrapping traces.

    Parameter
    ---------
    mode
        boolean whether to perform aikit.Array conversions

    Examples
    --------
    >>> aikit.set_show_func_wrapper_trace_mode(False)
    >>> aikit.show_func_wrapper_trace_mode
    False

    >>> aikit.set_show_func_wrapper_trace_mode(True)
    >>> aikit.show_func_wrapper_trace_mode
    True
    """
    global show_func_wrapper_trace_mode_stack
    aikit.utils.assertions.check_isinstance(mode, bool)
    show_func_wrapper_trace_mode_stack.append(mode)
    aikit.__setattr__("show_func_wrapper_trace_mode", mode, True)


@handle_exceptions
def unset_show_func_wrapper_trace_mode() -> None:
    """Reset the mode of whether to show the full stack trace with function
    wrapping traces.

    Examples
    --------
    >>> aikit.set_show_func_wrapper_trace_mode(False)
    >>> aikit.show_func_wrapper_trace_mode
    False

    >>> aikit.unset_show_func_wrapper_trace_mode()
    >>> aikit.show_func_wrapper_trace_mode
    True
    """
    global show_func_wrapper_trace_mode_stack
    if show_func_wrapper_trace_mode_stack:
        show_func_wrapper_trace_mode_stack.pop(-1)
        mode = (
            show_func_wrapper_trace_mode_stack[-1]
            if show_func_wrapper_trace_mode_stack
            else True
        )
        aikit.__setattr__("show_func_wrapper_trace_mode", mode, True)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_native_arrays
@handle_array_function
@handle_device
def array_equal(
    x0: Union[aikit.Array, aikit.NativeArray],
    x1: Union[aikit.Array, aikit.NativeArray],
    /,
) -> bool:
    """Determine whether two input arrays are equal across all elements.

    Parameters
    ----------
    x0
        The first input array to compare.
    x1
        The second input array to compare.

    Returns
    -------
    ret
        Boolean, whether or not the input arrays are equal across all elements.

    Examples
    --------
    >>> x = aikit.array([1,0,1])
    >>> y = aikit.array([1,0,-1])
    >>> z = aikit.array_equal(x,y)
    >>> print(z)
    False

    >>> a = aikit.array([1, 2])
    >>> b = aikit.array([1, 2])
    >>> c = aikit.array_equal(a,b)
    >>> print(c)
    True

    >>> i = aikit.array([1, 2])
    >>> j = aikit.array([1, 2, 3])
    >>> k = aikit.array_equal(i,j)
    >>> print(k)
    False
    """
    return current_backend(x0).array_equal(x0, x1)


@handle_exceptions
@handle_nestable
@inputs_to_aikit_arrays
@handle_array_function
def all_equal(
    *xs: Iterable[Any], equality_matrix: bool = False
) -> Union[bool, aikit.Array, aikit.NativeArray]:
    """Determine whether the inputs are all equal.

    Parameters
    ----------
    xs
        inputs to compare.
    equality_matrix
        Whether to return a matrix of equalities comparing each input with every other.
        Default is ``False``.

    Returns
    -------
    ret
        Boolean, whether or not the inputs are equal, or matrix array of booleans if
        equality_matrix=True is set.

    Examples
    --------
    With :class:`aikit.Array` inputs:

    >>> x1 = aikit.array([1, 1, 0, 0, 1, -1])
    >>> x2 = aikit.array([1, 1, 0, 0, 1, -1])
    >>> y = aikit.all_equal(x1, x2)
    >>> print(y)
    True

    >>> x1 = aikit.array([0, 0])
    >>> x2 = aikit.array([0, 0])
    >>> x3 = aikit.array([1, 0])
    >>> y = aikit.all_equal(x1, x2, x3, equality_matrix=True)
    >>> print(y)
    aikit.array([[ True,  True, False],
       [ True,  True, False],
       [False, False,  True]])

    With one :class:`aikit.Container` inputs:

    >>> x1 = aikit.Container(a=aikit.array([0, 0, -1, 1, 0]),
    ...                    b=aikit.array([0, 0, -1, 1, 0]))
    >>> x2 = aikit.array([0, 0, -1, 1, 0])
    >>> y = aikit.all_equal(x1, x2, equality_matrix=False)
    >>> print(y)
    {
        a: True,
        b: True
    }

    With multiple :class:`aikit.Container` inputs:

    >>> x1 = aikit.Container(a=aikit.array([1, 0, 1, 1]),
    ...                    b=aikit.array([1, 0, 0, 1]))
    >>> x2 = aikit.Container(a=aikit.array([1, 0, 1, 1]),
    ...                    b=aikit.array([1, 0, -1, -1]))
    >>> y = aikit.all_equal(x1, x2, equality_matrix=False)
    >>> print(y)
    {
        a: True,
        b: False
    }
    """
    equality_fn = aikit.array_equal if aikit.is_array(xs[0]) else lambda a, b: a == b
    if equality_matrix:
        num_arrays = len(xs)
        mat = [[None for _ in range(num_arrays)] for _ in range(num_arrays)]
        for i, xa in enumerate(xs):
            for j_, xb in enumerate(xs[i:]):
                j = j_ + i
                res = equality_fn(xa, xb)
                if aikit.is_native_array(res):
                    # noinspection PyTypeChecker
                    res = aikit.to_scalar(res)
                # noinspection PyTypeChecker
                mat[i][j] = res
                # noinspection PyTypeChecker
                mat[j][i] = res
        return aikit.array(mat)
    x0 = xs[0]
    for x in xs[1:]:
        if not equality_fn(x0, x):
            return False
    return True


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_native_arrays
@handle_array_function
@handle_device
def to_numpy(
    x: Union[aikit.Array, aikit.NativeArray], /, *, copy: bool = True
) -> np.ndarray:
    """Convert an array into a numpy array.

    Parameters
    ----------
    x
        input array
    copy
        whether to copy the array to a new address or not.
        Default is ``True``.

    Returns
    -------
    ret
        a numpy array copying all the element of the array ``x``.

    Examples
    --------
    With :class:`aikit.Array` inputs:

    >>> x = aikit.array([-1, 0, 1])
    >>> y = aikit.to_numpy(x, copy=True)
    >>> print(y)
    [-1  0  1]

    >>> x = aikit.array([[-1, 0, 1],[-1, 0, 1], [1,0,-1]])
    >>> y = aikit.to_numpy(x, copy=True)
    >>> print(y)
    [[-1  0  1]
    [-1  0  1]
    [ 1  0 -1]]

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([-1, 0, 1]))
    >>> y = aikit.to_numpy(x)
    >>> print(y)
    {
        a: array([-1, 0, 1], dtype=int32)
    }

    >>> x = aikit.Container(a=aikit.array([[-1.0, 0., 1.], [-1, 0, 1], [1, 0, -1]]),
    ...                   b=aikit.array([[-1, 0, 0], [1, 0, 1], [1, 1, 1]]))
    >>> y = aikit.to_numpy(x)
    >>> print(y)
    {
        a: array([[-1., 0., 1.],
                  [-1., 0., 1.],
                  [1., 0., -1.]], dtype=float32),
        b: array([[-1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 1]], dtype=int32)
    }
    """
    return current_backend(x).to_numpy(x, copy=copy)


@handle_exceptions
@handle_nestable
def isscalar(x: Any, /) -> bool:
    return np.isscalar(x)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_native_arrays
@handle_array_function
@handle_device
def to_scalar(x: Union[aikit.Array, aikit.NativeArray], /) -> Number:
    """Convert an array with a single element into a scalar.

    Parameters
    ----------
    x
        Input array with a single element.

    Returns
    -------
    ret
        a scalar copying the element of the array ``x``.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([3])
    >>> y = aikit.to_scalar(x)
    >>> print(y)
    3

    With a mix of :class:`aikit.Container` and :class:`aikit.Array` input:

    >>> x = aikit.Container(a=aikit.array([-1]), b=aikit.array([3]))
    >>> y = aikit.to_scalar(x)
    >>> print(y)
    {
        a: -1,
        b: 3
    }

    >>> x = aikit.Container(a=aikit.array([1]), b=aikit.array([0]),
    ...                   c=aikit.array([-1]))
    >>> y = aikit.to_scalar(x)
    >>> print(y)
    {
        a: 1,
        b: 0,
        c: -1
    }
    """
    return current_backend(x).to_scalar(x)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_native_arrays
@handle_array_function
@handle_device
def to_list(x: Union[aikit.Array, aikit.NativeArray], /) -> List:
    """Create a (possibly nested) list from input array.

    Parameters
    ----------
    x
        Input array.

    Returns
    -------
    ret
        A list representation of the input array ``x``.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([-1, 0, 1])
    >>> y = aikit.to_list(x)
    >>> print(y)
    [-1, 0, 1]

    >>> x = aikit.array([[ 1.1,  2.2,  3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> y = aikit.to_list(x)
    >>> print(y)
    [[1.100000023841858,2.200000047683716,3.299999952316284],
    [-4.400000095367432,-5.5,-6.599999904632568]]

    >>> x = aikit.array([[[-1,  0,  1],
    ...                 [ 1,  0, -1]],
    ...                [[ 1, -1,  0],
    ...                 [ 1,  0, -1]]])
    >>> y = aikit.to_list(x)
    >>> print(y)
    [[[-1, 0, 1], [1, 0, -1]], [[1, -1, 0], [1, 0, -1]]]

    With a mix of :class:`aikit.Container` and :class:`aikit.Array` input:

    >>> x = aikit.Container(a=aikit.array([-1, 0, 1]))
    >>> y = aikit.to_list(x)
    >>> print(y)
    {
        a: [-1, 0, 1]
    }

    >>> x = aikit.Container(a=aikit.array([[-1, 0, 1],
    ...                                [-1, 0, 1],
    ...                                [1, 0, -1]]))
    >>> y = aikit.to_list(x)
    >>> print(y)
    {
        a: [[-1, 0, 1], [-1, 0, 1], [1,0,-1]]
    }

    >>> x = aikit.Container(a=aikit.array([[[-1, 0, 1],[1, 0, -1]],
    ...                                [[1, -1, 0],[1, 0, -1]]]))
    >>> y = aikit.to_list(x)
    >>> print(y)
    {
        a: [[[-1, 0, 1], [1, 0, -1]], [[1, -1, 0], [1, 0, -1]]]
    }
    """
    return current_backend(x).to_list(x)


@handle_exceptions
@handle_nestable
@inputs_to_aikit_arrays
@handle_array_function
def clip_vector_norm(
    x: Union[aikit.Array, aikit.NativeArray],
    max_norm: float,
    /,
    *,
    p: float = 2.0,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Clips (limits) the vector p-norm of an array.

    Parameters
    ----------
    x
        Input array containing elements to clip.
    max_norm
        The maximum value of the array norm.
    p
        The p-value for computing the p-norm.
        Default is 2.
    out
        optional output array, for writing the result to.
        It must have a shape that the inputs broadcast to.

    Returns
    -------
    ret
        An array with the vector norm downscaled to the max norm if needed.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([0., 1., 2.])
    >>> y = aikit.clip_vector_norm(x, 2.0)
    >>> print(y)
    aikit.array([0.   , 0.894, 1.79 ])

    >>> x = aikit.array([0.5, -0.7, 2.4])
    >>> y = aikit.clip_vector_norm(x, 3.0, p=1.0)
    >>> print(y)
    aikit.array([ 0.417, -0.583,  2.   ])

    >>> x = aikit.array([[[0., 0.], [1., 3.], [2., 6.]],
    ...                [[3., 9.], [4., 12.], [5., 15.]]])
    >>> y = aikit.zeros(((2, 3, 2)))
    >>> aikit.clip_vector_norm(x, 4.0, p=1.0, out=y)
    >>> print(y)
    aikit.array([[[0.    , 0.    ],
                [0.0667, 0.2   ],
                [0.133 , 0.4   ]],
               [[0.2   , 0.6   ],
                [0.267 , 0.8   ],
                [0.333 , 1.    ]]])

    >>> x = aikit.array([[1.1, 2.2, 3.3],
    ...                [-4.4, -5.5, -6.6]])
    >>> aikit.clip_vector_norm(x, 1.0, p=3.0, out=x)
    >>> print(x)
    aikit.array([[ 0.131,  0.263,  0.394],
               [-0.526, -0.657, -0.788]])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),
    ...                   b=aikit.array([3., 4., 5.]))
    >>> y = aikit.clip_vector_norm(x, 2.0)
    >>> print(y)
    {
        a: aikit.array([0., 0.894, 1.79]),
        b: aikit.array([0.849, 1.13, 1.41])
    }

    With multiple :class:`aikit.Container` inputs:

    >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),
    ...                   b=aikit.array([3., 4., 5.]))
    >>> max_norm = aikit.Container(a=2, b=3)
    >>> y = aikit.clip_vector_norm(x, max_norm)
    >>> print(y)
    {
        a: aikit.array([0., 0.894, 1.79]),
        b: aikit.array([2.449, 2.65, 2.83])
    }
    """
    norm = aikit.vector_norm(x, keepdims=True, ord=p)
    ratio = aikit.stable_divide(max_norm, norm)
    if ratio < 1:
        ret = ratio * x
    else:
        ret = aikit.copy_array(x)
    if out is not None:
        ret = aikit.inplace_update(out, ret)
    return ret


@handle_exceptions
@handle_nestable
@inputs_to_aikit_arrays
@handle_array_function
def clip_matrix_norm(
    x: Union[aikit.Array, aikit.NativeArray],
    max_norm: float,
    /,
    *,
    p: float = 2.0,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Clips (limits) the matrix norm of an array.

    Parameters
    ----------
    x
        Input array containing elements to clip.
    max_norm
        The maximum value of the array norm.
    p
        The p-value for computing the p-norm.
        Default is 2.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        An array with the matrix norm downscaled to the max norm if needed.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([[0., 1., 2.]])
    >>> y = aikit.clip_matrix_norm(x, 2.0)
    >>> print(y)
    aikit.array([[0.   , 0.894, 1.79 ]])

    >>> x = aikit.array([[0.1, -1.2, 3.7], [0., 7.3, -0.5]])
    >>> y = aikit.clip_matrix_norm(x, 3.0, p=1.0)
    >>> print(y)
    aikit.array([[ 0.0353, -0.424 ,  1.31  ],
               [ 0.    ,  2.58  , -0.176 ]])

    >>> x = aikit.array([[[5., 4.], [-2., 6.]],
    ...                [[3., 7.], [0., -5.]]])
    >>> y = aikit.empty((2, 2, 2))
    >>> y = aikit.clip_matrix_norm(x, 0.5, p=2.0)
    >>> print(y)
    aikit.array([[[ 0.339,  0.271],
                [-0.135,  0.406]],
               [[ 0.168,  0.391],
                [ 0.   , -0.279]]])

    >>> x = aikit.array([[0., 1.],
    ...                [2., 3.]])
    >>> aikit.clip_matrix_norm(x, 5.0, p=1.0, out=x)
    >>> print(x)
    aikit.array([[0., 1.],
               [2., 3.]])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([[0., 1., 2.]]),
    ...                   b=aikit.array([[3., 4., 5.]]))
    >>> y = aikit.clip_matrix_norm(x, 2.0)
    >>> print(y)
    {
        a: aikit.array([[0., 0.894, 1.79]]),
        b: aikit.array([[0.849, 1.13, 1.41]])
    }
    """
    norms = aikit.matrix_norm(x, ord=p, keepdims=True)
    ratios = aikit.minimum(aikit.stable_divide(max_norm, norms), 1.0)
    return aikit.multiply(ratios, x, out=out)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_aikit_arrays
@handle_array_function
def fourier_encode(
    x: Union[aikit.Array, aikit.NativeArray],
    max_freq: Union[float, aikit.Array, aikit.NativeArray],
    /,
    *,
    num_bands: int = 4,
    linear: bool = False,
    concat: bool = True,
    flatten: bool = False,
) -> Union[aikit.Array, aikit.NativeArray, Tuple]:
    """Pad an array with fourier encodings.

    Parameters
    ----------
    x
        Input array to encode.
    max_freq
        The maximum frequency of the encoding.
    num_bands
        The number of frequency bands for the encoding.
        Default is 4.
    linear
        Whether to space the frequency bands linearly as opposed to geometrically.
        Default is ``False``.
    concat
        Whether to concatenate the position, sin and cos values, or return separately.
        Default is ``True``.
    flatten
        Whether to flatten the position dimension into the batch dimension.
        Default is False.

    Returns
    -------
    ret
        New array with the final dimension expanded, and the encodings stored in this
        channel.

    Examples
    --------
    >>> x = aikit.array([1,2,3])
    >>> y = 1.5
    >>> z = aikit.fourier_encode(x,y)
    >>> print(z)
    aikit.array([[ 1.0000000e+00, 1.2246468e-16, 0.0000000e+00, 0.0000000e+00,
                 0.0000000e+00, -1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
                 1.0000000e+00],
               [ 2.0000000e+00, -2.4492936e-16, 0.0000000e+00, 0.0000000e+00,
                 0.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
                 1.0000000e+00],
               [ 3.0000000e+00, 3.6739404e-16, 0.0000000e+00, 0.0000000e+00,
                 0.0000000e+00, -1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
                 1.0000000e+00]])


    >>> x = aikit.array([3,10])
    >>> y = 2.5
    >>> z = aikit.fourier_encode(x, y, num_bands=3)
    >>> print(z)
    aikit.array([[ 3.0000000e+00,  3.6739404e-16,  3.6739404e-16, 3.6739404e-16,
                -1.0000000e+00, -1.0000000e+00, -1.0000000e+00],
               [ 1.0000000e+01, -1.2246468e-15, -1.2246468e-15, -1.2246468e-15,
                 1.0000000e+00,  1.0000000e+00,  1.0000000e+00]])
    """
    x_in = x
    dim = x.shape[-1]
    x = aikit.expand_dims(x, axis=-1)
    orig_x = x
    if linear:
        scales = aikit.linspace(1.0, max_freq / 2, num_bands, device=dev(x))
    elif aikit.backend == "torch" and isinstance(max_freq, float):
        scales = aikit.logspace(
            0.0,
            aikit.log(aikit.array(max_freq / 2)) / math.log(10),
            num_bands,
            base=10,
            device=dev(x),
        )
    else:
        scales = aikit.logspace(
            0.0,
            aikit.log(max_freq / 2) / math.log(10),
            num_bands,
            base=10,
            device=dev(x),
        )
    scales = aikit.astype(scales, aikit.dtype(x))
    scales = scales[(*((None,) * (len(x.shape) - len(scales.shape))), Ellipsis)]
    x = x * scales * math.pi
    sin_x = aikit.sin(x)
    cos_x = aikit.cos(x)
    if flatten:
        orig_x = x_in
        sin_x = aikit.reshape(sin_x, [-1, num_bands * dim])
        cos_x = aikit.reshape(cos_x, [-1, num_bands * dim])
    if concat:
        return aikit.concat([orig_x, sin_x, cos_x], axis=-1)
    return sin_x, cos_x


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_aikit_arrays
@handle_array_function
def value_is_nan(
    x: Union[aikit.Array, aikit.NativeArray, Number],
    /,
    *,
    include_infs: bool = True,
) -> bool:
    """Determine whether the single valued array or scalar is of nan type.

    Parameters
    ----------
    x
        The input to check Input array.
    include_infs
        Whether to include infs and -infs in the check.
        Default is ``True``.

    Returns
    -------
    ret
        Boolean as to whether the input value is a nan or not.

    Examples
    --------
    >>> x = aikit.array([451])
    >>> y = aikit.value_is_nan(x)
    >>> print(y)
    False

    >>> x = aikit.array([float('inf')])
    >>> y = aikit.value_is_nan(x)
    >>> print(y)
    True

    >>> x = aikit.array([float('inf')])
    >>> y = aikit.value_is_nan(x, include_infs=False)
    >>> print(y)
    False

    >>> x = aikit.array([float('nan')])
    >>> y = aikit.value_is_nan(x, include_infs=False)
    >>> print(y)
    True

    >>> x = aikit.array([0])
    >>> y = aikit.value_is_nan(x)
    >>> print(y)
    False
    """
    x_scalar = aikit.to_scalar(x) if aikit.is_array(x) else x
    if x_scalar != x:
        return True
    if include_infs and (x_scalar in [INF, -INF]):
        return True
    return False


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_aikit_arrays
@handle_array_function
def has_nans(
    x: Union[aikit.Array, aikit.NativeArray], /, *, include_infs: bool = True
) -> bool:
    """Determine whether the array contains any nans, as well as infs or -infs
    if specified.

    Parameters
    ----------
    x
        Input array.
    include_infs
        Whether to include ``+infinity`` and ``-infinity`` in the check.
        Default is ``True``.

    Returns
    -------
    ret
        Boolean as to whether the array contains nans.


    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([1, 2, 3])
    >>> y = aikit.has_nans(x)
    >>> print(y)
    False

    >>> x = aikit.array([float('nan'), 2, 3])
    >>> y = aikit.has_nans(x)
    >>> print(y)
    True

    >>> x = aikit.array([float('inf'), 2, 3])
    >>> y = aikit.has_nans(x)
    >>> print(y)
    True

    >>> x = aikit.array([float('inf'), 2, 3])
    >>> y = aikit.has_nans(x, include_infs=False)
    >>> print(y)
    False

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([0., 1., 2.]), b=aikit.array([3., 4., 5.]))
    >>> y = aikit.has_nans(x)
    >>> print(y)
    {
        a: False,
        b: False
    }
    """
    return aikit.value_is_nan(aikit.sum(x), include_infs=include_infs)


@handle_exceptions
def exists(x: Any, /) -> bool:
    """Check as to whether the input is None or not.

    Parameters
    ----------
    x
        Input to check.

    Returns
    -------
    ret
        True if x is not None, else False.

    Examples
    --------
    With :code:`Any` input:

    >>> x = None
    >>> y = aikit.exists(x)
    >>> print(y)
    False

    >>> x = ""
    >>> y = aikit.exists(x)
    >>> print(y)
    True

    >>> x = []
    >>> y = aikit.exists(x)
    >>> print(y)
    True

    >>> x = 1
    >>> y = aikit.exists(x)
    >>> print(y)
    True

    >>> x = "abc"
    >>> y = aikit.exists(x)
    >>> print(y)
    True

    >>> x = [1, 0, -1, 1]
    >>> y = aikit.exists(x)
    >>> print(y)
    True

    >>> x = aikit.array([1, 2, 3, 1.2])
    >>> y = aikit.exists(x)
    >>> print(y)
    True

    With a mix of :class:`aikit.Container` and :code:`Any` input:

    >>> x = aikit.Container(a=None, b=None)
    >>> y = aikit.exists(x)
    >>> print(y)
    True

    >>> x = aikit.Container(a=None, b="")
    >>> y = aikit.exists(x)
    >>> print(y)
    True

    >>> x = aikit.Container(a=123, b="")
    >>> y = aikit.exists(x)
    >>> print(y)
    True
    """
    return x is not None


@handle_exceptions
def default(
    x: Any,
    /,
    default_val: Any,
    *,
    catch_exceptions: bool = False,
    rev: bool = False,
    with_callable: bool = False,
) -> Any:
    """Return x provided it exists (is not None), else returns default value.

    Parameters
    ----------
    x
        Input which may or may not exist (be None).
    default_val
        The default value.
    catch_exceptions
        Whether to catch exceptions from callable x.
        Default is ``False``.
    rev
        Whether to reverse the input x and default_val.
        Default is ``False``.
    with_callable
        Whether either of the arguments might be callable functions.
        Default is ``False``.

    Returns
    -------
    ret
        x if x exists (is not None), else default.

    Examples
    --------
    With :code:`Any` input:

    >>> x = None
    >>> y = aikit.default(x, "default_string")
    >>> print(y)
    default_string

    >>> x = ""
    >>> y = aikit.default(x, "default_string")
    >>> print(y)


    >>> x = aikit.array([4, 5, 6])
    >>> y = aikit.default(x, aikit.array([1, 2, 3]), rev=True)
    >>> print(y)
    aikit.array([1, 2, 3])

    >>> x = lambda: aikit.array([1, 2, 3])
    >>> y = aikit.default(x, aikit.array([4, 5, 6]), with_callable=True)
    >>> print(y)
    aikit.array([1, 2, 3])

    >>> x = lambda: None
    >>> y = aikit.default(x, lambda: aikit.array([1, 2, 3]), with_callable=True)
    >>> print(y)
    aikit.array([1, 2, 3])

    >>> x = lambda: None
    >>> y = aikit.default(x, lambda: aikit.array([1, 2, 3]), catch_exceptions=True)
    >>> print(y)
    aikit.array([1, 2, 3])

    >>> x = lambda a, b: a + b
    >>> y = aikit.default(x, lambda: aikit.array([1, 2, 3]), with_callable=True,
    ...                 catch_exceptions=True)
    >>> print(y)
    aikit.array([1, 2, 3])

    >>> x = lambda a, b: a + b
    >>> y = aikit.default(x, lambda: aikit.array([1, 2, 3]), with_callable=True,
    ...                 catch_exceptions=True, rev=True)
    >>> print(y)
    aikit.array([1, 2, 3])
    """
    with_callable = catch_exceptions or with_callable
    if rev:
        x, default_val = default_val, x
    if with_callable:
        x_callable = callable(x)
        default_callable = callable(default_val)
    else:
        x_callable = False
        default_callable = False
    if catch_exceptions:
        # noinspection PyBroadException
        try:
            x = x() if x_callable else x
        except Exception:
            return default_val() if default_callable else default_val
    else:
        x = x() if x_callable else x
    return x if exists(x) else default_val() if default_callable else default_val


@handle_exceptions
def to_aikit_shape(shape: Union[aikit.Shape, aikit.NativeShape]) -> aikit.Shape:
    """Return the input shape in aikit.Shape form.

    Parameters
    ----------
    shape
        The input to be converted

    Returns
    -------
     ret
        the input in aikit.Shape form
    """
    if isinstance(shape, aikit.Shape):
        return shape
    return aikit.Shape(shape)


@handle_exceptions
def to_native_shape(
    shape: Union[aikit.Array, aikit.Shape, aikit.NativeShape, tuple, int, list]
) -> aikit.NativeShape:
    """Return the input shape in its native backend framework form.

    Parameters
    ----------
    shape
        The input to be converted

    Returns
    -------
     ret
        the input in its native framework form
    """
    native_shape_type = (aikit.NativeShape,)
    if aikit.current_backend_str() == "torch":
        native_shape_type += (tuple,)
    if len(backend_stack) != 0 and isinstance(shape, native_shape_type):
        return shape
    aikit.utils.assertions.check_isinstance(
        shape, (int, list, tuple, aikit.Array, aikit.NativeArray, aikit.Shape)
    )
    if isinstance(shape, int):
        shape = (shape,)
    elif isinstance(shape, list):
        shape = tuple(shape)
    elif is_array(shape):
        shape = aikit.to_numpy(shape).tolist()
    elif isinstance(shape, aikit.Shape):
        shape = shape.shape
    aikit.utils.assertions.check_all(
        [isinstance(v, int) for v in shape if not is_array(v)],
        "shape must take integers only",
        as_array=False,
    )
    aikit.utils.assertions.check_true(
        not is_array(shape) or aikit.is_int_dtype(shape), "shape must take integers only"
    )
    return aikit.NativeShape(shape) if len(backend_stack) != 0 else aikit.Shape(shape)


@handle_exceptions
@handle_nestable
def try_else_none(fn: Callable, *args: Any, **kwargs: Any) -> Union[Callable, None]:
    """Try and return the function, otherwise return None if an exception was
    raised during function execution.

    Parameters
    ----------
    fn
        Function to try and call and return.
    args
        list of arguments.
    kwargs
        dictionary of keyword arguments

    Returns
    -------
        Either the function itself or None if an exception was raised
        during function execution.

    Examples
    --------
    with a function that is executed without any exception:

    >>> x = aikit.array([1, 2, 3])
    >>> y = aikit.array([4, 5, 6])
    >>> z = aikit.try_else_none(aikit.add, x, y)
    >>> print(z.__name__)
    add

    with a function that is executed with an exception:

    >>> x = aikit.array([1, 2, 3])
    >>> y = 'hemant'
    >>> z = aikit.try_else_none(aikit.add,x, y)
    >>> print(z)
    None
    """
    try:
        _ = fn(*args, **kwargs)
        return fn
    except Exception:
        return None


@handle_exceptions
def arg_names(receiver):
    """Get the expected keyword arguments for a function or class constructor.

    Parameters
    ----------
    receiver
        Function or class constructor

    Returns
    -------
    ret
        List containing the keyword arguments' names for a function or class constructor

    Examples
    --------
    >>> x = aikit.arg_names(aikit.tan)
    >>> print(x)
    ['x', 'out']

    >>> x = aikit.arg_names(aikit.optimizers.Adam)
    >>> print(x)
    ['lr', 'beta1', 'beta2', 'epsilon', 'inplace',
    'stop_gradients', 'trace_on_next_step', 'device']
    """
    return list(inspect.signature(receiver).parameters.keys())


@handle_exceptions
def match_kwargs(
    kwargs: Dict, *receivers: Iterable[Callable], allow_duplicates: bool = False
) -> Union[List[Dict], Dict]:
    """Match keyword arguments to either class or function receivers.

    Parameters
    ----------
    kwargs
        Keyword arguments to match.
    receivers
        Functions and/or classes to match the keyword arguments to.
    allow_duplicates
        Whether to allow one keyword argument to be used for multiple receivers.
        Default is ``False``.

    Returns
    -------
    ret
        Sequence of keyword arguments split as best as possible.

    Examples
    --------
    >>> o = aikit.zeros(3)
    >>> kwargs = {'out': o, 'bias': aikit.arange(3)}
    >>> x = aikit.match_kwargs(kwargs, aikit.add, aikit.linear)
    >>> print(x)
    [{'out': aikit.array([0., 0., 0.])}, {'bias': aikit.array([0, 1, 2])}]

    >>> o = aikit.zeros(3)
    >>> kwargs = {'out': o, 'bias': aikit.arange(3)}
    >>> x = aikit.match_kwargs(kwargs, aikit.linear, aikit.add)
    >>> print(x)
    [{'out': aikit.array([0., 0., 0.]), 'bias': aikit.array([0, 1, 2])}, {}]
    """
    split_kwargs = []
    for receiver in receivers:
        expected_kwargs = arg_names(receiver)
        found_kwargs = {k: v for k, v in kwargs.items() if k in expected_kwargs}
        if not allow_duplicates:
            for k in found_kwargs:
                del kwargs[k]
        split_kwargs.append(found_kwargs)
    if len(split_kwargs) == 1:
        return split_kwargs[0]
    return split_kwargs


@handle_exceptions
def cache_fn(func: Callable) -> Callable:
    """Cache function outputs.

    A decorator to wrap a function, such that computed outputs are cached to avoid
    recalculating them later.

    Parameters
    ----------
    func
        The function to wrap, whose output should be cached for later.

    Returns
    -------
    ret
        The newly cache wrapped function.

    Examples
    --------
    With positional arguments only:

    >>> def my_sum(val1:float, val2:float)->float: return val1 + val2
    >>> cached_sum = aikit.cache_fn(my_sum)
    >>> print(cached_sum(3, 5))
    8

    With keyword arguments:

    >>> def line_eq(x:float, /, *, slp:float=2, itc:float=0)->float: return x*slp+itc
    >>> cached_line_eq = aikit.cache_fn(line_eq)
    >>> print(cached_line_eq(3, itc=5, slp=2))
    11
    """
    global FN_CACHE
    if func not in FN_CACHE:
        FN_CACHE[func] = {}

    @wraps(func)
    def cached_fn(*args, **kwargs):
        key = "".join(
            ([f"{str(i)}, " for i in args] + [" kw, "])
            + [f"{str(i)}, " for i in sorted(kwargs.items())]
        )
        cache = FN_CACHE[func]
        if key in cache:
            return cache[key]
        ret = func(*args, **kwargs)
        cache[key] = ret
        return ret

    return cached_fn


@handle_exceptions
def current_backend_str() -> Union[str, None]:
    """Return framework string.

    Returns
    -------
    ret
        The framework string.
    """
    fw = current_backend()
    if not backend_stack:
        return ""
    return fw.current_backend_str()


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_aikit_arrays
@handle_array_function
def einops_rearrange(
    x: Union[aikit.Array, aikit.NativeArray],
    pattern: str,
    /,
    *,
    out: Optional[aikit.Array] = None,
    **axes_lengths: Dict[str, int],
) -> aikit.Array:
    """Perform einops rearrange operation on input array x.

    Parameters
    ----------
    x
        Input array to be re-arranged.
    pattern
        Rearrangement pattern.
    axes_lengths
        Any additional specifications for dimensions.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        New array with einops.rearrange having been applied.

    Examples
    --------
    With :class:`aikit.Array` instance method:

    >>> x = aikit.array([[1, 2, 3],
    ...               [-4, -5, -6]])
    >>> y = x.einops_rearrange("height width -> width height")
    >>> print(y)
    aikit.array([[ 1, -4],
           [ 2, -5],
           [ 3, -6]])

    >>> x = aikit.array([[[ 1,  2,  3],
    ...                  [ 4,  5,  6]],
    ...               [[ 7,  8,  9],
    ...                  [10, 11, 12]]])
    >>> y = x.einops_rearrange("c h w -> c (h w)")
    >>> print(y)
    aikit.array([[ 1,  2,  3,  4,  5,  6],
           [ 7,  8,  9, 10, 11, 12]])

    >>> x = aikit.array([[1, 2, 3, 4, 5, 6],
    ...            [7, 8, 9, 10, 11, 12]])
    >>> y = aikit.zeros((4,3))
    >>> x.einops_rearrange("c (h w) -> (c h) w", out=y, h=2, w=3)
    >>> print(y)
    aikit.array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [10, 11, 12]])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([[-4.47, 0.93, -3.34],
    ...                            [3.66, 24.29, 3.64]]),
    ...               b=aikit.array([[4.96, 1.52, -10.67],
    ...                            [4.36, 13.96, 0.3]]))
    >>> y = aikit.einops_rearrange(x, 'a b -> b a')
    >>> print(y)
    {
        a: aikit.array([[-4.46999979, 3.66000009],
                      [0.93000001, 24.29000092],
                      [-3.33999991, 3.6400001]]),
        b: aikit.array([[4.96000004, 4.36000013],
                      [1.51999998, 13.96000004],
                      [-10.67000008, 0.30000001]])
    }

    With varying pattern:

    Suppose we have a set of 32 images in "h w c" format (height-width-channel)
    and concatenate images along height (vertical axis), 960 = 32 * 30

    >>> images = aikit.asarray([aikit.random_normal(shape=(30, 40, 3)) for _ in range(32)])
    >>> x = aikit.einops_rearrange(images, 'b h w c -> (b h) w c')
    >>> print(x.shape)
    (960, 40, 3)

    # Concatenate images along horizontal axis, 1280 = 32 * 40

    >>> images = aikit.asarray([aikit.random_normal(shape=(30, 40, 3)) for _ in range(32)])
    >>> x = aikit.einops_rearrange(images, 'b h w c -> h (b w) c')
    >>> print(x.shape)
    (30, 1280, 3)

    # Reorder axes to "b c h w" format for deep learning

    >>> images = aikit.asarray([aikit.random_normal(shape=(30, 40, 3)) for _ in range(32)])
    >>> x = aikit.einops_rearrange(images, 'b h w c -> b c h w')
    >>> print(x.shape)
    (32, 3, 30, 40)

    # Flatten each image into a vector, 3600 = 30 * 40 * 3

    >>> images = aikit.asarray([aikit.random_normal(shape=(30, 40, 3)) for _ in range(32)])
    >>> x = aikit.einops_rearrange(images, 'b h w c -> b (c h w)')
    >>> print(x.shape)
    (32, 3600)

    # Split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right),
    # 128 = 32 * 2 * 2

    >>> images = aikit.asarray([aikit.random_normal(shape=(30, 40, 3)) for _ in range(32)])
    >>> x = aikit.einops_rearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c',
    ... h1=2, w1=2)
    >>> print(x.shape)
    (128, 15, 20, 3)

    # Space-to-depth operation
    >>> images = aikit.asarray([aikit.random_normal(shape=(30, 40, 3)) for _ in range(32)])
    >>> x = aikit.einops_rearrange(images, 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2,
    ... w1=2)
    >>> print(x.shape)
    (32, 15, 20, 12)
    """
    ret = einops.rearrange(x._data, pattern, **axes_lengths)
    ret = aikit.array(ret, dtype=x.dtype)
    if aikit.exists(out):
        return aikit.inplace_update(out, ret)
    return ret


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_native_arrays
@handle_array_function
def einops_reduce(
    x: Union[aikit.Array, aikit.NativeArray],
    pattern: str,
    reduction: Union[str, Callable],
    /,
    *,
    out: Optional[aikit.Array] = None,
    **axes_lengths: Dict[str, int],
) -> aikit.Array:
    """Perform einops reduce operation on input array x.

    Parameters
    ----------
    x
        Input array to be reduced.
    pattern
        Reduction pattern.
    reduction
        One of available reductions ('min', 'max', 'sum', 'mean', 'prod'), or callable.
    axes_lengths
        Any additional specifications for dimensions.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        New array with einops.reduce having been applied.

    This function is *nestable*, and therefore also accepts :code:'aikit.Container'
    instance in place of the argument.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([[-4.47, 0.93, -3.34],
    ...                [3.66, 24.29, 3.64]])
    >>> reduced = aikit.einops_reduce(x, 'a b -> b', 'mean')
    >>> print(reduced)
    aikit.array([-0.40499985, 12.61000061, 0.1500001 ])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([[-4.47, 0.93, -3.34],
    ...                                [3.66, 24.29, 3.64]]),
    ...                   b=aikit.array([[4.96, 1.52, -10.67],
    ...                                [4.36, 13.96, 0.3]]))
    >>> reduced = aikit.einops_reduce(x, 'a b -> a', 'mean')
    >>> print(reduced)
    {
        a: aikit.array([-2.29333329, 10.53000069]),
        b: aikit.array([-1.39666676, 6.20666695])
    }
    """
    ret = einops.reduce(x, pattern, reduction, **axes_lengths)
    ret = aikit.array(ret, dtype=x.dtype)
    if aikit.exists(out):
        return aikit.inplace_update(out, ret)
    return ret


# IMPORTANT: assign attribute directly to function instead of wrapper here
einops_reduce.unsupported_dtypes = {
    "torch": ("float16",),
    "tensorflow": ("complex",),
    "paddle": ("complex", "uint8", "int8", "int16", "float16"),
}


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_aikit_arrays
@handle_array_function
def einops_repeat(
    x: Union[aikit.Array, aikit.NativeArray],
    pattern: str,
    /,
    *,
    out: Optional[aikit.Array] = None,
    **axes_lengths: Dict[str, int],
) -> aikit.Array:
    """Perform einops repeat operation on input array x.

    Parameters
    ----------
    x
        Input array to be repeated.
    pattern
        Rearrangement pattern.
    axes_lengths
        Any additional specifications for dimensions.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        New array with einops.repeat having been applied.

    This function is *nestable*, and therefore also accepts :code:'aikit.Container'
    instance in place of the argument.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([1, 2, 3, 4])
    >>> repeated = aikit.einops_repeat(x, 'a -> b a', b=2)
    >>> print(repeated)
    aikit.array([[1, 2, 3, 4],
               [1, 2, 3, 4]])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([[4,5],
    ...                                [1, 3]]),
    ...                    b=aikit.array([[9, 10],
    ...                                 [4, 2]]))
    >>> repeated = aikit.einops_repeat(x, 'h w -> h (c w)', c=2)
    >>> print(repeated)
    {
        a: aikit.array([[4, 5, 4, 5],
                      [1, 3, 1, 3]]),
        b: aikit.array([[9, 10, 9, 10],
                      [4, 2, 4, 2]])
    }
    """
    ret = einops.repeat(x._data, pattern, **axes_lengths)
    ret = aikit.array(ret, dtype=x.dtype)
    if aikit.exists(out):
        return aikit.inplace_update(out, ret)
    return ret


aikit.min_denominator = min_denominator_stack[-1] if min_denominator_stack else 1e-12


@handle_exceptions
@handle_array_function
def set_min_denominator(val: float) -> None:
    """Set the global minimum denominator used by aikit for numerically stable
    division.

    Parameters
    ----------
    val
        The value to set the global minimum denominator to.

    Examples
    --------
    >>> x = aikit.min_denominator
    >>> print(x)
    1e-12

    >>> aikit.set_min_denominator(1e-13)
    >>> y = aikit.min_denominator
    >>> print(y)
    1e-13
    """
    global min_denominator_stack
    aikit.utils.assertions.check_isinstance(val, (int, float))
    min_denominator_stack.append(val)
    aikit.__setattr__("min_denominator", val, True)


@handle_exceptions
def unset_min_denominator() -> None:
    """Reset the global minimum denominator used by aikit for numerically stable
    division to the previous value.

    Examples
    --------
    >>> aikit.set_min_denominator(1e-10)
    >>> y = aikit.min_denominator
    >>> print(y)
    1e-10

    >>> aikit.unset_min_denominator()
    >>> aikit.min_denominator
    1e-12
    """
    global min_denominator_stack
    if min_denominator_stack:
        min_denominator_stack.pop(-1)
        val = min_denominator_stack[-1] if min_denominator_stack else 1e-12
        aikit.__setattr__("min_denominator", val, True)


aikit.min_base = min_base_stack[-1] if min_base_stack else 1e-05


@handle_exceptions
@handle_array_function
def set_min_base(val: float) -> None:
    """Set the global minimum base used by aikit for numerically stable power
    raising.

    Parameters
    ----------
    val
        The new value to set the minimum base to.

    Examples
    --------
    Retrieve the minimum base
    >>> x = aikit.min_base
    >>> print(x)
    1e-05

    Set the minimum base to 1e-04:
    >>> aikit.set_min_base(1e-04)

    Retrieve the minimum base:
    >>> y = aikit.min_base
    >>> print(y)
    1e-04
    """
    global min_base_stack

    # Ensure val is an instance of 'float' or 'int'
    aikit.utils.assertions.check_isinstance(val, (int, float))

    # Access and modify min_base_stack
    min_base_stack.append(val)

    # Set the min_base attribute
    aikit.__setattr__("min_base", val, True)


@handle_exceptions
def unset_min_base() -> None:
    """Reset the global minimum base used by aikit for numerically stable power
    raising to the previous value.

    Examples
    --------
    >>> aikit.set_min_base(1e-07)
    >>> y = aikit.min_base
    >>> print(y)
    1e-07

    >>> aikit.unset_min_base()
    >>> aikit.min_base
    1e-05
    """
    global min_base_stack
    if min_base_stack:
        min_base_stack.pop(-1)
        val = min_base_stack[-1] if min_base_stack else 1e-05
        aikit.__setattr__("min_base", val, True)


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_aikit_arrays
@handle_array_function
def stable_divide(
    numerator: Union[Number, aikit.Array, aikit.NativeArray],
    denominator: Union[Number, aikit.Array, aikit.NativeArray],
    /,
    *,
    min_denominator: Union[Number, aikit.Array, aikit.NativeArray] = None,
) -> Union[Number, aikit.Array]:
    """Divide the numerator by the denominator, with min denominator added to
    the denominator for numerical stability.

    Parameters
    ----------
    numerator
        The numerator of the division.
    denominator
        The denominator of the division.
    min_denominator
        The minimum denominator to use, use global aikit._MIN_DENOMINATOR (1e-12)
        by default.

    Returns
    -------
    ret
        The new item following the numerically stable division.

    Examples
    --------
    With :code:`int` input:

    >>> x = aikit.stable_divide(1, 2)
    >>> print(x)
    0.49999999999975

    >>> x = aikit.stable_divide(1, 4, min_denominator=1)
    >>> print(x)
    0.2

    With float input:

    >>> x = aikit.stable_divide(5.0, 3.33)
    >>> print(x)
    1.5015015015010504

    With :code:`complex` input:

    >>> x = aikit.stable_divide(1+1j, 1-1j)
    >>> print(x)
    (5.000444502911705e-13+0.9999999999995j)

    With :class:`aikit.Array` input:

    >>> x = aikit.asarray([[10., 20., 30.],
    ...                  [40., 50., 60.]])
    >>> y = aikit.stable_divide(x, 10.)
    >>> print(y)
    aikit.array([[1., 2., 3.],
              [4., 5., 6.]])


    >>> x = aikit.asarray([1,2,3])
    >>> y = np.array((1., 3., 5.))
    >>> z = aikit.stable_divide(x, y)
    >>> print(z)
    aikit.array([1.   , 0.667, 0.6  ])

    >>> x = aikit.asarray([1., 2., 4.])
    >>> y = aikit.asarray([1., 0.5, 0.25])
    >>> z = aikit.asarray([0.01, 0.02, 0.03])
    >>> w = aikit.stable_divide(x, y, min_denominator=z)
    >>> print(w)
    aikit.array([ 0.99,  3.85, 14.3 ])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.asarray([10., 15.]), b=aikit.asarray([20., 25.]))
    >>> y = aikit.stable_divide(x, 0.5)
    >>> print(y)
    {
        a: aikit.array([20., 30.]),
        b: aikit.array([40., 50.])
    }


    >>> x = aikit.Container(a=aikit.asarray([1., 2.]), b=aikit.asarray([3., 4.]))
    >>> y = aikit.Container(a=aikit.asarray([0.5, 2.5]), b=aikit.asarray([3.5, 0.4]))
    >>> z = aikit.stable_divide(x, y)
    >>> print(z)
    {
        a: aikit.array([2., 0.8]),
        b: aikit.array([0.857, 10.])
    }
    """
    return numerator / (denominator + default(min_denominator, aikit.min_denominator))


@handle_exceptions
@handle_nestable
@inputs_to_aikit_arrays
@handle_array_function
def stable_pow(
    base: Union[Number, aikit.Array, aikit.NativeArray],
    exponent: Union[Number, aikit.Array, aikit.NativeArray],
    /,
    *,
    min_base: Optional[float] = None,
) -> Any:
    """Raise the base by the power, with aikit.min_base added to the base when
    exponent > 1 for numerical stability.

    Parameters
    ----------
    base
        The base number.
    exponent
        The exponent number.
    min_base
        The minimum base to use, use global aikit.min_base by default.

    Returns
    -------
    ret
        The new item following the numerically stable power.

    Examples
    --------
    With :code:`int` input:

    >>> x = aikit.stable_pow(2, 2)
    >>> print(x)
    aikit.array(4.00004)

    >>> x = aikit.stable_pow(2, 2, min_base=2)
    >>> print(x)
    aikit.array(16)

    With float input:

    >>> x = aikit.stable_pow(4.0, .5)
    >>> print(x)
    aikit.array(2.00000262)

    With :code:`complex` input:

    >>> x = aikit.stable_pow(3+4j, 2j)
    >>> print(x)
    aikit.array(-0.15605032-0.01208451j)

    With :class:`aikit.Array` input:

    >>> x = aikit.asarray([[2, 4],
    ...                  [6, 8]])
    >>> y = aikit.stable_pow(x, 2)
    >>> print(y)
    aikit.array([[ 4.00004, 16.00008],
           [36.00012, 64.00016]])

    >>> x = aikit.asarray([2, 4, 6])
    >>> y = aikit.asarray([2, 3, 4])
    >>> z = aikit.stable_pow(x, y)
    >>> print(z)
    aikit.array([   4.00004,   64.00048, 1296.00864])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.asarray([2, 4]), b=aikit.asarray([6, 8]))
    >>> y = aikit.stable_pow(x, 2)
    >>> print(y)
    {
        a: aikit.array([4.00004, 16.00008]),
        b: aikit.array([36.00012, 64.00016])
    }

    >>> x = aikit.Container(a=aikit.asarray([2, 4]), b=aikit.asarray([6, 8]))
    >>> y = aikit.Container(a=aikit.asarray([1, 3]), b=aikit.asarray([4, 5]))
    >>> z = aikit.stable_pow(x, y)
    >>> print(z)
    {
        a: aikit.array([2.00001, 64.00048]),
        b: aikit.array([1296.00864, 32768.2048])
    }
    """
    return_dtype = aikit.promote_types(
        aikit.default_dtype(item=base),
        aikit.default_dtype(item=default(min_base, aikit.min_base)),
    )
    return_dtype = aikit.promote_types(return_dtype, aikit.default_dtype(item=exponent))
    ret = (base + default(min_base, aikit.min_base)) ** aikit.array(exponent)
    return ret.astype(return_dtype)


stable_pow.unsupported_dtypes = ("bfloat16",)


@handle_exceptions
def get_all_arrays_in_memory() -> List[Union[aikit.Array, aikit.NativeArray]]:
    """Get all arrays which are currently alive.

    Returns
    -------
    ret
        All arrays which are alive.

    Examples
    --------
    >>> aikit.get_all_arrays_in_memory()
    []
    >>> x = aikit.get_all_arrays_in_memory()
    >>> x
    []
    >>> y = aikit.array([0, 1, 2])
    >>> x
    [aikit.array([0, 1, 2])]
    """
    all_arrays = []
    for obj in gc.get_objects():
        try:
            if aikit.current_backend_str() in ["", "numpy"]:
                if aikit.is_aikit_array(obj):
                    all_arrays.append(obj)
            else:
                if aikit.is_native_array(obj):
                    all_arrays.append(obj)

        except Exception:
            pass
    return all_arrays


@handle_exceptions
def num_arrays_in_memory() -> int:
    """Return the number of arrays which are currently alive.

    Returns
    -------
    ret
        Number of all arrays which are alive.

    Examples
    --------
    >>> aikit.num_arrays_in_memory()
    0
    >>> x = aikit.num_arrays_in_memory()
    >>> x
    0
    >>> y = aikit.array([0, 1, 2])
    >>> x
    1
    """
    return len(get_all_arrays_in_memory())


@handle_exceptions
def print_all_arrays_in_memory():
    """Print all native Aikit arrays in memory to the console.

    Gets all the native Aikit arrays which are currently alive(in the
    garbage collector) from get_all_arrays_in_memory() function and
    prints them to the console.
    """
    for arr in get_all_arrays_in_memory():
        print(type(arr), arr.shape)


aikit.queue_timeout = queue_timeout_stack[-1] if queue_timeout_stack else 15.0


@handle_exceptions
@handle_array_function
def set_queue_timeout(timeout: float):
    """Set a timeout value (in seconds) for the global queue.

    Set the global queue timeout value (in seconds) Default value without this function
    being called is 15 seconds.

    Parameters
    ----------
    timeout
        The timeout when waiting for containers to arrive from the queues.
        To be set in seconds.

    Examples
    --------
    >>> x = aikit.set_queue_timeout(10)
    >>> x = aikit.queue_timeout
    >>> print(x)
    10.0

    >>> aikit.set_queue_timeout(30)
    >>> y = aikit.queue_timeout
    >>> print(y)
    30
    """
    global queue_timeout_stack
    aikit.utils.assertions.check_isinstance(timeout, (int, float))
    queue_timeout_stack.append(timeout)
    aikit.__setattr__("queue_timeout", timeout, True)


@handle_exceptions
def unset_queue_timeout() -> None:
    """Reset the global queue timeout value (in seconds) to the previous state.

    Examples
    --------
    >>> aikit.set_queue_timeout(10.0)
    >>> y = aikit.queue_timeout
    >>> print(y)
    10.0

    >>> aikit.unset_queue_timeout()
    >>> aikit.queue_timeout
    15.0
    """
    global queue_timeout_stack
    if queue_timeout_stack:
        queue_timeout_stack.pop(-1)
        timeout = queue_timeout_stack[-1] if queue_timeout_stack else 15.0
        aikit.__setattr__("queue_timeout", timeout, True)


aikit.tmp_dir = tmp_dir_stack[-1] if tmp_dir_stack else "/tmp"


@handle_exceptions
def set_tmp_dir(tmp_dr: str) -> None:
    """Set the directory for saving temporary files.

    Parameters
    ----------
    tmp_dr
        The new directory for saving temporary files

    Examples
    --------
    >>> x = aikit.tmp_dir
    >>> print(x)
    /tmp

    >>> aikit.set_tmp_dir("/my_tmp")
    >>> y = aikit.tmp_dir
    >>> print(y)
    /my_tmp
    """
    global tmp_dir_stack
    aikit.utils.assertions.check_isinstance(tmp_dr, str)
    tmp_dir_stack.append(tmp_dr)
    aikit.__setattr__("tmp_dir", tmp_dr, True)


@handle_exceptions
def unset_tmp_dir() -> None:
    """Reset the directory for saving temporary files to the previous value.

    Examples
    --------
    >>> aikit.set_tmp_dir("/my_dir")
    >>> y = aikit.tmp_dir
    >>> print(y)
    /my_dir

    >>> aikit.unset_tmp_dir()
    >>> aikit.tmp_dir
    /tmp
    """
    global tmp_dir_stack
    if tmp_dir_stack:
        tmp_dir_stack.pop(-1)
        tmp_dr = tmp_dir_stack[-1] if tmp_dir_stack else "/tmp"
        aikit.__setattr__("tmp_dir", tmp_dr, True)


@handle_exceptions
def container_types():
    """Summary.

    Returns
    -------
    ret
        a key-value structure, and exposes public methods .keys(), .values() and
        items().
    """
    # noinspection PyBroadException
    try:
        return current_backend().container_types()
    except ValueError:
        return []


@handle_exceptions
def inplace_arrays_supported() -> bool:
    """Determine whether inplace arrays are supported for the current backend
    framework.

    Returns
    -------
    ret
        Boolean, whether or not inplace arrays are supported.
    """
    return current_backend().inplace_arrays_supported()


@handle_exceptions
def inplace_variables_supported() -> bool:
    """Determine whether inplace variables are supported for the current
    backend framework.

    Returns
    -------
    ret
        Boolean, whether or not inplace variables are supported.
    """
    return current_backend().inplace_variables_supported()


@handle_exceptions
@handle_nestable
@inputs_to_native_arrays
@handle_array_function
def supports_inplace_updates(x: Union[aikit.Array, aikit.NativeArray], /) -> bool:
    """Return if in-place operations are supported for x's data type.

    Determine whether in-place operations are supported for x's data type, by the
    current backend framework setting.

    Parameters
    ----------
    x
        Input variable for whose data type we check whether the current backend
        framework supports in-place operations.

    Returns
    -------
    ret
        Value depends on whether in-place operations are supported for
        data type of x.

    Raises
    ------
    AikitException
        If x isn't a class instance of aikit.Array or aikit.NativeArray, an exception will
        be raised.

    This function is *nestable*, and therefore also accepts :code:'aikit.Container'
    instance in place of the argument.

    Examples
    --------
    With :class:`aikit.Array` input and default backend set as `numpy`:

    >>> x = aikit.array([0, 1, 2])
    >>> y = aikit.supports_inplace_updates(x)
    >>> print(y)
    True

    With :class:`aikit.Container` input and backend set as `torch`:

    >>> x = aikit.Container(a=aikit.array([5., 6.]), b=aikit.array([7., 8.]))
    >>> y = aikit.supports_inplace_updates(x)
    >>> print(y)
    {
        a: True,
        b: True
    }

    With `aikit.Array` input and backend set as "tensorflow":

    >>> x = aikit.array([1., 4.2, 2.2])
    >>> ret = x.supports_inplace_updates()
    >>> print(ret)
    False
    """
    if _is_variable(x):
        return aikit.inplace_variables_supported()
    elif aikit.is_native_array(x):
        return aikit.inplace_arrays_supported()
    raise aikit.utils.exceptions.AikitException(
        "Input x must be either a variable or an array."
    )


@handle_exceptions
@handle_nestable
@inputs_to_native_arrays
@handle_array_function
def assert_supports_inplace(x: Union[aikit.Array, aikit.NativeArray], /) -> bool:
    """Assert that inplace operations are supported for x.

    Parameters
    ----------
    x
        Input variable or array to check for inplace support for.

    Returns
    -------
    ret
        True if supports, raises AikitBackendException otherwise

    This function is *nestable*, and therefore also accepts :code:'aikit.Container'
    instance in place of the argument.

    Examples
    --------
    With :class:`aikit.Array` input and default backend set as `numpy`:

    >>> aikit.set_backend("numpy")
    >>> x = aikit.array([1, 2, 3])
    >>> print(x.assert_supports_inplace())
    True

    With :class:`aikit.Array` input and default backend set as `torch`:

    >>> aikit.set_backend("torch")
    >>> x = aikit.array([1, 2, 3])
    >>> print(x.assert_supports_inplace())
    True

    With :class:`aikit.Container` input and default backend set as `numpy`:

    >>> aikit.set_backend("numpy")
    >>> x = aikit.Container(a=aikit.array([5, 6]), b=aikit.array([7, 8]))
    >>> print(x.assert_supports_inplace())
    {
        a: True,
        b: True
    }

    With :class:`aikit.Container` input and default backend set as `torch`:

    >>> aikit.set_backend("torch")
    >>> x = aikit.Container(a=aikit.array([5, 6]), b=aikit.array([7, 8]))
    >>> print(x.assert_supports_inplace())
    {
        a: True,
        b: True
    }
    """
    aikit.utils.assertions.check_true(
        aikit.supports_inplace_updates(x),
        f"Inplace operations are not supported {type(x)} types with"
        f" {aikit.current_backend_str()} backend",
    )
    return True


@handle_nestable
@handle_partial_mixed_function
@handle_view_indexing
@inputs_to_aikit_arrays
@handle_array_function
@handle_device
def get_item(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    query: Union[aikit.Array, aikit.NativeArray, Tuple],
    *,
    copy: Optional[bool] = None,
) -> aikit.Array:
    """Gather slices from x according to query array, identical to x[query].

    Parameters
    ----------
    x
        array, the array from which to gather values.
    query
        array, index array, integer indices or boolean mask.
    copy
        boolean indicating whether or not to copy the input array.
        If True, the function must always copy.
        If False, the function must never copy.
        In case copy is False we avoid copying by returning a view of the input array.

    Returns
    -------
    ret
        New array with the values gathered at the specified indices.

    Examples
    --------
    >>> x = aikit.array([0, -1, 20])
    >>> query = aikit.array([0, 1])
    >>> print(aikit.get_item(x, query))
    aikit.array([ 0, -1])

    >>> x = aikit.array([[4, 5], [20, 128], [-2, -10]])
    >>> query = aikit.array([[True, False], [False, False], [True, True]])
    >>> print(aikit.get_item(x, query))
    aikit.array([  4,  -2, -10])
    """
    if aikit.is_array(query) and aikit.is_bool_dtype(query):
        if query.ndim == 0:
            if query is False:
                return aikit.zeros(shape=(0,) + x.shape, dtype=x.dtype)
            return x[None]  # equivalent to aikit.expand_dims(x, axis=0)
        query = aikit.nonzero(query, as_tuple=False)
        ret = aikit.gather_nd(x, query)
    else:
        indices, target_shape = _parse_query(query, x.shape)
        if indices is None:
            return aikit.empty(target_shape, dtype=x.dtype)
        ret = aikit.gather_nd(x, indices)
        ret = aikit.reshape(ret, target_shape)
    return ret


get_item.mixed_backend_wrappers = {
    "to_add": (
        "handle_backend_invalid",
        "inputs_to_native_arrays",
        "outputs_to_aikit_arrays",
    ),
    "to_skip": ("inputs_to_aikit_arrays",),
}


@handle_nestable
@handle_partial_mixed_function
@inputs_to_aikit_arrays
@handle_array_function
def set_item(
    x: Union[aikit.Array, aikit.NativeArray],
    query: Union[aikit.Array, aikit.NativeArray, Tuple],
    val: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    copy: Optional[bool] = False,
) -> aikit.Array:
    """Replace slices of x (defined by query) with val, identical to x[query] =
    val.

    Parameters
    ----------
    x
        the array to be updated.
    query
        either an index array, or a tuple of integers or slices.
    val
        the array containing the values to be infused into x
    copy
        boolean indicating whether to copy x.
        If True, the function will update and return a copy of x.
        If False, the function will update x inplace.

    Returns
    -------
    ret
        the array with updated values at the specified indices.

    Examples
    --------
    >>> x = aikit.array([0, -1, 20])
    >>> query = aikit.array([0, 1])
    >>> val = aikit.array([10, 10])
    >>> aikit.set_item(x, query, val)
    >>> print(x)

    aikit.array([10, 10, 20])
    >>> x = aikit.array([[0, -1, 20], [5, 2, -8]])
    >>> query = ([1, 1])
    >>> val = aikit.array([10, 10])
    >>> y = aikit.set_item(x, query, val, copy=True)
    >>> print(y)
    aikit.array([[ 0, -1, 20],
           [10, 10, 10]])
    """
    if copy:
        x = aikit.copy_array(x)
    if not aikit.is_array(val):
        val = aikit.array(val)
    if 0 in x.shape or 0 in val.shape:
        return x
    if aikit.is_array(query) and aikit.is_bool_dtype(query):
        if not len(query.shape):
            query = aikit.tile(query, (x.shape[0],))
        target_shape = aikit.get_item(x, query).shape
        indices = aikit.nonzero(query, as_tuple=False)
    else:
        indices, target_shape = _parse_query(query, x.shape)
        if indices is None:
            return x
    val = _broadcast_to(val, target_shape).astype(x.dtype)
    ret = aikit.scatter_nd(indices, val, reduction="replace", out=x)
    return ret


set_item.mixed_backend_wrappers = {
    "to_add": (
        "handle_backend_invalid",
        "inputs_to_native_arrays",
        "outputs_to_aikit_arrays",
    ),
    "to_skip": ("inputs_to_aikit_arrays",),
}


def _parse_query(query, x_shape):
    query = query if isinstance(query, tuple) else (query,)
    query_ = tuple(q.to_numpy() if aikit.is_array(q) else q for q in query)

    # array containing all of x's flat indices
    x_ = aikit.arange(0, _numel(x_shape)).reshape(x_shape)

    # use numpy's __getitem__ to get the queried indices
    x_idxs = aikit.array(x_.to_numpy()[query_])
    target_shape = x_idxs.shape

    if 0 in x_idxs.shape or 0 in x_shape:
        return None, target_shape

    # convert the flat indices to multi-D indices
    x_idxs = aikit.unravel_index(x_idxs, x_shape)

    # stack the multi-D indices to bring them to gather_nd/scatter_nd format
    x_idxs = aikit.stack(x_idxs, axis=-1).astype(aikit.int64)

    return x_idxs, target_shape


def _numel(shape):
    shape = tuple(shape)
    return aikit.prod(shape).to_scalar() if shape != () else 1


def _broadcast_to(input, target_shape):
    if _numel(tuple(input.shape)) == _numel(tuple(target_shape)):
        return aikit.reshape(input, target_shape)
    else:
        input = input if len(input.shape) else aikit.expand_dims(input, axis=0)
        new_dims = ()
        i_i = len(input.shape) - 1
        for i_t in range(len(target_shape) - 1, -1, -1):
            if len(input.shape) + len(new_dims) >= len(target_shape):
                break
            if i_i < 0 or target_shape[i_t] != input.shape[i_i]:
                new_dims += (i_t,)
            else:
                i_i -= 1
        input = aikit.expand_dims(input, axis=new_dims)
        return aikit.broadcast_to(input, target_shape)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@inputs_to_aikit_arrays
@handle_array_function
@handle_device
def inplace_update(
    x: Union[aikit.Array, aikit.NativeArray],
    val: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    ensure_in_backend: bool = False,
    keep_input_dtype: bool = False,
) -> aikit.Array:
    """Perform in-place update for the input array.

    This will always be performed on aikit.Array instances pass in the input, and will
    also be performed on the native array classes in the backend when the backend
    supports this. If the backend does not natively support inplace updates, and x is an
    aikit.NativeArray instance, then an
    exception will be thrown.

    Parameters
    ----------
    x
        The variable to update.
    val
        The array to update the variable with.
    ensure_in_backend
        Whether or not to ensure that the `aikit.NativeArray` is also inplace updated.
        In cases where it should be, backends which do not natively support inplace
        updates will raise an exception.
    keep_input_dtype
        Whether or not to preserve `x` data type after the update, otherwise `val`
        data type will be applied. Defaults to False.

    Returns
    -------
    ret
        The array following the in-place update.

    Raises
    ------
    AikitException
        If backend set doesn't natively support inplace updates and ensure_in_backend is
        True, above exception will be raised.

    This function is *nestable*, and therefore also accepts :code:'aikit.Container'
    instance in place of the arguments.

    Examples
    --------
    With :class:`aikit.Array` input and default backend set as `numpy`:

    >>> aikit.set_backend("numpy")
    >>> x = aikit.array([1, 2, 3])
    >>> y = aikit.array([0])
    >>> aikit.inplace_update(x, y)
    >>> print(x)
    aikit.array([0])

    With :class:`aikit.Array` input and default backend set as `numpy`:

    >>> aikit.set_backend("numpy")
    >>> x = aikit.array([1, 2, 3], dtype=aikit.float32)
    >>> y = aikit.array([0, 0, 0], dtype=aikit.int32)
    >>> aikit.inplace_update(x, y, keep_input_dtype=True)
    >>> print(x)
    aikit.array([0., 0., 0.])

    With :class:`aikit.Container` instances:, and backend set as `torch`:

    >>> aikit.set_backend("torch")
    >>> x = aikit.Container(a=aikit.array([5, 6]), b=aikit.array([7, 8]))
    >>> y = aikit.Container(a=aikit.array([1]), b=aikit.array([2]))
    >>> aikit.inplace_update(x, y)
    >>> print(x)
    {
        a: aikit.array([1, 1]),
        b: aikit.array([2, 2])
    }

    With mix of :class:`aikit.Array` and :class:`aikit.Container` instances:, and backend
    set as `torch`:

    >>> aikit.set_backend("torch")
    >>> x = aikit.Container(a=aikit.array([5, 6]), b=aikit.array([7, 8]))
    >>> y = aikit.array([1, 2])
    >>> aikit.inplace_update(x, y)
    >>> print(x)
    {
        a: aikit.array([1, 2]),
        b: aikit.array([1, 2])
    }
    """
    return current_backend(x).inplace_update(
        x,
        val,
        ensure_in_backend=ensure_in_backend,
        keep_input_dtype=keep_input_dtype,
    )


inplace_update.unsupported_dtypes = {"torch": ("bfloat16",)}

aikit.inplace_mode = inplace_mode_stack[-1] if inplace_mode_stack else "lenient"


@handle_exceptions
def set_inplace_mode(mode: str = "lenient") -> None:
    """Set the memory management behavior for in-place updates in Aikit.

    By default, Aikit creates new arrays in the backend for in-place updates.
    However, this behavior can be controlled by the user
    using the 'inplace_mode' parameter.

    Parameters
    ----------
    mode : str
        The mode for memory management during in-place updates.
        - 'lenient': (Default) In this mode, new arrays will be created during
                    in-place updates to avoid breaking existing code.
                    This is the default behavior.
        - 'strict': In this mode, an error will be raised if the
                    'inplace_update' function is called
                    in a backend that doesn't support inplace updates natively.

    Returns
    -------
    None

    Examples
    --------
    >>> set_inplace_mode('lenient')
    >>> aikit.inplace_mode
    'lenient'

    >>> set_inplace_mode('strict')
    >>> aikit.inplace_mode
    'strict'

    Note
    ----
    Enabling strict mode can help users have more control over memory management
    but may lead to errors if the backend doesn't support inplace updates natively.
    """
    global inplace_mode_stack
    inplace_modes = ["lenient", "strict"]
    aikit.utils.assertions.check_elem_in_list(
        mode, inplace_modes, False, f"inplace mode must be one of {inplace_modes}"
    )
    inplace_mode_stack.append(mode)
    aikit.__setattr__("inplace_mode", mode, True)


@handle_exceptions
def unset_inplace_mode() -> None:
    """Reset the memory management behavior for in-place updates in Aikit to the
    previous state.

    Examples
    --------
    >>> set_inplace_mode('strict')
    >>> aikit.inplace_mode
    'strict'

    >>> unset_inplace_mode()
    >>> aikit.inplace_mode
    'lenient'
    """
    global inplace_mode_stack
    if inplace_mode_stack:
        inplace_mode_stack.pop(-1)
        mode = inplace_mode_stack[-1] if inplace_mode_stack else "lenient"
        aikit.__setattr__("inplace_mode", mode, True)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@inputs_to_aikit_arrays
@handle_array_function
@handle_device
def inplace_decrement(
    x: Union[aikit.Array, aikit.NativeArray],
    val: Union[aikit.Array, aikit.NativeArray],
) -> aikit.Array:
    """Perform in-place decrement for the input array.

    Parameters
    ----------
    x
        The input array to be decremented by the defined value.
    val
        The value of decrement.

    Returns
    -------
    ret
        The array following the in-place decrement.


    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([[5.3, 7., 0.],[6.8, 8, 3.9],[0., 10., 6.3]])
    >>> y = aikit.inplace_decrement(x, 1.25)
    >>> print(y)
    aikit.array([[ 4.05,  5.75, -1.25],
       [ 5.55,  6.75,  2.65],
       [-1.25,  8.75,  5.05]])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([0.5, -5., 30.]), b=aikit.array([0., -25., 50.]))
    >>> y = aikit.inplace_decrement(x, 1.5)
    >>> print(y)
    {
        a: aikit.array([-1., -6.5, 28.5]),
        b: aikit.array([-1.5, -26.5, 48.5])
    }

    >>> x = aikit.Container(a=aikit.array([0., 15., 30.]), b=aikit.array([0., 25., 50.]))
    >>> y = aikit.Container(a=aikit.array([0., 15., 30.]), b=aikit.array([0., 25., 50.]))
    >>> z = aikit.inplace_decrement(x, y)
    >>> print(z)
    {
        a: aikit.array([0., 0., 0.]),
        b: aikit.array([0., 0., 0.])
    }

    >>> x = aikit.Container(a=aikit.array([3., 7., 10.]), b=aikit.array([0., 75., 5.5]))
    >>> y = aikit.Container(a=aikit.array([2., 5.5, 7.]), b=aikit.array([0., 25., 2.]))
    >>> z = aikit.inplace_decrement(x, y)
    >>> print(z)
    {
        a: aikit.array([1., 1.5, 3.]),
        b: aikit.array([0., 50., 3.5])
    }
    """
    return current_backend(x).inplace_decrement(x, val)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@inputs_to_aikit_arrays
@handle_array_function
@handle_device
def inplace_increment(
    x: Union[aikit.Array, aikit.NativeArray],
    val: Union[aikit.Array, aikit.NativeArray],
) -> aikit.Array:
    """Perform in-place increment for the input array.

    Parameters
    ----------
    x
        The input array to be incremented by the defined value.
    val
        The value of increment.

    Returns
    -------
    ret
        The array following the in-place increment.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([[5.3, 7., 0.],[6.8, 8, 3.9],[0., 10., 6.3]])
    >>> y = aikit.inplace_increment(x, 3.)
    >>> print(y)
    aikit.array([[ 8.3, 10.,  3.],
       [ 9.8, 11.,  6.9],
       [ 3., 13.,  9.3]])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([0., 15., 30.]), b=aikit.array([0., 25., 50.]))
    >>> y = aikit.inplace_increment(x, 2.5)
    >>> print(y)
    {
        a: aikit.array([2.5, 17.5, 32.5]),
        b: aikit.array([2.5, 27.5, 52.5])
    }


    >>> x = aikit.Container(a=aikit.array([0., 15., 30.]), b=aikit.array([0., 25., 50.]))
    >>> y = aikit.Container(a=aikit.array([0., 15., 30.]), b=aikit.array([0., 25., 50.]))
    >>> z = aikit.inplace_increment(x, y)
    >>> print(z)
    {
        a: aikit.array([0., 30., 60.]),
        b: aikit.array([0., 50., 100.])
    }
    """
    return current_backend(x).inplace_increment(x, val)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
@handle_array_function
@handle_device
def scatter_flat(
    indices: Union[aikit.Array, aikit.NativeArray],
    updates: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    size: Optional[int] = None,
    reduction: str = "sum",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Scatter flat updates into a new flat array according to flat indices.

    Parameters
    ----------
    indices
        Indices for the new values to occupy.
    updates
        Values for the new array to hold.
    size
        The size of the result. Default is `None`, in which case tensor
        argument out must be provided.
    reduction
        The reduction method for the scatter, one of 'sum', 'min', 'max' or 'replace'
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        New array of given shape, with the values scattered at the indices.

    This function is *nestable*, and therefore also accepts :code:'aikit.Container'
    instance in place of the argument.

    Examples
    --------
    With :class:`aikit.Array` input:
    >>> indices = aikit.array([0, 0, 1, 0, 2, 2, 3, 3])
    >>> updates = aikit.array([5, 1, 7, 2, 3, 2, 1, 3])
    >>> out = aikit.array([0, 0, 0, 0, 0, 0, 0, 0])
    >>> aikit.scatter_flat(indices, updates, out=out)
    >>> print(out)
    aikit.array([8, 7, 5, 4, 0, 0, 0, 0])


    With :class:`aikit.Array` input:
    >>> indices = aikit.array([1, 0, 1, 0, 2, 2, 3, 3])
    >>> updates = aikit.array([9, 2, 0, 2, 3, 2, 1, 8])
    >>> size = 8
    >>> print(aikit.scatter_flat(indices, updates, size=size))
    aikit.array([2, 0, 2, 8, 0, 0, 0, 0])


    With :class:`aikit.Container` and :class:`aikit.Array` input:
    >>> indices = aikit.array([1, 0, 1, 0, 2, 2, 3, 3])
    >>> updates = aikit.Container(a=aikit.array([9, 2, 0, 2, 3, 2, 1, 8]),
    ...                 b=aikit.array([5, 1, 7, 2, 3, 2, 1, 3]))
    >>> size = 8
    >>> print(aikit.scatter_flat(indices, updates, size=size))
    {
        a: aikit.array([2, 0, 2, 8, 0, 0, 0, 0]),
        b: aikit.array([2, 7, 2, 3, 0, 0, 0, 0])
    }


    With :class:`aikit.Container` input:
    >>> indices = aikit.Container(a=aikit.array([1, 0, 1, 0, 2, 2, 3, 3]),
    ...                 b=aikit.array([0, 0, 1, 0, 2, 2, 3, 3]))
    >>> updates = aikit.Container(a=aikit.array([9, 2, 0, 2, 3, 2, 1, 8]),
    ...                 b=aikit.array([5, 1, 7, 2, 3, 2, 1, 3]))
    >>> size = 8
    >>> print(aikit.scatter_flat(indices, updates, size=size))
    {
        a: aikit.array([2, 0, 2, 8, 0, 0, 0, 0]),
        b: aikit.array([2, 7, 2, 3, 0, 0, 0, 0])
    }
    """
    return current_backend(indices).scatter_flat(
        indices, updates, size=size, reduction=reduction, out=out
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@inputs_to_native_shapes
@to_native_arrays_and_back
@handle_array_function
@handle_device
def scatter_nd(
    indices: Union[aikit.Array, aikit.NativeArray],
    updates: Union[aikit.Array, aikit.NativeArray],
    /,
    shape: Optional[Union[tuple, list, aikit.Array, aikit.Shape, aikit.NativeShape]] = None,
    *,
    reduction: str = "sum",
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Scatter updates into a new array according to indices.

    Parameters
    ----------
    indices
        Indices for the new values to occupy.
    updates
        Values for the new array to hold.
    shape
        The shape of the result. Default is ``None``, in which case tensor
        argument must be provided.
    reduction
        The reduction method for the scatter, one of 'sum', 'min', 'max' or 'replace'
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        New array of given shape, with the values scattered at the indices.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> indices = aikit.array([[4], [3], [7], [7]])
    >>> updates = aikit.array([9, 12, 11, 10])
    >>> shape = aikit.array([8])
    >>> scatter = aikit.scatter_nd(indices, updates, shape)
    >>> print(scatter)
    aikit.array([ 0,  0,  0, 12,  9,  0,  0, 21])

    >>> indices = aikit.array([[0, 1], [1, 0], [1, 1], [1, 1]])
    >>> updates = aikit.array([9, 11, 12, 10])
    >>> shape = (2, 2)
    >>> scatter = aikit.scatter_nd(indices, updates, shape, reduction="max")
    >>> print(scatter)
    aikit.array([[ 0,  9], [11, 12]])

    >>> indices = aikit.array([[[0], [1]], [[2], [1]]])
    >>> updates = aikit.array([[9, 12], [11, 10]])
    >>> shape = [4]
    >>> scatter = aikit.scatter_nd(indices, updates, shape, reduction="replace")
    >>> print(scatter)
    aikit.array([ 9, 10, 11,  0])

    >>> indices = aikit.array([[[1, 1], [0, 0]], [[1, 1], [0, 0]]])
    >>> updates = aikit.array([[-1, 12], [11, 10]])
    >>> shape = aikit.Shape([2, 2])
    >>> result = aikit.zeros([2, 2])
    >>> scatter = aikit.scatter_nd(indices, updates, shape, reduction="min", out=result)
    >>> print(result)
    aikit.array([[ 0.,  0.], [ 0., -1.]])

    With :class:`aikit.Container` input:

    >>> indices = aikit.Container(a=aikit.array([[4],[3],[6]]),
    ...                         b=aikit.array([[5],[1],[2]]))
    >>> updates = aikit.Container(a=aikit.array([100, 200, 200]),
    ...                         b=aikit.array([20, 30, 40]))
    >>> shape = aikit.Container(a=aikit.array([10]),
    ...                       b=aikit.array([10]))
    >>> z = aikit.scatter_nd(indices, updates, shape=shape)
    >>> print(z)
    {
        a: aikit.array([0, 0, 0, 200, 100, 0, 200, 0, 0, 0]),
        b: aikit.array([0, 30, 40, 0, 0, 20, 0, 0, 0, 0])
    }

    With :class:`aikit.Container` and :class:`aikit.Array` input:

    >>> indices = aikit.array([[4],[3],[1]])
    >>> updates = aikit.Container(a=aikit.array([10, 20, 30]),
    ...                         b=aikit.array([200, 300, 400]))
    >>> z = aikit.Container(a=aikit.array([1, 2, 3, 4, 5]),
    ...                   b=aikit.array([10, 20, 30, 40, 50]))
    >>> aikit.scatter_nd(indices, updates, reduction="replace", out=z)
    >>> print(z)
    {
        a: aikit.array([1, 30, 3, 20, 10]),
        b: aikit.array([10, 400, 30, 300, 200])
    }
    """
    return current_backend(indices).scatter_nd(
        indices, updates, shape=shape, reduction=reduction, out=out
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def gather(
    params: Union[aikit.Array, aikit.NativeArray],
    indices: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    axis: int = -1,
    batch_dims: int = 0,
    out: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
) -> Union[aikit.Array, aikit.NativeArray]:
    """Gather slices from params at axis according to indices.

    Parameters
    ----------
    params
        The array from which to gather values.
    indices
        The array which indicates the indices that will be gathered along
        the specified axis.
    axis
        Optional int, the axis from which to gather from.
        Default is ``-1``.
    batch_dims
        Optional int, lets you gather different items from each element of a batch.
        Default is ``0``.
    out
        Optional array, for writing the result to. It must have a shape
        that the inputs broadcast to.

    Returns
    -------
    ret
        New array with the values gathered at the specified indices along the
        specified axis.


    Both the description and the type hints above assumes an array input for
    simplicity, but this function is *nestable*, and therefore also accepts
    :class:`aikit.Container` instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([0., 1., 2.])
    >>> y = aikit.array([1, 2])
    >>> print(aikit.gather(x, y))
    aikit.array([1., 2.])

    >>> x = aikit.array([[0., 1., 2.],[3., 4., 5.]])
    >>> y = aikit.array([[0, 1],[1, 2]])
    >>> z = aikit.zeros((2, 2, 2))
    >>> aikit.gather(x, y, out=z)
    >>> print(z)
    aikit.array([[[0., 1.],[1., 2.]],[[3., 4.],[4., 5.]]])

    >>> x = aikit.array([[[0., 1.], [2., 3.]],
    ...                [[8., 9.], [10., 11.]]])
    >>> y = aikit.array([[0, 1]])
    >>> z = aikit.zeros((1, 2, 2, 2))
    >>> aikit.gather(x, y, axis=0, out=z)
    >>> print(z)
    aikit.array(
        [[[[ 0.,  1.],
           [ 2.,  3.]],
          [[ 8.,  9.],
           [10., 11.]]]])

    >>> x = aikit.array([[0, 10, 20, 0, 0],
    ...                [0, 0, 0, 30, 40],
    ...                [0, 10, 0, 0, 40]])
    >>> y = aikit.array([[1, 2],[3, 4],[1, 4]])
    >>> z = aikit.gather(x, y, batch_dims=1)
    >>> print(z)
    aikit.array([[10, 20], [30, 40],[10, 40]])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a = aikit.array([0., 1., 2.]),
    ...                   b = aikit.array([4., 5., 6.]))
    >>> y = aikit.Container(a = aikit.array([0, 1]),
    ...                   b = aikit.array([1, 2]))
    >>> print(aikit.gather(x, y))
    {
        a: aikit.array([0., 1.]),
        b: aikit.array([5., 6.])
    }

    With a mix of :class:`aikit.Array` and :class:`aikit.Container` inputs:

    >>> x = aikit.Container(a = aikit.array([0., 1., 2.]),
    ...                   b = aikit.array([4., 5., 6.]))
    >>> y = aikit.array([0, 1])
    >>> print(aikit.gather(x, y))
    {
        a: aikit.array([0., 1.]),
        b: aikit.array([4., 5.])
    }
    """
    return current_backend(params, indices).gather(
        params, indices, axis=axis, batch_dims=batch_dims, out=out
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def gather_nd(
    params: Union[aikit.Array, aikit.NativeArray],
    indices: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    batch_dims: int = 0,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Gather slices from params into a array with shape specified by indices.

    Parameters
    ----------
    params
        The array from which to gather values.
    indices
        Index array.
    batch_dims
        optional int, lets you gather different items from each element of a batch.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        New array of given shape, with the values gathered at the indices.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([0., 1., 2., 3., 4., 5., 6.])
    >>> y = aikit.array([1])
    >>> print(aikit.gather_nd(x, y))
    aikit.array(1.)

    >>> x = aikit.array([[0., 1.], [2., 3.], [4., 5.]])
    >>> y = aikit.array([[0],[1],[1]], dtype='int32')
    >>> z = aikit.gather_nd(x,y,batch_dims=1)
    aikit.array([0., 3., 5.])

    With a mix of :class:`aikit.Array` and :class:`aikit.Container` inputs:

    >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),b=aikit.array([4., 5., 6.]))
    >>> y = aikit.array([1])
    >>> print(aikit.gather_nd(x, y))
    {
        a: aikit.array(1.),
        b: aikit.array(5.)
    }

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([[0., 10., 20.],[30.,40.,50.]]),
    ...                   b=aikit.array([[0., 100., 200.],[300.,400.,500.]]))
    >>> y = aikit.Container(a=aikit.array([1,0]),
    ...                   b=aikit.array([0]))
    >>> print(aikit.gather_nd(x, y))
    {
        a: aikit.array(30.),
        b: aikit.array([0., 100., 200.])
    }
    """
    res = current_backend(params, indices).gather_nd(
        params, indices, batch_dims=batch_dims
    )
    if aikit.exists(out):
        return aikit.inplace_update(out, res)
    return res


@handle_exceptions
@handle_nestable
@handle_array_function
def multiprocessing(context: Optional[str] = None):
    """Return backend-specific multiprocessing module.

    Parameters
    ----------
    context
        The context of the multiprocessing, either 'fork', 'forkserver' or 'spawn'.
        Default is ``None``.

    Returns
    -------
    ret
        Multiprocessing module

    Examples
    --------
    >>> import aikit

    Using the default context (None):

    >>> mp_default = aikit.multiprocessing()
    >>> print(mp_default)
    <multiprocessing.context.DefaultContext object at 0x7f4e3193e520>

    Specifying 'fork' as the context:

    >>> mp_fork = aikit.multiprocessing(context='fork')
    >>> print(mp_fork)
    <multiprocessing.context.ForkContext object at 0x7f4e3193e580>

    Specifying 'spawn' as the context:

    >>> mp_spawn = aikit.multiprocessing(context='spawn')
    >>> print(mp_spawn)
    <multiprocessing.context.SpawnContext object at 0x7f4e3193e5e0>

    Specifying 'forkserver' as the context:

    >>> mp_forkserver = aikit.multiprocessing(context='forkserver')
    >>> print(mp_forkserver)
    <multiprocessing.context.ForkServerContext object at 0x7f4e3193e640>
    """
    return current_backend().multiprocessing(context)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_native_arrays
@outputs_to_aikit_shapes
@outputs_to_aikit_arrays
@handle_array_function
@handle_device
def shape(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    as_array: bool = False,
) -> Union[aikit.Shape, aikit.NativeShape]:
    """Return the shape of the array ``x``.

    Parameters
    ----------
    x
        Input array to infer the shape of.
    as_array
        Whether to return the shape as an array.
        Default is False.

    Returns
    -------
    ret
        Shape of the array ``x``.

    Examples
    --------
    >>> x = aikit.array([[-1, 0, 1], [1, 0, -1]])
    >>> y = aikit.shape(x)
    >>> z = aikit.shape(x, as_array = True)
    >>> print(y)
    (2, 3)

    >>> print(z)
    aikit.array([2, 3])
    """
    return current_backend(x).shape(x, as_array=as_array)


aikit.shape_array_mode = shape_array_mode_stack[-1] if shape_array_mode_stack else False


@handle_exceptions
def set_shape_array_mode(mode: bool) -> None:
    """Set the mode of returning shape as aikit.Array to the given mode instance.

    Parameter
    ---------
    mode
        boolean whether to return shape as aikit.Array

    Examples
    --------
    >>> aikit.set_shape_array_mode(False)
    >>> aikit.shape_array_mode
    False

    >>> aikit.set_shape_array_mode(True)
    >>> aikit.shape_array_mode
    True
    """
    global shape_array_mode_stack
    aikit.utils.assertions.check_isinstance(mode, bool)
    shape_array_mode_stack.append(mode)
    aikit.__setattr__("shape_array_mode", mode, True)


@handle_exceptions
def unset_shape_array_mode() -> None:
    """Reset the mode of returning shape as aikit.Array to the previous state.

    Examples
    --------
    >>> aikit.set_shape_array_mode(True)
    >>> aikit.shape_array_mode
    True

    >>> aikit.unset_shape_array_mode()
    >>> aikit.shape_array_mode
    False
    """
    global shape_array_mode_stack
    if shape_array_mode_stack:
        shape_array_mode_stack.pop(-1)
        mode = shape_array_mode_stack[-1] if shape_array_mode_stack else False
        aikit.__setattr__("shape_array_mode", mode, True)


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@to_native_arrays_and_back
@handle_array_function
@handle_device
def get_num_dims(
    x: Union[aikit.Array, aikit.NativeArray], /, *, as_array: bool = False
) -> int:
    """Return the number of dimensions of the array x.

    Parameters
    ----------
    x
        Input array to infer the number of dimensions for.
    as_array
        Whether to return the shape as a array, default False.

    Returns
    -------
    ret
        Shape of the array

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> a = aikit.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ...                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ...                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    >>> b = aikit.get_num_dims(a, as_array=False)
    >>> print(b)
    3

    With :class:`aikit.Container` input:

    >>> a = aikit.Container(b = aikit.asarray([[0.,1.,1.],[1.,0.,0.],[8.,2.,3.]]))
    >>> print(aikit.get_num_dims(a))
    {
        b: 2
    }

    >>> b = aikit.get_num_dims(a, as_array=True)
    >>> print(b)
    {
        b: aikit.array(2)
    }
    """
    return current_backend(x).get_num_dims(x, as_array=as_array)


@handle_exceptions
def arg_info(fn: Callable, *, name: Optional[str] = None, idx: Optional[int] = None):
    """Return the index and `inspect.Parameter` representation of the specified
    argument. In the form of a dict with keys "idx" and "param".

    Parameters
    ----------
    fn
        The function to retrieve the argument information for
    name
        The name of the argument
    idx
        the index of the argument in the inputs

    Returns
    -------
    ret
        a `dict` containing the idx, and the `inspect.Parameter` for the argument,
        which itself contains the parameter name, type, and other helpful information.
    """
    aikit.utils.assertions.check_all_or_any_fn(
        name,
        idx,
        fn=aikit.exists,
        type="any",
        limit=[1],
        message="exactly one of the keyword arguments name or idx must be provided",
        as_array=False,
    )
    params = inspect.signature(fn).parameters
    if aikit.exists(name):
        return {"idx": list(params).index(name), "param": params[name]}
    return {"idx": idx, "param": list(params.values())[idx]}


def _valid_attrib_combinations(fn, backend, dnd_dict, first_attr_name, other_attr_name):
    attr_list = ()
    if hasattr(fn, other_attr_name):
        attr_list = getattr(fn, other_attr_name)
        if isinstance(attr_list, dict):
            attr_list = attr_list.get(backend, ())
    aikit.utils.assertions.check_false(
        dnd_dict and attr_list,
        f"Cannot specify both {first_attr_name} and {other_attr_name} "
        "cannot both be defined for the same function",
    )


def _is_valid_device_and_dtypes_attributes(fn: Callable) -> bool:
    fn_unsupported_dnd = {}
    fn_supported_dnd = {}
    backend = aikit.current_backend_str()
    if hasattr(fn, "unsupported_device_and_dtype"):
        fn_unsupported_dnd = fn.unsupported_device_and_dtype
        # if it's a nested dict, unwrap for the current backend
        if fn_unsupported_dnd and isinstance(
            list(fn_unsupported_dnd.__get__().values())[0], dict
        ):
            fn_unsupported_dnd = fn_unsupported_dnd.get(backend, {})
    if hasattr(fn, "supported_device_and_dtype"):
        fn_supported_dnd = fn.supported_device_and_dtype
        # if it's a nested dict, unwrap for the current backend
        if fn_supported_dnd and isinstance(
            list(fn_supported_dnd.__get__().values())[0], dict
        ):
            fn_supported_dnd = fn_supported_dnd.get(backend, {})

    aikit.utils.assertions.check_false(
        fn_unsupported_dnd and fn_supported_dnd,
        "unsupported_device_and_dtype and supported_device_and_dtype cannot"
        " both be defined for the same function",
    )

    us = "unsupported_device_and_dtype"
    _valid_attrib_combinations(fn, backend, fn_unsupported_dnd, us, "supported_devices")
    _valid_attrib_combinations(fn, backend, fn_unsupported_dnd, us, "supported_dtypes")

    ss = "supported_device_and_dtype"
    _valid_attrib_combinations(fn, backend, fn_supported_dnd, ss, "unsupported_device")
    _valid_attrib_combinations(fn, backend, fn_supported_dnd, ss, "unsupported_dtypes")

    return True


def _all_dnd_combinations():
    all_comb = {}
    for device in aikit.all_devices:
        all_comb[device] = aikit.all_dtypes
    return all_comb


def _dnd_dict_intersection(a, b):
    res = {}
    for device in a:
        if device in b:
            intersection = set.intersection(set(a[device]), set(b[device]))
            if intersection:
                res[device] = tuple(intersection)
    return res


def _dnd_dict_difference(a, b):
    res = a
    for device in list(a):
        if device in b:
            difference = set.difference(set(a[device]), set(b[device]))
            if difference:
                res[device] = tuple(difference)
            else:
                del res[device]
    return res


def _dnd_dict_union(a, b):
    res = {}
    for device in set(list(a) + list(b)):
        u1 = set(a.get(device, ()))
        u2 = set(b.get(device, ()))
        res[device] = tuple(set.union(u1, u2))

    return res


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


def _get_devices_and_dtypes(fn, recurse=False, complement=True):
    supported_devices = aikit.function_supported_devices(fn, recurse=recurse)
    supported_dtypes = aikit.function_supported_dtypes(fn, recurse=recurse)

    if hasattr(fn, "partial_mixed_handler"):
        supported_devices = supported_devices["primary"]
        supported_dtypes = supported_dtypes["primary"]

    supported = {}
    # Generate a base supported set from other attributes
    for device in supported_devices:
        supported[device] = supported_dtypes

    is_backend_fn = "backend" in fn.__module__
    is_frontend_fn = "frontend" in fn.__module__
    is_einops_fn = "einops" in fn.__name__
    if not is_backend_fn and not is_frontend_fn and not is_einops_fn:
        if complement:
            all_comb = _all_dnd_combinations()
            supported = _dnd_dict_difference(all_comb, supported)
        return supported

    backend = aikit.current_backend_str()

    # Their values are formatted like either
    # 1. fn.supported_device_and_dtype = {"cpu":("float16",)}
    if hasattr(fn, "supported_device_and_dtype"):
        fn_supported_dnd = fn.supported_device_and_dtype.__get__()

        if "einops" in fn.__name__ and isinstance(fn_supported_dnd, dict):
            fn_supported_dnd = fn_supported_dnd.get(backend, supported)

        if fn_supported_dnd:
            aikit.utils.assertions.check_isinstance(
                list(fn_supported_dnd.values())[0], tuple
            )

        if isinstance(fn_supported_dnd, dict):
            for device, dtypes in fn_supported_dnd.items():
                fn_supported_dnd[device] = tuple(_expand_typesets(dtypes))

        # dict intersection
        supported = _dnd_dict_intersection(supported, fn_supported_dnd)

    if hasattr(fn, "unsupported_device_and_dtype"):
        fn_unsupported_dnd = fn.unsupported_device_and_dtype.__get__()

        if "einops" in fn.__name__ and isinstance(fn_unsupported_dnd, dict):
            fn_unsupported_dnd = fn_unsupported_dnd.get(backend, supported)

        if fn_unsupported_dnd:
            aikit.utils.assertions.check_isinstance(
                list(fn_unsupported_dnd.values())[0], tuple
            )

        if isinstance(fn_unsupported_dnd, dict):
            for device, dtypes in fn_unsupported_dnd.items():
                fn_unsupported_dnd[device] = tuple(_expand_typesets(dtypes))

        # dict difference
        supported = _dnd_dict_difference(supported, fn_unsupported_dnd)

    if complement:
        # dict difference
        all_comb = _all_dnd_combinations()
        supported = _dnd_dict_difference(all_comb, supported)
    return supported


@handle_exceptions
@handle_nestable
def function_supported_devices_and_dtypes(fn: Callable, recurse: bool = True) -> Dict:
    """Return the supported combination of devices and dtypes of the current
    backend's function. The function returns a dict containing the supported
    combination of devices and dtypes of the primary and compositional
    implementations in case of partial mixed functions.

    Parameters
    ----------
    fn
        The function to check for the supported device and dtype attribute
    recurse
        Whether to recurse into used aikit functions.
        Default is ``True``.

    Returns
    -------
    ret
        Tuple or dict containing the supported devices and dtypes of the function
    """
    aikit.utils.assertions.check_true(
        _is_valid_device_and_dtypes_attributes(fn),
        "supported_device_and_dtypes and unsupported_device_and_dtypes "
        "attributes cannot both exist in a particular backend",
    )

    if hasattr(fn, "partial_mixed_handler"):
        return {
            "compositional": function_supported_devices_and_dtypes(
                fn.compos, recurse=recurse
            ),
            "primary": _get_devices_and_dtypes(fn, complement=False),
        }
    else:
        supported_devices_dtypes = _get_devices_and_dtypes(fn, complement=False)
        if recurse:
            supported_devices_dtypes = aikit.functional.data_type._nested_get(
                fn,
                supported_devices_dtypes,
                _dnd_dict_intersection,
                function_supported_devices_and_dtypes,
                wrapper=lambda x: x,
            )

    return supported_devices_dtypes


@handle_exceptions
@handle_nestable
def function_unsupported_devices_and_dtypes(fn: Callable, recurse: bool = True) -> Dict:
    """Return the unsupported combination of devices and dtypes of the current
    backend's function. The function returns a dict containing the unsupported
    combination of devices and dtypes of the primary and compositional
    implementations in case of partial mixed functions.

    Parameters
    ----------
    fn
        The function to check for the unsupported device and dtype attribute
    recurse
        Whether to recurse into used aikit functions.
        Default is ``True``.

    Returns
    -------
    ret
        Tuple or dict containing the unsupported devices and dtypes of the function
    """
    aikit.utils.assertions.check_true(
        _is_valid_device_and_dtypes_attributes(fn),
        "supported_device_and_dtypes and unsupported_device_and_dtypes "
        "attributes cannot both exist in a particular backend",
    )
    if hasattr(fn, "partial_mixed_handler"):
        return {
            "compositional": function_unsupported_devices_and_dtypes(
                fn.compos, recurse=recurse
            ),
            "primary": _get_devices_and_dtypes(fn, complement=True),
        }
    else:
        unsupported_devices_dtypes = _get_devices_and_dtypes(fn, complement=True)
        if recurse:
            unsupported_devices_dtypes = aikit.functional.data_type._nested_get(
                fn,
                unsupported_devices_dtypes,
                _dnd_dict_union,
                function_unsupported_devices_and_dtypes,
                wrapper=lambda x: x,
            )
    return unsupported_devices_dtypes


@handle_exceptions
def vmap(
    func: Callable,
    in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
    out_axes: int = 0,
) -> Callable:
    """Vectorizing map. Creates a function which maps func over argument axes.

    Parameters
    ----------
    func
        Function to be mapped over additional axes.
    in_axes
       An integer, None, or (nested) standard Python container
       (tuple/list) thereof specifying which input array
       axes to map over.If each positional argument to fun
       is an array, then in_axes can be an integer, a None,
       or a tuple of integers and Nones with length equal
       to the number of positional arguments to fun. An
       integer or None indicates which array axis to map
       over for all arguments (with None indicating not to map any axis),
       and a tuple indicates which axis to map for each
       corresponding positional argument. Axis integers must
       be in the range [-ndim, ndim) for each array,
       where ndim is the number of dimensions (axes) of the
       corresponding input array.
    out_axes
        An integer indicating where the mapped axis should appear in the output.

    Returns
    -------
    ret
        Batched/vectorized version of func with arguments
        that correspond to those of func, but with extra
        array axes at positions indicated by in_axes,
        and a return value that corresponds
        to that of fun, but with extra array axes
        at positions indicated by out_axes.


    This docstring is a summarised version of the `docstring
    <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax-vmap>`_
    for vmap from JAX documentation.

    Examples
    --------
    With :func:`aikit.matmul` and :class:`aikit.Array` input:

    >>> x = aikit.array(aikit.arange(60).reshape((3, 5, 4)))
    >>> y = aikit.array(aikit.arange(40).reshape((5, 4, 2)))
    >>> z = aikit.vmap(aikit.matmul, (1, 0), 1)(x, y)
    >>> print(z.shape)
    (3, 5, 2)
    """
    # TODO: optimize in the numpy and tensorflow backends and extend functionality
    return current_backend().vmap(func, in_axes, out_axes)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@to_native_arrays_and_back
@handle_device
def isin(
    elements: Union[aikit.Array, aikit.NativeArray],
    test_elements: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    assume_unique: bool = False,
    invert: bool = False,
) -> aikit.Array:
    """Test if each element of elements is in test_elements.

    Parameters
    ----------
    elements
        input array
    test_elements
        values against which to test for each input element
    assume_unique
        If True, assumes both elements and test_elements contain unique elements,
        which can speed up the calculation. Default value is False.
    invert
        If True, inverts the boolean return array, resulting in True values for
        elements not in test_elements. Default value is False.

    Returns
    -------
    ret
        output a boolean array of the same shape as elements that is True for elements
        in test_elements and False otherwise.

    Examples
    --------
    >>> x = aikit.array([[10, 7, 4], [3, 2, 1]])
    >>> y = aikit.array([1, 2, 3])
    >>> aikit.isin(x, y)
    aikit.array([[False, False, False], [ True,  True,  True]])

    >>> x = aikit.array([3, 2, 1, 0])
    >>> y = aikit.array([1, 2, 3])
    >>> aikit.isin(x, y, invert=True)
    aikit.array([False, False, False,  True])
    """
    return aikit.current_backend(elements, test_elements).isin(
        elements, test_elements, assume_unique=assume_unique, invert=invert
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@inputs_to_native_arrays
@handle_device
def itemsize(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
) -> int:
    """Return the size of the input array's elements.

    Parameters
    ----------
    x
       The input array.

    Returns
    -------
    ret
        An integer specifying the element size in bytes.

    Examples
    --------
    >>> x = aikit.array([1,2,3], dtype=aikit.float64)
    >>> aikit.itemsize(x)
    8

    >>> x = aikit.array([1,2,3], dtype=aikit.complex128)
    >>> aikit.itemsize(x)
    16
    """
    return aikit.current_backend(x).itemsize(x)


@handle_exceptions
@handle_nestable
@handle_device
def strides(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
) -> Tuple[int]:
    """Return the input array's strides across each dimension.

    Parameters
    ----------
    x
       The input array.

    Returns
    -------
    ret
        A tuple containing the strides.

    Examples
    --------
    >>> x = aikit.array([[1, 5, 9], [2, 6, 10]])
    >>> aikit.strides(x)
    (4, 8)
    """
    if aikit.is_native_array(x) or (aikit.is_aikit_array(x) and x.base is None):
        return aikit.to_numpy(x).strides
    # if x is an aikit array with a base,
    # convert it to a numpy array with the same base:
    ret = aikit.to_numpy(x.base)
    aikit_numpy = aikit.with_backend("numpy")
    for fn, args, kwargs, index in x._manipulation_stack:
        ret = aikit_numpy.__dict__[fn](ret, *args, **kwargs)
        ret = ret[index] if aikit.exists(index) else ret
    return ret.to_native().strides


def is_aikit_nested_array(x: Any, /) -> bool:
    """Determine whether the input x is an Aikit Nested Array.

    Parameters
    ----------
    x
        The input to check

    Returns
    -------
    ret
        Boolean, whether or not x is an aikit nested array.
    """
    return isinstance(x, aikit.NestedArray)
