# global
import functools
from typing import Callable, Union, Sequence

# local
import aikit
from aikit import (
    inputs_to_aikit_arrays,
    handle_nestable,
    handle_array_like_without_promotion,
    handle_array_function,
)
from aikit.utils.exceptions import handle_exceptions


def _correct_aikit_callable(func):
    # get the current backend of the given aikit callable
    if aikit.nested_any(
        func,
        lambda x: hasattr(x, "__module__")
        and x.__module__.startswith("aikit")
        and not x.__module__.startswith("aikit.functional.frontends"),
    ):
        return aikit.__dict__[func.__name__]
    return func


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_aikit_arrays
@handle_array_function
def reduce(
    operand: Union[aikit.Array, aikit.NativeArray],
    init_value: Union[int, float],
    computation: Callable,
    /,
    *,
    axes: Union[int, Sequence[int]] = 0,
    keepdims: bool = False,
) -> aikit.Array:
    """Reduces the input array's dimensions by applying a function along one or
    more axes.

    Parameters
    ----------
    operand
        The array to act on.
    init_value
        The value with which to start the reduction.
    computation
        The reduction function.
    axes
        The dimensions along which the reduction is performed.
    keepdims
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one.

    Returns
    -------
    ret
        The reduced array.

    Examples
    --------
    >>> x = aikit.array([[1, 2, 3], [4, 5, 6]])
    >>> aikit.reduce(x, 0, aikit.add, 0)
    aikit.array([6, 15])

    >>> x = aikit.array([[1, 2, 3], [4, 5, 6]])
    >>> aikit.reduce(x, 0, aikit.add, 1)
    aikit.array([5, 7, 9])
    """
    axes = (axes,) if isinstance(axes, int) else axes
    axes = [a + operand.ndim if a < 0 else a for a in axes]
    axes = sorted(axes, reverse=True)
    init_value = aikit.array(init_value)
    op_dtype = operand.dtype
    computation = _correct_aikit_callable(computation)
    for axis in axes:
        temp = aikit.moveaxis(operand, axis, 0).reshape((operand.shape[axis], -1))
        temp = functools.reduce(computation, temp, init_value)
        operand = aikit.reshape(temp, operand.shape[:axis] + operand.shape[axis + 1 :])
    if keepdims:
        operand = aikit.expand_dims(operand, axis=axes)
    return operand.astype(op_dtype)


reduce.mixed_backend_wrappers = {
    "to_add": (
        "handle_backend_invalid",
        "inputs_to_native_arrays",
        "outputs_to_aikit_arrays",
        "handle_device",
    ),
    "to_skip": ("inputs_to_aikit_arrays",),
}
