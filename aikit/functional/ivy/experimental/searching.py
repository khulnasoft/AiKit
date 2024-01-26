from typing import Optional, Union, Tuple
import aikit
from aikit.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
    handle_device,
    handle_backend_invalid,
)
from aikit.utils.exceptions import handle_exceptions


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_device
def unravel_index(
    indices: Union[aikit.Array, aikit.NativeArray],
    shape: Tuple[int],
    /,
    *,
    out: Optional[aikit.Array] = None,
) -> Tuple[aikit.Array]:
    """Convert a flat index or array of flat indices into a tuple of coordinate
    arrays.

    Parameters
    ----------
    indices
        Input array.
    shape
        The shape of the array to use for unraveling indices.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Tuple with arrays of type int32 that have the same shape as the indices array.

    Examples
    --------
    >>> indices = aikit.array([22, 41, 37])
    >>> aikit.unravel_index(indices, (7,6))
    (aikit.array([3, 6, 6]), aikit.array([4, 5, 1]))
    """
    return aikit.current_backend(indices).unravel_index(indices, shape, out=out)
