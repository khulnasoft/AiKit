# global
from typing import Optional

# local
import aikit
from aikit import handle_out_argument, handle_nestable
from aikit.utils.exceptions import handle_exceptions


@handle_out_argument
@handle_nestable
@handle_exceptions
def optional_get_element(
    x: Optional[aikit.Array] = None,
    /,
    *,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """If the input is a tensor or sequence type, it returns the input. If the
    input is an optional type, it outputs the element in the input. It is an
    error if the input is an empty optional-type (i.e. does not have an
    element) and the behavior is undefined in this case.

    Parameters
    ----------
    x
        Input array
    out
        Optional output array, for writing the result to.

    Returns
    -------
    ret
        Input array if it is not None
    """
    if x is None:
        raise aikit.utils.exceptions.AikitError(
            "The requested optional input has no value."
        )
    return x
