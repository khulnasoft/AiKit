# local

import aikit

from aikit.functional.frontends.numpy import promote_types_of_numpy_inputs

from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)


# --- Helpers --- #
# --------------- #


# nanargmin and nanargmax composition helper
def _nanargminmax(a, axis=None):
    # check nans
    nans = aikit.isnan(a).astype(aikit.bool)
    # replace nans with inf
    a = aikit.where(nans, aikit.inf, a)
    if nans is not None:
        nans = aikit.all(nans, axis=axis)
        if aikit.any(nans):
            raise aikit.utils.exceptions.IvyError("All-NaN slice encountered")
    return a


# --- Main --- #
# ------------ #


@handle_numpy_out
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def argmax(
    a,
    /,
    *,
    axis=None,
    out=None,
    keepdims=False,
):
    return aikit.argmax(a, axis=axis, out=out, keepdims=keepdims)


@handle_numpy_out
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def argmin(a, /, *, axis=None, keepdims=False, out=None):
    return aikit.argmin(a, axis=axis, out=out, keepdims=keepdims)


@to_aikit_arrays_and_back
def argwhere(a):
    return aikit.argwhere(a)


@to_aikit_arrays_and_back
def extract(cond, arr, /):
    if cond.dtype == "bool":
        return arr[cond]
    else:
        return arr[cond != 0]


@to_aikit_arrays_and_back
def flatnonzero(a):
    return aikit.nonzero(aikit.reshape(a, (-1,)))


@handle_numpy_out
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanargmax(a, /, *, axis=None, out=None, keepdims=False):
    a = _nanargminmax(a, axis=axis)
    return aikit.argmax(a, axis=axis, keepdims=keepdims, out=out)


@handle_numpy_out
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanargmin(a, /, *, axis=None, out=None, keepdims=False):
    a = _nanargminmax(a, axis=axis)
    return aikit.argmin(a, axis=axis, keepdims=keepdims, out=out)


@to_aikit_arrays_and_back
def nonzero(a):
    return aikit.nonzero(a)


@to_aikit_arrays_and_back
def searchsorted(a, v, side="left", sorter=None):
    return aikit.searchsorted(a, v, side=side, sorter=sorter)


@to_aikit_arrays_and_back
def where(cond, x1=None, x2=None, /):
    if x1 is None and x2 is None:
        # numpy where behaves as np.asarray(condition).nonzero() when x and y
        # not included
        return aikit.asarray(cond).nonzero()
    elif x1 is not None and x2 is not None:
        x1, x2 = promote_types_of_numpy_inputs(x1, x2)
        return aikit.where(cond, x1, x2)
    else:
        raise aikit.utils.exceptions.IvyException("where takes either 1 or 3 arguments")
