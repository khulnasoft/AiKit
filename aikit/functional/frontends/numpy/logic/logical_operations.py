# global
import aikit
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_numpy_casting,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)


# --- Helpers --- #
# --------------- #


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _logical_and(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    ret = aikit.logical_and(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _logical_not(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    ret = aikit.logical_not(x, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _logical_or(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    ret = aikit.logical_or(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _logical_xor(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    ret = aikit.logical_xor(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret
