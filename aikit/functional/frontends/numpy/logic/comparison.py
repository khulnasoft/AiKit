# global
import aikit

# local
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    inputs_to_aikit_arrays,
    handle_numpy_casting,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)


# --- Helpers --- #
# --------------- #


@handle_numpy_out
@to_aikit_arrays_and_back
@handle_numpy_dtype
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _equal(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    ret = aikit.equal(x1, x2, out=out)
    if aikit.is_array(where):
        where = aikit.asarray(where, dtype=aikit.bool)
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@to_aikit_arrays_and_back
@handle_numpy_dtype
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _greater(
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
    ret = aikit.greater(x1, x2, out=out)
    if aikit.is_array(where):
        where = aikit.asarray(where, dtype=aikit.bool)
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@to_aikit_arrays_and_back
@handle_numpy_dtype
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _greater_equal(
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
    ret = aikit.greater_equal(x1, x2, out=out)
    if aikit.is_array(where):
        where = aikit.asarray(where, dtype=aikit.bool)
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@to_aikit_arrays_and_back
@handle_numpy_dtype
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _less(
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
    ret = aikit.less(x1, x2, out=out)
    if aikit.is_array(where):
        where = aikit.asarray(where, dtype=aikit.bool)
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@to_aikit_arrays_and_back
@handle_numpy_dtype
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _less_equal(
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
    ret = aikit.less_equal(x1, x2, out=out)
    if aikit.is_array(where):
        where = aikit.asarray(where, dtype=aikit.bool)
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@to_aikit_arrays_and_back
@handle_numpy_dtype
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _not_equal(
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
    ret = aikit.not_equal(x1, x2, out=out)
    if aikit.is_array(where):
        where = aikit.asarray(where, dtype=aikit.bool)
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


# --- Main --- #
# ------------ #


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def array_equal(a1, a2, equal_nan=False):
    if not equal_nan:
        return aikit.array(aikit.array_equal(a1, a2))
    a1nan, a2nan = aikit.isnan(a1), aikit.isnan(a2)

    if not (a1nan == a2nan).all():
        return False
    return aikit.array(aikit.array_equal(a1 * ~a1nan, a2 * ~a2nan))


@inputs_to_aikit_arrays
@from_zero_dim_arrays_to_scalar
def array_equiv(a1, a2):
    if len(aikit.shape(a1)) < len(aikit.shape(a2)):
        a1 = aikit.broadcast_to(a1, aikit.shape(a2))
    else:
        a2 = aikit.broadcast_to(a2, aikit.shape(a1))
    return aikit.array_equal(a1, a2)
