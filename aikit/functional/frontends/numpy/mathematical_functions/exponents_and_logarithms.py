# global
import aikit

# local
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
def _exp(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    ret = aikit.exp(x, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _exp2(
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
    ret = aikit.pow(2, x, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _expm1(
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
    ret = aikit.expm1(x, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _frexp(
    x,
    /,
    out1_2=(None, None),
    out=(None, None),
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    mant, exp = aikit.frexp(x, out=out)
    if aikit.is_array(where):
        mant = aikit.where(
            where, mant, aikit.default(out[0], aikit.zeros_like(mant)), out=out[0]
        )
        exp = aikit.where(
            where, exp, aikit.default(out[1], aikit.zeros_like(exp)), out=out[1]
        )
    return mant, exp


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _ldexp(
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
    ret = aikit.ldexp(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _log(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    ret = aikit.log(x, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _log10(
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
    ret = aikit.log10(x, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _log1p(
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
    ret = aikit.log1p(x, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _log2(
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
    ret = aikit.log2(x, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _logaddexp(
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
    ret = aikit.logaddexp(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _logaddexp2(
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
    ret = aikit.logaddexp2(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


# --- Main --- #
# ------------ #


@to_aikit_arrays_and_back
def i0(x):
    return aikit.i0(x)
