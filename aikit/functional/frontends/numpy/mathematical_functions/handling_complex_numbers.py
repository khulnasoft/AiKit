# global
import aikit
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_numpy_out,
    handle_numpy_dtype,
    handle_numpy_casting,
    from_zero_dim_arrays_to_scalar,
)


# --- Helpers --- #
# --------------- #


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _conj(
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
    ret = aikit.conj(x, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


# --- Main --- #
# ------------ #


@to_aikit_arrays_and_back
def angle(z, deg=False):
    angle = aikit.angle(z, deg=deg)
    if deg and len(z.shape) == 0:
        angle = aikit.astype(angle, aikit.float64)
    return angle


@to_aikit_arrays_and_back
def imag(val):
    return aikit.imag(val)


@to_aikit_arrays_and_back
def real(val):
    return aikit.real(val)
