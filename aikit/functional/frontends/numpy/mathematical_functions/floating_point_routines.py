# global
import aikit

# local
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_numpy_out,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_casting,
)


# --- Helpers --- #
# --------------- #


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _nextafter(
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
    return aikit.nextafter(x1, x2, out=out)


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _signbit(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="safe",
    order="K",
    dtype=None,
    subok=True,
):
    x = aikit.astype(x, aikit.float64)
    return aikit.logical_or(aikit.less(x, 0), aikit.atan2(0.0, x) == aikit.pi, out=out)


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _spacing(
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
    # Implement the frontend function using Ivy compositions
    if dtype is None:
        dtype = aikit.dtype(x)
    y = aikit.floor(aikit.log2(aikit.abs(x + 1)))
    spacing = aikit.multiply(aikit.finfo(dtype).eps, aikit.pow(2, y))
    if dtype != "float16":
        spacing = aikit.sign(x) * spacing
    return spacing
