# global
import aikit

from aikit.functional.frontends.numpy import promote_types_of_numpy_inputs

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
def _add(
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
    x1, x2 = promote_types_of_numpy_inputs(x1, x2)
    ret = aikit.add(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _divide(
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
    x1, x2 = promote_types_of_numpy_inputs(x1, x2)
    ret = aikit.divide(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _divmod(
    x1,
    x2,
    /,
    out1_2=(None, None),
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    if dtype:
        x1 = aikit.astype(aikit.array(x1), aikit.as_aikit_dtype(dtype))
        x2 = aikit.astype(aikit.array(x2), aikit.as_aikit_dtype(dtype))

    ret = [aikit.floor_divide(x1, x2, out=out), aikit.remainder(x1, x2, out=out)]
    if aikit.is_array(where):
        ret = aikit.where(
            where,
            ret,
            (
                [
                    aikit.default(out, aikit.zeros_like(ret[0])),
                    aikit.default(out, aikit.zeros_like(ret[1])),
                ]
            ),
            out=out,
        )
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _float_power(
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
    x1, x2 = promote_types_of_numpy_inputs(x1, x2)
    ret = aikit.float_power(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _floor_divide(
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
    if dtype:
        x1 = aikit.astype(aikit.array(x1), aikit.as_aikit_dtype(dtype))
        x2 = aikit.astype(aikit.array(x2), aikit.as_aikit_dtype(dtype))
    ret = aikit.floor_divide(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _fmod(
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
    if dtype:
        x1 = aikit.astype(aikit.array(x1), aikit.as_aikit_dtype(dtype))
        x2 = aikit.astype(aikit.array(x2), aikit.as_aikit_dtype(dtype))
    ret = aikit.fmod(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _mod(
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
    if dtype:
        x1 = aikit.astype(aikit.array(x1), aikit.as_aikit_dtype(dtype))
        x2 = aikit.astype(aikit.array(x2), aikit.as_aikit_dtype(dtype))
    ret = aikit.remainder(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _modf(
    x,
    /,
    out1_2=(None, None),
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    if dtype:
        x = aikit.astype(aikit.array(x), aikit.as_aikit_dtype(dtype))

    integral_part = aikit.floor(x)
    fractional_part = x - integral_part

    if aikit.is_array(where):
        integral_part = aikit.where(
            where,
            integral_part,
            aikit.default(out, aikit.zeros_like(integral_part)),
            out=out,
        )
        fractional_part = aikit.where(
            where,
            fractional_part,
            aikit.default(out, aikit.zeros_like(fractional_part)),
            out=out,
        )

    return fractional_part, integral_part


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _multiply(
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
    x1, x2 = promote_types_of_numpy_inputs(x1, x2)
    ret = aikit.multiply(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _negative(
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
    ret = aikit.negative(x, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _positive(
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
    ret = aikit.positive(x, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _power(
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
    x1, x2 = promote_types_of_numpy_inputs(x1, x2)
    ret = aikit.pow(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _reciprocal(
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
    if dtype is None:
        dtype = aikit.as_aikit_dtype(x.dtype)
    ret = aikit.reciprocal(x, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret.astype(dtype)


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _remainder(
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
    if dtype:
        x1 = aikit.astype(aikit.array(x1), aikit.as_aikit_dtype(dtype))
        x2 = aikit.astype(aikit.array(x2), aikit.as_aikit_dtype(dtype))
    ret = aikit.remainder(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _subtract(
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
    x1, x2 = promote_types_of_numpy_inputs(x1, x2)
    ret = aikit.subtract(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


# --- Main --- #
# ------------ #


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def vdot(
    a,
    b,
    /,
):
    a, b = promote_types_of_numpy_inputs(a, b)
    return aikit.multiply(a, b).sum()
