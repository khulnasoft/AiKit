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
from aikit.func_wrapper import with_supported_dtypes


# --- Helpers --- #
# --------------- #


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _absolute(
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
    ret = aikit.abs(x)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _cbrt(
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
    all_positive = aikit.pow(aikit.abs(x), 1.0 / 3.0)
    ret = aikit.where(aikit.less(x, 0.0), aikit.negative(all_positive), all_positive)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _clip(
    a,
    a_min,
    a_max,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    aikit.utils.assertions.check_all_or_any_fn(
        a_min,
        a_max,
        fn=aikit.exists,
        type="any",
        limit=[1, 2],
        message="at most one of a_min and a_max can be None",
    )
    if a_min is None:
        ret = aikit.minimum(a, a_max, out=out)
    elif a_max is None:
        ret = aikit.maximum(a, a_min, out=out)
    else:
        ret = aikit.clip(a, a_min, a_max, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _copysign(
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
    ret = aikit.copysign(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _fabs(
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
    ret = aikit.abs(x)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
@with_supported_dtypes(
    {"1.26.3 and below": ("int8", "int16", "int32", "int64")}, "numpy"
)  # Add
def _gcd(
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
    ret = aikit.gcd(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _heaviside(
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
    ret = aikit.heaviside(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def _lcm(
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
    ret = aikit.lcm(x1, x2, out=out)
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
    ret = aikit.reciprocal(x)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _sign(
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
    ret = aikit.sign(x, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _sqrt(
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
    ret = aikit.sqrt(x)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _square(
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
    ret = aikit.square(x)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


# --- Main --- #
# ------------ #


@to_aikit_arrays_and_back
def convolve(a, v, mode="full"):
    if a.ndim != 1 or v.ndim != 1:
        raise ValueError("convolve() only support 1-dimensional inputs.")
    if a.shape[0] < v.shape[0]:
        a, v = v, a
    v = aikit.flip(v)

    out_order = slice(None)

    if mode == "valid":
        padding = [(0, 0)]
    elif mode == "same":
        padding = [(v.shape[0] // 2, v.shape[0] - v.shape[0] // 2 - 1)]
    elif mode == "full":
        padding = [(v.shape[0] - 1, v.shape[0] - 1)]

    result = aikit.conv_general_dilated(
        a[None, None, :],
        v[:, None, None],
        (1,),
        padding,
        dims=1,
        data_format="channel_first",
    )
    return result[0, 0, out_order]


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def interp(x, xp, fp, left=None, right=None, period=None):
    return aikit.interp(x, xp, fp, left=left, right=right, period=period)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    bounds = aikit.finfo(x.dtype)
    if posinf is None:
        posinf = bounds.max
    if neginf is None:
        neginf = bounds.min
    pos_where = aikit.isinf(x, detect_negative=False)
    neg_where = aikit.isinf(x, detect_positive=False)
    nan_where = aikit.isnan(x)
    ret = aikit.where(nan_where, nan, x)
    ret = aikit.where(pos_where, posinf, ret)
    ret = aikit.where(neg_where, neginf, ret)
    ret = ret.astype(x.dtype, copy=False)
    if not copy:
        return aikit.inplace_update(x, ret)
    return ret


@to_aikit_arrays_and_back
def real_if_close(a, tol=100):
    a = aikit.array(a, dtype=a.dtype)
    dtype_ = a.dtype

    if not aikit.is_complex_dtype(dtype_):
        return a

    if tol > 1:
        f = aikit.finfo(dtype_)
        tol = f.eps * tol

    if aikit.all(aikit.abs(aikit.imag(a)) < tol):
        a = aikit.real(a)

    return a
