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
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def _fmax(
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
    ret = aikit.fmax(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _fmin(
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
    ret = aikit.fmin(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _maximum(
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
    ret = aikit.maximum(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _minimum(
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
    ret = aikit.minimum(x1, x2, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


# --- Main --- #
# ------------ #


@handle_numpy_out
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def amax(
    a,
    /,
    *,
    axis=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    out_dtype = aikit.dtype(a)
    where_mask = None
    if initial is not None:
        if aikit.is_array(where):
            a = aikit.where(where, a, a.full_like(initial))
            where_mask = aikit.all(aikit.logical_not(where), axis=axis, keepdims=keepdims)
        s = aikit.shape(a, as_array=True)
        if axis is not None:
            if isinstance(axis, (tuple, list)) or aikit.is_array(axis):
                # introducing the initial in one dimension is enough
                ax = axis[0] % len(s)
                s[ax] = 1
            else:
                ax = axis % len(s)
                s[ax] = 1
        header = aikit.full(aikit.Shape(s.to_list()), initial, dtype=aikit.dtype(a))
        if axis:
            if isinstance(axis, (tuple, list)) or aikit.is_array(axis):
                a = aikit.concat([a, header], axis=axis[0])
            else:
                a = aikit.concat([a, header], axis=axis)
        else:
            a = aikit.concat([a, header], axis=0)
    res = aikit.max(a, axis=axis, keepdims=keepdims, out=out)
    if where_mask is not None and aikit.any(where_mask):
        res = aikit.where(aikit.logical_not(where_mask), res, initial, out=out)
    return aikit.astype(res, out_dtype, out=out, copy=False)


@handle_numpy_out
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def amin(
    a,
    /,
    *,
    axis=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    out_dtype = aikit.dtype(a)
    where_mask = None
    if initial is not None:
        if aikit.is_array(where):
            a = aikit.where(where, a, a.full_like(initial))
            where_mask = aikit.all(aikit.logical_not(where), axis=axis, keepdims=keepdims)
        s = aikit.shape(a, as_array=True)
        if axis is not None:
            if isinstance(axis, (tuple, list)) or aikit.is_array(axis):
                # introducing the initial in one dimension is enough
                ax = axis[0] % len(s)
                s[ax] = 1
            else:
                ax = axis % len(s)
                s[ax] = 1
        header = aikit.full(aikit.Shape(s.to_list()), initial, dtype=aikit.dtype(a))
        if axis:
            if isinstance(axis, (tuple, list)) or aikit.is_array(axis):
                a = aikit.concat([a, header], axis=axis[0])
            else:
                a = aikit.concat([a, header], axis=axis)
        else:
            a = aikit.concat([a, header], axis=0)
    res = aikit.min(a, axis=axis, keepdims=keepdims, out=out)
    if where_mask is not None and aikit.any(where_mask):
        res = aikit.where(aikit.logical_not(where_mask), res, initial, out=out)
    return aikit.astype(res, out_dtype, out=out, copy=False)


@handle_numpy_out
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def max(
    a,
    /,
    *,
    axis=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    return amax(a, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)


@handle_numpy_out
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def min(
    a,
    /,
    *,
    axis=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    return amin(a, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)


@handle_numpy_out
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanmax(
    a,
    axis=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    out_dtype = aikit.dtype(a)
    nan_mask = aikit.isnan(a)
    a = aikit.where(aikit.logical_not(nan_mask), a, a.full_like(-aikit.inf))
    where_mask = None
    if initial is not None:
        if aikit.is_array(where):
            a = aikit.where(where, a, a.full_like(initial))
            where_mask = aikit.all(aikit.logical_not(where), axis=axis, keepdims=keepdims)
        s = aikit.shape(a, as_array=True)
        if axis is not None:
            if isinstance(axis, (tuple, list)) or aikit.is_array(axis):
                # introducing the initial in one dimension is enough
                ax = axis[0] % len(s)
                s[ax] = 1
            else:
                ax = axis % len(s)
                s[ax] = 1
        header = aikit.full(aikit.Shape(s.to_list()), initial, dtype=aikit.dtype(a))
        if axis:
            if isinstance(axis, (tuple, list)) or aikit.is_array(axis):
                a = aikit.concat([a, header], axis=axis[0])
            else:
                a = aikit.concat([a, header], axis=axis)
        else:
            a = aikit.concat([a, header], axis=0)
    res = aikit.max(a, axis=axis, keepdims=keepdims, out=out)
    if nan_mask is not None:
        nan_mask = aikit.all(nan_mask, axis=axis, keepdims=keepdims, out=out)
        if aikit.any(nan_mask):
            res = aikit.where(
                aikit.logical_not(nan_mask),
                res,
                initial if initial is not None else aikit.nan,
                out=out,
            )
    if where_mask is not None and aikit.any(where_mask):
        res = aikit.where(aikit.logical_not(where_mask), res, aikit.nan, out=out)
    return aikit.astype(res, out_dtype, out=out, copy=False)


@handle_numpy_out
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanmin(
    a,
    axis=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    out_dtype = aikit.dtype(a)
    nan_mask = aikit.isnan(a)
    a = aikit.where(aikit.logical_not(nan_mask), a, a.full_like(+aikit.inf))
    where_mask = None
    if initial is not None:
        if aikit.is_array(where):
            a = aikit.where(where, a, a.full_like(initial))
            where_mask = aikit.all(aikit.logical_not(where), axis=axis, keepdims=keepdims)
        s = aikit.shape(a, as_array=True)
        if axis is not None:
            if isinstance(axis, (tuple, list)) or aikit.is_array(axis):
                # introducing the initial in one dimension is enough
                ax = axis[0] % len(s)
                s[ax] = 1
            else:
                ax = axis % len(s)
                s[ax] = 1
        header = aikit.full(aikit.Shape(s.to_list()), initial, dtype=aikit.dtype(a))
        if axis:
            if isinstance(axis, (tuple, list)) or aikit.is_array(axis):
                a = aikit.concat([a, header], axis=axis[0])
            else:
                a = aikit.concat([a, header], axis=axis)
        else:
            a = aikit.concat([a, header], axis=0)
    res = aikit.min(a, axis=axis, keepdims=keepdims, out=out)
    if nan_mask is not None:
        nan_mask = aikit.all(nan_mask, axis=axis, keepdims=keepdims, out=out)
        if aikit.any(nan_mask):
            res = aikit.where(
                aikit.logical_not(nan_mask),
                res,
                initial if initial is not None else aikit.nan,
                out=out,
            )
    if where_mask is not None and aikit.any(where_mask):
        res = aikit.where(aikit.logical_not(where_mask), res, aikit.nan, out=out)
    return aikit.astype(res, out_dtype, out=out, copy=False)
