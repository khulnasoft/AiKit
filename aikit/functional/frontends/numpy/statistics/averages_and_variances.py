# global
import aikit
from aikit.func_wrapper import with_unsupported_dtypes
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def average(a, /, *, axis=None, weights=None, returned=False, keepdims=False):
    axis = tuple(axis) if isinstance(axis, list) else axis
    avg = 0

    if keepdims is None:
        keepdims_kw = {}
    else:
        keepdims_kw = {"keepdims": keepdims}

    dtype = a.dtype
    if weights is None:
        avg = a.mean(axis, **keepdims_kw)
        weights_sum = avg.dtype.type(a.count(axis))
    else:
        if a.shape != weights.shape:
            if axis is None:
                return 0
            weights = aikit.broadcast_to(weights, (a.ndim - 1) * (1,) + weights.shape)
            weights = weights.swapaxes(-1, axis)
        weights_sum = weights.sum(axis=axis, **keepdims_kw)
        mul = aikit.multiply(a, weights)
        avg = aikit.sum(mul, axis=axis, **keepdims_kw) / weights_sum

    if returned:
        if weights_sum.shape != avg.shape:
            weights_sum = aikit.broadcast_to(weights_sum, avg.shape).copy()
        return avg.astype(dtype), weights_sum
    else:
        return avg.astype(dtype)


@to_aikit_arrays_and_back
def cov(
    m,
    y=None,
    /,
    *,
    rowvar=True,
    bias=False,
    ddof=None,
    fweights=None,
    aweights=None,
    dtype=None,
):
    return aikit.cov(
        m,
        y,
        rowVar=rowvar,
        bias=bias,
        ddof=ddof,
        fweights=fweights,
        aweights=aweights,
        dtype=dtype,
    )


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    axis = tuple(axis) if isinstance(axis, list) else axis
    dtype = dtype or a.dtype if not aikit.is_int_dtype(a.dtype) else aikit.float64
    where = aikit.where(where, aikit.ones_like(a), 0)
    if where is not True:
        a = aikit.where(where, a, 0.0)
        sum = aikit.sum(a, axis=axis, keepdims=keepdims, dtype=dtype)
        cnt = aikit.sum(where, axis=axis, keepdims=keepdims, dtype=int)
        ret = aikit.divide(sum, cnt, out=out)
    else:
        ret = aikit.mean(a.astype(dtype), axis=axis, keepdims=keepdims, out=out)

    return ret.astype(dtype)


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanmean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    where = ~aikit.isnan(a) & where
    ret = mean(a, axis, dtype, keepdims=keepdims, where=where).aikit_array
    if out is not None:
        out.data = ret.data
    return ret


# nanmedian
@handle_numpy_out
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanmedian(
    a,
    /,
    *,
    axis=None,
    keepdims=False,
    out=None,
    overwrite_input=False,
):
    ret = aikit.nanmedian(
        a, axis=axis, keepdims=keepdims, out=out, overwrite_input=overwrite_input
    )
    return ret


@handle_numpy_out
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanstd(
    a, /, *, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True
):
    a = aikit.nan_to_num(a)
    axis = tuple(axis) if isinstance(axis, list) else axis

    if dtype:
        a = aikit.astype(aikit.array(a), aikit.as_aikit_dtype(dtype))

    ret = aikit.std(a, axis=axis, correction=ddof, keepdims=keepdims, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)

    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.25.0 and below": ("float16", "bfloat16")}, "tensorflow")
def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True):
    is_nan = aikit.isnan(a)
    axis = tuple(axis) if isinstance(axis, list) else axis

    if aikit.any(is_nan):
        a = [i for i in a if aikit.isnan(i) is False]

    if dtype is None:
        dtype = "float" if aikit.is_int_dtype(a) else a.dtype

    a = aikit.astype(aikit.array(a), aikit.as_aikit_dtype(dtype))
    ret = aikit.var(a, axis=axis, correction=ddof, keepdims=keepdims, out=out)

    if aikit.is_array(where):
        where = aikit.array(where, dtype=aikit.bool)
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)

    if aikit.all(aikit.isnan(ret)):
        ret = aikit.astype(ret, aikit.array([float("inf")]))

    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def std(
    x,
    /,
    *,
    axis=None,
    ddof=0.0,
    keepdims=False,
    out=None,
    dtype=None,
    where=True,
):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if dtype is None:
        if aikit.is_int_dtype(x.dtype):
            dtype = aikit.float64
        else:
            dtype = x.dtype
    ret = aikit.std(x, axis=axis, correction=ddof, keepdims=keepdims, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret.astype(dtype, copy=False)


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def var(x, /, *, axis=None, ddof=0.0, keepdims=False, out=None, dtype=None, where=True):
    axis = tuple(axis) if isinstance(axis, list) else axis
    dtype = (
        dtype
        if dtype is not None
        else aikit.float64
        if aikit.is_int_dtype(x.dtype)
        else x.dtype
    )
    ret = aikit.var(x, axis=axis, correction=ddof, keepdims=keepdims, out=out)
    ret = (
        aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
        if aikit.is_array(where)
        else ret
    )
    return ret.astype(dtype, copy=False)
