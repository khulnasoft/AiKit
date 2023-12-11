# global
import aikit

# local
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)
import aikit.functional.frontends.numpy as np_frontend


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
def cumprod(a, /, axis=None, dtype=None, out=None):
    return aikit.cumprod(a, axis=axis, dtype=dtype, out=out)


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
def cumsum(a, /, axis=None, dtype=None, out=None):
    return aikit.cumsum(a, axis=axis, dtype=dtype, out=out)


@to_aikit_arrays_and_back
def diff(x, /, *, n=1, axis=-1, prepend=None, append=None):
    return aikit.diff(x, n=n, axis=axis, prepend=prepend, append=append)


@to_aikit_arrays_and_back
def ediff1d(ary, to_end=None, to_begin=None):
    diffs = aikit.diff(ary)
    if to_begin is not None:
        if not isinstance(to_begin, (list, tuple)):
            to_begin = [to_begin]
        to_begin = aikit.array(to_begin)
        diffs = aikit.concat((to_begin, diffs))
    if to_end is not None:
        if not isinstance(to_end, (list, tuple)):
            to_end = [to_end]
        to_end = aikit.array(to_end)
        diffs = aikit.concat((diffs, to_end))
    return diffs


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
def nancumprod(a, /, axis=None, dtype=None, out=None):
    a = aikit.where(aikit.isnan(a), aikit.ones_like(a), a)
    return aikit.cumprod(a, axis=axis, dtype=dtype, out=out)


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
def nancumsum(a, /, axis=None, dtype=None, out=None):
    a = aikit.where(aikit.isnan(a), aikit.zeros_like(a), a)
    return aikit.cumsum(a, axis=axis, dtype=dtype, out=out)


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanprod(
    a, /, *, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None
):
    fill_values = aikit.ones_like(a)
    a = aikit.where(aikit.isnan(a), fill_values, a)
    if aikit.is_array(where):
        a = aikit.where(where, a, aikit.default(out, fill_values), out=out)
    if initial is not None:
        a[axis] = 1
        s = aikit.shape(a, as_array=False)
        header = aikit.full(s, initial)
        a = aikit.concat([header, a], axis=axis)
    return aikit.prod(a, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nansum(
    a, /, *, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None
):
    fill_values = aikit.zeros_like(a)
    a = aikit.where(aikit.isnan(a), fill_values, a)
    if aikit.is_array(where):
        a = aikit.where(where, a, aikit.default(out, fill_values), out=out)
    if initial is not None:
        a[axis] = 1
        s = aikit.shape(a, as_array=False)
        header = aikit.full(s, initial)
        a = aikit.concat([header, a], axis=axis)
    return aikit.sum(a, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def prod(
    x,
    /,
    *,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    if where is not True:
        x = aikit.where(where, x, aikit.default(out, aikit.ones_like(x)), out=out)
    if initial is not None:
        initial = np_frontend.array(initial, dtype=dtype).tolist()
        if axis is not None:
            s = aikit.to_list(aikit.shape(x, as_array=True))
            s[axis] = 1
            header = aikit.full(aikit.Shape(tuple(s)), initial)
            x = aikit.concat([header, x], axis=axis)
        else:
            x[0] *= initial
    return aikit.prod(x, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def sum(
    x,
    /,
    *,
    axis=None,
    dtype=None,
    keepdims=False,
    out=None,
    initial=None,
    where=True,
):
    if aikit.is_array(where):
        x = aikit.where(where, x, aikit.default(out, aikit.zeros_like(x)), out=out)
    if initial is not None:
        s = aikit.to_list(aikit.shape(x, as_array=True))
        s[axis] = 1
        header = aikit.full(aikit.Shape(tuple(s)), initial)
        if aikit.is_array(where):
            x = aikit.where(where, x, aikit.default(out, aikit.zeros_like(x)), out=out)
        x = aikit.concat([header, x], axis=axis)
    else:
        x = aikit.where(aikit.isnan(x), aikit.zeros_like(x), x)
    return aikit.sum(x, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@to_aikit_arrays_and_back
def trapz(y, x=None, dx=1.0, axis=-1):
    return aikit.trapz(y, x=x, dx=dx, axis=axis)
