# global
import aikit
import numbers
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)
import aikit.functional.frontends.numpy as np_frontend


@handle_numpy_out
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def all(
    a,
    axis=None,
    out=None,
    keepdims=False,
    *,
    where=None,
):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if where is not None:
        a = aikit.where(where, a, True)
    ret = aikit.all(a, axis=axis, keepdims=keepdims, out=out)
    return ret


@handle_numpy_out
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def any(
    a,
    axis=None,
    out=None,
    keepdims=False,
    *,
    where=None,
):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if where is not None:
        a = aikit.where(where, a, False)
    ret = aikit.any(a, axis=axis, keepdims=keepdims, out=out)
    return ret


@to_aikit_arrays_and_back
def iscomplex(x):
    return aikit.bitwise_invert(aikit.isreal(x))


@to_aikit_arrays_and_back
def iscomplexobj(x):
    if x.ndim == 0:
        return aikit.is_complex_dtype(aikit.dtype(x))
    for ele in x:
        return bool(aikit.is_complex_dtype(aikit.dtype(ele)))


@to_aikit_arrays_and_back
def isfortran(a):
    return a.flags.fnc


@to_aikit_arrays_and_back
def isreal(x):
    return aikit.isreal(x)


@to_aikit_arrays_and_back
def isrealobj(x: any):
    return not aikit.is_complex_dtype(aikit.dtype(x))


@to_aikit_arrays_and_back
def isscalar(element):
    return isinstance(
        element,
        (
            int,
            float,
            complex,
            bool,
            bytes,
            str,
            memoryview,
            numbers.Number,
            np_frontend.generic,
        ),
    )
