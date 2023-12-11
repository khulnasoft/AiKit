# local
import aikit
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_numpy_casting,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)
import aikit.functional.frontends.numpy as np_frontend


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def concatenate(arrays, /, *, axis=0, out=None, dtype=None, casting="same_kind"):
    if dtype is not None:
        out_dtype = aikit.as_aikit_dtype(dtype)
    else:
        out_dtype = aikit.dtype(arrays[0])
        for i in arrays:
            out_dtype = aikit.as_aikit_dtype(
                np_frontend.promote_numpy_dtypes(i.dtype, out_dtype)
            )
    return aikit.concat(arrays, axis=axis, out=out).astype(out_dtype, copy=False)


@to_aikit_arrays_and_back
def hstack(tup):
    out_dtype = aikit.dtype(tup[0])
    for i in tup:
        out_dtype = aikit.as_aikit_dtype(
            np_frontend.promote_numpy_dtypes(i.dtype, out_dtype)
        )
    return aikit.hstack(tup)


@handle_numpy_out
@to_aikit_arrays_and_back
def stack(arrays, /, *, axis=0, out=None):
    out_dtype = aikit.dtype(arrays[0])
    for i in arrays:
        out_dtype = aikit.as_aikit_dtype(
            np_frontend.promote_numpy_dtypes(i.dtype, out_dtype)
        )
    return aikit.stack(arrays, axis=axis, out=out).astype(out_dtype, copy=False)


@to_aikit_arrays_and_back
def vstack(tup):
    out_dtype = aikit.dtype(tup[0])
    for i in tup:
        out_dtype = aikit.as_aikit_dtype(
            np_frontend.promote_numpy_dtypes(i.dtype, out_dtype)
        )
    return aikit.vstack(tup)


row_stack = vstack
