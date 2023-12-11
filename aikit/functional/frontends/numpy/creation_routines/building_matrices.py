import aikit
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_numpy_dtype,
)


@to_aikit_arrays_and_back
def diag(v, k=0):
    return aikit.diag(v, k=k)


# diagflat
@to_aikit_arrays_and_back
def diagflat(v, k=0):
    ret = aikit.diagflat(v, offset=k)
    while len(aikit.shape(ret)) < 2:
        ret = ret.expand_dims(axis=0)
    return ret


@handle_numpy_dtype
@to_aikit_arrays_and_back
def tri(N, M=None, k=0, dtype="float64", *, like=None):
    if M is None:
        M = N
    ones = aikit.ones((N, M), dtype=dtype)
    return aikit.tril(ones, k=k)


@to_aikit_arrays_and_back
def tril(m, k=0):
    return aikit.tril(m, k=k)


@to_aikit_arrays_and_back
def triu(m, k=0):
    return aikit.triu(m, k=k)


@to_aikit_arrays_and_back
def vander(x, N=None, increasing=False):
    if aikit.is_float_dtype(x):
        x = x.astype(aikit.float64)
    elif aikit.is_bool_dtype or aikit.is_int_dtype(x):
        x = x.astype(aikit.int64)
    return aikit.vander(x, N=N, increasing=increasing)
