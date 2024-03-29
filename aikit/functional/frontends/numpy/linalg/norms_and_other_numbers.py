# local
import aikit
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    from_zero_dim_arrays_to_scalar,
)

from aikit.func_wrapper import with_unsupported_dtypes


# det
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def det(a):
    return aikit.det(a)


# matrix_rank
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def matrix_rank(A, tol=None, hermitian=False):
    return aikit.matrix_rank(A, atol=tol, hermitian=hermitian)


# solve
@with_unsupported_dtypes({"1.26.3 and below": ("float16",)}, "numpy")
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def norm(x, ord=None, axis=None, keepdims=False):
    if axis is None and (ord is not None):
        if x.ndim not in (1, 2):
            raise ValueError("Improper number of dimensions to norm.")
        else:
            if x.ndim == 1:
                ret = aikit.vector_norm(x, axis=axis, keepdims=keepdims, ord=ord)
            else:
                ret = aikit.matrix_norm(x, axis=axis, keepdims=keepdims, ord=ord)
    elif axis is None and ord is None:
        x = aikit.flatten(x)
        ret = aikit.vector_norm(x, axis=0, keepdims=keepdims, ord=2)
    if isinstance(axis, int):
        ret = aikit.vector_norm(x, axis=axis, keepdims=keepdims, ord=ord)
    elif isinstance(axis, tuple):
        ret = aikit.matrix_norm(x, axis=axis, keepdims=keepdims, ord=ord)
    return ret


# slogdet
@with_unsupported_dtypes({"1.26.3 and below": ("float16",)}, "numpy")
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def slogdet(a):
    sign, logabsdet = aikit.slogdet(a)
    return sign, logabsdet


# trace
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def trace(a, offset=0, axis1=0, axis2=1, out=None):
    ret = aikit.trace(a, offset=offset, axis1=axis1, axis2=axis2, out=out)
    return ret
