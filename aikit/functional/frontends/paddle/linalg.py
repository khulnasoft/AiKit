# global
import aikit
from aikit.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from aikit.functional.frontends.paddle import promote_types_of_paddle_inputs
from aikit.functional.frontends.paddle.func_wrapper import (
    to_aikit_arrays_and_back,
)


@with_supported_dtypes({"2.4.1 and above": ("int64",)}, "paddle")
@to_aikit_arrays_and_back
def bincount(x, weights=None, minlength=0, name=None):
    return aikit.bincount(x, weights=weights, minlength=minlength)


# bmm
@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def bmm(x, y, transpose_x=False, transpose_y=False, name=None):
    if len(aikit.shape(x)) != 3 or len(aikit.shape(y)) != 3:
        raise RuntimeError("input must be 3D matrices")
    x, y = promote_types_of_paddle_inputs(x, y)
    return aikit.matmul(x, y, transpose_a=transpose_x, transpose_b=transpose_y)


# cholesky
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def cholesky(x, /, *, upper=False, name=None):
    return aikit.cholesky(x, upper=upper)


# cholesky_solve
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def cholesky_solve(x, y, /, *, upper=False, name=None):
    if upper:
        y = aikit.matrix_transpose(y)
    Y = aikit.solve(y, x)
    return aikit.solve(aikit.matrix_transpose(y), Y)


# cond
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def cond(x, p=None, name=None):
    ret = aikit.cond(x, p=p, out=name)
    if ret.shape == ():
        ret = ret.reshape((1,))
    return ret


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def cross(x, y, /, *, axis=9, name=None):
    x, y = promote_types_of_paddle_inputs(x, y)
    return aikit.cross(x, y, axis=axis)


# diagonal
@with_supported_dtypes(
    {
        "2.6.0 and below": (
            "int32",
            "int64",
            "float64",
            "complex128",
            "float32",
            "complex64",
            "bool",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
def diagonal(x, offset=0, axis1=0, axis2=1, name=None):
    return aikit.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


@with_supported_dtypes({"2.4.1 and above": ("float64", "float32")}, "paddle")
@to_aikit_arrays_and_back
def dist(x, y, p=2):
    ret = aikit.vector_norm(aikit.subtract(x, y), ord=p)
    return aikit.reshape(ret, (1,))


# dot
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def dot(x, y, name=None):
    x, y = promote_types_of_paddle_inputs(x, y)
    out = aikit.multiply(x, y)
    return aikit.sum(out, axis=aikit.get_num_dims(x) - 1, keepdims=False)


# eig
@to_aikit_arrays_and_back
def eig(x, name=None):
    return aikit.eig(x)


# eigh
@to_aikit_arrays_and_back
def eigh(x, UPLO="L", name=None):
    return aikit.eigh(x, UPLO=UPLO)


# eigvals
@to_aikit_arrays_and_back
def eigvals(x, name=None):
    return aikit.eigvals(x)


# eigvalsh
@to_aikit_arrays_and_back
def eigvalsh(x, UPLO="L", name=None):
    return aikit.eigvalsh(x, UPLO=UPLO)


@to_aikit_arrays_and_back
def lu_unpack(lu_data, lu_pivots, unpack_datas=True, unpack_pivots=True, *, out=None):
    A = lu_data
    n = A.shape
    m = len(lu_pivots)
    pivot_matrix = aikit.eye(m)
    L = aikit.tril(A)
    L.fill_diagonal(1.000)
    U = aikit.triu(A)
    for i in range(m):
        if i != lu_pivots[i] - 1:
            pivot_matrix[[i, lu_pivots[i] - 1]] = pivot_matrix[[lu_pivots[i] - 1, i]]
        P = pivot_matrix
    if not unpack_datas:
        L = aikit.zeros(n)
        U = aikit.zeros(n)
        if not unpack_pivots:
            P = aikit.zeros(n)
        else:
            P = pivot_matrix
        result = f"P={P}\n" + f"L={L}\n" + f"U={U}"
        return result
    elif not unpack_pivots:
        P = aikit.zeros(n)
        result = f"P={P}\n" + f"L={L}\n" + f"U={U}"
        return result
    else:
        result = f"P={P}\n" + f"L={L}\n" + f"U={U}"
        return result


# matmul
@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def matmul(x, y, transpose_x=False, transpose_y=False, name=None):
    x, y = promote_types_of_paddle_inputs(x, y)
    return aikit.matmul(x, y, transpose_a=transpose_x, transpose_b=transpose_y)


# matrix_power
@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def matrix_power(x, n, name=None):
    return aikit.matrix_power(x, n)


# mv
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def mv(x, vec, name=None):
    return aikit.dot(x, vec)


# norm
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def norm(x, p="fro", axis=None, keepdim=False, name=None):
    if axis is None and p is not None:
        if p == "fro":
            p = 2
        ret = aikit.vector_norm(x.flatten(), ord=p, axis=-1)
        if keepdim:
            ret = ret.reshape([1] * len(x.shape))
        return ret

    if isinstance(axis, tuple):
        axis = list(axis)
    if isinstance(axis, list) and len(axis) == 1:
        axis = axis[0]

    if isinstance(axis, int):
        if p == "fro":
            p = 2
        if p in [0, 1, 2, aikit.inf, -aikit.inf]:
            ret = aikit.vector_norm(x, ord=p, axis=axis, keepdims=keepdim)
        elif isinstance(p, (int, float)):
            ret = aikit.pow(
                aikit.sum(aikit.pow(aikit.abs(x), p), axis=axis, keepdims=keepdim),
                float(1.0 / p),
            )

    elif isinstance(axis, list) and len(axis) == 2:
        if p == 0:
            raise ValueError
        elif p == 1:
            ret = aikit.sum(aikit.abs(x), axis=axis, keepdims=keepdim)
        elif p in [2, "fro"]:
            ret = aikit.matrix_norm(x, ord="fro", axis=axis, keepdims=keepdim)
        elif p == aikit.inf:
            ret = aikit.max(aikit.abs(x), axis=axis, keepdims=keepdim)
        elif p == -aikit.inf:
            ret = aikit.min(aikit.abs(x), axis=axis, keepdims=keepdim)
        elif isinstance(p, (int, float)) and p > 0:
            ret = aikit.pow(
                aikit.sum(aikit.pow(aikit.abs(x), p), axis=axis, keepdims=keepdim),
                float(1.0 / p),
            )
        else:
            raise ValueError

    else:
        raise ValueError

    return ret


# pinv
@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def pinv(x, rcond=1e-15, hermitian=False, name=None):
    # TODO: Add hermitian functionality
    return aikit.pinv(x, rtol=rcond)


# qr
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def qr(x, mode="reduced", name=None):
    return aikit.qr(x, mode=mode)


# solve
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def solve(x, y, name=None):
    return aikit.solve(x, y)


# transpose
@with_unsupported_dtypes({"2.6.0 and below": ("uint8", "int8", "int16")}, "paddle")
@to_aikit_arrays_and_back
def transpose(x, perm, name=None):
    return aikit.permute_dims(x, axes=perm)
