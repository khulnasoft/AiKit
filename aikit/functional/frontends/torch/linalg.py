# local
import math
import aikit
import aikit.functional.frontends.torch as torch_frontend
from aikit.functional.frontends.torch.func_wrapper import to_aikit_arrays_and_back
from aikit.func_wrapper import with_supported_dtypes, with_unsupported_dtypes
from collections import namedtuple


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def cholesky(input, *, upper=False, out=None):
    return aikit.cholesky(input, upper=upper, out=out)


@to_aikit_arrays_and_back
def cholesky_ex(input, *, upper=False, check_errors=False, out=None):
    try:
        matrix = aikit.cholesky(input, upper=upper, out=out)
        info = aikit.zeros(input.shape[:-2], dtype=aikit.int32)
        return matrix, info
    except RuntimeError as e:
        if check_errors:
            raise RuntimeError(e)
        else:
            matrix = input * math.nan
            info = aikit.ones(input.shape[:-2], dtype=aikit.int32)
            return matrix, info


@to_aikit_arrays_and_back
@with_supported_dtypes({"2.1.1 and below": ("float32", "float64", "complex")}, "torch")
def cond(input, p=None, *, out=None):
    return aikit.cond(input, p=p, out=out)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def cross(input, other, *, dim=None, out=None):
    return torch_frontend.miscellaneous_ops.cross(input, other, dim=dim, out=out)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def det(A, *, out=None):
    return aikit.det(A, out=out)


@to_aikit_arrays_and_back
def diagonal(A, *, offset=0, dim1=-2, dim2=-1):
    return torch_frontend.diagonal(A, offset=offset, dim1=dim1, dim2=dim2)


@to_aikit_arrays_and_back
def divide(input, other, *, rounding_mode=None, out=None):
    return aikit.divide(input, other, out=out)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("bfloat16", "float16")}, "torch")
def eig(input, *, out=None):
    return aikit.eig(input, out=out)


@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64", "complex128")},
    "torch",
)
def eigh(A, UPLO="L", *, out=None):
    return aikit.eigh(A, UPLO=UPLO, out=out)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def eigvals(input, *, out=None):
    ret = aikit.eigvals(input)
    if aikit.exists(out):
        return aikit.inplace_update(out, ret)
    return ret


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def eigvalsh(input, UPLO="L", *, out=None):
    ret = aikit.eigvalsh(input, UPLO=UPLO, out=out)
    if "complex64" in aikit.as_aikit_dtype(ret.dtype):
        ret = aikit.astype(ret, aikit.float32)
    elif "complex128" in aikit.as_aikit_dtype(ret.dtype):
        ret = aikit.astype(ret, aikit.float64)
    return ret


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def inv(A, *, out=None):
    return aikit.inv(A, out=out)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def inv_ex(A, *, check_errors=False, out=None):
    if aikit.any(aikit.det(A) == 0):
        if check_errors:
            raise RuntimeError("Singular Matrix")
        else:
            inv = A * math.nan
            # TODO: info should return an array containing the diagonal element of the
            # LU decomposition of the input matrix that is exactly zero
            info = aikit.ones(A.shape[:-2], dtype=aikit.int32)
    else:
        inv = aikit.inv(A, out=out)
        info = aikit.zeros(A.shape[:-2], dtype=aikit.int32)
    return inv, info


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def lu_factor(A, *, pivot=True, out=None):
    return aikit.lu_factor(A, pivot=pivot, out=out)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def matmul(input, other, *, out=None):
    return aikit.matmul(input, other, out=out)


@to_aikit_arrays_and_back
@with_supported_dtypes({"2.1.1 and below": ("float32", "float64", "complex")}, "torch")
def matrix_exp(A):
    return aikit.matrix_exp(A)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def matrix_norm(input, ord="fro", dim=(-2, -1), keepdim=False, *, dtype=None, out=None):
    if "complex" in aikit.as_aikit_dtype(input.dtype):
        input = aikit.abs(input)
    if dtype:
        input = aikit.astype(input, aikit.as_aikit_dtype(dtype))
    return aikit.matrix_norm(input, ord=ord, axis=dim, keepdims=keepdim, out=out)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def matrix_power(A, n, *, out=None):
    return aikit.matrix_power(A, n, out=out)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def matrix_rank(A, *, atol=None, rtol=None, hermitian=False, out=None):
    return aikit.matrix_rank(A, atol=atol, rtol=rtol, hermitian=hermitian, out=out)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def multi_dot(tensors, *, out=None):
    return aikit.multi_dot(tensors, out=out)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex64", "complex128")}, "torch"
)
def norm(input, ord=None, dim=None, keepdim=False, *, dtype=None, out=None):
    if dim is None and (ord is not None):
        if input.ndim == 1:
            ret = aikit.vector_norm(input, axis=dim, keepdims=keepdim, ord=ord)
        else:
            ret = aikit.matrix_norm(input, keepdims=keepdim, ord=ord)
    elif dim is None and ord is None:
        input = aikit.flatten(input)
        ret = aikit.vector_norm(input, axis=0, keepdims=keepdim, ord=2)
    if isinstance(dim, int):
        ret = aikit.vector_norm(input, axis=dim, keepdims=keepdim, ord=ord)
    elif isinstance(dim, tuple):
        ret = aikit.matrix_norm(input, axis=dim, keepdims=keepdim, ord=ord)
    return ret


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def pinv(input, *, atol=None, rtol=None, hermitian=False, out=None):
    # TODO: add handling for hermitian
    if atol is None:
        return aikit.pinv(input, rtol=rtol, out=out)
    else:
        sigma = aikit.svdvals(input)[0]
        if rtol is None:
            rtol = atol / sigma
        else:
            if atol > rtol * sigma:
                rtol = atol / sigma

    return aikit.pinv(input, rtol=rtol, out=out)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def qr(A, mode="reduced", *, out=None):
    if mode == "reduced":
        ret = aikit.qr(A, mode="reduced")
    elif mode == "r":
        Q, R = aikit.qr(A, mode="r")
        Q = []
        ret = Q, R
    elif mode == "complete":
        ret = aikit.qr(A, mode="complete")
    if aikit.exists(out):
        return aikit.inplace_update(out, ret)
    return ret


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def slogdet(A, *, out=None):
    sign, logabsdet = aikit.slogdet(A)
    if "complex64" in aikit.as_aikit_dtype(logabsdet.dtype):
        logabsdet = aikit.astype(logabsdet, aikit.float32)
    if "complex128" in aikit.as_aikit_dtype(logabsdet.dtype):
        logabsdet = aikit.astype(logabsdet, aikit.float64)
    ret = namedtuple("slogdet", ["sign", "logabsdet"])(sign, logabsdet)
    if aikit.exists(out):
        return aikit.inplace_update(out, ret, keep_input_dtype=True)
    return ret


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def solve(A, B, *, left=True, out=None):
    if left:
        return aikit.solve(A, B, out=out)

    A_t = aikit.linalg.matrix_transpose(A)
    B_t = aikit.linalg.matrix_transpose(B if B.ndim > 1 else aikit.reshape(B, (-1, 1)))
    X_t = aikit.solve(A_t, B_t)
    return aikit.linalg.matrix_transpose(X_t, out=out)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def solve_ex(A, B, *, left=True, check_errors=False, out=None):
    try:
        if left:
            result = aikit.solve(A, B, out=out)
        else:
            A_t = aikit.linalg.matrix_transpose(A)
            B_t = aikit.linalg.matrix_transpose(
                B if B.ndim > 1 else aikit.reshape(B, (-1, 1))
            )
            X_t = aikit.solve(A_t, B_t)
            result = aikit.linalg.matrix_transpose(X_t, out=out)

        info = aikit.zeros(A.shape[:-2], dtype=aikit.int32)
        return result, info
    except RuntimeError as e:
        if check_errors:
            raise RuntimeError(e)
        else:
            result = A * math.nan
            info = aikit.ones(A.shape[:-2], dtype=aikit.int32)

            return result, info


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def svd(A, /, *, full_matrices=True, driver=None, out=None):
    # TODO: add handling for driver and out
    return aikit.svd(A, compute_uv=True, full_matrices=full_matrices)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def svdvals(A, *, driver=None, out=None):
    if driver in ["gesvd", "gesvdj", "gesvda", None]:
        return aikit.svdvals(A, driver=driver, out=out)
    else:
        raise ValueError("Unsupported SVD driver")


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def tensorinv(input, ind=2, *, out=None):
    not_invertible = "Reshaped tensor is not invertible"
    prod_cond = "Tensor shape must satisfy prod(A.shape[:ind]) == prod(A.shape[ind:])"
    positive_ind_cond = "Expected a strictly positive integer for 'ind'"
    input_shape = aikit.shape(input)
    assert ind > 0, f"{positive_ind_cond}"
    shape_ind_end = input_shape[:ind]
    shape_ind_start = input_shape[ind:]
    prod_ind_end = 1
    prod_ind_start = 1
    for i in shape_ind_start:
        prod_ind_start *= i
    for j in shape_ind_end:
        prod_ind_end *= j
    assert prod_ind_end == prod_ind_start, f"{prod_cond}."
    inverse_shape = shape_ind_start + shape_ind_end
    input = aikit.reshape(input, shape=(prod_ind_end, -1))
    inverse_shape_tuple = (*inverse_shape,)
    assert inv_ex(input, check_errors=True), f"{not_invertible}."
    inverse_tensor = aikit.inv(input)
    return aikit.reshape(inverse_tensor, shape=inverse_shape_tuple, out=out)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def tensorsolve(A, B, dims=None, *, out=None):
    return aikit.tensorsolve(A, B, axes=dims, out=out)


@to_aikit_arrays_and_back
@with_supported_dtypes({"2.1.1 and below": ("integer", "float", "complex")}, "torch")
def vander(x, N=None):
    if len(x.shape) < 1:
        raise RuntimeError("Input dim must be greater than or equal to 1.")

    # pytorch always return int64 for integers
    if "int" in x.dtype:
        x = aikit.astype(x, aikit.int64)

    if len(x.shape) == 1:
        # torch always returns the powers in ascending order
        return aikit.vander(x, N=N, increasing=True)

    # support multi-dimensional array
    original_shape = x.shape
    if N is None:
        N = x.shape[-1]

    # store the vander output
    x = aikit.reshape(x, (-1, x.shape[-1]))
    output = []

    for i in range(x.shape[0]):
        output.append(aikit.vander(x[i], N=N, increasing=True))

    output = aikit.stack(output)
    output = aikit.reshape(output, (*original_shape, N))
    output = aikit.astype(output, x.dtype)
    return output


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def vecdot(x, y, *, dim=-1, out=None):
    if "complex" in aikit.as_aikit_dtype(x.dtype):
        x = aikit.conj(x)
    return aikit.sum(aikit.multiply(x, y), axis=dim)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "complex32", "complex64")}, "torch"
)
def vector_norm(input, ord=2, dim=None, keepdim=False, *, dtype=None, out=None):
    return aikit.vector_norm(
        input, axis=dim, keepdims=keepdim, ord=ord, out=out, dtype=dtype
    )
