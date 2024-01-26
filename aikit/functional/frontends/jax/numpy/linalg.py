# local
import aikit
from aikit.functional.frontends.jax import Array
from aikit.functional.frontends.jax.func_wrapper import to_aikit_arrays_and_back
from aikit.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from aikit.functional.frontends.jax.numpy import promote_types_of_jax_inputs
from aikit.functional.frontends.numpy.linalg import lstsq as numpy_lstsq


@to_aikit_arrays_and_back
def cholesky(a):
    return aikit.cholesky(a)


@to_aikit_arrays_and_back
def cond(x, p=None):
    return aikit.cond(x, p=p)


@to_aikit_arrays_and_back
def det(a):
    return aikit.det(a)


@to_aikit_arrays_and_back
def eig(a):
    return aikit.eig(a)


@to_aikit_arrays_and_back
def eigh(a, UPLO="L", symmetrize_input=True):
    def symmetrize(x):
        # TODO : Take Hermitian transpose after complex numbers added
        return (x + aikit.swapaxes(x, -1, -2)) / 2

    if symmetrize_input:
        a = symmetrize(a)

    return aikit.eigh(a, UPLO=UPLO)


@to_aikit_arrays_and_back
def eigvals(a):
    return aikit.eigvals(a)


@to_aikit_arrays_and_back
def eigvalsh(a, UPLO="L"):
    return aikit.eigvalsh(a, UPLO=UPLO)


@to_aikit_arrays_and_back
def inv(a):
    return aikit.inv(a)


# TODO: replace this with function from API
# As the composition provides numerically unstable results
@to_aikit_arrays_and_back
def lstsq(a, b, rcond=None, *, numpy_resid=False):
    if numpy_resid:
        return numpy_lstsq(a, b, rcond=rcond)
    least_squares_solution = aikit.matmul(
        aikit.pinv(a, rtol=1e-15).astype(aikit.float64), b.astype(aikit.float64)
    )
    residuals = aikit.sum((b - aikit.matmul(a, least_squares_solution)) ** 2).astype(
        aikit.float64
    )
    svd_values = aikit.svd(a, compute_uv=False)
    rank = aikit.matrix_rank(a).astype(aikit.int32)
    return (least_squares_solution, residuals, rank, svd_values[0])


@to_aikit_arrays_and_back
def matrix_power(a, n):
    return aikit.matrix_power(a, n)


@to_aikit_arrays_and_back
def matrix_rank(M, tol=None):
    return aikit.matrix_rank(M, atol=tol)


@to_aikit_arrays_and_back
def multi_dot(arrays, *, precision=None):
    return aikit.multi_dot(arrays)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"0.4.23 and below": ("float32", "float64")},
    "jax",
)
def norm(x, ord=None, axis=None, keepdims=False):
    if ord is None:
        ord = 2
    if type(axis) in [list, tuple] and len(axis) == 2:
        return Array(aikit.matrix_norm(x, ord=ord, axis=axis, keepdims=keepdims))
    return Array(aikit.vector_norm(x, ord=ord, axis=axis, keepdims=keepdims))


@to_aikit_arrays_and_back
def pinv(a, rcond=None):
    return aikit.pinv(a, rtol=rcond)


@to_aikit_arrays_and_back
def qr(a, mode="reduced"):
    return aikit.qr(a, mode=mode)


@to_aikit_arrays_and_back
def slogdet(a, method=None):
    return aikit.slogdet(a)


@to_aikit_arrays_and_back
def solve(a, b):
    return aikit.solve(a, b)


@to_aikit_arrays_and_back
def svd(a, /, *, full_matrices=True, compute_uv=True, hermitian=None):
    if not compute_uv:
        return aikit.svdvals(a)
    return aikit.svd(a, full_matrices=full_matrices)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"0.4.23 and below": ("float16", "bfloat16")}, "jax")
def tensorinv(a, ind=2):
    old_shape = aikit.shape(a)
    prod = 1
    if ind > 0:
        invshape = old_shape[ind:] + old_shape[:ind]
        for k in old_shape[ind:]:
            prod *= k
    else:
        raise ValueError("Invalid ind argument.")
    a = aikit.reshape(a, shape=(prod, -1))
    ia = aikit.inv(a)
    new_shape = (*invshape,)
    return Array(aikit.reshape(ia, shape=new_shape))


@to_aikit_arrays_and_back
def tensorsolve(a, b, axes=None):
    a, b = promote_types_of_jax_inputs(a, b)
    return aikit.tensorsolve(a, b, axes=axes)
