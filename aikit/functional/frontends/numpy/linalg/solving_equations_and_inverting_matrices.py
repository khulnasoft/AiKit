# global

# local
import aikit
from aikit.functional.frontends.numpy.func_wrapper import to_aikit_arrays_and_back

from aikit.func_wrapper import with_unsupported_dtypes
from aikit.functional.frontends.numpy import promote_types_of_numpy_inputs
from aikit.functional.frontends.numpy.linalg.norms_and_other_numbers import matrix_rank


# inv
@with_unsupported_dtypes({"1.26.2 and below": ("float16",)}, "numpy")
@to_aikit_arrays_and_back
def inv(a):
    return aikit.inv(a)


# TODO: replace this with function from API
# As the compositon provides unstable results
@to_aikit_arrays_and_back
@with_unsupported_dtypes({"1.26.2 and below": ("float16",)}, "numpy")
def lstsq(a, b, rcond="warn"):
    solution = aikit.matmul(
        aikit.pinv(a, rtol=1e-15).astype(aikit.float64), b.astype(aikit.float64)
    )
    svd = aikit.svd(a, compute_uv=False)
    rank = matrix_rank(a).astype(aikit.int32)
    residuals = aikit.sum((b - aikit.matmul(a, solution)) ** 2).astype(aikit.float64)
    return (solution, residuals, rank, svd[0])


# pinv
# TODO: add hermitian functionality
@with_unsupported_dtypes({"1.26.2 and below": ("float16",)}, "numpy")
@to_aikit_arrays_and_back
def pinv(a, rcond=1e-15, hermitian=False):
    return aikit.pinv(a, rtol=rcond)


# solve
@with_unsupported_dtypes({"1.26.2 and below": ("float16",)}, "numpy")
@to_aikit_arrays_and_back
def solve(a, b):
    a, b = promote_types_of_numpy_inputs(a, b)
    return aikit.solve(a, b)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"1.26.2 and below": ("float16", "blfloat16")}, "numpy")
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
    return aikit.reshape(ia, shape=new_shape)
