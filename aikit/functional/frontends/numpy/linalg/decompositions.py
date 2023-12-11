# local
import aikit
from aikit.functional.frontends.numpy.func_wrapper import to_aikit_arrays_and_back


@to_aikit_arrays_and_back
def cholesky(a):
    return aikit.cholesky(a)


@to_aikit_arrays_and_back
def qr(a, mode="reduced"):
    return aikit.qr(a, mode=mode)


@to_aikit_arrays_and_back
def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    # Todo: conpute_uv and hermitian handling
    return aikit.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)
