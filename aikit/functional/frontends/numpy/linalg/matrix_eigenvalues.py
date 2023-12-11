# local
import aikit
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    from_zero_dim_arrays_to_scalar,
)


@to_aikit_arrays_and_back
def eig(a):
    return aikit.eig(a)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def eigh(a, /, UPLO="L"):
    return aikit.eigh(a, UPLO=UPLO)


@to_aikit_arrays_and_back
def eigvals(a):
    return aikit.eig(a)[0]


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def eigvalsh(a, /, UPLO="L"):
    return aikit.eigvalsh(a, UPLO=UPLO)
