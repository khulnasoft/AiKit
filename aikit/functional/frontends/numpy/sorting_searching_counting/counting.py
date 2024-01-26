# global
import aikit
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    from_zero_dim_arrays_to_scalar,
)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def count_nonzero(a, axis=None, *, keepdims=False):
    x = aikit.array(a)
    zero = aikit.zeros(aikit.shape(x), dtype=x.dtype)
    return aikit.sum(
        aikit.astype(aikit.not_equal(x, zero), aikit.int64),
        axis=axis,
        keepdims=keepdims,
    )
