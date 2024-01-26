import aikit
from aikit.functional.frontends.torch.func_wrapper import (
    to_aikit_arrays_and_back,
    outputs_to_native_arrays,
)
from aikit.func_wrapper import outputs_to_aikit_arrays


def vmap(func, in_dims=0, out_dims=0, randomness="error", *, chunk_size=None):
    fun = outputs_to_native_arrays(func)
    return to_aikit_arrays_and_back(
        outputs_to_aikit_arrays(aikit.vmap(fun, in_axes=in_dims, out_axes=out_dims))
    )
