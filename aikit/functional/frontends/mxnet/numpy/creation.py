import aikit
from aikit.functional.frontends.mxnet.func_wrapper import (
    to_aikit_arrays_and_back,
)
from aikit.functional.frontends.numpy.func_wrapper import handle_numpy_dtype


@handle_numpy_dtype
@to_aikit_arrays_and_back
def array(object, dtype=None, ctx=None):
    if not aikit.is_array(object) and not dtype:
        return aikit.array(object, dtype="float32", device=ctx)
    return aikit.array(object, dtype=dtype, device=ctx)
