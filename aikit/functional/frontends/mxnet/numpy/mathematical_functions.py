# local
import aikit
from aikit.functional.frontends.mxnet.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_mxnet_out,
)
from aikit.functional.frontends.mxnet.numpy import promote_types_of_mxnet_inputs


@handle_mxnet_out
@to_aikit_arrays_and_back
def add(x1, x2, out=None):
    x1, x2 = promote_types_of_mxnet_inputs(x1, x2)
    return aikit.add(x1, x2, out=out)


@handle_mxnet_out
@to_aikit_arrays_and_back
def sin(x, out=None, **kwargs):
    return aikit.sin(x, out=out, **kwargs)
