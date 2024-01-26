import aikit
from aikit.functional.frontends.mxnet.func_wrapper import to_aikit_arrays_and_back


@to_aikit_arrays_and_back
def tensordot(a, b, axes=2):
    return aikit.tensordot(a, b, axes=axes)
