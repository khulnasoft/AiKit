import aikit
from aikit.functional.frontends.mxnet.func_wrapper import to_aikit_arrays_and_back


@to_aikit_arrays_and_back
def diagonal(a, offset=0, axis1=0, axis2=1):
    return aikit.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)
