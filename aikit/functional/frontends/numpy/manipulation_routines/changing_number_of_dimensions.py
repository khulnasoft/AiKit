# global
import aikit
from aikit.functional.frontends.numpy.func_wrapper import to_aikit_arrays_and_back


# atleast_1d
@to_aikit_arrays_and_back
def atleast_1d(
    *arys,
):
    return aikit.atleast_1d(*arys)


@to_aikit_arrays_and_back
def atleast_2d(*arys):
    return aikit.atleast_2d(*arys)


@to_aikit_arrays_and_back
def atleast_3d(*arys):
    return aikit.atleast_3d(*arys)


# broadcast_arrays
@to_aikit_arrays_and_back
def broadcast_arrays(*args):
    return aikit.broadcast_arrays(*args)


# expand_dims
@to_aikit_arrays_and_back
def expand_dims(
    a,
    axis,
):
    return aikit.expand_dims(a, axis=axis)


# squeeze
@to_aikit_arrays_and_back
def squeeze(
    a,
    axis=None,
):
    return aikit.squeeze(a, axis=axis)
