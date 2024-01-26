# local
import aikit
from aikit.functional.frontends.numpy.func_wrapper import to_aikit_arrays_and_back


@to_aikit_arrays_and_back
def flip(m, axis=None):
    return aikit.flip(m, axis=axis, out=None)


@to_aikit_arrays_and_back
def fliplr(m):
    return aikit.fliplr(m, out=None)


@to_aikit_arrays_and_back
def flipud(m):
    return aikit.flipud(m, out=None)


@to_aikit_arrays_and_back
def roll(a, shift, axis=None):
    return aikit.roll(a, shift, axis=axis)


@to_aikit_arrays_and_back
def rot90(m, k=1, axes=(0, 1)):
    return aikit.rot90(m, k=k, axes=axes)
