# local
import aikit
from aikit.functional.frontends.numpy.func_wrapper import to_aikit_arrays_and_back


@to_aikit_arrays_and_back
def repeat(a, repeats, axis=None):
    return aikit.repeat(a, repeats, axis=axis)


@to_aikit_arrays_and_back
def tile(A, reps):
    return aikit.tile(A, reps)
