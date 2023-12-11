# local
import aikit
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
)


@to_aikit_arrays_and_back
def asanyarray(a, dtype=None, order=None, like=None):
    return aikit.asarray(a)


@to_aikit_arrays_and_back
def asarray_chkfinite(a, dtype=None, order=None):
    a = aikit.asarray(a, dtype=dtype)
    if not aikit.all(aikit.isfinite(a)):
        raise ValueError("array must not contain infs or NaNs")
    return a


@to_aikit_arrays_and_back
def asfarray(a, dtype=aikit.float64):
    return aikit.asarray(a, dtype=aikit.float64)


@to_aikit_arrays_and_back
def broadcast_to(array, shape, subok=False):
    return aikit.broadcast_to(array, shape)


@to_aikit_arrays_and_back
def moveaxis(a, source, destination):
    return aikit.moveaxis(a, source, destination)


@to_aikit_arrays_and_back
def ravel(a, order="C"):
    return aikit.reshape(a, shape=(-1,), order=order)


@to_aikit_arrays_and_back
def require(a, dtype=None, requirements=None, *, like=None):
    return aikit.asarray(a, dtype=dtype)


@to_aikit_arrays_and_back
def reshape(x, /, newshape, order="C"):
    return aikit.reshape(x, shape=newshape, order=order)


@to_aikit_arrays_and_back
def resize(x, newshape, /, refcheck=True):
    if isinstance(newshape, int):
        newshape = (newshape,)
    x_new = aikit.reshape(x, shape=(-1,), order="C")
    total_size = 1
    for diff_size in newshape:
        total_size *= diff_size
        if diff_size < 0:
            raise ValueError("values must not be negative")
    if x_new.size == 0 or total_size == 0:
        return aikit.zeros_like(x_new)
    repetition = -(-total_size // len(x_new))
    conc = (x_new,) * repetition
    x_new = aikit.concat(conc)[:total_size]
    y = aikit.reshape(x_new, shape=newshape, order="C")
    return y
