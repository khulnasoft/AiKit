import aikit
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_numpy_dtype,
)


@handle_numpy_dtype
@to_aikit_arrays_and_back
def array(object, dtype=None, *, copy=True, order="K", subok=False, ndmin=0, like=None):
    ret = aikit.array(object, copy=copy, dtype=dtype)
    if aikit.get_num_dims(ret) < ndmin:
        ret = aikit.expand_dims(ret, axis=list(range(ndmin - aikit.get_num_dims(ret))))
    return ret


@handle_numpy_dtype
@to_aikit_arrays_and_back
def asarray(
    a,
    dtype=None,
    order=None,
    *,
    like=None,
):
    return aikit.asarray(a, dtype=dtype)


@to_aikit_arrays_and_back
def copy(a, order="K", subok=False):
    return aikit.copy_array(a)


@handle_numpy_dtype
def frombuffer(buffer, dtype=float, count=-1, offset=0, *, like=None):
    return aikit.frombuffer(buffer)
