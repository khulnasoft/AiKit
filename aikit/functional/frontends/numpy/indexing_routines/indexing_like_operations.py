import aikit
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    inputs_to_aikit_arrays,
    handle_numpy_out,
)


@to_aikit_arrays_and_back
@handle_numpy_out
def compress(condition, a, axis=None, out=None):
    condition_arr = aikit.asarray(condition).astype(bool)
    if condition_arr.ndim != 1:
        raise aikit.utils.exceptions.AikitException("Condition must be a 1D array")
    if axis is None:
        arr = aikit.asarray(a).flatten()
        axis = 0
    else:
        arr = aikit.moveaxis(a, axis, 0)
    if condition_arr.shape[0] > arr.shape[0]:
        raise aikit.utils.exceptions.AikitException(
            "Condition contains entries that are out of bounds"
        )
    arr = arr[: condition_arr.shape[0]]
    return aikit.moveaxis(arr[condition_arr], 0, axis)


def diag(v, k=0):
    return aikit.diag(v, k=k)


@to_aikit_arrays_and_back
def diagonal(a, offset, axis1, axis2):
    return aikit.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)


@to_aikit_arrays_and_back
def fill_diagonal(a, val, wrap=False):
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    end = None
    if a.ndim == 2:
        # Explicit, fast formula for the common case.  For 2-d arrays, we
        # accept rectangular ones.
        step = a.shape[1] + 1
        # This is needed to don't have tall matrix have the diagonal wrap.
        if not wrap:
            end = a.shape[1] * a.shape[1]
    else:
        # For more than d=2, the strided formula is only valid for arrays with
        # all dimensions equal, so we check first.
        if not aikit.all(aikit.diff(a.shape) == 0):
            raise ValueError("All dimensions of input must be of equal length")
        step = 1 + aikit.sum(aikit.cumprod(a.shape[:-1]))

    # Write the value out into the diagonal.
    shape = a.shape
    a = aikit.reshape(a, a.size)
    a[:end:step] = val
    a = aikit.reshape(a, shape)


@to_aikit_arrays_and_back
def indices(dimensions, dtype=int, sparse=False):
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,) * N
    if sparse:
        res = ()
    else:
        res = aikit.empty((N,) + dimensions, dtype=dtype)
    for i, dim in enumerate(dimensions):
        idx = aikit.arange(dim, dtype=dtype).reshape(shape[:i] + (dim,) + shape[i + 1 :])
        if sparse:
            res = res + (idx,)
        else:
            res[i] = idx
    return res


@inputs_to_aikit_arrays
def put_along_axis(arr, indices, values, axis):
    aikit.put_along_axis(arr, indices, values, axis)


@to_aikit_arrays_and_back
@handle_numpy_out
def take(a, indices, /, *, axis=None, out=None, mode="raise"):
    return aikit.take(a, indices, axis=axis, out=out, mode=mode)


@to_aikit_arrays_and_back
def take_along_axis(arr, indices, axis):
    return aikit.take_along_axis(arr, indices, axis)


@to_aikit_arrays_and_back
def tril_indices(n, k=0, m=None):
    return aikit.tril_indices(n, m, k)


# unravel_index
@to_aikit_arrays_and_back
def unravel_index(indices, shape, order="C"):
    ret = [x.astype("int64") for x in aikit.unravel_index(indices, shape)]
    return tuple(ret)
