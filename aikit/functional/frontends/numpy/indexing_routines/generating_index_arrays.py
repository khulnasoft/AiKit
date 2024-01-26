import inspect

import aikit
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
)


@to_aikit_arrays_and_back
def diag_indices(n, ndim=2):
    idx = aikit.arange(n)
    res = aikit.array((idx,) * ndim)
    res = tuple(res.astype("int64"))
    return res


@to_aikit_arrays_and_back
def indices(dimensions, dtype=int, sparse=False):
    return aikit.indices(dimensions, dtype=dtype, sparse=sparse)


@to_aikit_arrays_and_back
def mask_indices(n, mask_func, k=0):
    mask_func_obj = inspect.unwrap(mask_func)
    mask_func_name = mask_func_obj.__name__
    try:
        aikit_mask_func_obj = getattr(aikit.functional.frontends.numpy, mask_func_name)
        a = aikit.ones((n, n))
        mask = aikit_mask_func_obj(a, k=k)
        indices = aikit.argwhere(mask.aikit_array)
        ret = indices[:, 0], indices[:, 1]
        return tuple(ret)
    except AttributeError as e:
        print(f"Attribute error: {e}")


@to_aikit_arrays_and_back
def tril_indices(n, k=0, m=None):
    return aikit.tril_indices(n, m, k)


@to_aikit_arrays_and_back
def tril_indices_from(arr, k=0):
    return aikit.tril_indices(arr.shape[0], arr.shape[1], k)


# unravel_index
@to_aikit_arrays_and_back
def unravel_index(indices, shape, order="C"):
    ret = [x.astype("int64") for x in aikit.unravel_index(indices, shape)]
    return tuple(ret)
