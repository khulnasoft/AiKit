# global
import aikit
from aikit.functional.frontends.numpy.func_wrapper import to_aikit_arrays_and_back


@to_aikit_arrays_and_back
def argsort(
    x,
    /,
    *,
    axis=-1,
    kind=None,
    order=None,
):
    return aikit.argsort(x, axis=axis)


@to_aikit_arrays_and_back
def lexsort(keys, /, *, axis=-1):
    return aikit.lexsort(keys, axis=axis)


@to_aikit_arrays_and_back
def msort(a):
    return aikit.msort(a)


@to_aikit_arrays_and_back
def partition(a, kth, axis=-1, kind="introselect", order=None):
    sorted_arr = aikit.sort(a, axis=axis)
    for k in kth:
        index_to_remove = aikit.argwhere(a == sorted_arr[k])[0, 0]
        if len(a) == 1:
            a = aikit.array([], dtype=a.dtype)
        else:
            a = aikit.concat((a[:index_to_remove], a[index_to_remove + 1 :]))
        left = aikit.array([], dtype=a.dtype)
        right = aikit.array([], dtype=a.dtype)
        equal = aikit.array([], dtype=a.dtype)
        for i in range(len(a)):
            if a[i] < sorted_arr[k]:
                left = aikit.concat((left, aikit.array([a[i]], dtype=a.dtype)))
            elif a[i] > sorted_arr[k]:
                right = aikit.concat((right, aikit.array([a[i]], dtype=a.dtype)))
            else:
                equal = aikit.concat((equal, aikit.array([a[i]], dtype=a.dtype)))
        for j in range(len(equal)):
            if len(left) == len(sorted_arr[:k]):
                right = aikit.concat((right, aikit.array([equal[j]], dtype=a.dtype)))
            else:
                left = aikit.concat((left, aikit.array([equal[j]], dtype=a.dtype)))
        a = aikit.concat((left, aikit.array([sorted_arr[k]], dtype=a.dtype), right))
    return a


@to_aikit_arrays_and_back
def sort(a, axis=-1, kind=None, order=None):
    return aikit.sort(a, axis=axis)


@to_aikit_arrays_and_back
def sort_complex(a):
    return aikit.sort(a)
