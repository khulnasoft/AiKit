# global
import aikit
from aikit.func_wrapper import with_supported_dtypes
from aikit.functional.frontends.paddle.func_wrapper import (
    to_aikit_arrays_and_back,
)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int16", "int32", "int64", "uint8")},
    "paddle",
)
@to_aikit_arrays_and_back
def argmax(x, /, *, axis=None, keepdim=False, dtype="int64", name=None):
    return aikit.argmax(x, axis=axis, keepdims=keepdim, dtype=dtype)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int16", "int32", "int64", "uint8")},
    "paddle",
)
@to_aikit_arrays_and_back
def argmin(x, /, *, axis=None, keepdim=False, dtype="int64", name=None):
    return aikit.argmin(x, axis=axis, keepdims=keepdim, dtype=dtype)


@with_supported_dtypes(
    {"2.4.2 and below": ("float32", "float64", "int16", "int32", "int64", "uint8")},
    "paddle",
)
@to_aikit_arrays_and_back
def argsort(x, /, *, axis=-1, descending=False, name=None):
    return aikit.argsort(x, axis=axis, descending=descending)


@with_supported_dtypes(
    {"2.5.2 and below": ("int32", "int64", "float32", "float64")},
    "paddle",
)
@to_aikit_arrays_and_back
def index_sample(x, index):
    return x[aikit.arange(x.shape[0])[:, None], index]


# kthvalue
@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def kthvalue(x, k, axis=None, keepdim=False, name=None):
    if axis is None:
        axis = -1
    sorted_input = aikit.sort(x, axis=axis)
    sort_indices = aikit.argsort(x, axis=axis)

    values = aikit.gather(sorted_input, aikit.array(k - 1), axis=axis)
    indices = aikit.gather(sort_indices, aikit.array(k - 1), axis=axis)

    if keepdim:
        values = aikit.expand_dims(values, axis=axis)
        indices = aikit.expand_dims(indices, axis=axis)

    ret = (values, indices)
    return ret


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")},
    "paddle",
)
@to_aikit_arrays_and_back
def masked_select(x, mask, name=None):
    return aikit.flatten(x[mask])


@with_supported_dtypes(
    {"2.4.2 and below": ("float32", "float64", "int16", "int32", "int64", "uint8")},
    "paddle",
)
@to_aikit_arrays_and_back
def nonzero(input, /, *, as_tuple=False):
    ret = aikit.nonzero(input)
    if as_tuple is False:
        ret = aikit.matrix_transpose(aikit.stack(ret))
    return ret


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")},
    "paddle",
)
@to_aikit_arrays_and_back
def searchsorted(sorted_sequence, values, out_int32=False, right=False, name=None):
    if right:
        side = "right"
    else:
        side = "left"
    ret = aikit.searchsorted(sorted_sequence, values, side=side)
    if out_int32:
        ret = aikit.astype(ret, "int32")
    return ret


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")},
    "paddle",
)
@to_aikit_arrays_and_back
def sort(x, /, *, axis=-1, descending=False, name=None):
    return aikit.sort(x, axis=axis, descending=descending)


@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")},
    "paddle",
)
@to_aikit_arrays_and_back
def topk(x, k, axis=None, largest=True, sorted=True, name=None):
    return aikit.top_k(x, k, axis=axis, largest=largest, sorted=sorted)


# where
@with_supported_dtypes(
    {"2.5.2 and below": ("float32", "float64", "int32", "int64")},
    "paddle",
)
@to_aikit_arrays_and_back
def where(condition, x, y, name=None):
    return aikit.where(condition, x, y)
