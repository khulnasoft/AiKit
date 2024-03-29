# local
from ..manipulation import *  # noqa: F401
import aikit
from aikit.functional.frontends.paddle.func_wrapper import (
    to_aikit_arrays_and_back,
)
from aikit.func_wrapper import with_unsupported_dtypes


@with_supported_dtypes(
    {"2.5.1 and below": ("bool", "int32", "int64", "float16", "float32", "float64")},
    "paddle",
)
@to_aikit_arrays_and_back
def index_add_(x, index, axis, value, *, name=None):
    x = aikit.swapaxes(x, axis, 0)
    value = aikit.swapaxes(value, axis, 0)
    _to_adds = []
    index = sorted(zip(aikit.to_list(index), range(len(index))), key=(lambda i: i[0]))
    while index:
        _curr_idx = index[0][0]
        while len(_to_adds) < _curr_idx:
            _to_adds.append(aikit.zeros_like(value[0]))
        _to_add_cum = aikit.get_item(value, index[0][1])
        while (len(index)) > 1 and (index[0][0] == index[1][0]):
            _to_add_cum = _to_add_cum + aikit.get_item(value, index.pop(1)[1])
        index.pop(0)
        _to_adds.append(_to_add_cum)
    while len(_to_adds) < x.shape[0]:
        _to_adds.append(aikit.zeros_like(value[0]))
    _to_adds = aikit.stack(_to_adds)
    if len(x.shape) < 2:
        # Added this line due to the paddle backend treating scalars as 1-d arrays
        _to_adds = aikit.flatten(_to_adds)

    ret = aikit.add(x, _to_adds)
    ret = aikit.swapaxes(ret, axis, 0)
    x = ret
    return x


# NOTE:
# Only inplace functions are to be added in this file.
# Please add non-inplace counterparts to `/frontends/paddle/manipulation.py`.


@with_unsupported_dtypes(
    {"2.6.0 and below": ("int8", "uint8", "int16", "uint16", "float16", "bfloat16")},
    "paddle",
)
@to_aikit_arrays_and_back
def reshape_(x, shape):
    aikit.reshape(x, shape)
    return x
