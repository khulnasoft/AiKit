# local
from ..manipulation import *  # noqa: F401
import aikit
from aikit.functional.frontends.paddle.func_wrapper import (
    to_aikit_arrays_and_back,
)
from aikit.func_wrapper import with_unsupported_dtypes

# NOTE:
# Only inplace functions are to be added in this file.
# Please add non-inplace counterparts to `/frontends/paddle/manipulation.py`.


@with_unsupported_dtypes(
    {"2.5.2 and below": ("int8", "uint8", "int16", "uint16", "float16", "bfloat16")},
    "paddle",
)
@to_aikit_arrays_and_back
def reshape_(x, shape):
    ret = aikit.reshape(x, shape)
    aikit.inplace_update(x, ret)
    return x
