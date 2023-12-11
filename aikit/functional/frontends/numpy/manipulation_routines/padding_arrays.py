# local
import aikit
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
)


@to_aikit_arrays_and_back
def pad(array, pad_width, mode="constant", **kwargs):
    return aikit.pad(array, pad_width, mode=mode, **kwargs)
