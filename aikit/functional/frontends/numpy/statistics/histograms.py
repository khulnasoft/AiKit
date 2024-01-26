import aikit
from aikit.functional.frontends.numpy.func_wrapper import to_aikit_arrays_and_back
from aikit.func_wrapper import with_supported_dtypes


@with_supported_dtypes({"1.26.3 and below": ("int64",)}, "numpy")
@to_aikit_arrays_and_back
def bincount(x, /, weights=None, minlength=0):
    return aikit.bincount(x, weights=weights, minlength=minlength)
