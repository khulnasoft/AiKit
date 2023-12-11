# local
import aikit
from aikit.func_wrapper import with_unsupported_dtypes
from aikit.functional.frontends.torch.func_wrapper import to_aikit_arrays_and_back


@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def linear(input, weight, bias=None):
    return aikit.linear(input, weight, bias=bias)
