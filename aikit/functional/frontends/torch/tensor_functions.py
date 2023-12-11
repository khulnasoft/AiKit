# local
import aikit
from aikit.func_wrapper import with_supported_dtypes
from aikit.functional.frontends.torch.func_wrapper import to_aikit_arrays_and_back


@to_aikit_arrays_and_back
def is_complex(input):
    return aikit.is_complex_dtype(input)


@to_aikit_arrays_and_back
def is_floating_point(input):
    return aikit.is_float_dtype(input)


@to_aikit_arrays_and_back
def is_nonzero(input):
    return aikit.nonzero(input)[0].size != 0


@to_aikit_arrays_and_back
def is_tensor(obj):
    return aikit.is_array(obj)


@to_aikit_arrays_and_back
def numel(input):
    return aikit.astype(aikit.array(input.size), aikit.int64)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "int32", "int64")}, "torch"
)
def scatter(input, dim, index, src):
    return aikit.put_along_axis(input, index, src, dim, mode="replace")


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "int32", "int64")}, "torch"
)
def scatter_add(input, dim, index, src):
    return aikit.put_along_axis(input, index, src, dim, mode="sum")


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.1 and below": ("float32", "float64", "int32", "int64")}, "torch"
)
def scatter_reduce(input, dim, index, src, reduce, *, include_self=True):
    mode_mappings = {
        "sum": "sum",
        "amin": "min",
        "amax": "max",
        "prod": "mul",
        "replace": "replace",
    }
    reduce = mode_mappings.get(reduce, reduce)
    return aikit.put_along_axis(input, index, src, dim, mode=reduce)
