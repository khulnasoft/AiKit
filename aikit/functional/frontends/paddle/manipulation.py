# global
import aikit
from aikit.functional.frontends.paddle.func_wrapper import (
    to_aikit_arrays_and_back,
)
from aikit.func_wrapper import (
    with_unsupported_dtypes,
    with_supported_dtypes,
    with_supported_device_and_dtypes,
)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def abs(x, name=None):
    return aikit.abs(x)


@with_supported_dtypes(
    {"2.5.2 and below": ("bool", "float32", "float64", "int32", "int64")},
    "paddle",
)
@to_aikit_arrays_and_back
def broadcast_to(x, shape, name=None):
    return aikit.broadcast_to(x, shape)


@with_supported_dtypes(
    {
        "2.5.2 and below": (
            "bool",
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
            "uint8",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
def cast(x, dtype):
    return aikit.astype(x, dtype)


@with_unsupported_dtypes({"2.5.2 and below": ("int8", "int16")}, "paddle")
@to_aikit_arrays_and_back
def concat(x, axis, name=None):
    return aikit.concat(x, axis=axis)


@with_supported_dtypes(
    {"2.5.2 and below": ("bool", "float32", "float64", "int32", "int64")},
    "paddle",
)
@to_aikit_arrays_and_back
def expand(x, shape, name=None):
    return aikit.expand(x, shape)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("int8", "uint8", "int16", "float16")},
    "paddle",
)
@to_aikit_arrays_and_back
def flip(x, axis, name=None):
    return aikit.flip(x, axis=axis)


@with_supported_dtypes(
    {"2.5.2 and below": ("bool", "float32", "float64", "int32", "int64")},
    "paddle",
)
@to_aikit_arrays_and_back
def gather(params, indices, axis=-1, batch_dims=0, name=None):
    return aikit.gather(params, indices, axis=axis, batch_dims=batch_dims)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("int8", "uint8", "int16", "uint16", "float16", "bfloat16")},
    "paddle",
)
@to_aikit_arrays_and_back
def gather_nd(x, index, name=None):
    return aikit.gather_nd(x, index)


@to_aikit_arrays_and_back
def put_along_axis(arr, indices, values, axis, reduce="assign"):
    result = aikit.put_along_axis(arr, indices, values, axis)
    return result


@with_supported_dtypes(
    {"2.5.2 and below": ("int32", "int64", "float32", "float64")},
    "paddle",
)
@to_aikit_arrays_and_back
def repeat_interleave(x, repeats, axis=None, name=None):
    return aikit.repeat(x, repeats, axis=axis)


@to_aikit_arrays_and_back
def reshape(x, shape, name=None):
    return aikit.reshape(x, shape)


@with_supported_dtypes(
    {
        "2.5.0 and below": (
            "float32",
            "float64",
            "int32",
            "int64",
            "complex64",
            "complex128",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
def roll(x, shifts, axis=None, name=None):
    return aikit.roll(x, shifts, axis=axis)


@with_supported_device_and_dtypes(
    {
        "2.5.2 and above": {
            "cpu": (
                "bool",
                "int32",
                "int64",
                "float32",
                "float64",
            ),
            "gpu": ("float16",),
        },
    },
    "paddle",
)
@to_aikit_arrays_and_back
def rot90(x, k=1, axes=(0, 1), name=None):
    return aikit.rot90(x, k=k, axes=axes)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("int16", "complex64", "complex128")},
    "paddle",
)
@to_aikit_arrays_and_back
def split(x, num_or_sections, axis=0, name=None):
    return aikit.split(x, num_or_size_splits=num_or_sections, axis=axis)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("float16", "bfloat16", "int8", "int16")},
    "paddle",
)
@to_aikit_arrays_and_back
def squeeze(x, axis=None, name=None):
    return aikit.squeeze(x, axis=axis)


@to_aikit_arrays_and_back
def stack(x, axis=0, name=None):
    return aikit.stack(x, axis=axis)


def take_along_axis(arr, indices, axis):
    return aikit.take_along_axis(arr, indices, axis)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("int8", "uint8", "int16", "float16")},
    "paddle",
)
@to_aikit_arrays_and_back
def tile(x, repeat_times, name=None):
    return aikit.tile(x, repeats=repeat_times)


@to_aikit_arrays_and_back
def tolist(x):
    return aikit.to_list(x)


@with_supported_dtypes(
    {"2.5.2 and below": ("bool", "int32", "int64", "float16", "float32", "float64")},
    "paddle",
)
@to_aikit_arrays_and_back
def unbind(input, axis=0):
    shape = list(input.shape)
    num_splits = shape[axis]
    shape.pop(axis)
    return tuple(x.reshape(tuple(shape)) for x in split(input, num_splits, axis=axis))


@with_supported_dtypes(
    {"2.5.2 and below": ("bool", "int32", "int64", "float16", "float32", "float64")},
    "paddle",
)
@to_aikit_arrays_and_back
def unique_consecutive(x, axis=0):
    return aikit.unique_consecutive(x, axis=axis)


@with_supported_dtypes(
    {
        "2.5.2 and below": (
            "float32",
            "float64",
            "int32",
            "int64",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
def unstack(x, axis=0, name=None):
    return aikit.unstack(x, axis=axis)


absolute = abs
