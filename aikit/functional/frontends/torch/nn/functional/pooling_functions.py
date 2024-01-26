# global
from functools import reduce

# local
import aikit
from aikit import with_unsupported_dtypes
from aikit.functional.frontends.torch.func_wrapper import (
    to_aikit_arrays_and_back,
)


@with_unsupported_dtypes(
    {
        "2.1.2 and below": (
            "bfloat16",
            "float16",
        )
    },
    "torch",
)
@to_aikit_arrays_and_back
def adaptive_avg_pool1d(input, output_size):
    return aikit.adaptive_avg_pool1d(input, output_size)


@with_unsupported_dtypes(
    {
        "2.1.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
@to_aikit_arrays_and_back
def adaptive_avg_pool2d(input, output_size):
    return aikit.adaptive_avg_pool2d(input, output_size, data_format="NCHW")


@with_unsupported_dtypes(
    {
        "2.1.2 and below": (
            "bfloat16",
            "float16",
        )
    },
    "torch",
)
@to_aikit_arrays_and_back
def adaptive_max_pool2d(
    input,
    output_size,
    return_indices=False,
):
    # ToDo: Add return_indices once superset is implemented
    return aikit.adaptive_max_pool2d(input, output_size)


@with_unsupported_dtypes(
    {"2.1.2 and below": ("float16",)},
    "torch",
)
@to_aikit_arrays_and_back
def avg_pool1d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
):
    if not isinstance(padding, int):
        padding = [(pad, pad) for pad in padding]
    return aikit.avg_pool1d(
        input,
        kernel_size,
        stride if stride is not None else kernel_size,
        padding,
        data_format="NCW",
        count_include_pad=count_include_pad,
        ceil_mode=ceil_mode,
    )


@with_unsupported_dtypes(
    {"2.1.2 and below": ("float16",)},
    "torch",
)
@to_aikit_arrays_and_back
def avg_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    if not isinstance(padding, int):
        padding = [(pad, pad) for pad in padding]
    return aikit.avg_pool2d(
        input,
        kernel_size,
        stride if stride is not None else kernel_size,
        padding,
        data_format="NCHW",
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@with_unsupported_dtypes(
    {"2.1.2 and below": ("float16", "bfloat16")},
    "torch",
)
@to_aikit_arrays_and_back
def avg_pool3d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    if not isinstance(padding, int):
        padding = [(pad, pad) for pad in padding]
    return aikit.avg_pool3d(
        input,
        kernel_size,
        stride if stride is not None else kernel_size,
        padding,
        data_format="NCDHW",
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


@with_unsupported_dtypes(
    {
        "2.1.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
@to_aikit_arrays_and_back
def lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    data_format = "NCW"
    padding = "VALID"
    if stride is None:
        stride = kernel_size
    if not isinstance(kernel_size, int):
        kernel_mul = reduce(lambda x, y: x * y, kernel_size)
    else:
        kernel_mul = kernel_size

    out = aikit.avg_pool1d(
        aikit.pow(input, norm_type),
        kernel_size,
        stride,
        padding,
        data_format=data_format,
        ceil_mode=ceil_mode,
    )
    p = 1.0 / norm_type if norm_type != 0 else 1.0
    return aikit.pow(aikit.multiply(out, kernel_mul), p)


@with_unsupported_dtypes(
    {
        "2.1.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
@to_aikit_arrays_and_back
def lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    data_format = "NCHW"
    padding = "VALID"
    if stride is None:
        stride = kernel_size
    out = aikit.avg_pool2d(
        aikit.pow(input, norm_type),
        kernel_size,
        stride,
        padding,
        data_format=data_format,
        ceil_mode=ceil_mode,
    )
    if not isinstance(kernel_size, int):
        kernel_mul = reduce(lambda x, y: x * y, kernel_size)
    else:
        kernel_mul = kernel_size
    p = aikit.divide(1.0, norm_type) if norm_type != 0 else 1.0
    return aikit.pow(aikit.multiply(out, kernel_mul), p).astype(input.dtype)


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def max_pool1d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    if stride is None:
        stride = kernel_size
    if not isinstance(padding, int):
        padding = [(pad, pad) for pad in padding]
    return aikit.max_pool1d(
        input,
        kernel_size,
        stride,
        padding,
        data_format="NCW",
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def max_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    if stride is None:
        stride = kernel_size
    if not isinstance(padding, int):
        padding = [(pad, pad) for pad in padding]
    return aikit.max_pool2d(
        input,
        kernel_size,
        stride,
        padding,
        data_format="NCHW",
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def max_pool3d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    if stride is None:
        stride = kernel_size
    if not isinstance(padding, int):
        padding = [(pad, pad) for pad in padding]
    return aikit.max_pool3d(
        input,
        kernel_size,
        stride,
        padding,
        data_format="NCDHW",
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
