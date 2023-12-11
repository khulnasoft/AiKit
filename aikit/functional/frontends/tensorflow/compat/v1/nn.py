# local
import aikit
from aikit.functional.frontends.tensorflow.func_wrapper import to_aikit_arrays_and_back
from aikit.func_wrapper import with_supported_dtypes, with_unsupported_dtypes
import aikit.functional.frontends.tensorflow.nn as tf_nn


@with_unsupported_dtypes({"2.15.0 and below": ("float16",)}, "tensorflow")
def depthwise_conv2d(
    input,
    filter,
    strides,
    padding,
    rate=None,
    name=None,
    data_format=None,
    dilations=None,
):
    if rate:
        dilations = rate
    return tf_nn.depthwise_conv2d(
        input,
        filter,
        strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
    )


# should have float16 as well but sqrt doesn't support it
@to_aikit_arrays_and_back
@with_supported_dtypes({"2.15.0 and below": ("float32",)}, "tensorflow")
def fused_batch_norm(
    x,
    scale,
    offset,
    mean=None,
    variance=None,
    epsilon=1e-3,
    data_format="NHWC",
    is_training=True,
    name=None,
    exponential_avg_factor=1.0,
):
    min_epsilon = 1.001e-5
    epsilon = epsilon if epsilon > min_epsilon else min_epsilon

    dims = len(x.shape)
    if data_format[1] == "C":
        if dims == 4:
            x = aikit.permute_dims(x, axes=(0, 2, 3, 1))
        elif dims == 5:
            x = aikit.permute_dims(x, axes=(0, 2, 3, 4, 1))
        else:
            raise aikit.utils.exceptions.AikitException(
                f"input tensor must be of 4 or 5 dimensions, got {dims}"
            )

    scale = scale.astype(aikit.float32)
    offset = offset.astype(aikit.float32)
    old_mean = mean.astype(aikit.float32)
    old_var = variance.astype(aikit.float32)
    x = x.astype(aikit.float32)

    if is_training:
        depth = x.shape[-1]
        rest_size = aikit.prod(x.shape) // depth
        x_rest_by_depth = aikit.reshape(x, [rest_size, depth])
        mean = aikit.mean(x_rest_by_depth, axis=0, keepdims=True)
        variance = aikit.var(x_rest_by_depth, axis=0, keepdims=True)
        y = aikit.reshape(
            scale * (x_rest_by_depth - mean) / aikit.sqrt(variance + epsilon) + offset,
            x.shape,
        )
        float_rest_size = aikit.astype(rest_size, x.dtype)
        variance = (
            variance * float_rest_size / (float_rest_size - 1)
            if rest_size > 1
            else variance
        )
        mean = aikit.reshape(
            mean * exponential_avg_factor + old_mean * (1 - exponential_avg_factor),
            old_mean.shape,
        )
        variance = aikit.reshape(
            variance * exponential_avg_factor + old_var * (1 - exponential_avg_factor),
            old_var.shape,
        )
    else:
        y = scale * (x - old_mean) / aikit.sqrt(old_var + epsilon) + offset

    # permute dimensions back
    if data_format[1] == "C":
        if dims == 4:
            y = aikit.permute_dims(y, axes=(0, 3, 1, 2))
        elif dims == 5:
            y = aikit.permute_dims(y, axes=(0, 4, 1, 2, 3))

    if is_training:
        return y, mean, variance
    else:
        return y, old_mean, old_var


@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {"2.15.0 and below": ("float16",)},
    "tensorflow",
)
def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None, input=None):
    if input is not None and value is not None:
        raise aikit.utils.exceptions.AikitException(
            "Cannot specify both 'value' and 'input'."
        )
    return tf_nn.max_pool2d(
        input if input is not None else value,
        ksize,
        strides,
        padding,
        data_format=data_format,
    )


@with_unsupported_dtypes(
    {
        "2.15.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "tensorflow",
)
def separable_conv2d(
    input,
    depthwise_filter,
    pointwise_filter,
    strides,
    padding,
    rate=None,
    name=None,
    data_format=None,
    dilations=None,
):
    if rate:
        dilations = rate
    return tf_nn.separable_conv2d(
        input,
        depthwise_filter,
        pointwise_filter,
        strides,
        padding,
        data_format=data_format,
        dilations=dilations,
    )
