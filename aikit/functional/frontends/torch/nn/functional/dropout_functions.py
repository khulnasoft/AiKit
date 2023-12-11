# local
import aikit
from aikit.func_wrapper import with_unsupported_dtypes

from aikit.functional.frontends.torch.func_wrapper import to_aikit_arrays_and_back


# ToDo: this function will be simplified once aikit.alpha_dropout is implemented
@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
def alpha_dropout(input, p=0.5, training=False, inplace=False):
    if p == 0.0 or not training or input.shape == () or input.shape == (0,):
        return input
    neg_saturation = aikit.log1p(aikit.exp(-aikit.square(input)))
    mask = aikit.where(
        aikit.random_uniform(shape=input.shape, device=aikit.dev(input)) < p,
        0.0,
        1.0,
    )
    if inplace:
        aikit.inplace_update(input, mask * input + (1 - mask) * neg_saturation)
        aikit.inplace_update(input, input / aikit.sqrt(1 - p / (1 - p + 1e-5)))
        return input
    else:
        masked = mask * input + (1 - mask) * neg_saturation
        return masked / aikit.sqrt(1 - p / (1 - p + 1e-5))


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def dropout(input, p=0.5, training=True, inplace=False):
    return aikit.dropout(input, p, scale=True, training=training)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def dropout1d(input, p=0.5, training=True, inplace=False):
    if inplace:
        return aikit.dropout1d(input, p, training=training, data_format="NCW", out=input)
    return aikit.dropout1d(input, p, training=training, data_format="NCW")


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("float16",)}, "torch")
def dropout2d(input, p=0.5, training=True, inplace=False):
    if input.ndim < 2:
        raise ValueError("Feature dropout requires at least 2 dimensions in the input")

    ret = aikit.dropout2d(input, p, training=training, data_format="NCHW")
    if inplace:
        aikit.inplace_update(input, ret)
        return input
    return ret


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.1 and below": ("float16", "bfloat16")}, "torch")
def dropout3d(input, p=0.5, training=True, inplace=False):
    if inplace:
        return aikit.dropout3d(
            input, p, training=training, data_format="NDHWC", out=input
        )
    return aikit.dropout3d(input, p, training=training, data_format="NDHWC")
