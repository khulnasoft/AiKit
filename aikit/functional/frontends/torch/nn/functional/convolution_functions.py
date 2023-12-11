# global
import math

# local
import aikit
from aikit.func_wrapper import with_unsupported_dtypes
from aikit.functional.frontends.torch.func_wrapper import to_aikit_arrays_and_back


# --- Helpers --- #
# --------------- #


def _conv(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    dims = len(input.shape) - 2
    _valid_shapes(input, weight, bias, stride, padding, groups)

    if isinstance(padding, str):
        padding = padding.upper()
    else:
        if isinstance(padding, int):
            padding = [*[(padding, padding) for _ in range(dims)]]
        else:
            padding = [*[(p, p) for p in padding]]

    ret = aikit.conv_general_dilated(
        input,
        weight,
        stride,
        padding,
        dims=dims,
        data_format="channel_first",
        filter_format="channel_first",
        dilations=dilation,
        feature_group_count=groups,
        bias=bias,
    )
    return ret


def _conv_transpose(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    dims = len(input.shape) - 2
    weight = aikit.permute_dims(weight, axes=(*range(2, dims + 2), 0, 1))
    for i in range(dims):
        weight = aikit.flip(weight, axis=i)
    padding, output_padding, stride, dilation = map(
        lambda x: [x] * dims if isinstance(x, int) else x,
        [padding, output_padding, stride, dilation],
    )

    pad_widths = [
        (
            (weight.shape[i] - 1) * dilation[i]
            + max([output_padding[i] - padding[i], 0]),
        )
        * 2
        for i in range(dims)
    ]

    ret = aikit.conv_general_dilated(
        input,
        weight,
        1,
        pad_widths,
        dims=dims,
        data_format="channel_first",
        feature_group_count=groups,
        x_dilations=stride,
        dilations=dilation,
        bias=bias,
    )
    unpad_slice = (slice(None),) * 2
    for i in range(dims):
        unpad_slice += (
            slice(
                max([padding[i] - (dilation[i] // 2), padding[i], output_padding[i]]),
                ret.shape[2 + i] - padding[i] + output_padding[i] + (dilation[i] // 2),
                1,
            ),
        )
    ret = ret[unpad_slice]
    return ret


def _get_transpose_pad(padding, output_padding, dims):
    (
        padding,
        output_padding,
    ) = map(
        lambda x: [x] * dims if isinstance(x, int) else x, [padding, output_padding]
    )
    asymmetric_padding = [
        [pad, pad - output_pad] for pad, output_pad in zip(padding, output_padding)
    ]
    return asymmetric_padding


def _valid_shapes(input, weight, bias, stride, padding, groups, transpose=False):
    in_channels = input.shape[1]
    out_channels = weight.shape[0] if not transpose else weight.shape[1] * groups

    aikit.utils.assertions.check_equal(
        in_channels % groups,
        0,
        message="in_channels must be divisible by groups",
        as_array=False,
    )
    aikit.utils.assertions.check_equal(
        out_channels % groups,
        0,
        message="out_channels must be divisible by groups",
        as_array=False,
    )

    if bias is not None:
        aikit.utils.assertions.check_equal(
            bias.shape[0],
            out_channels,
            message="bias must be same shape as out_channels",
            as_array=False,
        )

    if padding == "same":
        if isinstance(stride, int):
            aikit.utils.assertions.check_equal(
                stride,
                1,
                message="padding cannot be 'same' for stride > 1",
                as_array=False,
            )
        else:
            for i in stride:
                aikit.utils.assertions.check_equal(
                    i,
                    1,
                    message="padding cannot be 'same' for stride > 1",
                    as_array=False,
                )

    if not transpose:
        in_channels_by_groups = weight.shape[1]
        aikit.utils.assertions.check_equal(
            in_channels,
            in_channels_by_groups * groups,
            message="in_channels must be consistent between input and weight",
            as_array=False,
        )
    else:
        aikit.utils.assertions.check_equal(
            in_channels,
            weight.shape[0],
            message="in_channels must be consistent between input and weight",
            as_array=False,
        )


# --- Main --- #
# ------------ #


@with_unsupported_dtypes({"2.1.1 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return _conv(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@with_unsupported_dtypes({"2.1.1 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return _conv(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@with_unsupported_dtypes({"2.1.1 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return _conv(
        input,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@with_unsupported_dtypes({"2.1.1 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def conv_transpose1d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    if aikit.current_backend_str() in ["torch"]:
        # this backend supports explicit padding, no need for conv_general_dilated
        return aikit.conv_general_transpose(
            input,
            weight,
            stride,
            _get_transpose_pad(padding, output_padding, 1),
            dims=1,
            filter_format="channel_first",
            data_format="channel_first",
            dilations=dilation,
            feature_group_count=groups,
            bias=bias,
        )
    else:
        return _conv_transpose(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )


@with_unsupported_dtypes({"2.1.1 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    if aikit.current_backend_str() in ["torch", "tensorflow"]:
        # these two backends support explicit padding, no need for conv_general_dilated
        return aikit.conv_general_transpose(
            input,
            weight,
            stride,
            _get_transpose_pad(padding, output_padding, 2),
            dims=2,
            filter_format="channel_first",
            data_format="channel_first",
            dilations=dilation,
            feature_group_count=groups,
            bias=bias,
        )
    else:
        return _conv_transpose(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )


@with_unsupported_dtypes({"2.1.1 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def conv_transpose3d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    if aikit.current_backend_str() in ["torch"]:
        # this backend supports explicit padding, no need for conv_general_dilated
        return aikit.conv_general_transpose(
            input,
            weight,
            stride,
            _get_transpose_pad(padding, output_padding, 3),
            dims=3,
            filter_format="channel_first",
            data_format="channel_first",
            dilations=dilation,
            feature_group_count=groups,
            bias=bias,
        )
    else:
        return _conv_transpose(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )


@to_aikit_arrays_and_back
def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    orig_ndim = input.ndim
    if orig_ndim == 2:
        input = aikit.expand_dims(input, axis=0)
    elif orig_ndim != 3:
        raise aikit.utils.exceptions.AikitException(
            "only 2D or batched 3D inputs are supported"
        )
    stride = [stride] * 2 if isinstance(stride, int) else stride
    dilation = [dilation] * 2 if isinstance(dilation, int) else dilation
    padding = [padding] * 2 if isinstance(padding, int) else padding
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    output_size = [output_size] * 2 if isinstance(output_size, int) else output_size
    input_shape = [
        (output_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1)
        // stride[i]
        + 1
        for i in range(2)
    ]
    n_batches = input.shape[0]
    n_channels = input.shape[1] // math.prod(kernel_size)
    output = aikit.zeros((n_batches, n_channels, *output_size), dtype=input.dtype)
    output_padded = aikit.zero_pad(
        output,
        ((0, 0), (0, 0), (padding[0],) * 2, (padding[1],) * 2),
    )
    k = 0
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            i_in = i * stride[0]
            j_in = j * stride[1]
            patch = input[:, :, k].reshape((n_batches, n_channels, *kernel_size))
            output_padded[
                :,
                :,
                i_in : i_in + kernel_size[0] * dilation[0] : dilation[0],
                j_in : j_in + kernel_size[1] * dilation[1] : dilation[1],
            ] += patch
            k += 1
    ret = aikit.array(
        output_padded[
            :,
            :,
            padding[0] : output_padded.shape[2] - padding[0],
            padding[1] : output_padded.shape[3] - padding[1],
        ]
    )
    if orig_ndim == 2:
        return aikit.squeeze(ret, axis=0)
    return ret


@to_aikit_arrays_and_back
def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    # TODO: refactor this function to use aikit.sliding_window, but ensure that the
    # function is transpilable to all backends with varying batch size(see issue #25796)
    if input.ndim != 4:
        raise aikit.utils.exceptions.AikitException("only batched 4D inputs are supported")
    stride = [stride] * 2 if isinstance(stride, int) else stride
    dilation = [dilation] * 2 if isinstance(dilation, int) else dilation
    padding = [padding] * 2 if isinstance(padding, int) else padding
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    output_shape = [
        (input.shape[i + 2] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1)
        // stride[i]
        + 1
        for i in range(2)
    ]
    ret = aikit.zeros((*input.shape[0:2], *kernel_size, *output_shape), dtype=input.dtype)
    input_padded = aikit.zero_pad(
        input,
        ((0, 0), (0, 0), (padding[0],) * 2, (padding[1],) * 2),
    )
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            i_in = i * stride[0]
            j_in = j * stride[1]
            ret[:, :, :, :, i, j] = input_padded[
                :,
                :,
                i_in : i_in + kernel_size[0] * dilation[0] : dilation[0],
                j_in : j_in + kernel_size[1] * dilation[1] : dilation[1],
            ]
    return aikit.reshape(
        ret, (input.shape[0], input.shape[1] * math.prod(kernel_size), -1)
    )
