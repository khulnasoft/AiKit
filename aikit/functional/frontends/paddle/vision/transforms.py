import aikit
from aikit.func_wrapper import (
    with_supported_dtypes,
    with_unsupported_device_and_dtypes,
)
from ..tensor.tensor import Tensor
from aikit.functional.frontends.paddle.func_wrapper import (
    to_aikit_arrays_and_back,
)


# --- Helpers --- #
# --------------- #


def _blend_images(img1, img2, ratio):
    # TODO: aikit.check_float(img1) returns False for aikit array
    # TODO: when lerp supports int type and when the above issue is fixed,
    # replace this with aikit.check_float(img1)
    max_value = (
        1.0 if aikit.dtype(img1) == "float32" or aikit.dtype(img1) == "float64" else 255.0
    )
    return aikit.astype(
        aikit.lerp(img2, img1, float(ratio)).clip(0, max_value), aikit.dtype(img1)
    )


# helpers
def _get_image_c_axis(data_format):
    if data_format.lower() == "chw":
        return -3
    elif data_format.lower() == "hwc":
        return -1


def _get_image_num_channels(img, data_format):
    return aikit.shape(img)[_get_image_c_axis(data_format)]


def _hsv_to_rgb(img):
    h, s, v = img[0], img[1], img[2]
    f = h * 6.0
    i = aikit.floor(f)
    f = f - i
    i = aikit.astype(i, aikit.int32) % 6

    p = aikit.clip(v * (1.0 - s), 0.0, 1.0)
    q = aikit.clip(v * (1.0 - s * f), 0.0, 1.0)
    t = aikit.clip(v * (1.0 - s * (1.0 - f)), 0.0, 1.0)

    mask = aikit.astype(
        aikit.equal(
            aikit.expand_dims(i, axis=-3),
            aikit.reshape(aikit.arange(6, dtype=aikit.dtype(i)), (-1, 1, 1)),
        ),
        aikit.dtype(img),
    )
    matrix = aikit.stack(
        [
            aikit.stack([v, q, p, p, t, v], axis=-3),
            aikit.stack([t, v, v, q, p, p], axis=-3),
            aikit.stack([p, p, t, v, v, q], axis=-3),
        ],
        axis=-4,
    )
    return aikit.einsum("...ijk, ...xijk -> ...xjk", mask, matrix)


def _rgb_to_hsv(img):
    maxc = aikit.max(img, axis=-3)
    minc = aikit.min(img, axis=-3)

    is_equal = aikit.equal(maxc, minc)
    one_divisor = aikit.ones_like(maxc)
    c_delta = maxc - minc
    s = c_delta / aikit.where(is_equal, one_divisor, maxc)

    r, g, b = img[0], img[1], img[2]
    c_delta_divisor = aikit.where(is_equal, one_divisor, c_delta)

    rc = (maxc - r) / c_delta_divisor
    gc = (maxc - g) / c_delta_divisor
    bc = (maxc - b) / c_delta_divisor

    hr = aikit.where((maxc == r), bc - gc, aikit.zeros_like(maxc))
    hg = aikit.where(
        ((maxc == g) & (maxc != r)),
        rc - bc + 2.0,
        aikit.zeros_like(maxc),
    )
    hb = aikit.where(
        ((maxc != r) & (maxc != g)),
        gc - rc + 4.0,
        aikit.zeros_like(maxc),
    )

    h = (hr + hg + hb) / 6.0 + 1.0
    h = h - aikit.trunc(h)

    return aikit.stack([h, s, maxc], axis=-3)


# --- Main --- #
# ------------ #


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def adjust_brightness(img, brightness_factor):
    assert brightness_factor >= 0, "brightness_factor should be non-negative."
    assert _get_image_num_channels(img, "CHW") in [
        1,
        3,
    ], "channels of input should be either 1 or 3."

    extreme_target = aikit.zeros_like(img)
    return _blend_images(img, extreme_target, brightness_factor)


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64", "uint8")}, "paddle")
@to_aikit_arrays_and_back
def adjust_hue(img, hue_factor):
    assert -0.5 <= hue_factor <= 0.5, "hue_factor should be in range [-0.5, 0.5]"

    channels = _get_image_num_channels(img, "CHW")

    if channels == 1:
        return img
    elif channels == 3:
        if aikit.dtype(img) == "uint8":
            img = aikit.astype(img, "float32") / 255.0

        img_hsv = _rgb_to_hsv(img)
        h, s, v = img_hsv[0], img_hsv[1], img_hsv[2]

        h = h + hue_factor
        h = h - aikit.floor(h)

        img_adjusted = _hsv_to_rgb(aikit.stack([h, s, v], axis=-3))

    else:
        raise ValueError("channels of input should be either 1 or 3.")

    return img_adjusted


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def hflip(img):
    img = aikit.array(img)
    return aikit.flip(img, axis=-1)


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
def normalize(img, mean, std, data_format="CHW", to_rgb=False):
    if aikit.is_array(img):
        if data_format == "HWC":
            permuted_axes = [2, 0, 1]
        else:
            permuted_axes = [0, 1, 2]

        img_np = aikit.permute(img, permuted_axes)
        normalized_img = aikit.divide(aikit.subtract(img_np, mean), std)
        return normalized_img
    else:
        raise ValueError("Unsupported input format")


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def pad(img, padding, fill=0, padding_mode="constant"):
    dim_size = img.ndim
    if not hasattr(padding, "__len__"):
        if dim_size == 2:
            trans_padding = ((padding, padding), (padding, padding))
        elif dim_size == 3:
            trans_padding = ((0, 0), (padding, padding), (padding, padding))
    elif len(padding) == 2:
        if dim_size == 2:
            trans_padding = ((padding[1], padding[1]), (padding[0], padding[0]))
        elif dim_size == 3:
            trans_padding = ((0, 0), (padding[1], padding[1]), (padding[0], padding[0]))
    elif len(padding) == 4:
        if dim_size == 2:
            trans_padding = ((padding[1], padding[3]), (padding[0], padding[2]))
        elif dim_size == 3:
            trans_padding = ((0, 0), (padding[1], padding[3]), (padding[0], padding[2]))
    else:
        raise ValueError("padding can only be 1D with size 1, 2, 4 only")

    if padding_mode in ["constant", "edge", "reflect", "symmetric"]:
        return aikit.pad(img, trans_padding, mode=padding_mode, constant_values=fill)
    else:
        raise ValueError("Unsupported padding_mode")


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def to_tensor(pic, data_format="CHW"):
    array = aikit.array(pic)
    return Tensor(array)


@with_unsupported_device_and_dtypes(
    {
        "2.6.0 and below": {
            "cpu": ("int8", "uint8", "int16", "float16", "bfloat16", "bool")
        }
    },
    "paddle",
)
@to_aikit_arrays_and_back
def vflip(img, data_format="CHW"):
    if data_format.lower() == "chw":
        axis = -2
    elif data_format.lower() == "hwc":
        axis = -3
    return aikit.flip(img, axis=axis)
