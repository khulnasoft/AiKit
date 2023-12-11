# global

# local
import aikit
from aikit import with_unsupported_dtypes, with_supported_dtypes
from aikit.functional.frontends.torch.func_wrapper import to_aikit_arrays_and_back


# --- Helpers --- #
# --------------- #


def _handle_padding_shape(padding, n, mode):
    padding = tuple(
        [
            (padding[i * 2], padding[i * 2 + 1])
            for i in range(int(len(padding) / 2) - 1, -1, -1)
        ]
    )
    if mode == "circular":
        padding = padding + ((0, 0),) * (n - len(padding))
    else:
        padding = ((0, 0),) * (n - len(padding)) + padding
    if mode == "circular":
        padding = tuple(list(padding)[::-1])
    return padding


# --- Main --- #
# ------------ #


@with_unsupported_dtypes({"2.1.1 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def affine_grid(theta, size, align_corners=False):
    if len(size) == 4:
        N, C, H, W = size
        base_grid = aikit.empty((N, H, W, 3))
        if align_corners:
            base_grid[:, :, :, 0] = aikit.linspace(-1, 1, W)
            base_grid[:, :, :, 1] = aikit.expand_dims(aikit.linspace(-1, 1, H), axis=-1)
            base_grid[:, :, :, 2] = aikit.full((H, W), 1)
            grid = aikit.matmul(base_grid.view((N, H * W, 3)), theta.swapaxes(1, 2))
            return grid.view((N, H, W, 2))
        else:
            base_grid[:, :, :, 0] = aikit.linspace(-1, 1, W) * (W - 1) / W
            base_grid[:, :, :, 1] = aikit.expand_dims(
                aikit.linspace(-1, 1, H) * (H - 1) / H, axis=-1
            )
            base_grid[:, :, :, 2] = aikit.full((H, W), 1)
        grid = aikit.matmul(base_grid.view((N, H * W, 3)), aikit.swapaxes(theta, 1, 2))
        return grid.view((N, H, W, 2))
    else:
        N, C, D, H, W = size
        base_grid = aikit.empty((N, D, H, W, 4))
        if align_corners:
            base_grid[:, :, :, :, 0] = aikit.linspace(-1, 1, W)
            base_grid[:, :, :, :, 1] = aikit.expand_dims(aikit.linspace(-1, 1, H), axis=-1)
            base_grid[:, :, :, :, 2] = aikit.expand_dims(
                aikit.expand_dims(aikit.linspace(-1, 1, D), axis=-1), axis=-1
            )
            base_grid[:, :, :, :, 3] = aikit.full((D, H, W), 1)
            grid = aikit.matmul(base_grid.view((N, D * H * W, 4)), theta.swapaxes(1, 2))
            return grid.view((N, D, H, W, 3))
        else:
            base_grid[:, :, :, :, 0] = aikit.linspace(-1, 1, W) * (W - 1) / W
            base_grid[:, :, :, :, 1] = aikit.expand_dims(
                aikit.linspace(-1, 1, H) * (H - 1) / H, axis=-1
            )
            base_grid[:, :, :, :, 2] = aikit.expand_dims(
                aikit.expand_dims(aikit.linspace(-1, 1, D) * (D - 1) / D, axis=-1), axis=-1
            )
            base_grid[:, :, :, :, 3] = aikit.full((D, H, W), 1)
            grid = aikit.matmul(base_grid.view((N, D * H * W, 4)), theta.swapaxes(1, 2))
            return grid.view((N, D, H, W, 3))


def bicubic_interp(x, t, alpha=-0.75):
    n, h, w = t.shape
    coeffs = []
    coeffs.append(aikit.reshape(cubic_conv2(alpha, t + 1), (n, 1, h, w)))
    coeffs.append(aikit.reshape(cubic_conv1(alpha, t), (n, 1, h, w)))
    coeffs.append(aikit.reshape(cubic_conv1(alpha, 1 - t), (n, 1, h, w)))
    coeffs.append(aikit.reshape(cubic_conv2(alpha, 2 - t), (n, 1, h, w)))
    return x[0] * coeffs[0] + x[1] * coeffs[1] + x[2] * coeffs[2] + x[3] * coeffs[3]


def cubic_conv1(A, x):
    return ((A + 2) * x - (A + 3)) * x * x + 1


def cubic_conv2(A, x):
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A


@with_supported_dtypes({"2.1.1 and below": ("float32", "float64")}, "torch")
@to_aikit_arrays_and_back
def grid_sample(
    input, grid, mode="bilinear", padding_mode="zeros", align_corners=False
):
    input_clone = aikit.copy_array(input)
    grid_clone = aikit.copy_array(grid)

    if aikit.get_num_dims(input_clone) == 4:  # sample from 2D images
        n, c, h, w = input_clone.shape
        n, to_h, to_w, gc = grid_clone.shape

        # Un-normalize 2D grid
        if align_corners:  # to range[0, size - 1]
            grid_clone[..., 0] = ((grid_clone[..., 0] + 1) / 2) * (w - 1)
            grid_clone[..., 1] = ((grid_clone[..., 1] + 1) / 2) * (h - 1)

        elif not align_corners:  # to range[0.5, size - 0.5]
            grid_clone[..., 0] = ((grid_clone[..., 0] + 1) * w - 1) / 2
            grid_clone[..., 1] = ((grid_clone[..., 1] + 1) * h - 1) / 2

        batch_coor = aikit.reshape(aikit.arange(n), (-1, 1))
        batch_coor = aikit.repeat(batch_coor, to_h * to_w, axis=1)
        batch_coor = aikit.reshape(batch_coor, (n, to_h, to_w))
        padding = [(0, 0) for _ in range(2)] + [(4, 4) for _ in range(2)]
        input_clone = aikit.pad(input_clone, padding, mode="constant", constant_values=0)

        if mode == "bicubic":
            grid_floor = aikit.floor(grid_clone)
            distance = grid_clone - grid_floor

            tx, ty = distance[..., 0], distance[..., 1]

            grid_floor -= 1
            grid_floor = [
                grid_sample_padding(
                    grid_floor + i, padding_mode, align_corners, borders=[w, h]
                )
                for i in range(4)
            ]

            w_cubic = [
                aikit.astype(grid_floor[i][..., 0] + 4, aikit.int64) for i in range(4)
            ]
            h_cubic = [
                aikit.astype(grid_floor[i][..., 1] + 4, aikit.int64) for i in range(4)
            ]

            coeffs = [
                bicubic_interp(
                    [
                        aikit.permute_dims(
                            input_clone[batch_coor, :, h_cubic[i], w_cubic[0]],
                            (0, 3, 1, 2),
                        ),
                        aikit.permute_dims(
                            input_clone[batch_coor, :, h_cubic[i], w_cubic[1]],
                            (0, 3, 1, 2),
                        ),
                        aikit.permute_dims(
                            input_clone[batch_coor, :, h_cubic[i], w_cubic[2]],
                            (0, 3, 1, 2),
                        ),
                        aikit.permute_dims(
                            input_clone[batch_coor, :, h_cubic[i], w_cubic[3]],
                            (0, 3, 1, 2),
                        ),
                    ],
                    tx,
                )
                for i in range(4)
            ]
            return bicubic_interp(coeffs, ty)

        else:
            grid_clone = grid_sample_padding(
                grid_clone, padding_mode, align_corners, borders=[w, h]
            )

        if mode == "bilinear":
            grid_clone += 4
            w_coor = aikit.reshape(grid_clone[..., 0], (n, to_h, to_w))
            h_coor = aikit.reshape(grid_clone[..., 1], (n, to_h, to_w))

            w0 = aikit.astype(aikit.floor(w_coor), aikit.int64)
            h0 = aikit.astype(aikit.floor(h_coor), aikit.int64)
            w1 = w0 + 1
            h1 = h0 + 1

            v00 = aikit.permute_dims(input_clone[batch_coor, :, h0, w0], (0, 3, 1, 2))
            v01 = aikit.permute_dims(input_clone[batch_coor, :, h0, w1], (0, 3, 1, 2))
            v10 = aikit.permute_dims(input_clone[batch_coor, :, h1, w0], (0, 3, 1, 2))
            v11 = aikit.permute_dims(input_clone[batch_coor, :, h1, w1], (0, 3, 1, 2))

            alpha = aikit.reshape(w_coor - w0, (n, 1, to_h, to_w))
            beta = aikit.reshape(h_coor - h0, (n, 1, to_h, to_w))

            alpha = aikit.astype(alpha, aikit.float32)
            beta = aikit.astype(beta, aikit.float32)

            v0 = v00 * (1 - alpha) + v01 * alpha
            v1 = v10 * (1 - alpha) + v11 * alpha

            return v0 * (1 - beta) + v1 * beta

        elif mode == "nearest":
            w_coor = aikit.reshape(grid_clone[..., 0], (n, to_h, to_w))
            h_coor = aikit.reshape(grid_clone[..., 1], (n, to_h, to_w))

            w_coor = aikit.astype(aikit.round(w_coor), aikit.int64) + 4
            h_coor = aikit.astype(aikit.round(h_coor), aikit.int64) + 4
            return aikit.permute_dims(
                input_clone[batch_coor, :, h_coor, w_coor], (0, 3, 1, 2)
            )

        else:
            raise aikit.exceptions.AikitError(f"Not supported mode {mode}")

    elif aikit.get_num_dims(input_clone) == 5:  # sample from 3D images
        n, c, d, h, w = input_clone.shape
        n, to_d, to_h, to_w, gc = grid_clone.shape

        # Un-normalize 3D grid
        if align_corners:  # to range[0, size - 1]
            grid_clone[..., 0] = ((grid_clone[..., 0] + 1) / 2) * (w - 1)
            grid_clone[..., 1] = ((grid_clone[..., 1] + 1) / 2) * (h - 1)
            grid_clone[..., 2] = ((grid_clone[..., 2] + 1) / 2) * (d - 1)
        elif not align_corners:  # to range[0.5, size - 0.5]
            grid_clone[..., 0] = ((grid_clone[..., 0] + 1) * w - 1) / 2
            grid_clone[..., 1] = ((grid_clone[..., 1] + 1) * h - 1) / 2
            grid_clone[..., 2] = ((grid_clone[..., 2] + 1) * d - 1) / 2

        batch_coor = aikit.reshape(aikit.arange(n), (-1, 1))
        batch_coor = aikit.repeat(batch_coor, to_d * to_h * to_w, axis=1)
        batch_coor = aikit.reshape(batch_coor, (n, to_d, to_h, to_w))
        padding = [(0, 0) for _ in range(2)] + [(3, 3) for _ in range(3)]
        input_clone = aikit.pad(input_clone, padding, mode="constant", constant_values=0)

        grid_clone = grid_sample_padding(
            grid_clone, padding_mode, align_corners, borders=[w, h, d]
        )

        if mode == "bilinear":
            grid_clone += 3
            w_coor = aikit.reshape(grid_clone[..., 0], (n, to_d, to_h, to_w))
            h_coor = aikit.reshape(grid_clone[..., 1], (n, to_d, to_h, to_w))
            d_coor = aikit.reshape(grid_clone[..., 2], (n, to_d, to_h, to_w))

            w0 = aikit.astype(aikit.floor(w_coor), aikit.int64)
            h0 = aikit.astype(aikit.floor(h_coor), aikit.int64)
            d0 = aikit.astype(aikit.floor(d_coor), aikit.int64)
            w1 = w0 + 1
            h1 = h0 + 1
            d1 = d0 + 1

            v000 = aikit.permute_dims(
                input_clone[batch_coor, :, d0, h0, w0], (0, 4, 1, 2, 3)
            )  # tnw
            v001 = aikit.permute_dims(
                input_clone[batch_coor, :, d0, h0, w1], (0, 4, 1, 2, 3)
            )  # tne
            v010 = aikit.permute_dims(
                input_clone[batch_coor, :, d0, h1, w0], (0, 4, 1, 2, 3)
            )  # tsw
            v011 = aikit.permute_dims(
                input_clone[batch_coor, :, d0, h1, w1], (0, 4, 1, 2, 3)
            )  # tse
            v100 = aikit.permute_dims(
                input_clone[batch_coor, :, d1, h0, w0], (0, 4, 1, 2, 3)
            )  # bnw
            v101 = aikit.permute_dims(
                input_clone[batch_coor, :, d1, h0, w1], (0, 4, 1, 2, 3)
            )  # bne
            v110 = aikit.permute_dims(
                input_clone[batch_coor, :, d1, h1, w0], (0, 4, 1, 2, 3)
            )  # bsw
            v111 = aikit.permute_dims(
                input_clone[batch_coor, :, d1, h1, w1], (0, 4, 1, 2, 3)
            )  # bse

            alpha = aikit.reshape(w_coor - w0, (n, 1, to_d, to_h, to_w))
            beta = aikit.reshape(h_coor - h0, (n, 1, to_d, to_h, to_w))
            gamma = aikit.reshape(d_coor - d0, (n, 1, to_d, to_h, to_w))

            alpha = aikit.astype(alpha, aikit.float32)
            beta = aikit.astype(beta, aikit.float32)
            gamma = aikit.astype(gamma, aikit.float32)

            v = (alpha * beta * gamma) * v111
            v += ((1 - alpha) * beta * gamma) * v110
            v += (alpha * (1 - beta) * gamma) * v101
            v += ((1 - alpha) * (1 - beta) * gamma) * v100

            v += (alpha * beta * (1 - gamma)) * v011
            v += ((1 - alpha) * beta * (1 - gamma)) * v010
            v += (alpha * (1 - beta) * (1 - gamma)) * v001
            v += ((1 - alpha) * (1 - beta) * (1 - gamma)) * v000
            return v

        elif mode == "nearest":
            ceil_mask = grid_clone % 1 == 0.5
            grid_clone[ceil_mask] = aikit.astype(
                aikit.ceil(grid_clone[ceil_mask]), aikit.int64
            )

            w_coor = aikit.reshape(grid_clone[..., 0], (n, to_d, to_h, to_w))
            h_coor = aikit.reshape(grid_clone[..., 1], (n, to_d, to_h, to_w))
            d_coor = aikit.reshape(grid_clone[..., 2], (n, to_d, to_h, to_w))

            w_coor = aikit.astype(aikit.round(w_coor), aikit.int64) + 3
            h_coor = aikit.astype(aikit.round(h_coor), aikit.int64) + 3
            d_coor = aikit.astype(aikit.round(d_coor), aikit.int64) + 3
            return aikit.permute_dims(
                input_clone[batch_coor, :, d_coor, h_coor, w_coor], (0, 4, 1, 2, 3)
            )

        elif mode == "bicubic":
            raise aikit.exceptions.AikitError("Bicubic is not support in 3D grid sampling")

    else:
        raise aikit.exceptions.AikitError(f"Not supported input shape {input_clone.shape}")


def grid_sample_padding(grid, padding_mode, align_corners, borders=None):
    if padding_mode == "reflection":
        if align_corners:
            for idx, border in enumerate(borders):
                grid[..., idx] = reflect(grid[..., idx], 0, 2 * (border - 1))
                grid[..., idx] = aikit.clip(grid[..., idx], 0, border - 1)

        else:
            for idx, border in enumerate(borders):
                grid[..., idx] = reflect(grid[..., idx], -1, 2 * border - 1)
                grid[..., idx] = aikit.clip(grid[..., idx], 0, border - 1)

    elif padding_mode == "border":
        for idx, border in enumerate(borders):
            grid[..., idx] = aikit.clip(grid[..., idx], 0, border - 1)

    masks = []
    for idx, border in enumerate(borders):
        masks.append(aikit.bitwise_or(grid[..., idx] < -4, grid[..., idx] > border + 2))
        borders[idx] += 1

    zeros_mask = masks[0]
    for i in range(1, len(borders)):
        zeros_mask = aikit.bitwise_or(zeros_mask, masks[i])

    if grid[zeros_mask].shape[0] > 0:
        grid[zeros_mask] = aikit.array(borders)
    return grid


@with_unsupported_dtypes(
    {
        "2.1.1 and below": (
            "bfloat16",
            "float16",
        )
    },
    "torch",
)
@to_aikit_arrays_and_back
def interpolate(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
    antialias=False,
):
    if (
        mode not in ["linear", "bilinear", "bicubic", "trilinear"]
        and align_corners is not None
    ):
        raise aikit.utils.exceptions.AikitException(
            "align_corners option can only be set with the interpolating"
            f"modes: linear | bilinear | bicubic | trilinear (got {mode})"
        )
    aikit.utils.assertions.check_elem_in_list(
        aikit.get_num_dims(input),
        range(3, 6),
        message=(
            "Input Error: Only 3D, 4D and 5D input Tensors supported (got"
            f" {aikit.get_num_dims(input)}D) for the modes: nearest | linear | bilinear |"
            f" bicubic | trilinear | area | nearest-exact (got {mode})"
        ),
    )
    return aikit.interpolate(
        input,
        size,
        mode=mode,
        scale_factor=scale_factor,
        recompute_scale_factor=recompute_scale_factor,
        align_corners=True if align_corners else False,
        antialias=antialias,
    )


@to_aikit_arrays_and_back
def pad(input, pad, mode="constant", value=0):
    mode_dict = {
        "constant": "constant",
        "reflect": "reflect",
        "replicate": "edge",
        "circular": "wrap",
    }
    if mode not in mode_dict:
        raise ValueError(f"Unsupported padding mode: {mode}")
    pad = _handle_padding_shape(pad, len(input.shape), mode)
    return aikit.pad(input, pad, mode=mode_dict[mode], constant_values=value)


@to_aikit_arrays_and_back
def pixel_shuffle(input, upscale_factor):
    input_shape = aikit.shape(input)

    aikit.utils.assertions.check_equal(
        aikit.get_num_dims(input),
        4,
        message=(
            "pixel_shuffle expects 4D input, but got input with sizes"
            f" {str(input_shape)}"
        ),
        as_array=False,
    )
    b = input_shape[0]
    c = input_shape[1]
    h = input_shape[2]
    w = input_shape[3]
    upscale_factor_squared = upscale_factor * upscale_factor
    aikit.utils.assertions.check_equal(
        c % upscale_factor_squared,
        0,
        message="pixel_shuffle expects input channel to be divisible by square "
        + "of upscale_factor, but got input with sizes "
        + str(input_shape)
        + ", upscale_factor="
        + str(upscale_factor)
        + ", and self.size(1)="
        + str(c)
        + " is not divisible by "
        + str(upscale_factor_squared),
        as_array=False,
    )
    oc = int(c / upscale_factor_squared)
    oh = h * upscale_factor
    ow = w * upscale_factor

    input_reshaped = aikit.reshape(input, (b, oc, upscale_factor, upscale_factor, h, w))
    return aikit.reshape(
        aikit.permute_dims(input_reshaped, (0, 1, 4, 2, 5, 3)), (b, oc, oh, ow)
    )


@to_aikit_arrays_and_back
def pixel_unshuffle(input, downscale_factor):
    input_shape = aikit.shape(input)

    aikit.utils.assertions.check_equal(
        aikit.get_num_dims(input),
        4,
        message=(
            f"pixel_unshuffle expects 4D input, but got input with sizes {input_shape}"
        ),
        as_array=False,
    )

    b = input_shape[0]
    c = input_shape[1]
    h = input_shape[2]
    w = input_shape[3]
    downscale_factor_squared = downscale_factor * downscale_factor

    aikit.utils.assertions.check_equal(
        [h % downscale_factor, w % downscale_factor],
        [0, 0],  # Assert h % downscale_factor == 0 and w % downscale_factor == 0
        message=(
            "pixel_unshuffle expects input height and width to be divisible by "
            f"downscale_factor, but got input with sizes {input_shape}"
            f", downscale_factor= {downscale_factor}"
            f", and either self.size(2)= {h}"
            f" or self.size(3)= {w}"
            f" is not divisible by {downscale_factor}"
        ),
        as_array=False,
    )
    oc = c * downscale_factor_squared
    oh = int(h / downscale_factor)
    ow = int(w / downscale_factor)

    input_reshaped = aikit.reshape(
        input, (b, c, oh, downscale_factor, ow, downscale_factor)
    )
    return aikit.reshape(
        aikit.permute_dims(input_reshaped, (0, 1, 3, 5, 2, 4)), (b, oc, oh, ow)
    )


def reflect(x, low2, high2):
    min = low2 / 2
    span = (high2 - low2) / 2
    x = aikit.abs(x - min)
    frac_in = aikit.abs(x / span)
    extra = (frac_in - aikit.floor(frac_in)) * aikit.abs(span)
    flips = aikit.floor(x / span)
    x[flips % 2 == 0] = (extra + min)[flips % 2 == 0]
    x[flips % 2 != 0] = (span - extra + min)[flips % 2 != 0]
    return x


@with_unsupported_dtypes({"2.1.1 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def upsample(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
):
    return interpolate(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )


@with_unsupported_dtypes({"2.1.1 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def upsample_bilinear(input, size=None, scale_factor=None):
    return interpolate(
        input, size=size, scale_factor=scale_factor, mode="bilinear", align_corners=True
    )


@with_unsupported_dtypes({"2.1.1 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def upsample_nearest(input, size=None, scale_factor=None):
    return interpolate(input, size=size, scale_factor=scale_factor, mode="nearest")
