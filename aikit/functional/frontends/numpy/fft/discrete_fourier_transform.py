import aikit
from aikit.functional.frontends.numpy.func_wrapper import to_aikit_arrays_and_back
from aikit.func_wrapper import with_unsupported_dtypes


_SWAP_DIRECTION_MAP = {
    None: "forward",
    "backward": "forward",
    "ortho": "ortho",
    "forward": "backward",
}


# --- Helpers --- #
# --------------- #


def _swap_direction(norm):
    try:
        return _SWAP_DIRECTION_MAP[norm]
    except KeyError:
        raise ValueError(
            f'Invalid norm value {norm}; should be "backward", "ortho" or "forward".'
        ) from None


# --- Main --- #
# ------------ #


@to_aikit_arrays_and_back
def fft(a, n=None, axis=-1, norm=None):
    return aikit.fft(aikit.astype(a, aikit.complex128), axis, norm=norm, n=n)


@with_unsupported_dtypes({"1.26.2 and below": ("int",)}, "numpy")
@to_aikit_arrays_and_back
def fftfreq(n, d=1.0):
    if not isinstance(
        n, (int, type(aikit.int8), type(aikit.int16), type(aikit.int32), type(aikit.int64))
    ):
        raise TypeError("n should be an integer")

    N = (n - 1) // 2 + 1
    val = 1.0 / (n * d)
    results = aikit.empty((n,), dtype=int)

    p1 = aikit.arange(0, N, dtype=int)
    results[:N] = p1
    p2 = aikit.arange(-(n // 2), 0, dtype=int)
    results[N:] = p2

    return results * val


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"1.26.2 and below": ("float16",)}, "numpy")
def fftshift(x, axes=None):
    x = aikit.asarray(x)

    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [(dim // 2) for dim in x.shape]
    elif isinstance(
        axes,
        (int, type(aikit.uint8), type(aikit.uint16), type(aikit.uint32), type(aikit.uint64)),
    ):
        shift = x.shape[axes] // 2
    else:
        shift = [(x.shape[ax] // 2) for ax in axes]

    roll = aikit.roll(x, shift, axis=axes)

    return roll


@to_aikit_arrays_and_back
def ifft(a, n=None, axis=-1, norm=None):
    a = aikit.array(a, dtype=aikit.complex128)
    if norm is None:
        norm = "backward"
    return aikit.ifft(a, axis, norm=norm, n=n)


@with_unsupported_dtypes({"1.24.3 and below": ("float16",)}, "numpy")
@to_aikit_arrays_and_back
def ifft2(a, s=None, axes=(-2, -1), norm=None):
    a = aikit.asarray(a, dtype=aikit.complex128)
    a = aikit.ifftn(a, s=s, axes=axes, norm=norm)
    return a


@with_unsupported_dtypes({"1.24.3 and below": ("float16",)}, "numpy")
@to_aikit_arrays_and_back
def ifftn(a, s=None, axes=None, norm=None):
    a = aikit.asarray(a, dtype=aikit.complex128)
    a = aikit.ifftn(a, s=s, axes=axes, norm=norm)
    return a


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"1.26.2 and below": ("float16",)}, "numpy")
def ifftshift(x, axes=None):
    x = aikit.asarray(x)

    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(
        axes,
        (int, type(aikit.uint8), type(aikit.uint16), type(aikit.uint32), type(aikit.uint64)),
    ):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]

    roll = aikit.roll(x, shift, axis=axes)

    return roll


@with_unsupported_dtypes({"1.26.2 and below": ("float16",)}, "numpy")
@to_aikit_arrays_and_back
def ihfft(a, n=None, axis=-1, norm=None):
    if n is None:
        n = a.shape[axis]
    norm = _swap_direction(norm)
    output = aikit.conj(rfft(a, n, axis, norm=norm).aikit_array)
    return output


@with_unsupported_dtypes({"1.26.2 and below": ("float16",)}, "numpy")
@to_aikit_arrays_and_back
def rfft(a, n=None, axis=-1, norm=None):
    if norm is None:
        norm = "backward"
    a = aikit.array(a, dtype=aikit.float64)
    return aikit.dft(a, axis=axis, inverse=False, onesided=True, dft_length=n, norm=norm)


@to_aikit_arrays_and_back
def rfftfreq(n, d=1.0):
    if not isinstance(
        n, (int, type(aikit.int8), type(aikit.int16), type(aikit.int32), type(aikit.int64))
    ):
        raise TypeError("n should be an integer")

    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = aikit.arange(0, N, dtype=int)
    return results * val


@with_unsupported_dtypes({"1.24.3 and below": ("float16",)}, "numpy")
@to_aikit_arrays_and_back
def rfftn(a, s=None, axes=None, norm=None):
    a = aikit.asarray(a, dtype=aikit.complex128)
    return aikit.rfftn(a, s=s, axes=axes, norm=norm)
