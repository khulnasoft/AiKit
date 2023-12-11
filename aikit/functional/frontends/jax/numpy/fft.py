# local
import aikit
from aikit.functional.frontends.jax.func_wrapper import to_aikit_arrays_and_back
from aikit.func_wrapper import with_unsupported_dtypes


@to_aikit_arrays_and_back
def fft(a, n=None, axis=-1, norm=None):
    if norm is None:
        norm = "backward"
    return aikit.fft(a, axis, norm=norm, n=n)


@to_aikit_arrays_and_back
def fft2(a, s=None, axes=(-2, -1), norm=None):
    if norm is None:
        norm = "backward"
    return aikit.array(aikit.fft2(a, s=s, dim=axes, norm=norm), dtype=aikit.dtype(a))


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
def fftfreq(n, d=1.0, *, dtype=None):
    if not isinstance(
        n, (int, type(aikit.int8), type(aikit.int16), type(aikit.int32), type(aikit.int64))
    ):
        raise TypeError("n should be an integer")

    dtype = aikit.float64 if dtype is None else aikit.as_aikit_dtype(dtype)

    N = (n - 1) // 2 + 1
    val = 1.0 / (n * d)

    results = aikit.zeros((n,), dtype=dtype)
    results[:N] = aikit.arange(0, N, dtype=dtype)
    results[N:] = aikit.arange(-(n // 2), 0, dtype=dtype)

    return results * val


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
def fftshift(x, axes=None, name=None):
    shape = x.shape

    if axes is None:
        axes = tuple(range(x.ndim))
        shifts = [(dim // 2) for dim in shape]
    elif isinstance(axes, int):
        shifts = shape[axes] // 2
    else:
        shifts = [shape[ax] // 2 for ax in axes]

    roll = aikit.roll(x, shifts, axis=axes)

    return roll


@to_aikit_arrays_and_back
def ifft(a, n=None, axis=-1, norm=None):
    if norm is None:
        norm = "backward"
    return aikit.ifft(a, axis, norm=norm, n=n)


@to_aikit_arrays_and_back
def ifft2(a, s=None, axes=(-2, -1), norm=None):
    if norm is None:
        norm = "backward"
    return aikit.array(aikit.ifft2(a, s=s, dim=axes, norm=norm), dtype=aikit.dtype(a))


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"1.25.2 and below": ("float16", "bfloat16")}, "numpy")
def rfft(a, n=None, axis=-1, norm=None):
    if n is None:
        n = a.shape[axis]
    if norm is None:
        norm = "backward"
    result = aikit.dft(
        a, axis=axis, inverse=False, onesided=False, dft_length=n, norm=norm
    )
    slices = [slice(0, a) for a in result.shape]
    slices[axis] = slice(0, int(aikit.shape(result, as_array=True)[axis] // 2 + 1))
    result = result[tuple(slices)]
    return result
