# global
import aikit
from aikit.func_wrapper import with_supported_dtypes
from aikit.functional.frontends.paddle.func_wrapper import (
    to_aikit_arrays_and_back,
)


@with_supported_dtypes(
    {"2.6.0 and below": ("complex64", "complex128")},
    "paddle",
)
@to_aikit_arrays_and_back
def fft(x, n=None, axis=-1.0, norm="backward", name=None):
    ret = aikit.fft(aikit.astype(x, "complex128"), axis, norm=norm, n=n)
    return aikit.astype(ret, x.dtype)


@with_supported_dtypes(
    {
        "2.6.0 and below": (
            "int32",
            "int64",
            "float32",
            "float64",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
def fft2(x, s=None, axes=(-2, -1), norm="backward", name=None):
    if axes is None:
        axes = (-2, -1)
    ret = aikit.fft2(x, s=s, dim=axes, norm=norm)
    return ret


@with_supported_dtypes(
    {
        "2.6.0 and below": (
            "int32",
            "int64",
            "float32",
            "float64",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
def fftfreq(n, d=1.0, dtype=None, name=None):
    if d * n == 0:
        raise ValueError("d or n should not be 0.")

    if dtype is None:
        dtype = aikit.default_dtype()
    val = 1.0 / (n * d)
    pos_max = (n + 1) // 2
    neg_max = n // 2
    indices = aikit.arange(-neg_max, pos_max, dtype=dtype)
    indices = aikit.roll(indices, -neg_max)
    return aikit.multiply(indices, val)


@with_supported_dtypes(
    {
        "2.6.0 and below": (
            "int32",
            "int64",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
def fftshift(x, axes=None, name=None):
    shape = x.shape

    if axes is None:
        axes = tuple(range(x.ndim))
        shifts = [(dim // 2) for dim in shape]
    elif isinstance(axes, int):
        shifts = shape[axes] // 2
    else:
        shifts = aikit.concat([shape[ax] // 2 for ax in axes])

    roll = aikit.roll(x, shifts, axis=axes)

    return roll


@with_supported_dtypes(
    {"2.6.0 and below": ("complex64", "complex128")},
    "paddle",
)
@to_aikit_arrays_and_back
def hfft(x, n=None, axes=-1, norm="backward", name=None):
    """Compute the FFT of a signal that has Hermitian symmetry, resulting in a
    real spectrum."""
    # Determine the input shape and axis length
    input_shape = x.shape
    input_len = input_shape[axes]

    # Calculate n if not provided
    if n is None:
        n = 2 * (input_len - 1)

    # Perform the FFT along the specified axis
    result = aikit.fft(x, axes, n=n, norm=norm)

    return aikit.real(result)


@with_supported_dtypes(
    {"2.6.0 and below": "complex64"},
    "paddle",
)
@to_aikit_arrays_and_back
def hfft2(x, s=None, axis=(-2, -1), norm="backward"):
    # check if the input tensor x is a hermitian complex
    if not aikit.allclose(aikit.conj(aikit.matrix_transpose(x)), x):
        raise ValueError("Input tensor x must be Hermitian complex.")

    fft_result = aikit.fft2(x, s=s, dim=axis, norm=norm)

    # Depending on the norm, apply scaling and normalization
    if norm == "forward":
        fft_result /= aikit.sqrt(aikit.prod(aikit.shape(fft_result)))
    elif norm == "ortho":
        fft_result /= aikit.sqrt(aikit.prod(aikit.shape(x)))

    return aikit.real(fft_result)  # Return the real part of the result


@with_supported_dtypes(
    {"2.6.0 and below": ("complex64", "complex128")},
    "paddle",
)
@to_aikit_arrays_and_back
def ifft(x, n=None, axis=-1.0, norm="backward", name=None):
    ret = aikit.ifft(aikit.astype(x, "complex128"), axis, norm=norm, n=n)
    return aikit.astype(ret, x.dtype)


@with_supported_dtypes(
    {"2.6.0 and below": ("complex64", "complex128")},
    "paddle",
)
@to_aikit_arrays_and_back
def ifftn(x, s=None, axes=None, norm="backward", name=None):
    ret = aikit.ifftn(aikit.astype(x, "complex128"), s=s, axes=axes, norm=norm)
    return aikit.astype(ret, x.dtype)


@with_supported_dtypes(
    {
        "2.6.0 and below": (
            "int32",
            "int64",
            "float32",
            "float64",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
def ifftshift(x, axes=None, name=None):
    shape = x.shape

    if axes is None:
        axes = tuple(range(x.ndim))
        shifts = [-(dim // 2) for dim in shape]
    elif isinstance(axes, int):
        shifts = -(shape[axes] // 2)
    else:
        shifts = aikit.concat([-shape[ax] // 2 for ax in axes])

    roll = aikit.roll(x, shifts, axis=axes)

    return roll


@with_supported_dtypes(
    {
        "2.6.0 and below": (
            "int32",
            "int64",
            "float32",
            "float64",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
def ihfft2(x, s=None, axes=(-2, -1), norm="backward", name=None):
    # check if the input array is two-dimensional and real
    if len(aikit.array(x).shape) != 2 or aikit.is_complex_dtype(x):
        raise ValueError("input must be a two-dimensional real array")

    # cast the input to the same float64 type so that there are no backend issues
    x_ = aikit.astype(x, aikit.float64)

    ihfft2_result = 0
    # Compute the complex conjugate of the 2-dimensional discrete Fourier Transform
    if norm == "backward":
        ihfft2_result = aikit.conj(aikit.rfftn(x_, s=s, axes=axes, norm="forward"))
    if norm == "forward":
        ihfft2_result = aikit.conj(aikit.rfftn(x_, s=s, axes=axes, norm="backward"))
    if norm == "ortho":
        ihfft2_result = aikit.conj(aikit.rfftn(x_, s=s, axes=axes, norm="ortho"))

    if x.dtype in [aikit.float32, aikit.int32, aikit.int64]:
        return aikit.astype(ihfft2_result, aikit.complex64)
    if x.dtype == aikit.float64:
        return aikit.astype(ihfft2_result, aikit.complex128)


@to_aikit_arrays_and_back
def ihfftn(x, s=None, axes=None, norm="backward", name=None):
    # cast the input to the same float64 type so that there are no backend issues
    x_ = aikit.astype(x, aikit.float64)

    ihfftn_result = 0
    # Compute the complex conjugate of the 2-dimensional discrete Fourier Transform
    if norm == "backward":
        ihfftn_result = aikit.conj(aikit.rfftn(x_, s=s, axes=axes, norm="forward"))
    if norm == "forward":
        ihfftn_result = aikit.conj(aikit.rfftn(x_, s=s, axes=axes, norm="backward"))
    if norm == "ortho":
        ihfftn_result = aikit.conj(aikit.rfftn(x_, s=s, axes=axes, norm="ortho"))

    if x.dtype in [aikit.float32, aikit.int32, aikit.int64]:
        return aikit.astype(ihfftn_result, aikit.complex64)
    if x.dtype == aikit.float64:
        return aikit.astype(ihfftn_result, aikit.complex128)


@with_supported_dtypes(
    {"2.6.0 and below": ("complex64", "complex128")},
    "paddle",
)
@to_aikit_arrays_and_back
def irfft(x, n=None, axis=-1.0, norm="backward", name=None):
    if n is None:
        n = 2 * (x.shape[axis] - 1)

    pos_freq_terms = aikit.take_along_axis(x, range(n // 2 + 1), axis)
    neg_freq_terms = aikit.conj(pos_freq_terms[1:-1][::-1])
    combined_freq_terms = aikit.concat((pos_freq_terms, neg_freq_terms), axis=axis)
    time_domain = aikit.ifft(combined_freq_terms, axis, norm=norm, n=n)
    if aikit.isreal(x):
        time_domain = aikit.real(time_domain)
    return time_domain


@with_supported_dtypes(
    {
        "2.6.0 and below": (
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    },
    "paddle",
)
@to_aikit_arrays_and_back
def irfft2(x, s=None, axes=(-2, -1), norm="backward"):
    # Handle values if None
    if s is None:
        s = x.shape
    if axes is None:
        axes = (-2, -1)

    # Calculate the normalization factor 'n' based on the shape 's'
    n = aikit.prod(aikit.array(s))

    result = aikit.ifftn(x, axes=axes[0], norm=norm)

    # Normalize the result based on the 'norm' parameter
    if norm == "backward":
        result /= n
    elif norm == "forward":
        result *= n
    elif norm == "ortho":
        result /= aikit.sqrt(n)
    return result


@with_supported_dtypes(
    {"2.6.0 and below": ("complex64", "complex128")},
    "paddle",
)
@to_aikit_arrays_and_back
def irfftn(x, s=None, axes=None, norm="backward", name=None):
    x = aikit.array(x)

    if axes is None:
        axes = list(range(len(x.shape)))

    include_last_axis = len(x.shape) - 1 in axes

    if s is None:
        s = [
            x.shape[axis] if axis != (len(x.shape) - 1) else 2 * (x.shape[axis] - 1)
            for axis in axes
        ]

    real_result = x
    remaining_axes = [axis for axis in axes if axis != (len(x.shape) - 1)]

    if remaining_axes:
        real_result = aikit.ifftn(
            x,
            s=[s[axes.index(axis)] for axis in remaining_axes],
            axes=remaining_axes,
            norm=norm,
        )

    if include_last_axis:
        axis = len(x.shape) - 1
        size = s[axes.index(axis)]
        freq_domain = aikit.moveaxis(real_result, axis, -1)
        slices = [slice(None)] * aikit.get_num_dims(freq_domain)
        slices[-1] = slice(0, size // 2 + 1)
        pos_freq_terms = freq_domain[tuple(slices)]
        slices[-1] = slice(1, -1)
        neg_freq_terms = aikit.conj(pos_freq_terms[tuple(slices)][..., ::-1])
        combined_freq_terms = aikit.concat((pos_freq_terms, neg_freq_terms), axis=-1)
        real_result = aikit.ifftn(combined_freq_terms, s=[size], axes=[-1], norm=norm)
        real_result = aikit.moveaxis(real_result, -1, axis)

    if aikit.is_complex_dtype(x.dtype):
        output_dtype = "float32" if x.dtype == "complex64" else "float64"
    else:
        output_dtype = "float32"

    result_t = aikit.astype(real_result, output_dtype)
    return result_t


@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
@to_aikit_arrays_and_back
def rfft(x, n=None, axis=-1, norm="backward", name=None):
    return aikit.dft(x, axis=axis, inverse=False, onesided=True, dft_length=n, norm=norm)


@to_aikit_arrays_and_back
def rfftfreq(n, d=1.0, dtype=None, name=None):
    dtype = aikit.default_dtype()
    val = 1.0 / (n * d)
    pos_max = n // 2 + 1
    indices = aikit.arange(0, pos_max, dtype=dtype)
    return indices * val
