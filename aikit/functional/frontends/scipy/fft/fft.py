# global
from aikit.functional.frontends.numpy.func_wrapper import to_aikit_arrays_and_back

import aikit


# dct
@to_aikit_arrays_and_back
def dct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, orthogonalize=None):
    return aikit.dct(x, type=type, n=n, axis=axis, norm=norm)


# fft
@to_aikit_arrays_and_back
def fft(x, n=None, axis=-1, norm="backward", overwrite_x=False):
    return aikit.fft(x, axis, norm=norm, n=n)


@to_aikit_arrays_and_back
def fft2(x, s=None, axes=(-2, -1), norm=None, overwrite_x=False):
    return aikit.fft2(x, s=s, dim=axes, norm=norm)


# idct
@to_aikit_arrays_and_back
def idct(x, type=2, n=None, axis=-1, norm=None, overwrite_x=False, orthogonalize=None):
    inverse_type = {1: 1, 2: 3, 3: 2, 4: 4}[type]
    return aikit.dct(x, type=inverse_type, n=n, axis=axis, norm=norm)


# ifft
@to_aikit_arrays_and_back
def ifft(x, n=None, axis=-1, norm=None, overwrite_x=False):
    return aikit.ifft(x, axis, norm=norm, n=n)


@to_aikit_arrays_and_back
def ifftn(
    x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *, plan=None
):
    return aikit.ifftn(x, s=s, axes=axes, norm=norm)


@to_aikit_arrays_and_back
def rfftn(
    x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *, plan=None
):
    return aikit.rfftn(x, s=s, axes=axes, norm=norm)
