# global
import aikit
from aikit.functional.frontends.paddle.func_wrapper import (
    to_aikit_arrays_and_back,
)


@to_aikit_arrays_and_back
def imag(x):
    return aikit.imag(x)


@to_aikit_arrays_and_back
def is_complex(x):
    return aikit.is_complex_dtype(x)


@to_aikit_arrays_and_back
def is_floating_point(x):
    return aikit.is_float_dtype(x)


@to_aikit_arrays_and_back
def is_integer(x):
    return aikit.is_int_dtype(x)


@to_aikit_arrays_and_back
def rank(input):
    return aikit.get_num_dims(input)


@to_aikit_arrays_and_back
def real(x):
    return aikit.real(x)
