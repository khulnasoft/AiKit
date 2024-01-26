# global
import aikit

from aikit.functional.frontends.numpy import promote_types_of_numpy_inputs
from aikit import with_unsupported_dtypes
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_numpy_casting,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)


# --- Helpers --- #
# --------------- #


@handle_numpy_out
@handle_numpy_dtype
@to_aikit_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _matmul(
    x1, x2, /, out=None, *, casting="same_kind", order="K", dtype=None, subok=True
):
    return aikit.matmul(x1, x2, out=out)


# --- Main --- #
# ------------ #


@to_aikit_arrays_and_back
def cross(a, b, *, axisa=-1, axisb=-1, axisc=-1, axis=None):
    return aikit.cross(a, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)


@handle_numpy_out
@to_aikit_arrays_and_back
def dot(a, b, out=None):
    a, b = promote_types_of_numpy_inputs(a, b)
    return aikit.matmul(a, b, out=out)


@handle_numpy_out
@to_aikit_arrays_and_back
def einsum(
    subscripts,
    *operands,
    out=None,
    dtype=None,
    order="K",
    casting="safe",
    optimize=False,
):
    return aikit.einsum(subscripts, *operands, out=out)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def inner(a, b, /):
    a, b = promote_types_of_numpy_inputs(a, b)
    return aikit.inner(a, b)


@to_aikit_arrays_and_back
def kron(a, b):
    a, b = promote_types_of_numpy_inputs(a, b)
    return aikit.kron(a, b)


@to_aikit_arrays_and_back
def matrix_power(a, n):
    return aikit.matrix_power(a, n)


@with_unsupported_dtypes({"2.0.0 and below": ("float16",)}, "torch")
@handle_numpy_out
@to_aikit_arrays_and_back
def multi_dot(arrays, *, out=None):
    return aikit.multi_dot(arrays, out=out)


@handle_numpy_out
@to_aikit_arrays_and_back
def outer(a, b, out=None):
    a, b = promote_types_of_numpy_inputs(a, b)
    return aikit.outer(a, b, out=out)


@to_aikit_arrays_and_back
def tensordot(a, b, axes=2):
    return aikit.tensordot(a, b, axes=axes)


@to_aikit_arrays_and_back
def tensorsolve(a, b, axes=2):
    return aikit.tensorsolve(a, b, axes=axes)
