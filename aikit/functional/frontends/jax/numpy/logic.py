# local
import aikit
from aikit.functional.frontends.jax.func_wrapper import (
    to_aikit_arrays_and_back,
)
from aikit.functional.frontends.jax.numpy import (
    promote_types_of_jax_inputs as promote_jax_arrays,
)
from aikit.utils.exceptions import AikitNotImplementedException
from aikit.func_wrapper import with_unsupported_dtypes


# --- Helpers --- #
# --------------- #


def _packbits_nested_list_padding(arr, pad_length):
    if arr.ndim > 1:
        nested_list = []
        for sub_arr in arr:
            nested_list.append(_packbits_nested_list_padding(sub_arr, pad_length))
        return nested_list
    else:
        return arr.zero_pad(pad_width=[[0, pad_length]])


# --- Main --- #
# ------------ #


@to_aikit_arrays_and_back
def all(a, axis=None, out=None, keepdims=False, *, where=False):
    return aikit.all(a, axis=axis, keepdims=keepdims, out=out)


@to_aikit_arrays_and_back
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    a, b = promote_jax_arrays(a, b)
    return aikit.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@to_aikit_arrays_and_back
def any(a, axis=None, out=None, keepdims=False, *, where=None):
    # TODO: Out not supported
    ret = aikit.any(a, axis=axis, keepdims=keepdims)
    if aikit.is_array(where):
        where = aikit.array(where, dtype=aikit.bool)
        ret = aikit.where(where, ret, aikit.default(None, aikit.zeros_like(ret)))
    return ret


@to_aikit_arrays_and_back
def array_equal(a1, a2, equal_nan: bool) -> bool:
    a1, a2 = promote_jax_arrays(a1, a2)
    if aikit.shape(a1) != aikit.shape(a2):
        return False
    eq = aikit.asarray(a1 == a2)
    if equal_nan:
        eq = aikit.logical_or(eq, aikit.logical_and(aikit.isnan(a1), aikit.isnan(a2)))
    return aikit.all(eq)


@to_aikit_arrays_and_back
def array_equiv(a1, a2) -> bool:
    a1, a2 = promote_jax_arrays(a1, a2)
    try:
        eq = aikit.equal(a1, a2)
    except ValueError:
        # shapes are not broadcastable
        return False
    return aikit.all(eq)


@to_aikit_arrays_and_back
def bitwise_and(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return aikit.bitwise_and(x1, x2)


@to_aikit_arrays_and_back
def bitwise_not(x, /):
    return aikit.bitwise_invert(x)


@to_aikit_arrays_and_back
def bitwise_or(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return aikit.bitwise_or(x1, x2)


@to_aikit_arrays_and_back
def bitwise_xor(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return aikit.bitwise_xor(x1, x2)


@to_aikit_arrays_and_back
def equal(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return aikit.equal(x1, x2)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"0.4.23 and below": ("bfloat16",)}, "jax")
def fromfunction(function, shape, *, dtype=float, **kwargs):
    def canonicalize_shape(shape, context="shape argument"):
        if isinstance(shape, int):
            return (shape,)
        elif isinstance(shape, list):
            return tuple(shape)
        elif isinstance(shape, tuple):
            return shape
        else:
            msg = f"{context} must be an int, list, or tuple, but got {type(shape)}."
            raise TypeError(msg)

    arr = aikit.zeros(shape, dtype=dtype)
    shape = canonicalize_shape(shape)
    # Iterate over the indices of the array
    for indices in aikit.ndindex(shape):
        f_indices = indices
        aikit.set_nest_at_index(
            arr, f_indices, aikit.asarray(function(*indices, **kwargs), dtype=dtype)
        )
    return arr


@to_aikit_arrays_and_back
def greater(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return aikit.greater(x1, x2)


@to_aikit_arrays_and_back
def greater_equal(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return aikit.greater_equal(x1, x2)


@to_aikit_arrays_and_back
def invert(x, /):
    return aikit.bitwise_invert(x)


@to_aikit_arrays_and_back
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    a, b = promote_jax_arrays(a, b)
    return aikit.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@to_aikit_arrays_and_back
def iscomplex(x: any):
    return aikit.bitwise_invert(aikit.isreal(x))


@to_aikit_arrays_and_back
def iscomplexobj(x):
    return aikit.is_complex_dtype(aikit.dtype(x))


@to_aikit_arrays_and_back
def isfinite(x, /):
    return aikit.isfinite(x)


@to_aikit_arrays_and_back
def isin(element, test_elements, assume_unique=False, invert=False):
    return aikit.isin(element, test_elements, assume_unique=assume_unique, invert=invert)


@to_aikit_arrays_and_back
def isinf(x, /):
    return aikit.isinf(x)


@to_aikit_arrays_and_back
def isnan(x, /):
    return aikit.isnan(x)


@to_aikit_arrays_and_back
def isneginf(x, /, out=None):
    return aikit.isinf(x, detect_positive=False, out=out)


@to_aikit_arrays_and_back
def isposinf(x, /, out=None):
    return aikit.isinf(x, detect_negative=False, out=out)


@to_aikit_arrays_and_back
def isreal(x, out=None):
    return aikit.isreal(x, out=out)


@to_aikit_arrays_and_back
def isrealobj(x: any):
    return not aikit.is_complex_dtype(aikit.dtype(x))


@to_aikit_arrays_and_back
def isscalar(x, /):
    return aikit.isscalar(x)


@to_aikit_arrays_and_back
def left_shift(x1, x2):
    # TODO: implement
    raise AikitNotImplementedException()


@to_aikit_arrays_and_back
def less(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return aikit.less(x1, x2)


@to_aikit_arrays_and_back
def less_equal(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return aikit.less_equal(x1, x2)


@to_aikit_arrays_and_back
# known issue in jnp's documentation of arguments
# https://github.com/google/jax/issues/9119
def logical_and(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    if x1.dtype == "complex128" or x2.dtype == "complex128":
        x1 = aikit.astype(x1, aikit.complex128)
        x2 = aikit.astype(x2, aikit.complex128)
    else:
        x1, x2 = promote_jax_arrays(x1, x2)
    return aikit.logical_and(x1, x2)


@to_aikit_arrays_and_back
def logical_not(x, /):
    return aikit.logical_not(x)


@to_aikit_arrays_and_back
def logical_or(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return aikit.logical_or(x1, x2)


@to_aikit_arrays_and_back
def logical_xor(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return aikit.logical_xor(x1, x2)


@to_aikit_arrays_and_back
def not_equal(x1, x2, /):
    x1, x2 = promote_jax_arrays(x1, x2)
    return aikit.not_equal(x1, x2)


@to_aikit_arrays_and_back
def packbits(x, /, *, axis=None, bitorder="big"):
    x = aikit.greater(x, aikit.zeros_like(x)).astype("uint8")
    bits = aikit.arange(8, dtype="uint8")
    if bitorder == "big":
        bits = bits[::-1]
    if axis is None:
        x = aikit.flatten(x)
        axis = 0
    x = aikit.swapaxes(x, axis, -1)

    remainder = x.shape[-1] % 8
    if remainder:
        x = _packbits_nested_list_padding(x, 8 - remainder)
        x = aikit.array(x)

    x = aikit.reshape(x, list(x.shape[:-1]) + [x.shape[-1] // 8, 8])
    bits = aikit.expand_dims(bits, axis=tuple(range(x.ndim - 1)))
    packed = (x << bits).sum(axis=-1).astype("uint8")
    return aikit.swapaxes(packed, axis, -1)


@to_aikit_arrays_and_back
def right_shift(x1, x2, /):
    return aikit.bitwise_right_shift(x1, x2)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"0.4.23 and below": ("bfloat16", "bool")}, "jax")
def setxor1d(ar1, ar2, assume_unique=False):
    common_dtype = aikit.promote_types(aikit.dtype(ar1), aikit.dtype(ar2))
    ar1 = aikit.asarray(ar1, dtype=common_dtype)
    ar2 = aikit.asarray(ar2, dtype=common_dtype)
    if not assume_unique:
        ar1 = aikit.unique_values(ar1)
        ar2 = aikit.unique_values(ar2)
    ar1 = aikit.reshape(ar1, (-1,))
    ar2 = aikit.reshape(ar2, (-1,))
    aux = aikit.concat([ar1, ar2], axis=0)
    if aux.size == 0:
        return aux
    aux = aikit.sort(aux)
    flag = aikit.concat(
        (aikit.array([True]), aikit.not_equal(aux[1:], aux[:-1]), aikit.array([True])), axis=0
    )
    mask = flag[1:] & flag[:-1]
    if aikit.all(aikit.logical_not(mask)):
        ret = aikit.asarray([], dtype=common_dtype)
    else:
        ret = aux[mask]
    return ret


alltrue = all
sometrue = any
