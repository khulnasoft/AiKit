import aikit
from aikit.func_wrapper import with_unsupported_dtypes
from aikit.functional.frontends.jax.array import Array
import aikit.functional.frontends.jax.numpy as jnp_frontend
from aikit.functional.frontends.jax.func_wrapper import (
    to_aikit_arrays_and_back,
    outputs_to_frontend_arrays,
    handle_jax_dtype,
    inputs_to_aikit_arrays,
)

from aikit.func_wrapper import handle_out_argument
from aikit import with_unsupported_device_and_dtypes

ndarray = Array


@with_unsupported_device_and_dtypes(
    {
        "0.4.21 and below": {
            "cpu": (
                "float16",
                "bflooat16",
                "complex64",
                "complex128",
            ),
            "gpu": (
                "complex64",
                "complex128",
            ),
        }
    },
    "jax",
)
@handle_jax_dtype
@outputs_to_frontend_arrays
def arange(start, stop=None, step=1, dtype=None):
    return aikit.arange(start, stop, step=step, dtype=dtype)


@handle_jax_dtype
@to_aikit_arrays_and_back
def array(object, dtype=None, copy=True, order="K", ndmin=0):
    if order is not None and order != "K":
        raise aikit.utils.exceptions.AikitNotImplementedException(
            "Only implemented for order='K'"
        )
    device = aikit.default_device()
    if aikit.is_array(object):
        device = aikit.dev(object)
    ret = aikit.array(object, dtype=dtype, device=device)
    if aikit.get_num_dims(ret) < ndmin:
        ret = aikit.expand_dims(ret, axis=list(range(ndmin - aikit.get_num_dims(ret))))

    if ret.shape == () and dtype is None:
        return Array(ret, weak_type=True)
    return Array(ret)


@handle_jax_dtype
@to_aikit_arrays_and_back
def asarray(a, dtype=None, order=None):
    return array(a, dtype=dtype, order=order)


@to_aikit_arrays_and_back
def bool_(x):
    return aikit.astype(x, aikit.bool)


@to_aikit_arrays_and_back
def cdouble(x):
    return aikit.astype(x, aikit.complex128)


@to_aikit_arrays_and_back
@handle_out_argument
def compress(condition, a, *, axis=None, out=None):
    condition_arr = aikit.asarray(condition).astype(bool)
    if condition_arr.ndim != 1:
        raise aikit.utils.exceptions.AikitException("Condition must be a 1D array")
    if axis is None:
        arr = aikit.asarray(a).flatten()
        axis = 0
    else:
        arr = aikit.moveaxis(a, axis, 0)
    if condition_arr.shape[0] > arr.shape[0]:
        raise aikit.utils.exceptions.AikitException(
            "Condition contains entries that are out of bounds"
        )
    arr = arr[: condition_arr.shape[0]]
    return aikit.moveaxis(arr[condition_arr], 0, axis)


@to_aikit_arrays_and_back
def copy(a, order=None):
    return array(a, order=order)


@to_aikit_arrays_and_back
def csingle(x):
    return aikit.astype(x, aikit.complex64)


@to_aikit_arrays_and_back
def double(x):
    return aikit.astype(x, aikit.float64)


@handle_jax_dtype
@to_aikit_arrays_and_back
def empty(shape, dtype=None):
    return Array(aikit.empty(shape=shape, dtype=dtype))


@handle_jax_dtype
@to_aikit_arrays_and_back
def empty_like(prototype, dtype=None, shape=None):
    # XLA cannot create uninitialized arrays
    # jax.numpy.empty_like returns an array initialized with zeros.
    if shape:
        return aikit.zeros(shape, dtype=dtype)
    return aikit.zeros_like(prototype, dtype=dtype)


@handle_jax_dtype
@to_aikit_arrays_and_back
def eye(N, M=None, k=0, dtype=None):
    return Array(aikit.eye(N, M, k=k, dtype=dtype))


@to_aikit_arrays_and_back
def from_dlpack(x):
    return aikit.from_dlpack(x)


@to_aikit_arrays_and_back
def frombuffer(buffer, dtype="float", count=-1, offset=0):
    return aikit.frombuffer(buffer, dtype, count, offset)


@to_aikit_arrays_and_back
def full(shape, fill_value, dtype=None):
    return aikit.full(shape, fill_value, dtype=dtype)


@to_aikit_arrays_and_back
def full_like(a, fill_value, dtype=None, shape=None):
    return aikit.full_like(a, fill_value, dtype=dtype)


@to_aikit_arrays_and_back
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    cr = aikit.log(stop / start) / (num - 1 if endpoint else num)
    x = aikit.linspace(
        0, cr * (num - 1 if endpoint else num), num, endpoint=endpoint, axis=axis
    )
    x = aikit.exp(x)
    x = start * x
    x[0] = (start * cr) / cr
    if endpoint:
        x[-1] = stop
    return x.asarray(dtype=dtype)


@handle_jax_dtype
@to_aikit_arrays_and_back
def hstack(tup, dtype=None):
    # TODO: dtype supported in JAX v0.3.20
    return aikit.hstack(tup)


@handle_jax_dtype
@to_aikit_arrays_and_back
def identity(n, dtype=None):
    return aikit.eye(n, dtype=dtype)


@to_aikit_arrays_and_back
def in1d(ar1, ar2, assume_unique=False, invert=False):
    del assume_unique
    ar1_flat = aikit.flatten(ar1)
    ar2_flat = aikit.flatten(ar2)
    if invert:
        return (ar1_flat[:, None] != ar2_flat[None, :]).all(axis=-1)
    else:
        return (ar1_flat[:, None] == ar2_flat[None, :]).any(axis=-1)


@inputs_to_aikit_arrays
def iterable(y):
    return hasattr(y, "__iter__") and y.ndim > 0


@handle_jax_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.21 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    ret = aikit.linspace(start, stop, num, axis=axis, endpoint=endpoint, dtype=dtype)
    if retstep:
        if endpoint:
            num -= 1
        step = aikit.divide(aikit.subtract(stop, start), num)
        return ret, step
    return ret


@handle_jax_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.21 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    if not endpoint:
        interval = (stop - start) / num
        stop -= interval
    return aikit.logspace(start, stop, num, base=base, axis=axis, dtype=dtype)


@to_aikit_arrays_and_back
def meshgrid(*x, copy=True, sparse=False, indexing="xy"):
    # TODO: handle 'copy' argument when aikit.meshgrid supports it
    aikit_meshgrid = aikit.meshgrid(*x, sparse=sparse, indexing=indexing)
    return aikit_meshgrid


@to_aikit_arrays_and_back
def ndim(a):
    if not isinstance(a, aikit.Array):
        return 0
    return aikit.astype(aikit.array(a.ndim), aikit.int64)


@handle_jax_dtype
@to_aikit_arrays_and_back
def ones(shape, dtype=None):
    return Array(aikit.ones(shape, dtype=dtype))


@handle_jax_dtype
@to_aikit_arrays_and_back
def ones_like(a, dtype=None, shape=None):
    if shape:
        return aikit.ones(shape, dtype=dtype)
    return aikit.ones_like(a, dtype=dtype)


@to_aikit_arrays_and_back
def setdiff1d(ar1, ar2, assume_unique=False, *, size=None, fill_value=None):
    fill_value = aikit.array(0 if fill_value is None else fill_value, dtype=ar1.dtype)
    if ar1.size == 0:
        return aikit.full(size or 0, fill_value, dtype=ar1.dtype)
    if not assume_unique:
        val = (
            aikit.to_scalar(aikit.all(ar1))
            if aikit.is_bool_dtype(ar1.dtype)
            else aikit.to_scalar(aikit.min(ar1))
        )
        ar1 = jnp_frontend.unique(ar1, size=size and ar1.size, fill_value=val).aikit_array
    mask = in1d(ar1, ar2, invert=True).aikit_array
    if size is None:
        return ar1[mask]
    else:
        if not (assume_unique):
            # Set mask to zero at locations corresponding to unique() padding.
            n_unique = ar1.size + 1 - (ar1 == ar1[0]).sum(dtype=aikit.int64)
            mask = aikit.where(aikit.arange(ar1.size) < n_unique, mask, False)
        return aikit.where(
            aikit.arange(size) < mask.sum(dtype=aikit.int64),
            ar1[jnp_frontend.where(mask, size=size)[0].aikit_array],
            fill_value,
        )


@to_aikit_arrays_and_back
def single(x):
    return aikit.astype(x, aikit.float32)


@to_aikit_arrays_and_back
def size(a, axis=None):
    aikit.set_default_int_dtype("int64")
    if axis is not None:
        sh = aikit.shape(a)
        return sh[axis]
    return a.size


@to_aikit_arrays_and_back
def triu(m, k=0):
    return aikit.triu(m, k=k)


@to_aikit_arrays_and_back
def vander(x, N=None, increasing=False):
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array")
    if N == 0:
        return aikit.array([], dtype=x.dtype).reshape((x.shape[0], 0))
    else:
        return aikit.vander(x, N=N, increasing=increasing)


@handle_jax_dtype
@to_aikit_arrays_and_back
def zeros(shape, dtype=None):
    return Array(aikit.zeros(shape, dtype=dtype))


@handle_jax_dtype
@to_aikit_arrays_and_back
def zeros_like(a, dtype=None, shape=None):
    if shape:
        return aikit.zeros(shape, dtype=dtype)
    return aikit.zeros_like(a, dtype=dtype)
