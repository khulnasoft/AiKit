"""Collection of Jax general functions, wrapped to fit Aikit syntax and
signature."""

# global
import jax
import numpy as np
import jax.numpy as jnp
from numbers import Number
from operator import mul
from functools import reduce as _reduce
from typing import Optional, Union, Sequence, Callable, Tuple
import multiprocessing as _multiprocessing
import importlib


# local
import aikit
from aikit.func_wrapper import with_unsupported_dtypes
from aikit.functional.backends.jax.device import _to_array, _to_device
from aikit.functional.aikit.general import _broadcast_to
from aikit.functional.backends.jax import JaxArray, NativeArray
from aikit.utils.exceptions import _check_inplace_update_support
from . import backend_version


def container_types():
    flat_mapping_spec = importlib.util.find_spec(
        "FlatMapping", "haiku._src.data_structures"
    )
    if not flat_mapping_spec:
        from haiku._src.data_structures import FlatMapping
    else:
        FlatMapping = importlib.util.module_from_spec(flat_mapping_spec)
    return [FlatMapping]


def current_backend_str() -> str:
    return "jax"


def is_native_array(x, /, *, exclusive=False):
    if exclusive:
        return isinstance(x, NativeArray)
    return isinstance(
        x,
        (
            NativeArray,
            jax.interpreters.ad.JVPTracer,
            jax.core.ShapedArray,
            jax.interpreters.partial_eval.DynamicJaxprTracer,
        ),
    )


def _mask_to_index(query, x):
    if query.shape != x.shape:
        if len(query.shape) > len(x.shape):
            raise aikit.exceptions.AikitException("too many indices")
        elif not len(query.shape):
            query = jnp.tile(query, x.shape[0])
    return jnp.where(query)


def get_item(
    x: JaxArray,
    /,
    query: Union[JaxArray, Tuple],
    *,
    copy: Optional[bool] = None,
) -> JaxArray:
    if aikit.is_array(query) and aikit.is_bool_dtype(query):
        if not len(query.shape):
            if not query:
                return jnp.array([], dtype=x.dtype)
            else:
                return jnp.expand_dims(x, 0)
        query = _mask_to_index(query, x)
    elif isinstance(query, list):
        query = (query,)
    return x.__getitem__(query)


def set_item(
    x: JaxArray,
    query: Union[JaxArray, Tuple],
    val: JaxArray,
    /,
    *,
    copy: Optional[bool] = False,
) -> JaxArray:
    if aikit.is_array(query) and aikit.is_bool_dtype(query):
        query = _mask_to_index(query, x)
    expected_shape = x[query].shape
    if aikit.is_array(val):
        val = _broadcast_to(val, expected_shape)._data
    ret = x.at[query].set(val)
    if copy:
        return ret
    return aikit.inplace_update(x, _to_device(ret))


def array_equal(x0: JaxArray, x1: JaxArray, /) -> bool:
    return bool(jnp.array_equal(x0, x1))


@with_unsupported_dtypes({"0.4.23 and below": ("bfloat16",)}, backend_version)
def to_numpy(x: JaxArray, /, *, copy: bool = True) -> np.ndarray:
    if copy:
        return np.array(_to_array(x))
    else:
        return np.asarray(_to_array(x))


def to_scalar(x: JaxArray, /) -> Number:
    if isinstance(x, Number):
        return x
    else:
        return _to_array(x).item()


def to_list(x: JaxArray, /) -> list:
    return _to_array(x).tolist()


def gather(
    params: JaxArray,
    indices: JaxArray,
    /,
    *,
    axis: int = -1,
    batch_dims: int = 0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    axis %= len(params.shape)
    batch_dims %= len(params.shape)
    aikit.utils.assertions.check_gather_input_valid(params, indices, axis, batch_dims)
    result = []
    if batch_dims == 0:
        result = jnp.take(params, indices, axis)
    else:
        for b in range(batch_dims):
            if b == 0:
                zip_list = [(p, i) for p, i in zip(params, indices)]
            else:
                zip_list = [
                    (p, i) for z in [zip(p1, i1) for p1, i1 in zip_list] for p, i in z
                ]
        for z in zip_list:
            p, i = z
            r = jnp.take(p, i, axis - batch_dims)
            result.append(r)
        result = jnp.array(result)
        result = result.reshape([*params.shape[0:batch_dims], *result.shape[1:]])
    return result


def gather_nd_helper(params, indices):
    indices_shape = indices.shape
    params_shape = params.shape
    if len(indices.shape) == 0:
        num_index_dims = 1
    else:
        num_index_dims = indices_shape[-1]
    res_dim_sizes_list = [
        _reduce(mul, params_shape[i + 1 :], 1) for i in range(len(params_shape) - 1)
    ] + [1]
    result_dim_sizes = jnp.array(res_dim_sizes_list)
    implicit_indices_factor = int(result_dim_sizes[num_index_dims - 1].item())
    flat_params = jnp.reshape(params, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = jnp.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = jnp.tile(
        jnp.reshape(jnp.sum(indices * indices_scales, -1, keepdims=True), (-1, 1)),
        (1, implicit_indices_factor),
    )
    implicit_indices = jnp.tile(
        jnp.expand_dims(jnp.arange(implicit_indices_factor), 0),
        (indices_for_flat_tiled.shape[0], 1),
    )
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = jnp.reshape(indices_for_flat, (-1,)).astype(jnp.int32)
    flat_gather = jnp.take(flat_params, flat_indices_for_flat, 0)
    new_shape = list(indices_shape[:-1]) + list(params_shape[num_index_dims:])
    ret = jnp.reshape(flat_gather, new_shape)
    return ret


def gather_nd(
    params: JaxArray,
    indices: JaxArray,
    /,
    *,
    batch_dims: int = 0,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    aikit.utils.assertions.check_gather_nd_input_valid(params, indices, batch_dims)
    batch_dims = batch_dims % len(params.shape)
    result = []
    if batch_dims == 0:
        result = gather_nd_helper(params, indices)
    else:
        for b in range(batch_dims):
            if b == 0:
                zip_list = [(p, i) for p, i in zip(params, indices)]
            else:
                zip_list = [
                    (p, i) for z in [zip(p1, i1) for p1, i1 in zip_list] for p, i in z
                ]
        for z in zip_list:
            p, i = z
            r = gather_nd_helper(p, i)
            result.append(r)
        result = jnp.array(result)
        result = result.reshape([*params.shape[0:batch_dims], *result.shape[1:]])
    return result


def get_num_dims(x: JaxArray, /, *, as_array: bool = False) -> Union[JaxArray, int]:
    return jnp.asarray(len(jnp.shape(x))) if as_array else len(x.shape)


def inplace_arrays_supported():
    return False


def inplace_decrement(
    x: Union[aikit.Array, JaxArray], val: Union[aikit.Array, JaxArray]
) -> aikit.Array:
    (x_native, val_native), _ = aikit.args_to_native(x, val)
    if aikit.is_aikit_array(x):
        x.data -= val_native
    else:
        x = aikit.Array(x_native - val_native)
    return x


def inplace_increment(
    x: Union[aikit.Array, JaxArray], val: Union[aikit.Array, JaxArray]
) -> aikit.Array:
    (x_native, val_native), _ = aikit.args_to_native(x, val)
    if aikit.is_aikit_array(x):
        x.data += val_native
    else:
        x = aikit.Array(x_native + val_native)
    return x


def inplace_update(
    x: Union[aikit.Array, JaxArray],
    val: Union[aikit.Array, JaxArray],
    /,
    *,
    ensure_in_backend: bool = False,
    keep_input_dtype: bool = False,
) -> aikit.Array:
    if aikit.is_array(x) and aikit.is_array(val):
        _check_inplace_update_support(x, ensure_in_backend)
        if keep_input_dtype:
            val = aikit.astype(val, x.dtype)
        (x_native, val_native), _ = aikit.args_to_native(x, val)
        if aikit.is_aikit_array(x):
            x.data = val_native
            # Handle view updates
            if aikit.exists(x._base):
                base = x._base
                base_idx = aikit.arange(base.size).reshape(base.shape)
                for fn, args, kwargs, index in x._manipulation_stack:
                    kwargs["copy"] = True
                    base_idx = aikit.__dict__[fn](base_idx, *args, **kwargs)
                    base_idx = base_idx[index] if aikit.exists(index) else base_idx
                base_flat = base.data.flatten()
                base_flat = base_flat.at[base_idx.data.flatten()].set(
                    val_native.flatten()
                )

                base.data = base_flat.reshape(base.shape)

                for ref in base._view_refs:
                    view = ref()
                    if aikit.exists(view) and view is not x:
                        _update_view(view, base)

            else:
                for ref in x._view_refs:
                    view = ref()
                    if aikit.exists(view):
                        _update_view(view, x)
        return x
    else:
        return val


def _update_view(view, base):
    for fn, args, kwargs, index in view._manipulation_stack:
        base = aikit.__dict__[fn](base, *args, **kwargs)
        base = base[index] if aikit.exists(index) else base
    view.data = base.data
    return view


def inplace_variables_supported():
    return False


def multiprocessing(context: Optional[str] = None):
    return (
        _multiprocessing if context is None else _multiprocessing.get_context(context)
    )


def scatter_flat(
    indices: JaxArray,
    updates: JaxArray,
    /,
    *,
    size: Optional[int] = None,
    reduction: str = "sum",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    target = out
    target_given = aikit.exists(target)
    if aikit.exists(size) and aikit.exists(target):
        aikit.utils.assertions.check_equal(len(target.shape), 1, as_array=False)
        aikit.utils.assertions.check_equal(target.shape[0], size, as_array=False)
    if not target_given:
        reduction = "replace"
    if reduction == "sum":
        target = target.at[indices].add(updates)
    elif reduction == "replace":
        if not target_given:
            target = jnp.zeros([size], dtype=updates.dtype)
        target = target.at[indices].set(updates)
    elif reduction == "min":
        target = target.at[indices].min(updates)
    elif reduction == "max":
        target = target.at[indices].max(updates)
    else:
        raise aikit.utils.exceptions.AikitException(
            f'reduction is {reduction}, but it must be one of "sum", "min", "max" or'
            ' "replace"'
        )
    if target_given:
        return aikit.inplace_update(out, target)
    return target


scatter_flat.support_native_out = True


def scatter_nd(
    indices: JaxArray,
    updates: JaxArray,
    /,
    shape: Optional[Union[aikit.NativeShape, Sequence[int]]] = None,
    *,
    reduction: str = "sum",
    out: Optional[JaxArray] = None,
) -> JaxArray:
    updates = jnp.array(
        updates,
        dtype=(
            aikit.dtype(out, as_native=True)
            if aikit.exists(out)
            else aikit.default_dtype(item=updates)
        ),
    )
    indices_flat = indices.reshape(-1, indices.shape[-1]).T
    indices_tuple = tuple(indices_flat) + (Ellipsis,)
    target = out
    target_given = aikit.exists(target)
    if aikit.exists(shape) and aikit.exists(target):
        aikit.utils.assertions.check_equal(
            aikit.Shape(target.shape), aikit.Shape(shape), as_array=False
        )
    shape = list(shape) if aikit.exists(shape) else list(out.shape)
    if not target_given:
        target = jnp.zeros(shape, dtype=updates.dtype)
    updates = _broadcast_to(updates, target[indices_tuple].shape)._data
    if reduction == "sum":
        target = target.at[indices_tuple].add(updates)
    elif reduction == "replace":
        target = target.at[indices_tuple].set(updates)
    elif reduction == "min":
        target = target.at[indices_tuple].min(updates)
    elif reduction == "max":
        target = target.at[indices_tuple].max(updates)
    elif reduction == "mul":
        target = target.at[indices_tuple].mul(updates)
    else:
        raise aikit.utils.exceptions.AikitException(
            f'reduction is {reduction}, but it must be one of "sum", "min", "max",'
            ' "mul" or "replace"'
        )
    if aikit.exists(out):
        return aikit.inplace_update(out, target)
    return target


scatter_nd.support_native_out = True


def shape(
    x: JaxArray,
    /,
    *,
    as_array: bool = False,
) -> Union[aikit.Shape, aikit.Array]:
    if as_array:
        return aikit.array(jnp.shape(x), dtype=aikit.default_int_dtype())
    else:
        return aikit.Shape(x.shape)


def vmap(
    func: Callable,
    in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
    out_axes: int = 0,
) -> Callable:
    func = aikit.output_to_native_arrays(func)
    return aikit.inputs_to_native_arrays(
        jax.vmap(func, in_axes=in_axes, out_axes=out_axes)
    )


@with_unsupported_dtypes({"0.4.23 and below": ("float16", "bfloat16")}, backend_version)
def isin(
    elements: JaxArray,
    test_elements: JaxArray,
    /,
    *,
    assume_unique: bool = False,
    invert: bool = False,
) -> JaxArray:
    return jnp.isin(elements, test_elements, assume_unique=assume_unique, invert=invert)


def itemsize(x: JaxArray) -> int:
    return x.itemsize
