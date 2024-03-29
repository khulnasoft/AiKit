"""Tensorflow general functions.

Collection of TensorFlow general functions, wrapped to fit Aikit syntax
and signature.
"""

# global
from typing import Optional, Union, Sequence, Callable, Tuple
import numpy as np
import multiprocessing as _multiprocessing
from numbers import Number
import tensorflow as tf

# local
import aikit
from aikit.functional.aikit.gradients import _is_variable
from aikit.func_wrapper import with_unsupported_dtypes
from aikit.utils.exceptions import _check_inplace_update_support
from . import backend_version
from ...aikit.general import _broadcast_to

_round = round


def is_native_array(x, /, *, exclusive=False):
    if isinstance(x, (tf.Tensor, tf.Variable, tf.TensorArray)):
        if exclusive and isinstance(x, tf.Variable):
            return False
        return True
    return False


def array_equal(
    x0: Union[tf.Tensor, tf.Variable],
    x1: Union[tf.Tensor, tf.Variable],
    /,
) -> bool:
    x0, x1 = aikit.promote_types_of_inputs(x0, x1)
    return bool(tf.experimental.numpy.array_equal(x0, x1))


def container_types():
    return []


def current_backend_str() -> str:
    return "tensorflow"


def _check_query(query):
    return not isinstance(query, list) and (
        not (aikit.is_array(query) and bool(query.ndim > 0))
    )


def get_item(
    x: Union[tf.Tensor, tf.Variable],
    /,
    query: Union[tf.Tensor, tf.Variable, Tuple],
    *,
    copy: Optional[bool] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if aikit.is_array(query) and aikit.is_bool_dtype(query):
        if not len(query.shape):
            return tf.expand_dims(x, 0)
    return x.__getitem__(query)


get_item.partial_mixed_handler = lambda x, query, **kwargs: (
    all(_check_query(i) for i in query)
    and len({i.shape for i in query if aikit.is_array(i)}) <= 1
    if isinstance(query, tuple)
    else _check_query(query)
)


def to_numpy(x: Union[tf.Tensor, tf.Variable], /, *, copy: bool = True) -> np.ndarray:
    # TensorFlow fails to convert bfloat16 tensor when it has 0 dimensions
    if (
        aikit.is_array(x)
        and get_num_dims(x) == 0
        and aikit.as_native_dtype(x.dtype) is tf.bfloat16
    ):
        x = tf.expand_dims(x, 0)
        if copy:
            return np.squeeze(np.array(tf.convert_to_tensor(x)), 0)
        else:
            return np.squeeze(np.asarray(tf.convert_to_tensor(x)), 0)
    if copy:
        return np.array(tf.convert_to_tensor(x))
    else:
        return np.asarray(tf.convert_to_tensor(x))


def to_scalar(x: Union[tf.Tensor, tf.Variable], /) -> Number:
    ret = to_numpy(x).item()
    if x.dtype == tf.bfloat16:
        return float(ret)
    return ret


def to_list(x: Union[tf.Tensor, tf.Variable], /) -> list:
    return x.numpy().tolist()


def gather(
    params: Union[tf.Tensor, tf.Variable],
    indices: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: int = -1,
    batch_dims: int = 0,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    axis %= len(params.shape)
    batch_dims %= len(params.shape)
    aikit.utils.assertions.check_gather_input_valid(params, indices, axis, batch_dims)
    return tf.gather(params, indices, axis=axis, batch_dims=batch_dims)


def gather_nd_helper(params, indices):
    indices_shape = tf.shape(indices)
    params_shape = tf.shape(params)
    num_index_dims = indices_shape[-1]
    result_dim_sizes_list = [
        tf.math.reduce_prod(params_shape[i + 1 :]) for i in range(len(params_shape) - 1)
    ] + [1]
    result_dim_sizes = tf.convert_to_tensor(result_dim_sizes_list, dtype=indices.dtype)
    implicit_indices_factor = result_dim_sizes[num_index_dims - 1]
    flat_params = tf.reshape(params, (-1,))
    new_shape = [1] * (len(indices_shape) - 1) + [num_index_dims]
    indices_scales = tf.reshape(result_dim_sizes[0:num_index_dims], new_shape)
    indices_for_flat_tiled = tf.reshape(
        tf.reduce_sum(indices * indices_scales, -1, keepdims=True), (-1, 1)
    )
    indices_for_flat_tiled = tf.repeat(
        indices_for_flat_tiled, implicit_indices_factor, axis=1
    )
    implicit_indices = tf.repeat(
        tf.expand_dims(tf.range(implicit_indices_factor), 0),
        indices_for_flat_tiled.shape[0],
        axis=0,
    )
    indices_for_flat = indices_for_flat_tiled + implicit_indices
    flat_indices_for_flat = tf.reshape(indices_for_flat, (-1,))
    flat_gather = tf.gather(flat_params, flat_indices_for_flat)
    res = tf.reshape(
        flat_gather, tf.concat([indices_shape[:-1], params_shape[num_index_dims:]], 0)
    )
    return res


def gather_nd(
    params: Union[tf.Tensor, tf.Variable],
    indices: Union[tf.Tensor, tf.Variable],
    /,
    *,
    batch_dims: int = 0,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    aikit.utils.assertions.check_gather_nd_input_valid(params, indices, batch_dims)
    try:
        return tf.gather_nd(params, indices, batch_dims=batch_dims)
    except Exception:  # fall back to compositional implementation
        batch_dims %= len(params.shape)
        result = []
        if batch_dims == 0:
            result = gather_nd_helper(params, indices)
        else:
            for b in range(batch_dims):
                if b == 0:
                    zip_list = list(zip(params, indices))
                else:
                    zip_list = [
                        (p, i)
                        for z in [zip(p1, i1) for p1, i1 in zip_list]
                        for p, i in z
                    ]
            for z in zip_list:
                p, i = z
                r = gather_nd_helper(p, i)
                result.append(r)
            result = tf.stack(result)
            result = tf.reshape(
                result, tf.concat([params.shape[0:batch_dims], result.shape[1:]], 0)
            )
        return result


def get_num_dims(x, /, *, as_array=False):
    return (
        tf.cast(tf.shape(tf.shape(x))[0], tf.int64)
        if as_array
        else int(tf.shape(tf.shape(x)))
    )


def inplace_arrays_supported():
    return False


def inplace_decrement(
    x: Union[aikit.Array, tf.Tensor], val: Union[aikit.Array, tf.Tensor]
) -> aikit.Array:
    (x_native, val_native), _ = aikit.args_to_native(x, val)
    if _is_variable(x_native):
        x_native.assign(x_native - val_native)
        if aikit.is_aikit_array(x):
            x.data = x_native
        else:
            x = aikit.Array(x_native)
    elif aikit.is_aikit_array(x):
        x.data -= val_native
    else:
        x = aikit.Array(val_native)
    return x


def inplace_increment(
    x: Union[aikit.Array, tf.Tensor], val: Union[aikit.Array, tf.Tensor]
) -> aikit.Array:
    (x_native, val_native), _ = aikit.args_to_native(x, val)
    if _is_variable(x_native):
        x_native.assign(x_native + val_native)
        if aikit.is_aikit_array(x):
            x.data = x_native
        else:
            x = aikit.Array(x_native)
    else:
        x_native += val_native
        if aikit.is_aikit_array(x):
            x._data = x_native
        else:
            x = aikit.Array(x_native)
    return x


def inplace_update(
    x: Union[aikit.Array, tf.Tensor],
    val: Union[aikit.Array, tf.Tensor],
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
        if _is_variable(x_native):
            x_native.assign(val_native)
            if aikit.is_aikit_array(x):
                x.data = x_native
            else:
                x = aikit.Array(x_native)
        elif aikit.is_aikit_array(x):
            x.data = val_native
            # Handle view updates
            if aikit.exists(x._base):
                base = x._base
                base_idx = aikit.arange(base.size).reshape(base.shape)
                for fn, args, kwargs, index in x._manipulation_stack:
                    kwargs["copy"] = True
                    base_idx = aikit.__dict__[fn](base_idx, *args, **kwargs)
                    base_idx = base_idx[index] if aikit.exists(index) else base_idx
                base_flat = tf.reshape(base.data, -1)
                base_flat = tf.tensor_scatter_nd_update(
                    base_flat,
                    tf.reshape(base_idx.data, (-1, 1)),
                    tf.reshape(val_native, -1),
                )

                base.data = tf.reshape(base_flat, base.shape)
                for ref in base._view_refs:
                    view = ref()
                    if aikit.exists(view) and view is not x:
                        _update_view(view, base)
            else:
                for ref in x._view_refs:
                    view = ref()
                    if aikit.exists(view):
                        _update_view(view, x)
        else:
            x = aikit.to_aikit(x_native)
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
    return True


def multiprocessing(context: Optional[str] = None):
    return (
        _multiprocessing if context is None else _multiprocessing.get_context(context)
    )


def scatter_flat(
    indices: Union[tf.Tensor, tf.Variable],
    updates: Union[tf.Tensor, tf.Variable],
    /,
    *,
    size: Optional[int] = None,
    reduction: str = "sum",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if indices.dtype != tf.int32 or indices.dtype != tf.int64:
        if indices.dtype in [tf.int8, tf.int16, tf.uint8, tf.uint16]:
            indices = tf.cast(indices, tf.int32)
        else:
            indices = tf.cast(indices, tf.int64)
    target = out
    target_given = aikit.exists(target)
    if aikit.exists(size) and aikit.exists(target):
        aikit.utils.assertions.check_equal(len(target.shape), 1, as_array=False)
        aikit.utils.assertions.check_equal(target.shape[0], size, as_array=False)
    if not target_given:
        target = tf.zeros([size], dtype=updates.dtype)
        res = tf.tensor_scatter_nd_update(target, tf.expand_dims(indices, -1), updates)
    elif reduction == "max":
        res = tf.tensor_scatter_nd_max(target, tf.expand_dims(indices, -1), updates)
    elif reduction == "min":
        res = tf.tensor_scatter_nd_min(target, tf.expand_dims(indices, -1), updates)
    elif reduction == "replace":
        res = tf.tensor_scatter_nd_update(target, tf.expand_dims(indices, -1), updates)
    elif reduction == "sum":
        res = tf.tensor_scatter_nd_add(target, tf.expand_dims(indices, -1), updates)
    else:
        raise aikit.utils.exceptions.AikitException(
            f'reduction is {reduction}, but it must be one of "sum", "min", "max" or'
            ' "replace"'
        )
    return aikit.inplace_update(out, res) if aikit.exists(out) else res


scatter_flat.support_native_out = True


@with_unsupported_dtypes({"2.15.0 and below": ("bfloat16", "complex")}, backend_version)
def scatter_nd(
    indices: Union[tf.Tensor, tf.Variable],
    updates: Union[tf.Tensor, tf.Variable],
    /,
    shape: Optional[Union[aikit.NativeShape, Sequence[int]]] = None,
    *,
    reduction: str = "sum",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    updates_dtype = updates.dtype
    if aikit.exists(out):
        dtype = aikit.promote_types(out.dtype, updates_dtype)
    updates = tf.cast(
        updates,
        (aikit.as_native_dtype(dtype) if aikit.exists(out) else updates_dtype),
    )

    expected_shape = (
        list(indices.shape[:-1]) + list(out.shape[indices.shape[-1] :])
        if aikit.exists(out)
        else list(indices.shape[:-1]) + list(shape[indices.shape[-1] :])
    )
    updates = _broadcast_to(updates, expected_shape)._data
    if len(updates.shape) == 0:
        indices = tf.expand_dims(indices, 0)
        updates = tf.expand_dims(updates, 0)

    # implementation
    target = out
    target_given = aikit.exists(target)
    if aikit.exists(shape) and target_given:
        aikit.utils.assertions.check_equal(
            aikit.Shape(target.shape), aikit.Shape(shape), as_array=False
        )
    if not target_given:
        shape = list(shape) if aikit.exists(shape) else list(out.shape)
        target = tf.zeros(shape, dtype=updates.dtype)
    if reduction == "sum":
        res = tf.tensor_scatter_nd_add(target, indices, updates)
    elif reduction == "min":
        res = tf.tensor_scatter_nd_min(target, indices, updates)
    elif reduction == "max":
        res = tf.tensor_scatter_nd_max(target, indices, updates)
    elif reduction == "mul":
        updates = aikit.multiply(aikit.gather_nd(target, indices), updates).data
        res = tf.tensor_scatter_nd_update(target, indices, updates)
    elif reduction == "replace":
        res = tf.tensor_scatter_nd_update(target, indices, updates)
    else:
        raise aikit.utils.exceptions.AikitException(
            f'reduction is {reduction}, but it must be one of "sum", "min", "max",'
            ' "mul" or "replace"'
        )
    if aikit.exists(out):
        return aikit.inplace_update(out, res)
    return res


scatter_nd.support_native_out = True


def shape(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    as_array: bool = False,
) -> Union[tf.Tensor, aikit.Shape, aikit.Array]:
    if as_array:
        return aikit.array(tf.shape(x), dtype=aikit.default_int_dtype())
    else:
        return aikit.Shape(x.shape)


def vmap(
    func: Callable,
    in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
    out_axes: int = 0,
) -> Callable:
    @aikit.output_to_native_arrays
    @aikit.inputs_to_native_arrays
    def _vmap(*args, **kwargs):
        # convert args tuple to list to allow mutability using moveaxis ahead.
        args = list(args)

        # if in_axis is a non-integer, its length should be equal to pos args.
        if isinstance(in_axes, (list, tuple)):
            aikit.utils.assertions.check_equal(
                len(args),
                len(in_axes),
                message="""in_axes should have a length equivalent to the number
                of positional arguments to the function being vectorized or it
                should be an integer""",
                as_array=False,
            )

        # checking axis_size consistency
        axis_size = set()

        if isinstance(in_axes, int):
            for arg in args:
                axis_size.add(arg.shape[in_axes])
        elif isinstance(in_axes, (list, tuple)):
            for arg, axis in zip(args, in_axes):
                if axis is not None:
                    axis_size.add(arg.shape[axis])

        if len(axis_size) > 1:
            raise aikit.utils.exceptions.AikitException(
                """Inconsistent sizes. All mapped axes should have the same size"""
            )

        # Making sure not all in_axes are None
        if isinstance(in_axes, (list, tuple)):
            aikit.utils.assertions.check_any(
                [aikit.exists(ax) for ax in in_axes],
                message="At least one of the axes should be specified (not None)",
                as_array=False,
            )
        else:
            aikit.utils.assertions.check_exists(
                in_axes, message="single value in_axes should not be None"
            )

        # Handling None in in_axes by broadcasting the axis_size
        if isinstance(in_axes, (tuple, list)) and None in in_axes:
            none_axis_index = []
            for index, axis in enumerate(in_axes):
                if axis is None:
                    none_axis_index.append(index)

            for none_mapped_axis in none_axis_index:
                args[none_mapped_axis] = tf.broadcast_to(
                    args[none_mapped_axis],
                    (tuple(axis_size) + args[none_mapped_axis].shape),
                )

        # set up the axis to be mapped
        if isinstance(in_axes, (tuple, list)):
            for i in range(len(in_axes)):
                args[i] = tf.experimental.numpy.moveaxis(args[i], in_axes[i], 0)
        elif isinstance(in_axes, int):
            args[0] = tf.experimental.numpy.moveaxis(args[0], in_axes, 0)

        # vectorisation - applying map_fn if only one arg provided as reduce requires
        # two elements to begin with.
        arr_results = []
        for arrays in zip(*args):
            single_op = func(*arrays)
            arr_results.append(single_op)
        res = aikit.stack(arr_results)

        if out_axes:
            res = tf.experimental.numpy.moveaxis(res, 0, out_axes)

        return res

    return _vmap


@with_unsupported_dtypes({"2.15.0 and below": ("bfloat16", "complex")}, backend_version)
def isin(
    elements: tf.Tensor,
    test_elements: tf.Tensor,
    /,
    *,
    assume_unique: bool = False,
    invert: bool = False,
) -> tf.Tensor:
    input_shape = elements.shape

    if tf.rank(elements) == 0:
        elements = tf.reshape(elements, [1])
    if tf.rank(test_elements) == 0:
        test_elements = tf.reshape(test_elements, [1])
    if not assume_unique:
        test_elements = tf.unique(tf.reshape(test_elements, [-1]))[0]

    elements = tf.reshape(elements, [-1])
    test_elements = tf.reshape(test_elements, [-1])

    output = tf.reduce_any(
        tf.equal(tf.expand_dims(elements, -1), test_elements), axis=-1
    )
    return tf.reshape(output, input_shape) ^ invert


def itemsize(x: Union[tf.Tensor, tf.Variable]) -> int:
    return x.dtype.size
