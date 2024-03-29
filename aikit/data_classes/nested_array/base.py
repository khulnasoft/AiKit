# global
import abc
from typing import List, Tuple

# local
import aikit


class NestedArrayBase(abc.ABC):
    """Base class for nested array objects."""

    def __init__(self, data, nested_rank, inner_shape, dtype, device, internal=False):
        if not internal:
            raise RuntimeError(
                "NestedArray is an abstract class "
                "and should not be instantiated directly."
                "Please use one of the factory methods instead"
            )
        self._data = data
        self._nested_rank = nested_rank
        self._inner_shape = inner_shape
        self._shape = [len(self._data)] + [None] * self._nested_rank + self._inner_shape
        self._dtype = dtype
        self._device = device
        self._pre_repr = "aikit.NestedArray"

    @classmethod
    def nested_array(
        cls, data, nested_rank=None, inner_shape=None, dtype=None, device=None
    ):
        dtype = aikit.default_dtype(dtype=dtype, item=data)
        device = aikit.default_device(device, item=data)

        # convert all the leaf lists to aikit arrays, determine inner_shape and depth
        det_inner_shape = []

        # ToDo: add check for depth being the same for all nests
        def _seq_to_aikit(x, depth=0):
            if nested_rank is not None and depth >= nested_rank:
                x = aikit.array(x, dtype=dtype, device=device)
                depth += x.ndim - 1
                if x.ndim > 1:
                    det_inner_shape.append(list(x.shape[1:]))
                else:
                    det_inner_shape.append([])
            elif (
                isinstance(x, (list, tuple))
                and len(x) != 0
                and isinstance(x[0], (list, tuple))
            ):
                depth_ret = None
                for i, item in enumerate(x):
                    x = list(x) if isinstance(x, tuple) else x
                    x[i], depth_ret = _seq_to_aikit(item, depth=depth + 1)

                depth = depth_ret if depth_ret else depth
            else:
                x = aikit.array(x, dtype=dtype, device=device)
                if x.ndim > 1:
                    det_inner_shape.append(list(x.shape[1:]))
                else:
                    det_inner_shape.append([])
            return x, depth

        if isinstance(data, (list, tuple)):
            data, depth = _seq_to_aikit(data)
            depth += 1
            # make sure that all the elements of det_inner_shape are the same
            if len(det_inner_shape) > 0:
                if [det_inner_shape[0]] * len(det_inner_shape) != det_inner_shape:
                    raise ValueError(
                        "All the elements of the nested array must have the same "
                        f"inner shape, got: {det_inner_shape}"
                    )
                det_inner_shape = det_inner_shape[0]

            # defining default values for nested_rank and inner_shape
            default_nested_rank = (
                max(0, depth - 1)
                if inner_shape is None
                else max(0, depth - 1 - len(inner_shape))
            )
            default_inner_shape = [] if nested_rank is None else det_inner_shape

            # determining actual values for nested_rank and inner_shape
            nested_rank = (
                nested_rank if nested_rank is not None else default_nested_rank
            )
            inner_shape = (
                list(inner_shape) if inner_shape is not None else default_inner_shape
            )
        elif isinstance(data, cls):
            data = data._data
            nested_rank = nested_rank if nested_rank is not None else data.nested_rank
            inner_shape = (
                list(inner_shape) if list(inner_shape) is not None else data.inner_shape
            )
        else:
            raise TypeError(f"Input data must be pylist or tuple, got: {type(data)}")

        return cls(data, nested_rank, inner_shape, dtype, device, internal=True)

    @staticmethod
    def ragged_multi_map_in_function(fn, *args, **kwargs):
        arg_nest_idxs = aikit.nested_argwhere(
            args, aikit.is_aikit_nested_array, to_ignore=aikit.NestedArray
        )
        kwarg_nest_idxs = aikit.nested_argwhere(
            kwargs, aikit.is_aikit_nested_array, to_ignore=aikit.NestedArray
        )
        # retrieve all the nested_array in args and kwargs
        arg_nest = aikit.multi_index_nest(args, arg_nest_idxs)
        kwarg_nest = aikit.multi_index_nest(kwargs, kwarg_nest_idxs)
        num_arg_nest, num_kwarg_nest = len(arg_nest), len(kwarg_nest)
        num_nest = num_arg_nest + num_kwarg_nest
        inspect_fn = fn
        if isinstance(fn, str):
            inspect_fn = aikit.__dict__[fn]
        nests = arg_nest + kwarg_nest

        def map_fn(vals):
            arg_vals = vals[:num_arg_nest]
            a = aikit.copy_nest(args, to_mutable=True)
            aikit.set_nest_at_indices(a, arg_nest_idxs, arg_vals)
            kwarg_vals = vals[num_arg_nest:]
            kw = aikit.copy_nest(kwargs, to_mutable=True)
            aikit.set_nest_at_indices(kw, kwarg_nest_idxs, kwarg_vals)
            return inspect_fn(*a, **kw)

        if num_nest == 0:
            raise ValueError(
                f"No RaggedArrays found in args or kwargs of function {fn}"
            )
        ret = aikit.NestedArray.ragged_multi_map(map_fn, nests)
        return ret

    @staticmethod
    def ragged_multi_map(fn, ragged_arrays):
        args = []
        for ragged in ragged_arrays:
            args.append(aikit.copy_nest(ragged.data))
        ret = aikit.nested_multi_map(lambda x, _: fn(x), args)
        # infer dtype, shape, and device from the first array in the ret data
        broadcasted_shape = aikit.NestedArray.broadcast_shapes(
            [arg.shape for arg in ragged_arrays]
        )
        # infer ragged_rank from broadcasted shape
        for i, dim in enumerate(broadcasted_shape[::-1]):
            if dim is None:
                nested_rank = len(broadcasted_shape) - i - 1
                break
        inner_shape = broadcasted_shape[nested_rank:]
        arr0_id = aikit.nested_argwhere(ret, aikit.is_aikit_array, stop_after_n_found=1)[0]
        arr0 = aikit.index_nest(ret, arr0_id)
        ragged_ret = aikit.NestedArray.nested_array(
            ret, nested_rank, inner_shape, arr0.dtype, arr0.device
        )
        return ragged_ret

    @staticmethod
    def replace_aikit_arrays(ragged_array, arrays):
        data = ragged_array.data
        aikit_idxs = aikit.nested_argwhere(data, aikit.is_aikit_array)
        arr0 = arrays[0]
        inner_shape, dev, dtype = arr0.shape.as_list(), arr0.device, arr0.dtype
        ret = aikit.set_nest_at_indices(data, aikit_idxs, arrays, shallow=False)
        return aikit.NestedArray.nested_array(
            ret, ragged_array.nested_rank, inner_shape, dtype, dev
        )

    @staticmethod
    def broadcast_shapes(shapes):
        z = []
        max_length = max(len(x) for x in shapes)
        shape_list = list(shapes)
        # making every shape the same length
        for i, shape in enumerate(shapes):
            if len(shape) != max_length:
                shape_list[i] = [1] * (max_length - len(shape)) + shape
        # broadcasting
        for x in zip(*shape_list):
            if None in x:
                for dims in x:
                    if dims is not None and dims != 1:
                        raise ValueError(
                            f"Shapes {shapes[0]} and {shapes[1]} are not broadcastable"
                        )
                z.append(None)
            elif 1 in x:
                dim_exist = False
                for dims in x:
                    if dims != 1:
                        z.append(dims)
                        if dim_exist:
                            raise ValueError(
                                f"Shapes {shapes[0]} and {shapes[1]} are not"
                                " broadcastable"
                            )
                        else:
                            dim_exist = True
                if not dim_exist:
                    z.append(1)
            elif len(set(x)) == 1:
                z.append(x[0])
            else:
                raise ValueError(
                    f"Shapes {shapes[0]} and {shapes[1]} are not broadcastable"
                )
        return z

    def ragged_map(self, fn):
        arg = aikit.copy_nest(self._data)
        aikit.nested_map(lambda x: fn(x), arg, shallow=True)
        # infer dtype, shape, and device from the first array in the ret data
        arr0_id = aikit.nested_argwhere(arg, aikit.is_aikit_array, stop_after_n_found=1)[0]
        arr0 = aikit.index_nest(arg, arr0_id)
        inner_shape = arr0.shape.as_list()[1:]
        ragged_ret = aikit.NestedArray.nested_array(
            arg, self._nested_rank, inner_shape, arr0.dtype, arr0.device
        )
        return ragged_ret

    def unbind(self):
        return tuple(aikit.copy_nest(self._data))

    # Properties #
    # ---------- #

    @property
    def data(self) -> aikit.NativeArray:
        """The native array being wrapped in self."""
        return self._data

    @property
    def dtype(self) -> aikit.Dtype:
        """Data type of the array elements."""
        return self._dtype

    @property
    def device(self) -> aikit.Device:
        """Hardware device the array data resides on."""
        return self._device

    @property
    def shape(self) -> List:
        """Array dimensions."""
        return self._shape

    @property
    def ndim(self) -> int:
        """Number of array dimensions (axes)."""
        return len(tuple(self._shape))

    @property
    def nested_rank(self) -> int:
        """Nested Rank."""
        return self._nested_rank

    @property
    def inner_shape(self) -> Tuple[int]:
        """Inner Shape."""
        return self._inner_shape

    # Built-ins #
    # ----------#

    def __repr__(self):
        rep = self._data.__repr__().replace("[aikit.array", "[")
        rep = rep.replace("aikit.array", "\n\t").replace("(", "").replace(")", "")
        ret = self._pre_repr + "(\n\t" + rep + "\n)"
        return ret

    def __getitem__(self, query):
        ret = self._data[query]
        if isinstance(ret, list):
            return self.__class__.nested_array(
                ret, self._nested_rank - 1, dtype=self._dtype, device=self._device
            )
        return ret
