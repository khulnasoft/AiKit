# local
import aikit
from aikit.func_wrapper import with_unsupported_dtypes
from aikit.functional.frontends.torch.func_wrapper import (
    to_aikit_arrays_and_back,
    numpy_to_torch_style_args,
    to_aikit_shape,
)
import aikit.functional.frontends.torch as torch_frontend


@to_aikit_arrays_and_back
def adjoint(input):
    return aikit.adjoint(input)


@to_aikit_arrays_and_back
def argwhere(input):
    return aikit.argwhere(input)


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
def cat(tensors, dim=0, *, out=None):
    return aikit.concat(tensors, axis=dim, out=out)


@to_aikit_arrays_and_back
def chunk(input, chunks, dim=0):
    if aikit.shape(input) == ():
        return [input]
    else:
        dim_size = aikit.shape(input)[dim]
        chunk_size = dim_size // chunks
        if chunk_size == 0:
            return aikit.split(input, num_or_size_splits=dim_size, axis=dim)
        else:
            remainder = dim_size % chunks
            if remainder == 0:
                return aikit.split(input, num_or_size_splits=chunks, axis=dim)
            else:
                return aikit.split(
                    input,
                    num_or_size_splits=tuple(
                        [chunk_size + remainder] + [chunk_size] * (chunks - 1)
                    ),
                    axis=dim,
                )


@to_aikit_arrays_and_back
def column_stack(tensors, *, out=None):
    reshaped_tensors = []
    for t in tensors:
        dim_num = aikit.get_num_dims(t, as_array=False)
        if dim_num <= 1:
            reshaped_tensor = aikit.reshape(t, (-1, 1))
        else:
            reshaped_tensor = t
        reshaped_tensors.append(reshaped_tensor)
    return aikit.hstack(reshaped_tensors, out=out)


@to_aikit_arrays_and_back
def concat(tensors, dim=0, *, out=None):
    return aikit.concat(tensors, axis=dim, out=out)


@to_aikit_arrays_and_back
def conj(input):
    return aikit.conj(input)


# diagonal_scatter
@with_unsupported_dtypes(
    {
        "2.1.1 and below": (
            "bfloat16",
            "float16",
        )
    },
    "torch",
)
@to_aikit_arrays_and_back
def diagonal_scatter(input, src, offset=0, dim1=0, dim2=1):
    input = aikit.copy_array(input)
    input_shape = input.shape
    indices = aikit.arange(0, input.size)
    diagonal_indices = aikit.diagonal(
        indices.reshape(input.shape), offset=offset, axis1=dim1, axis2=dim2
    )
    if not (src.shape == diagonal_indices.shape):
        raise aikit.utils.exceptions.AikitException(
            "src must have shape equal to specified diagonal of input. src size ="
            f" {src.shape}, diagonal size = {diagonal_indices.shape}"
        )
    input = input.reshape((-1,))
    input[diagonal_indices.reshape((-1,))] = src.reshape((-1,))
    input = input.reshape(input_shape)
    return input


@to_aikit_arrays_and_back
def dsplit(input, indices_or_sections, /):
    if isinstance(indices_or_sections, (list, tuple, aikit.Array)):
        indices_or_sections = (
            aikit.diff(indices_or_sections, prepend=[0], append=[input.shape[2]])
            .astype(aikit.int8)
            .to_list()
        )
    return tuple(aikit.dsplit(input, indices_or_sections))


@to_aikit_arrays_and_back
def dstack(tensors, *, out=None):
    return aikit.dstack(tensors, out=out)


@to_aikit_arrays_and_back
def gather(input, dim, index, *, sparse_grad=False, out=None):
    if sparse_grad:
        raise aikit.utils.exceptions.AikitException(
            "Gather does not yet support the sparse grad functionality"
        )

    dim = dim % len(input.shape)
    all_indices = aikit.argwhere(aikit.full(index.shape, True))
    gather_locations = aikit.reshape(index, [aikit.prod(aikit.array(index.shape))])

    gather_indices = []
    for axis in range(len(index.shape)):
        if axis == dim:
            gather_indices.append(aikit.array(gather_locations, dtype=index.dtype))
        else:
            gather_indices.append(aikit.array(all_indices[:, axis], dtype=index.dtype))

    gather_indices = aikit.stack(gather_indices, axis=-1)
    gathered = aikit.gather_nd(input, gather_indices)
    reshaped = aikit.reshape(gathered, index.shape)
    return reshaped


@to_aikit_arrays_and_back
def hsplit(input, indices_or_sections=None, /):
    if isinstance(indices_or_sections, (list, tuple, aikit.Array)):
        if input.ndim == 1:
            indices_or_sections = (
                aikit.diff(indices_or_sections, prepend=[0], append=[input.shape[0]])
                .astype(aikit.int8)
                .to_list()
            )
        else:
            indices_or_sections = (
                aikit.diff(indices_or_sections, prepend=[0], append=[input.shape[1]])
                .astype(aikit.int8)
                .to_list()
            )
    return tuple(aikit.hsplit(input, indices_or_sections))


@to_aikit_arrays_and_back
def hstack(tensors, *, out=None):
    return aikit.hstack(tensors, out=out)


@to_aikit_arrays_and_back
def index_add(input, dim, index, source, *, alpha=1, out=None):
    input = aikit.swapaxes(input, dim, 0)
    source = aikit.swapaxes(source, dim, 0)
    _to_adds = []
    index = sorted(zip(aikit.to_list(index), range(len(index))), key=(lambda x: x[0]))
    while index:
        _curr_idx = index[0][0]
        while len(_to_adds) < _curr_idx:
            _to_adds.append(aikit.zeros_like(source[0]))
        _to_add_cum = aikit.get_item(source, index[0][1])
        while (len(index) > 1) and (index[0][0] == index[1][0]):
            _to_add_cum = _to_add_cum + aikit.get_item(source, index.pop(1)[1])
        index.pop(0)
        _to_adds.append(_to_add_cum)
    while len(_to_adds) < input.shape[0]:
        _to_adds.append(aikit.zeros_like(source[0]))
    _to_adds = aikit.stack(_to_adds)
    if len(input.shape) < 2:
        # Added this line due to the paddle backend treating scalars as 1-d arrays
        _to_adds = aikit.flatten(_to_adds)

    ret = aikit.add(input, _to_adds, alpha=alpha)
    ret = aikit.swapaxes(ret, 0, dim, out=out)
    return ret


@to_aikit_arrays_and_back
def index_copy(input, dim, index, source, *, out=None):
    input = aikit.swapaxes(input, dim, 0)
    source = aikit.swapaxes(source, dim, 0)
    index = sorted(zip(aikit.to_list(index), range(len(index))), key=(lambda x: x[0]))
    res = []
    while index:
        _curr_idx = index[0][0]
        for i in range(len(res), _curr_idx):
            res.append(aikit.get_item(input, i))
        while (len(index) > 1) and (index[0][0] == index[1][0]):
            index.pop(0)
        res.append(aikit.get_item(source, index[0][1]))
        index.pop(0)
    for i in range(len(res), input.shape[0]):
        res.append(aikit.get_item(input, i))
    res = aikit.stack(res)
    if len(input.shape) < 2:
        res = aikit.flatten(res)

    return aikit.swapaxes(res, 0, dim, out=out)


@with_unsupported_dtypes(
    {
        "2.1.1 and below": (
            "uint16",
            "uint32",
            "uint64",
            "bfloat16",
            "complex128",
            "complex64",
        )
    },
    "torch",
)
@to_aikit_arrays_and_back
def index_reduce(input, dim, index, source, reduce, *, include_self=True, out=None):
    result = aikit.copy_array(input)
    counts = (
        aikit.ones_like(result, dtype=result.dtype)
        if include_self
        else aikit.zeros_like(result, dtype=result.dtype)
    )

    index = index.astype(aikit.int64)

    def init_val(reduce):
        if reduce == "prod":
            return 1
        elif reduce == "amax":
            return -aikit.inf
        elif reduce == "amin":
            return aikit.inf
        else:
            return 0

    if not include_self:
        result[index, ...] = init_val(reduce)

    numel = index.size
    index_contig = aikit.copy_array(index)

    def update_counts(reduce, counts, dim, input_index):
        if reduce == "mean":
            counts_slice = [slice(None)] * counts.ndim
            counts_slice[dim] = input_index
            counts[tuple(counts_slice)] += 1
        return counts

    def update_result(result, reduce, input_data, source_data):
        if reduce == "prod":
            return input_data * source_data
        elif reduce == "amin":
            return aikit.minimum(input_data, source_data)
        elif reduce == "amax":
            return aikit.maximum(input_data, source_data)
        else:
            return input_data + source_data

    if result.ndim > 1:
        for i in range(numel):
            input_index = index_contig[i]
            if not (0 <= input_index < result.shape[dim]):
                raise IndexError("Index out of range in self")

            input_data = aikit.gather(result, [input_index], axis=dim)
            source_data = aikit.gather(source, [i], axis=dim)

            result_slice = [slice(None)] * result.ndim
            result_slice[dim] = input_index

            update_data = update_result(result, reduce, input_data, source_data)
            slide_shape = result[tuple(result_slice)].shape
            result[tuple(result_slice)] = aikit.reshape(update_data, slide_shape)

            counts = update_counts(reduce, counts, dim, input_index)

    elif result.ndim == 1:
        for i in range(numel):
            input_index = index_contig[i]
            if not (0 <= input_index < result.size):
                raise IndexError("Index out of range in self")

            input_data = aikit.flatten(result)[input_index]
            source_data = aikit.flatten(source)[i]

            result[input_index] = update_result(result, reduce, input_data, source_data)
            counts[input_index] += 1

    if reduce == "mean":
        if aikit.any(counts == aikit.array(0)):
            counts[counts == aikit.array(0)] = aikit.array(1)
        result /= counts

        if not input.is_float_dtype():
            result = aikit.floor(result)
            result = result.astype(input.dtype)

    return result


@to_aikit_arrays_and_back
def index_select(input, dim, index, *, out=None):
    return aikit.gather(input, index, axis=dim, out=out)


@to_aikit_arrays_and_back
def masked_select(input, mask, out=None):
    return aikit.flatten(input[mask], out=out)


@to_aikit_arrays_and_back
def moveaxis(input, source, destination):
    return aikit.moveaxis(input, source, destination)


@to_aikit_arrays_and_back
def movedim(input, source, destination):
    return aikit.moveaxis(input, source, destination)


@to_aikit_arrays_and_back
def narrow(input, dim, start, length):
    num_dims = aikit.get_num_dims(input)
    slices = [slice(None)] * num_dims
    slices[dim] = slice(start, start + length)
    return input[tuple(slices)]


@to_aikit_arrays_and_back
def nonzero(input, *, out=None, as_tuple=False):
    ret = aikit.nonzero(input)
    if as_tuple is False:
        ret = aikit.matrix_transpose(aikit.stack(ret))

    if aikit.exists(out):
        return aikit.inplace_update(out, ret)
    return ret


@to_aikit_arrays_and_back
def permute(input, dims):
    return aikit.permute_dims(input, axes=dims, copy=False)


@to_aikit_shape
@to_aikit_arrays_and_back
def reshape(input, shape):
    return aikit.reshape(input, shape)


@to_aikit_arrays_and_back
def row_stack(tensors, *, out=None):
    return aikit.vstack(tensors, out=out)


@to_aikit_arrays_and_back
def select(input, dim, index):
    num_dims = aikit.get_num_dims(input)
    slices = [slice(None)] * num_dims
    slices[dim] = index
    return input[tuple(slices)]


@to_aikit_arrays_and_back
def split(tensor, split_size_or_sections, dim=0):
    if isinstance(split_size_or_sections, int):
        split_size = split_size_or_sections
        split_size_or_sections = [split_size] * (tensor.shape[dim] // split_size)
        if tensor.shape[dim] % split_size:
            split_size_or_sections.append(tensor.shape[dim] % split_size)
    return tuple(
        aikit.split(
            tensor,
            num_or_size_splits=split_size_or_sections,
            axis=dim,
            with_remainder=True,
        )
    )


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
def squeeze(input, dim=None):
    if isinstance(dim, int) and input.ndim > 0:
        if input.shape[dim] > 1:
            return input
    return aikit.squeeze(input, axis=dim)


@to_aikit_arrays_and_back
def stack(tensors, dim=0, *, out=None):
    return aikit.stack(tensors, axis=dim, out=out)


@to_aikit_arrays_and_back
def swapaxes(input, axis0, axis1):
    return aikit.swapaxes(input, axis0, axis1)


@to_aikit_arrays_and_back
def swapdims(input, dim0, dim1):
    return aikit.swapaxes(input, dim0, dim1)


@to_aikit_arrays_and_back
def t(input):
    if input.ndim > 2:
        raise aikit.utils.exceptions.AikitException(
            "t(input) expects a tensor with <= 2 dimensions, but self is %dD"
            % input.ndim
        )
    if input.ndim == 2:
        return aikit.swapaxes(input, 0, 1)
    else:
        return input


@to_aikit_arrays_and_back
def take(input, index):
    input = aikit.reshape(input, (-1,))
    return aikit.gather(input, index, axis=0)


@to_aikit_arrays_and_back
def take_along_dim(input, indices, dim, *, out=None):
    return aikit.take_along_axis(input, indices, dim, out=out)


@to_aikit_arrays_and_back
def tensor_split(input, indices_or_sections, dim=0):
    if isinstance(indices_or_sections, (list, tuple, aikit.Array)):
        indices_or_sections = (
            aikit.diff(indices_or_sections, prepend=[0], append=[input.shape[dim]])
            .astype(aikit.int8)
            .to_list()
        )
    return aikit.split(
        input, num_or_size_splits=indices_or_sections, axis=dim, with_remainder=True
    )


@to_aikit_arrays_and_back
def tile(input, dims):
    try:
        tup = tuple(dims)
    except TypeError:
        tup = (dims,)
    d = len(tup)
    res = 0
    if len(input.shape) > len([dims]) - 1:
        res = input
    if d < input.ndim:
        tup = (1,) * (input.ndim - d) + tup
        res = aikit.tile(input, tup)

    else:
        res = aikit.tile(input, repeats=dims, out=None)
    return res


@to_aikit_arrays_and_back
def transpose(input, dim0, dim1):
    return aikit.swapaxes(input, dim0, dim1)


@to_aikit_arrays_and_back
def unbind(input, dim=0):
    shape = list(input.shape)
    shape.pop(dim)
    return tuple([x.reshape(tuple(shape)) for x in split(input, 1, dim=dim)])


@to_aikit_arrays_and_back
def unsqueeze(input, dim=0):
    return aikit.expand_dims(input, axis=dim)


@to_aikit_arrays_and_back
def vsplit(input, indices_or_sections=None, /):
    if isinstance(indices_or_sections, (list, tuple, aikit.Array)):
        indices_or_sections = (
            aikit.diff(indices_or_sections, prepend=[0], append=[input.shape[0]])
            .astype(aikit.int8)
            .to_list()
        )
    return tuple(aikit.vsplit(input, indices_or_sections))


@to_aikit_arrays_and_back
def vstack(tensors, *, out=None):
    return aikit.vstack(tensors, out=out)


@to_aikit_arrays_and_back
def where(condition, input=None, other=None):
    if not aikit.exists(input) and not aikit.exists(other):
        return nonzero(condition, as_tuple=True)
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.where(condition, input, other)
