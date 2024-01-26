import aikit
from aikit.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from aikit.functional.frontends.torch.func_wrapper import (
    to_aikit_arrays_and_back,
    numpy_to_torch_style_args,
)
from collections import namedtuple
import aikit.functional.frontends.torch as torch_frontend


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
def all(input, dim=None, keepdim=False, *, out=None):
    input_dtype = aikit.as_aikit_dtype(input.dtype)
    ret = aikit.all(input, axis=dim, keepdims=keepdim, out=out)
    if aikit.is_uint_dtype(input_dtype):
        ret = aikit.astype(ret, input_dtype, out=out)
    return ret


@with_unsupported_dtypes({"2.1.2 and below": ("complex",)}, "torch")
@numpy_to_torch_style_args
@to_aikit_arrays_and_back
def amax(input, dim=None, keepdim=False, *, out=None):
    return aikit.max(input, axis=dim, keepdims=keepdim, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("complex",)}, "torch")
@numpy_to_torch_style_args
@to_aikit_arrays_and_back
def amin(input, dim=None, keepdim=False, *, out=None):
    return aikit.min(input, axis=dim, keepdims=keepdim, out=out)


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {"2.1.2 and below": ("float16", "bfloat16", "complex")}, "torch"
)
def aminmax(input, *, dim=None, keepdim=False, out=None):
    minmax_tuple = namedtuple("minmax", ["min", "max"])
    return minmax_tuple(
        aikit.min(input, axis=dim, keepdims=keepdim, out=out),
        aikit.max(input, axis=dim, keepdims=keepdim, out=out),
    )


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
def any(input, dim=None, keepdim=False, *, out=None):
    input_dtype = aikit.as_aikit_dtype(input.dtype)
    ret = aikit.any(input, axis=dim, keepdims=keepdim, out=out)
    if aikit.is_uint_dtype(input_dtype):
        ret = aikit.astype(ret, input_dtype, out=out)
    return ret


@with_unsupported_dtypes({"2.1.2 and below": ("complex", "bool")}, "torch")
@numpy_to_torch_style_args
@to_aikit_arrays_and_back
def argmax(input, dim=None, keepdim=False):
    return aikit.argmax(input, axis=dim, keepdims=keepdim)


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
def argmin(input, dim=None, keepdim=False):
    return aikit.argmin(input, axis=dim, keepdims=keepdim).astype(aikit.int64)


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {"2.1.2 and below": ("uint8", "int8")},
    "torch",
)
def count_nonzero(input, dim=None):
    return aikit.count_nonzero(input, axis=dim).astype(aikit.int64)


@to_aikit_arrays_and_back
def dist(input, other, p=2):
    return aikit.vector_norm(aikit.subtract(input, other), ord=p)


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
def logsumexp(input, dim, keepdim=False, *, out=None):
    c = aikit.max(input, axis=dim, keepdims=True)
    if aikit.get_num_dims(c) > 0:
        c = aikit.where(aikit.isinf(c), aikit.zeros_like(c), c)
    elif not aikit.isinf(c):
        c = 0
    exponential = aikit.exp(input - c)
    sum = aikit.sum(exponential, axis=dim, keepdims=keepdim)
    ret = aikit.log(sum)
    if not keepdim:
        c = aikit.squeeze(c, axis=dim)
    ret = aikit.add(ret, c, out=out)
    return ret


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
def max(*input, dim=None, keepdim=False, out=None):
    if len(input) == 1:
        input = input[0]
    elif len(input) == 2:
        return torch_frontend.maximum(*input)
    if dim is None:
        return aikit.max(input, axis=dim, keepdims=keepdim, out=out)
    elif out is not None:
        aikit.max(input, axis=dim, keepdims=keepdim, out=out[0])
        aikit.argmax(input, axis=dim, keepdims=keepdim, out=out[1])
        return out
    else:
        max_tuple = namedtuple("max", ["values", "indices"])
        return max_tuple(
            aikit.max(input, axis=dim, keepdims=keepdim),
            aikit.argmax(input, axis=dim, keepdims=keepdim),
        )


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
def mean(input, dim=None, keepdim=False, *, dtype=None, out=None):
    if dtype is not None:
        input = input.astype(dtype)
        if out is not None:
            out = out.astype(dtype)
    return aikit.mean(input, axis=dim, keepdims=keepdim, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("complex", "float16", "bool")}, "torch")
@numpy_to_torch_style_args
@to_aikit_arrays_and_back
def median(input, dim=None, keepdim=False, *, out=None):
    if dim is None:
        input = aikit.reshape(input, (-1,))
        sorted_input = aikit.sort(input)
        return sorted_input[(sorted_input.shape[0] - 1) // 2]

    median_tuple = namedtuple("median", ["values", "indices"])

    if input.ndim == 0:
        result = median_tuple(input, aikit.array(0))
    else:
        sorted_indices = aikit.argsort(input, axis=dim)
        median_indices = aikit.gather(
            sorted_indices, (sorted_indices.shape[dim] - 1) // 2, axis=dim
        )
        median_values = aikit.take_along_axis(
            input, aikit.expand_dims(median_indices, axis=dim), dim
        ).squeeze(axis=dim)

        if keepdim:
            median_values = aikit.expand_dims(median_values, axis=dim)
            median_indices = aikit.expand_dims(median_indices, axis=dim)

        result = median_tuple(median_values, median_indices)
    if out is not None:
        aikit.inplace_update(out[0], result.values)
        aikit.inplace_update(out[1], result.indices)
        return out
    return result


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {"2.1.2 and below": ("complex64", "complex128")},
    "torch",
)
def min(*input, dim=None, keepdim=False, out=None):
    if len(input) == 1:
        input = input[0]
    elif len(input) == 2:
        return torch_frontend.minimum(*input)
    if dim is None:
        return aikit.min(input, axis=dim, keepdims=keepdim, out=out)
    elif out is not None:
        aikit.min(input, axis=dim, keepdims=keepdim, out=out[0])
        aikit.argmin(input, axis=dim, keepdims=keepdim, out=out[1])
        return out
    else:
        min_tuple = namedtuple("min", ["values", "indices"])
        return min_tuple(
            aikit.min(input, axis=dim, keepdims=keepdim),
            aikit.argmin(input, axis=dim, keepdims=keepdim),
        )


@to_aikit_arrays_and_back
def moveaxis(input, source, destination):
    return aikit.moveaxis(input, source, destination)


@with_supported_dtypes({"2.1.2 and below": ("float",)}, "torch")
@numpy_to_torch_style_args
@to_aikit_arrays_and_back
def nanmean(input, dim=None, keepdim=False, *, dtype=None, out=None):
    return aikit.nanmean(input, axis=dim, keepdims=keepdim, dtype=dtype, out=out)


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
def nanmedian(input, dim=None, keepdim=False, *, out=None):
    if dim is None:
        flattened_input = aikit.flatten(input)
        sorted_input = aikit.sort(flattened_input)
        nonnan_index = int(sorted_input.shape[0] - aikit.isnan(sorted_input).sum())
        return sorted_input[(nonnan_index - 1) // 2]

    nanmedian_tuple = namedtuple("nanmedian", ["values", "indices"])

    if input.ndim == 0:
        result = nanmedian_tuple(input, aikit.array(0))
    else:
        sorted_indices = aikit.argsort(input, axis=dim)
        nonnan_index = (
            sorted_indices.shape[dim] - aikit.isnan(input).sum(axis=1) - 1
        ) // 2
        nonnan_index = aikit.expand_dims(nonnan_index, axis=1)
        nanmedian_indices = aikit.gather_nd(sorted_indices, nonnan_index, batch_dims=1)
        nanmedian_values = aikit.take_along_axis(
            input, aikit.expand_dims(nanmedian_indices, axis=dim), dim
        ).squeeze(axis=dim)

        if keepdim:
            nanmedian_values = aikit.expand_dims(nanmedian_values, axis=dim)
            nanmedian_indices = aikit.expand_dims(nanmedian_tuple, axis=dim)

        result = nanmedian_tuple(nanmedian_values, nanmedian_indices)
    if out is not None:
        aikit.inplace_update(out[0], result.values)
        aikit.inplace_update(out[1], result.indices)
        return out
    return result


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.2 and below": ("float", "int")},
    "torch",
)
def nansum(input, dim=None, keepdim=False, *, dtype=None):
    input = aikit.where(aikit.isnan(input), aikit.zeros_like(input), input)
    return aikit.sum(input, axis=dim, dtype=dtype, keepdims=keepdim, out=None)


@to_aikit_arrays_and_back
@with_supported_dtypes(
    {"2.1.2 and below": ("float", "complex")},
    "torch",
)
def norm(input, p="fro", dim=None, keepdim=False, out=None, dtype=None):
    if dtype is None or not aikit.is_float_dtype(dtype):
        dtype = "float64" if "128" in str(dtype) else "float32"
    if (
        p == "fro" and (dim is None or isinstance(dim, int) or len(dim) <= 2)
    ) or p is None:
        p = 2
    if isinstance(p, str):
        if dim is None:
            dim = tuple(range(input.dim()))
        return aikit.matrix_norm(
            input, ord=p, axis=dim, keepdims=keepdim, out=out
        ).astype(dtype)
    else:
        return aikit.vector_norm(
            input, ord=p, axis=dim, keepdims=keepdim, dtype=dtype, out=out
        )


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.1.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def prod(input, dim=None, keepdim=False, *, dtype=None):
    if not dtype:
        if "int" in input.dtype:
            dtype = aikit.int64
    return aikit.prod(input, axis=dim, dtype=dtype, keepdims=keepdim)


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.2 and below": ("float16", "bfloat16")}, "torch")
def quantile(input, q, dim=None, keepdim=False, *, interpolation="linear", out=None):
    return aikit.quantile(
        input, q, axis=dim, keepdims=keepdim, interpolation=interpolation, out=out
    )


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.2 and below": ("float16", "bool", "integer")}, "torch")
def std(input, dim=None, unbiased=True, keepdim=False, *, out=None):
    return aikit.std(input, axis=dim, correction=int(unbiased), keepdims=keepdim, out=out)


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.2 and below": ("bfloat16",)}, "torch")
def std_mean(input, dim, unbiased, keepdim=False, *, out=None):
    temp_std = aikit.std(
        input, axis=dim, correction=int(unbiased), keepdims=keepdim, out=out
    )
    temp_mean = aikit.mean(input, axis=dim, keepdims=keepdim, out=out)
    return temp_std, temp_mean


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
def sum(input, dim=None, keepdim=False, *, dtype=None, out=None):
    return aikit.sum(input, axis=dim, dtype=dtype, keepdims=keepdim, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("complex",)}, "torch")
@to_aikit_arrays_and_back
def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    if dim is not None:
        sorted = True
    results = aikit.unique_all(input, axis=dim, by_value=sorted)
    ret = (results.values,) if return_counts or return_inverse else results.values
    if return_inverse:
        inverse_indices = results.inverse_indices
        if dim is None:
            inverse_indices = inverse_indices.reshape(input.shape)
        ret += (inverse_indices,)
    if return_counts:
        ret += (results.counts,)
    return ret


@with_unsupported_dtypes(
    {
        "2.1.2 and below": (
            "float16",
            "complex",
        )
    },
    "torch",
)
@to_aikit_arrays_and_back
def unique_consecutive(input, return_inverse=False, return_counts=False, dim=None):
    output, inverse_indices, counts = aikit.unique_consecutive(input, axis=dim)
    ret = (output,)
    if return_inverse:
        ret += (inverse_indices,)
    if return_counts:
        ret += (counts,)
    return ret


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.1.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def var(input, dim, unbiased, keepdim=False, *, out=None):
    return aikit.var(input, axis=dim, correction=int(unbiased), keepdims=keepdim, out=out)


@numpy_to_torch_style_args
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.1.2 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def var_mean(input, dim, unbiased, keepdim=False, *, out=None):
    temp_var = aikit.var(
        input, axis=dim, correction=int(unbiased), keepdims=keepdim, out=out
    )
    temp_mean = aikit.mean(input, axis=dim, keepdims=keepdim, out=out)
    return (temp_var, temp_mean)
