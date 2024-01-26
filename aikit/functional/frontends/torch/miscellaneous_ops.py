import aikit
from aikit.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from aikit.functional.frontends.torch.func_wrapper import to_aikit_arrays_and_back
from aikit.functional.frontends.torch import promote_types_of_torch_inputs
import aikit.functional.frontends.torch as torch_frontend


@to_aikit_arrays_and_back
def atleast_1d(*tensors):
    return aikit.atleast_1d(*tensors)


@to_aikit_arrays_and_back
def atleast_2d(*tensors):
    return aikit.atleast_2d(*tensors)


@to_aikit_arrays_and_back
def atleast_3d(*tensors):
    return aikit.atleast_3d(*tensors)


# TODO: Add Ivy function for block_diag but only scipy.linalg and \
# and torch supports block_diag currently
@to_aikit_arrays_and_back
def block_diag(*tensors):
    shapes_list = [aikit.shape(t) for t in tensors]
    # TODO: Add aikit function to return promoted dtype for multiple tensors at once
    promoted_dtype = aikit.as_aikit_dtype(tensors[0].dtype)
    for idx in range(1, len(tensors)):
        promoted_dtype = torch_frontend.promote_types_torch(
            tensors[idx - 1].dtype, tensors[idx].dtype
        )

    inp_tensors = [aikit.asarray(t, dtype=promoted_dtype) for t in tensors]
    tensors_2d = []
    result_dim_0, result_dim_1 = 0, 0
    for idx, t_shape in enumerate(shapes_list):
        dim_0, dim_1 = 1, 1
        if len(t_shape) > 2:
            raise aikit.exceptions.IvyError(
                "Input tensors must have 2 or fewer dimensions."
                f"Input {idx} has {len(t_shape)} dimensions"
            )
        elif len(t_shape) == 2:
            dim_0, dim_1 = t_shape
            tensors_2d.append(inp_tensors[idx])
        elif len(t_shape) == 1:
            dim_1 = t_shape[0]
            tensors_2d.append(aikit.reshape(inp_tensors[idx], shape=(dim_0, dim_1)))
        else:
            tensors_2d.append(aikit.reshape(inp_tensors[idx], shape=(dim_0, dim_1)))

        result_dim_0 += dim_0
        result_dim_1 += dim_1
        shapes_list[idx] = (dim_0, dim_1)

    ret = aikit.zeros((result_dim_0, result_dim_1), dtype=promoted_dtype)
    ret_dim_0 = 0
    ret_dim_1 = 0
    for idx, t_shape in enumerate(shapes_list):
        dim_0, dim_1 = t_shape
        ret[
            ret_dim_0 : ret_dim_0 + dim_0, ret_dim_1 : ret_dim_1 + dim_1
        ] = aikit.copy_array(tensors_2d[idx])
        ret_dim_0 += dim_0
        ret_dim_1 += dim_1

    return ret


@to_aikit_arrays_and_back
def broadcast_shapes(*shapes):
    return aikit.broadcast_shapes(*shapes)


@with_unsupported_dtypes({"2.1.2 and below": ("bfloat16",)}, "torch")
@to_aikit_arrays_and_back
def broadcast_to(tensor, shape):
    return aikit.broadcast_to(tensor, shape)


@to_aikit_arrays_and_back
def cartesian_prod(*tensors):
    if len(tensors) == 1:
        return tensors

    ret = aikit.meshgrid(*tensors, indexing="ij")
    ret = aikit.stack(ret, axis=-1)
    ret = aikit.reshape(ret, shape=(-1, len(tensors)))

    return ret


@with_unsupported_dtypes({"2.1.2 and below": "float16"}, "torch")
@to_aikit_arrays_and_back
def cdist(x1, x2, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary"):
    if len(x1.shape) == 2 and len(x2.shape) == 2:
        x1_first_dim, x2_first_dim = x1.shape[0], x2.shape[0]
        if (
            compute_mode == "use_mm_for_euclid_dist_if_necessary"
            and (x1_first_dim > 25 or x2_first_dim > 25)
            or compute_mode == "use_mm_for_euclid_dist"
        ):
            return aikit.vector_norm(x1[:, None, :] - x2[None, :, :], axis=-1, ord=p)
        else:
            distances = aikit.zeros((x1_first_dim, x2_first_dim), dtype=x1.dtype)
            for i in range(x1_first_dim):
                for j in range(x2_first_dim):
                    distances[i, j] = aikit.vector_norm(x1[i, :] - x2[j, :], ord=p)
            return distances
    if p == 2:
        B, P, M = x1.shape
        _, R, _ = x2.shape
        if (
            compute_mode == "use_mm_for_euclid_dist_if_necessary"
            and (P > 25 or R > 25)
            or compute_mode == "use_mm_for_euclid_dist"
        ):
            return aikit.vector_norm(
                x1[:, :, None, :] - x2[:, None, :, :], axis=-1, ord=p
            )
        else:
            distances = aikit.zeros((B, P, R), dtype=x1.dtype)
            for b in range(B):
                for i in range(P):
                    for j in range(R):
                        distances[b, i, j] = aikit.vector_norm(
                            x1[b, i, :] - x2[b, j, :], ord=p
                        )
            return distances
    else:
        return aikit.vector_norm(x1[:, :, None, :] - x2[:, None, :, :], axis=-1, ord=p)


@to_aikit_arrays_and_back
def clone(input, *, memory_format=None):
    return aikit.copy_array(input)


@with_unsupported_dtypes({"2.1.2 and below": ("float16", "bool")}, "torch")
@to_aikit_arrays_and_back
def corrcoef(input):
    if len(aikit.shape(input)) > 2:
        raise aikit.exceptions.IvyError(
            "corrcoef(): expected input to have two or fewer dimensions but got an"
            f" input with {aikit.shape(input)} dimensions"
        )
    return aikit.corrcoef(input, y=None, rowvar=True)


@to_aikit_arrays_and_back
def cov(input, /, *, correction=1, fweights=None, aweights=None):
    return aikit.cov(input, ddof=correction, fweights=fweights, aweights=aweights)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
def cross(input, other, dim=None, *, out=None):
    if dim is None:
        dim = -1
    input, other = promote_types_of_torch_inputs(input, other)
    return aikit.cross(input, other, axisa=-1, axisb=-1, axisc=-1, axis=dim, out=out)


@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.1.2 and below": (
            "uint16",
            "uint32",
            "uint64",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
        )
    },
    "torch",
)
def cummax(input, dim, *, out=None):
    input_dtype = input.dtype
    result_values, result_indices = aikit.cummax(input, axis=dim, out=out)
    result_values = result_values.astype(input_dtype)
    return result_values, result_indices


@to_aikit_arrays_and_back
def cumprod(input, dim, *, dtype=None, out=None):
    if not dtype and "int" in input.dtype:
        dtype = aikit.int64
    return aikit.cumprod(input, axis=dim, dtype=dtype, out=out)


@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {"2.1.2 and below": ("uint8", "bfloat16", "float16"), "1.12.1": ()},
    "torch",
)
def cumsum(input, dim, *, dtype=None, out=None):
    if not dtype and "int" in input.dtype:
        dtype = aikit.int64
    return aikit.cumsum(input, axis=dim, dtype=dtype, out=out)


@to_aikit_arrays_and_back
def diag(input, diagonal=0, *, out=None):
    return aikit.diag(input, k=diagonal)


@with_supported_dtypes(
    {"2.1.2 and below": ("float32", "float64", "int32", "int64")}, "torch"
)
@to_aikit_arrays_and_back
def diagflat(x, offset=0, name=None):
    arr = aikit.diagflat(x, offset=offset)
    return arr


@with_unsupported_dtypes({"2.1.2 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def diagonal(input, offset=0, dim1=0, dim2=1):
    return aikit.diagonal(input, offset=offset, axis1=dim1, axis2=dim2)


@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {"2.1.2 and below": ("int8", "float16", "bfloat16", "bool")}, "torch"
)
def diff(input, n=1, dim=-1, prepend=None, append=None):
    return aikit.diff(input, n=n, axis=dim, prepend=prepend, append=append, out=None)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
def einsum(equation, *operands):
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = operands[0]
    return aikit.einsum(equation, *operands)


@to_aikit_arrays_and_back
def finfo(dtype):
    return aikit.finfo(dtype)


@to_aikit_arrays_and_back
def flatten(input, start_dim=0, end_dim=-1):
    return aikit.flatten(input, start_dim=start_dim, end_dim=end_dim)


@to_aikit_arrays_and_back
def flip(input, dims):
    return aikit.flip(input, axis=dims, copy=True)


@to_aikit_arrays_and_back
def fliplr(input):
    aikit.utils.assertions.check_greater(
        len(input.shape),
        2,
        allow_equal=True,
        message="requires tensor to be at least 2D",
        as_array=False,
    )
    return aikit.fliplr(input, copy=True)


@to_aikit_arrays_and_back
def flipud(input):
    aikit.utils.assertions.check_greater(
        len(input.shape),
        1,
        allow_equal=True,
        message="requires tensor to be at least 1D",
        as_array=False,
    )
    return aikit.flipud(input, copy=True)


@to_aikit_arrays_and_back
def gcd(input, other, *, out=None):
    return aikit.gcd(input, other, out=out)


@to_aikit_arrays_and_back
def kron(input, other, *, out=None):
    return aikit.kron(input, other, out=out)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.2 and below": ("int8",)}, "torch")
def lcm(input, other, *, out=None):
    return aikit.lcm(input, other, out=out)


@with_unsupported_dtypes(
    {
        "2.1.2 and below": (
            "float16",
            "bfloat16",
            "integer",
        )
    },
    "torch",
)
@to_aikit_arrays_and_back
def logcumsumexp(input, dim, *, out=None):
    if len(input.shape) == 0:
        ret = input
    else:
        # For numerical stability, cast to float64
        # We cast back to the original type at the end.
        original_dtype = input.dtype
        exp_input = aikit.exp(input.astype("float64"))
        summed_exp_input = aikit.cumsum(exp_input, axis=dim)
        ret = aikit.log(summed_exp_input).astype(original_dtype)
    if aikit.exists(out):
        aikit.inplace_update(out, ret)
    return ret


@to_aikit_arrays_and_back
def meshgrid(*tensors, indexing=None):
    if indexing is None:
        indexing = "ij"
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tensors[0]
    return tuple(aikit.meshgrid(*tensors, indexing=indexing))


@to_aikit_arrays_and_back
def ravel(input):
    return aikit.reshape(input, (-1,))


@with_unsupported_dtypes({"2.1.2 and below": ("float16",)}, "torch")
@to_aikit_arrays_and_back
def renorm(input, p, dim, maxnorm, *, out=None):
    # Torch hardcodes this magic number
    epsilon = 1e-07

    # To iterate through the n-th dimension of `input`, it is easiest to swap
    # the dimension that we wish to iterate through to be first, then iterate
    # through the re-ordered data. This re-ordering is fine for our purposes
    # as we calculate the p-norms and they are all order agnostic. That is,
    # we may re-order the elements of any vector, and as long as none are
    # added, edited, or removed, the p-norm will be the same.
    input_swapped = aikit.swapaxes(input, 0, dim)
    individual_tensors = [input_swapped[i, ...] for i in range(input_swapped.shape[0])]
    ret = []
    for individual_tensor in individual_tensors:
        # These tensors may be multidimensional, but must be treated as a single vector.
        original_shape = individual_tensor.shape
        tensor_flattened = aikit.flatten(individual_tensor)

        # Don't scale up to the maximum norm, only scale down to it.
        norm = aikit.vector_norm(tensor_flattened, axis=0, ord=p)
        multiplier = aikit.minimum(maxnorm / (norm + epsilon), aikit.ones_like(norm))

        # Store the result in its original shape
        ret.append(
            aikit.reshape(aikit.multiply(tensor_flattened, multiplier), original_shape)
        )

    # We must undo our axis swap from the start.
    ret = aikit.asarray(ret, dtype=ret[0].dtype)
    ret = aikit.swapaxes(ret, 0, dim)
    ret = aikit.reshape(ret, input.shape)

    if aikit.exists(out):
        aikit.inplace_update(out, ret)
    return ret


@with_supported_dtypes(
    {
        "2.1.2 and below": (
            "int32",
            "int64",
        )
    },
    "torch",
)
@to_aikit_arrays_and_back
def repeat_interleave(input, repeats, dim=None, *, output_size=None):
    return aikit.repeat(input, repeats, axis=dim)


@to_aikit_arrays_and_back
def roll(input, shifts, dims=None):
    return aikit.roll(input, shifts, axis=dims)


@to_aikit_arrays_and_back
def rot90(input, k, dims):
    total_dims = aikit.get_num_dims(input)
    total_rot_dims = len(dims)

    aikit.utils.assertions.check_greater(
        total_dims,
        2,
        allow_equal=True,
        message="expected total dims >= 2, but got total dims = " + str(total_dims),
        as_array=False,
    )

    aikit.utils.assertions.check_equal(
        total_rot_dims,
        2,
        message="expected total rotation dims == 2, but got dims = "
        + str(total_rot_dims),
        as_array=False,
    )

    aikit.utils.assertions.check_equal(
        dims[0],
        dims[1],
        inverse=True,
        message="expected rotation dims to be different, but got dim0 = "
        + str(dims[0])
        + " and dim1 = "
        + str(dims[1]),
        as_array=False,
    )

    aikit.utils.assertions.check_equal(
        aikit.abs(dims[0] - dims[1]),
        total_dims,
        inverse=True,
        message="expected rotation dims to be different, but got dim0 = "
        + str(dims[0])
        + " and dim1 = "
        + str(dims[1]),
    )

    # range of dims
    aikit.utils.assertions.check_less(
        dims[0],
        total_dims,
        message="Rotation dim0 out of range, dim0 = " + str(dims[0]),
        as_array=False,
    )

    aikit.utils.assertions.check_greater(
        dims[0],
        -total_dims,
        allow_equal=True,
        message="Rotation dim0 out of range, dim0 = " + str(dims[0]),
        as_array=False,
    )

    aikit.utils.assertions.check_less(
        dims[1],
        total_dims,
        message="Rotation dim1 out of range, dim1 = " + str(dims[1]),
        as_array=False,
    )

    aikit.utils.assertions.check_greater(
        dims[1],
        -total_dims,
        allow_equal=True,
        message="Rotation dim1 out of range, dim1 = " + str(dims[1]),
        as_array=False,
    )

    k = (4 + (k % 4)) % 4
    new_axes = list(range(total_dims))
    new_axes[min(dims)], new_axes[max(dims)] = max(dims), min(dims)
    if k == 1:
        flipped = aikit.flip(input, axis=dims[1])
        return aikit.permute_dims(flipped, axes=new_axes, copy=True)
    elif k == 2:
        return aikit.flip(input, axis=dims, copy=True)
    elif k == 3:
        flipped = aikit.flip(input, axis=dims[0])
        return aikit.permute_dims(flipped, axes=new_axes, copy=True)
    else:
        return input


@to_aikit_arrays_and_back
def searchsorted(
    sorted_sequence,
    values,
    /,
    *,
    out_int32=False,
    right=False,
    side="left",
    out=None,
    sorter=None,
):
    if right and side == "left":
        raise aikit.exceptions.IvyError(
            "side and right can't be set to opposites, got side of left"
            " while right was True"
        )
    if right:
        side = "right"
    ret = aikit.searchsorted(sorted_sequence, values, side=side, out=out, sorter=sorter)
    if out_int32:
        ret = aikit.astype(ret, "int32")
    return ret


@with_unsupported_dtypes({"2.1.2 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def tensordot(a, b, dims=2, out=None):
    a, b = promote_types_of_torch_inputs(a, b)
    return aikit.tensordot(a, b, axes=dims, out=out)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"2.1.2 and below": ("float16", "bfloat16")}, "torch")
def trace(input):
    if "int" in input.dtype:
        input = input.astype("int64")
    target_type = "int64" if "int" in input.dtype else input.dtype
    return aikit.astype(aikit.trace(input), target_type)


@with_supported_dtypes({"2.5.0 and below": ("int8", "int16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def tril(input, diagonal=0, *, out=None):
    return aikit.tril(input, k=diagonal, out=out)


@with_unsupported_dtypes({"2.1.2 and below": ("int8", "uint8", "int16")}, "torch")
@to_aikit_arrays_and_back
def tril_indices(row, col, offset=0, *, dtype=aikit.int64, device="cpu", layout=None):
    sample_matrix = aikit.tril(aikit.ones((row, col), device=device), k=offset)
    return aikit.stack(aikit.nonzero(sample_matrix)).astype(dtype)


@with_supported_dtypes(
    {"2.5.0 and below": ("float64", "float32", "int32", "int64")}, "paddle"
)
@to_aikit_arrays_and_back
def triu(input, diagonal=0, *, out=None):
    return aikit.triu(input, k=diagonal, out=out)


@to_aikit_arrays_and_back
def triu_indices(row, col, offset=0, dtype="int64", device="cpu", layout=None):
    # TODO: Handle layout flag when possible.
    sample_matrix = aikit.triu(aikit.ones((row, col), device=device), k=offset)
    return aikit.stack(aikit.nonzero(sample_matrix)).astype(dtype)


@to_aikit_arrays_and_back
def unflatten(input, /, *, dim, sizes):
    return aikit.unflatten(input, dim=dim, shape=sizes, out=None)


@to_aikit_arrays_and_back
def vander(x, N=None, increasing=False):
    # if N == 0:
    #     return aikit.array([], dtype=x.dtype)
    # else:
    return aikit.vander(x, N=N, increasing=increasing, out=None)


@to_aikit_arrays_and_back
@with_supported_dtypes({"2.1.2 and below": ("float32", "float64")}, "torch")
def view_as_complex(input):
    if aikit.shape(input)[-1] != 2:
        raise aikit.exceptions.IvyError("The last dimension must have a size of 2")

    real, imaginary = aikit.split(
        aikit.stop_gradient(input, preserve_type=False),
        num_or_size_splits=2,
        axis=aikit.get_num_dims(input) - 1,
    )
    dtype = aikit.complex64 if input.dtype == aikit.float32 else aikit.complex128
    real = aikit.squeeze(real, axis=aikit.get_num_dims(real) - 1).astype(dtype)
    imag = aikit.squeeze(imaginary, axis=aikit.get_num_dims(imaginary) - 1).astype(dtype)
    complex_ = real + imag * 1j
    return aikit.array(complex_, dtype=dtype)


@with_supported_dtypes(
    {"2.1.2 and below": ("complex64", "complex128")},
    "torch",
)
@to_aikit_arrays_and_back
def view_as_real(input):
    if not aikit.is_complex_dtype(input):
        raise aikit.exceptions.IvyError(
            "view_as_real is only supported for complex tensors"
        )
    re_part = aikit.real(input)
    im_part = aikit.imag(input)
    return aikit.stack((re_part, im_part), axis=-1)
