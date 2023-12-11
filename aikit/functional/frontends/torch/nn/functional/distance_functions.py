import aikit
import aikit.functional.frontends.torch as torch_frontend
from aikit.functional.frontends.torch.func_wrapper import to_aikit_arrays_and_back
from aikit.func_wrapper import with_unsupported_dtypes


@with_unsupported_dtypes({"2.1.1 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def cosine_similarity(x1, x2, *, dim=1, eps=1e-08):
    x1, x2 = torch_frontend.promote_types_of_torch_inputs(x1, x2)

    if len(x1.shape) == len(x2.shape) and len(x2.shape) >= 2:
        numerator = aikit.sum(x1 * x2, axis=dim)
        x1_squared_norm = aikit.sum(aikit.square(x1), axis=dim)
        x2_squared_norm = aikit.sum(aikit.square(x2), axis=dim)
    else:
        numerator = aikit.sum(x1 * x2)
        x1_squared_norm = aikit.sum(aikit.square(x1))
        x2_squared_norm = aikit.sum(aikit.square(x2))

    x1_norm = aikit.sqrt(x1_squared_norm)
    x2_norm = aikit.sqrt(x2_squared_norm)
    norm_mm = x1_norm * x2_norm
    norm_mm, eps = torch_frontend.promote_types_of_torch_inputs(norm_mm, eps)
    denominator = aikit.maximum(norm_mm, eps)

    cosine = numerator / denominator
    return cosine


@with_unsupported_dtypes({"2.1.1 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def pairwise_distance(x1, x2, *, p=2.0, eps=1e-06, keepdim=False):
    x1, x2 = torch_frontend.promote_types_of_torch_inputs(x1, x2)
    x1_dim = len(x1.shape)
    x2_dim = len(x2.shape)
    if x1_dim > x2_dim:
        output_dim = x1_dim
    else:
        output_dim = x2_dim

    return aikit.vector_norm(x1 - x2 + eps, ord=p, axis=output_dim - 1, keepdims=keepdim)


@with_unsupported_dtypes({"2.1.1 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def pdist(input, p=2):
    x = aikit.array(
        [
            abs(input[i] - input[j])
            for i in range(len(input) - 1)
            for j in range(i + 1, len(input))
        ]
    )
    return aikit.vector_norm(x, ord=p, axis=1)
