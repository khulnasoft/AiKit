# local
import aikit
from aikit.func_wrapper import with_supported_dtypes
from aikit.functional.frontends.paddle.func_wrapper import to_aikit_arrays_and_back


@to_aikit_arrays_and_back
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
def layer_norm(x, normalized_shape, weight=None, bias=None, epsilon=1e-05, name=None):
    return aikit.layer_norm(x, normalized_shape, scale=weight, offset=bias, eps=epsilon)


@to_aikit_arrays_and_back
@with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
def normalize(x, p=2, axis=1, epsilon=1e-12, name=None):
    if axis < 0:
        axis = aikit.get_num_dims(x) + axis
    return aikit.lp_normalize(x, p=p, axis=axis)
