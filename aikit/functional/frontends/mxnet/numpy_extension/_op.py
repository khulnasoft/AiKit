import aikit
from aikit.functional.frontends.mxnet.func_wrapper import to_aikit_arrays_and_back
from aikit.functional.frontends.numpy.func_wrapper import handle_numpy_dtype


@handle_numpy_dtype
@to_aikit_arrays_and_back
def softmax(data, length=None, axis=-1, temperature=None, use_length=False, dtype=None):
    ret = aikit.softmax(data, axis=axis)
    if dtype:
        aikit.utils.assertions.check_elem_in_list(
            dtype, ["float16", "float32", "float64"]
        )
        ret = aikit.astype(ret, dtype)
    return ret
