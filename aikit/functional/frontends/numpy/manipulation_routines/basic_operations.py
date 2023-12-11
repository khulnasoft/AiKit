# local
import aikit
from aikit.functional.frontends.numpy.func_wrapper import (
    inputs_to_aikit_arrays,
    _assert_no_array,
    _assert_array,
)


@inputs_to_aikit_arrays
def copyto(dst, src, /, *, casting="same_kind", where=True):
    # Handle casting
    # Numpy copyto doesn't cast the inputs
    # It just checks casting rules
    aikit.utils.assertions.check_elem_in_list(
        casting,
        ["no", "equiv", "safe", "same_kind", "unsafe"],
        message="casting must be one of [no, equiv, safe, same_kind, unsafe]",
    )

    args = [dst, src]
    args_idxs = aikit.nested_argwhere(args, aikit.is_array)
    args_to_check = aikit.multi_index_nest(args, args_idxs)
    dtype = args_to_check[0].dtype

    if casting in ["no", "equiv"]:
        _assert_no_array(
            args_to_check,
            dtype,
        )
    elif casting in ["same_kind", "safe"]:
        _assert_array(
            args_to_check,
            dtype,
            casting=casting,
        )

    aikit.where(where, src, dst, out=dst)


@inputs_to_aikit_arrays
def shape(array, /):
    return aikit.shape(array)
