from typing import Callable
import functools


import aikit
import aikit.functional.frontends.paddle as paddle_frontend


# --- Helpers --- #
# --------------- #


def _from_aikit_array_to_paddle_frontend_tensor(x, nested=False, include_derived=None):
    if nested:
        return aikit.nested_map(
            _from_aikit_array_to_paddle_frontend_tensor, x, include_derived, shallow=False
        )
    elif isinstance(x, aikit.Array) or aikit.is_native_array(x):
        a = paddle_frontend.Tensor(x)
        return a
    return x


def _to_aikit_array(x):
    # if x is a native array return it as an aikit array
    if isinstance(x, aikit.NativeArray):
        return aikit.array(x)

    # else if x is a frontend torch Tensor (or any frontend "Tensor" actually) return the wrapped aikit array # noqa: E501
    elif hasattr(x, "aikit_array"):
        return x.aikit_array

    # else just return x
    return x


# --- Main --- #
# ------------ #


def inputs_to_aikit_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        """Convert `Tensor` into `aikit.Array` instances.

        Convert all `Tensor` instances in both the positional and keyword arguments
        into `aikit.Array` instances, and then call the function with the updated
        arguments.
        """
        # convert all input arrays to aikit.Array instances
        new_args = aikit.nested_map(
            _to_aikit_array, args, include_derived={"tuple": True}, shallow=False
        )
        new_kwargs = aikit.nested_map(
            _to_aikit_array, kwargs, include_derived={"tuple": True}, shallow=False
        )

        return fn(*new_args, **new_kwargs)

    return new_fn


def outputs_to_frontend_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):
        """Convert `aikit.Array` into `Tensor` instances.

        Call the function, and then convert all `aikit.Array` instances returned by the
        function into `Tensor` instances.
        """
        # call unmodified function
        # ToDo: Remove this default dtype setting
        #  once frontend specific backend setting is added
        # aikit.set_default_int_dtype("int64")
        # aikit.set_default_float_dtype(paddle_frontend.get_default_dtype())
        try:
            ret = fn(*args, **kwargs)
        finally:
            aikit.unset_default_int_dtype()
            aikit.unset_default_float_dtype()
        # convert all arrays in the return to `paddle_frontend.Tensor` instances
        return _from_aikit_array_to_paddle_frontend_tensor(
            ret, nested=True, include_derived={"tuple": True}
        )

    return new_fn


def to_aikit_arrays_and_back(fn: Callable) -> Callable:
    """Wrap `fn` so it receives and returns `aikit.Array` instances.

    Wrap `fn` so that input arrays are all converted to `aikit.Array` instances and
    return arrays are all converted to `Tensor` instances.
    """
    return outputs_to_frontend_arrays(inputs_to_aikit_arrays(fn))
