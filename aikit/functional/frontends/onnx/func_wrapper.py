import functools
from typing import Callable

import aikit
import aikit.functional.frontends.onnx as onnx_frontend


# --- Helpers --- #
# --------------- #


def _from_aikit_array_to_onnx_frontend_tensor(x, nested=False, include_derived=None):
    if nested:
        return aikit.nested_map(
            _from_aikit_array_to_onnx_frontend_tensor, x, include_derived, shallow=False
        )
    elif isinstance(x, aikit.Array) or aikit.is_native_array(x):
        a = onnx_frontend.Tensor(x)
        return a
    return x


def _aikit_array_to_onnx(x):
    if isinstance(x, aikit.Array) or aikit.is_native_array(x):
        return onnx_frontend.Tensor(x)
    return x


def _native_to_aikit_array(x):
    if isinstance(x, aikit.NativeArray):
        return aikit.array(x)
    return x


def _onnx_frontend_array_to_aikit(x):
    if hasattr(x, "aikit_array"):
        return x.aikit_array
    return x


def _to_aikit_array(x):
    return _onnx_frontend_array_to_aikit(_native_to_aikit_array(x))


# --- Main --- #
# ------------ #


def inputs_to_aikit_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _inputs_to_aikit_arrays_onnx(*args, **kwargs):
        """Convert `Tensor` into `aikit.Array` instances.

        Convert all `Tensor` instances in both the positional and
        keyword arguments into `aikit.Array` instances, and then calls the
        function with the updated arguments.
        """
        # convert all arrays in the inputs to aikit.Array instances
        new_args = aikit.nested_map(
            _to_aikit_array, args, include_derived={"tuple": True}, shallow=False
        )
        new_kwargs = aikit.nested_map(
            _to_aikit_array, kwargs, include_derived={"tuple": True}, shallow=False
        )
        return fn(*new_args, **new_kwargs)

    return _inputs_to_aikit_arrays_onnx


def outputs_to_frontend_arrays(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _outputs_to_frontend_arrays_onnx(*args, **kwargs):
        """Convert `aikit.Array` into `Tensor` instances.

        Call the function, and then converts all `aikit.Array` instances
        returned by the function into `Tensor` instances.
        """
        # call unmodified function
        ret = fn(*args, **kwargs)

        # convert all arrays in the return to `frontend.onnx.Tensor` instances
        return _from_aikit_array_to_onnx_frontend_tensor(
            ret, nested=True, include_derived={"tuple": True}
        )

    return _outputs_to_frontend_arrays_onnx


def to_aikit_arrays_and_back(fn: Callable) -> Callable:
    """Wrap `fn` so it receives and returns `aikit.Array` instances.

    Wrap `fn` so that input arrays are all converted to `aikit.Array`
    instances and return arrays are all converted to `ndarray.NDArray`
    instances.
    """
    return outputs_to_frontend_arrays(inputs_to_aikit_arrays(fn))
