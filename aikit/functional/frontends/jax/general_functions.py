import aikit
from aikit.functional.frontends.jax.func_wrapper import (
    to_aikit_arrays_and_back,
    outputs_to_native_arrays,
)
from aikit.func_wrapper import outputs_to_aikit_arrays


@to_aikit_arrays_and_back
def device_get(x):
    if aikit.dev(x) != "cpu":
        x = aikit.to_device(x, "cpu")
    return x


@to_aikit_arrays_and_back
def device_put(x, device=None, *, src=None):
    if device is not None:
        cur_dev = aikit.dev(x)
        device = aikit.as_aikit_dev(device)
        if cur_dev != device:
            x = aikit.to_device(x, device)
    return x


def vmap(
    fun, in_axes=0, out_axes=0, axis_name=None, axis_size=None, spmd_axis_name=None
):
    fun = outputs_to_native_arrays(fun)
    return to_aikit_arrays_and_back(
        outputs_to_aikit_arrays(aikit.vmap(fun, in_axes=in_axes, out_axes=out_axes))
    )
