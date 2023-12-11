import aikit
from aikit.functional.frontends.jax.func_wrapper import to_aikit_arrays_and_back


@to_aikit_arrays_and_back
def stop_gradient(x):
    return aikit.stop_gradient(x)
