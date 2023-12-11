# global
import jax
from typing import Callable

# local
import aikit
from aikit.func_wrapper import inputs_to_native_arrays


def bind_custom_gradient_function(func, custom_grad_fn):
    def custom_forward(x):
        ret = func(x)
        return aikit.to_native((ret, (x, ret)), nested=True, include_derived=True)

    def custom_backward(*args):
        return (custom_grad_fn(*args),)

    func = jax.custom_vjp(func)
    func.defvjp(custom_forward, custom_backward)
    return inputs_to_native_arrays(func)


def vjp(func: Callable, *primals):
    def grad_fn(*x_in):
        return aikit.to_native(
            func(*aikit.to_aikit(x_in, nested=True)), nested=True, include_derived=True
        )

    primals_out, _vjpfun = aikit.outputs_to_aikit_arrays(jax.vjp)(
        grad_fn, *aikit.to_native(primals, nested=True)
    )

    def vjpfun(x_in):
        return aikit.to_aikit(
            _vjpfun(aikit.to_native(x_in, nested=True)), nested=True, include_derived=True
        )

    return (primals_out, vjpfun)


def jvp(func: Callable, primals, tangents):
    def grad_fn(*x_in):
        return aikit.to_native(
            func(*aikit.to_aikit(x_in, nested=True)), nested=True, include_derived=True
        )

    primals_out, tangents_out = aikit.outputs_to_aikit_arrays(jax.jvp)(
        grad_fn,
        aikit.to_native(primals, nested=True),
        aikit.to_native(tangents, nested=True),
    )

    return (primals_out, tangents_out)
