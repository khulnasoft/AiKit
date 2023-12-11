# global
import tensorflow as tf
from typing import Callable

# local
import aikit
from aikit.func_wrapper import inputs_to_native_arrays
from aikit.functional.aikit.gradients import _get_required_float_variables
from aikit.functional.aikit.gradients import (
    _flatten_containers,
    _rebuild_flattened_containers,
)


def bind_custom_gradient_function(func, custom_grad_fn):
    @tf.custom_gradient
    def custom_module(x):
        x, _, _, _, _ = _get_required_float_variables(x, xs_grad_idxs=None)
        ret = func(x)

        def grad(upstream):
            return custom_grad_fn((x, ret), upstream)

        return aikit.to_native((ret, grad), nested=True, include_derived=True)

    return inputs_to_native_arrays(custom_module)


def vjp(func: Callable, *primals):
    flattened_primals, ret_idxs = _flatten_containers(primals)
    native_flattened_primals = aikit.to_native(flattened_primals, nested=True)

    def grad_fn(*x_in):
        return _flatten_containers(
            aikit.to_native(
                func(
                    *aikit.to_aikit(
                        _rebuild_flattened_containers(x_in, ret_idxs), nested=True
                    )
                ),
                nested=True,
                include_derived=True,
            )
        )

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(native_flattened_primals)
        flat_primals_out, func_ret_idxs = grad_fn(*native_flattened_primals)

    primals_out = _rebuild_flattened_containers(flat_primals_out, func_ret_idxs)

    def vjpfun(x_in):
        grads = tape.gradient(
            flat_primals_out,
            native_flattened_primals,
            output_gradients=aikit.to_native(_flatten_containers(x_in)[0], nested=True),
        )
        return _rebuild_flattened_containers(
            aikit.to_aikit(grads, nested=True, include_derived=True), ret_idxs
        )

    return (aikit.to_aikit(primals_out, nested=True, include_derived=True), vjpfun)


def jvp(func: Callable, primals, tangents):
    flattened_primals, ret_idxs = _flatten_containers(primals)
    flattened_tangents, _ = _flatten_containers(tangents)

    def grad_fn(*x_in):
        return _flatten_containers(
            aikit.to_native(
                func(
                    *aikit.to_aikit(
                        _rebuild_flattened_containers(x_in, ret_idxs), nested=True
                    )
                ),
                nested=True,
                include_derived=True,
            )
        )

    flattened_primals = aikit.to_native(flattened_primals, nested=True)
    flattened_tangents = aikit.to_native(flattened_tangents, nested=True)

    with tf.autodiff.ForwardAccumulator(
        flattened_primals,
        flattened_tangents,
    ) as acc:
        flat_primals_out, func_ret_idxs = grad_fn(*flattened_primals)
    tangents_out = acc.jvp(flat_primals_out)

    return aikit.to_aikit(
        (
            _rebuild_flattened_containers(flat_primals_out, func_ret_idxs),
            _rebuild_flattened_containers(tangents_out, func_ret_idxs),
        ),
        nested=True,
        include_derived=True,
    )
