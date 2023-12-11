# global
from typing import Callable
import mxnet as mx

# local
import aikit
from aikit.functional.aikit.gradients import (
    _flatten_containers,
    _rebuild_flattened_containers,
)
from aikit.utils.exceptions import AikitNotImplementedException


def bind_custom_gradient_function(func, custom_grad_fn):
    raise AikitNotImplementedException()


def vjp(func: Callable, *primals):
    flattened_primals, ret_idxs = _flatten_containers(primals)

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

    with mx.autograd.record():
        flat_primals_out, func_ret_idxs = grad_fn(
            *aikit.to_native(flattened_primals, nested=True)
        )

    primals_out = _rebuild_flattened_containers(flat_primals_out, func_ret_idxs)

    def vjpfun(x_in):
        grads = mx.autograd.grad(
            flat_primals_out,
            aikit.to_native(flattened_primals, nested=True),
            head_grads=aikit.to_native(_flatten_containers(x_in)[0], nested=True),
        )

        return _rebuild_flattened_containers(
            aikit.to_aikit(grads, nested=True, include_derived=True)
        )

    return (aikit.to_aikit(primals_out, nested=True, include_derived=True), vjpfun)


def jvp(func: Callable, primals, tangents):
    raise AikitNotImplementedException()
