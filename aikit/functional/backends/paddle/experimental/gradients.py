# global
from typing import Callable
import paddle

# local
import aikit
from aikit.func_wrapper import inputs_to_native_arrays
from aikit.functional.aikit.gradients import (
    _flatten_containers,
    _rebuild_flattened_containers,
)
from aikit.utils.exceptions import AikitNotImplementedException


def bind_custom_gradient_function(func, custom_grad_fn):
    class _CustomModule(paddle.autograd.PyLayer):
        @staticmethod
        def forward(ctx, x):
            ret = aikit.to_native(func(x), nested=True, include_derived=True)
            ctx.save_for_backward(x, ret)
            return ret

        @staticmethod
        def backward(ctx, upstream):
            grads = custom_grad_fn(
                *aikit.to_aikit(
                    (ctx.saved_tensor(), upstream), nested=True, include_derived=True
                )
            )
            return aikit.to_native(grads, nested=True, include_derived=True)

    custom_module = _CustomModule.apply
    return inputs_to_native_arrays(custom_module)


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
        )[0]

    # primals_out = _rebuild_flattened_containers(
    #     grad_fn(*aikit.to_aikit(flattened_primals, nested=True)), ret_idxs
    # )
    primals_out = func(*aikit.to_aikit(primals, nested=True))

    def vjpfun(x_in):
        _, vjp_result = aikit.to_aikit(
            paddle.incubate.autograd.vjp(
                grad_fn,
                aikit.to_native(flattened_primals, nested=True),
                aikit.to_native(_flatten_containers(x_in)[0], nested=True),
            )
        )
        return aikit.to_aikit(
            _rebuild_flattened_containers(vjp_result, ret_idxs),
            nested=True,
            include_derived=True,
        )

    return (aikit.to_aikit(primals_out, nested=True, include_derived=True), vjpfun)


def jvp(func: Callable, primals, tangents):
    raise AikitNotImplementedException()
