"""Collection of Paddle gradient functions, wrapped to fit Aikit syntax and
signature."""

# global

from typing import Optional, Callable
import paddle
import aikit.functional.backends.paddle as paddle_backend
from itertools import chain

# local
import aikit
from aikit.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version
from aikit.functional.aikit.gradients import (
    _get_required_float_variables,
    _get_y_and_ret_idxs,
    _get_native_y,
    _set_duplicates,
    _process_func_ret_and_grads,
)


def variable(x, /):
    if not x.is_leaf:
        ret = x.detach()
        ret.stop_gradient = False
        return ret
    ret = paddle_backend.copy_array(x).to_native()
    ret.stop_gradient = False
    return ret


def is_variable(x, /, *, exclusive: bool = False):
    return isinstance(x, paddle.Tensor) and not x.stop_gradient


def variable_data(x: paddle.Tensor, /) -> paddle.Tensor:
    return x.value()


def _grad_func(y, xs, retain_grads):
    """Gradient calculation function."""
    # Creating a zero gradient nest for the case where no gradients are computed
    grads_ = aikit.nested_map(
        lambda x: (paddle.to_tensor([0.0]) if x is None else paddle.zeros_like(x)),
        xs,
        include_derived=True,
        shallow=False,
    )

    # Gradient calculation
    if isinstance(xs, paddle.Tensor):
        grads = paddle.grad(
            outputs=y,
            inputs=xs,
            retain_graph=True,
            create_graph=retain_grads,
            allow_unused=True,
        )[0]
        grads = grads_ if grads is None else grads
    elif isinstance(xs, aikit.Container):
        grads = xs.cont_from_flat_list(
            list(
                paddle.grad(
                    outputs=[y],
                    inputs=[
                        paddle.to_tensor([0.0]) if v is None else v
                        for k, v in xs.cont_to_iterator()
                    ],
                    retain_graph=True,
                    create_graph=retain_grads,
                    allow_unused=True,
                )
            )
        )
        # Returning zeros if no gradients are computed for consistent results
        if isinstance(grads, aikit.Container):
            grads = aikit.nested_map(
                lambda x: 0 if x is None else x, grads, include_derived=True
            )
            grads = aikit.add(grads, grads_)
        else:
            grads = grads_ if grads is None else grads
    else:

        def grad_(x):
            x = paddle.to_tensor([0.0]) if x is None else x
            grad = paddle.grad(
                outputs=y,
                inputs=paddle.to_tensor([0.0]) if x is None else x,
                retain_graph=True,
                create_graph=retain_grads,
                allow_unused=True,
            )[0]
            return grad if grad is not None else paddle.zeros_like(x)

        grads = aikit.nested_map(grad_, xs, include_derived=True, shallow=False)
        grads = aikit.nested_multi_map(
            lambda x, _: (paddle_backend.add(x[0], x[1])), [grads, grads_]
        )
    return grads


@with_unsupported_device_and_dtypes(
    {"2.6.0 and below": {"cpu": ("float16",)}}, backend_version
)
def execute_with_gradients(
    func, xs, /, *, retain_grads=False, xs_grad_idxs=((0,),), ret_grad_idxs=((0,),)
):
    # Conversion of required arrays to float variables and duplicate index chains
    xs, xs_grad_idxs, xs1, required_duplicate_index_chains, _ = (
        _get_required_float_variables(xs, xs_grad_idxs)
    )
    func_ret = func(xs)
    xs = xs1
    if isinstance(xs, aikit.Container):
        duplicate_indices = list(
            chain.from_iterable([
                map(lambda x: x.split("/"), duplicate_index_chain[1:])
                for duplicate_index_chain in required_duplicate_index_chains
            ])
        )
        xs = aikit.set_nest_at_indices(xs, duplicate_indices, None, shallow=False)

    # Getting the relevant outputs from the function return for gradient calculation
    ret_grad_idxs, y, ret_idxs = _get_y_and_ret_idxs(
        func_ret, ret_grad_idxs, create_var=True
    )

    if isinstance(y, aikit.NativeArray):
        # Gradient calculation for a single output
        grads = _set_duplicates(
            _grad_func(paddle.clone(y), xs, retain_grads),
            required_duplicate_index_chains,
        )
    else:
        # Gradient calculation for multiple outputs
        #
        y = _get_native_y(y)
        grad_arr_idxs = aikit.nested_argwhere(y, lambda x: aikit.is_native_array(x))
        grad_arr_values = aikit.multi_index_nest(y, grad_arr_idxs)
        grads_ = [
            _grad_func(paddle.clone(arr_value), xs, retain_grads)
            for arr_value in grad_arr_values
        ]
        grads = grads_
        if isinstance(ret_idxs, list) and len(ret_idxs):
            grads = {
                ret_idxs[i]: _set_duplicates(grad, required_duplicate_index_chains)
                for i, grad in enumerate(grads_)
            }

    # Stop further gradient propagation if not retaining gradients

    return _process_func_ret_and_grads(func_ret, grads, retain_grads)


def value_and_grad(func):
    def grad_fn(xs):
        return aikit.to_native(func(xs))

    def callback_fn(xs):
        y = grad_fn(xs)

        def autograd_fn(x):
            x = aikit.to_native(x)
            grad = paddle.grad(y, x, allow_unused=True)[0]
            grad = grad if grad is not None else paddle.zeros_like(x)
            grad = aikit.to_aikit(grad)
            return grad

        grads = aikit.nested_map(autograd_fn, xs, include_derived=True, shallow=False)
        y = aikit.to_aikit(y)
        return y, grads

    return callback_fn


def stop_gradient(
    x: Optional[paddle.Tensor],
    /,
    *,
    preserve_type: bool = True,
    out: Optional[paddle.Tensor] = None,
):
    is_var = is_variable(x)
    x.stop_gradient = True
    if is_var and preserve_type:
        return variable(x)
    return x


def _get_jac_one_arg_fn(grad_fn, xs, out_idx):
    nested_indices = iter(aikit.all_nested_indices(xs))

    def one_arg_fn(x):
        idx = next(nested_indices)
        new_xs = aikit.set_nest_at_index(xs, idx, x, shallow=False) if idx else x
        ret = grad_fn(new_xs)
        for i in out_idx:
            ret = ret[i]
        return ret

    return one_arg_fn


def _get_one_out_fn(grad_fn, xs, fn_ret):
    out_nested_indices = iter(aikit.all_nested_indices(fn_ret))

    def one_out_fn(o):
        out_idx = next(out_nested_indices)
        out_shape = aikit.index_nest(grad_fn(xs), out_idx).shape
        one_arg_fn = _get_jac_one_arg_fn(grad_fn, xs, out_idx)
        jacobian = aikit.nested_map(
            lambda x: jacobian_to_aikit(
                paddle.incubate.autograd.Jacobian(
                    one_arg_fn, aikit.to_native(x.expand_dims())
                ),
                x.shape,
                out_shape,
            ),
            xs,
            shallow=False,
        )
        return jacobian

    return one_out_fn


def jacobian_to_aikit(jacobian, in_shape, out_shape):
    jac_aikit = aikit.to_aikit(jacobian[:])
    jac_shape = out_shape + in_shape
    jac_reshaped = jac_aikit.reshape(jac_shape)
    return jac_reshaped


def jac(func: Callable):
    def grad_fn(x_in):
        return aikit.to_native(
            func(aikit.to_aikit(x_in, nested=True)), nested=True, include_derived=True
        )

    def callback_fn(xs):
        fn_ret = grad_fn(xs)
        one_out_fn = _get_one_out_fn(grad_fn, xs, fn_ret)
        jacobian = aikit.nested_map(one_out_fn, fn_ret)
        return jacobian

    return callback_fn


def grad(f, argnums=0):
    if grad.nth == 0:
        grad.f_original = f

    # ToDo: Return grads on nth chained calls rather than None. issue with paddle.grad.
    def _nth_derivative(n):
        def _inner(x):
            x = aikit.to_native(x)
            if n == 0:
                x.stop_gradient = False
                ret = grad.f_original(x) if grad.f_original is not None else f(x)
                grad.nth = 0
                return ret
            else:
                x.stop_gradient = False
                y = _nth_derivative(n - 1)(x)
                y = aikit.to_native(y)
                y_ones = paddle.ones_like(y)
                y_ones.stop_gradient = False
                y.stop_gradient = False
                dy_dx = paddle.grad(
                    outputs=[y],
                    inputs=[x],
                    create_graph=True,
                    grad_outputs=y_ones,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
            return dy_dx

        return _inner

    grad.nth += 1

    return _nth_derivative(grad.nth)


grad.f_original = None
grad.nth = 0
