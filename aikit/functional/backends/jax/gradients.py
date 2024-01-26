"""Collection of Jax gradient functions, wrapped to fit Ivy syntax and
signature."""

# global
import jax
import jax.lax as jlax
from aikit.functional.backends.jax import JaxArray, NativeArray
from typing import Optional, Callable, Sequence, Union, Tuple


# local
import aikit
from aikit.functional.aikit.gradients import (
    _get_required_float_variables,
    _get_y_and_ret_idxs,
    _get_native_variables_and_indices,
    _set_duplicates,
    _process_func_ret_and_grads,
)


# ToDo: modify these functions to track whether variable() has been called
def variable(x, /):
    return x


def is_variable(x, /, *, exclusive=False):
    if exclusive:
        return False
    return isinstance(x, NativeArray)


def variable_data(x: JaxArray, /) -> JaxArray:
    return x


def _forward_fn(
    xs, x, func, duplicate_index_chains, xs_grad_idxs=None, ret_grad_idxs=None
):
    """Forward function for gradient calculation."""
    # Setting x(relevant variables) into xs(all variables)
    x = aikit.nested_map(aikit.to_aikit, x, include_derived=True)
    x_arr_idxs = aikit.nested_argwhere(x, aikit.is_array)
    x_arr_values = aikit.multi_index_nest(x, x_arr_idxs)
    if xs_grad_idxs is not None:
        xs_grad_arr_idxs = []
        for grad_idx in xs_grad_idxs:
            xs_grad_arr_idx = aikit.nested_argwhere(
                aikit.index_nest(xs, grad_idx), aikit.is_array
            )
            for idx in xs_grad_arr_idx:
                xs_grad_arr_idxs.append(list(grad_idx) + idx)
        aikit.set_nest_at_indices(xs, xs_grad_arr_idxs, x_arr_values)
    elif aikit.is_array(xs):
        xs = x
    else:
        xs_arr_idxs = aikit.nested_argwhere(xs, lambda x: aikit.is_array(x))
        aikit.set_nest_at_indices(xs, xs_arr_idxs, x_arr_values)

    # Setting duplicates to ensure same references as in the original input
    if not aikit.is_array(xs):
        xs = _set_duplicates(xs, duplicate_index_chains)
    ret = func(xs)

    # Getting the relevant outputs from the function return for gradient calculation
    _, ret_values = _get_native_variables_and_indices(ret, idxs=ret_grad_idxs)
    if isinstance(ret_values, list) and len(ret_values) == 1 and ret_grad_idxs is None:
        ret_values = ret_values[0]
    return ret_values


def execute_with_gradients(
    func,
    xs: JaxArray,
    /,
    *,
    retain_grads: bool = False,
    xs_grad_idxs: Sequence[Sequence[Union[str, int]]] = ((0,),),
    ret_grad_idxs: Sequence[Sequence[Union[str, int]]] = ((0,),),
):
    # Conversion of required arrays to float variables and duplicate index chains
    (
        xs,
        xs_grad_idxs,
        xs_required,
        required_duplicate_index_chains,
        duplicate_index_chains,
    ) = _get_required_float_variables(xs, xs_grad_idxs)

    func_ret = func(xs)

    # Getting the relevant outputs from the function return for gradient calculation
    ret_grad_idxs, y, ret_idxs = _get_y_and_ret_idxs(func_ret, ret_grad_idxs)

    if isinstance(y, aikit.NativeArray):
        # Gradient calculation for a single output
        grad_fn = jax.grad(
            lambda x: _forward_fn(
                xs,
                x,
                func,
                duplicate_index_chains,
                xs_grad_idxs=xs_grad_idxs,
                ret_grad_idxs=ret_grad_idxs,
            )
        )
        grads = _set_duplicates(grad_fn(xs_required), required_duplicate_index_chains)
    else:
        # Gradient calculation for multiple outputs
        grad_fn = jax.jacrev(
            lambda x: _forward_fn(
                xs,
                x,
                func,
                duplicate_index_chains,
                xs_grad_idxs=xs_grad_idxs,
                ret_grad_idxs=ret_grad_idxs,
            )
        )
        grads_ = grad_fn(xs_required)
        grads = grads_
        if isinstance(ret_idxs, list) and len(ret_idxs):
            grads = {
                ret_idxs[i]: _set_duplicates(grad, required_duplicate_index_chains)
                for i, grad in enumerate(grads_)
            }

    return _process_func_ret_and_grads(func_ret, grads, retain_grads)


def value_and_grad(func):
    def grad_fn(xs):
        return aikit.to_native(func(xs))

    def callback_fn(xs):
        xs = aikit.nested_map(lambda x: aikit.to_native(x), xs, include_derived=True)
        value, grad = jax.value_and_grad(grad_fn)(xs)
        return aikit.to_aikit(value), aikit.to_aikit(grad)

    return callback_fn


def stop_gradient(
    x: JaxArray, /, *, preserve_type: bool = True, out: Optional[JaxArray] = None
) -> JaxArray:
    return jlax.stop_gradient(x)


def jac(func: Callable):
    def grad_fn(x_in):
        return aikit.to_native(
            func(aikit.to_aikit(x_in, nested=True)), nested=True, include_derived=True
        )

    def callback_fn(x_in):
        return aikit.to_aikit(
            jax.jacfwd(grad_fn)(aikit.to_native(x_in, nested=True)),
            nested=True,
            include_derived=True,
        )

    return callback_fn


def grad(func: Callable, argnums: Union[int, Tuple[int]] = 0):
    def grad_fn(x_in):
        return aikit.to_native(func(x_in))

    def callback_fn(x_in):
        return aikit.to_aikit(jax.grad(grad_fn, argnums)(aikit.to_native(x_in)))

    return callback_fn
