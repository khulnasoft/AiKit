"""Collection of MXNet gradient functions, wrapped to fit Aikit syntax and
signature."""

# global
from typing import Sequence, Union
import mxnet as mx

# local
from aikit.utils.exceptions import AikitNotImplementedException


def variable(x, /):
    return x


def is_variable(x, /, *, exclusive=False):
    return isinstance(x, mx.ndarray.NDArray)


def variable_data(x, /):
    raise AikitNotImplementedException()


def execute_with_gradients(
    func,
    xs,
    /,
    *,
    retain_grads: bool = False,
    xs_grad_idxs: Sequence[Sequence[Union[str, int]]] = ((0,),),
    ret_grad_idxs: Sequence[Sequence[Union[str, int]]] = ((0,),),
):
    raise AikitNotImplementedException()


def value_and_grad(func):
    raise AikitNotImplementedException()


def jac(func):
    raise AikitNotImplementedException()


def grad(func, argnums=0):
    raise AikitNotImplementedException()


def stop_gradient(x, /, *, preserve_type=True, out=None):
    raise AikitNotImplementedException()
