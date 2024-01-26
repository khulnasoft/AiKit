# global
import aikit
from aikit.functional.frontends.jax.func_wrapper import to_aikit_arrays_and_back


@to_aikit_arrays_and_back
def cond(pred, true_fun, false_fun, *operands, operand=None, linear=None):
    if operand is not None:
        if operands:
            raise aikit.utils.exceptions.AikitException(
                "if `operand` is passed, positional `operands` should not be passed"
            )
        operands = (operand,)

    if pred:
        return true_fun(*operands)
    return false_fun(*operands)


@to_aikit_arrays_and_back
def fori_loop(lower, upper, body_fun, init_val):
    if not callable(body_fun):
        raise aikit.exceptions.AikitException(
            "jax.lax.fori_loop: Argument body_fun should be callable."
        )
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


@to_aikit_arrays_and_back
def map(f, xs):
    return aikit.stack([f(x) for x in xs])


@to_aikit_arrays_and_back
def scan(f, init, xs, length=None, reverse=False, unroll=1):
    if not (callable(f)):
        raise aikit.exceptions.AikitException(
            "jax.lax.scan: Argument f should be callable."
        )
    if xs is None and length is None:
        raise aikit.exceptions.AikitException(
            "jax.lax.scan: Either xs or length must be provided."
        )

    if length is not None and (not isinstance(length, int) or length < 0):
        raise aikit.exceptions.AikitException(
            "jax.lax.scan: length must be a non-negative integer."
        )
    if xs is None:
        xs = [None] * length

    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, aikit.stack(ys)


@to_aikit_arrays_and_back
def switch(index, branches, *operands, operand=None):
    if operand is not None:
        if operands:
            raise aikit.utils.exceptions.AikitException(
                "if `operand` is passed, positional `operands` should not be passed"
            )
        operands = (operand,)

    index = max(index, 0)
    index = min(len(branches) - 1, index)
    return branches[index](*operands)


@to_aikit_arrays_and_back
def while_loop(cond_fun, body_fun, init_val):
    if not (callable(body_fun) and callable(cond_fun)):
        raise aikit.exceptions.AikitException(
            "jax.lax.while_loop: Arguments body_fun and cond_fun should be callable."
        )
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val
