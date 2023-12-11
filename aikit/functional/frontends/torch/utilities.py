import aikit
from aikit.functional.frontends.torch.func_wrapper import to_aikit_arrays_and_back
from aikit.func_wrapper import with_supported_dtypes
import inspect


# --- Helpers --- #
# --------------- #


@to_aikit_arrays_and_back
def _assert(condition, message):
    if not condition:
        raise Exception(message)
    else:
        return True


# --- Main --- #
# ------------ #


@with_supported_dtypes({"2.1.1 and above": ("int64",)}, "torch")
@to_aikit_arrays_and_back
def bincount(x, weights=None, minlength=0):
    return aikit.bincount(x, weights=weights, minlength=minlength)


def if_else(cond_fn, body_fn, orelse_fn, vars):
    cond_keys = inspect.getargspec(cond_fn).args
    cond_vars = dict(zip(cond_keys, vars))
    return aikit.if_else(cond_fn, body_fn, orelse_fn, cond_vars)


@to_aikit_arrays_and_back
def result_type(tensor, other):
    return aikit.result_type(tensor, other)
