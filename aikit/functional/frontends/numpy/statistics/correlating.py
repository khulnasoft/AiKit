# global
import aikit
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_numpy_dtype,
)


@handle_numpy_dtype
@to_aikit_arrays_and_back
def corrcoef(x, y=None, /, *, rowvar=True, bias=None, ddof=None, dtype="float64"):
    if (bias is not None) or (ddof is not None):
        aikit.warn("bias and ddof are deprecated and have no effect")

    x = x.astype("float64")
    if y is not None:
        y = y.astype("float64")

    return aikit.corrcoef(x, y=y, rowvar=rowvar).astype(dtype)


@to_aikit_arrays_and_back
def correlate(a, v, mode=None, *, old_behavior=False):
    dtypes = [x.dtype for x in [a, v]]
    mode = mode if mode is not None else "valid"
    aikit.utils.assertions.check_equal(a.ndim, 1, as_array=False)
    aikit.utils.assertions.check_equal(v.ndim, 1, as_array=False)
    n = min(a.shape[0], v.shape[0])
    m = max(a.shape[0], v.shape[0])
    if a.shape[0] >= v.shape[0]:
        if mode == "full":
            r = n + m - 1
            for j in range(0, n - 1):
                a = aikit.concat((aikit.array([0]), a), axis=0)
        elif mode == "same":
            r = m
            right_pad = (n - 1) // 2
            left_pad = (n - 1) - (n - 1) // 2
            for _ in range(0, left_pad):
                a = aikit.concat((aikit.array([0]), a), axis=0)
            for _ in range(0, right_pad):
                a = aikit.concat((a, aikit.array([0])), axis=0)
        elif mode == "valid":
            r = m - n + 1
        else:
            raise aikit.utils.exceptions.IvyException("invalid mode")
        ret = aikit.array(
            [aikit.to_list((v[:n] * aikit.roll(a, -t)[:n]).sum()) for t in range(0, r)],
            dtype=max(dtypes),
        )
    else:
        if mode == "full":
            r = n + m - 1
            for j in range(0, n - 1):
                v = aikit.concat((aikit.array([0]), v), axis=0)
        elif mode == "same":
            r = m
            right_pad = (n - 1) // 2
            left_pad = (n - 1) - (n - 1) // 2
            for _ in range(0, left_pad):
                v = aikit.concat((aikit.array([0]), v), axis=0)
            for _ in range(0, right_pad):
                v = aikit.concat((v, aikit.array([0])), axis=0)
        elif mode == "valid":
            r = m - n + 1
        else:
            raise aikit.utils.exceptions.IvyException("invalid mode")
        ret = aikit.flip(
            aikit.array(
                [aikit.to_list((a[:n] * aikit.roll(v, -t)[:n]).sum()) for t in range(0, r)],
                dtype=max(dtypes),
            )
        )
    return ret
