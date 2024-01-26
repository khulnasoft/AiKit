# global
import aikit
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_numpy_out,
)


# --- Helpers --- #
# --------------- #


def _cpercentile(N, percent, key=lambda x: x):
    """Find the percentile   of a list of values.

    @parameter N - is a list of values. Note N MUST BE already sorted.
    @parameter percent - a float value from 0.0 to 1.0.
    @parameter key - optional key function to compute value from each element of N.

    @return - the percentile  of the values
    """
    N.sort()
    k = (len(N) - 1) * percent
    f = aikit.math.floor(k)
    c = aikit.math.ceil(k)
    if f == c:
        return key(N[int(k)])
    d0 = key(N[int(f)]) * (c - k)
    d1 = key(N[int(c)]) * (k - f)
    return d0 + d1


def _quantile_is_valid(q):
    # avoid expensive reductions, relevant for arrays with < O(1000) elements
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if not (0.0 <= q[i] <= 1.0):
                return False
    else:
        if not (aikit.all(q >= 0) and aikit.all(q <= 1)):
            return False
    return True


# --- Main --- #
# ------------ #


def nanpercentile(
    a,
    /,
    *,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    interpolation=None,
):
    a = aikit.array(a)
    q = aikit.divide(q, 100.0)
    q = aikit.array(q)

    if not _quantile_is_valid(q):
        # raise ValueError("percentile s must be in the range [0, 100]")
        aikit.logging.warning("percentile s must be in the range [0, 100]")
        return []

    if axis is None:
        resultarray = []
        nanlessarray = []
        for x in a:
            for i in x:
                if not aikit.isnan(i):
                    nanlessarray.append(i)

        for i in q:
            resultarray.append(_cpercentile(nanlessarray, i))
        return resultarray
    elif axis == 1:
        resultarray = []
        nanlessarrayofarrays = []
        for i in a:
            nanlessarray = []
            for t in i:
                if not aikit.isnan(t):
                    nanlessarray.append(t)
            nanlessarrayofarrays.append(nanlessarray)
        for i in q:
            arrayofpercentiles = []
            for ii in nanlessarrayofarrays:
                arrayofpercentiles.append(_cpercentile(ii, i))
            resultarray.append(arrayofpercentiles)
        return resultarray
    elif axis == 0:
        resultarray = []

        try:
            a = aikit.swapaxes(a, 0, 1)
        except aikit.utils.exceptions.IvyError:
            aikit.logging.warning("axis is 0 but couldn't swap")

        finally:
            nanlessarrayofarrays = []
            for i in a:
                nanlessarray = []
                for t in i:
                    if not aikit.isnan(t):
                        nanlessarray.append(t)
                nanlessarrayofarrays.append(nanlessarray)
            for i in q:
                arrayofpercentiles = []
                for ii in nanlessarrayofarrays:
                    arrayofpercentiles.append(_cpercentile(ii, i))
                resultarray.append(arrayofpercentiles)
        return resultarray


@to_aikit_arrays_and_back
@handle_numpy_out
def ptp(a, axis=None, out=None, keepdims=False):
    x = aikit.max(a, axis=axis, keepdims=keepdims)
    y = aikit.min(a, axis=axis, keepdims=keepdims)
    ret = aikit.subtract(x, y)
    return ret.astype(a.dtype, copy=False)
