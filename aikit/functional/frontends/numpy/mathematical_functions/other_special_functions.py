# global
import aikit

# local
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    from_zero_dim_arrays_to_scalar,
)


@to_aikit_arrays_and_back
def sinc(x):
    if aikit.get_num_dims(x) == 0:
        x = aikit.astype(x, aikit.float64)
    return aikit.sinc(x)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def unwrap(p, discont=None, axis=-1, *, period=2 * aikit.pi):
    p = aikit.asarray(p)
    nd = p.ndim
    dd = aikit.diff(p, axis=axis)
    if discont is None:
        discont = period / 2
    slice1 = [slice(None, None)] * nd  # full slices
    slice1[axis] = aikit.slice(1, None)
    slice1 = aikit.tuple(slice1)
    dtype = aikit.result_type(dd, period)
    if aikit.issubdtype(dtype, aikit.integer):
        interval_high, rem = aikit.divmod(period, 2)
        boundary_ambiguous = rem == 0
    else:
        interval_high = period / 2
        boundary_ambiguous = True
    interval_low = -interval_high
    ddmod = aikit.mod(dd - interval_low, period) + interval_low
    if boundary_ambiguous:
        aikit.copyto(ddmod, interval_high, where=(ddmod == interval_low) & (dd > 0))
    ph_correct = ddmod - dd
    aikit.copyto(ph_correct, 0, where=aikit.abs(dd) < discont)
    up = aikit.array(p, copy=True, dtype=dtype)
    up[slice1] = p[slice1] + ph_correct.cumsum(axis)
    return up
