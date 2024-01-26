# local
import aikit
from aikit.functional.frontends.jax.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_jax_dtype,
)
from aikit.functional.frontends.jax.numpy import promote_types_of_jax_inputs


@to_aikit_arrays_and_back
def append(arr, values, axis=None):
    if axis is None:
        return aikit.concat((aikit.flatten(arr), aikit.flatten(values)), axis=0)
    else:
        return aikit.concat((arr, values), axis=axis)


@to_aikit_arrays_and_back
def array_split(ary, indices_or_sections, axis=0):
    return aikit.split(
        ary, num_or_size_splits=indices_or_sections, axis=axis, with_remainder=True
    )


@to_aikit_arrays_and_back
def atleast_1d(*arys):
    return aikit.atleast_1d(*arys)


@to_aikit_arrays_and_back
def atleast_2d(*arys):
    return aikit.atleast_2d(*arys)


@to_aikit_arrays_and_back
def atleast_3d(*arys):
    return aikit.atleast_3d(*arys)


@to_aikit_arrays_and_back
def bartlett(M):
    if M < 1:
        return aikit.array([])
    if M == 1:
        return aikit.ones(M, dtype=aikit.float64)
    res = aikit.arange(0, M)
    res = aikit.where(
        aikit.less_equal(res, (M - 1) / 2.0),
        2.0 * res / (M - 1),
        2.0 - 2.0 * res / (M - 1),
    )
    return res


@to_aikit_arrays_and_back
def blackman(M):
    if M < 1:
        return aikit.array([])
    if M == 1:
        return aikit.ones((1,))
    n = aikit.arange(0, M)
    alpha = 0.16
    a0 = (1 - alpha) / 2
    a1 = 1 / 2
    a2 = alpha / 2
    ret = (
        a0
        - a1 * aikit.cos(2 * aikit.pi * n / (M - 1))
        + a2 * aikit.cos(4 * aikit.pi * n / (M - 1))
    )
    return ret


@to_aikit_arrays_and_back
def block(arr):
    # TODO: reimplement block
    raise aikit.utils.exceptions.AikitNotImplementedError()


@to_aikit_arrays_and_back
def broadcast_arrays(*args):
    return aikit.broadcast_arrays(*args)


@to_aikit_arrays_and_back
def broadcast_shapes(*shapes):
    return aikit.broadcast_shapes(*shapes)


@to_aikit_arrays_and_back
def broadcast_to(array, shape):
    return aikit.broadcast_to(array, shape)


@to_aikit_arrays_and_back
def clip(a, a_min=None, a_max=None, out=None):
    aikit.utils.assertions.check_all_or_any_fn(
        a_min,
        a_max,
        fn=aikit.exists,
        type="any",
        limit=[1, 2],
        message="at most one of a_min or a_max can be None",
    )
    a = aikit.array(a)
    if a_min is None:
        a, a_max = promote_types_of_jax_inputs(a, a_max)
        return aikit.minimum(a, a_max, out=out)
    if a_max is None:
        a, a_min = promote_types_of_jax_inputs(a, a_min)
        return aikit.maximum(a, a_min, out=out)
    return aikit.clip(a, a_min, a_max, out=out)


@to_aikit_arrays_and_back
def column_stack(tup):
    if len(aikit.shape(tup[0])) == 1:
        ys = []
        for t in tup:
            ys += [aikit.reshape(t, (aikit.shape(t)[0], 1))]
        return aikit.concat(ys, axis=1)
    return aikit.concat(tup, axis=1)


@handle_jax_dtype
@to_aikit_arrays_and_back
def concatenate(arrays, axis=0, dtype=None):
    ret = aikit.concat(arrays, axis=axis)
    if dtype:
        ret = aikit.array(ret, dtype=dtype)
    return ret


@to_aikit_arrays_and_back
def diagflat(v, k=0):
    ret = aikit.diagflat(v, offset=k)
    while len(aikit.shape(ret)) < 2:
        ret = ret.expand_dims(axis=0)
    return ret


@to_aikit_arrays_and_back
def dsplit(ary, indices_or_sections):
    if isinstance(indices_or_sections, (list, tuple, aikit.Array)):
        indices_or_sections = (
            aikit.diff(indices_or_sections, prepend=[0], append=[ary.shape[2]])
            .astype(aikit.int8)
            .to_list()
        )
    return aikit.dsplit(ary, indices_or_sections)


@to_aikit_arrays_and_back
def dstack(tup, dtype=None):
    return aikit.dstack(tup)


@to_aikit_arrays_and_back
def expand_dims(a, axis):
    return aikit.expand_dims(a, axis=axis)


@to_aikit_arrays_and_back
def flip(m, axis=None):
    return aikit.flip(m, axis=axis)


@to_aikit_arrays_and_back
def fliplr(m):
    return aikit.fliplr(m)


@to_aikit_arrays_and_back
def flipud(m):
    return aikit.flipud(m, out=None)


def hamming(M):
    if M <= 1:
        return aikit.ones([M], dtype=aikit.float64)
    n = aikit.arange(M)
    ret = 0.54 - 0.46 * aikit.cos(2.0 * aikit.pi * n / (M - 1))
    return ret


@to_aikit_arrays_and_back
def hanning(M):
    if M <= 1:
        return aikit.ones([M], dtype=aikit.float64)
    n = aikit.arange(M)
    ret = 0.5 * (1 - aikit.cos(2.0 * aikit.pi * n / (M - 1)))
    return ret


@to_aikit_arrays_and_back
def hsplit(ary, indices_or_sections):
    if isinstance(indices_or_sections, (list, tuple, aikit.Array)):
        if ary.ndim == 1:
            indices_or_sections = (
                aikit.diff(indices_or_sections, prepend=[0], append=[ary.shape[0]])
                .astype(aikit.int8)
                .to_list()
            )
        else:
            indices_or_sections = (
                aikit.diff(indices_or_sections, prepend=[0], append=[ary.shape[1]])
                .astype(aikit.int8)
                .to_list()
            )
    return aikit.hsplit(ary, indices_or_sections)


@to_aikit_arrays_and_back
def kaiser(M, beta):
    if M <= 1:
        return aikit.ones([M], dtype=aikit.float64)
    n = aikit.arange(M)
    alpha = 0.5 * (M - 1)
    ret = aikit.i0(beta * aikit.sqrt(1 - ((n - alpha) / alpha) ** 2)) / aikit.i0(beta)
    return ret


@to_aikit_arrays_and_back
def moveaxis(a, source, destination):
    return aikit.moveaxis(a, source, destination)


@to_aikit_arrays_and_back
def pad(array, pad_width, mode="constant", **kwargs):
    return aikit.pad(array, pad_width, mode=mode, **kwargs)


@to_aikit_arrays_and_back
def ravel(a, order="C"):
    return aikit.reshape(a, shape=(-1,), order=order)


@to_aikit_arrays_and_back
def repeat(a, repeats, axis=None, *, total_repeat_length=None):
    return aikit.repeat(a, repeats, axis=axis)


@to_aikit_arrays_and_back
def reshape(a, newshape, order="C"):
    return aikit.reshape(a, shape=newshape, order=order)


@to_aikit_arrays_and_back
def resize(a, new_shape):
    a = aikit.array(a)
    resized_a = aikit.reshape(a, new_shape)
    return resized_a


@to_aikit_arrays_and_back
def roll(a, shift, axis=None):
    return aikit.roll(a, shift, axis=axis)


@to_aikit_arrays_and_back
def rot90(m, k=1, axes=(0, 1)):
    return aikit.rot90(m, k=k, axes=axes)


@to_aikit_arrays_and_back
def row_stack(tup):
    if len(aikit.shape(tup[0])) == 1:
        xs = []
        for t in tup:
            xs += [aikit.reshape(t, (1, aikit.shape(t)[0]))]
        return aikit.concat(xs, axis=0)
    return aikit.concat(tup, axis=0)


@to_aikit_arrays_and_back
def split(ary, indices_or_sections, axis=0):
    if isinstance(indices_or_sections, (list, tuple, aikit.Array)):
        indices_or_sections = (
            aikit.diff(indices_or_sections, prepend=[0], append=[ary.shape[axis]])
            .astype(aikit.int8)
            .to_list()
        )
    return aikit.split(
        ary, num_or_size_splits=indices_or_sections, axis=axis, with_remainder=False
    )


@to_aikit_arrays_and_back
def squeeze(a, axis=None):
    return aikit.squeeze(a, axis=axis)


@to_aikit_arrays_and_back
def stack(arrays, axis=0, out=None, dtype=None):
    if dtype:
        return aikit.astype(
            aikit.stack(arrays, axis=axis, out=out), aikit.as_aikit_dtype(dtype)
        )
    return aikit.stack(arrays, axis=axis, out=out)


@to_aikit_arrays_and_back
def swapaxes(a, axis1, axis2):
    return aikit.swapaxes(a, axis1, axis2)


@to_aikit_arrays_and_back
def take(
    a,
    indices,
    axis=None,
    out=None,
    mode=None,
    unique_indices=False,
    indices_are_sorted=False,
    fill_value=None,
):
    return aikit.gather(a, indices, axis=axis, out=out)


@to_aikit_arrays_and_back
def tile(A, reps):
    return aikit.tile(A, reps)


@to_aikit_arrays_and_back
def transpose(a, axes=None):
    if aikit.isscalar(a):
        return aikit.array(a)
    elif a.ndim == 1:
        return a
    if not axes:
        axes = list(range(len(a.shape)))[::-1]
    if isinstance(axes, int):
        axes = [axes]
    if (len(a.shape) == 0 and not axes) or (len(a.shape) == 1 and axes[0] == 0):
        return a
    return aikit.permute_dims(a, axes, out=None)


@handle_jax_dtype
@to_aikit_arrays_and_back
def tri(N, M=None, k=0, dtype="float64"):
    if M is None:
        M = N
    ones = aikit.ones((N, M), dtype=dtype)
    return aikit.tril(ones, k=k)


@to_aikit_arrays_and_back
def tril(m, k=0):
    return aikit.tril(m, k=k)


@to_aikit_arrays_and_back
def trim_zeros(flit, trim="fb"):
    start_index = 0
    end_index = aikit.shape(flit)[0]
    trim = trim.lower()
    if "f" in trim:
        for item in flit:
            if item == 0:
                start_index += 1
            else:
                break
    if "b" in trim:
        for item in flit[::-1]:
            if item == 0:
                end_index -= 1
            else:
                break
    return flit[start_index:end_index]


@to_aikit_arrays_and_back
def vsplit(ary, indices_or_sections):
    if isinstance(indices_or_sections, (list, tuple, aikit.Array)):
        indices_or_sections = (
            aikit.diff(indices_or_sections, prepend=[0], append=[ary.shape[0]])
            .astype(aikit.int8)
            .to_list()
        )
    return aikit.vsplit(ary, indices_or_sections)
