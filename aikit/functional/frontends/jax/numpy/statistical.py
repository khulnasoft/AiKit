# local

import aikit
from aikit.func_wrapper import with_unsupported_dtypes
from aikit.functional.frontends.jax.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_jax_dtype,
)
from aikit.functional.frontends.jax.numpy import promote_types_of_jax_inputs
from aikit.functional.backends.jax.experimental.elementwise import _normalize_axis_tuple


@to_aikit_arrays_and_back
def argmin(a, axis=None, out=None, keepdims=None):
    return aikit.argmin(a, axis=axis, out=out, keepdims=keepdims)


@to_aikit_arrays_and_back
def average(a, axis=None, weights=None, returned=False, keepdims=False):
    # canonicalize_axis to ensure axis or the values in axis > 0
    if isinstance(axis, (tuple, list)):
        a_ndim = len(aikit.shape(a))
        new_axis = [0] * len(axis)
        for i, v in enumerate(axis):
            if not -a_ndim <= v < a_ndim:
                raise ValueError(
                    f"axis {v} is out of bounds for array of dimension {a_ndim}"
                )
            new_axis[i] = v + a_ndim if v < 0 else v
        axis = tuple(new_axis)

    if weights is None:
        ret = aikit.mean(a, axis=axis, keepdims=keepdims)
        if axis is None:
            fill_value = int(a.size) if aikit.is_int_dtype(ret) else float(a.size)
            weights_sum = aikit.full((), fill_value, dtype=ret.dtype)
        else:
            if isinstance(axis, tuple):
                # prod with axis has dtype Sequence[int]
                fill_value = 1
                for d in axis:
                    fill_value *= a.shape[d]
            else:
                fill_value = a.shape[axis]
            weights_sum = aikit.full_like(ret, fill_value=fill_value)
    else:
        a = aikit.asarray(a, copy=False)
        weights = aikit.asarray(weights, copy=False)
        a, weights = promote_types_of_jax_inputs(a, weights)

        a_shape = aikit.shape(a)
        a_ndim = len(a_shape)
        weights_shape = aikit.shape(weights)

        # Make sure the dimensions work out
        if a_shape != weights_shape:
            if len(weights_shape) != 1:
                raise ValueError(
                    "1D weights expected when shapes of a and weights differ."
                )
            if axis is None:
                raise ValueError(
                    "Axis must be specified when shapes of a and weights differ."
                )
            elif isinstance(axis, tuple):
                raise ValueError(
                    "Single axis expected when shapes of a and weights differ"
                )
            elif weights.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis."
                )

            weights = aikit.broadcast_to(
                weights, shape=(a_ndim - 1) * (1,) + weights_shape
            )
            weights = aikit.moveaxis(weights, -1, axis)

        weights_sum = aikit.sum(weights, axis=axis)
        ret = aikit.sum(a * weights, axis=axis, keepdims=keepdims) / weights_sum

    if returned:
        if ret.shape != weights_sum.shape:
            weights_sum = aikit.broadcast_to(weights_sum, shape=ret.shape)
        return ret, weights_sum

    return ret


@to_aikit_arrays_and_back
def bincount(x, weights=None, minlength=0, *, length=None):
    x_list = [int(x[i]) for i in range(x.shape[0])]
    max_val = int(aikit.max(aikit.array(x_list)))
    ret = [x_list.count(i) for i in range(0, max_val + 1)]
    ret = aikit.array(ret)
    ret = aikit.astype(ret, aikit.as_aikit_dtype(aikit.int64))
    return ret


@to_aikit_arrays_and_back
def corrcoef(x, y=None, rowvar=True):
    return aikit.corrcoef(x, y=y, rowvar=rowvar)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"0.4.21 and below": ("float16", "bfloat16")}, "jax")
def correlate(a, v, mode="valid", precision=None):
    if aikit.get_num_dims(a) != 1 or aikit.get_num_dims(v) != 1:
        raise ValueError("correlate() only support 1-dimensional inputs.")
    if a.shape[0] == 0 or v.shape[0] == 0:
        raise ValueError(
            f"correlate: inputs cannot be empty, got shapes {a.shape} and {v.shape}."
        )
    if v.shape[0] > a.shape[0]:
        need_flip = True
        a, v = v, a
    else:
        need_flip = False

    out_order = slice(None)

    if mode == "valid":
        padding = [(0, 0)]
    elif mode == "same":
        padding = [(v.shape[0] // 2, v.shape[0] - v.shape[0] // 2 - 1)]
    elif mode == "full":
        padding = [(v.shape[0] - 1, v.shape[0] - 1)]
    else:
        raise ValueError("mode must be one of ['full', 'same', 'valid']")

    result = aikit.conv_general_dilated(
        a[None, None, :],
        v[:, None, None],
        (1,),
        padding,
        dims=1,
        data_format="channel_first",
    )
    return aikit.flip(result[0, 0, out_order]) if need_flip else result[0, 0, out_order]


@to_aikit_arrays_and_back
def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
    return aikit.cov(
        m, y, rowVar=rowvar, bias=bias, ddof=ddof, fweights=fweights, aweights=aweights
    )


@handle_jax_dtype
@to_aikit_arrays_and_back
def cumprod(a, axis=None, dtype=None, out=None):
    if dtype is None:
        dtype = aikit.as_aikit_dtype(a.dtype)
    return aikit.cumprod(a, axis=axis, dtype=dtype, out=out)


@handle_jax_dtype
@to_aikit_arrays_and_back
def cumsum(a, axis=0, dtype=None, out=None):
    if dtype is None:
        dtype = aikit.uint8
    return aikit.cumsum(a, axis, dtype=dtype, out=out)


@to_aikit_arrays_and_back
def einsum(
    subscripts,
    *operands,
    out=None,
    optimize="optimal",
    precision=None,
    preferred_element_type=None,
    _use_xeinsum=False,
    _dot_general=None,
):
    return aikit.einsum(subscripts, *operands, out=out)


@to_aikit_arrays_and_back
def max(a, axis=None, out=None, keepdims=False, where=None):
    ret = aikit.max(a, axis=axis, out=out, keepdims=keepdims)
    if aikit.is_array(where):
        where = aikit.array(where, dtype=aikit.bool)
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_jax_dtype
@to_aikit_arrays_and_back
def mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if dtype is None:
        dtype = "float32" if aikit.is_int_dtype(a) else a.dtype
    ret = aikit.mean(a, axis=axis, keepdims=keepdims, out=out)
    if aikit.is_array(where):
        where = aikit.array(where, dtype=aikit.bool)
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return aikit.astype(ret, aikit.as_aikit_dtype(dtype), copy=False)


@to_aikit_arrays_and_back
def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    return aikit.median(a, axis=axis, out=out, keepdims=keepdims)


@to_aikit_arrays_and_back
def min(a, axis=None, out=None, keepdims=False, where=None):
    ret = aikit.min(a, axis=axis, out=out, keepdims=keepdims)
    if aikit.is_array(where):
        where = aikit.array(where, dtype=aikit.bool)
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return ret


@handle_jax_dtype
@to_aikit_arrays_and_back
def nancumprod(a, axis=None, dtype=None, out=None):
    a = aikit.where(aikit.isnan(a), aikit.zeros_like(a), a)
    return aikit.cumprod(a, axis=axis, dtype=dtype, out=out)


@handle_jax_dtype
@to_aikit_arrays_and_back
def nancumsum(a, axis=None, dtype=None, out=None):
    a = aikit.where(aikit.isnan(a), aikit.zeros_like(a), a)
    return aikit.cumsum(a, axis=axis, dtype=dtype, out=out)


@to_aikit_arrays_and_back
def nanmax(
    a,
    axis=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    nan_mask = aikit.isnan(a)
    a = aikit.where(aikit.logical_not(nan_mask), a, a.full_like(-aikit.inf))
    where_mask = None
    if initial is not None:
        if aikit.is_array(where):
            a = aikit.where(where, a, a.full_like(initial))
            where_mask = aikit.all(aikit.logical_not(where), axis=axis, keepdims=keepdims)
        s = aikit.shape(a, as_array=True)
        if axis is not None:
            if isinstance(axis, (tuple, list)) or aikit.is_array(axis):
                # introducing the initial in one dimension is enough
                ax = axis[0] % len(s)
            else:
                ax = axis % len(s)
            s[ax] = aikit.array(1)
        header = aikit.full(aikit.Shape(s.to_list()), initial, dtype=aikit.dtype(a))
        if axis:
            if isinstance(axis, (tuple, list)) or aikit.is_array(axis):
                a = aikit.concat([a, header], axis=axis[0])
            else:
                a = aikit.concat([a, header], axis=axis)
        else:
            a = aikit.concat([a, header], axis=0)
    res = aikit.max(a, axis=axis, keepdims=keepdims, out=out)
    if nan_mask is not None:
        nan_mask = aikit.all(nan_mask, axis=axis, keepdims=keepdims, out=out)
        if aikit.any(nan_mask):
            res = aikit.where(
                aikit.logical_not(nan_mask),
                res,
                initial if initial is not None else aikit.nan,
                out=out,
            )
    if where_mask is not None and aikit.any(where_mask):
        res = aikit.where(aikit.logical_not(where_mask), res, aikit.nan, out=out)
    return res.astype(aikit.dtype(a))


@handle_jax_dtype
@to_aikit_arrays_and_back
def nanmean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if dtype is None:
        dtype = "float64" if aikit.is_int_dtype(a) else a.dtype
    if aikit.is_array(where):
        where1 = aikit.array(where, dtype=aikit.bool)
        a = aikit.where(where1, a, aikit.full_like(a, aikit.nan))
    nan_mask1 = aikit.isnan(a)
    not_nan_mask1 = ~aikit.isnan(a)
    b1 = aikit.where(aikit.logical_not(nan_mask1), a, aikit.zeros_like(a))
    array_sum1 = aikit.sum(b1, axis=axis, dtype=dtype, keepdims=keepdims, out=out)
    not_nan_mask_count1 = aikit.sum(
        not_nan_mask1, axis=axis, dtype=dtype, keepdims=keepdims, out=out
    )
    count_zero_handel = aikit.where(
        not_nan_mask_count1 != 0,
        not_nan_mask_count1,
        aikit.full_like(not_nan_mask_count1, aikit.nan),
    )
    return aikit.divide(array_sum1, count_zero_handel)


@to_aikit_arrays_and_back
def nanmedian(
    a,
    /,
    *,
    axis=None,
    keepdims=False,
    out=None,
    overwrite_input=False,
):
    return aikit.nanmedian(
        a, axis=axis, keepdims=keepdims, out=out, overwrite_input=overwrite_input
    ).astype(a.dtype)


@to_aikit_arrays_and_back
def nanmin(
    a,
    axis=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    nan_mask = aikit.isnan(a)
    a = aikit.where(aikit.logical_not(nan_mask), a, a.full_like(+aikit.inf))
    where_mask = None
    if initial is not None:
        if aikit.is_array(where):
            a = aikit.where(where, a, a.full_like(initial))
            where_mask = aikit.all(aikit.logical_not(where), axis=axis, keepdims=keepdims)
        s = aikit.shape(a, as_array=True)
        if axis is not None:
            if isinstance(axis, (tuple, list)) or aikit.is_array(axis):
                # introducing the initial in one dimension is enough
                ax = axis[0] % len(s)
            else:
                ax = axis % len(s)

            s[ax] = aikit.array(1)
        header = aikit.full(aikit.Shape(s.to_list()), initial, dtype=aikit.dtype(a))
        if axis:
            if isinstance(axis, (tuple, list)) or aikit.is_array(axis):
                a = aikit.concat([a, header], axis=axis[0])
            else:
                a = aikit.concat([a, header], axis=axis)
        else:
            a = aikit.concat([a, header], axis=0)
    res = aikit.min(a, axis=axis, keepdims=keepdims, out=out)
    if nan_mask is not None:
        nan_mask = aikit.all(nan_mask, axis=axis, keepdims=keepdims, out=out)
        if aikit.any(nan_mask):
            res = aikit.where(
                aikit.logical_not(nan_mask),
                res,
                initial if initial is not None else aikit.nan,
                out=out,
            )
    if where_mask is not None and aikit.any(where_mask):
        res = aikit.where(aikit.logical_not(where_mask), res, aikit.nan, out=out)
    return res.astype(aikit.dtype(a))


@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {"0.4.14 and below": ("complex64", "complex128", "bfloat16", "bool", "float16")},
    "jax",
)
def nanpercentile(
    a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=None
):
    def _remove_nan_1d(arr1d, overwrite_input=False):
        if arr1d.dtype == object:
            c = aikit.not_equal(arr1d, arr1d)
        else:
            c = aikit.isnan(arr1d)
        s = aikit.nonzero(c)[0]
        if s.size == arr1d.size:
            return arr1d[:0], True
        elif s.size == 0:
            return arr1d, overwrite_input
        else:
            if not overwrite_input:
                arr1d = arr1d.copy()

                enonan = arr1d[-s.size :][~c[-s.size :]]
                arr1d[s[: enonan.size]] = enonan

                return arr1d[: -s.size], True

    def _nanquantile_1d(arr1d, q, overwrite_input=False, method="linear"):
        arr1d, overwrite_input = _remove_nan_1d(arr1d, overwrite_input=overwrite_input)
        if arr1d.size == 0:
            return aikit.full(q.shape, aikit.nan)
        return aikit.quantile(arr1d, q, interpolation=method)

    def apply_along_axis(func1d, axis, arr, *args, **kwargs):
        ndim = aikit.get_num_dims(arr)
        if axis is None:
            raise ValueError("Axis must be an integer.")
        if not -ndim <= axis < ndim:
            raise ValueError(
                f"axis {axis} is out of bounds for array of dimension {ndim}"
            )
        if axis < 0:
            axis = axis + ndim

        func = lambda elem: func1d(elem, *args, **kwargs)
        for i in range(1, ndim - axis):
            func = aikit.vmap(func, in_axes=i, out_axes=-1)
        for i in range(axis):
            func = aikit.vmap(func, in_axes=0, out_axes=0)

        return aikit.asarray(func(arr))

    def _nanquantile_ureduce_func(
        a, q, axis=None, out=None, overwrite_input=False, method="linear"
    ):
        if axis is None or a.ndim == 1:
            part = a.ravel()
            result = _nanquantile_1d(
                part, q, overwrite_input=overwrite_input, method=method
            )
        else:
            result = apply_along_axis(
                _nanquantile_1d, axis, a, q, overwrite_input, method
            )

            if q.ndim != 0:
                result = aikit.moveaxis(result, axis, 0)

        if out is not None:
            out[...] = result

        return result

    def _ureduce(a, func, keepdims=False, **kwargs):
        axis = kwargs.get("axis", None)
        out = kwargs.get("out", None)

        if keepdims is None:
            keepdims = False

        nd = a.ndim
        if axis is not None:
            axis = _normalize_axis_tuple(axis, nd)

            if keepdims:
                if out is not None:
                    index_out = tuple(
                        0 if i in axis else slice(None) for i in range(nd)
                    )
                    kwargs["out"] = out[(Ellipsis,) + index_out]

            if len(axis) == 1:
                kwargs["axis"] = axis[0]
            else:
                keep = set(range(nd)) - set(axis)
                nkeep = len(keep)
                # swap axis that should not be reduced to front
                for i, s in enumerate(sorted(keep)):
                    a = a.swapaxes(i, s)
                # merge reduced axis
                a = a.reshape(a.shape[:nkeep] + (-1,))
                kwargs["axis"] = -1
        else:
            if keepdims:
                if out is not None:
                    index_out = (0,) * nd
                    kwargs["out"] = out[(Ellipsis,) + index_out]

        r = func(a, **kwargs)

        if out is not None:
            return out

        if keepdims:
            if axis is None:
                index_r = (aikit.newaxis,) * nd
            else:
                index_r = tuple(
                    aikit.newaxis if i in axis else slice(None) for i in range(nd)
                )
            r = r[(Ellipsis,) + index_r]

        return r

    def _nanquantile_unchecked(
        a,
        q,
        axis=None,
        out=None,
        overwrite_input=False,
        method="linear",
        keepdims=None,
    ):
        """Assumes that q is in [0, 1], and is an ndarray."""
        if a.size == 0:
            return aikit.nanmean(a, axis=axis, out=out, keepdims=keepdims)
        return _ureduce(
            a,
            func=_nanquantile_ureduce_func,
            q=q,
            keepdims=keepdims,
            axis=axis,
            out=out,
            overwrite_input=overwrite_input,
            method=method,
        )

    a = aikit.array(a)
    q = aikit.divide(q, 100.0)
    q = aikit.array(q)
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if not (0.0 <= q[i] <= 1.0):
                aikit.logging.warning("percentile s must be in the range [0, 100]")
                return []
    else:
        if not (aikit.all(q >= 0) and aikit.all(q <= 1)):
            aikit.logging.warning("percentile s must be in the range [0, 100]")
            return []
    return _nanquantile_unchecked(a, q, axis, out, overwrite_input, method, keepdims)


@handle_jax_dtype
@to_aikit_arrays_and_back
def nanstd(
    a, /, *, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True
):
    a = aikit.nan_to_num(a)
    axis = tuple(axis) if isinstance(axis, list) else axis

    if dtype:
        a = aikit.astype(aikit.array(a), aikit.as_aikit_dtype(dtype))

    ret = aikit.std(a, axis=axis, correction=ddof, keepdims=keepdims, out=out)
    if aikit.is_array(where):
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)

    return ret


@handle_jax_dtype
@to_aikit_arrays_and_back
def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True):
    is_nan = aikit.isnan(a)
    if dtype is None:
        dtype = "float16" if aikit.is_int_dtype(a) else a.dtype
    if aikit.any(is_nan):
        a = [i for i in a if aikit.isnan(i) is False]

    if dtype:
        a = aikit.astype(aikit.array(a), aikit.as_aikit_dtype(dtype))

    ret = aikit.var(a, axis=axis, correction=ddof, keepdims=keepdims, out=out)
    if aikit.is_array(where):
        where = aikit.array(where, dtype=aikit.bool)
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)

    all_nan = aikit.isnan(ret)
    if aikit.all(all_nan):
        ret = aikit.astype(ret, aikit.array([float("inf")]))
    return ret


@to_aikit_arrays_and_back
def ptp(a, axis=None, out=None, keepdims=False):
    x = aikit.max(a, axis=axis, keepdims=keepdims)
    y = aikit.min(a, axis=axis, keepdims=keepdims)
    return aikit.subtract(x, y)


@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {"0.4.21 and below": ("complex64", "complex128", "bfloat16", "bool", "float16")},
    "jax",
)
def quantile(
    a,
    q,
    /,
    *,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    interpolation=None,
):
    if method == "nearest":
        return aikit.quantile(
            a, q, axis=axis, keepdims=keepdims, interpolation="nearest_jax", out=out
        )
    return aikit.quantile(
        a, q, axis=axis, keepdims=keepdims, interpolation=method, out=out
    )


@handle_jax_dtype
@with_unsupported_dtypes({"0.4.21 and below": ("bfloat16",)}, "jax")
@to_aikit_arrays_and_back
def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if dtype is None:
        dtype = "float32" if aikit.is_int_dtype(a) else a.dtype
    std_a = aikit.std(a, axis=axis, correction=ddof, keepdims=keepdims, out=out)
    if aikit.is_array(where):
        where = aikit.array(where, dtype=aikit.bool)
        std_a = aikit.where(
            where, std_a, aikit.default(out, aikit.zeros_like(std_a)), out=out
        )
    return aikit.astype(std_a, aikit.as_aikit_dtype(dtype), copy=False)


@handle_jax_dtype
@to_aikit_arrays_and_back
def sum(
    a,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=None,
    promote_integers=True,
):
    # TODO: promote_integers is only supported from JAX v0.4.10
    if dtype is None and promote_integers:
        if aikit.is_bool_dtype(a.dtype):
            dtype = aikit.default_int_dtype()
        elif aikit.is_uint_dtype(a.dtype):
            dtype = "uint64"
            a = aikit.astype(a, dtype)
        elif aikit.is_int_dtype(a.dtype):
            dtype = "int64"
            a = aikit.astype(a, dtype)
        else:
            dtype = a.dtype
    elif dtype is None and not promote_integers:
        dtype = "float32" if aikit.is_int_dtype(a.dtype) else aikit.as_aikit_dtype(a.dtype)

    if initial:
        if axis is None:
            a = aikit.reshape(a, (1, -1))
            axis = 0
        s = list(aikit.shape(a))
        s[axis] = 1
        header = aikit.full(s, initial)
        a = aikit.concat([a, header], axis=axis)

    ret = aikit.sum(a, axis=axis, keepdims=keepdims, out=out)

    if aikit.is_array(where):
        where = aikit.array(where, dtype=aikit.bool)
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return aikit.astype(ret, aikit.as_aikit_dtype(dtype))


@handle_jax_dtype
@to_aikit_arrays_and_back
def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if dtype is None:
        dtype = "float32" if aikit.is_int_dtype(a) else a.dtype
    ret = aikit.var(a, axis=axis, correction=ddof, keepdims=keepdims, out=out)
    if aikit.is_array(where):
        where = aikit.array(where, dtype=aikit.bool)
        ret = aikit.where(where, ret, aikit.default(out, aikit.zeros_like(ret)), out=out)
    return aikit.astype(ret, aikit.as_aikit_dtype(dtype), copy=False)


amax = max
amin = min
cumproduct = cumprod
