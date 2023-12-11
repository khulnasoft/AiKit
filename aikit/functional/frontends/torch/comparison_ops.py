# global
import aikit
import aikit.functional.frontends.torch as torch_frontend
from aikit.func_wrapper import with_unsupported_dtypes
from aikit.functional.frontends.torch.func_wrapper import to_aikit_arrays_and_back

# local
from collections import namedtuple


# --- Helpers --- #
# --------------- #


def _compute_allclose_with_tol(input, other, rtol, atol):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.all(
        aikit.less_equal(
            aikit.abs(aikit.subtract(input, other)),
            aikit.add(atol, aikit.multiply(rtol, aikit.abs(other))),
        )
    )


def _compute_isclose_with_tol(input, other, rtol, atol):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.less_equal(
        aikit.abs(aikit.subtract(input, other)),
        aikit.add(atol, aikit.multiply(rtol, aikit.abs(other))),
    )


# --- Main --- #
# ------------ #


@to_aikit_arrays_and_back
def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    finite_input = aikit.isfinite(input)
    finite_other = aikit.isfinite(other)
    if aikit.all(finite_input) and aikit.all(finite_other):
        ret = _compute_allclose_with_tol(input, other, rtol, atol)
        return aikit.all_equal(True, ret)
    else:
        finites = aikit.bitwise_and(finite_input, finite_other)
        ret = aikit.zeros_like(finites)
        ret_ = ret.astype(int)
        input = input * aikit.ones_like(ret_)
        other = other * aikit.ones_like(ret_)
        ret[finites] = _compute_allclose_with_tol(
            input[finites], other[finites], rtol, atol
        )
        nans = aikit.bitwise_invert(finites)
        ret[nans] = aikit.equal(input[nans], other[nans])
        if equal_nan:
            both_nan = aikit.bitwise_and(aikit.isnan(input), aikit.isnan(other))
            ret[both_nan] = both_nan[both_nan]
        return aikit.all(ret)


@to_aikit_arrays_and_back
def argsort(input, dim=-1, descending=False):
    return aikit.argsort(input, axis=dim, descending=descending)


@to_aikit_arrays_and_back
def eq(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.equal(input, other, out=out)


@to_aikit_arrays_and_back
def equal(input, other):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.all_equal(input, other)


@to_aikit_arrays_and_back
def fmax(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.where(
        aikit.bitwise_or(aikit.greater(input, other), aikit.isnan(other)),
        input,
        other,
        out=out,
    )


@to_aikit_arrays_and_back
def fmin(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.where(
        aikit.bitwise_or(aikit.less(input, other), aikit.isnan(other)),
        input,
        other,
        out=out,
    )


@with_unsupported_dtypes({"2.1.1 and below": ("complex64", "complex128")}, "torch")
@to_aikit_arrays_and_back
def greater(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.greater(input, other, out=out)


@with_unsupported_dtypes({"2.1.1 and below": ("complex64", "complex128")}, "torch")
@to_aikit_arrays_and_back
def greater_equal(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.greater_equal(input, other, out=out)


@to_aikit_arrays_and_back
def isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    finite_input = aikit.isfinite(input)
    finite_other = aikit.isfinite(other)
    if aikit.all(finite_input) and aikit.all(finite_other):
        return _compute_isclose_with_tol(input, other, rtol, atol)

    else:
        finites = aikit.bitwise_and(finite_input, finite_other)
        ret = aikit.zeros_like(finites)
        ret_ = ret.astype(int)
        input = input * aikit.ones_like(ret_)
        other = other * aikit.ones_like(ret_)
        ret[finites] = _compute_isclose_with_tol(
            input[finites], other[finites], rtol, atol
        )
        nans = aikit.bitwise_invert(finites)
        ret[nans] = aikit.equal(input[nans], other[nans])
        if equal_nan:
            both_nan = aikit.bitwise_and(aikit.isnan(input), aikit.isnan(other))
            ret[both_nan] = both_nan[both_nan]
        return ret


@to_aikit_arrays_and_back
def isfinite(input):
    return aikit.isfinite(input)


@with_unsupported_dtypes({"2.1.1 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def isin(elements, test_elements, *, assume_unique=False, invert=False):
    input_elements_copy = aikit.reshape(aikit.to_aikit(elements), (-1,))
    test_elements_copy = aikit.reshape(aikit.to_aikit(test_elements), (-1,))

    if (
        aikit.shape(test_elements_copy)[0]
        < 10 * aikit.shape(input_elements_copy)[0] ** 0.145
    ):
        if invert:
            mask = aikit.ones(aikit.shape(input_elements_copy[0]), dtype=bool)
            for a in test_elements_copy:
                mask &= input_elements_copy != a
        else:
            mask = aikit.zeros(aikit.shape(input_elements_copy[0]), dtype=bool)
            for a in test_elements_copy:
                mask |= input_elements_copy == a
        return aikit.reshape(mask, aikit.shape(elements))

    if not assume_unique:
        input_elements_copy, rev_idx = aikit.unique_inverse(input_elements_copy)
        test_elements_copy = aikit.sort(aikit.unique_values(test_elements_copy))

    ar = aikit.concat((input_elements_copy, test_elements_copy))

    order = aikit.argsort(ar, stable=True)
    sar = ar[order]
    if invert:
        bool_ar = sar[1:] != sar[:-1]
    else:
        bool_ar = sar[1:] == sar[:-1]
    flag = aikit.concat((bool_ar, aikit.array([invert])))
    ret = aikit.empty(aikit.shape(ar), dtype=bool)
    ret[order] = flag

    if assume_unique:
        return aikit.reshape(
            ret[: aikit.shape(input_elements_copy)[0]], aikit.shape(elements)
        )
    else:
        return aikit.reshape(ret[rev_idx], aikit.shape(elements))


@to_aikit_arrays_and_back
def isinf(input):
    return aikit.isinf(input)


@to_aikit_arrays_and_back
def isnan(input):
    return aikit.isnan(input)


@to_aikit_arrays_and_back
def isneginf(input, *, out=None):
    is_inf = aikit.isinf(input)
    neg_sign_bit = aikit.less(input, 0)
    return aikit.logical_and(is_inf, neg_sign_bit, out=out)


@to_aikit_arrays_and_back
def isposinf(input, *, out=None):
    is_inf = aikit.isinf(input)
    pos_sign_bit = aikit.bitwise_invert(aikit.less(input, 0))
    return aikit.logical_and(is_inf, pos_sign_bit, out=out)


@with_unsupported_dtypes({"2.1.1 and below": ("bfloat16",)}, "torch")
@to_aikit_arrays_and_back
def isreal(input):
    return aikit.isreal(input)


@with_unsupported_dtypes(
    {"2.1.1 and below": ("bfloat16", "float16", "bool", "complex")}, "torch"
)
@to_aikit_arrays_and_back
def kthvalue(input, k, dim=-1, keepdim=False, *, out=None):
    sorted_input = aikit.sort(input, axis=dim)
    sort_indices = aikit.argsort(input, axis=dim)

    values = aikit.asarray(
        aikit.gather(sorted_input, aikit.array(k - 1), axis=dim), dtype=input.dtype
    )
    indices = aikit.asarray(
        aikit.gather(sort_indices, aikit.array(k - 1), axis=dim), dtype="int64"
    )

    if keepdim:
        values = aikit.expand_dims(values, axis=dim)
        indices = aikit.expand_dims(indices, axis=dim)

    ret = namedtuple("sort", ["values", "indices"])(values, indices)
    if aikit.exists(out):
        return aikit.inplace_update(out, ret)
    return ret


@with_unsupported_dtypes({"2.1.1 and below": ("complex64", "complex128")}, "torch")
@to_aikit_arrays_and_back
def less(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.less(input, other, out=out)


@with_unsupported_dtypes({"2.1.1 and below": ("complex64", "complex128")}, "torch")
@to_aikit_arrays_and_back
def less_equal(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.less_equal(input, other, out=out)


@to_aikit_arrays_and_back
def maximum(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.maximum(input, other, out=out)


@to_aikit_arrays_and_back
def minimum(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.minimum(input, other, out=out)


@to_aikit_arrays_and_back
def msort(input, *, out=None):
    return aikit.sort(input, axis=0, out=out)


@with_unsupported_dtypes({"2.1.1 and below": ("float16", "bfloat16")}, "torch")
@to_aikit_arrays_and_back
def not_equal(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return aikit.not_equal(input, other, out=out)


@to_aikit_arrays_and_back
# TODO: the original torch.sort places * right before `out`
def sort(input, *, dim=-1, descending=False, stable=False, out=None):
    values = aikit.sort(input, axis=dim, descending=descending, stable=stable, out=out)
    indices = aikit.argsort(input, axis=dim, descending=descending)
    return namedtuple("sort", ["values", "indices"])(values, indices)


@with_unsupported_dtypes({"2.1.1 and below": ("float16", "complex")}, "torch")
@to_aikit_arrays_and_back
def topk(input, k, dim=None, largest=True, sorted=True, *, out=None):
    if dim is None:
        dim = -1
    return aikit.top_k(input, k, axis=dim, largest=largest, sorted=sorted, out=out)


ge = greater_equal
gt = greater
le = less_equal
lt = less
ne = not_equal
