# local
import aikit
from aikit.functional.frontends.jax.func_wrapper import (
    to_aikit_arrays_and_back,
)
from aikit.func_wrapper import with_unsupported_dtypes
from aikit.functional.frontends.jax.numpy import promote_types_of_jax_inputs
from aikit.functional.frontends.numpy.manipulation_routines import trim_zeros
from aikit.utils.einsum_path_helpers import (
    parse_einsum_input,
    compute_size_by_dict,
    flop_count,
    greedy_path,
    optimal_path,
    find_contraction,
    can_dot,
)


@to_aikit_arrays_and_back
def absolute(x, /):
    return aikit.abs(x)


@to_aikit_arrays_and_back
def add(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.add(x1, x2)


@to_aikit_arrays_and_back
def angle(z, deg=False):
    return aikit.angle(z, deg=deg)


@to_aikit_arrays_and_back
def arccos(x, /):
    return aikit.acos(x)


@to_aikit_arrays_and_back
def arccosh(x, /):
    return aikit.acosh(x)


@to_aikit_arrays_and_back
def arcsin(x, /):
    return aikit.asin(x)


@to_aikit_arrays_and_back
def arcsinh(x, /):
    return aikit.asinh(x)


@to_aikit_arrays_and_back
def arctan(x, /):
    return aikit.atan(x)


@to_aikit_arrays_and_back
def arctan2(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.atan2(x1, x2)


@to_aikit_arrays_and_back
def arctanh(x, /):
    return aikit.atanh(x)


@to_aikit_arrays_and_back
def around(a, decimals=0, out=None):
    ret_dtype = a.dtype
    return aikit.round(a, decimals=decimals, out=out).astype(ret_dtype, copy=False)


@with_unsupported_dtypes(
    {"0.4.23 and below": ("bfloat16",)},
    "jax",
)
@to_aikit_arrays_and_back
def cbrt(x, /):
    all_positive = aikit.pow(aikit.abs(x), 1.0 / 3.0)
    return aikit.where(aikit.less(x, 0.0), aikit.negative(all_positive), all_positive)


@to_aikit_arrays_and_back
def ceil(x, /):
    return aikit.ceil(x)


@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
@to_aikit_arrays_and_back
def clip(a, a_min=None, a_max=None, out=None):
    return aikit.array(aikit.clip(a, a_min, a_max), dtype=a.dtype)


@to_aikit_arrays_and_back
def conj(x, /):
    return aikit.conj(x)


@to_aikit_arrays_and_back
def conjugate(x, /):
    return aikit.conj(x)


@to_aikit_arrays_and_back
def convolve(a, v, mode="full", *, precision=None):
    a, v = promote_types_of_jax_inputs(a, v)

    if len(a) < len(v):
        a, v = v, a
    v = aikit.flip(v)

    out_order = slice(None)

    if mode == "valid":
        padding = [(0, 0)]
    elif mode == "same":
        padding = [(v.shape[0] // 2, v.shape[0] - v.shape[0] // 2 - 1)]
    elif mode == "full":
        padding = [(v.shape[0] - 1, v.shape[0] - 1)]

    a = a.reshape([1, 1, a.shape[0]])
    v = v.reshape([v.shape[0], 1, 1])

    result = aikit.conv_general_dilated(
        a,
        v,
        (1,),
        padding,
        dims=1,
        data_format="channel_first",
    )
    return result[0, 0, out_order]


@to_aikit_arrays_and_back
def copysign(x1, x2, /):
    return aikit.copysign(x1, x2)


@to_aikit_arrays_and_back
def cos(x, /):
    return aikit.cos(x)


@to_aikit_arrays_and_back
def cosh(x, /):
    return aikit.cosh(x)


@to_aikit_arrays_and_back
def deg2rad(x, /):
    return aikit.deg2rad(x)


@to_aikit_arrays_and_back
def degrees(x, /):
    return aikit.rad2deg(x)


@to_aikit_arrays_and_back
def diff(a, n=1, axis=-1, prepend=None, append=None):
    return aikit.diff(a, n=n, axis=axis, prepend=prepend, append=append, out=None)


@to_aikit_arrays_and_back
def divide(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    if aikit.dtype(x1) in ["int64", "uint64"]:
        x1 = aikit.astype(x1, aikit.float64)
    elif aikit.is_int_dtype(x1):
        x1 = aikit.astype(x1, aikit.float32)

    return aikit.divide(x1, x2).astype(x1.dtype)


@to_aikit_arrays_and_back
def divmod(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return (aikit.floor_divide(x1, x2), aikit.remainder(x1, x2))


@to_aikit_arrays_and_back
def dot(a, b, *, precision=None):
    a, b = promote_types_of_jax_inputs(a, b)
    return aikit.matmul(a, b)


@to_aikit_arrays_and_back
def ediff1d(ary, to_end=None, to_begin=None):
    diffs = aikit.diff(ary)
    diffs_dtype = diffs.dtype
    if to_begin is not None:
        if not isinstance(to_begin, (list, tuple)):
            to_begin = [to_begin]
        to_begin = aikit.array(to_begin, dtype=diffs_dtype)
        diffs = aikit.concat((to_begin, diffs))
    if to_end is not None:
        if not isinstance(to_end, (list, tuple)):
            to_end = [to_end]
        to_end = aikit.array(to_end, dtype=diffs_dtype)
        diffs = aikit.concat((diffs, to_end))
    return diffs


@to_aikit_arrays_and_back
def einsum_path(subscripts, *operands, optimize="greedy"):
    # Figure out what the path really is
    path_type = optimize
    if path_type is True:
        path_type = "greedy"
    if path_type is None:
        path_type = False

    explicit_einsum_path = False
    memory_limit = None

    # No optimization or a named path algorithm
    if (path_type is False) or isinstance(path_type, str):
        pass

    # Given an explicit path
    elif len(path_type) and (path_type[0] == "einsum_path"):
        explicit_einsum_path = True

    # Path tuple with memory limit
    elif (
        (len(path_type) == 2)
        and isinstance(path_type[0], str)
        and isinstance(path_type[1], (int, float))
    ):
        memory_limit = int(path_type[1])
        path_type = path_type[0]

    else:
        raise TypeError(f"Did not understand the path: {str(path_type)}")

    # Python side parsing
    if subscripts:
        input_subscripts, output_subscript, operands = parse_einsum_input(
            operands, subscripts=subscripts
        )
    else:
        input_subscripts, output_subscript, operands = parse_einsum_input(operands)

    # Build a few useful list and sets
    input_list = input_subscripts.split(",")
    input_sets = [set(x) for x in input_list]
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(",", ""))

    # Get length of each unique dimension and ensure all dimensions are correct
    dimension_dict = {}
    broadcast_indices = [[] for x in range(len(input_list))]
    for tnum, term in enumerate(input_list):
        sh = operands[tnum].shape
        if len(sh) != len(term):
            raise ValueError(
                "Einstein sum subscript %s does not contain the "
                "correct number of indices for operand %d."
                % (input_subscripts[tnum], tnum)
            )
        for cnum, char in enumerate(term):
            dim = sh[cnum]

            # Build out broadcast indices
            if dim == 1:
                broadcast_indices[tnum].append(char)

            if char in dimension_dict.keys():
                # For broadcasting cases we always want the largest dim size
                if dimension_dict[char] == 1:
                    dimension_dict[char] = dim
                elif dim not in (1, dimension_dict[char]):
                    raise ValueError(
                        "Size of label '%s' for operand %d (%d) "
                        "does not match previous terms (%d)."
                        % (char, tnum, dimension_dict[char], dim)
                    )
            else:
                dimension_dict[char] = dim

    # Convert broadcast inds to sets
    broadcast_indices = [set(x) for x in broadcast_indices]

    # Compute size of each input array plus the output array
    size_list = [
        compute_size_by_dict(term, dimension_dict)
        for term in input_list + [output_subscript]
    ]
    max_size = max(size_list)

    if memory_limit is None:
        memory_arg = max_size
    else:
        memory_arg = memory_limit

    # Compute naive cost
    # This isn't quite right, need to look into exactly how einsum does this
    inner_product = (sum(len(x) for x in input_sets) - len(indices)) > 0
    naive_cost = flop_count(indices, inner_product, len(input_list), dimension_dict)

    # Compute the path
    if explicit_einsum_path:
        path = path_type[1:]
    elif (path_type is False) or (len(input_list) in [1, 2]) or (indices == output_set):
        # Nothing to be optimized, leave it to einsum
        path = [tuple(range(len(input_list)))]
    elif path_type == "greedy":
        path = greedy_path(input_sets, output_set, dimension_dict, memory_arg)
    elif path_type == "optimal":
        path = optimal_path(input_sets, output_set, dimension_dict, memory_arg)
    else:
        raise KeyError(f"Path name {path_type} not found")

    cost_list, scale_list, size_list, contraction_list = [], [], [], []

    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    for cnum, contract_inds in enumerate(path):
        # Make sure we remove inds from right to left
        contract_inds = tuple(sorted(contract_inds, reverse=True))

        contract = find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract

        cost = flop_count(idx_contract, idx_removed, len(contract_inds), dimension_dict)
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(compute_size_by_dict(out_inds, dimension_dict))

        bcast = set()
        tmp_inputs = []
        for x in contract_inds:
            tmp_inputs.append(input_list.pop(x))
            bcast |= broadcast_indices.pop(x)

        new_bcast_inds = bcast - idx_removed

        # If we're broadcasting, nix blas
        if not len(idx_removed & bcast):
            do_blas = can_dot(tmp_inputs, out_inds, idx_removed)
        else:
            do_blas = False

        # Last contraction
        if (cnum - len(path)) == -1:
            idx_result = output_subscript
        else:
            sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
            idx_result = "".join([x[1] for x in sorted(sort_result)])

        input_list.append(idx_result)
        broadcast_indices.append(new_bcast_inds)
        einsum_str = ",".join(tmp_inputs) + "->" + idx_result

        contraction = (contract_inds, idx_removed, einsum_str, input_list[:], do_blas)
        contraction_list.append(contraction)

    opt_cost = sum(cost_list) + 1

    if len(input_list) != 1:
        # Explicit "einsum_path" is usually trusted, but we detect this kind of
        # mistake in order to prevent from returning an intermediate value.
        raise RuntimeError(
            f"Invalid einsum_path is specified: {len(input_list) - 1} "
            "more operands has to be contracted."
        )

    # Return the path along with a nice string representation
    overall_contraction = input_subscripts + "->" + output_subscript
    header = ("scaling", "current", "remaining")

    speedup = naive_cost / opt_cost
    max_i = max(size_list)

    path_print = f"  Complete contraction:  {overall_contraction}\n"
    path_print += f"         Naive scaling:  {len(indices)}\n"
    path_print += f"     Optimized scaling:  {max(scale_list)}\n"
    path_print += f"      Naive FLOP count:  {naive_cost:.3e}\n"
    path_print += f"  Optimized FLOP count:  {opt_cost:.3e}\n"
    path_print += f"   Theoretical speedup:  {speedup:3.3f}\n"
    path_print += f"  Largest intermediate:  {max_i:.3e} elements\n"
    path_print += "-" * 74 + "\n"
    path_print += "%6s %24s %40s\n" % header
    path_print += "-" * 74

    for n, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, remaining, blas = contraction
        remaining_str = ",".join(remaining) + "->" + output_subscript
        path_run = (scale_list[n], einsum_str, remaining_str)
        path_print += "\n%4d    %24s %40s" % path_run

    ret = (path, path_print)
    return ret


@to_aikit_arrays_and_back
def exp(
    x,
    /,
):
    return aikit.exp(x)


@to_aikit_arrays_and_back
def exp2(x, /):
    return aikit.exp2(x)


@to_aikit_arrays_and_back
def expm1(
    x,
    /,
):
    return aikit.expm1(x)


@with_unsupported_dtypes(
    {"0.4.23 and below": ("uint16",)},
    "jax",
)
@to_aikit_arrays_and_back
def fabs(x, /):
    return aikit.abs(x)


@to_aikit_arrays_and_back
def fix(x, out=None):
    return aikit.fix(x, out=out)


@to_aikit_arrays_and_back
def float_power(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.float_power(x1, x2).astype(x1.dtype, copy=False)


@to_aikit_arrays_and_back
def floor(x, /):
    return aikit.floor(x)


@to_aikit_arrays_and_back
def floor_divide(x1, x2, /, out=None):
    return aikit.floor_divide(x1, x2, out=out)


@to_aikit_arrays_and_back
def fmax(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.fmax(x1, x2)


@to_aikit_arrays_and_back
def fmin(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.fmin(x1, x2)


@to_aikit_arrays_and_back
def fmod(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.fmod(x1, x2)


@to_aikit_arrays_and_back
def frexp(x, /):
    return aikit.frexp(x)


@to_aikit_arrays_and_back
def gcd(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.gcd(x1, x2)


@to_aikit_arrays_and_back
def gradient(f, *varargs, axis=None, edge_order=None):
    edge_order = edge_order if edge_order is not None else 1
    return aikit.gradient(f, spacing=varargs, axis=axis, edge_order=edge_order)


@to_aikit_arrays_and_back
def heaviside(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.heaviside(x1, x2)


@to_aikit_arrays_and_back
def hypot(x1, x2, /):
    return aikit.hypot(x1, x2)


@to_aikit_arrays_and_back
def i0(x):
    return aikit.i0(x)


@to_aikit_arrays_and_back
def imag(val, /):
    return aikit.imag(val)


@to_aikit_arrays_and_back
def inner(a, b):
    a, b = promote_types_of_jax_inputs(a, b)
    return aikit.inner(a, b)


@to_aikit_arrays_and_back
def interp(x, xp, fp, left=None, right=None, period=None):
    return aikit.interp(x, xp, fp, left=left, right=right, period=period)


@to_aikit_arrays_and_back
def kron(a, b):
    a, b = promote_types_of_jax_inputs(a, b)
    return aikit.kron(a, b)


@to_aikit_arrays_and_back
def lcm(x1, x2):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.lcm(x1, x2)


@to_aikit_arrays_and_back
def ldexp(x1, x2, /):
    return aikit.ldexp(x1, x2)


@to_aikit_arrays_and_back
def log(x, /):
    return aikit.log(x)


@to_aikit_arrays_and_back
def log10(x, /):
    return aikit.log10(x)


@to_aikit_arrays_and_back
def log1p(x, /):
    return aikit.log1p(x)


@to_aikit_arrays_and_back
def log2(x, /):
    return aikit.log2(x)


@to_aikit_arrays_and_back
def logaddexp(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.logaddexp(x1, x2)


@to_aikit_arrays_and_back
def logaddexp2(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.logaddexp2(x1, x2)


@to_aikit_arrays_and_back
def matmul(a, b, *, precision=None):
    a, b = promote_types_of_jax_inputs(a, b)
    return aikit.matmul(a, b)


@to_aikit_arrays_and_back
def maximum(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.maximum(x1, x2)


@to_aikit_arrays_and_back
def minimum(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.minimum(x1, x2)


@to_aikit_arrays_and_back
@with_unsupported_dtypes({"0.4.23 and below": ("complex",)}, "jax")
def mod(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.remainder(x1, x2)


@to_aikit_arrays_and_back
def modf(x, /, out=None):
    y1 = aikit.where(x >= 0, aikit.floor(x), aikit.ceil(x))  # integral part
    y2 = x - y1  # fractional part
    dtype_str = str(x.dtype)
    if "float" in dtype_str:
        return y2, y1
    # floats return as they were. u/ints (8, 16, 32) return as float32, 64 as float64.
    dtype_size = x.itemsize * 8
    if "int8" in dtype_str or "int16" in dtype_str:
        dtype_size = 32
    ret_type = f"float{dtype_size}"
    return y2.astype(ret_type), y1.astype(ret_type)


@to_aikit_arrays_and_back
def multiply(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.multiply(x1, x2)


@to_aikit_arrays_and_back
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    return aikit.nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf)


@to_aikit_arrays_and_back
def negative(
    x,
    /,
):
    return aikit.negative(x)


@with_unsupported_dtypes(
    {
        "0.4.23 and below": (
            "bfloat16",
            "float16",
        )
    },
    "jax",
)
@to_aikit_arrays_and_back
def nextafter(x1, x2, /):
    return aikit.nextafter(x1, x2)


@to_aikit_arrays_and_back
def outer(a, b, out=None):
    return aikit.outer(a, b, out=out)


@to_aikit_arrays_and_back
def poly(seq_of_zeros):
    seq_of_zeros = aikit.atleast_1d(seq_of_zeros)
    sh = seq_of_zeros.shape
    if len(sh) == 2 and sh[0] == sh[1] and sh[0] != 0:
        seq_of_zeros = aikit.eigvals(seq_of_zeros)
    if seq_of_zeros.ndim != 1:
        raise ValueError("input must be 1d or non-empty square 2d array.")
    dt = seq_of_zeros.dtype
    if len(seq_of_zeros) == 0:
        return aikit.ones((), dtype=dt)
    a = aikit.ones((1,), dtype=dt)
    for k in range(len(seq_of_zeros)):
        a = convolve(
            a, aikit.asarray([aikit.array(1), -seq_of_zeros[k]], dtype=dt), mode="full"
        )
    return a


@to_aikit_arrays_and_back
def polyadd(a1, a2):
    d = max(a1.size, a2.size)
    a1 = aikit.pad(a1, (d - a1.size, 0), mode="constant")
    a2 = aikit.pad(a2, (d - a2.size, 0), mode="constant")
    return a1 + a2


@with_unsupported_dtypes(
    {"0.4.23 and below": ("float16",)},
    "jax",
)
@to_aikit_arrays_and_back
def polyder(p, m=1):
    if m < 0:
        raise ValueError("Order of derivative must be positive.")

    if m == 0:
        return p
    p_dtype = p.dtype
    coeff = aikit.prod(
        aikit.expand_dims(aikit.arange(m, len(p), dtype=p_dtype))
        - aikit.expand_dims(aikit.arange(m, dtype=p_dtype), axis=1),
        axis=0,
    )
    return (p[:-m] * coeff[::-1]).astype(p_dtype)


@with_unsupported_dtypes(
    {"0.3.14 and below": ("float16",)},
    "jax",
)
@to_aikit_arrays_and_back
def polydiv(u, v, *, trim_leading_zeros=False):
    u, v_arr = aikit.promote_types_of_inputs(u, v)
    n = v_arr.shape[0] - 1
    m = u.shape[0] - 1
    scale = 1.0 / v_arr[0]
    q = aikit.zeros((max(m - n + 1, 1),), dtype=u.dtype)
    r = aikit.copy_array(u)
    for k in range(0, m - n + 1):
        d = scale * r[k]
        q[k] = d
        r[k : k + n + 1] = r[k : k + n + 1] - (d * v_arr)
    # if trim_leading_zeros:
    #    r = trim_zeros_tol(r, trim='f')
    # TODO: need to control tolerance of this function to handle the argument
    return q, r


@with_unsupported_dtypes(
    {"0.4.23 and below": ("float16",)},
    "jax",
)
@to_aikit_arrays_and_back
def polyint(p, m=1, k=None):
    p = aikit.asarray(p)
    m = int(m)
    if m == 0:
        return p
    if k is None:
        k_arr = aikit.zeros((m,), dtype=p.dtype)
    elif isinstance(k, (int, float)):
        k_arr = aikit.full((m,), k, dtype=p.dtype)
    elif aikit.asarray(k).shape == (1,):
        k_arr = aikit.full((m,), aikit.asarray(k)[0], dtype=p.dtype)
    elif aikit.asarray(k).shape == (m,):
        k_arr = aikit.asarray(k, dtype=p.dtype)
    else:
        raise ValueError("k must be a scalar or a rank-1 array of length 1 or m.")
    grid = (
        aikit.arange(p.size + m, dtype=p.dtype)[aikit.newaxis]
        - aikit.arange(m, dtype=p.dtype)[:, aikit.newaxis]
    )
    coeff = aikit.maximum(1, grid).prod(axis=0)[::-1]
    return aikit.divide(aikit.concat((p, k_arr)), coeff).astype(p.dtype)


@to_aikit_arrays_and_back
def polymul(a1, a2, *, trim_leading_zeros=False):
    a1, a2 = aikit.atleast_1d(a1), aikit.atleast_1d(a2)
    if trim_leading_zeros and (len(a1) > 1 or len(a2) > 1):
        a1, a2 = trim_zeros(a1, trim="f"), trim_zeros(a2, trim="f")
    if len(a1) == 0:
        a1 = aikit.asarray([0], dtype=a1.dtype)
    if len(a2) == 0:
        a2 = aikit.asarray([0], dtype=a2.dtype)
    return convolve(a1, a2, mode="full")


@to_aikit_arrays_and_back
def polysub(a1, a2):
    n = max(a1.size, a2.size) - 1
    a1 = aikit.pad(a1, (0, n - a1.size + 1), mode="constant")
    a2 = aikit.pad(a2, (0, n - a2.size + 1), mode="constant")
    return a1 - a2


@to_aikit_arrays_and_back
def positive(
    x,
    /,
):
    return aikit.positive(x)


@to_aikit_arrays_and_back
def power(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.pow(x1, x2)


@to_aikit_arrays_and_back
def product(
    a,
    *,
    axis=None,
    dtype=None,
    keepdims=False,
    initial=None,
    where=None,
    promote_integers=True,
    out=None,
):
    if aikit.is_array(where):
        a = aikit.where(where, a, aikit.default(out, aikit.ones_like(a)), out=out)
    if promote_integers:
        if aikit.is_uint_dtype(a.dtype):
            dtype = "uint64"
        elif aikit.is_int_dtype(a.dtype):
            dtype = "int64"
    if initial is not None:
        if axis is not None:
            s = aikit.to_list(aikit.shape(a, as_array=True))
            s[axis] = 1
            header = aikit.full(aikit.Shape(tuple(s)), initial)
            a = aikit.concat([header, a], axis=axis)
        else:
            a[0] *= initial
    return aikit.prod(a, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@to_aikit_arrays_and_back
def rad2deg(
    x,
    /,
):
    return aikit.rad2deg(x)


@to_aikit_arrays_and_back
def radians(x, /):
    return aikit.deg2rad(x)


@to_aikit_arrays_and_back
def real(val, /):
    return aikit.real(val)


@to_aikit_arrays_and_back
def reciprocal(x, /):
    return aikit.reciprocal(x)


@to_aikit_arrays_and_back
def remainder(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.remainder(x1, x2)


@to_aikit_arrays_and_back
def round(a, decimals=0, out=None):
    return aikit.round(a, decimals=decimals, out=out)


# sign
@to_aikit_arrays_and_back
def sign(x, /):
    return aikit.sign(x, out=None)


@to_aikit_arrays_and_back
def signbit(x, /):
    x = aikit.array(x)
    return aikit.signbit(x)


@to_aikit_arrays_and_back
def sin(x, /):
    return aikit.sin(x)


@to_aikit_arrays_and_back
def sinc(x, /):
    return aikit.sinc(x)


@to_aikit_arrays_and_back
def sinh(x, /):
    return aikit.sinh(x)


@to_aikit_arrays_and_back
def sqrt(x, /):
    return aikit.sqrt(x)


@to_aikit_arrays_and_back
def square(x, /):
    return aikit.square(x)


@to_aikit_arrays_and_back
def subtract(x1, x2, /):
    x1, x2 = promote_types_of_jax_inputs(x1, x2)
    return aikit.subtract(x1, x2)


@to_aikit_arrays_and_back
def tan(x, /):
    return aikit.tan(x)


@to_aikit_arrays_and_back
def tanh(x, /):
    return aikit.tanh(x)


@to_aikit_arrays_and_back
def tensordot(a, b, axes=2):
    a, b = promote_types_of_jax_inputs(a, b)
    return aikit.tensordot(a, b, axes=axes)


@to_aikit_arrays_and_back
def trace(a, offset=0, axis1=0, axis2=1, out=None):
    return aikit.trace(a, offset=offset, axis1=axis1, axis2=axis2, out=out)


@to_aikit_arrays_and_back
def trapz(y, x=None, dx=1.0, axis=-1, out=None):
    return aikit.trapz(y, x=x, dx=dx, axis=axis, out=out)


@to_aikit_arrays_and_back
def trunc(x):
    return aikit.trunc(x)


@to_aikit_arrays_and_back
def vdot(a, b):
    a, b = promote_types_of_jax_inputs(a, b)
    return aikit.multiply(a, b).sum()


abs = absolute
true_divide = divide
