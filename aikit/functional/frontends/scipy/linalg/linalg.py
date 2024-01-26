# global
import aikit
from aikit.functional.frontends.numpy.func_wrapper import to_aikit_arrays_and_back


# --- Helpers --- #
# --------------- #


def _check_finite(a):
    if not aikit.all(aikit.isfinite(a)):
        raise ValueError("Array must not contain infs or NaNs")


# --- Main --- #
# ------------ #


# eigh_tridiagonal
@to_aikit_arrays_and_back
def eigh_tridiagonal(
    d,
    e,
    /,
    *,
    eigvals_only=False,
    select="a",
    select_range=None,
    check_finite=True,
    tol=0.0,
):
    if check_finite:
        _check_finite(d)
        _check_finite(e)

    return aikit.eigh_tridiagonal(
        d,
        e,
        eigvals_only=eigvals_only,
        select=select,
        select_range=select_range,
        tol=tol,
    )


# inv
@to_aikit_arrays_and_back
def inv(a, /, *, overwrite_a=False, check_finite=True):
    if check_finite:
        _check_finite(a)

    if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Expected a square matrix")

    return aikit.inv(a)


# kron
@to_aikit_arrays_and_back
def kron(a, b):
    return aikit.kron(a, b)


# lu_factor
@to_aikit_arrays_and_back
def lu_factor(a, /, *, overwrite_a=False, check_finite=True):
    if check_finite:
        _check_finite(a)
    return aikit.lu_factor(a)


# norm
@to_aikit_arrays_and_back
def norm(a, /, *, ord=None, axis=None, keepdims=False, check_finite=True):
    if check_finite:
        _check_finite(a)

    if axis is None and ord is not None:
        if a.ndim not in (1, 2):
            raise ValueError("Improper number of dimensions to norm.")
        else:
            if a.ndim == 1:
                ret = aikit.vector_norm(a, axis=axis, keepdims=keepdims, ord=ord)
            else:
                ret = aikit.matrix_norm(a, keepdims=keepdims, ord=ord)
    elif axis is None and ord is None:
        a = aikit.flatten(a)
        ret = aikit.vector_norm(a, axis=0, keepdims=keepdims, ord=2)
    if isinstance(axis, int):
        ret = aikit.vector_norm(a, axis=axis, keepdims=keepdims, ord=ord)
    elif isinstance(axis, tuple):
        ret = aikit.matrix_norm(a, axis=axis, keepdims=keepdims, ord=ord)
    return ret


# pinv
@to_aikit_arrays_and_back
def pinv(
    a,
    /,
    *,
    atol=None,
    rtol=None,
    return_rank=False,
    cond=None,
    rcond=None,
    check_finite=True,
):
    if check_finite:
        _check_finite(a)

    if (rcond or cond) and (atol is None) and (rtol is None):
        atol = rcond or cond
        rtol = 0.0

    inverse = aikit.pinv(a, rtol=rtol)

    if return_rank:
        rank = aikit.matrix_rank(a)
        return inverse, rank

    return inverse


# svd
@to_aikit_arrays_and_back
def svd(
    a, /, *, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True
):
    if check_finite:
        _check_finite(a)
    return aikit.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)


# svdvals
@to_aikit_arrays_and_back
def svdvals(a, /, *, overwrite_a=False, check_finite=True):
    if check_finite:
        _check_finite(a)
    return aikit.svdvals(a)


# Functions #
# --------- #


# tril
@to_aikit_arrays_and_back
def tril(m, /, *, k=0):
    return aikit.tril(m, k=k)


# triu
@to_aikit_arrays_and_back
def triu(m, /, *, k=0):
    return aikit.triu(m, k=k)
