# global

import mxnet as mx
from typing import Union, Optional, Tuple, Literal, List, Sequence
from collections import namedtuple


# local
from aikit import inf
from aikit.utils.exceptions import AikitNotImplementedException


def cholesky(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    upper: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def cross(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axisa: int = (-1),
    axisb: int = (-1),
    axisc: int = (-1),
    axis: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def det(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    return mx.nd.linalg.det(x)


def diagonal(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    offset: int = 0,
    axis1: int = (-2),
    axis2: int = (-1),
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def eig(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Tuple[Union[(None, mx.ndarray.NDArray)]]:
    raise AikitNotImplementedException()


def eigh(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    UPLO: str = "L",
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Tuple[Union[(None, mx.ndarray.NDArray)]]:
    raise AikitNotImplementedException()


def eigvalsh(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    UPLO: str = "L",
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def inner(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def inv(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    adjoint: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def matmul(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    transpose_a: bool = False,
    transpose_b: bool = False,
    adjoint_a: bool = False,
    adjoint_b: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def matrix_norm(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    ord: Union[(int, float, Literal[(inf, (-inf), "fro", "nuc")])] = "fro",
    axis: Tuple[(int, int)] = ((-2), (-1)),
    keepdims: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def matrix_power(
    x: Union[(None, mx.ndarray.NDArray)],
    n: int,
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def matrix_rank(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    atol: Optional[Union[(float, Tuple[float])]] = None,
    rtol: Optional[Union[(float, Tuple[float])]] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def matrix_transpose(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    conjugate: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def outer(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def pinv(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    rtol: Optional[Union[(float, Tuple[float])]] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def qr(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    mode: str = "reduced",
    out: Optional[
        Tuple[(Union[(None, mx.ndarray.NDArray)], Union[(None, mx.ndarray.NDArray)])]
    ] = None,
) -> Tuple[(Union[(None, mx.ndarray.NDArray)], Union[(None, mx.ndarray.NDArray)])]:
    res = namedtuple("qr", ["Q", "R"])
    q, r = mx.np.linalg.qr(x, mode=mode)
    return res(q, r)


def slogdet(
    x: Union[(None, mx.ndarray.NDArray)], /
) -> Tuple[(Union[(None, mx.ndarray.NDArray)], Union[(None, mx.ndarray.NDArray)])]:
    raise AikitNotImplementedException()


def solve(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    adjoint: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def svd(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
) -> Union[
    (Union[(None, mx.ndarray.NDArray)], Tuple[(Union[(None, mx.ndarray.NDArray)], ...)])
]:
    raise AikitNotImplementedException()


def svdvals(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    driver: Optional[str] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    # TODO: handling the driver argument
    raise AikitNotImplementedException()


def tensordot(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axes: Union[(int, Tuple[(List[int], List[int])])] = 2,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def trace(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def vecdot(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: int = (-1),
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def vector_norm(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: Optional[Union[(int, Sequence[int])]] = None,
    keepdims: bool = False,
    ord: Union[(int, float, Literal[(inf, (-inf))])] = 2,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def diag(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    k: int = 0,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def vander(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    N: Optional[int] = None,
    increasing: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def vector_to_skew_symmetric_matrix(
    vector: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()
