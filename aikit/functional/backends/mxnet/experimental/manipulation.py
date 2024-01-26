from typing import Union, Optional, Sequence, Tuple, List
from numbers import Number
import mxnet as mx

from aikit.utils.exceptions import AikitNotImplementedException


def moveaxis(
    a: Union[(None, mx.ndarray.NDArray)],
    source: Union[(int, Sequence[int])],
    destination: Union[(int, Sequence[int])],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def heaviside(
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def flipud(
    m: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def vstack(
    arrays: Union[(Sequence[None], Sequence[mx.ndarray.NDArray])],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def hstack(
    arrays: Union[(Sequence[None], Sequence[mx.ndarray.NDArray])],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def rot90(
    m: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: Optional[bool] = None,
    k: int = 1,
    axes: Tuple[(int, int)] = (0, 1),
    out: Union[(None, mx.ndarray.NDArray)] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def top_k(
    x: None,
    k: int,
    /,
    *,
    axis: int = -1,
    largest: bool = True,
    sorted: bool = True,
    out: Optional[Tuple[(None, None)]] = None,
) -> Tuple[(None, None)]:
    raise AikitNotImplementedException()


def fliplr(
    m: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def i0(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def vsplit(
    ary: Union[(None, mx.ndarray.NDArray)],
    indices_or_sections: Union[(int, Tuple[(int, ...)])],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise AikitNotImplementedException()


def dsplit(
    ary: Union[(None, mx.ndarray.NDArray)],
    indices_or_sections: Union[(int, Tuple[(int, ...)])],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise AikitNotImplementedException()


def atleast_1d(
    *arys: Union[(None, mx.ndarray.NDArray, bool, Number)], copy: Optional[bool] = None
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise AikitNotImplementedException()


def dstack(
    arrays: Union[(Sequence[None], Sequence[mx.ndarray.NDArray])],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def atleast_2d(
    *arys: Union[(None, mx.ndarray.NDArray)], copy: Optional[bool] = None
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise AikitNotImplementedException()


def atleast_3d(
    *arys: Union[(None, mx.ndarray.NDArray, bool, Number)], copy: Optional[bool] = None
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise AikitNotImplementedException()


def take(
    x: Union[int, List, Union[(None, mx.ndarray.NDArray)]],
    indices: Union[int, List, Union[(None, mx.ndarray.NDArray)]],
    /,
    *,
    axis: Optional[int] = None,
    mode: str = "clip",
    fill_value: Optional[Number] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def take_along_axis(
    arr: Union[(None, mx.ndarray.NDArray)],
    indices: Union[(None, mx.ndarray.NDArray)],
    axis: int,
    /,
    *,
    mode: str = "fill",
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def hsplit(
    ary: Union[(None, mx.ndarray.NDArray)],
    indices_or_sections: Union[(int, Tuple[(int, ...)])],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[Union[(None, mx.ndarray.NDArray)]]:
    raise AikitNotImplementedException()


def broadcast_shapes(*shapes: Union[(List[int], List[Tuple])]) -> Tuple[(int, ...)]:
    raise AikitNotImplementedException()


def expand(
    x: Union[(None, mx.ndarray.NDArray)],
    shape: Union[(List[int], List[Tuple])],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def concat_from_sequence(
    input_sequence: Union[(Tuple[None], List[None])],
    /,
    *,
    new_axis: int = 0,
    axis: int = 0,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()
