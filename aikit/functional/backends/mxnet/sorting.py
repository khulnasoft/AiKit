from typing import Union, Optional, Literal, List
import mxnet as mx

import aikit
from aikit.utils.exceptions import AikitNotImplementedException


def argsort(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: int = (-1),
    descending: bool = False,
    stable: bool = True,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def sort(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: int = (-1),
    descending: bool = False,
    stable: bool = True,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def msort(
    a: Union[(None, mx.ndarray.NDArray, list, tuple)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def searchsorted(
    x: Union[(None, mx.ndarray.NDArray)],
    v: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    side: Literal[("left", "right")] = "left",
    sorter: Optional[Union[(aikit.Array, aikit.NativeArray, List[int])]] = None,
    ret_dtype: None = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()
