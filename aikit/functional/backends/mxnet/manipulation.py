import mxnet as mx
from numbers import Number
from typing import Union, Tuple, Optional, List, Sequence

import aikit
from aikit.utils.exceptions import AikitNotImplementedException


def concat(
    xs: Union[(Tuple[(None, ...)], List[None])],
    /,
    *,
    axis: int = 0,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def expand_dims(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: Optional[bool] = None,
    axis: Union[(int, Sequence[int])] = 0,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def flip(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: Optional[bool] = None,
    axis: Optional[Union[(int, Sequence[int])]] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def permute_dims(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    axes: Tuple[(int, ...)],
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def reshape(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    shape: Union[(aikit.NativeShape, Sequence[int])],
    *,
    copy: Optional[bool] = None,
    order: str = "C",
    allowzero: bool = True,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def roll(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    shift: Union[(int, Sequence[int])],
    *,
    axis: Optional[Union[(int, Sequence[int])]] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def squeeze(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    return mx.nd.squeeze(x, axis=axis)


def stack(
    arrays: Union[(Tuple[None], List[None])],
    /,
    *,
    axis: int = 0,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def split(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: Optional[bool] = None,
    num_or_size_splits: Optional[Union[(int, Sequence[int])]] = None,
    axis: int = 0,
    with_remainder: bool = False,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def repeat(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    repeats: Union[(int, List[int])],
    *,
    axis: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def tile(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    repeats: Sequence[int],
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    return mx.nd.tile(x, repeats)


def constant_pad(
    x, /, pad_width, *, value=0, out: Optional[Union[(None, mx.ndarray.NDArray)]] = None
):
    raise AikitNotImplementedException()


def zero_pad(
    x, /, pad_width, *, out: Optional[Union[(None, mx.ndarray.NDArray)]] = None
):
    raise AikitNotImplementedException()


def swapaxes(
    x,
    axis0,
    axis1,
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
):
    raise AikitNotImplementedException()


def clip(
    x: Union[(None, mx.ndarray.NDArray)],
    x_min: Union[(Number, None, mx.ndarray.NDArray)],
    x_max: Union[(Number, None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    return mx.nd.clip(x, x_min, x_max)


def unstack(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: Optional[bool] = None,
    axis: int = 0,
    keepdims: bool = False,
) -> List[None]:
    raise AikitNotImplementedException()
