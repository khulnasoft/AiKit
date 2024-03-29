from typing import Union, Optional
import mxnet as mx

from aikit.utils.exceptions import AikitNotImplementedException


def lexsort(
    keys: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: int = (-1),
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()
