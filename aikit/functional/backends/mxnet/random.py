"""MXNet random functions.

Collection of MXNet random functions, wrapped to fit Aikit syntax and
signature.
"""

import mxnet as mx
from typing import Optional, Union, Sequence
import aikit

from aikit.utils.exceptions import AikitNotImplementedException


def random_uniform(
    *,
    low: Union[(float, None, mx.ndarray.NDArray)] = 0.0,
    high: Union[(float, None, mx.ndarray.NDArray)] = 1.0,
    shape: Optional[Union[(aikit.NativeShape, Sequence[int], None)]] = None,
    dtype: None,
    device: str,
    seed: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def random_normal(
    *,
    mean: Union[(float, None, mx.ndarray.NDArray)] = 0.0,
    std: Union[(float, None, mx.ndarray.NDArray)] = 1.0,
    shape: Optional[Union[(aikit.NativeShape, Sequence[int])]] = None,
    dtype: None,
    seed: Optional[int] = None,
    device: str,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def multinomial(
    population_size: int,
    num_samples: int,
    /,
    *,
    batch_size: int = 1,
    probs: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
    replace: bool = True,
    device: str,
    seed: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def randint(
    low: Union[(float, None, mx.ndarray.NDArray)],
    high: Union[(float, None, mx.ndarray.NDArray)],
    /,
    *,
    shape: Optional[Union[(aikit.NativeShape, Sequence[int])]] = None,
    device: str,
    dtype: Optional[Union[(None, aikit.Dtype)]] = None,
    seed: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()


def seed(*, seed_value: int = 0) -> None:
    raise AikitNotImplementedException()


def shuffle(
    x: Union[(None, mx.ndarray.NDArray)],
    axis: Optional[int] = 0,
    /,
    *,
    seed: Optional[int] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise AikitNotImplementedException()
