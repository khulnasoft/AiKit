# global
from typing import Optional, Union, Sequence
import paddle
from aikit import with_unsupported_device_and_dtypes
from aikit.functional.backends.paddle import backend_version
from aikit.utils.exceptions import AikitNotImplementedException
from aikit.functional.aikit.random import _check_bounds_and_get_shape

# local
import aikit
from paddle.device import core
from aikit import with_supported_device_and_dtypes

# dirichlet


@with_unsupported_device_and_dtypes(
    {
        "2.6.0 and below": {
            "cpu": (
                "int8",
                "int16",
                "uint8",
                "float16",
                "complex64",
                "complex128",
                "bool",
            )
        }
    },
    backend_version,
)
def dirichlet(
    alpha: Union[paddle.Tensor, float, Sequence[float]],
    /,
    *,
    size: Optional[Union[aikit.NativeShape, Sequence[int]]] = None,
    out: Optional[paddle.Tensor] = None,
    seed: Optional[int] = None,
    dtype: Optional[paddle.dtype] = None,
) -> paddle.Tensor:
    size = size if size is not None else len(alpha)
    dtype = dtype if dtype is not None else paddle.float64
    if seed is not None:
        paddle.seed(seed)
    res = paddle.to_tensor(
        paddle.distribution.Dirichlet(concentration=alpha).sample(shape=size),
        dtype=dtype,
    )
    return res


# beta
def beta(
    alpha: Union[float, paddle.Tensor],
    beta: Union[float, paddle.Tensor],
    /,
    *,
    shape: Optional[Union[aikit.NativeShape, Sequence[int]]] = None,
    dtype: Optional[Union[paddle.dtype, aikit.Dtype]] = None,
    device: core.Place = None,
    seed: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if seed is not None:
        paddle.seed(seed)
    shape = _check_bounds_and_get_shape(alpha, beta, shape)
    dtype = paddle.float32 if dtype is None else dtype
    beta = paddle.cast(beta, alpha.dtype)
    dist = paddle.distribution.Beta(alpha, beta)
    sample = dist.sample(shape)
    sample = paddle.cast(sample, dtype)
    return sample


def gamma(
    alpha: Union[float, paddle.Tensor],
    beta: Union[float, paddle.Tensor],
    /,
    *,
    shape: Optional[Union[aikit.NativeShape, Sequence[int]]] = None,
    dtype: Optional[Union[paddle.dtype, aikit.Dtype]] = None,
    device: core.Place = None,
    seed: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise AikitNotImplementedException()


def poisson(
    lam: Union[float, paddle.Tensor],
    *,
    shape: Optional[Union[aikit.NativeShape, Sequence[int]]] = None,
    device: core.Place = None,
    dtype: paddle.dtype = None,
    seed: Optional[int] = None,
    fill_value: Optional[Union[float, int]] = 0,
    out: Optional[paddle.Tensor] = None,
):
    raise AikitNotImplementedException()


# bernoulli
@with_supported_device_and_dtypes(
    {
        "2.5.0 and above": {
            "cpu": ("float32", "float64"),
            "gpu": ("bfloat16", "float16", "float32", "float64"),
        },
        "2.4.2 and below": {
            "cpu": (
                "float32",
                "float64",
            ),
            "gpu": ("float16", "float32", "float64"),
        },
    },
    backend_version,
)
def bernoulli(
    probs: Union[float, paddle.Tensor],
    *,
    logits: Union[float, paddle.Tensor] = None,
    shape: Optional[Union[aikit.NativeArray, Sequence[int]]] = None,
    device: core.Place = None,
    dtype: paddle.dtype,
    seed: Optional[int] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if seed is not None:
        paddle.seed(seed)
    if probs is not None:
        probs = probs
    elif logits is not None:
        probs = aikit.softmax(logits)
    probs = paddle.cast(probs, dtype)
    probs = paddle.unsqueeze(probs, 0) if len(probs.shape) == 0 else probs
    probs = paddle.maximum(probs, paddle.full_like(probs, 1e-6))
    sample = paddle.bernoulli(probs)
    return sample
