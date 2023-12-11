import paddle
import paddle.nn.functional as F
import aikit
from aikit.utils.exceptions import AikitNotImplementedException
from typing import Optional, Tuple
from aikit.func_wrapper import with_supported_dtypes
from aikit.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version


# TODO: add support for the rest of the dtypes
# use numpy implementation with aikit functions
@with_unsupported_device_and_dtypes(
    {
        "2.5.2 and below": {
            "cpu": (
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "float16",
                "complex",
                "bool",
            )
        }
    },
    backend_version,
)
def batch_norm(
    x: paddle.Tensor,
    mean: paddle.Tensor,
    variance: paddle.Tensor,
    /,
    *,
    scale: Optional[paddle.Tensor] = None,
    offset: Optional[paddle.Tensor] = None,
    training: Optional[bool] = False,
    eps: Optional[float] = 1e-5,
    momentum: Optional[float] = 1e-1,
    data_format: Optional[str] = "NSC",
    out: Optional[Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    if x.dtype not in [paddle.float32, paddle.float64]:
        x, mean, variance, scale, offset = (
            t.cast("float32") for t in [x, mean, variance, scale, offset]
        )
    runningmean = mean
    runningvariance = variance
    data_formats = ["NC", "NCL", "NCHW", "NCDHW", "NLC", "NHWC", "NDHWC"]

    try:
        data_format = (
            data_formats[4:][x.ndim - 3]
            if data_format[-1] == "C"
            else data_formats[0:4][x.ndim - 2]
        )
    except IndexError:
        raise IndexError(
            "data_format must be one of 'NC', 'NCL', 'NCHW', 'NCDHW', 'NLC', 'NHWC',"
            f" 'NDHWC' but receive {data_format}"
        )

    with aikit.ArrayMode(False):
        if training:
            x_shape = paddle.to_tensor(x.shape)
            x_size = paddle.prod(x_shape)
            n = (x_size if x.ndim == 1 else aikit.divide(x_size, x_shape[-1])).cast(
                x.dtype
            )
            dims = (0, *range(1, x.ndim - 1))
            mean = aikit.mean(x, axis=dims)
            variance = aikit.var(x, axis=dims)
            # runningmean = (1 - momentum) * runningmean + momentum * mean
            runningmean = aikit.add(
                aikit.multiply(aikit.subtract(1, momentum), runningmean),
                aikit.multiply(momentum, mean),
            )
            # runningvariance = (
            #    1 - momentum
            # ) * runningvariance + momentum * variance * n / (n - 1)
            runningvariance = aikit.add(
                aikit.multiply(aikit.subtract(1, momentum), runningvariance),
                aikit.divide(aikit.multiply(aikit.multiply(momentum, variance), n), n - 1),
            )

    xnormalized = F.batch_norm(
        x,
        running_mean=mean,
        running_var=variance,
        weight=scale,
        bias=offset,
        training=training,
        momentum=momentum,
        epsilon=eps,
        data_format=data_format,
    ).cast(x.dtype)
    return xnormalized, runningmean, runningvariance


batch_norm.partial_mixed_handler = lambda x, *args, scale, offset, **kwargs: (
    (x.ndim > 1 and x.ndim < 6)
    and (scale is not None and scale.ndim == 1)
    and (offset is not None and offset.ndim == 1)
)


@with_supported_dtypes({"2.5.2 and below": ("float32", "float64")}, backend_version)
def l1_normalize(
    x: paddle.Tensor, /, *, axis: Optional[int] = None, out: paddle.Tensor = None
) -> paddle.Tensor:
    if not isinstance(x, paddle.Tensor):
        x = paddle.to_tensor(x)
    if axis is None:
        axis = list(range(x.ndim))
    elif isinstance(axis, int):
        axis = [axis]
    else:
        axis = list(axis)

    # Compute the L1 norm along the given axis
    norm = paddle.norm(x, p=1, axis=axis, keepdim=True)

    # Divide x by the L1 norm to obtain the normalized array
    norm = paddle.where(norm == 0, paddle.to_tensor([1], dtype=x.dtype), norm)
    if out is None:
        return x / norm
    else:
        out[:] = x / norm
        return out


def l2_normalize(
    x: paddle.Tensor, /, *, axis: Optional[int] = None, out: paddle.Tensor = None
) -> paddle.Tensor:
    raise AikitNotImplementedException()


def instance_norm(
    x: paddle.Tensor,
    mean: paddle.Tensor,
    variance: paddle.Tensor,
    /,
    *,
    scale: Optional[paddle.Tensor] = None,
    offset: Optional[paddle.Tensor] = None,
    training: Optional[bool] = False,
    eps: Optional[float] = 1e-5,
    momentum: Optional[float] = 1e-1,
    data_format: Optional[str] = "NSC",
    out: Optional[
        Tuple[
            paddle.Tensor,
            paddle.Tensor,
            paddle.Tensor,
        ]
    ] = None,
) -> Tuple[
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
]:
    raise AikitNotImplementedException()


def lp_normalize(
    x: paddle.Tensor,
    /,
    *,
    p: float = 2,
    axis: Optional[int] = None,
    out: paddle.Tensor = None,
) -> paddle.Tensor:
    raise AikitNotImplementedException()
