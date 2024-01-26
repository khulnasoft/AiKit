# local
import aikit
import aikit.functional.frontends.paddle as paddle_frontend
from aikit.func_wrapper import (
    with_supported_dtypes,
    with_unsupported_dtypes,
    with_supported_device_and_dtypes,
)
from aikit.functional.frontends.paddle.func_wrapper import _to_aikit_array


class Tensor:
    def __init__(self, array, dtype=None, place="cpu", stop_gradient=True):
        self._aikit_array = (
            aikit.array(array, dtype=dtype, device=place)
            if not isinstance(array, aikit.Array)
            else array
        )
        self._dtype = dtype
        self._place = place
        self._stop_gradient = stop_gradient

    def __repr__(self):
        return (
            f"aikit.frontends.paddle.Tensor(shape={self.shape}, dtype={self.dtype}, "
            + str(self.aikit_array.__repr__()).replace("aikit.array(", "")
        )

    # Properties #
    # ---------- #

    @property
    def aikit_array(self):
        return self._aikit_array

    @property
    def place(self):
        return self.aikit_array.device

    @property
    def dtype(self):
        return self._aikit_array.dtype

    @property
    def shape(self):
        return list(self.aikit_array.shape.shape)

    @property
    def ndim(self):
        return self.dim()

    # Setters #
    # --------#

    @aikit_array.setter
    def aikit_array(self, array):
        self._aikit_array = (
            aikit.array(array) if not isinstance(array, aikit.Array) else array
        )

    # Special Methods #
    # -------------------#

    @with_unsupported_dtypes(
        {"2.6.0 and below": ("bool", "unsigned", "int8", "float16", "bfloat16")},
        "paddle",
    )
    def __add__(self, y, /, name=None):
        return paddle_frontend.add(self, y)

    @with_unsupported_dtypes(
        {"2.6.0 and below": ("bool", "unsigned", "int8", "float16", "bfloat16")},
        "paddle",
    )
    def __radd__(self, x, /, name=None):
        return paddle_frontend.add(self, x)

    @with_unsupported_dtypes(
        {"2.6.0 and below": ("bool", "unsigned", "int8", "float16", "bfloat16")},
        "paddle",
    )
    def __sub__(self, y, /, name=None):
        return paddle_frontend.subtract(self, y)

    @with_unsupported_dtypes(
        {"2.6.0 and below": ("uint8", "int8", "int16", "float16", "bfloat16")},
        "paddle",
    )
    def __mul__(self, y, /, name=None):
        return paddle_frontend.multiply(self, y)

    @with_unsupported_dtypes(
        {
            "2.6.0 and below": (
                "bool",
                "uint8",
                "int8",
                "int16",
                "complex64",
                "complex128",
            )
        },
        "paddle",
    )
    def __gt__(self, y, /, name=None):
        return paddle_frontend.logic.greater_than(self, y)

    @with_unsupported_dtypes(
        {
            "2.6.0 and below": (
                "bool",
                "uint8",
                "int8",
                "int16",
                "complex64",
                "complex128",
            )
        },
        "paddle",
    )
    def __lt__(self, y, /, name=None):
        return paddle_frontend.logic.less_than(self, y)

    @with_unsupported_dtypes(
        {
            "2.6.0 and below": (
                "bool",
                "uint8",
                "int8",
                "int16",
                "complex64",
                "complex128",
            )
        },
        "paddle",
    )
    def __ge__(self, y, /, name=None):
        return paddle_frontend.logic.greater_equal(self, y)

    @with_unsupported_dtypes(
        {
            "2.6.0 and below": (
                "bool",
                "uint8",
                "int8",
                "int16",
                "complex64",
                "complex128",
            )
        },
        "paddle",
    )
    def __le__(self, y, /, name=None):
        return paddle_frontend.logic.less_equal(self, y)

    @with_supported_dtypes(
        {
            "2.6.0 and below": (
                "bool",
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
            )
        },
        "paddle",
    )
    def __or__(self, y, /, name=None):
        return paddle_frontend.logic.bitwise_or(self, y)

    @with_unsupported_dtypes(
        {"2.6.0 and below": ("bool", "unsigned", "int8", "float16", "bfloat16")},
        "paddle",
    )
    def __rsub__(self, x, /, name=None):
        return paddle_frontend.subtract(x, self)

    def __getitem__(self, item):
        aikit_args = aikit.nested_map(_to_aikit_array, [self, item])
        ret = aikit.get_item(*aikit_args)
        return paddle_frontend.Tensor(ret)

    def __setitem__(self, item, value):
        raise aikit.utils.exceptions.IvyException(
            "aikit.functional.frontends.paddle.Tensor object doesn't support assignment"
        )

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def __floordiv__(self, y, /, name=None):
        return paddle_frontend.floor_divide(self, y)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def __ne__(self, y, /, name=None):
        return paddle_frontend.not_equal(self, y)

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d tensor not supported")
        for i in range(self.shape[0]):
            yield self[i]

    @with_unsupported_dtypes(
        {"2.6.0 and below": ("bool", "unsigned", "int8", "float16", "bfloat16")},
        "paddle",
    )
    def __rmul__(self, y, /, name=None):
        return paddle_frontend.multiply(self, y)

    @with_unsupported_dtypes(
        {"2.6.0 and below": ("bool", "unsigned", "int8", "float16", "bfloat16")},
        "paddle",
    )
    def __float__(self):
        return float(self._aikit_array)

    def __xor__(self, y, /, name=None):
        return paddle_frontend.logic.bitwise_xor(self, y)

    def __invert__(self, out=None, name=None):
        return paddle_frontend.logic.bitwise_not(self)

    def __len__(self):
        return len(self._aikit_array)

    def __neg__(self):
        return paddle_frontend.neg(self)

    @with_unsupported_dtypes(
        {"2.6.0 and below": ("bool", "unsigned", "int8", "float16", "bfloat16")},
        "paddle",
    )
    def __rdiv__(self, y, /, name=None):
        return paddle_frontend.divide(y, self)

    @with_unsupported_dtypes(
        {"2.6.0 and below": ("bool", "unsigned", "int8", "float16", "bfloat16")},
        "paddle",
    )
    def __rtruediv__(self, y, /, name=None):
        return paddle_frontend.divide(y, self)

    @with_unsupported_dtypes(
        {"2.6.0 and below": ("bool", "unsigned", "int8", "float16", "bfloat16")},
        "paddle",
    )
    def __int__(self):
        return int(self._aikit_array)

    @with_unsupported_dtypes(
        {
            "2.6.0 and below": (
                "bool",
                "unsigned",
                "int8",
                "int32",
                "int64",
                "float16",
                "bfloat16",
            )
        },
        "paddle",
    )
    def __long__(self):
        return int(self._aikit_array)

    # Instance Methods #
    # ---------------- #

    def reshape(self, *args, shape=None):
        if args and shape:
            raise TypeError("reshape() got multiple values for argument 'shape'")
        if shape is not None:
            return paddle_frontend.reshape(self, shape)
        if args:
            if isinstance(args[0], (tuple, list)):
                shape = args[0]
                return paddle_frontend.reshape(self, shape)
            else:
                return paddle_frontend.reshape(self, args)
        else:
            raise ValueError("reshape() got no values for argument 'shape'")

    def reshape_(self, *args, shape=None):
        if args and shape:
            raise TypeError("reshape() got multiple values for argument 'shape'")
        if shape is not None:
            self.aikit_array = paddle_frontend.reshape(
                self._aikit_array, shape=shape
            ).aikit_array
            return self
        if args:
            if isinstance(args[0], (tuple, list)):
                shape = args[0]
                self.aikit_array = paddle_frontend.reshape(
                    self._aikit_array, shape=shape
                ).aikit_array
                return self
            else:
                self.aikit_array = paddle_frontend.reshape(
                    self._aikit_array, args
                ).aikit_array
                return self
        else:
            raise ValueError("reshape_() got no values for argument 'shape'")

    def dim(self):
        return self.aikit_array.ndim

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def abs(self):
        return paddle_frontend.abs(self)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def acosh(self, name=None):
        return paddle_frontend.acosh(self)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def add_n(self, inputs, name=None):
        inputs = aikit.array(inputs)
        return aikit.sum(inputs, dtype=inputs.dtype, axis=0)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def ceil(self):
        return paddle_frontend.ceil(self)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def ceil_(self):
        self.aikit_array = self.ceil().aikit_array
        return self

    @with_unsupported_dtypes({"2.6.0 and below": ("complex", "int8")}, "paddle")
    def numel(self):
        return paddle_frontend.numel(self)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16",)}, "paddle")
    def asinh(self, name=None):
        return paddle_frontend.asinh(self)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def asin(self, name=None):
        return paddle_frontend.asin(self)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def cosh(self, name=None):
        return paddle_frontend.cosh(self)

    @with_supported_dtypes(
        {
            "2.6.0 and below": (
                "int32",
                "int64",
                "float64",
                "complex128",
                "float32",
                "complex64",
                "bool",
            )
        },
        "paddle",
    )
    def diagonal(self, offset, axis1=0, axis2=1, name=None):
        return paddle_frontend.diagonal(self, offset=offset, axis1=axis1, axis2=axis2)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def log(self, name=None):
        return paddle_frontend.log(self)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def sin(self, name=None):
        return paddle_frontend.sin(self)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def sinh(self, name=None):
        return paddle_frontend.sinh(self)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def lerp(self, y, weight, name=None):
        return paddle_frontend.lerp(self, y, weight)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def lerp_(self, y, weight, name=None):
        self.aikit_array = paddle_frontend.lerp(self, y, weight).aikit_array
        return self

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def argmax(self, axis=None, keepdim=False, dtype=None, name=None):
        return paddle_frontend.argmax(self, axis=axis, keepdim=keepdim, dtype=dtype)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "uint16")}, "paddle")
    def unsqueeze(self, axis=None, name=None):
        return paddle_frontend.Tensor(aikit.expand_dims(self._aikit_array, axis=axis))

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def sqrt(self, name=None):
        return paddle_frontend.sqrt(self)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def sqrt_(self, name=None):
        self.aikit_array = self.sqrt().aikit_array
        return self

    @with_unsupported_dtypes({"2.6.0 and below": ("bfloat16", "uint16")}, "paddle")
    def zero_(self):
        self.aikit_array = paddle_frontend.zeros_like(self).aikit_array
        return self

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def cos(self, name=None):
        return paddle_frontend.cos(self)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def exp(self, name=None):
        return paddle_frontend.exp(self)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def exp_(self, name=None):
        self.aikit_array = self.exp().aikit_array
        return self

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def erf(self, name=None):
        return paddle_frontend.erf(self)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def subtract(self, y, name=None):
        return paddle_frontend.subtract(self, y)

    @with_unsupported_dtypes(
        {"2.6.0 and below": ("float16", "uint8", "int8", "bool")}, "paddle"
    )
    def subtract_(self, y, name=None):
        self.aikit_array = self.subtract(y).aikit_array
        return self

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def log10(self, name=None):
        return paddle_frontend.Tensor(aikit.log10(self._aikit_array))

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def argsort(self, axis=-1, descending=False, name=None):
        return paddle_frontend.argsort(self, axis=axis, descending=descending)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def floor(self, name=None):
        return paddle_frontend.floor(self)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def floor_(self):
        self.aikit_array = self.floor().aikit_array
        return self

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def round_(self, name=None):
        self.aikit_array = paddle_frontend.round(self).aikit_array
        return self

    @with_supported_dtypes(
        {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
    )
    def clip(self, min=None, max=None, name=None):
        aikit.utils.assertions.check_all_or_any_fn(
            min,
            max,
            fn=aikit.exists,
            type="any",
            limit=[1, 2],
            message="at most one of min or max can be None",
        )
        if min is None:
            ret = aikit.minimum(self._aikit_array, max)
        elif max is None:
            ret = aikit.maximum(self._aikit_array, min)
        else:
            ret = aikit.clip(self._aikit_array, min, max)
        return paddle_frontend.Tensor(ret)

    @with_supported_dtypes(
        {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
    )
    def clip_(self, min=None, max=None, name=None):
        self._aikit_array = self.clip(min, max).aikit_array
        return self

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def tanh(self, name=None):
        return paddle_frontend.tanh(self)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def add(self, y, name=None):
        return paddle_frontend.Tensor(aikit.add(self._aikit_array, _to_aikit_array(y)))

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def add_(self, y, name=None):
        self.aikit_array = paddle_frontend.add(self, y).aikit_array
        return self

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def addmm(self, x, y, beta=1.0, alpha=1.0, name=None):
        return paddle_frontend.addmm(self, x, y, beta, alpha)

    @with_supported_dtypes(
        {"2.6.0 and below": ("float16", "float32", "float64", "int32", "int64")},
        "paddle",
    )
    def isinf(self, name=None):
        return paddle_frontend.isinf(self)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "uint16")}, "paddle")
    def unsqueeze_(self, axis=None, name=None):
        self.aikit_array = self.unsqueeze(axis=axis).aikit_array
        return self

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def square(self, name=None):
        return paddle_frontend.square(self)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def remainder_(self, y, name=None):
        self.aikit_array = paddle_frontend.remainder(self, y).aikit_array
        return self

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def cholesky(self, upper=False, name=None):
        return paddle_frontend.cholesky(self, upper=upper)

    @with_unsupported_dtypes(
        {"2.6.0 and below": ("float16", "uint16", "int16")}, "paddle"
    )
    def squeeze(self, axis=None, name=None):
        if isinstance(axis, int) and self.ndim > 0:
            if self.shape[axis] > 1:
                return self
        if len(self.shape) == 0:
            return self
        return paddle_frontend.squeeze(self, axis=axis)

    @with_unsupported_dtypes(
        {"2.6.0 and below": ("float16", "uint16", "int16")}, "paddle"
    )
    def squeeze_(self, axis=None, name=None):
        self.aikit_array = paddle_frontend.squeeze(self, axis=axis).aikit_array
        return self

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def multiply(self, y, name=None):
        return paddle_frontend.multiply(self, y)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def matmul(self, y, transpose_x=False, transpose_y=False, name=None):
        return paddle_frontend.matmul(
            self, y, transpose_x=transpose_x, transpose_y=transpose_y
        )

    @with_supported_dtypes(
        {"2.6.0 and below": ("float16", "float32", "float64", "int32", "int64")},
        "paddle",
    )
    def isfinite(self, name=None):
        return paddle_frontend.isfinite(self)

    @with_supported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
    def all(self, axis=None, keepdim=False, name=None):
        return paddle_frontend.Tensor(
            aikit.all(self.aikit_array, axis=axis, keepdims=keepdim)
        )

    @with_supported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False, name=None):
        return paddle_frontend.allclose(
            self, other, rtol=rtol, atol=atol, equal_nan=equal_nan
        )

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def sort(self, axis=-1, descending=False, name=None):
        return paddle_frontend.sort(self, axis=axis, descending=descending)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def log1p(self, name=None):
        return paddle_frontend.log1p(self)

    @with_supported_dtypes(
        {
            "2.4.2 and below": (
                "bool",
                "uint8",
                "int8",
                "int16",
                "int32",
                "int64",
            )
        },
        "paddle",
    )
    def bitwise_and(self, y, out=None, name=None):
        return paddle_frontend.bitwise_and(self, y)

    @with_supported_dtypes(
        {
            "2.6.0 and below": (
                "bool",
                "int8",
                "int16",
                "int32",
                "int64",
                "float32",
                "float64",
            )
        },
        "paddle",
    )
    def logical_or(self, y, out=None, name=None):
        return paddle_frontend.logical_or(self, y, out=out)

    @with_supported_dtypes(
        {"2.6.0 and below": ("bool", "uint8", "int8", "int16", "int32", "int64")},
        "paddle",
    )
    def bitwise_xor(self, y, out=None, name=None):
        return paddle_frontend.bitwise_xor(self, y)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def any(self, axis=None, keepdim=False, name=None):
        return paddle_frontend.any(self, axis=axis, keepdim=keepdim)

    @with_unsupported_dtypes({"2.6.0 and below": "bfloat16"}, "paddle")
    def astype(self, dtype):
        return paddle_frontend.Tensor(aikit.astype(self._aikit_array, dtype))

    @with_supported_dtypes(
        {"2.6.0 and below": ("bool", "uint8", "int8", "int16", "int32", "int64")},
        "paddle",
    )
    def bitwise_not(self, out=None, name=None):
        return paddle_frontend.bitwise_not(self, out=out)

    @with_supported_dtypes(
        {
            "2.6.0 and below": (
                "bool",
                "int8",
                "int16",
                "int32",
                "int64",
            )
        },
        "paddle",
    )
    def bitwise_or(self, y, out=None, name=None):
        return paddle_frontend.bitwise_or(self, y, out=out)

    @with_supported_dtypes(
        {
            "2.6.0 and below": (
                "bool",
                "int8",
                "int16",
                "int32",
                "int64",
                "float32",
                "float64",
            )
        },
        "paddle",
    )
    def logical_xor(self, y, out=None, name=None):
        return paddle_frontend.logical_xor(self, y, out=out)

    @with_supported_dtypes(
        {"2.6.0 and below": ("float16", "float32", "float64", "int32", "int64")},
        "paddle",
    )
    def isnan(self, name=None):
        return paddle_frontend.isnan(self)

    @with_unsupported_dtypes(
        {
            "2.6.0 and below": (
                "bool",
                "uint8",
                "int8",
                "int16",
                "complex64",
                "complex128",
            )
        },
        "paddle",
    )
    def greater_than(self, y, name=None):
        return paddle_frontend.greater_than(self, y)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def rsqrt(self, name=None):
        return paddle_frontend.rsqrt(self)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def rsqrt_(self, name=None):
        self.aikit_array = self.rsqrt().aikit_array
        return self

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def reciprocal(self, name=None):
        return paddle_frontend.reciprocal(self)

    @with_supported_dtypes(
        {
            "2.6.0 and below": (
                "bool",
                "int8",
                "int16",
                "int32",
                "int64",
                "float32",
                "float64",
            )
        },
        "paddle",
    )
    def logical_and(self, y, out=None, name=None):
        return paddle_frontend.logical_and(self, y, out=out)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def divide(self, y, name=None):
        return paddle_frontend.divide(self, y)

    @with_supported_dtypes(
        {"2.6.0 and below": ("float32", "float64", "complex64", "complex128")},
        "paddle",
    )
    def eigvals(self, name=None):
        return paddle_frontend.eigvals(self)

    @with_unsupported_dtypes(
        {
            "2.6.0 and below": (
                "bool",
                "uint8",
                "int8",
                "int16",
                "complex64",
                "complex128",
            )
        },
        "paddle",
    )
    def less_than(self, y, name=None):
        return paddle_frontend.less_than(self, y)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def cumprod(self, dim=None, dtype=None, name=None):
        return paddle_frontend.cumprod(self, dim=dim, dtype=dtype)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def cumsum(self, axis=None, dtype=None, name=None):
        return paddle_frontend.Tensor(
            aikit.cumsum(self._aikit_array, axis=axis, dtype=dtype)
        )

    @with_supported_dtypes(
        {"2.6.0 and below": ("complex64", "complex128", "float32", "float64")},
        "paddle",
    )
    def angle(self, name=None):
        return paddle_frontend.angle(self)

    @with_unsupported_dtypes(
        {
            "2.6.0 and below": (
                "uint8",
                "int8",
                "int16",
                "complex64",
                "complex128",
            )
        },
        "paddle",
    )
    def equal(self, y, name=None):
        return paddle_frontend.equal(self, y)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def rad2deg(self, name=None):
        return paddle_frontend.rad2deg(self)

    @with_unsupported_dtypes(
        {
            "2.6.0 and below": (
                "uint8",
                "int8",
                "int16",
                "float16",
                "complex64",
                "complex128",
            )
        },
        "paddle",
    )
    def equal_all(self, y, name=None):
        return paddle_frontend.equal_all(self, y)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def maximum(self, other, name=None):
        return paddle_frontend.maximum(self, other)

    @with_unsupported_dtypes({"2.6.0 and below": "bfloat16"}, "paddle")
    def fmax(self, y, name=None):
        return paddle_frontend.fmax(self, y)

    @with_unsupported_dtypes({"2.6.0 and below": "bfloat16"}, "paddle")
    def fmin(self, y, name=None):
        return paddle_frontend.fmin(self, y)

    @with_supported_dtypes(
        {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
    )
    def minimum(self, y, name=None):
        return paddle_frontend.minimum(self, y)

    @with_supported_dtypes(
        {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
    )
    def max(self, axis=None, keepdim=False, name=None):
        return paddle_frontend.max(self, axis=axis, keepdim=keepdim)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def deg2rad(self, name=None):
        return paddle_frontend.deg2rad(self)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def digamma(self, name=None):
        return paddle_frontend.digamma(self)

    @with_supported_dtypes(
        {"2.6.0 and below": ("float32", "float64", "int32", "int64", "bool")}, "paddle"
    )
    def rot90(self, k=1, axes=(0, 1), name=None):
        return paddle_frontend.rot90(self, k=k, axes=axes)

    @with_supported_dtypes(
        {"2.6.0 and below": ("complex64", "complex128")},
        "paddle",
    )
    def imag(self, name=None):
        return paddle_frontend.imag(self)

    def is_tensor(self):
        return paddle_frontend.is_tensor(self)

    @with_supported_dtypes(
        {
            "2.6.0 and below": (
                "float32",
                "float64",
            )
        },
        "paddle",
    )
    def isclose(self, y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None):
        return paddle_frontend.isclose(
            self, y, rtol=rtol, atol=atol, equal_nan=equal_nan
        )

    @with_supported_dtypes({"2.6.0 and below": ("int32", "int64")}, "paddle")
    def floor_divide(self, y, name=None):
        return paddle_frontend.floor_divide(self, y)

    @with_supported_dtypes({"2.6.0 and below": ("int32", "int64")}, "paddle")
    def mod(self, y, name=None):
        return paddle_frontend.Tensor(aikit.fmod(self._aikit_array, _to_aikit_array(y)))

    @with_supported_dtypes(
        {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
    )
    def floor_mod(self, y, name=None):
        return paddle_frontend.remainder(self, y)

    # cond
    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def cond(self, p=None, name=None):
        return paddle_frontend.cond(self, p=p, name=name)

    @with_unsupported_dtypes({"2.4.2 and below": ("int16", "float16")}, "paddle")
    def conj(self, name=None):
        return paddle_frontend.conj(self)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def log2(self, name=None):
        return paddle_frontend.log2(self)

    @with_unsupported_dtypes(
        {"2.4.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
    )
    def neg(self, name=None):
        return paddle_frontend.neg(self)

    @with_supported_dtypes(
        {
            "2.6.0 and below": (
                "bool",
                "int8",
                "int16",
                "int32",
                "int64",
                "float32",
                "float64",
            )
        },
        "paddle",
    )
    def logical_not(self, out=None, name=None):
        return paddle_frontend.logical_not(self)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def sign(self, name=None):
        return paddle_frontend.sign(self)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def var(self, axis=None, unbiased=True, keepdim=False, name=None):
        return paddle_frontend.var(self, axis=axis, unbiased=unbiased, keepdim=keepdim)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def sgn(self, name=None):
        return paddle_frontend.sgn(self)

    def tolist(self):
        return paddle_frontend.Tensor(aikit.to_list(self._aikit_array))

    @with_supported_dtypes(
        {"2.6.0 and below": ("float32", "float64", "int32", "int64")},
        "paddle",
    )
    def min(self, axis=None, keepdim=False, name=None):
        return paddle_frontend.min(self, axis=axis, keepdim=keepdim)

    @with_supported_dtypes(
        {"2.6.0 and below": ("int32", "int64", "float32", "float64")}, "paddle"
    )
    def pow(self, y, name=None):
        return paddle_frontend.pow(self, y)

    @with_supported_dtypes(
        {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
    )
    def prod(self, axis=None, keepdim=False, dtype=None, name=None):
        return paddle_frontend.Tensor(
            aikit.prod(self._aikit_array, axis=axis, keepdims=keepdim, dtype=dtype)
        )

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def atan(self, name=None):
        return paddle_frontend.atan(self)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def atanh(self, name=None):
        return paddle_frontend.atanh(self)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def std(self, axis=None, unbiased=True, keepdim=False, name=None):
        return paddle_frontend.std(self, axis=axis, unbiased=unbiased, keepdim=keepdim)

    @with_supported_dtypes(
        {"2.6.0 and below": ("int32", "int64", "float32", "float64")}, "paddle"
    )
    def trunc(self, name=None):
        return paddle_frontend.trunc(self)

    @with_supported_dtypes({"2.6.0 and below": ("complex64", "complex128")}, "paddle")
    def as_real(self, name=None):
        if not aikit.is_complex_dtype(self._aikit_array):
            raise aikit.exceptions.IvyError(
                "as_real is only supported for complex tensors"
            )
        re_part = aikit.real(self._aikit_array)
        im_part = aikit.imag(self._aikit_array)
        return paddle_frontend.Tensor(aikit.stack((re_part, im_part), axis=-1))

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def stanh(self, scale_a=0.67, scale_b=1.7159, name=None):
        return paddle_frontend.stanh(self, scale_a=scale_a, scale_b=scale_b)

    @with_supported_dtypes(
        {"2.6.0 and below": ("int32", "int64", "float32", "float64")}, "paddle"
    )
    def trace(self, offset=0, axis1=0, axis2=1, name=None):
        return paddle_frontend.Tensor(
            aikit.trace(self._aikit_array, offset=offset, axis1=axis1, axis2=axis2)
        )

    @with_supported_dtypes(
        {
            "2.6.0 and below": (
                "bfloat16",
                "float32",
                "float64",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
            )
        },
        "paddle",
    )
    def flatten(self, start_axis=0, stop_axis=-1, name=None):
        if len(self.shape) == 0:
            return self.unsqueeze(axis=0)
        return paddle_frontend.Tensor(
            aikit.flatten(self.aikit_array, start_dim=start_axis, end_dim=stop_axis)
        )

    @with_supported_dtypes(
        {
            "2.6.0 and below": (
                "float32",
                "float64",
                "int16",
                "int32",
                "int64",
                "uint8",
            )
        },
        "paddle",
    )
    def argmin(self, axis=None, keepdim=False, dtype=None, name=None):
        return paddle_frontend.argmin(self, axis=axis, keepdim=keepdim, dtype=dtype)

    @with_supported_dtypes(
        {"2.6.0 and below": ("float32", "float64", "int32", "int64")},
        "paddle",
    )
    def topk(self, k, axis=None, largest=True, sorted=True, name=None):
        return paddle_frontend.topk(self, k, axis=axis, largest=largest, sorted=sorted)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def remainder(self, y, name=None):
        return paddle_frontend.remainder(self, y)

    def is_floating_point(self):
        return paddle_frontend.is_floating_point(self)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def tanh_(self, name=None):
        y = self.tanh(self)
        return aikit.inplace_update(self, y)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def reciprocal_(self, name=None):
        y = self.reciprocal(self)
        return aikit.inplace_update(self, y)

    @with_unsupported_dtypes(
        {"2.6.0 and below": ("complex", "uint8", "uint16")}, "paddle"
    )
    def numpy(self):
        return self.aikit_array.to_numpy()

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def nonzero(self):
        return paddle_frontend.nonzero(self)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def inner(self, y, name=None):
        return paddle_frontend.inner(self, y, name)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def acos(self, name=None):
        return paddle_frontend.Tensor(aikit.acos(self._aikit_array))

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def mean(self, axis=None, keepdim=False, name=None):
        return paddle_frontend.mean(self, axis=axis, keepdim=keepdim)

    @with_supported_dtypes({"2.6.0 and below": ("float32", "float64")}, "paddle")
    def as_complex(self, name=None):
        if self.aikit_array.shape[-1] != 2:
            raise aikit.exceptions.IvyError(
                "The size of the last dimension of tensor does not equals 2"
            )
        dtype = (
            aikit.complex64 if aikit.dtype(self.aikit_array) == "float32" else aikit.complex128
        )
        re_part = self.aikit_array[..., 0]
        im_part = aikit.multiply(1j, self.aikit_array[..., 1])
        value = paddle_frontend.Tensor(aikit.add(re_part, im_part).astype(dtype))
        return value

    @with_supported_dtypes(
        {"2.6.0 and below": ("int32", "int64", "float32", "float64", "bool")}, "paddle"
    )
    def not_equal(self, y, name=None):
        return paddle_frontend.not_equal(self._aikit_array, y)

    @with_supported_dtypes(
        {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
    )
    def less_equal(self, y, name=None):
        return paddle_frontend.less_equal(self._aikit_array, y)

    @with_supported_dtypes({"2.6.0 and below": ("complex64", "complex128")}, "paddle")
    def real(self, name=None):
        return paddle_frontend.real(self._aikit_array)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def t(self, name=None):
        axes = list(range(len(self.aikit_array.shape)))[::-1]
        return aikit.permute_dims(self.aikit_array, axes=axes)

    @with_supported_dtypes(
        {
            "2.6.0 and below": (
                "bool",
                "float16",
                "float32",
                "float64",
                "int32",
                "int64",
                "uint8",
            )
        },
        "paddle",
    )
    def cast(self, dtype):
        return paddle_frontend.cast(self, dtype)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def bmm(self, y, transpose_x=False, transpose_y=False, name=None):
        return paddle_frontend.bmm(self, y, transpose_x, transpose_y)

    @with_supported_dtypes(
        {"2.6.0 and below": ("float16", "float32", "float64", "int32", "int64")},
        "paddle",
    )
    def fill_(self, value):
        filled_tensor = paddle_frontend.full_like(self, value)
        return aikit.inplace_update(self, filled_tensor)

    @with_supported_dtypes(
        {
            "2.6.0 and below": (
                "bool",
                "int32",
                "int64",
                "float16",
                "float32",
                "float64",
            )
        },
        "paddle",
    )
    def unbind(self, axis=0):
        return paddle_frontend.unbind(self._aikit_array, axis=axis)

    @with_supported_dtypes(
        {
            "2.6.0 and below": (
                "bool",
                "int32",
                "int64",
                "float16",
                "float32",
                "float64",
            )
        },
        "paddle",
    )
    def unique_consecutive(self, axis=0):
        return paddle_frontend.unique_consecutive(self._aikit_array, axis=axis)

    def cpu(self):
        self.aikit_array = aikit.to_device(self.aikit_array, aikit.as_aikit_dev("cpu"))
        return self

    @with_unsupported_dtypes(
        {"2.6.0 and below": ("int16", "complex64", "complex128")},
        "paddle",
    )
    def split(self, num_or_sections, axis=0, name=None):
        return paddle_frontend.split(self._aikit_array, num_or_sections, axis, name)

    @with_supported_dtypes(
        {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
    )
    def frac(self, name=None):
        return paddle_frontend.frac(self._aikit_array)

    @with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
    def gather(self, y, name=None):
        return paddle_frontend.gather(self, y)

    def is_complex(self):
        return paddle_frontend.is_complex(self)

    @with_unsupported_dtypes(
        {"2.6.0 and below": ("float16", "uint8", "int8", "bool")}, "paddle"
    )
    def gather_(self, y, name=None):
        res = self.gather(self, y)
        return aikit.inplace_update(self, res)

    @with_supported_dtypes(
        {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, "paddle"
    )
    def heaviside(self, y, name=None):
        return paddle_frontend.heaviside(self, y)

    @with_supported_dtypes(
        {"2.6.0 and below": ("bool", "int32", "int64", "float32", "float64")}, "paddle"
    )
    def expand(self, shape, name=None):
        return paddle_frontend.expand(self._aikit_array, shape)

    @with_supported_device_and_dtypes(
        {
            "2.6.0 and below": {
                "cpu": (
                    "bool",
                    "int32",
                    "int64",
                    "float32",
                    "float64",
                    "complex64",
                    "complex128",
                )
            }
        },
        "paddle",
    )
    def tile(self, repeat_times):
        return paddle_frontend.Tensor(aikit.tile(self._aikit_array, repeats=repeat_times))

    @with_supported_dtypes(
        {
            "2.6.0 and below": (
                "bool",
                "float16",
                "float32",
                "float64",
                "int8",
                "int16",
                "int32",
                "int64",
            )
        },
        "paddle",
    )
    def chunk(self, chunks, axis=0, name=None):
        return paddle_frontend.split(self._aikit_array, num_or_sections=chunks, axis=axis)
