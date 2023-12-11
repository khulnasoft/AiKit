# global
import struct
import warnings

# local
import aikit
import aikit.functional.frontends.numpy as np_frontend
from aikit.functional.frontends.numpy.func_wrapper import _to_aikit_array
from aikit.func_wrapper import (
    with_supported_device_and_dtypes,
)


# --- Classes ---#
# ---------------#


class ndarray:
    def __init__(self, shape, dtype="float32", order=None, _init_overload=False):
        if isinstance(dtype, np_frontend.dtype):
            dtype = dtype.aikit_dtype

        # in this case shape is actually the desired array
        if _init_overload:
            self._aikit_array = (
                aikit.array(shape) if not isinstance(shape, aikit.Array) else shape
            )
        else:
            self._aikit_array = aikit.empty(shape=shape, dtype=dtype)

        aikit.utils.assertions.check_elem_in_list(
            order,
            ["C", "F", None],
            message="order must be one of 'C', 'F'",
        )
        if order == "F":
            self._f_contiguous = True
        else:
            self._f_contiguous = False

    def __repr__(self):
        return str(self.aikit_array.__repr__()).replace(
            "aikit.array", "aikit.frontends.numpy.ndarray"
        )

    # Properties #
    # ---------- #

    @property
    def aikit_array(self):
        return self._aikit_array

    @property
    def T(self):
        return np_frontend.transpose(self)

    @property
    def shape(self):
        return tuple(self.aikit_array.shape.shape)

    @property
    def size(self):
        return self.aikit_array.size

    @property
    def dtype(self):
        return np_frontend.dtype(self.aikit_array.dtype)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def flat(self):
        self = self.flatten()
        return self

    # Setters #
    # --------#

    @aikit_array.setter
    def aikit_array(self, array):
        self._aikit_array = (
            aikit.array(array) if not isinstance(array, aikit.Array) else array
        )

    # Instance Methods #
    # ---------------- #

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        aikit.utils.assertions.check_elem_in_list(
            order,
            ["C", "F", "A", "K"],
            message="order must be one of 'C', 'F', or 'A'",
        )
        if copy and self._f_contiguous:
            ret = np_frontend.array(self.aikit_array, order="F")
        else:
            ret = np_frontend.array(self.aikit_array) if copy else self

        dtype = np_frontend.to_aikit_dtype(dtype)
        if np_frontend.can_cast(ret, dtype, casting=casting):
            ret.aikit_array = ret.aikit_array.astype(dtype)
        else:
            raise aikit.utils.exceptions.AikitException(
                f"Cannot cast array data from dtype('{ret.aikit_array.dtype}')"
                f" to dtype('{dtype}') according to the rule '{casting}'"
            )
        if order == "F":
            ret._f_contiguous = True
        elif order == "C":
            ret._f_contiguous = False
        return ret

    def argmax(
        self,
        /,
        *,
        axis=None,
        out=None,
        keepdims=False,
    ):
        return np_frontend.argmax(
            self,
            axis=axis,
            out=out,
            keepdims=keepdims,
        )

    def reshape(self, newshape, /, *, order="C"):
        aikit.utils.assertions.check_elem_in_list(
            order,
            ["C", "F", "A"],
            message="order must be one of 'C', 'F', or 'A'",
        )
        if (order == "A" and self._f_contiguous) or order == "F":
            return np_frontend.reshape(self, newshape, order="F")
        else:
            return np_frontend.reshape(self, newshape, order="C")

    def resize(self, newshape, /, *, refcheck=True):
        return np_frontend.resize(self, newshape, refcheck)

    def transpose(self, axes, /):
        if axes and isinstance(axes[0], tuple):
            axes = axes[0]
        return np_frontend.transpose(self, axes=axes)

    def swapaxes(self, axis1, axis2, /):
        return np_frontend.swapaxes(self, axis1, axis2)

    def all(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        if not (dtype is None or aikit.is_bool_dtype(dtype)):
            raise TypeError(
                "No loop matching the specified signature and "
                "casting was found for ufunc logical_or"
            )
        return np_frontend.all(self, axis, out, keepdims, where=where)

    def any(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        if not (dtype is None or aikit.is_bool_dtype(dtype)):
            raise TypeError(
                "No loop matching the specified signature and "
                "casting was found for ufunc logical_or"
            )
        return np_frontend.any(self, axis, out, keepdims, where=where)

    def argsort(self, *, axis=-1, kind=None, order=None):
        return np_frontend.argsort(self, axis=axis, kind=kind, order=order)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        return np_frontend.mean(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            where=where,
        )

    def min(self, *, axis=None, out=None, keepdims=False, initial=None, where=True):
        return np_frontend.amin(
            self,
            axis=axis,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def max(self, *, axis=None, out=None, keepdims=False, initial=None, where=True):
        return np_frontend.amax(
            self,
            axis=axis,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def argmin(
        self,
        /,
        *,
        axis=None,
        keepdims=False,
        out=None,
    ):
        return np_frontend.argmin(
            self,
            axis=axis,
            keepdims=keepdims,
            out=out,
        )

    def clip(
        self,
        min,
        max,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        return np_frontend.clip(
            self,
            min,
            max,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def compress(self, condition, axis=None, out=None):
        return np_frontend.compress(
            condition=condition,
            a=self,
            axis=axis,
            out=out,
        )

    def conjugate(
        self,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        return np_frontend.conjugate(
            self.aikit_array,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def cumprod(self, *, axis=None, dtype=None, out=None):
        return np_frontend.cumprod(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
        )

    def cumsum(self, *, axis=None, dtype=None, out=None):
        return np_frontend.cumsum(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
        )

    def dot(self, b, out=None):
        return np_frontend.dot(self, b, out=out)

    def diagonal(self, *, offset=0, axis1=0, axis2=1):
        return np_frontend.diagonal(
            self,
            offset=offset,
            axis1=axis1,
            axis2=axis2,
        )

    def sort(self, *, axis=-1, kind=None, order=None):
        return np_frontend.sort(self, axis=axis, kind=kind, order=order)

    def copy(self, order="C"):
        return np_frontend.copy(self, order=order)

    def nonzero(
        self,
    ):
        return np_frontend.nonzero(self)[0]

    def ravel(self, order="C"):
        aikit.utils.assertions.check_elem_in_list(
            order,
            ["C", "F", "A", "K"],
            message="order must be one of 'C', 'F', 'A', or 'K'",
        )
        if (order in ["K", "A"] and self._f_contiguous) or order == "F":
            return np_frontend.ravel(self, order="F")
        else:
            return np_frontend.ravel(self, order="C")

    def flatten(self, order="C"):
        aikit.utils.assertions.check_elem_in_list(
            order,
            ["C", "F", "A", "K"],
            message="order must be one of 'C', 'F', 'A', or 'K'",
        )
        if (order in ["K", "A"] and self._f_contiguous) or order == "F":
            return np_frontend.ravel(self, order="F")
        else:
            return np_frontend.ravel(self, order="C")

    def fill(self, num, /):
        self.aikit_array = np_frontend.full(self.shape, num).aikit_array
        return

    def repeat(self, repeats, axis=None):
        return np_frontend.repeat(self, repeats, axis=axis)

    def searchsorted(self, v, side="left", sorter=None):
        return np_frontend.searchsorted(self, v, side=side, sorter=sorter)

    def squeeze(self, axis=None):
        return np_frontend.squeeze(self, axis=axis)

    def std(
        self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True
    ):
        return np_frontend.std(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=where,
        )

    def tobytes(self, order="C"):
        return _to_bytes_helper(self.aikit_array, order=order)

    def tostring(self, order="C"):
        warnings.warn(
            "DeprecationWarning: tostring() is deprecated. Use tobytes() instead."
        )
        return self.tobytes(order=order)

    def prod(
        self,
        *,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        return np_frontend.prod(
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            initial=initial,
            where=where,
            out=out,
        )

    def sum(
        self,
        *,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        initial=None,
        where=True,
    ):
        return np_frontend.sum(
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            initial=initial,
            where=where,
            out=out,
        )

    def tofile(self, fid, /, sep="", format_="%s"):
        if self.ndim == 0:
            string = str(self)
        else:
            string = sep.join([str(item) for item in self.tolist()])
        with open(fid, "w") as f:
            f.write(string)

    def tolist(self) -> list:
        return self._aikit_array.to_list()

    @with_supported_device_and_dtypes(
        {
            "1.26.2 and below": {
                "cpu": (
                    "int64",
                    "float32",
                    "float64",
                    "bfloat16",
                    "complex64",
                    "complex128",
                    "uint64",
                )
            }
        },
        "numpy",
    )
    def trace(self, *, offset=0, axis1=0, axis2=1, out=None):
        return np_frontend.trace(
            self,
            offset=offset,
            axis1=axis1,
            axis2=axis2,
            out=out,
        )

    def view(self):
        return np_frontend.reshape(self, tuple(self.shape))

    def __add__(self, value, /):
        return np_frontend.add(self, value)

    def __radd__(self, value, /):
        return np_frontend.add(self, value)

    def __sub__(self, value, /):
        return np_frontend.subtract(self, value)

    def __mul__(self, value, /):
        return np_frontend.multiply(self, value)

    def __rmul__(self, value, /):
        return np_frontend.multiply(value, self)

    def __truediv__(self, value, /):
        return np_frontend.true_divide(self, value)

    def __floordiv__(self, value, /):
        return np_frontend.floor_divide(self, value)

    def __rtruediv__(self, value, /):
        return np_frontend.true_divide(value, self)

    def __pow__(self, value, /):
        return np_frontend.power(self, value)

    def __and__(self, value, /):
        return np_frontend.logical_and(self, value)

    def __or__(self, value, /):
        return np_frontend.logical_or(self, value)

    def __xor__(self, value, /):
        return np_frontend.logical_xor(self, value)

    def __matmul__(self, value, /):
        return np_frontend.matmul(self, value)

    def __copy__(
        self,
    ):
        return np_frontend.copy(self)

    def __deepcopy__(self, memo, /):
        return self.aikit_array.__deepcopy__(memo)

    def __neg__(
        self,
    ):
        return np_frontend.negative(self)

    def __pos__(
        self,
    ):
        return np_frontend.positive(self)

    def __bool__(
        self,
    ):
        if isinstance(self.aikit_array, int):
            return self.aikit_array != 0

        temp = aikit.squeeze(aikit.asarray(self.aikit_array), axis=None)
        if aikit.get_num_dims(temp) > 1:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous. "
                "Use a.any() or a.all()"
            )

        return temp != 0

    def __ne__(self, value, /):
        return np_frontend.not_equal(self, value)

    def __len__(self):
        return len(self.aikit_array)

    def __eq__(self, value, /):
        return np_frontend.equal(self, value)

    def __ge__(self, value, /):
        return np_frontend.greater_equal(self, value)

    def __gt__(self, value, /):
        return np_frontend.greater(self, value)

    def __le__(self, value, /):
        return np_frontend.less_equal(self, value)

    def __lt__(self, value, /):
        return np_frontend.less(self, value)

    def __int__(
        self,
    ):
        if "complex" in self.dtype.name:
            raise TypeError(
                "int() argument must be a string, a bytes-like object or a number, not"
                " 'complex"
            )
        return int(self.aikit_array)

    def __float__(
        self,
    ):
        if "complex" in self.dtype.name:
            raise TypeError(
                "float() argument must be a string or a real number, not 'complex"
            )
        return float(self.aikit_array)

    def __complex__(
        self,
    ):
        return complex(self.aikit_array)

    def __contains__(self, key, /):
        return np_frontend.any(self == key)

    def __iadd__(self, value, /):
        return np_frontend.add(self, value, out=self)

    def __isub__(self, value, /):
        return np_frontend.subtract(self, value, out=self)

    def __imul__(self, value, /):
        return np_frontend.multiply(self, value, out=self)

    def __itruediv__(self, value, /):
        return np_frontend.true_divide(self, value, out=self)

    def __ifloordiv__(self, value, /):
        return np_frontend.floor_divide(self, value, out=self)

    def __ipow__(self, value, /):
        return np_frontend.power(self, value, out=self)

    def __iand__(self, value, /):
        return np_frontend.logical_and(self, value, out=self)

    def __ior__(self, value, /):
        return np_frontend.logical_or(self, value, out=self)

    def __ixor__(self, value, /):
        return np_frontend.logical_xor(self, value, out=self)

    def __imod__(self, value, /):
        return np_frontend.mod(self, value, out=self)

    def __invert__(self, /):
        return aikit.bitwise_invert(self.aikit_array)

    def __abs__(self):
        return np_frontend.absolute(self)

    def __array__(self, dtype=None, /):
        if not dtype:
            return aikit.to_numpy(self.aikit_array)
        return aikit.to_numpy(self.aikit_array).astype(dtype)

    def __array_wrap__(self, array, context=None, /):
        return np_frontend.array(array)

    def __getitem__(self, key, /):
        aikit_args = aikit.nested_map(_to_aikit_array, [self, key])
        ret = aikit.get_item(*aikit_args)
        return np_frontend.ndarray(ret, _init_overload=True)

    def __setitem__(self, key, value, /):
        key, value = aikit.nested_map(_to_aikit_array, [key, value])
        self.aikit_array[key] = value

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d ndarray not supported")
        for i in range(self.shape[0]):
            yield self[i]

    def __mod__(self, value, /):
        return np_frontend.mod(self, value, out=self)

    def ptp(self, *, axis=None, out=None, keepdims=False):
        xmax = self.max(axis=axis, out=out, keepdims=keepdims)
        xmin = self.min(axis=axis, out=out, keepdims=keepdims)
        return np_frontend.subtract(xmax, xmin)

    def item(self, *args):
        if len(args) == 0:
            return self[0].aikit_array.to_scalar()
        elif len(args) == 1 and isinstance(args[0], int):
            index = args[0]
            return self.aikit_array.flatten()[index].to_scalar()
        else:
            out = self
            for index in args:
                out = out[index]
            return out.aikit_array.to_scalar()

    def __rshift__(self, value, /):
        return aikit.bitwise_right_shift(self.aikit_array, value)

    def __lshift__(self, value, /):
        return aikit.bitwise_left_shift(self.aikit_array, value)

    def __ilshift__(self, value, /):
        return aikit.bitwise_left_shift(self.aikit_array, value, out=self)

    def round(self, decimals=0, out=None):
        return np_frontend.round(self, decimals=decimals, out=out)

    def var(
        self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True
    ):
        return np_frontend.var(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=where,
        )

    def __irshift__(self, value, /):
        return aikit.bitwise_right_shift(self.aikit_array, value, out=self)


# --- Helpers --- #
# --------------- #


# tobytes helper function
def _to_bytes_helper(array, order="C"):
    def _integers_bytes_repr(item_val, /, *, dtype=None):
        if dtype == aikit.int8:
            return item_val.to_bytes(1, byteorder="big", signed=True)
        elif dtype == aikit.int16:
            return struct.pack("h", item_val)
        elif dtype == aikit.int32:
            return struct.pack("i", item_val)
        elif dtype == aikit.int64:
            return struct.pack("q", item_val)

    def _float_bytes_repr(item_val, /, *, dtype=None):
        if dtype == aikit.float16:
            return struct.pack("e", item_val)
        elif dtype == aikit.float32:
            return struct.pack("f", item_val)
        return struct.pack("d", item_val)

    def _bool_bytes_repr(item_val, /):
        return struct.pack("?", item_val)

    def _complex_bytes_repr(item_val, /, *, dtype=None):
        if dtype == aikit.complex64:
            # complex64 is represented as two 32-bit floats
            return struct.pack("ff", item_val.real, item_val.imag)

        elif dtype == aikit.complex128:
            # complex128 is represented as two 64-bit floats
            return struct.pack("dd", item_val.real, item_val.imag)

    def _unsigned_int_bytes_repr(item_val, /, *, dtype=None):
        if dtype == aikit.uint8:
            return item_val.to_bytes(1, byteorder="little", signed=False)
        elif dtype == aikit.uint16:
            return struct.pack("H", item_val)
        elif dtype == aikit.uint32:
            return struct.pack("I", item_val)
        elif dtype == aikit.uint64:
            return struct.pack("Q", item_val)

    if aikit.get_num_dims(array) == 0:
        scalar_value = aikit.to_scalar(array)
        dtype = aikit.dtype(array)
        if aikit.is_int_dtype(dtype) and not aikit.is_uint_dtype(dtype):
            return _integers_bytes_repr(scalar_value, dtype=dtype)

        elif aikit.is_float_dtype(dtype):
            return _float_bytes_repr(scalar_value, dtype=dtype)

        elif aikit.is_bool_dtype(dtype):
            return _bool_bytes_repr(scalar_value)

        elif aikit.is_complex_dtype(dtype):
            return _complex_bytes_repr(scalar_value, dtype=dtype)

        elif aikit.is_uint_dtype(dtype):
            return _unsigned_int_bytes_repr(scalar_value, dtype=dtype)
        else:
            raise ValueError("Unsupported data type for the array.")
    else:
        if order == "F":
            array = np_frontend.ravel(array, order="F").aikit_array
        array = aikit.flatten(array)
        if aikit.is_int_dtype(array) and not aikit.is_uint_dtype(array):
            bytes_reprs = [
                _integers_bytes_repr(item, dtype=aikit.dtype(array))
                for item in array.to_list()
            ]
            return b"".join(bytes_reprs)

        elif aikit.is_float_dtype(array):
            bytes_reprs = [
                _float_bytes_repr(item, dtype=aikit.dtype(array))
                for item in array.to_list()
            ]
            return b"".join(bytes_reprs)

        elif aikit.is_bool_dtype(array):
            bytes_reprs = [_bool_bytes_repr(item) for item in array.to_list()]
            return b"".join(bytes_reprs)

        elif aikit.is_complex_dtype(array):
            bytes_reprs = [
                _complex_bytes_repr(item, dtype=aikit.dtype(array))
                for item in array.to_list()
            ]
            return b"".join(bytes_reprs)

        elif aikit.is_uint_dtype(array):
            bytes_reprs = [
                _unsigned_int_bytes_repr(item, dtype=aikit.dtype(array))
                for item in array.to_list()
            ]
            return b"".join(bytes_reprs)
        else:
            raise ValueError("Unsupported data type for the array.")
