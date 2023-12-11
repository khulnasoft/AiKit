# global

# local
import aikit
import aikit.functional.frontends.jax as jax_frontend
from aikit.func_wrapper import with_unsupported_dtypes


class Array:
    def __init__(self, array, weak_type=False):
        self._aikit_array = array if isinstance(array, aikit.Array) else aikit.array(array)
        self.weak_type = weak_type

    def __repr__(self):
        main = (
            str(self.aikit_array.__repr__())
            .replace("aikit.array", "aikit.frontends.jax.Array")
            .replace(")", "")
            + ", dtype="
            + str(self.aikit_array.dtype)
        )
        if self.weak_type:
            return main + ", weak_type=True)"
        return main + ")"

    # Properties #
    # ---------- #

    @property
    def aikit_array(self):
        return self._aikit_array

    @property
    def dtype(self):
        return self.aikit_array.dtype

    @property
    def shape(self):
        return tuple(self.aikit_array.shape.shape)

    @property
    def at(self):
        return jax_frontend._src.numpy.lax_numpy._IndexUpdateHelper(self.aikit_array)

    @property
    def T(self):
        return self.aikit_array.T

    @property
    def ndim(self):
        return self.aikit_array.ndim

    # Instance Methods #
    # ---------------- #

    def copy(self, order=None):
        return jax_frontend.numpy.copy(self._aikit_array, order=order)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        return jax_frontend.numpy.diagonal(
            self._aikit_array, offset=offset, axis1=axis1, axis2=axis2
        )

    def all(self, *, axis=None, out=None, keepdims=False):
        return jax_frontend.numpy.all(
            self._aikit_array, axis=axis, keepdims=keepdims, out=out
        )

    def astype(self, dtype):
        try:
            return jax_frontend.numpy.asarray(self, dtype=dtype)
        except:  # noqa: E722
            raise aikit.utils.exceptions.AikitException(
                f"Dtype {self.dtype} is not castable to {dtype}"
            )

    @with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
    def argmax(
        self,
        /,
        *,
        axis=None,
        out=None,
        keepdims=False,
    ):
        return jax_frontend.numpy.argmax(
            self,
            axis=axis,
            out=out,
            keepdims=keepdims,
        )

    @with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
    def argmin(
        self,
        /,
        *,
        axis=None,
        out=None,
        keepdims=False,
    ):
        return jax_frontend.numpy.argmin(
            self,
            axis=axis,
            out=out,
            keepdims=keepdims,
        )

    def squeeze(self, axis=None):
        return jax_frontend.numpy.squeeze(self, axis=axis)

    def conj(self, /):
        return jax_frontend.numpy.conj(self._aikit_array)

    def conjugate(self, /):
        return jax_frontend.numpy.conjugate(self._aikit_array)

    def mean(self, *, axis=None, dtype=None, out=None, keepdims=False, where=None):
        return jax_frontend.numpy.mean(
            self._aikit_array,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            where=where,
        )

    def cumprod(self, axis=None, dtype=None, out=None):
        return jax_frontend.numpy.cumprod(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
        )

    def cumsum(self, axis=None, dtype=None, out=None):
        return jax_frontend.numpy.cumsum(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
        )

    def nonzero(self, *, size=None, fill_value=None):
        return jax_frontend.numpy.nonzero(
            self,
            size=size,
            fill_value=fill_value,
        )

    def prod(
        self,
        axis=None,
        dtype=None,
        keepdims=False,
        initial=None,
        where=None,
        promote_integers=True,
        out=None,
    ):
        return jax_frontend.numpy.product(
            self,
            axis=axis,
            dtype=self.dtype,
            keepdims=keepdims,
            initial=initial,
            where=where,
            promote_integers=promote_integers,
            out=out,
        )

    def ravel(self, order="C"):
        return jax_frontend.numpy.ravel(
            self,
            order=order,
        )

    flatten = ravel

    def sort(self, axis=-1, order=None):
        return jax_frontend.numpy.sort(
            self,
            axis=axis,
            order=order,
        )

    def sum(
        self,
        axis=None,
        dtype=None,
        out=None,
        keepdims=False,
        initial=None,
        where=None,
        promote_integers=True,
    ):
        return jax_frontend.numpy.sum(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
            promote_integers=promote_integers,
        )

    def argsort(self, axis=-1, kind="stable", order=None):
        return jax_frontend.numpy.argsort(self, axis=axis, kind=kind, order=order)

    def any(self, *, axis=None, out=None, keepdims=False, where=None):
        return jax_frontend.numpy.any(
            self._aikit_array, axis=axis, keepdims=keepdims, out=out, where=where
        )

    def reshape(self, *args, order="C"):
        if not isinstance(args[0], int):
            if len(args) > 1:
                raise TypeError(
                    "Shapes must be 1D sequences of concrete values of integer type,"
                    f" got {args}."
                )
            args = args[0]
        return jax_frontend.numpy.reshape(self, tuple(args), order)

    def __add__(self, other):
        return jax_frontend.numpy.add(self, other)

    def __radd__(self, other):
        return jax_frontend.numpy.add(other, self)

    def __sub__(self, other):
        return jax_frontend.lax.sub(self, other)

    def __rsub__(self, other):
        return jax_frontend.lax.sub(other, self)

    def __mul__(self, other):
        return jax_frontend.lax.mul(self, other)

    def __rmul__(self, other):
        return jax_frontend.lax.mul(other, self)

    def __div__(self, other):
        return jax_frontend.numpy.divide(self, other)

    def __rdiv__(self, other):
        return jax_frontend.numpy.divide(other, self)

    def __mod__(self, other):
        return jax_frontend.numpy.mod(self, other)

    def __rmod__(self, other):
        return jax_frontend.numpy.mod(other, self)

    def __truediv__(self, other):
        return jax_frontend.numpy.divide(self, other)

    def __rtruediv__(self, other):
        return jax_frontend.numpy.divide(other, self)

    def __matmul__(self, other):
        return jax_frontend.numpy.dot(self, other)

    def __rmatmul__(self, other):
        return jax_frontend.numpy.dot(other, self)

    def __pos__(self):
        return self

    def __neg__(self):
        return jax_frontend.lax.neg(self)

    def __eq__(self, other):
        return jax_frontend.lax.eq(self, other)

    def __ne__(self, other):
        return jax_frontend.lax.ne(self, other)

    def __lt__(self, other):
        return jax_frontend.lax.lt(self, other)

    def __le__(self, other):
        return jax_frontend.lax.le(self, other)

    def __gt__(self, other):
        return jax_frontend.lax.gt(self, other)

    def __ge__(self, other):
        return jax_frontend.lax.ge(self, other)

    def __abs__(self):
        return jax_frontend.numpy.abs(self)

    def __pow__(self, other):
        return jax_frontend.lax.pow(self, other)

    def __rpow__(self, other):
        other = aikit.asarray(other)
        return jax_frontend.lax.pow(other, self)

    def __and__(self, other):
        return jax_frontend.numpy.bitwise_and(self, other)

    def __rand__(self, other):
        return jax_frontend.numpy.bitwise_and(other, self)

    def __or__(self, other):
        return jax_frontend.numpy.bitwise_or(self, other)

    def __ror__(self, other):
        return jax_frontend.numpy.bitwise_or(other, self)

    def __xor__(self, other):
        return jax_frontend.lax.bitwise_xor(self, other)

    def __rxor__(self, other):
        return jax_frontend.lax.bitwise_xor(other, self)

    def __invert__(self):
        return jax_frontend.lax.bitwise_not(self)

    def __lshift__(self, other):
        return jax_frontend.lax.shift_left(self, other)

    def __rlshift__(self, other):
        return jax_frontend.lax.shift_left(other, self)

    def __rshift__(self, other):
        return jax_frontend.lax.shift_right_logical(self, other)

    def __rrshift__(self, other):
        return jax_frontend.lax.shift_right_logical(other, self)

    def __getitem__(self, idx):
        return self.at[idx].get()

    def __setitem__(self, idx, val):
        raise aikit.utils.exceptions.AikitException(
            "aikit.functional.frontends.jax.Array object doesn't support assignment"
        )

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d Array not supported")
        for i in range(self.shape[0]):
            yield self[i]

    def round(self, decimals=0):
        return jax_frontend.numpy.round(self, decimals)

    def repeat(self, repeats, axis=None, *, total_repeat_length=None):
        return jax_frontend.numpy.repeat(self, repeats, axis=axis)

    def searchsorted(self, v, side="left", sorter=None, *, method="scan"):
        return jax_frontend.numpy.searchsorted(self, v, side=side, sorter=sorter)

    def max(
        self,
        /,
        *,
        axis=None,
        out=None,
        keepdims=False,
        where=None,
    ):
        return jax_frontend.numpy.max(
            self, axis=axis, out=out, keepdims=keepdims, where=where
        )

    def ptp(self, *, axis=None, out=None, keepdims=False):
        return jax_frontend.numpy.ptp(self, axis=axis, keepdims=keepdims)

    def min(
        self,
        /,
        *,
        axis=None,
        out=None,
        keepdims=False,
        where=None,
    ):
        return jax_frontend.numpy.min(
            self, axis=axis, out=out, keepdims=keepdims, where=where
        )

    def std(
        self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=None
    ):
        return jax_frontend.numpy.std(
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=where,
        )

    def var(
        self, *, axis=None, dtype=None, out=None, ddof=False, keepdims=False, where=None
    ):
        return jax_frontend.numpy.var(
            self._aikit_array,
            axis=axis,
            dtype=dtype,
            out=out,
            ddof=int(ddof),
            keepdims=keepdims,
            where=where,
        )

    def swapaxes(self, axis1, axis2):
        return jax_frontend.numpy.swapaxes(self, axis1=axis1, axis2=axis2)


# Jax supports DeviceArray from 0.4.13 and below
# Hence aliasing it here
DeviceArray = Array
