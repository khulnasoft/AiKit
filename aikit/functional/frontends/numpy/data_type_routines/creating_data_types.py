# global

# local
import aikit
import aikit.functional.frontends.numpy as np_frontend


class dtype:
    def __init__(self, dtype_in, align=False, copy=False):
        self._aikit_dtype = (
            to_aikit_dtype(dtype_in)
            if not isinstance(dtype_in, dtype)
            else dtype_in._aikit_dtype
        )

    def __repr__(self):
        return "aikit.frontends.numpy.dtype('" + self._aikit_dtype + "')"

    def __ge__(self, other):
        try:
            other = dtype(other)
        except TypeError as e:
            raise aikit.utils.exceptions.AikitException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            ) from e

        return self == np_frontend.promote_numpy_dtypes(
            self._aikit_dtype, other._aikit_dtype
        )

    def __gt__(self, other):
        try:
            other = dtype(other)
        except TypeError as e:
            raise aikit.utils.exceptions.AikitException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            ) from e

        return self >= other and self != other

    def __lt__(self, other):
        try:
            other = dtype(other)
        except TypeError as e:
            raise aikit.utils.exceptions.AikitException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            ) from e

        return self != np_frontend.promote_numpy_dtypes(
            self._aikit_dtype, other._aikit_dtype
        )

    def __le__(self, other):
        try:
            other = dtype(other)
        except TypeError as e:
            raise aikit.utils.exceptions.AikitException(
                "Attempted to compare a dtype with something which"
                "couldn't be interpreted as a dtype"
            ) from e

        return self < other or self == other

    @property
    def type(self):
        return np_frontend.numpy_dtype_to_scalar[self._aikit_dtype]

    @property
    def alignment(self):
        if self._aikit_dtype.is_bool_dtype:
            return 1
        return self._aikit_dtype.dtype_bits // 8

    @property
    def base(self):
        return self

    @property
    def char(self):
        return np_frontend.numpy_type_to_str_and_num_table[self._aikit_dtype][0]

    @property
    def byteorder(self):
        if self._aikit_dtype[-1] == 8:
            return "|"
        else:
            return "="

    @property
    def itemsize(self):
        return self._aikit_dtype.dtype_bits // 8

    @property
    def kind(self):
        if self._aikit_dtype.is_bool_dtype:
            return "b"
        elif self._aikit_dtype.is_int_dtype:
            return "i"
        elif self._aikit_dtype.is_uint_dtype:
            return "u"
        elif self._aikit_dtype.is_float_dtype:
            return "f"
        else:
            return "V"

    @property
    def num(self):
        return np_frontend.numpy_type_to_str_and_num_table[self._aikit_dtype][1]

    @property
    def shape(self):
        return ()

    @property
    def str(self):
        if self._aikit_dtype.is_bool_dtype:
            return "|b1"
        elif self._aikit_dtype.is_uint_dtype:
            if self._aikit_dtype[4::] == "8":
                return "|u1"
            return "<u" + str(self.alignment)
        elif self._aikit_dtype.is_int_dtype:
            if self._aikit_dtype[3::] == "8":
                return "|i1"
            return "<i" + str(self.alignment)
        elif self._aikit_dtype.is_float_dtype:
            return "<f" + str(self.alignment)

    @property
    def subtype(self):
        return None

    @property
    def aikit_dtype(self):
        return self._aikit_dtype

    @property
    def name(self):
        return self._aikit_dtype.__repr__()


def to_aikit_dtype(dtype_in):
    if dtype_in is None:
        return
    if isinstance(dtype_in, aikit.Dtype):
        return dtype_in
    if isinstance(dtype_in, str):
        if dtype_in.strip("><=") in np_frontend.numpy_str_to_type_table:
            return aikit.Dtype(np_frontend.numpy_str_to_type_table[dtype_in.strip("><=")])
        return aikit.Dtype(dtype_in)
    if aikit.is_native_dtype(dtype_in):
        return aikit.as_aikit_dtype(dtype_in)
    if dtype_in in (int, float, bool):
        return {int: aikit.int64, float: aikit.float64, bool: aikit.bool}[dtype_in]
    if isinstance(dtype_in, np_frontend.dtype):
        return dtype_in.aikit_dtype
    if isinstance(dtype_in, type):
        if issubclass(dtype_in, np_frontend.generic):
            return np_frontend.numpy_scalar_to_dtype[dtype_in]
        if hasattr(dtype_in, "dtype"):
            return dtype_in.dtype.aikit_dtype
    else:
        return aikit.as_aikit_dtype(dtype_in)
