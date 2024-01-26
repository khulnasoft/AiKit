# local
import aikit
import aikit.functional.frontends.tensorflow as tf_frontend
import aikit.functional.frontends.numpy as np_frontend
from aikit.functional.frontends.tensorflow.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_tf_dtype,
)


class DType:
    def __init__(self, dtype_int):
        self._aikit_dtype = tf_frontend.tensorflow_enum_to_type[dtype_int]

    def __repr__(self):
        return "aikit.frontends.tensorflow." + self._aikit_dtype

    @property
    def aikit_dtype(self):
        return self._aikit_dtype

    @property
    def as_datatype_enum(self):
        return tf_frontend.tensorflow_type_to_enum[self._aikit_dtype]

    @property
    def as_numpy_dtype(self):
        return np_frontend.dtype(self._aikit_dtype)

    @property
    def base_dtype(self):
        return self

    @property
    def is_bool(self):
        return self._aikit_dtype.is_bool_dtype

    @property
    def is_complex(self):
        return "complex" in self._aikit_dtype

    @property
    def is_floating(self):
        return self._aikit_dtype.is_float_dtype

    @property
    def is_integer(self):
        return self._aikit_dtype.is_int_dtype

    @property
    def is_numpy_compatible(self):
        return self._aikit_dtype in np_frontend.numpy_type_to_str_and_num_table

    @property
    def is_unsigned(self):
        return self._aikit_dtype.is_uint_dtype

    @property
    def limits(self):
        if self._aikit_dtype is aikit.bool:
            return False, True
        if self._aikit_dtype.is_int_dtype:
            return 0, self._aikit_dtype.info.max
        if self._aikit_dtype.is_float_dtype:
            return 0, 1
        else:
            raise aikit.utils.exceptions.AikitException(
                f"{self._aikit_dtype} does not have defined limits"
            )

    @property
    def max(self):
        if self._aikit_dtype in (aikit.bool, aikit.complex128, aikit.complex64):
            raise aikit.utils.exceptions.AikitException(
                f"Cannot find maximum value of {self._aikit_dtype}"
            )
        if self._aikit_dtype is aikit.bfloat16:
            return float.fromhex("0x1.FEp127")
        return self._aikit_dtype.info.max

    @property
    def min(self):
        if self._aikit_dtype in (aikit.bool, aikit.complex128, aikit.complex64):
            raise aikit.utils.exceptions.AikitException(
                f"Cannot find maximum value of {self._aikit_dtype}"
            )
        if self._aikit_dtype is aikit.bfloat16:
            return float.fromhex("-0x1.FEp127")
        return self._aikit_dtype.info.min

    @property
    def real_dtype(self):
        if self._aikit_dtype is aikit.complex64:
            return DType(1)
        if self._aikit_dtype is aikit.complex128:
            return DType(2)
        else:
            return self

    def __eq__(self, other):
        if other is None:
            return False

        if not isinstance(other, DType):
            try:
                other = as_dtype(other)
            except aikit.utils.exceptions.AikitException:
                return False

        return self._aikit_dtype == other._aikit_dtype

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(repr(self))


def as_dtype(type_value):
    if isinstance(type_value, DType):
        return type_value
    if aikit.is_native_dtype(type_value):
        return DType(tf_frontend.tensorflow_type_to_enum[aikit.as_aikit_dtype(type_value)])
    if type_value in tf_frontend.tensorflow_enum_to_type:
        return DType(type_value)
    if type_value in tf_frontend.tensorflow_type_to_enum:
        return DType(tf_frontend.tensorflow_type_to_enum[type_value])
    if type_value is float:
        return DType(1)
    if type_value is bool:
        return DType(10)
    if isinstance(type_value, np_frontend.dtype):
        return DType(tf_frontend.tensorflow_type_to_enum[type_value.aikit_dtype])
    if issubclass(type_value, np_frontend.generic):
        return DType(
            tf_frontend.tensorflow_type_to_enum[
                np_frontend.numpy_scalar_to_dtype[type_value]
            ]
        )
    raise aikit.utils.exceptions.AikitException(
        f"Cannot convert the argument 'type_value': {type_value!r} "
        "to a TensorFlow Dtype"
    )


@handle_tf_dtype
@to_aikit_arrays_and_back
def cast(x, dtype, name=None):
    return aikit.astype(x, dtype, copy=False)
