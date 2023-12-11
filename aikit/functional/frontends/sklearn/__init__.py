from . import tree
import aikit
from aikit.functional.frontends.numpy import array

_int8 = aikit.IntDtype("int8")
_int16 = aikit.IntDtype("int16")
_int32 = aikit.IntDtype("int32")
_int64 = aikit.IntDtype("int64")
_uint8 = aikit.UintDtype("uint8")
_uint16 = aikit.UintDtype("uint16")
_uint32 = aikit.UintDtype("uint32")
_uint64 = aikit.UintDtype("uint64")
_bfloat16 = aikit.FloatDtype("bfloat16")
_float16 = aikit.FloatDtype("float16")
_float32 = aikit.FloatDtype("float32")
_float64 = aikit.FloatDtype("float64")
_complex64 = aikit.ComplexDtype("complex64")
_complex128 = aikit.ComplexDtype("complex128")
_bool = aikit.Dtype("bool")

_frontend_array = array
