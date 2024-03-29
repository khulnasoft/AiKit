# global
import sys
import numpy as np

# local
import aikit
from aikit.func_wrapper import _dtype_from_version

backend_version = {"version": np.__version__}

# noinspection PyUnresolvedReferences
if not aikit.is_local():
    _module_in_memory = sys.modules[__name__]
else:
    _module_in_memory = sys.modules[aikit.import_module_path].import_cache[__name__]

use = aikit.utils.backend.ContextManager(_module_in_memory)

# wrap __array_ufunc__ method of aikit.Array to prioritize Aikit array methods when using numpu backend


def wrap__array_ufunc__(func):
    def rep_method(self, ufunc, method, *inputs, **kwargs):
        methods = {
            "not_equal": "not_equal",
            "greater": "greater",
            "less": "less",
            "greater_equal": "greater_equal",
            "less_equal": "less_equal",
            "multiply": "multiply",
            "divide": "divide",
            "remainder": "remainder",
            "equal": "equal",
            "bitwise_and": "bitwise_and",
            "matmul": "matmul",
            "power": "pow",
            "subtract": "subtract",
            "add": "add",
        }
        if ufunc.__name__ in methods:
            return eval("aikit." + methods[ufunc.__name__] + "(*inputs, **kwargs)")
        return func(self, ufunc, method, *inputs, **kwargs)

    return rep_method


aikit.Array.__array_ufunc__ = wrap__array_ufunc__(aikit.Array.__array_ufunc__)

NativeArray = np.ndarray
NativeDevice = str
NativeDtype = np.dtype
NativeShape = tuple

NativeSparseArray = None


# devices
valid_devices = ("cpu",)

invalid_devices = ("gpu", "tpu")

# native data types
native_int8 = np.dtype("int8")
native_int16 = np.dtype("int16")
native_int32 = np.dtype("int32")
native_int64 = np.dtype("int64")
native_uint8 = np.dtype("uint8")
native_uint16 = np.dtype("uint16")
native_uint32 = np.dtype("uint32")
native_uint64 = np.dtype("uint64")
native_float16 = np.dtype("float16")
native_float32 = np.dtype("float32")
native_float64 = np.dtype("float64")
native_complex64 = np.dtype("complex64")
native_complex128 = np.dtype("complex128")
native_double = native_float64
native_bool = np.dtype("bool")

# valid data types
# ToDo: Add complex dtypes to valid_dtypes and fix all resulting failures.

# update these to add new dtypes
valid_dtypes = {
    "1.26.3 and below": (
        aikit.int8,
        aikit.int16,
        aikit.int32,
        aikit.int64,
        aikit.uint8,
        aikit.uint16,
        aikit.uint32,
        aikit.uint64,
        aikit.float16,
        aikit.float32,
        aikit.float64,
        aikit.complex64,
        aikit.complex128,
        aikit.bool,
    )
}
valid_numeric_dtypes = {
    "1.26.3 and below": (
        aikit.int8,
        aikit.int16,
        aikit.int32,
        aikit.int64,
        aikit.uint8,
        aikit.uint16,
        aikit.uint32,
        aikit.uint64,
        aikit.float16,
        aikit.float32,
        aikit.float64,
        aikit.complex64,
        aikit.complex128,
    )
}
valid_int_dtypes = {
    "1.26.3 and below": (
        aikit.int8,
        aikit.int16,
        aikit.int32,
        aikit.int64,
        aikit.uint8,
        aikit.uint16,
        aikit.uint32,
        aikit.uint64,
    )
}
valid_float_dtypes = {"1.26.3 and below": (aikit.float16, aikit.float32, aikit.float64)}
valid_uint_dtypes = {
    "1.26.3 and below": (aikit.uint8, aikit.uint16, aikit.uint32, aikit.uint64)
}
valid_complex_dtypes = {"1.26.3 and below": (aikit.complex64, aikit.complex128)}

# leave these untouched
valid_dtypes = _dtype_from_version(valid_dtypes, backend_version)
valid_numeric_dtypes = _dtype_from_version(valid_numeric_dtypes, backend_version)
valid_int_dtypes = _dtype_from_version(valid_int_dtypes, backend_version)
valid_float_dtypes = _dtype_from_version(valid_float_dtypes, backend_version)
valid_uint_dtypes = _dtype_from_version(valid_uint_dtypes, backend_version)
valid_complex_dtypes = _dtype_from_version(valid_complex_dtypes, backend_version)

# invalid data types
# update these to add new dtypes
invalid_dtypes = {"1.26.3 and below": (aikit.bfloat16,)}
invalid_numeric_dtypes = {"1.26.3 and below": (aikit.bfloat16,)}
invalid_int_dtypes = {"1.26.3 and below": ()}
invalid_float_dtypes = {"1.26.3 and below": (aikit.bfloat16,)}
invalid_uint_dtypes = {"1.26.3 and below": ()}
invalid_complex_dtypes = {"1.26.3 and below": ()}


# leave these untouched
invalid_dtypes = _dtype_from_version(invalid_dtypes, backend_version)
invalid_numeric_dtypes = _dtype_from_version(invalid_numeric_dtypes, backend_version)
invalid_int_dtypes = _dtype_from_version(invalid_int_dtypes, backend_version)
invalid_float_dtypes = _dtype_from_version(invalid_float_dtypes, backend_version)
invalid_uint_dtypes = _dtype_from_version(invalid_uint_dtypes, backend_version)
invalid_complex_dtypes = _dtype_from_version(invalid_complex_dtypes, backend_version)


native_inplace_support = True

supports_gradients = False


def closest_valid_dtype(type=None, /, as_native=False):
    if type is None:
        type = aikit.default_dtype()
    elif isinstance(type, str) and type in invalid_dtypes:
        type = {"bfloat16": aikit.float16}[type]
    return aikit.as_aikit_dtype(type) if not as_native else aikit.as_native_dtype(type)


backend = "numpy"


# local sub-modules
from . import activations
from .activations import *
from . import creation
from .creation import *
from . import data_type
from .data_type import *
from . import device
from .device import *
from . import elementwise
from .elementwise import *
from . import general
from .general import *
from . import gradients
from .gradients import *
from . import layers
from .layers import *
from . import linear_algebra as linalg
from .linear_algebra import *
from . import manipulation
from .manipulation import *
from . import random
from .random import *
from . import searching
from .searching import *
from . import set
from .set import *
from . import sorting
from .sorting import *
from . import statistical
from .statistical import *
from . import utility
from .utility import *
from . import experimental
from .experimental import *
from . import control_flow_ops
from .control_flow_ops import *
from . import module
from .module import *


# sub-backends

from . import sub_backends
from .sub_backends import *


NativeModule = None
