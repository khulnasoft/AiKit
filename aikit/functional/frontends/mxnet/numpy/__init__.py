import aikit
from aikit.utils.exceptions import handle_exceptions
from numbers import Number
from typing import Union, Tuple, Iterable


# Constructing dtypes are required as aikit.<dtype>
# will change dynamically on the backend and may not be available
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

mxnet_promotion_table = {
    (_bool, _bool): _bool,
    (_bool, _int8): _int8,
    (_bool, _int32): _int32,
    (_bool, _int64): _int64,
    (_bool, _uint8): _uint8,
    (_bool, _bfloat16): _bfloat16,
    (_bool, _float16): _float16,
    (_bool, _float32): _float32,
    (_bool, _float64): _float64,
    (_int8, _bool): _int8,
    (_int8, _int8): _int8,
    (_int8, _int32): _int32,
    (_int8, _int64): _int64,
    (_int32, _bool): _int32,
    (_int32, _int8): _int32,
    (_int32, _int32): _int32,
    (_int32, _int64): _int64,
    (_int64, _bool): _int64,
    (_int64, _int8): _int64,
    (_int64, _int32): _int64,
    (_int64, _int64): _int64,
    (_uint8, _bool): _uint8,
    (_uint8, _uint8): _uint8,
    (_int32, _uint8): _int32,
    (_int64, _uint8): _int64,
    (_uint8, _int32): _int32,
    (_uint8, _int64): _int64,
    (_float16, _bool): _float16,
    (_float16, _float16): _float16,
    (_float16, _float32): _float32,
    (_float16, _float64): _float64,
    (_float32, _bool): _float32,
    (_float32, _float16): _float32,
    (_float32, _float32): _float32,
    (_float32, _float64): _float64,
    (_float64, _bool): _float64,
    (_float64, _float16): _float64,
    (_float64, _float32): _float64,
    (_float64, _float64): _float64,
    (_int8, _float16): _float16,
    (_float16, _int8): _float16,
    (_int8, _float32): _float32,
    (_float32, _int8): _float32,
    (_int8, _float64): _float64,
    (_float64, _int8): _float64,
    (_int32, _float16): _float64,
    (_float16, _int32): _float64,
    (_int32, _float32): _float64,
    (_float32, _int32): _float64,
    (_int32, _float64): _float64,
    (_float64, _int32): _float64,
    (_int64, _float16): _float64,
    (_float16, _int64): _float64,
    (_int64, _float32): _float64,
    (_float32, _int64): _float64,
    (_int64, _float64): _float64,
    (_float64, _int64): _float64,
    (_uint8, _float16): _float16,
    (_float16, _uint8): _float16,
    (_uint8, _float32): _float32,
    (_float32, _uint8): _float32,
    (_uint8, _float64): _float64,
    (_float64, _uint8): _float64,
    (_bfloat16, _bfloat16): _bfloat16,
    (_bfloat16, _uint8): _bfloat16,
    (_uint8, _bfloat16): _bfloat16,
    (_bfloat16, _int8): _bfloat16,
    (_int8, _bfloat16): _bfloat16,
    (_bfloat16, _float32): _float32,
    (_float32, _bfloat16): _float32,
    (_bfloat16, _float64): _float64,
    (_float64, _bfloat16): _float64,
}


@handle_exceptions
def promote_types_mxnet(
    type1: Union[aikit.Dtype, aikit.NativeDtype],
    type2: Union[aikit.Dtype, aikit.NativeDtype],
    /,
) -> aikit.Dtype:
    """Promote the datatypes type1 and type2, returning the data type they
    promote to.

    Parameters
    ----------
    type1
        the first of the two types to promote
    type2
        the second of the two types to promote

    Returns
    -------
    ret
        The type that both input types promote to
    """
    try:
        ret = mxnet_promotion_table[(aikit.as_aikit_dtype(type1), aikit.as_aikit_dtype(type2))]
    except KeyError as e:
        raise aikit.utils.exceptions.AikitException(
            "these dtypes are not type promotable"
        ) from e
    return ret


@handle_exceptions
def promote_types_of_mxnet_inputs(
    x1: Union[aikit.Array, Number, Iterable[Number]],
    x2: Union[aikit.Array, Number, Iterable[Number]],
    /,
) -> Tuple[aikit.Array, aikit.Array]:
    """Promote the dtype of the given native array inputs to a common dtype
    based on type promotion rules.

    While passing float or integer values or any other non-array input
    to this function, it should be noted that the return will be an
    array-like object. Therefore, outputs from this function should be
    used as inputs only for those functions that expect an array-like or
    tensor-like objects, otherwise it might give unexpected results.
    """
    type1 = aikit.default_dtype(item=x1).strip("u123456789")
    type2 = aikit.default_dtype(item=x2).strip("u123456789")
    if hasattr(x1, "dtype") and not hasattr(x2, "dtype") and type1 == type2:
        x1 = aikit.asarray(x1)
        x2 = aikit.asarray(
            x2, dtype=x1.dtype, device=aikit.default_device(item=x1, as_native=False)
        )
    elif not hasattr(x1, "dtype") and hasattr(x2, "dtype") and type1 == type2:
        x1 = aikit.asarray(
            x1, dtype=x2.dtype, device=aikit.default_device(item=x2, as_native=False)
        )
        x2 = aikit.asarray(x2)
    else:
        x1 = aikit.asarray(x1)
        x2 = aikit.asarray(x2)
        promoted = promote_types_mxnet(x1.dtype, x2.dtype)
        x1 = aikit.asarray(x1, dtype=promoted)
        x2 = aikit.asarray(x2, dtype=promoted)
    return x1, x2


from . import random
from . import ndarray
from . import linalg
from .linalg import *
from . import mathematical_functions
from .mathematical_functions import *
from . import creation
from .creation import *
from . import symbol
from .symbol import *
