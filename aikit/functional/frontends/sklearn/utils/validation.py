import aikit
from aikit.functional.frontends.numpy.func_wrapper import to_aikit_arrays_and_back
from aikit.func_wrapper import with_unsupported_dtypes


@to_aikit_arrays_and_back
def as_float_array(X, *, copy=True, force_all_finite=True):
    if X.dtype in [aikit.float32, aikit.float64]:
        return X.copy_array() if copy else X
    if ("bool" in X.dtype or "int" in X.dtype or "uint" in X.dtype) and aikit.itemsize(
        X
    ) <= 4:
        return_dtype = aikit.float32
    else:
        return_dtype = aikit.float64
    return aikit.asarray(X, dtype=return_dtype)


@with_unsupported_dtypes({"1.3.0 and below": ("complex",)}, "sklearn")
@to_aikit_arrays_and_back
def column_or_1d(y, *, warn=False):
    shape = y.shape
    if len(shape) == 2 and shape[1] == 1:
        y = aikit.reshape(y, (-1,))
    elif len(shape) > 2:
        raise ValueError("y should be a 1d array or a column vector")
    return y
