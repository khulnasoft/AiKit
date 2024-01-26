# global
import aikit
import aikit.functional.frontends.tensorflow as tf_frontend
from aikit.functional.frontends.tensorflow import check_tensorflow_casting
from aikit.functional.frontends.tensorflow.func_wrapper import (
    to_aikit_arrays_and_back,
    map_raw_ops_alias,
    to_aikit_dtype,
)

from aikit.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from aikit.utils.exceptions import AikitNotImplementedException


Acos = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.acos))
Acosh = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.acosh))
Add = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.add))
AddN = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.add_n))
AddV2 = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.add))
ArgMax = to_aikit_arrays_and_back(
    with_unsupported_dtypes(
        {"2.15.0 and below": ("complex",)},
        "tensorflow",
    )(
        map_raw_ops_alias(
            tf_frontend.math.argmax, kwargs_to_update={"dimension": "axis"}
        )
    )
)
ArgMin = to_aikit_arrays_and_back(
    with_unsupported_dtypes(
        {"2.15.0 and below": ("complex",)},
        "tensorflow",
    )(
        map_raw_ops_alias(
            tf_frontend.math.argmin, kwargs_to_update={"dimension": "axis"}
        )
    )
)
Asin = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.asin))
Atan = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.atan))
Atan2 = to_aikit_arrays_and_back(
    with_unsupported_dtypes(
        {"2.15.0 and below": "float16"},
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.math.atan2))
)
ConcatV2 = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.concat))
Conj = to_aikit_arrays_and_back(
    with_supported_dtypes(
        {
            "2.13.0 and below": ("complex64", "complex128", "variant"),
        },
        "tensorflow",
    )(
        map_raw_ops_alias(
            tf_frontend.math.conj,
            kwargs_to_update={
                "input": "x",
            },
        )
    )
)
Cos = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.cos))
Cosh = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.cosh))
Cumprod = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.cumprod))
Cumsum = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.cumsum))
Digamma = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.digamma))
Div = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.divide))
Einsum = to_aikit_arrays_and_back(
    with_supported_dtypes(
        {
            "2.15.0 and below": (
                "bfloat16",
                "complex128 ",
                "complex64",
                "float64",
                "float32",
                "float16",
                "int64",
                "int32",
            ),
        },
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.general_functions.einsum))
)
Identity = to_aikit_arrays_and_back(
    map_raw_ops_alias(tf_frontend.general_functions.identity)
)
IdentityN = to_aikit_arrays_and_back(
    map_raw_ops_alias(tf_frontend.general_functions.identity_n)
)
Igamma = to_aikit_arrays_and_back(
    with_supported_dtypes(
        {
            "2.15.0 and below": (
                "float64",
                "float32",
                "half",
            ),
        },
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.math.igamma))
)
LeakyRelu = to_aikit_arrays_and_back(
    with_supported_dtypes(
        {
            "2.15.0 and below": ("bfloat16", "float16", "float32", "float64"),
        },
        "tensorflow",
    )(
        map_raw_ops_alias(
            tf_frontend.nn.leaky_relu,
        )
    )
)
LessEqual = to_aikit_arrays_and_back(
    with_unsupported_dtypes(
        {
            "2.15.0 and below": ("complex",),
        },
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.math.less_equal))
)
Log1p = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.log1p))
LogSoftmax = to_aikit_arrays_and_back(
    with_supported_dtypes(
        {
            "2.15.0 and below": (
                "bfloat16",
                "float32",
                "float64",
            ),
        },
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.math.log_softmax))
)
LogicalOr = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.logical_or))
MatrixDeterminant = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.linalg.det))
Max = to_aikit_arrays_and_back(
    with_unsupported_dtypes(
        {
            "2.15.0 and below": ("complex",),
        },
        "tensorflow",
    )(
        map_raw_ops_alias(
            tf_frontend.math.reduce_max,
            kwargs_to_update={
                "input": "input_tensor",
                "keep_dims": "keepdims",
            },
        )
    )
)
MaxPool3D = to_aikit_arrays_and_back(
    with_supported_dtypes(
        {
            "2.15.0 and below": ("float32",),
        },
        "tensorflow",
    )(
        map_raw_ops_alias(
            tf_frontend.nn.max_pool3d,
        )
    )
)
Maximum = to_aikit_arrays_and_back(
    with_unsupported_dtypes(
        {
            "2.15.0 and below": ("complex",),
        },
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.math.maximum))
)
Mean = to_aikit_arrays_and_back(
    map_raw_ops_alias(
        tf_frontend.math.reduce_mean,
        kwargs_to_update={
            "input": "input_tensor",
            "keep_dims": "keepdims",
        },
    )
)
Min = to_aikit_arrays_and_back(
    with_unsupported_dtypes(
        {
            "2.15.0 and below": ("complex",),
        },
        "tensorflow",
    )(
        map_raw_ops_alias(
            tf_frontend.math.reduce_min,
            kwargs_to_update={
                "input": "input_tensor",
                "keep_dims": "keepdims",
            },
        )
    )
)
Mod = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.mod))
Mul = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.multiply))
Neg = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.negative))
Pow = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.pow))
RealDiv = to_aikit_arrays_and_back(
    with_supported_dtypes(
        {
            "2.15.0 and below": (
                "complex",
                "bfloat16",
                "float16",
                "float64",
                "float32",
            ),
        },
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.general_functions.realdiv))
)
Reciprocal = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.reciprocal))
Relu = to_aikit_arrays_and_back(
    with_unsupported_dtypes(
        {
            "2.15.0 and below": ("complex", "float16"),
        },
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.nn.relu))
)
Relu6 = to_aikit_arrays_and_back(
    with_unsupported_dtypes(
        {
            "2.15.0 and below": ("complex", "float16"),
        },
        "tensorflow",
    )(
        map_raw_ops_alias(
            tf_frontend.nn.relu6,
        )
    )
)
Reshape = to_aikit_arrays_and_back(
    map_raw_ops_alias(tf_frontend.general_functions.reshape)
)
Roll = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.roll))
ShapeN = to_aikit_arrays_and_back(
    map_raw_ops_alias(tf_frontend.general_functions.shape_n)
)
Sigmoid = to_aikit_arrays_and_back(
    map_raw_ops_alias(tf_frontend.keras.activations.sigmoid)
)
Sin = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.sin))
Size = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.general_functions.size))
Slice = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.slice))
Softmax = to_aikit_arrays_and_back(
    with_unsupported_dtypes(
        {
            "2.15.0 and below": ("float16",),
        },
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.nn.softmax))
)
Split = to_aikit_arrays_and_back(
    map_raw_ops_alias(
        tf_frontend.split, kwargs_to_update={"num_split": "num_or_size_splits"}
    )
)
SquaredDifference = to_aikit_arrays_and_back(
    with_supported_dtypes(
        {
            "2.15.0 and below": (
                "complex",
                "bfloat16",
                "float16",
                "float64",
                "float32",
                "int32",
                "int64",
            ),
        },
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.math.squared_difference))
)
Squeeze = to_aikit_arrays_and_back(
    map_raw_ops_alias(tf_frontend.general_functions.squeeze)
)
Sub = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.subtract))
Tan = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.tan))
Tanh = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.tanh))
Tile = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.general_functions.tile))
Xlogy = to_aikit_arrays_and_back(map_raw_ops_alias(tf_frontend.math.xlogy))
Zeta = to_aikit_arrays_and_back(
    with_supported_dtypes(
        {
            "2.15.0 and below": ("float32", "float64"),
        },
        "tensorflow",
    )(map_raw_ops_alias(tf_frontend.math.zeta))
)


# --- Helpers --- #
# --------------- #


def _tf_to_aikit_aikit_arguments_for_conv(
    padding, ex_pading, strides, dilations, data_format
):
    if data_format.find("C") == 1:
        strides = strides[2:]
        dilations = dilations[2:]
        data_format = "channel_first"
        pad_index = [4, 8]
    else:
        strides = strides[1:-1]
        dilations = dilations[1:-1]
        data_format = "channel_last"
        pad_index = [2, 6]
    if padding == "EXPLICIT":
        padding = [
            (ex_pading[i], ex_pading[i + 1])
            for i in range(pad_index[0], pad_index[1], 2)
        ]
    return padding, strides, dilations, data_format


# --- Main --- #
# ------------ #


@to_aikit_arrays_and_back
def AccumulateNV2(inputs, shape, name="AccumulateNV2"):
    # TODO
    raise AikitNotImplementedException


@to_aikit_arrays_and_back
def Angle(
    *,
    input,
    Tout=aikit.float32,
    name="Angle",
):
    Tout = aikit.as_aikit_dtype(Tout) if Tout is not None else aikit.float32
    return aikit.astype(aikit.angle(input), Tout)


@with_unsupported_dtypes(
    {
        "2.15.0 and below": (
            "float16",
            "bool",
            "bfloat16",
        )
    },
    "tensorflow",
)
@to_aikit_arrays_and_back
def ApproximateEqual(
    *,
    x,
    y,
    tolerance=1e-05,
    name="ApproximateEqual",
):
    x, y = check_tensorflow_casting(x, y)
    return aikit.abs(x - y) < tolerance


@to_aikit_arrays_and_back
def Atanh(*, x, name="Atanh"):
    return aikit.atanh(x)


@to_aikit_arrays_and_back
def BandedTriangularSolve(
    matrix,
    rhs,
    lower=True,
    adjoint=False,
    name="BandedTriangularSolve",
):
    # TODO
    raise AikitNotImplementedException


@to_aikit_arrays_and_back
def BatchMatMul(x, y, adj_x=False, adj_y=False, name="BatchMatMul"):
    # TODO
    raise AikitNotImplementedException


@to_aikit_arrays_and_back
def BatchMatMulV2(x, y, adj_x=False, adj_y=False, name="BatchMatMulV2"):
    # TODO
    raise AikitNotImplementedException


@to_aikit_arrays_and_back
def BatchMatMulV3(x, y, Tout=aikit.Dtype, adj_x=False, adj_y=False, name="BatchMatMulV3"):
    # TODO
    raise AikitNotImplementedException


@to_aikit_arrays_and_back
def BitwiseAnd(*, x, y, name="BitwiseAnd"):
    x, y = check_tensorflow_casting(x, y)
    return aikit.bitwise_and(x, y)


@to_aikit_arrays_and_back
def BitwiseOr(*, x, y, name="BitwiseOr"):
    x, y = check_tensorflow_casting(x, y)
    return aikit.bitwise_or(x, y)


@to_aikit_arrays_and_back
def BitwiseXor(*, x, y, name="BitwiseXor"):
    x, y = check_tensorflow_casting(x, y)
    return aikit.bitwise_xor(x, y)


@to_aikit_arrays_and_back
def BroadcastTo(*, input, shape, name="BroadcastTo"):
    return aikit.broadcast_to(input, shape=shape)


@to_aikit_arrays_and_back
def Ceil(*, x, name=None):
    return aikit.ceil(x)


@to_aikit_arrays_and_back
def Cholesky(*, input, name="Cholesky"):
    return aikit.astype(aikit.cholesky(input), input.dtype)


@to_aikit_arrays_and_back
def Complex(real, imag, Tout=aikit.complex64, name="Complex"):
    # TODO
    raise AikitNotImplementedException


@to_aikit_arrays_and_back
def Concat(*, concat_dim, values, name="Concat"):
    return aikit.concat(values, axis=concat_dim)


@to_aikit_arrays_and_back
def Conv2D(
    *,
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu,
    explicit_paddings,
    data_format="NHWC",
    dilations=[1, 1, 1, 1],
    name="Conv2D",
):
    padding, strides, dilations, data_format = _tf_to_aikit_aikit_arguments_for_conv(
        padding, explicit_paddings, strides, dilations, data_format
    )
    return aikit.conv_general_dilated(
        input,
        filter,
        strides,
        padding,
        data_format=data_format,
        dilations=dilations,
        dims=2,
    )


@to_aikit_arrays_and_back
def Conv3D(
    *,
    input,
    filter,
    strides,
    padding,
    data_format="NDHWC",
    dilations=[1, 1, 1, 1, 1],
    name="Conv3D",
):
    # aikit.backends.tensorflow expects strides and dilations to be
    # a single integer value or a list of 3 values whereas the raw op
    # expects a list of 5 values
    if data_format == "NDHWC":
        strides = strides[1:-1]
        dilations = dilations[1:-1]
    elif data_format == "NCDHW":
        strides = strides[2:]
        dilations = dilations[2:]

    return tf_frontend.nn.conv3d(
        input,
        filter,
        strides,
        padding,
        data_format=data_format,
        dilations=dilations,
        name=name,
    )


@to_aikit_arrays_and_back
def Cross(*, a, b, name="Cross"):
    a, b = check_tensorflow_casting(a, b)
    return aikit.cross(a, b)


@to_aikit_arrays_and_back
def CumulativeLogsumexp(
    x, axis, exclusive=False, reverse=False, name="CumulativeLogsumexp"
):
    # TODO
    raise AikitNotImplementedException


@to_aikit_arrays_and_back
def DebugGradientIdentity(input, name="DebugGradientIdentity"):
    # TODO
    raise AikitNotImplementedException


@to_aikit_arrays_and_back
def Diag(*, diagonal, name="Diag"):
    return aikit.astype(aikit.diag(diagonal), diagonal.dtype)


@with_supported_dtypes(
    {"2.15.0 and below": ("bfloat16", "float16", "float32", "float64")},
    "tensorflow",
)
@to_aikit_arrays_and_back
def Elu(features, name=None):
    zeros = aikit.zeros_like(features, dtype=aikit.dtype(features))
    ones = aikit.ones_like(features, dtype=aikit.dtype(features))
    ret_val = aikit.where(
        # if x > 0 => x; else e^x - 1
        features > zeros,
        features,
        aikit.subtract(aikit.exp(features), ones),
    )
    return ret_val


@to_aikit_arrays_and_back
def Equal(*, x, y, incompatible_shape_error=True, name="Equal"):
    x, y = check_tensorflow_casting(x, y)
    if incompatible_shape_error:
        return aikit.equal(x, y)

    try:
        return aikit.equal(x, y)
    except (aikit.utils.exceptions.AikitError, aikit.utils.exceptions.AikitBackendException):
        return aikit.array(False)


@to_aikit_arrays_and_back
def EuclideanNorm(*, input, axis, keep_dims=False, name="EuclideanNorm"):
    return aikit.astype(
        aikit.vector_norm(input, axis=axis, keepdims=keep_dims), input.dtype
    )


@to_aikit_arrays_and_back
def Exp(*, x, name="Exp"):
    return aikit.exp(x)


@to_aikit_arrays_and_back
def Expm1(*, x, name="Expm1"):
    return aikit.expm1(x)


@to_aikit_arrays_and_back
def FFT(*, input, name="FFT"):
    return aikit.astype(aikit.fft(input, -1), input.dtype)


@to_aikit_arrays_and_back
def FFT2D(*, input, name="FFT2D"):
    return aikit.astype(aikit.fft2(input, dim=(-2, -1)), input.dtype)


@to_aikit_arrays_and_back
def FFT3D(*, input, name="FFT3D"):
    fft_result = aikit.fft(input, -1)
    fft_result = aikit.fft(fft_result, -2)
    fft_result = aikit.fft(fft_result, -3)
    return aikit.astype(fft_result, input.dtype)


@to_aikit_arrays_and_back
def Fill(*, dims, value, name="Full"):
    return aikit.full(dims, value)


@to_aikit_arrays_and_back
def Floor(*, x, name="Floor"):
    return aikit.floor(x)


@to_aikit_arrays_and_back
def FloorDiv(*, x, y, name="FloorDiv"):
    x, y = check_tensorflow_casting(x, y)
    return aikit.floor_divide(x, y)


@to_aikit_arrays_and_back
def FloorMod(*, x, y, name="FloorMod"):
    x, y = check_tensorflow_casting(x, y)
    return aikit.remainder(x, y)


@to_aikit_arrays_and_back
def Gather(*, params, indices, validate_indices=None, name="Gather"):
    return aikit.gather(params, indices, axis=0, batch_dims=0)


@to_aikit_arrays_and_back
def Greater(*, x, y, name="Greater"):
    x, y = check_tensorflow_casting(x, y)
    return aikit.greater(x, y)


@to_aikit_arrays_and_back
def GreaterEqual(*, x, y, name="GreaterEqual"):
    x, y = check_tensorflow_casting(x, y)
    return aikit.greater_equal(x, y)


@to_aikit_arrays_and_back
def Imag(
    *,
    input,
    Tout=aikit.float32,
    name="Imag",
):
    Tout = aikit.as_aikit_dtype(Tout) if Tout is not None else aikit.float32
    return aikit.astype(aikit.imag(input), Tout)


@to_aikit_arrays_and_back
def Inv(*, x, name="Inv"):
    return aikit.astype(aikit.reciprocal(x), x.dtype)


@to_aikit_arrays_and_back
def InvGrad(*, y, dy, name="InvGrad"):
    return aikit.multiply(aikit.negative(dy), aikit.multiply(y, y))


@to_aikit_arrays_and_back
def Invert(*, x, name="Invert"):
    return aikit.bitwise_invert(x)


@to_aikit_arrays_and_back
def LeftShift(*, x, y, name="LeftShift"):
    return aikit.bitwise_left_shift(x, y)


@to_aikit_arrays_and_back
def Less(*, x, y, name="Less"):
    x, y = check_tensorflow_casting(x, y)
    return aikit.less(x, y)


@to_aikit_arrays_and_back
def LinSpace(*, start, stop, num, name=None):
    return aikit.linspace(start, stop, num)


@to_aikit_arrays_and_back
def Log(*, x, name="Log"):
    return aikit.log(x)


@to_aikit_arrays_and_back
def LogicalNot(*, x, name="LogicalNot"):
    return aikit.logical_not(x)


@to_aikit_arrays_and_back
def MatMul(*, a, b, transpose_a=False, transpose_b=False, name="MatMul"):
    a, b = check_tensorflow_casting(a, b)
    return aikit.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)


@to_aikit_arrays_and_back
def MatrixInverse(*, input, adjoint=False, name="MatrixInverse"):
    return aikit.inv(input, adjoint=adjoint)


@to_aikit_arrays_and_back
def Minimum(*, x, y, name="Minimum"):
    return aikit.minimum(x, y)


@to_aikit_arrays_and_back
def NotEqual(*, x, y, incompatible_shape_error=True, name="NotEqual"):
    x, y = check_tensorflow_casting(x, y)
    if incompatible_shape_error:
        return aikit.not_equal(x, y)

    try:
        return aikit.not_equal(x, y)
    except (aikit.utils.exceptions.AikitError, aikit.utils.exceptions.AikitBackendException):
        return aikit.array(True)


@to_aikit_arrays_and_back
def NthElement(*, input, n, reverse=False, name="NthElement"):
    return aikit.astype(aikit.sort(input, descending=reverse)[..., n], input.dtype)


@to_aikit_arrays_and_back
def OnesLike(*, x, name="OnesLike"):
    return aikit.ones_like(x)


@to_aikit_arrays_and_back
def Pack(*, values, axis=0, name="Pack"):
    return aikit.stack(values, axis=axis)


@to_aikit_arrays_and_back
def Pad(*, input, paddings, name="Pad"):
    return aikit.constant_pad(input, paddings.to_list())


@to_aikit_arrays_and_back
def PadV2(*, input, paddings, constant_values, name="PadV2"):
    return aikit.constant_pad(input, paddings.to_list(), value=constant_values)


@to_aikit_arrays_and_back
def Prod(*, input, axis, keep_dims=False, name="Prod"):
    return aikit.astype(aikit.prod(input, axis=axis, keepdims=keep_dims), input.dtype)


@to_aikit_arrays_and_back
def Real(input, Tout=aikit.float32, name="Real"):
    # TODO
    raise AikitNotImplementedException


@to_aikit_arrays_and_back
def Reverse(*, tensor, dims, name="Reverse"):
    ret = tensor
    for dim in enumerate(dims):
        if dim[1]:
            ret = aikit.flip(ret, axis=dim[0])
    return ret


@to_aikit_arrays_and_back
def RightShift(*, x, y, name="RightShift"):
    return aikit.bitwise_right_shift(x, y)


@to_aikit_arrays_and_back
def Round(*, x, name="Round"):
    return aikit.round(x)


@to_aikit_arrays_and_back
def Rsqrt(*, x, name="Rsqrt"):
    return aikit.sqrt(aikit.reciprocal(x))


@to_aikit_arrays_and_back
def Shape(*, input, output_type=aikit.int32, name="Shape"):
    output_type = to_aikit_dtype(output_type)
    return aikit.astype(aikit.shape(input, as_array=True), output_type, copy=False)


@with_unsupported_dtypes(
    {"2.15.0 and below": ("unsigned",)},
    "tensorflow",
)
@to_aikit_arrays_and_back
def Sign(*, x, name="Sign"):
    return aikit.sign(x, np_variant=False)


@to_aikit_arrays_and_back
def Sinh(*, x, name="Sinh"):
    return aikit.sinh(x)


@to_aikit_arrays_and_back
def Softplus(*, features, name="Softplus"):
    return aikit.softplus(features)


# Softsign
@to_aikit_arrays_and_back
def Softsign(*, features, name="Softsign"):
    return aikit.softsign(features)


@to_aikit_arrays_and_back
def SplitV(*, value, size_splits, axis, num_split, name="SplitV"):
    return aikit.split(value, num_or_size_splits=size_splits, axis=axis)


@to_aikit_arrays_and_back
def Sqrt(*, x, name="Sqrt"):
    return aikit.sqrt(x)


@to_aikit_arrays_and_back
def Square(*, x, name="Square"):
    return aikit.square(x)


@to_aikit_arrays_and_back
def Sum(*, input, axis, keep_dims=False, name="Sum"):
    return aikit.astype(aikit.sum(input, axis=axis, keepdims=keep_dims), input.dtype)


@with_supported_dtypes(
    {"2.15.0 and below": ("float64", "float128", "halfcomplex64", "complex128")},
    "tensorflow",
)
@to_aikit_arrays_and_back
def Svd(*, input, full_matrices=False, compute_uv=True, name=None):
    return aikit.svd(input, compute_uv=compute_uv, full_matrices=full_matrices)


@to_aikit_arrays_and_back
def TanhGrad(*, y, dy, name="TanhGrad"):
    return aikit.multiply(dy, aikit.subtract(1, aikit.multiply(y, y)))


@to_aikit_arrays_and_back
def Transpose(*, x, perm, name="Transpose"):
    ret = aikit.permute_dims(x, axes=perm)
    return ret


@to_aikit_arrays_and_back
def TruncateDiv(*, x, y, name="TruncateDiv"):
    return aikit.astype(aikit.trunc_divide(x, y), x.dtype)


@with_unsupported_dtypes({"2.15.0 and below": ("float16", "bfloat16")}, "tensorflow")
@to_aikit_arrays_and_back
def Unpack(*, value, num, axis=0, name="Unpack"):
    return aikit.unstack(value, axis=axis)[:num]


@with_supported_dtypes(
    {
        "2.15.0 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    },
    "tensorflow",
)
@to_aikit_arrays_and_back
def UnsortedSegmentProd(*, data, segment_ids, num_segments, name=None):
    data = aikit.array(data)
    segment_ids = aikit.array(segment_ids)

    aikit.utils.assertions.check_equal(
        list(segment_ids.shape), [list(data.shape)[0]], as_array=False
    )
    aikit.utils.assertions.check_greater(int(num_segments), int(aikit.max(segment_ids)))

    shape = list(aikit.shape(data))
    shape[0] = int(num_segments)
    x = aikit.ones(shape, dtype=data.dtype)
    for i in range((segment_ids).shape[0]):
        x[segment_ids[i]] = aikit.multiply(x[segment_ids[i]], data[i])
    return x


@to_aikit_arrays_and_back
def Xdaikit(*, x, y, name="Xdaikit"):
    if (x == 0).all():
        return 0.0
    return aikit.divide(x, y)


@with_unsupported_dtypes({"2.15.0 and below": ("bfloat16",)}, "tensorflow")
@to_aikit_arrays_and_back
def Xlog1py(*, x, y, name="Xlog1py"):
    if (x == 0).all():
        return 0.0
    return aikit.multiply(x, aikit.log1p(y))


@to_aikit_arrays_and_back
def ZerosLike(*, x, name="ZerosLike"):
    return aikit.zeros_like(x)
