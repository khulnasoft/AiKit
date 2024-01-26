# global
from hypothesis import given, strategies as st

# local
import aikit
import aikit_tests.test_aikit.helpers as helpers
from aikit.functional.frontends.torch.func_wrapper import (
    inputs_to_aikit_arrays,
    outputs_to_frontend_arrays,
    to_aikit_arrays_and_back,
    numpy_to_torch_style_args,
)
from aikit.functional.frontends.torch.tensor import Tensor
import aikit.functional.frontends.torch as torch_frontend


# --- Helpers --- #
# --------------- #


def _fn(*args, dtype=None, check_default=False, inplace=False):
    if (
        check_default
        and all(not (aikit.is_array(i) or hasattr(i, "aikit_array")) for i in args)
        and not aikit.exists(dtype)
    ):
        aikit.utils.assertions.check_equal(
            aikit.default_float_dtype(),
            torch_frontend.get_default_dtype(),
            as_array=False,
        )
        aikit.utils.assertions.check_equal(
            aikit.default_int_dtype(), "int64", as_array=False
        )
    return args[0]


# --- Main --- #
# ------------ #


@numpy_to_torch_style_args
def mocked_func(dim=None, keepdim=None, input=None, other=None):
    return dim, keepdim, input, other


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0])
)
def test_torch_inputs_to_aikit_arrays(dtype_and_x, backend_fw):
    x_dtype, x = dtype_and_x

    aikit.set_backend(backend=backend_fw)

    # check for aikit array
    input_aikit = aikit.array(x[0], dtype=x_dtype[0])
    output = inputs_to_aikit_arrays(_fn)(input_aikit)
    assert isinstance(output, aikit.Array)
    assert input_aikit.dtype == output.dtype
    assert aikit.all(input_aikit == output)

    # check for native array
    input_native = aikit.native_array(input_aikit)
    output = inputs_to_aikit_arrays(_fn)(input_native)
    assert isinstance(output, aikit.Array)
    assert aikit.as_aikit_dtype(input_native.dtype) == str(output.dtype)
    assert aikit.all(input_native == output.data)

    # check for frontend array
    input_frontend = Tensor(x[0])
    input_frontend.aikit_array = input_aikit
    output = inputs_to_aikit_arrays(_fn)(input_frontend)
    assert isinstance(output, aikit.Array)
    assert str(input_frontend.dtype) == str(output.dtype)
    assert aikit.all(input_frontend.aikit_array == output)

    aikit.previous_backend()


@given(
    dim=st.integers(),
    keepdim=st.booleans(),
    input=st.lists(st.integers()),
    other=st.integers(),
)
def test_torch_numpy_to_torch_style_args(dim, keepdim, input, other):
    # PyTorch-style keyword arguments
    assert (dim, keepdim, input, other) == mocked_func(
        dim=dim, keepdim=keepdim, input=input, other=other
    )

    # NumPy-style keyword arguments
    assert (dim, keepdim, input, other) == mocked_func(
        axis=dim, keepdims=keepdim, x=input, x2=other
    )

    # Mixed-style keyword arguments
    assert (dim, keepdim, input, other) == mocked_func(
        axis=dim, keepdim=keepdim, input=input, x2=other
    )


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
    dtype=helpers.get_dtypes("valid", none=True, full=False, prune_function=False),
    generate_type=st.sampled_from(["frontend", "aikit", "native"]),
    inplace=st.booleans(),
)
def test_torch_outputs_to_frontend_arrays(
    dtype_and_x,
    dtype,
    generate_type,
    inplace,
    backend_fw,
):
    x_dtype, x = dtype_and_x

    aikit.set_backend(backend_fw)

    x = aikit.array(x[0], dtype=x_dtype[0])
    if generate_type == "frontend":
        x = Tensor(x)
    elif generate_type == "native":
        x = x.data

    if not len(x.shape):
        scalar_x = aikit.to_scalar(x.aikit_array if isinstance(x, Tensor) else x)
        outputs_to_frontend_arrays(_fn)(
            scalar_x, scalar_x, check_default=True, dtype=dtype
        )
        outputs_to_frontend_arrays(_fn)(scalar_x, x, check_default=True, dtype=dtype)
    output = outputs_to_frontend_arrays(_fn)(
        x, check_default=True, dtype=dtype, inplace=inplace
    )
    assert isinstance(output, Tensor)
    if inplace:
        if generate_type == "frontend":
            assert x is output
        elif generate_type == "native":
            assert x is output.aikit_array.data
        else:
            assert x is output.aikit_array
    else:
        assert aikit.as_aikit_dtype(x.dtype) == aikit.as_aikit_dtype(output.dtype)
        if generate_type == "frontend":
            assert aikit.all(x.aikit_array == output.aikit_array)
        elif generate_type == "native":
            assert aikit.all(x == output.aikit_array.data)
        else:
            assert aikit.all(x == output.aikit_array)

    assert aikit.default_float_dtype_stack == aikit.default_int_dtype_stack == []

    aikit.previous_backend()


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
    dtype=helpers.get_dtypes("valid", none=True, full=False, prune_function=False),
)
def test_torch_to_aikit_arrays_and_back(dtype_and_x, dtype, backend_fw):
    x_dtype, x = dtype_and_x

    aikit.set_backend(backend_fw)

    # check for aikit array
    input_aikit = aikit.array(x[0], dtype=x_dtype[0])
    if not len(input_aikit.shape):
        scalar_input_aikit = aikit.to_scalar(input_aikit)
        to_aikit_arrays_and_back(_fn)(
            scalar_input_aikit, scalar_input_aikit, check_default=True, dtype=dtype
        )
        to_aikit_arrays_and_back(_fn)(
            scalar_input_aikit, input_aikit, check_default=True, dtype=dtype
        )
    output = to_aikit_arrays_and_back(_fn)(input_aikit, check_default=True, dtype=dtype)
    assert isinstance(output, Tensor)
    assert str(input_aikit.dtype) == str(output.dtype)
    assert aikit.all(input_aikit == output.aikit_array)

    # check for native array
    input_native = aikit.native_array(input_aikit)
    if not len(input_native.shape):
        scalar_input_native = aikit.to_scalar(input_native)
        to_aikit_arrays_and_back(_fn)(
            scalar_input_native, scalar_input_native, check_default=True, dtype=dtype
        )
        to_aikit_arrays_and_back(_fn)(
            scalar_input_native, input_native, check_default=True, dtype=dtype
        )
    output = to_aikit_arrays_and_back(_fn)(input_native, check_default=True, dtype=dtype)
    assert isinstance(output, Tensor)
    assert aikit.as_aikit_dtype(input_native.dtype) == str(output.dtype)
    assert aikit.all(input_native == output.aikit_array.data)

    # check for frontend array
    input_frontend = Tensor(x[0])
    input_frontend.aikit_array = input_aikit
    if not len(input_frontend.shape):
        scalar_input_front = inputs_to_aikit_arrays(aikit.to_scalar)(input_frontend)
        to_aikit_arrays_and_back(_fn)(
            scalar_input_front, scalar_input_front, check_default=True, dtype=dtype
        )
        to_aikit_arrays_and_back(_fn)(
            scalar_input_front, input_frontend, check_default=True, dtype=dtype
        )
    output = to_aikit_arrays_and_back(_fn)(
        input_frontend, check_default=True, dtype=dtype
    )
    assert isinstance(output, Tensor)
    assert input_frontend.dtype == output.dtype
    assert aikit.all(input_frontend.aikit_array == output.aikit_array)

    assert aikit.default_float_dtype_stack == aikit.default_int_dtype_stack == []

    aikit.previous_backend()
