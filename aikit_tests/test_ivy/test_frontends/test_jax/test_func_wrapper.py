# global
from hypothesis import given

# local
import aikit
import aikit_tests.test_aikit.helpers as helpers
from aikit.functional.frontends.jax.func_wrapper import (
    inputs_to_aikit_arrays,
    outputs_to_frontend_arrays,
    to_aikit_arrays_and_back,
)
from aikit.functional.frontends.jax.array import Array
import aikit.functional.frontends.jax as jax_frontend


# --- Helpers --- #
# --------------- #


def _fn(x, check_default=False):
    if check_default and jax_frontend.config.jax_enable_x64:
        aikit.utils.assertions.check_equal(
            aikit.default_float_dtype(), "float64", as_array=False
        )
        aikit.utils.assertions.check_equal(
            aikit.default_int_dtype(), "int64", as_array=False
        )
    return x


# --- Main --- #
# ------------ #


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
)
def test_jax_inputs_to_aikit_arrays(dtype_and_x, backend_fw):
    aikit.set_backend(backend_fw)
    x_dtype, x = dtype_and_x

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
    assert aikit.as_aikit_dtype(input_native.dtype) == output.dtype
    assert aikit.all(aikit.equal(input_native, output.data))

    # check for frontend array
    input_frontend = Array(x[0])
    output = inputs_to_aikit_arrays(_fn)(input_frontend)
    assert isinstance(output, aikit.Array)
    assert input_frontend.dtype == output.dtype
    assert aikit.all(input_frontend.aikit_array == output)
    aikit.previous_backend()


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
)
def test_jax_outputs_to_frontend_arrays(dtype_and_x, backend_fw):
    aikit.set_backend(backend_fw)
    x_dtype, x = dtype_and_x

    # check for aikit array
    input_aikit = aikit.array(x[0], dtype=x_dtype[0])
    output = outputs_to_frontend_arrays(_fn)(input_aikit, check_default=True)
    assert isinstance(output, Array)
    assert input_aikit.dtype == output.dtype
    assert aikit.all(input_aikit == output.aikit_array)

    assert aikit.default_float_dtype_stack == aikit.default_int_dtype_stack == []
    aikit.previous_backend()


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
)
def test_jax_to_aikit_arrays_and_back(dtype_and_x, backend_fw):
    aikit.set_backend(backend_fw)
    x_dtype, x = dtype_and_x

    # check for aikit array
    input_aikit = aikit.array(x[0], dtype=x_dtype[0])
    output = to_aikit_arrays_and_back(_fn)(input_aikit, check_default=True)
    assert isinstance(output, Array)
    assert input_aikit.dtype == output.dtype
    assert aikit.all(input_aikit == output.aikit_array)

    # check for native array
    input_native = aikit.native_array(input_aikit)
    output = to_aikit_arrays_and_back(_fn)(input_native, check_default=True)
    assert isinstance(output, Array)
    assert aikit.as_aikit_dtype(input_native.dtype) == output.dtype
    assert aikit.all(aikit.equal(input_native, output.aikit_array.data))

    # check for frontend array
    input_frontend = Array(x[0])
    output = to_aikit_arrays_and_back(_fn)(input_frontend, check_default=True)
    assert isinstance(output, Array)
    assert str(input_frontend.dtype) == str(output.dtype)
    assert aikit.all(input_frontend.aikit_array == output.aikit_array)

    assert aikit.default_float_dtype_stack == aikit.default_int_dtype_stack == []
    aikit.previous_backend()
