# global
from hypothesis import given, strategies as st

# local
import aikit
import aikit_tests.test_aikit.helpers as helpers
from aikit_tests.test_aikit.helpers import BackendHandler
from aikit.functional.frontends.tensorflow.func_wrapper import (
    outputs_to_frontend_arrays,
    to_aikit_arrays_and_back,
    handle_tf_dtype,
)
from aikit.functional.frontends.tensorflow.tensor import EagerTensor
import aikit.functional.frontends.tensorflow as tf_frontend
import aikit.functional.frontends.numpy as np_frontend


# --- Helpers --- #
# --------------- #


@st.composite
def _dtype_helper(draw):
    return draw(
        st.sampled_from(
            [
                draw(helpers.get_dtypes("valid", prune_function=False, full=False))[0],
                aikit.as_native_dtype(
                    draw(helpers.get_dtypes("valid", prune_function=False, full=False))[
                        0
                    ]
                ),
                draw(
                    st.sampled_from(list(tf_frontend.tensorflow_enum_to_type.values()))
                ),
                draw(st.sampled_from(list(tf_frontend.tensorflow_enum_to_type.keys()))),
                np_frontend.dtype(
                    draw(helpers.get_dtypes("valid", prune_function=False, full=False))[
                        0
                    ]
                ),
                draw(st.sampled_from(list(np_frontend.numpy_scalar_to_dtype.keys()))),
            ]
        )
    )


def _fn(x=None, dtype=None):
    if aikit.exists(dtype):
        return dtype
    return x


# --- Main --- #
# ------------ #


@given(
    dtype=_dtype_helper(),
)
def test_tensorflow_handle_tf_dtype(dtype):
    ret_dtype = handle_tf_dtype(_fn)(dtype=dtype)
    assert isinstance(ret_dtype, aikit.Dtype)


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
)
def test_tensorflow_inputs_to_aikit_arrays(dtype_and_x, backend_fw):
    x_dtype, x = dtype_and_x

    with BackendHandler.update_backend(backend_fw) as aikit_backend:
        _import_fn = aikit_backend.utils.dynamic_import.import_module
        _import_fn("aikit.functional.frontends.tensorflow.func_wrapper")
        _tensor_module = _import_fn("aikit.functional.frontends.tensorflow.tensor")

        # check for aikit array
        input_aikit = aikit_backend.array(x[0], dtype=x_dtype[0])
        output = aikit_backend.inputs_to_aikit_arrays(_fn)(input_aikit)
        assert isinstance(output, aikit_backend.Array)
        assert input_aikit.dtype == output.dtype
        assert aikit_backend.all(input_aikit == output)

        # check for native array
        input_native = aikit_backend.native_array(input_aikit)
        output = aikit_backend.inputs_to_aikit_arrays(_fn)(input_native)
        assert isinstance(output, aikit_backend.Array)
        assert aikit_backend.as_aikit_dtype(input_native.dtype) == output.dtype
        assert aikit_backend.all(input_native == output.data)

        # check for frontend array
        input_frontend = _tensor_module.EagerTensor(x[0])
        output = aikit_backend.inputs_to_aikit_arrays(_fn)(input_frontend)
        assert isinstance(output, aikit_backend.Array)
        assert input_frontend.dtype.aikit_dtype == output.dtype
        assert aikit_backend.all(input_frontend.aikit_array == output)


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
)
def test_tensorflow_outputs_to_frontend_arrays(dtype_and_x):
    x_dtype, x = dtype_and_x

    # check for aikit array
    input_aikit = aikit.array(x[0], dtype=x_dtype[0])
    output = outputs_to_frontend_arrays(_fn)(input_aikit)
    assert isinstance(output, EagerTensor)
    assert input_aikit.dtype == output.dtype.aikit_dtype
    assert aikit.all(input_aikit == output.aikit_array)


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
)
def test_tensorflow_to_aikit_arrays_and_back(dtype_and_x):
    x_dtype, x = dtype_and_x

    # check for aikit array
    input_aikit = aikit.array(x[0], dtype=x_dtype[0])
    output = to_aikit_arrays_and_back(_fn)(input_aikit)
    assert isinstance(output, EagerTensor)
    assert input_aikit.dtype == output.dtype.aikit_dtype
    assert aikit.all(input_aikit == output.aikit_array)

    # check for native array
    input_native = aikit.native_array(input_aikit)
    output = to_aikit_arrays_and_back(_fn)(input_native)
    assert isinstance(output, EagerTensor)
    assert aikit.as_aikit_dtype(input_native.dtype) == output.dtype.aikit_dtype
    assert aikit.all(input_native == output.aikit_array.data)

    # check for frontend array
    input_frontend = EagerTensor(x[0])
    output = to_aikit_arrays_and_back(_fn)(input_frontend)
    assert isinstance(output, EagerTensor)
    assert input_frontend.dtype == output.dtype
    assert aikit.all(input_frontend.aikit_array == output.aikit_array)
