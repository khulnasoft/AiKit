# global
from hypothesis import given, strategies as st
import platform

# local
import aikit
import aikit_tests.test_aikit.helpers as helpers
from aikit.functional.frontends.numpy.func_wrapper import (
    inputs_to_aikit_arrays,
    outputs_to_frontend_arrays,
    to_aikit_arrays_and_back,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
)
from aikit.functional.frontends.numpy.ndarray import ndarray
import aikit.functional.frontends.numpy as np_frontend


# --- Helpers --- #
# --------------- #


@st.composite
def _dtype_helper(draw):
    return draw(
        st.sampled_from(
            [
                draw(st.sampled_from([int, float, bool])),
                aikit.as_native_dtype(
                    draw(helpers.get_dtypes("valid", full=False, prune_function=False))[
                        0
                    ]
                ),
                np_frontend.dtype(
                    draw(helpers.get_dtypes("valid", full=False, prune_function=False))[
                        0
                    ]
                ),
                draw(st.sampled_from(list(np_frontend.numpy_scalar_to_dtype.keys()))),
                draw(st.sampled_from(list(np_frontend.numpy_str_to_type_table.keys()))),
            ]
        )
    )


def _fn(*args, check_default=False, dtype=None):
    if (
        check_default
        and any(not (aikit.is_array(i) or hasattr(i, "aikit_array")) for i in args)
        and not aikit.exists(dtype)
    ):
        aikit.utils.assertions.check_equal(
            aikit.default_float_dtype(), "float64", as_array=False
        )
        if platform.system() != "Windows":
            aikit.utils.assertions.check_equal(
                aikit.default_int_dtype(), "int64", as_array=False
            )
        else:
            aikit.utils.assertions.check_equal(
                aikit.default_int_dtype(), "int32", as_array=False
            )
    if not aikit.exists(args[0]):
        return dtype
    return args[0]


def _zero_dim_to_scalar_checks(x, ret_x):
    if len(x.shape) > 0:
        assert aikit.all(aikit.array(ret_x) == aikit.array(x))
    else:
        assert issubclass(type(ret_x), np_frontend.generic)
        assert ret_x.aikit_array == aikit.array(x)


@st.composite
def _zero_dim_to_scalar_helper(draw):
    dtype = draw(
        helpers.get_dtypes("valid", prune_function=False, full=False).filter(
            lambda x: "bfloat16" not in x
        )
    )[0]
    shape = draw(helpers.get_shape())
    return draw(
        st.one_of(
            helpers.array_values(shape=shape, dtype=dtype),
            st.lists(helpers.array_values(shape=shape, dtype=dtype), min_size=1).map(
                tuple
            ),
        )
    )


# --- Main --- #
# ------------ #


@given(
    dtype=_dtype_helper(),
)
def test_handle_numpy_dtype(dtype, backend_fw):
    aikit.set_backend(backend_fw)
    ret_dtype = handle_numpy_dtype(_fn)(None, dtype=dtype)
    assert isinstance(ret_dtype, aikit.Dtype)
    aikit.previous_backend()


@given(x=_zero_dim_to_scalar_helper())
def test_numpy_from_zero_dim_arrays_to_scalar(x, backend_fw):
    aikit.set_backend(backend_fw)
    ret_x = from_zero_dim_arrays_to_scalar(_fn)(x)
    if isinstance(x, tuple):
        assert isinstance(ret_x, tuple)
        for x_i, ret_x_i in zip(x, ret_x):
            _zero_dim_to_scalar_checks(x_i, ret_x_i)
    else:
        _zero_dim_to_scalar_checks(x, ret_x)
    aikit.previous_backend()


@given(
    dtype_x_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ),
)
def test_numpy_inputs_to_aikit_arrays(dtype_x_shape, backend_fw):
    aikit.set_backend(backend_fw)
    x_dtype, x, shape = dtype_x_shape

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
    input_frontend = ndarray(shape)
    input_frontend.aikit_array = input_aikit
    output = inputs_to_aikit_arrays(_fn)(input_frontend)
    assert isinstance(output, aikit.Array)
    assert input_frontend.aikit_array.dtype == str(output.dtype)
    assert aikit.all(input_frontend.aikit_array == output)
    aikit.previous_backend()


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
    dtype=helpers.get_dtypes("valid", none=True, full=False, prune_function=False),
)
def test_numpy_outputs_to_frontend_arrays(dtype_and_x, dtype, backend_fw):
    aikit.set_backend(backend_fw)
    x_dtype, x = dtype_and_x

    # check for aikit array
    input_aikit = aikit.array(x[0], dtype=x_dtype[0])
    if not len(input_aikit.shape):
        scalar_input_aikit = aikit.to_scalar(input_aikit)
        outputs_to_frontend_arrays(_fn)(
            scalar_input_aikit, scalar_input_aikit, check_default=True, dtype=dtype
        )
        outputs_to_frontend_arrays(_fn)(
            scalar_input_aikit, input_aikit, check_default=True, dtype=dtype
        )
    output = outputs_to_frontend_arrays(_fn)(input_aikit, check_default=True, dtype=dtype)
    assert isinstance(output, ndarray)
    assert input_aikit.dtype == output.aikit_array.dtype
    assert aikit.all(input_aikit == output.aikit_array)

    assert aikit.default_float_dtype_stack == aikit.default_int_dtype_stack == []
    aikit.previous_backend()


@given(
    dtype_x_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ),
    dtype=helpers.get_dtypes("valid", none=True, full=False, prune_function=False),
)
def test_numpy_to_aikit_arrays_and_back(dtype_x_shape, dtype, backend_fw):
    aikit.set_backend(backend_fw)
    x_dtype, x, shape = dtype_x_shape

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
    assert isinstance(output, ndarray)
    assert input_aikit.dtype == output.aikit_array.dtype
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
    assert isinstance(output, ndarray)
    assert aikit.as_aikit_dtype(input_native.dtype) == output.aikit_array.dtype
    assert aikit.all(input_native == output.aikit_array.data)

    # check for frontend array
    input_frontend = ndarray(shape)
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
    assert isinstance(output, ndarray)
    assert input_frontend.aikit_array.dtype == output.aikit_array.dtype
    assert aikit.all(input_frontend.aikit_array == output.aikit_array)

    assert aikit.default_float_dtype_stack == aikit.default_int_dtype_stack == []
    aikit.previous_backend()
