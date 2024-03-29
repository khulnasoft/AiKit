# global
import aikit
from hypothesis import given
import pytest

# local
import aikit_tests.test_aikit.helpers as helpers
from aikit.functional.frontends.onnx import Tensor


@pytest.mark.skip("Testing pipeline not yet implemented")
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_onnx_tensor_property_device(
    dtype_x,
):
    _, data = dtype_x
    x = Tensor(data[0])
    x.aikit_array = data[0]
    aikit.utils.assertions.check_equal(
        x.device, aikit.dev(aikit.array(data[0])), as_array=False
    )


@pytest.mark.skip("Testing pipeline not yet implemented")
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_onnx_tensor_property_dtype(
    dtype_x,
):
    dtype, data = dtype_x
    x = Tensor(data[0])
    x.aikit_array = data[0]
    aikit.utils.assertions.check_equal(x.dtype, dtype[0], as_array=False)


@pytest.mark.skip("Testing pipeline not yet implemented")
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_onnx_tensor_property_aikit_array(
    dtype_x,
):
    _, data = dtype_x
    x = Tensor(data[0])
    x.aikit_array = data[0]
    ret = helpers.flatten_and_to_np(ret=x.aikit_array.data, backend="torch")
    ret_gt = helpers.flatten_and_to_np(ret=data[0], backend="torch")
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        backend="torch",
    )


@pytest.mark.skip("Testing pipeline not yet implemented")
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_onnx_tensor_property_ndim(
    dtype_x,
):
    dtype, data, shape = dtype_x
    x = Tensor(data[0])
    aikit.utils.assertions.check_equal(x.ndim, data[0].ndim, as_array=False)


@pytest.mark.skip("Testing pipeline not yet implemented")
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_onnx_tensor_property_shape(dtype_x):
    dtype, data, shape = dtype_x
    x = Tensor(data[0])
    aikit.utils.assertions.check_equal(
        x.aikit_array.shape, aikit.Shape(shape), as_array=False
    )
