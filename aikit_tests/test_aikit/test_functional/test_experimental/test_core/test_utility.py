# global
from hypothesis import strategies as st

# local
import aikit_tests.test_aikit.helpers as helpers
from aikit_tests.test_aikit.helpers import handle_test


@handle_test(
    fn_tree="functional.aikit.experimental.optional_get_element",
    dtype_and_x=helpers.dtype_and_values(),
    input_tensor=st.booleans(),
)
def test_optional_get_element(
    *,
    dtype_and_x,
    input_tensor,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    fn_input = x[0] if input_tensor else x

    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=fn_input,
    )
