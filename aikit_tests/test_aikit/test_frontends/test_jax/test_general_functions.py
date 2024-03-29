# local
import aikit
import aikit_tests.test_aikit.helpers as helpers
from aikit_tests.test_aikit.helpers import handle_frontend_test, BackendHandler
from aikit.functional.frontends.jax import vmap
from hypothesis import strategies as st
import jax


# --- Helpers --- #
# --------------- #


def _fn1(x, y):
    return aikit.matmul(x, y)


def _fn2(x, y):
    return aikit.vecdot(x, y)


def _fn3(x, y):
    return aikit.add(x, y)


# --- Main --- #
# ------------ #


# device_get
@handle_frontend_test(
    fn_tree="jax.general_functions.device_get",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_jax_device_get(
    *,
    dtype_and_x,
    test_flags,
    fn_tree,
    frontend,
    backend_fw,
    on_device,
):
    with BackendHandler.update_backend(backend_fw) as aikit_backend:
        dtype, x = dtype_and_x
        dtype = dtype[0]
        x = x[0]

        x = aikit_backend.asarray(x)
        if test_flags.as_variable and aikit_backend.is_float_dtype(dtype):
            x = aikit_backend.functional.aikit.gradients._variable(x)

        x_on_dev = aikit_backend.functional.frontends.jax.device_get(x).aikit_array
        dev_from_new_x = aikit_backend.dev(x_on_dev)

        # value test
        assert dev_from_new_x == "cpu"


# device_put
@handle_frontend_test(
    fn_tree="jax.general_functions.device_put",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_jax_device_put(
    *,
    dtype_and_x,
    test_flags,
    fn_tree,
    frontend,
    backend_fw,
    on_device,
):
    with BackendHandler.update_backend(backend_fw) as aikit_backend:
        dtype, x = dtype_and_x
        dtype = dtype[0]
        x = x[0]

        x = aikit_backend.asarray(x)
        if test_flags.as_variable and aikit_backend.is_float_dtype(dtype):
            x = aikit_backend.functional.aikit.gradients._variable(x)

        device = aikit_backend.dev(x)
        x_on_dev = aikit_backend.functional.frontends.jax.device_put(
            x, on_device
        ).aikit_array
        dev_from_new_x = aikit_backend.dev(x_on_dev)

        # value test
        assert dev_from_new_x == device


# vmap
@handle_frontend_test(
    fn_tree="jax.general_functions.vmap",
    func=st.sampled_from([_fn1, _fn2, _fn3]),
    dtype_and_arrays_and_axes=helpers.arrays_and_axes(
        allow_none=False,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=10,
        num=2,
        return_dtype=True,
    ),
    in_axes_as_cont=st.booleans(),
)
def test_jax_vmap(
    func,
    dtype_and_arrays_and_axes,
    in_axes_as_cont,
    backend_fw,
):
    dtype, generated_arrays, in_axes = dtype_and_arrays_and_axes
    aikit.set_backend(backend_fw)
    arrays = [aikit.native_array(array) for array in generated_arrays]
    if in_axes_as_cont:
        vmapped_func = vmap(func, in_axes=in_axes, out_axes=0)
    else:
        vmapped_func = vmap(func, in_axes=0, out_axes=0)

    assert callable(vmapped_func)

    try:
        fw_res = helpers.flatten_and_to_np(
            ret=vmapped_func(*arrays), backend=backend_fw
        )
        fw_res = fw_res if len(fw_res) else None
    except Exception:
        fw_res = None

    aikit.previous_backend()

    aikit.set_backend("jax")
    arrays = [aikit.native_array(array) for array in generated_arrays]
    if in_axes_as_cont:
        jax_vmapped_func = jax.vmap(func, in_axes=in_axes, out_axes=0)
    else:
        jax_vmapped_func = jax.vmap(func, in_axes=0, out_axes=0)

    assert callable(jax_vmapped_func)

    try:
        jax_res = helpers.flatten_and_to_np(
            ret=jax_vmapped_func(*arrays), backend="jax"
        )
        jax_res = jax_res if len(jax_res) else None
    except Exception:
        jax_res = None

    aikit.previous_backend()

    if fw_res is not None and jax_res is not None:
        helpers.value_test(
            ret_np_flat=fw_res,
            ret_np_from_gt_flat=jax_res,
            rtol=1e-1,
            atol=1e-1,
            backend=backend_fw,
            ground_truth_backend="jax",
        )

    elif fw_res is None and jax_res is None:
        pass
    else:
        assert False, "One of the results is None while other isn't"
