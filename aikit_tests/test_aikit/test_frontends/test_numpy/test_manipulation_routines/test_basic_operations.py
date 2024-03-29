# global
from hypothesis import strategies as st
import numpy as np

# local
import aikit_tests.test_aikit.helpers as helpers
import aikit_tests.test_aikit.test_frontends.test_numpy.helpers as np_frontend_helpers
from aikit_tests.test_aikit.helpers import handle_frontend_test, BackendHandler


# --- Helpers --- #
# --------------- #


@st.composite
def generate_copyto_args(draw):
    input_dtypes, xs, casting, _ = draw(
        np_frontend_helpers.dtypes_values_casting_dtype(
            arr_func=[
                lambda: helpers.dtype_and_values(
                    available_dtypes=helpers.get_dtypes("valid"),
                    num_arrays=2,
                    shared_dtype=True,
                    min_num_dims=1,
                )
            ],
        )
    )
    where = draw(np_frontend_helpers.where(shape=xs[0].shape))
    return input_dtypes, xs, casting, where


# copyto
@handle_frontend_test(
    fn_tree="numpy.copyto",
    test_with_out=st.just(False),
    copyto_args=generate_copyto_args(),
)
def test_numpy_copyto(
    copyto_args,
    backend_fw,
    frontend,
):
    _, xs, casting, where = copyto_args
    if isinstance(where, (list, tuple)):
        where = where[0]

    with BackendHandler.update_backend(backend_fw) as aikit_backend:
        src_aikit = aikit_backend.functional.frontends.numpy.array(xs[0])
        dst_aikit = aikit_backend.functional.frontends.numpy.array(xs[1])
        aikit_backend.functional.frontends.numpy.copyto(
            dst_aikit, src_aikit, where=where, casting=casting
        )

        src_np = np.array(xs[0])
        dst_np = np.array(xs[1])
        np.copyto(dst_np, src_np, where=where, casting=casting)

        assert dst_np.shape == dst_aikit.shape
        # value test
        dst_ = aikit_backend.to_numpy(dst_aikit.aikit_array)
        helpers.assert_all_close(
            dst_, dst_np, backend=backend_fw, ground_truth_backend=frontend
        )
        assert id(src_aikit) != id(dst_aikit)


# shape
@handle_frontend_test(
    fn_tree="numpy.shape",
    xs_n_input_dtypes_n_unique_idx=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    test_with_out=st.just(False),
)
def test_numpy_shape(
    *,
    xs_n_input_dtypes_n_unique_idx,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtypes, xs = xs_n_input_dtypes_n_unique_idx
    ret, ret_gt = helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        array=xs[0],
    )
    # Manually compare the shape here because aikit.shape doesn't return an array, so
    # aikit.to_numpy will narrow the bit-width, resulting in different dtypes. This is
    # not an issue with the front-end function, but how the testing framework converts
    # non-array function outputs to arrays.
    assert len(ret) == len(ret_gt)
    for i, j in zip(ret, ret_gt):
        assert i == j
