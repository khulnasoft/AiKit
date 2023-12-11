# global
import numpy as np

# local
import aikit.functional.frontends.numpy as np_frontend
import aikit_tests.test_aikit.helpers as helpers
from aikit_tests.test_aikit.helpers import handle_frontend_test, BackendHandler


@handle_frontend_test(
    fn_tree="numpy.asmatrix",
    arr=helpers.dtype_and_values(min_num_dims=2, max_num_dims=2),
)
def test_numpy_asmatrix(arr, backend_fw):
    with BackendHandler.update_backend(backend_fw) as aikit_backend:
        dtype, x = arr
        ret = np_frontend.asmatrix(x[0])
        ret_gt = np.asmatrix(x[0])
        assert ret.shape == ret_gt.shape
        assert aikit_backend.all(aikit_backend.flatten(ret._data) == np.ravel(ret_gt))


@handle_frontend_test(
    fn_tree="numpy.asscalar",
    arr=helpers.array_values(dtype=helpers.get_dtypes("numeric"), shape=1),
)
def test_numpy_asscalar(arr: np.ndarray):
    ret_1 = arr.item()
    ret_2 = np_frontend.asscalar(arr)
    assert ret_1 == ret_2