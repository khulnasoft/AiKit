from aikit.utils.exceptions import AikitNotImplementedException


def is_native_sparse_array(x):
    raise AikitNotImplementedException()


def native_sparse_array(
    data=None,
    *,
    coo_indices=None,
    crow_indices=None,
    col_indices=None,
    ccol_indices=None,
    row_indices=None,
    values=None,
    dense_shape=None,
    format="coo"
):
    raise AikitNotImplementedException()


def native_sparse_array_to_indices_values_and_shape(x):
    raise NotImplementedError(
        "mxnet.native_sparse_array_to_indices_values_and_shape Not Implemented"
    )
