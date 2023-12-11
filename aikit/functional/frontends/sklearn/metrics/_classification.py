import aikit
from aikit.functional.frontends.numpy.func_wrapper import to_aikit_arrays_and_back
from sklearn.utils.multiclass import type_of_target


@to_aikit_arrays_and_back
def accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None):
    # TODO: implement sample_weight
    y_type = type_of_target(y_true)
    if y_type.startswith("multilabel"):
        diff_labels = aikit.count_nonzero(y_true - y_pred, axis=1)
        ret = aikit.equal(diff_labels, 0).astype("int64")
    else:
        ret = aikit.equal(y_true, y_pred).astype("int64")
    ret = ret.sum().astype("int64")
    if normalize:
        ret = ret / y_true.shape[0]
        ret = ret.astype("float64")
    return ret
