from aikit.functional.frontends.sklearn.tree import DecisionTreeClassifier as aikit_DTC
import aikit
from hypothesis import given
import aikit_tests.test_aikit.helpers as helpers


# --- Helpers --- #
# --------------- #


# helper functions
def _get_sklearn_predict(X, y, max_depth, DecisionTreeClassifier):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    clf.fit(X, y)
    return clf.predict


# --- Main --- #
# ------------ #


# todo: integrate with already existing strats and generalize
@given(
    X=helpers.array_values(
        shape=(5, 2),
        dtype=helpers.get_dtypes("float", prune_function=False),
        safety_factor_scale="log",
    ),
    y=helpers.array_values(
        shape=(5,),
        dtype=helpers.get_dtypes("signed_integer", prune_function=False),
        safety_factor_scale="log",
    ),
    max_depth=helpers.ints(max_value=5, min_value=1),
)
def test_sklearn_tree_predict(X, y, max_depth):
    try:
        from sklearn.tree import DecisionTreeClassifier as sklearn_DTC
    except ImportError:
        print("sklearn not installed, skipping test_sklearn_tree_predict")
        return
    sklearn_pred = _get_sklearn_predict(X, y, max_depth, sklearn_DTC)(X)
    aikit_pred = _get_sklearn_predict(aikit.array(X), aikit.array(y), max_depth, aikit_DTC)(X)
    helpers.assert_same_type_and_shape([sklearn_pred, aikit_pred])
