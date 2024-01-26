"""Collection of tests for Aikit sequential."""

# global
import itertools

from hypothesis import strategies as st

# local
import aikit
from aikit_tests.test_aikit import helpers
from aikit_tests.test_aikit.helpers.testing_helpers import handle_method


class TrainableModule(aikit.Module):
    def __init__(self, in_size, hidden_size, out_size):
        self._linear0 = aikit.Linear(in_size, hidden_size)
        self._linear1 = aikit.Linear(hidden_size, out_size)
        aikit.Module.__init__(self)

    def _forward(self, x):
        x = self._linear0(x)
        return self._linear1(x)


# --- Helpers --- #
# --------------- #


def _copy_weights(v1, v2):
    # copy weights from layer1 to layer2
    v2.w = aikit.copy_array(v1.w)
    v2.b = aikit.copy_array(v1.b)


# Helpers #
###########
def _train(module, input_arr):
    def loss_fn(_v):
        return aikit.abs(aikit.mean(input_arr) - aikit.mean(module(input_arr, v=_v)))

    # initial loss
    loss_tm1, grads = aikit.execute_with_gradients(loss_fn, module.v)
    loss = None
    losses = []
    for i in range(5):
        loss, grads = aikit.execute_with_gradients(loss_fn, module.v)
        module.v = aikit.gradient_descent_update(module.v, grads, 1e-5)
        losses.append(loss)

    # loss is lower or very close to initial loss
    assert loss <= loss_tm1 or aikit.abs(loss - loss_tm1) < 1e-5

    return losses


# --- Main --- #
# ------------ #


@handle_method(
    method_tree="Sequential.__call__",
    input_array=st.lists(
        helpers.floats(
            min_value=-1,
            max_value=1,
            allow_nan=False,
            allow_inf=False,
            small_abs_safety_factor=1.5,
            safety_factor_scale="log",
        ),
        min_size=1,
        max_size=5,
    ),
    dims=st.lists(st.integers(1, 10), min_size=1, max_size=5),
    use_activation=st.booleans(),
)
def test_sequential_construction_and_value(
    input_array, dims, use_activation, on_device, backend_fw
):
    with aikit.utils.backend.ContextManager(backend_fw):
        dims = [len(input_array)] + dims
        layer_count = len(dims)
        layers = [
            aikit.Linear(dims[i], dims[i + 1], device=on_device)
            for i in range(layer_count - 1)
        ]

        if use_activation:
            activations = [aikit.GELU() for _ in range(layer_count - 1)]
            layers = itertools.chain.from_iterable(zip(layers, activations))

        module = aikit.Sequential(*layers)

        input_array = aikit.array(input_array, dtype="float32", device=on_device)

        if backend_fw != "numpy":
            _train(module, input_array)


@handle_method(
    method_tree="Sequential.__call__",
    input_array=st.lists(
        helpers.floats(
            min_value=0,
            max_value=1,
            allow_nan=False,
            allow_inf=False,
            small_abs_safety_factor=1.5,
            safety_factor_scale="log",
        ),
        min_size=1,
        max_size=5,
    ),
    dims=st.lists(st.integers(1, 10), min_size=2, max_size=2),
)
def test_sequential_same_as_class(input_array, dims, backend_fw):
    with aikit.utils.backend.ContextManager(backend_fw):
        dims = [len(input_array)] + dims
        layer_count = len(dims)
        layers = [aikit.Linear(dims[i], dims[i + 1]) for i in range(layer_count - 1)]

        m_sequential = aikit.Sequential(*layers)
        m_class = TrainableModule(dims[0], dims[1], dims[2])

        # copy weights
        _copy_weights(m_class.v.linear0, m_sequential.v.submodules.v0)
        _copy_weights(m_class.v.linear1, m_sequential.v.submodules.v1)

        input_array = aikit.array(input_array, dtype="float32")

        if backend_fw != "numpy":
            sequential_loss = _train(m_sequential, input_array)
            class_loss = _train(m_class, input_array)
            assert sequential_loss == class_loss
