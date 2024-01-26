"""Collection of tests for the demos."""

# global
import pytest

# local
import aikit
import aikit.functional.backends.numpy


# functional api
def test_array(on_device):
    import jax.numpy as jnp

    assert aikit.concat((jnp.ones((1,)), jnp.ones((1,))), axis=-1).shape == (2,)
    import tensorflow as tf

    assert aikit.concat((tf.ones((1,)), tf.ones((1,))), axis=-1).shape == (2,)
    import numpy as np

    assert aikit.concat((np.ones((1,)), np.ones((1,))), axis=-1).shape == (2,)
    import torch

    assert aikit.concat((torch.ones((1,)), torch.ones((1,))), axis=-1).shape == (2,)
    import paddle

    assert aikit.concat((paddle.ones((1,)), paddle.ones((1,))), axis=-1).shape == (2,)


# Tests #
# ------#


# training
def test_training_demo(on_device, backend_fw):
    if backend_fw == "numpy":
        # numpy does not support gradients
        pytest.skip()

    aikit.set_backend(backend_fw)

    class MyModel(aikit.Module):
        def __init__(self):
            self.linear0 = aikit.Linear(3, 64)
            self.linear1 = aikit.Linear(64, 1)
            aikit.Module.__init__(self)

        def _forward(self, x):
            x = aikit.relu(self.linear0(x))
            return aikit.sigmoid(self.linear1(x))

    model = MyModel()
    optimizer = aikit.Adam(1e-4)
    x_in = aikit.array([1.0, 2.0, 3.0])
    target = aikit.array([0.0])

    def loss_fn(v):
        out = model(x_in, v=v)
        return aikit.mean((out - target) ** 2)

    for step in range(100):
        loss, grads = aikit.execute_with_gradients(loss_fn, model.v)
        model.v = optimizer.step(model.v, grads)

    aikit.previous_backend()
