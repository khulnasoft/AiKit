import aikit
import aikit.functional.frontends.tensorflow as tf_frontend
from aikit.functional.frontends.tensorflow.func_wrapper import to_aikit_arrays_and_back
from aikit import with_supported_dtypes


ACTIVATION_FUNCTIONS = [
    "gelu",
    "leaky_relu",
    "log_softmax",
    "relu",
    "sigmoid",
    "silu",
    "softmax",
    "softplus",
]


@with_supported_dtypes(
    {"2.15.0 and below": ("float16", "float32", "float64")},
    "tensorflow",
)
def deserialize(name, custom_objects=None):
    if name is None:
        return None

    elif isinstance(name, str):
        if custom_objects and name in custom_objects:
            return custom_objects.get(name)

        # To replicate tensorflow framework
        elif (
            aikit.current_backend().__name__.split(".")[-1] == "tensorflow"
            and name in tf_frontend.keras.activations.__dict__
        ):  # noqa
            return tf_frontend.keras.activations.__dict__[name]

        # On other backends, query the function from global aikit dict
        elif name in ACTIVATION_FUNCTIONS:
            return aikit.__dict__[name]

        else:
            raise ValueError(f"Unknown activation function: {name}.")

    else:
        raise ValueError(f"Could not interpret activation function: {name}")


@with_supported_dtypes(
    {"2.15.0 and below": ("bfloat16", "float16", "float32", "float64")},
    "tensorflow",
)
@to_aikit_arrays_and_back
def elu(x, alpha=1.0):
    zeros = aikit.zeros_like(x, dtype=aikit.dtype(x))
    ones = aikit.ones_like(x, dtype=aikit.dtype(x))
    alpha = aikit.astype(aikit.array(alpha), aikit.dtype(x))
    ret_val = aikit.where(
        x > zeros, x, aikit.multiply(alpha, aikit.subtract(aikit.exp(x), ones))
    )
    return ret_val


@to_aikit_arrays_and_back
def gelu(x, approximate=False):
    return aikit.gelu(x, approximate=approximate)


def get(identifier):
    if identifier is None:
        return tf_frontend.keras.activations.linear

    elif isinstance(identifier, str):
        return tf_frontend.keras.activations.deserialize(identifier)

    elif callable(identifier):
        return identifier

    else:
        raise ValueError(f"Could not interpret function identifier: {identifier}")


@to_aikit_arrays_and_back
def hard_sigmoid(x):
    dtype_in = x.dtype
    point_two = aikit.full(x.shape, 0.2)
    point_five = aikit.full(x.shape, 0.5)
    x = aikit.multiply(x, point_two)
    x = aikit.add(x, point_five)
    x = aikit.clip(x, 0.0, 1.0)
    x = aikit.asarray(x, dtype=dtype_in)
    return x


@to_aikit_arrays_and_back
def linear(x):
    return aikit.array(x)


@to_aikit_arrays_and_back
def relu(x, alpha=0.0, max_value=None, threshold=0.0):
    return aikit.relu(x)


@with_supported_dtypes(
    {"2.15.0 and below": ("float16", "float32", "float64")},
    "tensorflow",
)
@to_aikit_arrays_and_back
def selu(x):
    return aikit.selu(x)


@with_supported_dtypes(
    {"2.15.0 and below": ("float16", "float32", "float64")},
    "tensorflow",
)
def serialize(activation, use_legacy_format=False, custom_objects=None):
    # If the activation function is None, return None
    if activation is None:
        return None

    # If the activation function is already a string, return it
    elif isinstance(activation, str):
        return activation

    # If the activation function is callable (a function), get its name
    elif callable(activation):
        # Check if the function is in the custom_objects dictionary
        if custom_objects:
            for name, custom_func in custom_objects.items():
                if custom_func == activation:
                    return name

        # Check if the function is in the ACTIVATION_FUNCTIONS list
        if activation.__name__ in ACTIVATION_FUNCTIONS:
            return activation.__name__

        # Check if the function is in the TensorFlow frontend activations
        elif activation in tf_frontend.keras.activations.__dict__.values():
            for name, tf_func in tf_frontend.keras.activations.__dict__.items():
                if tf_func == activation:
                    return name

        else:
            raise ValueError(f"Unknown activation function: {activation}.")

    else:
        raise ValueError(f"Could not interpret activation function: {activation}")


@to_aikit_arrays_and_back
def sigmoid(x):
    return aikit.sigmoid(x)


@to_aikit_arrays_and_back
def softmax(x, axis=-1):
    return aikit.softmax(x, axis=axis)


@to_aikit_arrays_and_back
def softplus(x):
    return aikit.softplus(x)


@to_aikit_arrays_and_back
def softsign(x):
    return aikit.divide(x, aikit.add(1, aikit.abs(x)))


@to_aikit_arrays_and_back
def swish(x):
    return aikit.multiply(x, aikit.sigmoid(x))


@to_aikit_arrays_and_back
def tanh(x):
    return aikit.tanh(x)
