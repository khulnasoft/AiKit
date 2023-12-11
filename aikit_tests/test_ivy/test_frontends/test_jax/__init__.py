# import jax
from aikit_tests.test_aikit.test_frontends import NativeClass


jax_classes_to_aikit_classes = {}


def convjax(argument):
    """Convert NativeClass in argument to aikit frontend counterpart for jax."""
    if isinstance(argument, NativeClass):
        return jax_classes_to_aikit_classes.get(argument._native_class)
    return argument
