# import tensorflow
from aikit_tests.test_aikit.test_frontends import NativeClass


tensorflow_classes_to_aikit_classes = {}


def convtensor(argument):
    """Convert NativeClass in argument to aikit frontend counterpart for
    tensorflow."""
    if isinstance(argument, NativeClass):
        return tensorflow_classes_to_aikit_classes.get(argument._native_class)
    return argument
