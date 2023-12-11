import numpy
from aikit_tests.test_aikit.test_frontends import NativeClass


numpy_classes_to_aikit_classes = {numpy._NoValue: None}


def convnumpy(argument):
    """Convert NativeClass in argument to aikit frontend counterpart for
    numpy."""
    if isinstance(argument, NativeClass):
        return numpy_classes_to_aikit_classes.get(argument._native_class)
    return argument
