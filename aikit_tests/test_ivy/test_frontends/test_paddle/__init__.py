# import paddle
from aikit_tests.test_aikit.test_frontends import NativeClass


paddle_classes_to_aikit_classes = {}


def convpaddle(argument):
    """Convert NativeClass in argument to aikit frontend counter part for
    paddle."""
    if isinstance(argument, NativeClass):
        return paddle_classes_to_aikit_classes.get(argument._native_class)
    return argument
