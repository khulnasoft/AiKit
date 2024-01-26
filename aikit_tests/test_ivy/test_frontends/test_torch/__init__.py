# import torch
from aikit_tests.test_aikit.test_frontends import NativeClass


torch_classes_to_aikit_classes = {}


def convtorch(argument):
    """Convert NativeClass in argument to aikit frontend counterpart for
    torch."""
    if isinstance(argument, NativeClass):
        return torch_classes_to_aikit_classes.get(argument._native_class)
    return argument
