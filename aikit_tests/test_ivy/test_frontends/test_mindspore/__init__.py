from aikit_tests.test_aikit.test_frontends import NativeClass


mindspore_classes_to_aikit_classes = {}


def convmindspore(argument):
    """Convert NativeClass in argument to aikit frontend counterpart for jax."""
    if isinstance(argument, NativeClass):
        return mindspore_classes_to_aikit_classes.get(argument._native_class)
    return argument
