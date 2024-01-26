from aikit_tests.test_aikit.test_frontends import NativeClass


scipy_classes_to_aikit_classes = {}


def convscipy(argument):
    """Convert NativeClass in argument to aikit frontend counterpart for
    scipy."""
    if isinstance(argument, NativeClass):
        return scipy_classes_to_aikit_classes.get(argument._native_class)
    return argument
