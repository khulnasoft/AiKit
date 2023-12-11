# NOQA
import aikit
from importlib import import_module as builtin_import


def import_module(name, package=None):
    if aikit.is_local():
        with aikit.utils._importlib.LocalAikitImporter():
            return aikit.utils._importlib._import_module(name=name, package=package)
    return builtin_import(name=name, package=package)
