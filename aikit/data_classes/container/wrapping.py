# local
import aikit

# global
from typing import Callable, Type, List, Iterable, Optional, Union, Sequence, Dict
from types import ModuleType

TO_IGNORE = ["is_aikit_array", "is_native_array", "is_array", "shape"]


def _wrap_function(function_name: str, static: bool) -> Callable:
    """Wrap the function called `function_name`.

    Parameters
    ----------
    function_name
        the name of the function e.g. "abs", "mean" etc.
    static
        whether the function being wrapped will be added as a static method.

    Returns
    -------
    new_function
        the wrapped function.
    """

    def new_function(
        *args,
        key_chains: Optional[
            Union[Sequence[str], Dict[str, str], aikit.Container]
        ] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
        **kwargs
    ):
        function = aikit.__dict__[function_name]
        data_idx = function.array_spec[0]
        if (
            not (data_idx[0][0] == 0 and len(data_idx[0]) == 1)
            and args
            and aikit.is_aikit_container(args[0])
            and not static
        ):
            # if the method has been called as an instance method, and self should not
            # be the first positional arg, then we need to re-arrange and place self
            # in the correct location in the args or kwargs
            self = args[0]
            args = args[1:]
            if len(args) > data_idx[0][0]:
                args = aikit.copy_nest(args, to_mutable=True)
                data_idx = [data_idx[0][0]] + [
                    0 if idx is int else idx for idx in data_idx[1:]
                ]
                aikit.insert_into_nest_at_index(args, data_idx, self)
            else:
                kwargs = aikit.copy_nest(kwargs, to_mutable=True)
                data_idx = [data_idx[0][1]] + [
                    0 if idx is int else idx for idx in data_idx[1:]
                ]
                aikit.insert_into_nest_at_index(kwargs, data_idx, self)

        # return function multi-mapped across the corresponding leaves of the containers
        return aikit.ContainerBase.cont_multi_map_in_function(
            function_name,
            *args,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **kwargs
        )

    return new_function


def add_aikit_container_instance_methods(
    cls: Type[aikit.Container],
    modules: Union[List[ModuleType], aikit.Container],
    static: Union[bool, aikit.Container] = False,
    to_ignore: Union[Iterable, aikit.Container] = (),
):
    """Loop over all aikit modules such as activations, general, etc. and add the
    module functions to aikit container as instance methods using _wrap_function.

    Parameters
    ----------
    cls
        the class we want to add the instance methods to.
    modules
        the modules to loop over: activations, general etc.
    static
        whether the function should be added as a static method.
    to_ignore
        any functions we don't want to add an instance method for.

    Examples
    --------
    As shown, `add_aikit_container_instance_methods` adds all the appropriate functions
    from the statistical module as instance methods to our toy `ContainerExample` class:

    >>> from aikit.functional.aikit import statistical
    >>> class ContainerExample:
    ...     pass
    >>> aikit.add_aikit_container_instance_methods(ContainerExample, [statistical])
    >>> print(hasattr(ContainerExample, "mean"), hasattr(ContainerExample, "var"))
    True True
    """
    to_ignore = TO_IGNORE + list(to_ignore)
    for module in modules:
        for key, value in module.__dict__.items():
            full_key = ("static_" if static else "") + key
            # skip cases where the function is protected, or first letter is uppercase
            # (i.e. is a class), or if the instance method already exists etc
            if (
                key.startswith("_")
                or key[0].isupper()
                or not callable(value)
                or full_key in cls.__dict__
                or hasattr(cls, full_key)
                or full_key in to_ignore
                or key not in aikit.__dict__
            ):
                continue
            try:
                setattr(cls, full_key, _wrap_function(key, static))
            except AttributeError:
                pass
