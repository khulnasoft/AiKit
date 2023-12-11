# global
import abc
from typing import Union, Callable, Sequence

# local
import aikit


class _ArrayWithGeneralExperimental(abc.ABC):
    def reduce(
        self: aikit.Array,
        init_value: Union[int, float],
        computation: Callable,
        /,
        *,
        axes: Union[int, Sequence[int]] = 0,
        keepdims: bool = False,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.reduce. This method simply
        wraps the function, and so the docstring for aikit.reduce also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            The array to act on.
        init_value
            The value with which to start the reduction.
        computation
            The reduction function.
        axes
            The dimensions along which the reduction is performed.
        keepdims
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one.

        Returns
        -------
        ret
            The reduced array.

        Examples
        --------
        >>> x = aikit.array([[1, 2, 3], [4, 5, 6]])
        >>> x.reduce(0, aikit.add, 0)
        aikit.array([6, 15])
        """
        return aikit.reduce(self, init_value, computation, axes=axes, keepdims=keepdims)
