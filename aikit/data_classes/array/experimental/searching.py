# global
import abc
from typing import Optional, Tuple

# local
import aikit


class _ArrayWithSearchingExperimental(abc.ABC):
    def unravel_index(
        self: aikit.Array,
        shape: Tuple[int],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> Tuple[aikit.Array]:
        """aikit.Array instance method variant of aikit.unravel_index. This method
        simply wraps the function, and so the docstring for aikit.unravel_index
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array.
        shape
            The shape of the array to use for unraveling indices.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Tuple with arrays that have the same shape as the indices array.

        Examples
        --------
        >>> indices = aikit.array([22, 41, 37])
        >>> indices.unravel_index((7,6))
        (aikit.array([3, 6, 6]), aikit.array([4, 5, 1]))
        """
        return aikit.unravel_index(self._data, shape, out=out)
