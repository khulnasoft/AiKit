# global
from typing import Optional, Union, List, Dict, Tuple

# local
import aikit
from aikit.data_classes.container.base import ContainerBase


class _ContainerWithSearchingExperimental(ContainerBase):
    @staticmethod
    def static_unravel_index(
        indices: aikit.Container,
        shape: Union[Tuple[int], aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[Union[aikit.Array, aikit.Container]] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.unravel_index. This
        method simply wraps the function, and so the docstring for
        aikit.unravel_index also applies to this method with minimal changes.

        Parameters
        ----------
        indices
            Input container including arrays.
        shape
            The shape of the array to use for unraveling indices.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Container with tuples that have arrays with the same shape as
            the arrays in the input container.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> indices = aikit.Container(a=aikit.array([22, 41, 37])), b=aikit.array([30, 2]))
        >>> aikit.Container.static_unravel_index(indices, (7,6))
        {
            a: (aikit.array([3, 6, 6]), aikit.array([4, 5, 1]))
            b: (aikit.array([5, 0], aikit.array([0, 2])))
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "unravel_index",
            indices,
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def unravel_index(
        self: aikit.Container,
        shape: Union[Tuple[int], aikit.Container],
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.unravel_index. This
        method simply wraps the function, and so the docstring for
        aikit.unravel_index also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container including arrays.
        shape
            The shape of the array to use for unraveling indices.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Container with tuples that have arrays with the same shape as
            the arrays in the input container.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> indices = aikit.Container(a=aikit.array([22, 41, 37])), b=aikit.array([30, 2]))
        >>> indices.unravel_index((7, 6))
        {
            a: (aikit.array([3, 6, 6]), aikit.array([4, 5, 1]))
            b: (aikit.array([5, 0], aikit.array([0, 2])))
        }
        """
        return self.static_unravel_index(self, shape, out=out)
