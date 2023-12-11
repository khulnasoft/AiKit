# global
from typing import Optional, List, Union, Dict

# local
from aikit.data_classes.container.base import ContainerBase
import aikit


class _ContainerWithSortingExperimental(ContainerBase):
    @staticmethod
    def static_invert_permutation(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.invert_permutation.

        This method simply wraps the function, and so the docstring for
        aikit.invert_permutation also applies to this method with minimal
        changes.
        """
        return ContainerBase.cont_multi_map_in_function(
            "invert_permutation",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def invert_permutation(
        self: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.invert_permutation.

        This method simply wraps the function, and so the docstring for
        aikit.invert_permutation also applies to this method with minimal
        changes.
        """
        return self.static_invert_permutation(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_lexsort(
        a: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        axis: Union[int, aikit.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.lexsort. This method
        simply wraps the function, and so the docstring for aikit.lexsort also
        applies to this method with minimal changes.

        Parameters
        ----------
        a
            array-like or container input to sort as keys.
        axis
            axis of each key to be indirectly sorted.
            By default, sort over the last axis of each key.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            a container containing sorted input arrays.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> a = aikit.Container(x = aikit.asarray([[9,4,0,4,0,2,1],[1,5,1,4,3,4,4]]),
        ...                   y = aikit.asarray([[1, 5, 2],[3, 4, 4]])
        >>> aikit.Container.static_lexsort(a)
        {
            x: aikit.array([2, 0, 4, 6, 5, 3, 1])),
            y: aikit.array([0, 2, 1])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "lexsort",
            a,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def lexsort(
        self: aikit.Container,
        /,
        *,
        axis: Union[int, aikit.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.lexsort. This method
        simply wraps the function, and so the docstring for aikit.lexsort also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container with array-like inputs to sort as keys.
        axis
            axis of each key to be indirectly sorted.
            By default, sort over the last axis of each key.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            a container containing the sorted input arrays.

        Examples
        --------
        >>> a = aikit.Container(x = aikit.asarray([[9,4,0,4,0,2,1],[1,5,1,4,3,4,4]]),
        ...                   y = aikit.asarray([[1, 5, 2],[3, 4, 4]])
        >>> a.lexsort()
        {
            x: aikit.array([2, 0, 4, 6, 5, 3, 1])),
            y: aikit.array([0, 2, 1])
        }
        """
        return self.static_lexsort(
            self,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
