"""Aikit wrapping functions for conversions.

Collection of Aikit functions for wrapping functions to accept and return
aikit.Array instances.
"""

# global
from typing import Union, Dict, Optional, List

# local
import aikit
from aikit.data_classes.container.base import ContainerBase


class _ContainerWithConversions(ContainerBase):
    @staticmethod
    def _static_to_native(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        nested: Union[bool, aikit.Container] = False,
        include_derived: Optional[Union[Dict[str, bool], aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.to_native.

        This method simply wraps the function, and so the docstring for aikit.to_native
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            The input to be converted.
        nested
            Whether to apply the conversion on arguments in a nested manner. If so, all
            dicts, lists and tuples will be traversed to their lowest leaves in search
            of aikit.Array instances. Default is ``False``.
        include_derived
            Whether to also recursive for classes derived from tuple, list and dict.
            Default is ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Container object with all sub-arrays converted to their native format.
        """
        return ContainerBase.cont_multi_map_in_function(
            "to_native",
            x,
            nested=nested,
            include_derived=include_derived,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def to_native(
        self: aikit.Container,
        nested: Union[bool, aikit.Container] = False,
        include_derived: Optional[Union[Dict[str, bool], aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.to_native.

        This method simply wraps the function, and so the docstring for aikit.to_native
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            The input to be converted.
        nested
            Whether to apply the conversion on arguments in a nested manner. If so, all
            dicts, lists and tuples will be traversed to their lowest leaves in search
            of aikit.Array instances. Default is ``False``.
        include_derived
            Whether to also recursive for classes derived from tuple, list and dict.
            Default is ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Container object with all sub-arrays converted to their native format.
        """
        return self._static_to_native(
            self,
            nested,
            include_derived,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def _static_to_aikit(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        nested: Union[bool, aikit.Container] = False,
        include_derived: Optional[Union[Dict[str, bool], aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.to_aikit.

        This method simply wraps the function, and so the docstring for aikit.to_aikit also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            The input to be converted.
        nested
            Whether to apply the conversion on arguments in a nested manner. If so, all
            dicts, lists and tuples will be traversed to their lowest leaves in search
            of aikit.Array instances. Default is ``False``.
        include_derived
            Whether to also recursive for classes derived from tuple, list and dict.
            Default is ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Container object with all native sub-arrays converted to their aikit.Array
            instances.
        """
        return ContainerBase.cont_multi_map_in_function(
            "to_aikit",
            x,
            nested=nested,
            include_derived=include_derived,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def to_aikit(
        self: aikit.Container,
        nested: Union[bool, aikit.Container] = False,
        include_derived: Optional[Union[Dict[str, bool], aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.to_aikit.

        This method simply wraps the function, and so the docstring for aikit.to_aikit also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            The input to be converted.
        nested
            Whether to apply the conversion on arguments in a nested manner. If so,
            all dicts, lists and tuples will be traversed to their lowest leaves in
            search of aikit.Array instances. Default is ``False``.
        include_derived
            Whether to also recursive for classes derived from tuple, list and dict.
            Default is ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Container object with all native sub-arrays converted to their aikit.Array
            instances.
        """
        return self._static_to_aikit(
            self,
            nested,
            include_derived,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )
