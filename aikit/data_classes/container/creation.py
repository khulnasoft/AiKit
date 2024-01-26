# global
from typing import Optional, Union, List, Tuple, Dict, Sequence
from numbers import Number
import numpy as np

# local
import aikit
from aikit.data_classes.container.base import ContainerBase


class _ContainerWithCreation(ContainerBase):
    @staticmethod
    def _static_arange(
        start: Union[Number, aikit.Container],
        /,
        stop: Optional[Union[Number, aikit.Container]] = None,
        step: Union[Number, aikit.Container] = 1,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "arange",
            start,
            stop=stop,
            step=step,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def _static_asarray(
        x: Union[
            aikit.Array,
            aikit.NativeArray,
            List[Number],
            Tuple[Number],
            np.ndarray,
            aikit.Container,
        ],
        /,
        copy: Optional[Union[bool, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.asarray. This method
        simply wraps the function, and so the docstring for aikit.asarray also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input data, in any form that can be converted to an array. This includes
            lists, lists of tuples, tuples, tuples of tuples, tuples of lists and
            ndarrays.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        dtype
            datatype, optional. Datatype is inferred from the input data.
        device
            device on which to place the created array. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            An array interpretation of ``self``.

        Examples
        --------
        With :class:`aikit.Container` as input:
        >>> x = aikit.Container(a = [(1,2),(3,4),(5,6)], b = ((1,2,3),(4,5,6)))
        >>> aikit.asarray(x)
        {
            a: aikit.array([[1, 2],
                          [3, 4],
                          [5, 6]]),
            b: aikit.array([[1, 2, 3],
                          [4, 5, 6]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "asarray",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            copy=copy,
            dtype=dtype,
            device=device,
            out=out,
        )

    def asarray(
        self: aikit.Container,
        /,
        copy: Optional[Union[bool, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return self._static_asarray(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            copy=copy,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def _static_zeros(
        shape: Union[int, Sequence[int], aikit.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        out: Optional[aikit.Container] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "zeros",
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _static_ones(
        shape: Union[int, Sequence[int], aikit.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        out: Optional[aikit.Container] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "ones",
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _static_empty(
        shape: Union[int, Sequence[int], aikit.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "empty",
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def _static_full(
        shape: Union[aikit.Shape, aikit.NativeShape, aikit.Container],
        fill_value: Union[float, bool, aikit.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "full",
            shape,
            fill_value,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _static_full_like(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        fill_value: Union[int, float, aikit.Container],
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.full_like. This method
        simply wraps the function, and so the docstring for aikit.full_like also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        fill_value
            Scalar fill value
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
        dtype
            output array data type. If ``dtype`` is `None`, the output array data type
            must be inferred from ``self``. Default: ``None``.
        device
            device on which to place the created array. If ``device`` is ``None``, the
            output array device must be inferred from ``self``. Default: ``None``.

        Returns
        -------
        ret
            an output container having the same data type as ``x`` and whose elements,
            relative to ``x``, are shifted.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a = aikit.array([1,2,3]) ,b = aikit.array([4,5,6]))
        >>> fill_value = 10
        >>> y = aikit.Container.static_full_like(fill_value)
        {
            a: aikit.array([10, 10, 10]),
            b: aikit.array([10, 10, 10])
        }

        >>> x = aikit.Container(a=aikit.array([1.2, 2.2324, 3.234]),
        ...                   b=aikit.array([4.123, 5.23, 6.23]))
        >>> fill_value = 15.0
        >>> y = aikit.Container.static_full_like(fill_value)
        >>> print(y)
        {
            a: aikit.array([15., 15., 15.]),
            b: aikit.array([15., 15., 15.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "full_like",
            x,
            fill_value=fill_value,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def full_like(
        self: aikit.Container,
        /,
        fill_value: Union[int, float, aikit.Container],
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        out: Optional[aikit.Container] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.full_like. This method
        simply wraps the function, and so the docstring for aikit.full_like also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        fill_value
            Scalar fill value
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
        dtype
            output array data type. If ``dtype`` is `None`, the output array data type
            must be inferred from ``self``. Default: ``None``.
        device
            device on which to place the created array. If ``device`` is ``None``, the
            output array device must be inferred from ``self``. Default: ``None``.

        Returns
        -------
        ret
            an output container having the same data type as ``x`` and whose elements,
            relative to ``x``, are shifted.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a = aikit.array([1,2,3]) ,b = aikit.array([4,5,6]))
        >>> fill_value = 10
        >>> y = x.full_like(fill_value)
        {
            a: aikit.array([10, 10, 10]),
            b: aikit.array([10, 10, 10])
        }

        >>> x = aikit.Container(a=aikit.array([1.2,2.2324,3.234]),
        ...                   b=aikit.array([4.123,5.23,6.23]))
        >>> fill_value = 15.0
        >>> y = x.full_like(fill_value)
        >>> print(y)
        {
            a: aikit.array([15., 15., 15.]),
            b: aikit.array([15., 15., 15.])
        }
        """
        return self._static_full_like(
            self,
            fill_value,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _static_ones_like(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.ones_like. This method
        simply wraps the function, and so the docstring for aikit.ones_like also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array from which to derive the output array shape.
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
        dtype
            output array data type. If ``dtype`` is ``None``, the output array data type
            must be inferred from ``self``. Default  ``None``.
        device
            device on which to place the created array. If device is ``None``, the
            output array device must be inferred from ``self``. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            a container having the same shape as ``self`` and filled with ones.
        """
        return ContainerBase.cont_multi_map_in_function(
            "ones_like",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def ones_like(
        self: aikit.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.ones_like. This method
        simply wraps the function, and so the docstring for aikit.ones_like also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array from which to derive the output array shape.
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
        dtype
            output array data type. If ``dtype`` is ``None``, the output array data type
            must be inferred from ``self``. Default  ``None``.
        device
            device on which to place the created array. If device is ``None``, the
            output array device must be inferred from ``self``. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            a container having the same shape as ``self`` and filled with ones.
        """
        return self._static_ones_like(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def _static_zeros_like(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.zeros_like. This method
        simply wraps the function, and so the docstring for aikit.zeros_like also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array or container from which to derive the output container shape.
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
        dtype
            output array data type. If ``dtype`` is ``None``, the output container
            data type must be inferred from ``self``. Default  ``None``.
        device
            device on which to place the created array. If device is ``None``, the
            output container device must be inferred from ``self``. Default: ``None``.
        out
            optional output container, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            an container having the same shape as ``x`` and filled with ``zeros``.
        """
        return ContainerBase.cont_multi_map_in_function(
            "zeros_like",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def zeros_like(
        self: aikit.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.zeros_like. This method
        simply wraps the function, and so the docstring for aikit.zeros_like also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container from which to derive the output container shape.
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
        dtype
            output array data type. If ``dtype`` is ``None``, the output container
            data type must be inferred from ``self``. Default: ``None``.
        device
            device on which to place the created array. If device is ``None``, the
            output container device must be inferred from ``self``. Default: ``None``.
        out
            optional output container, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            an container having the same shape as ``x`` and filled with ``zeros``.
        """
        return self._static_zeros_like(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _static_tril(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        k: Union[int, aikit.Container] = 0,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "tril",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            k=k,
            out=out,
        )

    def tril(
        self: aikit.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        k: Union[int, aikit.Container] = 0,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return self._static_tril(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            k=k,
            out=out,
        )

    @staticmethod
    def _static_triu(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        k: Union[int, aikit.Container] = 0,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "triu",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            k=k,
            out=out,
        )

    def triu(
        self: aikit.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        k: Union[int, aikit.Container] = 0,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return self._static_triu(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            k=k,
            out=out,
        )

    @staticmethod
    def _static_empty_like(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "empty_like",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def empty_like(
        self: aikit.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return self._static_empty_like(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def _static_eye(
        n_rows: Union[int, aikit.Container],
        n_cols: Optional[Union[int, aikit.Container]] = None,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        k: Union[int, aikit.Container] = 0,
        batch_shape: Optional[Union[int, Sequence[int], aikit.Container]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "eye",
            n_rows,
            n_cols,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            k=k,
            batch_shape=batch_shape,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def _static_linspace(
        start: Union[aikit.Array, aikit.NativeArray, float, aikit.Container],
        stop: Union[aikit.Array, aikit.NativeArray, float, aikit.Container],
        /,
        num: Union[int, aikit.Container],
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        axis: Optional[Union[int, aikit.Container]] = None,
        endpoint: Union[bool, aikit.Container] = True,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "linspace",
            start,
            stop,
            num,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            axis=axis,
            endpoint=endpoint,
            dtype=dtype,
            device=device,
            out=out,
        )

    def linspace(
        self: aikit.Container,
        stop: Union[aikit.Array, aikit.NativeArray, float, aikit.Container],
        /,
        num: Union[int, aikit.Container],
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        axis: Optional[Union[int, aikit.Container]] = None,
        endpoint: Union[bool, aikit.Container] = True,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return self._static_linspace(
            self,
            stop,
            num,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            axis=axis,
            endpoint=endpoint,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def _static_meshgrid(
        *arrays: Union[
            aikit.Array, aikit.NativeArray, List[Number], Tuple[Number], aikit.Container
        ],
        sparse: Union[bool, aikit.Container] = False,
        indexing: Union[str, aikit.Container] = "xy",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "meshgrid",
            *arrays,
            sparse=sparse,
            indexing=indexing,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def meshgrid(
        self: aikit.Container,
        *arrays: Union[
            aikit.Array, aikit.NativeArray, List[Number], Tuple[Number], aikit.Container
        ],
        sparse: Union[bool, aikit.Container] = False,
        indexing: Union[str, aikit.Container] = "xy",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        return self._static_meshgrid(
            self,
            *arrays,
            sparse=sparse,
            indexing=indexing,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_from_dlpack(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "from_dlpack",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def from_dlpack(
        self: aikit.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return self._static_from_dlpack(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_copy_array(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        to_aikit_array: Union[bool, aikit.Container] = True,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "copy_array",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            to_aikit_array=to_aikit_array,
            out=out,
        )

    def copy_array(
        self: aikit.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        to_aikit_array: Union[bool, aikit.Container] = True,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return self._static_copy_array(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            to_aikit_array=to_aikit_array,
            out=out,
        )

    @staticmethod
    def _static_native_array(
        x: Union[
            aikit.Array,
            aikit.NativeArray,
            List[Number],
            Tuple[Number],
            np.ndarray,
            aikit.Container,
        ],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "native_array",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
        )

    def native_array(
        self: aikit.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
    ) -> aikit.Container:
        return self._static_native_array(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _static_logspace(
        start: Union[aikit.Array, aikit.NativeArray, float, aikit.Container],
        stop: Union[aikit.Array, aikit.NativeArray, float, aikit.Container],
        /,
        num: Union[int, aikit.Container],
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        base: Union[float, aikit.Container] = 10.0,
        axis: Union[int, aikit.Container] = 0,
        endpoint: Union[bool, aikit.Container] = True,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.logspace. This method
        simply wraps the function, and so the docstring for aikit.logspace also
        applies to this method with minimal changes.

        Parameters
        ----------
        start
            Container for first value in the range in log space.
        stop
            Container for last value in the range in log space.
        num
            Number of values to generate.
        base
            The base of the log space. Default is 10.0
        axis
            Axis along which the operation is performed. Relevant only if values in
            start or stop containers are array-like. Default is 0.
        endpoint
            If True, stop is the last sample. Otherwise, it is not included. Default is
            True.
        dtype
            The data type of the output tensor. If None, the dtype of on_value is used
            or if that is None, the dtype of off_value is used, or if that is None,
            defaults to float32. Default is None.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Default
            is None.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to. Default is None.

        Returns
        -------
        ret
            a container having the same shape as ``start`` and filled with tensor of
            evenly-spaced values in log space.

        Examples
        --------
        >>> import aikit.container.creation.static_logspace as static_logspace
        >>> x = aikit.Container(a = 1, b = 0)
        >>> y = aikit.Container(a = 4, b = 1)
        >>> z = static_logspace(x, y, 4)
        {
            a: aikit.array([10.,  100.,  1000., 10000.]),
            b: aikit.array([ 1., 2.15443469, 4.64158883, 10.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "logspace",
            start,
            stop,
            num=num,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            base=base,
            axis=axis,
            endpoint=endpoint,
            dtype=dtype,
            device=device,
            out=out,
        )

    def logspace(
        self: aikit.Container,
        stop: Union[aikit.Array, aikit.NativeArray, float, aikit.Container],
        /,
        num: Union[int, aikit.Container],
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        base: Union[float, aikit.Container] = 10.0,
        axis: Union[int, aikit.Container] = None,
        endpoint: Union[bool, aikit.Container] = True,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.logspace. This method
        simply wraps the function, and so the docstring for aikit.logspace also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Container for first value in the range in log space.
        stop
            Container for last value in the range in log space.
        num
            Number of values to generate.
        base
            The base of the log space. Default is 10.0
        axis
            Axis along which the operation is performed. Relevant only if values in
            start or stop containers are array-like. Default is 0.
        endpoint
            If True, stop is the last sample. Otherwise, it is not included. Default is
            True.
        dtype
            The data type of the output tensor. If None, the dtype of on_value is used
            or if that is None, the dtype of off_value is used, or if that is None,
            defaults to float32. Default is None.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Default
            is None.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to. Default is None.

        Returns
        -------
        ret
            a container having the same shape as ``self`` and filled with tensor of
            evenly-spaced values in log space.

        Examples
        --------
        >>> x = aikit.Container(a = 1, b = 0)
        >>> y = aikit.Container(a = 4, b = 1)
        >>> z = x.logspace(y, 4)
        {
            a: aikit.array([10.,  100.,  1000., 10000.]),
            b: aikit.array([ 1., 2.15443469, 4.64158883, 10.])
        }

        >>> x = aikit.Container(a = 1, b = 0)
        >>> y = aikit.Container(a = 4, b = 1)
        >>> z = aikit.logspace(x, y, 4)
        {
            a: aikit.array([10.,  100.,  1000., 10000.]),
            b: aikit.array([ 1., 2.15443469, 4.64158883, 10.])
        }

        >>> u = aikit.Container(c = 0, d = 0)
        >>> v = aikit.Container(c = 1, d = 2)
        >>> x = aikit.Container(a = 1, b = u)
        >>> y = aikit.Container(a = 4, b = v)
        >>> z = x.logspace(y, 4)
        {
            a: aikit.array([10.,  100.,  1000., 10000.]),
            b:  {
                    c: aikit.array([ 1., 2.15443469, 4.64158883, 10.])
                    d: aikit.array([ 1., 4.64158883, 21.5443469, 100.])
                }
        }
        """
        return self._static_logspace(
            self,
            stop,
            num=num,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            base=base,
            axis=axis,
            endpoint=endpoint,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def _static_one_hot(
        indices: aikit.Container,
        depth: Union[int, aikit.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        on_value: Optional[Union[Number, aikit.Container]] = None,
        off_value: Optional[Union[Number, aikit.Container]] = None,
        axis: Optional[Union[int, aikit.Container]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.one_hot. This method
        simply wraps the function, and so the docstring for aikit.one_hot also
        applies to this method with minimal changes.

        Parameters
        ----------
        indices
            Indices for where the ones should be scattered *[batch_shape, dim]*
        depth
            Scalar defining the depth of the one-hot dimension.
        on_value
            Value to fill in output when indices[j] = i. If None, defaults to 1.
        off_value
            Value to fill in output when indices[j] != i. If None, defaults to 0.
        axis
            Axis to scatter on. The default is ``-1``, a new inner-most axis is created.
        dtype
            The data type of the output tensor. If None, defaults to the on_value dtype
            or the off_value dtype. If both are None, defaults to float32.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            container with tensors of zeros with the same shape and type as the inputs,
            unless dtype provided which overrides.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([1, 2]), \
            b=aikit.array([3, 1]), c=aikit.array([2, 3]))
        >>> y = 5
        >>> z = aikit.Container.static_one_hot(x, y)
        >>> print(z)
        {
            a: aikit.array([[0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 0.]]),
            b: aikit.array([[0., 0., 0., 1., 0.],
                        [0., 1., 0., 0., 0.]]),
            c: aikit.array([[0., 0., 1., 0., 0.],
                        [0., 0., 0., 1., 0.]])
        }

        >>> x = aikit.Container(a=aikit.array([1, 2]), \
            b=aikit.array([]), c=aikit.native_array([4]))
        >>> y = 5
        >>> z = aikit.Container.static_one_hot(x, y)
        >>> print(z)
        {
            a: aikit.array([[0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 0.]]),
            b: aikit.array([], shape=(0, 5)),
            c: aikit.array([[0., 0., 0., 0., 1.]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "one_hot",
            indices,
            depth,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            on_value=on_value,
            off_value=off_value,
            axis=axis,
            dtype=dtype,
            device=device,
            out=out,
        )

    def one_hot(
        self: aikit.Container,
        depth: Union[int, aikit.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        on_value: Optional[Union[Number, aikit.Container]] = None,
        off_value: Optional[Union[Number, aikit.Container]] = None,
        axis: Optional[Union[int, aikit.Container]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.one_hot. This method
        simply wraps the function, and so the docstring for aikit.one_hot also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Indices for where the ones should be scattered *[batch_shape, dim]*
        depth
            Scalar defining the depth of the one-hot dimension.
        on_value
            Value to fill in output when indices[j] == i. If None, defaults to 1.
        off_value
            Value to fill in output when indices[j] != i. If None, defaults to 0.
        axis
            Axis to scatter on. The default is ``-1``, a new inner-most axis is created.
        dtype
            The dtype of the returned tensor. If None, defaults to the on_value dtype
            or the off_value dtype. If both are None, defaults to float32.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            container with tensors of zeros with the same shape and type as the inputs,
            unless dtype provided which overrides.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([1, 2]), \
             b=aikit.array([3, 1]), c=aikit.array([2, 3]))
        >>> y = 5
        >>> z = x.one_hot(y)
        >>> print(z)
        {
            a: aikit.array([[0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 0.]]),
            b: aikit.array([[0., 0., 0., 1., 0.],
                        [0., 1., 0., 0., 0.]]),
            c: aikit.array([[0., 0., 1., 0., 0.],
                        [0., 0., 0., 1., 0.]])
        }

        >>> x = aikit.Container(a=aikit.array([1, 2]), \
             b=aikit.array([]), c=aikit.native_array([4]))
        >>> y = 5
        >>> z = x.one_hot(y)
        >>> print(z)
        {
            a: aikit.array([[0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 0.]]),
            b: aikit.array([], shape=(0, 5)),
            c: aikit.array([[0., 0., 0., 0., 1.]])
        }
        """
        return self._static_one_hot(
            self,
            depth,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            on_value=on_value,
            off_value=off_value,
            axis=axis,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def static_frombuffer(
        buffer: aikit.Container,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = float,
        count: Optional[Union[int, aikit.Container]] = -1,
        offset: Optional[Union[int, aikit.Container]] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        r"""aikit.Container static method variant of aikit.frombuffer. This method
        simply wraps the function, and so the docstring for aikit.frombuffer also
        applies to this method with minimal changes.

        Parameters
        ----------
        buffer
            An object that exposes the buffer interface.
        dtype
            Data-type of the returned array; default: float.
        count
            Number of items to read. -1 means all data in the buffer.
        offset
            Start reading the buffer from this offset (in bytes); default: 0.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        out
            1-dimensional array.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x = aikit.Container(
        ...     a = b'\x00\x00\x00\x00\x00\x00\xf0?',
        ...     b = b'\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@'
        ... )
        >>> y = aikit.Container.static_frombuffer(x)
        >>> print(y)
        {
            a: aikit.array([1.]),
            b: aikit.array([1., 2.])
        }

        >>> x = aikit.Container(
        ...     a = b'\x01\x02\x03\x04',
        ...     b = b'\x05\x04\x03\x03\x02'
        ... )
        >>> y = aikit.Container.static_frombuffer(x, dtype=aikit.int8, count=3, offset=1)
        >>> print(y)
        {
            a: aikit.array([2, 3, 4]),
            b: aikit.array([4, 3, 3])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "frombuffer",
            buffer,
            dtype=dtype,
            count=count,
            offset=offset,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def frombuffer(
        self: aikit.Container,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = float,
        count: Optional[Union[int, aikit.Container]] = -1,
        offset: Optional[Union[int, aikit.Container]] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        r"""aikit.Container instance method variant of aikit.frombuffer. This method
        simply wraps the function, and so the docstring for aikit.frombuffer also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            An object that exposes the buffer interface.
        dtype
            Data-type of the returned array; default: float.
        count
            Number of items to read. -1 means all data in the buffer.
        offset
            Start reading the buffer from this offset (in bytes); default: 0.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        out
            1-dimensional array.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x = aikit.Container(
        ...     a = b'\x00\x00\x00\x00\x00\x00\xf0?',
        ...     b = b'\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@'
        ... )
        >>> y = x.frombuffer(dtype=aikit.float64)
        >>> print(y)
        {
            a: aikit.array([1.]),
            b: aikit.array([1., 2.])
        }

        >>> x = aikit.Container(
        ...     a = b'\x01\x02\x03\x04',
        ...     b = b'\x05\x04\x03\x03\x02'
        ... )
        >>> y = x.frombuffer(dtype=aikit.int8, count=3, offset=1)
        >>> print(y)
        {
            a: aikit.array([2, 3, 4]),
            b: aikit.array([4, 3, 3])
        }
        """
        return self.static_frombuffer(
            self,
            dtype=dtype,
            count=count,
            offset=offset,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_triu_indices(
        n_rows: Union[int, aikit.Container],
        n_cols: Optional[Union[int, aikit.Container]] = None,
        k: Union[int, aikit.Container] = 0,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[Union[Tuple[aikit.Array], aikit.Container]] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "triu_indices",
            n_rows,
            n_cols,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            device=device,
            out=out,
        )

    def triu_indices(
        self: aikit.Container,
        n_rows: Union[int, aikit.Container],
        n_cols: Optional[Union[int, aikit.Container]] = None,
        k: Union[int, aikit.Container] = 0,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        out: Optional[Union[Tuple[aikit.Array], aikit.Container]] = None,
    ) -> aikit.Container:
        return self.static_triu_indices(
            n_rows,
            n_cols,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            device=device,
            out=out,
        )
