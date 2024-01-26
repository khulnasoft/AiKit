# global
from numbers import Number
from typing import Any, Union, List, Dict, Iterable, Optional, Callable

# local
from aikit.data_classes.container.base import ContainerBase
import aikit

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class _ContainerWithGeneral(ContainerBase):
    @staticmethod
    def _static_is_native_array(
        x: aikit.Container,
        /,
        *,
        exclusive: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.is_native_array. This
        method simply wraps the function, and so the docstring for
        aikit.is_native_array also applies to this method with minimal changes.

        Parameters
        ----------
        x
            The input to check
        exclusive
            Whether to check if the data type is exclusively an array, rather than a
            variable or traced array.
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

        Returns
        -------
        ret
            Boolean, whether or not x is a native array.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1]), b=aikit.native_array([2, 3]))
        >>> y = aikit.Container.static_is_native_array(x)
        >>> print(y)
        {
            a: false,
            b: true
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "is_native_array",
            x,
            exclusive=exclusive,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def is_native_array(
        self: aikit.Container,
        /,
        *,
        exclusive: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.is_native_array. This
        method simply wraps the function, and so the docstring for
        aikit.aikit.is_native_array also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            The input to check
        exclusive
            Whether to check if the data type is exclusively an array, rather than a
            variable or traced array.
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

        Returns
        -------
        ret
            Boolean, whether or not x is a native array.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1]), b=aikit.native_array([2, 3]))
        >>> y = x.is_native_array()
        >>> print(y)
        {
            a: False,
            b: True
        }
        """
        return self._static_is_native_array(
            self,
            exclusive=exclusive,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_is_aikit_array(
        x: aikit.Container,
        /,
        *,
        exclusive: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.is_aikit_array. This method
        simply wraps the function, and so the docstring for aikit.is_aikit_array
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            The input to check
        exclusive
            Whether to check if the data type is exclusively an array, rather than a
            variable or traced array.
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

        Returns
        -------
        ret
            Boolean, whether or not x is an array.

        >>> x = aikit.Container(a=aikit.array([1]), b=aikit.native_array([2, 3]))
        >>> y = aikit.Container.static_is_aikit_array(x)
        >>> print(y)
        {
            a: true,
            b: false
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "is_aikit_array",
            x,
            exclusive=exclusive,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def is_aikit_array(
        self: aikit.Container,
        /,
        *,
        exclusive: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.is_native_array. This
        method simply wraps the function, and so the docstring for
        aikit.aikit.is_native_array also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            The input to check
        exclusive
            Whether to check if the data type is exclusively an array, rather than a
            variable or traced array.
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

        Returns
        -------
        ret
            Boolean, whether or not x is an array.

        >>> x = aikit.Container(a=aikit.array([1]), b=aikit.native_array([2, 3]))
        >>> y = x.is_aikit_array()
        >>> print(y)
        {
            a: True,
            b: False
        }
        """
        return self._static_is_aikit_array(
            self,
            exclusive=exclusive,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_is_array(
        x: aikit.Container,
        /,
        *,
        exclusive: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.is_array. This method
        simply wraps the function, and so the docstring for aikit.aikit.is_array
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            The input to check
        exclusive
            Whether to check if the data type is exclusively an array, rather than a
            variable or traced array.
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
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            Boolean, whether or not x is an array.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1]), b=aikit.native_array([2, 3]))
        >>> y = aikit.Container.static_is_array(x)
        >>> print(y)
        {
            a: true,
            b: true
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "is_array",
            x,
            exclusive=exclusive,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def is_array(
        self: aikit.Container,
        /,
        *,
        exclusive: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.is_array. This method
        simply wraps the function, and so the docstring for aikit.is_array also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            The input to check
        exclusive
            Whether to check if the data type is exclusively an array, rather than a
            variable or traced array.
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

        Returns
        -------
        ret
            Boolean, whether or not x is an array.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1]), b=aikit.native_array([2, 3]))
        >>> y = x.is_array()
        >>> print(y)
        {
            a: True,
            b: True
        }
        """
        return self._static_is_array(
            self,
            exclusive=exclusive,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_clip_vector_norm(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        max_norm: Union[float, aikit.Container],
        /,
        *,
        p: Union[float, aikit.Container] = 2.0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.clip_vector_norm. This
        method simply wraps the function, and so the docstring for
        aikit.clip_vector_norm also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array
        max_norm
            float, the maximum value of the array norm.
        p
            optional float, the p-value for computing the p-norm.
            Default is 2.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped.
            Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the vector norm downscaled to the max norm if needed.

        Examples
        --------
        With :class:`aikit.Container` instance method:

        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),b=aikit.array([3., 4., 5.]))
        >>> y = aikit.Container.static_clip_vector_norm(x, 2.0)
        >>> print(y)
        {
            a: aikit.array([0., 0.894, 1.79]),
            b: aikit.array([0.849, 1.13, 1.41])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "clip_vector_norm",
            x,
            max_norm,
            p=p,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def clip_vector_norm(
        self: aikit.Container,
        max_norm: Union[float, aikit.Container],
        /,
        *,
        p: Union[float, aikit.Container] = 2.0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.clip_vector_norm. This
        method simply wraps the function, and so the docstring for
        aikit.clip_vector_norm also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array
        max_norm
            float, the maximum value of the array norm.
        p
            optional float, the p-value for computing the p-norm.
            Default is 2.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped.
            Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the vector norm downscaled to the max norm if needed.

        Examples
        --------
        With :class:`aikit.Container` instance method:

        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),
        ...                   b=aikit.array([3., 4., 5.]))
        >>> y = x.clip_vector_norm(2.0, p=1.0)
        >>> print(y)
        {
            a: aikit.array([0., 0.667, 1.33]),
            b: aikit.array([0.5, 0.667, 0.833])
        }
        """
        return self._static_clip_vector_norm(
            self,
            max_norm,
            p=p,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_inplace_update(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        val: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        ensure_in_backend: Union[bool, aikit.Container] = False,
        keep_input_dtype: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.inplace_update. This
        method simply wraps the function, and so the docstring for
        aikit.inplace_update also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container to be updated inplace
        val
            value to update the input container with
        ensure_in_backend
            Whether to ensure that the `aikit.NativeArray` is also inplace updated.
            In cases where it should be, backends which do not natively support inplace
            updates will raise an exception.
        keep_input_dtype
            Whether or not to preserve `x` data type after the update, otherwise `val`
            data type will be applied. Defaults to False.
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
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the vector norm downscaled to the max norm if needed.
        """
        # inplace update the leaves
        cont = x
        cont = ContainerBase.cont_multi_map_in_function(
            "inplace_update",
            cont,
            val,
            ensure_in_backend=ensure_in_backend,
            keep_input_dtype=keep_input_dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
        # inplace update the container
        x.cont_inplace_update(cont)
        return x

    def inplace_update(
        self: aikit.Container,
        val: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        ensure_in_backend: Union[bool, aikit.Container] = False,
        keep_input_dtype: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.inplace_update. This
        method simply wraps the function, and so the docstring for
        aikit.inplace_update also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container to be updated inplace
        val
            value to update the input container with
        ensure_in_backend
            Whether to ensure that the `aikit.NativeArray` is also inplace updated.
            In cases where it should be, backends which do not natively support inplace
            updates will raise an exception.
        keep_input_dtype
            Whether or not to preserve `x` data type after the update, otherwise `val`
            data type will be applied. Defaults to False.
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
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the vector norm downscaled to the max norm if needed.

        Examples
        --------
        With :class:`aikit.Container` input and default backend set as `numpy`:

        >>> x = aikit.Container(a=aikit.array([5, 6]), b=aikit.array([7, 8]))
        >>> y = aikit.Container(a=aikit.array([1]), b=aikit.array([2]))
        >>> x.inplace_update(y)
        >>> print(x)
        {
            a: aikit.array([1]),
            b: aikit.array([2])
        }
        """
        return self._static_inplace_update(
            self,
            val,
            ensure_in_backend=ensure_in_backend,
            keep_input_dtype=keep_input_dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_inplace_decrement(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        val: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.inplace_decrement. This
        method simply wraps the function, and so the docstring for
        aikit.inplace_decrement also applies to this method with minimal changes.

        Parameters
        ----------
        x
            The input array to be decremented by the defined value.
        val
            The value of decrement.
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
            The array following an in-place decrement.

        Examples
        --------
        Decrement by a value

        >>> x = aikit.Container(a=aikit.array([0.5, -5., 30.]),b=aikit.array([0., -25., 50.]))
        >>> y = aikit.inplace_decrement(x, 1.5)
        >>> print(y)
        {
            a: aikit.array([-1., -6.5, 28.5]),
            b: aikit.array([-1.5, -26.5, 48.5])
        }

        Decrement by a Container

        >>> x = aikit.Container(a=aikit.array([0., 15., 30.]), b=aikit.array([0., 25., 50.]))
        >>> y = aikit.Container(a=aikit.array([0., 15., 30.]), b=aikit.array([0., 25., 50.]))
        >>> z = aikit.inplace_decrement(x, y)
        >>> print(z)
        {
            a: aikit.array([0., 0., 0.]),
            b: aikit.array([0., 0., 0.])
        }

        >>> x = aikit.Container(a=aikit.array([3., 7., 10.]), b=aikit.array([0., 75., 5.5]))
        >>> y = aikit.Container(a=aikit.array([2., 5.5, 7.]), b=aikit.array([0., 25., 2.]))
        >>> z = aikit.inplace_decrement(x, y)
        >>> print(z)
        {
            a: aikit.array([1., 1.5, 3.]),
            b: aikit.array([0., 50., 3.5])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "inplace_decrement",
            x,
            val,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def inplace_decrement(
        self: aikit.Container,
        val: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.inplace_decrement. This
        method simply wraps the function, and so the docstring for
        aikit.inplace_decrement also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container to apply an in-place decrement.
        val
            The value of decrement.
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

        Returns
        -------
        ret
            A container with the array following the in-place decrement.

        Examples
        --------
        Using :class:`aikit.Container` instance method:
        >>> x = aikit.Container(a=aikit.array([-6.7, 2.4, -8.5]),
        ...                   b=aikit.array([1.5, -0.3, 0]),
        ...                   c=aikit.array([-4.7, -5.4, 7.5]))
        >>> y = x.inplace_decrement(2)
        >>> print(y)
        {
            a: aikit.array([-8.7, 0.4, -10.5]),
            b: aikit.array([-0.5, -2.3, -2]),
            c: aikit.array([-6.7, -7.4, 5.5])
        }
        """
        return self._static_inplace_decrement(
            self,
            val,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_inplace_increment(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        val: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.inplace_increment. This
        method simply wraps the function, and so the docstring for
        aikit.inplace_increment also applies to this method with minimal changes.

        Parameters
        ----------
        x
            The input array to be incremented by the defined value.
        val
            The value of increment.
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

        Returns
        -------
        ret
            The array following an in-place increment.

        Examples
        --------
        Increment by a value

        >>> x = aikit.Container(a=aikit.array([0.5, -5., 30.]),b=aikit.array([0., -25., 50.]))
        >>> y = aikit.inplace_increment(x, 1.5)
        >>> print(y)
        {
            a: aikit.array([2., -3.5, 31.5]),
            b: aikit.array([1.5, -23.5, 51.5])
        }

        Increment by a Container

        >>> x = aikit.Container(a=aikit.array([0., 15., 30.]), b=aikit.array([0., 25., 50.]))
        >>> y = aikit.Container(a=aikit.array([0., 15., 30.]), b=aikit.array([0., 25., 50.]))
        >>> z = aikit.inplace_increment(x, y)
        >>> print(z)
        {
            a: aikit.array([0., 30., 60.]),
            b: aikit.array([0., 50., 100.])
        }

        >>> x = aikit.Container(a=aikit.array([3., 7., 10.]), b=aikit.array([0., 75., 5.5]))
        >>> y = aikit.Container(a=aikit.array([2., 5.5, 7.]), b=aikit.array([0., 25., 2.]))
        >>> z = aikit.inplace_increment(x, y)
        >>> print(z)
        {
            a: aikit.array([5., 12.5, 17.]),
            b: aikit.array([0., 100., 7.5])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "inplace_increment",
            x,
            val,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def inplace_increment(
        self: aikit.Container,
        val: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.inplace_increment. This
        method wraps the function, and so the docstring for
        aikit.inplace_increment also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container to apply an in-place increment.
        val
            The value of increment.
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

        Returns
        -------
        ret
            A container with the array following the in-place increment.

        Examples
        --------
        Using :class:`aikit.Container` instance method:
        >>> x = aikit.Container(a=aikit.array([-6.7, 2.4, -8.5]),
        ...                   b=aikit.array([1.5, -0.3, 0]),
        ...                   c=aikit.array([-4.7, -5.4, 7.5]))
        >>> y = x.inplace_increment(2)
        >>> print(y)
        {
            a: aikit.array([-4.7, 4.4, -6.5]),
            b: aikit.array([3.5, 1.7, 2.]),
            c: aikit.array([-2.7, -3.4, 9.5])
        }
        """
        return self._static_inplace_increment(
            self,
            val,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_assert_supports_inplace(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.assert_supports_inplace.
        This method simply wraps the function, and so the docstring for
        aikit.assert_supports_inplace also applies to this method with minimal
        changes.

        Parameters
        ----------
        x
            input container to check for inplace support for.
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

        Returns
        -------
        ret
            True if support, raises exception otherwise`
        """
        return ContainerBase.cont_multi_map_in_function(
            "assert_supports_inplace",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def assert_supports_inplace(
        self: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of
        aikit.assert_supports_inplace. This method simply wraps the function, and
        so the docstring for aikit.assert_supports_inplace also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            input container to check for inplace support for.
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

        Returns
        -------
        ret
            An aikit.Container instance of True bool values if nodes of the Container \
            support in-place operations, raises AikitBackendException otherwise

        Examples
        --------
        >>> aikit.set_backend("numpy")
        >>> x = aikit.Container(a=aikit.array([5, 6]), b=aikit.array([7, 8]))
        >>> print(x.assert_supports_inplace())
        {
            a: True,
            b: True
        }
        """
        return self._static_assert_supports_inplace(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_all_equal(
        x1: aikit.Container,
        *xs: Union[Iterable[Any], aikit.Container],
        equality_matrix: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.all_equal. This method
        simply wraps the function, and so the docstring for aikit.all_equal also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input container.
        xs
            arrays or containers to be compared to ``x1``.
        equality_matrix
            Whether to return a matrix of equalities comparing each input with every
            other. Default is ``False``.
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

        Returns
        -------
        ret
            Boolean, whether or not the inputs are equal, or matrix container of
            booleans if equality_matrix=True is set.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x1 = aikit.Container(a=aikit.array([1, 0, 1, 1]), b=aikit.array([1, -1, 0, 0]))
        >>> x2 = aikit.array([1, 0, 1, 1])
        >>> y = aikit.Container.static_all_equal(x1, x2, equality_matrix= False)
        >>> print(y)
        {
            a: aikit.array([True, True, True, True]),
            b: aikit.array([True, False, False, False])
        }

        With multiple :class:`aikit.Container` input:

        >>> x1 = aikit.Container(a=aikit.array([1, 0, 1, 1]),
        ...                    b=aikit.native_array([1, 0, 0, 1]))
        >>> x2 = aikit.Container(a=aikit.native_array([1, 0, 1, 1]),
        ...                    b=aikit.array([1, 0, -1, -1]))
        >>> y = aikit.Container.static_all_equal(x1, x2, equality_matrix= False)
        >>> print(y)
        {
            a: aikit.array([True, True, True, True]),
            b: aikit.array([True, True, False, False])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "all_equal",
            x1,
            *xs,
            equality_matrix=equality_matrix,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def all_equal(
        self: aikit.Container,
        *xs: Union[Iterable[Any], aikit.Container],
        equality_matrix: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.all_equal. This method
        simply wraps the function, and so the docstring for aikit.all_equal also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        xs
            arrays or containers to be compared to ``self``.
        equality_matrix
            Whether to return a matrix of equalities comparing each input with every
            other. Default is ``False``.
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

        Returns
        -------
        ret
            Boolean, whether or not the inputs are equal, or matrix container of
            booleans if equality_matrix=True is set.

        Examples
        --------
        With one :class:`aikit.Container` instances:

        >>> x1 = aikit.Container(a=aikit.array([1, 0, 1, 1]), b=aikit.array([1, -1, 0, 0]))
        >>> x2 = aikit.array([1, 0, 1, 1])
        >>> y = x1.all_equal(x2, equality_matrix= False)
        >>> print(y)
        {
            a: True,
            b: False
        }

        >>> x1 = aikit.Container(a=aikit.array([1, 0, 1, 1]), b=aikit.array([1, -1, 0, 0]))
        >>> x2 = aikit.array([1, 0, 1, 1])
        >>> y = x1.all_equal(x2, equality_matrix= False)
        >>> print(y)
        {
            a: True,
            b: False
        }

        With multiple :class:`aikit.Container` instances:

        >>> x1 = aikit.Container(a=aikit.native_array([1, 0, 0]),
        ...                    b=aikit.array([1, 2, 3]))
        >>> x2 = aikit.Container(a=aikit.native_array([1, 0, 1]),
        ...                    b=aikit.array([1, 2, 3]))
        >>> y = x1.all_equal(x2, equality_matrix= False)
        >>> print(y)
        {
            a: False,
            b: True
        }

        >>> x1 = aikit.Container(a=aikit.native_array([1, 0, 0]),
        ...                    b=aikit.array([1, 2, 3]))
        >>> x2 = aikit.Container(a=aikit.native_array([1, 0, 1]),
        ...                    b=aikit.array([1, 2, 3]))
        >>> y = x1.all_equal(x2, equality_matrix= False)
        >>> print(y)
        {
            a: False,
            b: True
        }
        """
        return self._static_all_equal(
            self,
            *xs,
            equality_matrix=equality_matrix,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_fourier_encode(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        max_freq: Union[float, aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        num_bands: Union[int, aikit.Container] = 4,
        linear: Union[bool, aikit.Container] = False,
        flatten: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.fourier_encode. This
        method simply wraps the function, and so the docstring for
        aikit.fourier_encode also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container to apply fourier_encode.
        max_freq
            The maximum frequency of the encoding.
        num_bands
            The number of frequency bands for the encoding. Default is 4.
        linear
            Whether to space the frequency bands linearly as opposed to geometrically.
            Default is ``False``.
        flatten
            Whether to flatten the position dimension into the batch dimension.
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

        Returns
        -------
        ret
            New container with the final dimension expanded of arrays at its leaves,
            and the encodings stored in this channel.

        Examples
        --------
        >>> x = aikit.Container(a = aikit.array([1,2]),
        ...                   b = aikit.array([3,4]))
        >>> y = 1.5
        >>> z = aikit.Container.static_fourier_encode(x, y)
        >>> print(z)
        {
            a: (<classaikit.array.array.Array>shape=[2,9]),
            b: (<classaikit.array.array.Array>shape=[2,9])
        }

        >>> x = aikit.Container(a = aikit.array([3,10]),
        ...                   b = aikit.array([4,8]))
        >>> y = 2.5
        >>> z = aikit.Container.static_fourier_encode(x, y, num_bands=3)
        >>> print(z)
        {
            a: aikit.array([[ 3.0000000e+00, 3.6739404e-16, 3.6739404e-16,
                    3.6739404e-16, -1.0000000e+00, -1.0000000e+00, -1.0000000e+00],
                    [ 1.0000000e+01, -1.2246468e-15, -1.2246468e-15, -1.2246468e-15,
                    1.0000000e+00,  1.0000000e+00,  1.0000000e+00]]),
            b: aikit.array([[ 4.00000000e+00, -4.89858720e-16, -4.89858720e-16,
                    -4.89858720e-16, 1.00000000e+00,  1.00000000e+00,  1.00000000e+00],
                    [ 8.00000000e+00, -9.79717439e-16, -9.79717439e-16, -9.79717439e-16,
                    1.00000000e+00,  1.00000000e+00,  1.00000000e+00]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "fourier_encode",
            x,
            max_freq,
            num_bands=num_bands,
            linear=linear,
            concat=True,
            flatten=flatten,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def fourier_encode(
        self: aikit.Container,
        max_freq: Union[float, aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        num_bands: Union[int, aikit.Container] = 4,
        linear: Union[bool, aikit.Container] = False,
        flatten: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.fourier_encode. This
        method simply wraps the function, and so the docstring for
        aikit.fourier_encode also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container to apply fourier_encode at leaves.
        max_freq
            The maximum frequency of the encoding.
        num_bands
            The number of frequency bands for the encoding. Default is 4.
        linear
            Whether to space the frequency bands linearly as opposed to geometrically.
            Default is ``False``.
        flatten
            Whether to flatten the position dimension into the batch dimension.
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
        dtype
            Data type of the returned array. Default is ``None``.
        out
            Optional output container. Default is ``None``.

        Returns
        -------
        ret
            New container with the final dimension expanded of arrays at its leaves,
            and the encodings stored in this channel.

        Examples
        --------
        >>> x = aikit.Container(a = aikit.array([1,2]),
        ...                   b = aikit.array([3,4]))
        >>> y = 1.5
        >>> z = x.fourier_encode(y)
        >>> print(z)
        {
            a: (<class aikit.data_classes.array.array.Array> shape=[2, 9]),
            b: (<class aikit.data_classes.array.array.Array> shape=[2, 9])
        }

        >>> x = aikit.Container(a = aikit.array([3,10]),
        ...                   b = aikit.array([4,8]))
        >>> y = 2.5
        >>> z = x.fourier_encode(y,num_bands=3)
        >>> print(z)
        {
            a: (<class aikit.data_classes.array.array.Array> shape=[2, 7]),
            b: (<class aikit.data_classes.array.array.Array> shape=[2, 7])
        }
        """
        return self._static_fourier_encode(
            self,
            max_freq,
            num_bands=num_bands,
            linear=linear,
            flatten=flatten,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_gather(
        params: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        indices: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        axis: Union[int, aikit.Container] = -1,
        batch_dims: Union[int, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.gather. This method
        simply wraps the function, and so the docstring for aikit.gather also
        applies to this method with minimal changes.

        Parameters
        ----------
        params
            The container from which to gather values.
        indices
            The container or array which indicates the indices that will be
            gathered along the specified axis.
        axis
            The axis from which the indices will be gathered. Default is ``-1``.
        batch_dims
            optional int, lets you gather different items from each element of a batch.
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
            Optional array, for writing the result to. It must have a shape
            that the inputs broadcast to.


        Returns
        -------
        ret
            New container with the values gathered at the specified indices
            along the specified axis.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a = aikit.array([0., 1., 2.]),
        ...                   b = aikit.array([4., 5., 6.]))
        >>> y = aikit.Container(a = aikit.array([0, 1]),
        ...                   b = aikit.array([1, 2]))
        >>> print(aikit.Container.static_gather(x, y))
        {
            a: aikit.array([0., 1.]),
            b: aikit.array([5., 6.])
        }

        With a mix of :class:`aikit.Array` and :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a = aikit.array([0., 1., 2.]),
        ...                   b = aikit.array([4., 5., 6.]))
        >>> y = aikit.array([0, 1])
        >>> z = aikit.Container.static_gather(x, y)
        >>> print(z)
        {
            a: aikit.array([0., 1.]),
            b: aikit.array([4., 5.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "gather",
            params,
            indices,
            axis=axis,
            batch_dims=batch_dims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def gather(
        self: aikit.Container,
        indices: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        axis: Union[int, aikit.Container] = -1,
        batch_dims: Union[int, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.gather. This method
        simply wraps the function, and so the docstring for aikit.gather also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            The container from which to gather values.
        indices
            The container or array which indicates the indices that will be
            gathered along the specified axis.
        axis
            The axis from which the indices will be gathered. Default is ``-1``.
        batch_dims
            optional int, lets you gather different items from each element of a batch.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is
            False.
        out
            Optional array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            New container with the values gathered at the specified indices
            along the specified axis.

        Examples
        --------
        >>> x = aikit.Container(a = aikit.array([0., 1., 2.]),
        ...                   b = aikit.array([4., 5., 6.]))
        >>> y = aikit.Container(a = aikit.array([0, 1]),
        ...                   b = aikit.array([1, 2]))
        >>> z = x.gather(y)
        >>> print(z)
        {
            a: aikit.array([0., 1.]),
            b: aikit.array([5., 6.])
        }
        """
        return self._static_gather(
            self,
            indices,
            axis=axis,
            batch_dims=batch_dims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_has_nans(
        x: aikit.Container,
        /,
        *,
        include_infs: Union[bool, aikit.Container] = True,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """
        Determine whether arrays in the container contain any nans, as well as infs or
        -infs if specified.

        Parameters
        ----------
        x
            The container to check for nans.
        include_infs
            Whether to include infs and -infs in the check. Default is True.
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

        Returns
        -------
            Whether the container has any nans, applied either leafwise or across the
            entire container.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 2]), b=aikit.array([float('nan'), 2]))
        >>> y = aikit.Container.static_has_nans(x)
        >>> print(y)
        {
            a: false,
            b: true
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "has_nans",
            x,
            include_infs=include_infs,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def has_nans(
        self: aikit.Container,
        /,
        *,
        include_infs: Union[bool, aikit.Container] = True,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """
        Determine whether arrays in the container contain any nans, as well as infs or
        -infs if specified.

        Parameters
        ----------
        include_infs
            Whether to include infs and -infs in the check. Default is True.
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

        Returns
        -------
            Whether the container has any nans, applied across the entire container.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 2]), b=aikit.array([float('nan'), 2]))
        >>> y = x.has_nans()
        >>> print(y)
        {
            a: False,
            b: True
        }
        """
        return self._static_has_nans(
            self,
            include_infs=include_infs,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_scatter_nd(
        indices: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        updates: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        shape: Optional[Union[aikit.Array, aikit.NativeArray, aikit.Container]] = None,
        *,
        reduction: Union[str, aikit.Container] = "sum",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.scatter_nd. This method
        simply wraps the function, and so the docstring for aikit.scatter_nd also
        applies to this method with minimal changes.

        Parameters
        ----------
        indices
            Index array or container.
        updates
            values to update input tensor with
        shape
            The shape of the result. Default is ``None``, in which case tensor argument
            must be provided.
        reduction
            The reduction method for the scatter, one of 'sum', 'min', 'max'
            or 'replace'
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
        ref
            New container of given shape, with the values updated at the indices.

        Examples
        --------
        scatter into an empty array

        >>> indices = aikit.Container(a=aikit.array([[5],[6],[7]]),
        ...                         b=aikit.array([[2],[3],[4]]))
        >>> updates = aikit.Container(a=aikit.array([50, 60, 70]),
        ...                         b=aikit.array([20, 30, 40]))
        >>> shape = aikit.Container(a=aikit.array([10]),
        ...                       b=aikit.array([10]))
        >>> z = aikit.Container.static_scatter_nd(indices, updates, shape=shape)
        >>> print(z)
        {
            a: aikit.array([0, 0, 0, 0, 0, 50, 60, 70, 0, 0]),
            b: aikit.array([0, 0, 20, 30, 40, 0, 0, 0, 0, 0])
        }

        scatter into a container

        >>> indices = aikit.Container(a=aikit.array([[5],[6],[7]]),
        ...          b=aikit.array([[2],[3],[4]]))
        >>> updates = aikit.Container(a=aikit.array([50, 60, 70]),
        ...                         b=aikit.array([20, 30, 40]))
        >>> z = aikit.Container(a=aikit.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ...                   b=aikit.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        >>> aikit.Container.static_scatter_nd(indices, updates,
        ...                                    reduction='replace', out = z)
        >>> print(z)
        {
            a: aikit.array([1, 2, 3, 4, 5, 50, 60, 70, 9, 10]),
            b: aikit.array([1, 2, 20, 30, 40, 6, 7, 8, 9, 10])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "scatter_nd",
            indices,
            updates,
            shape=shape,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def scatter_nd(
        self: aikit.Container,
        updates: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        shape: Optional[Union[aikit.Array, aikit.NativeArray, aikit.Container]] = None,
        *,
        reduction: Union[str, aikit.Container] = "sum",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.scatter_nd. This method
        simply wraps the function, and so the docstring for aikit.scatter_nd also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Index array or container.
        updates
            values to update input tensor with
        shape
            The shape of the result. Default is ``None``, in which case tensor argument
            must be provided.
        reduction
            The reduction method for the scatter, one of 'sum', 'min', 'max'
            or 'replace'
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
            New container of given shape, with the values updated at the indices.

        Examples
        --------
        scatter into an empty container

        >>> indices = aikit.Container(a=aikit.array([[4],[3],[6]]),
        ...                         b=aikit.array([[5],[1],[2]]))
        >>> updates = aikit.Container(a=aikit.array([100, 200, 200]),
        ...                         b=aikit.array([20, 30, 40]))
        >>> shape = aikit.Container(a=aikit.array([10]),
        ...                       b=aikit.array([10]))
        >>> z = indices.scatter_nd(updates, shape=shape)
        >>> print(z)
        {
            a: aikit.array([0, 0, 0, 200, 100, 0, 200, 0, 0, 0]),
            b: aikit.array([0, 30, 40, 0, 0, 20, 0, 0, 0, 0])
        }

        With scatter into a container.

        >>> indices = aikit.Container(a=aikit.array([[5],[6],[7]]),
        ...                         b=aikit.array([[2],[3],[4]]))
        >>> updates = aikit.Container(a=aikit.array([50, 60, 70]),
        ...                         b=aikit.array([20, 30, 40]))
        >>> z = aikit.Container(a=aikit.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ...                   b=aikit.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        >>> indices.scatter_nd(updates,reduction='replace', out = z)
        >>> print(z)
        {
            a: aikit.array([1, 2, 3, 4, 5, 50, 60, 70, 9, 10]),
            b: aikit.array([1, 2, 20, 30, 40, 6, 7, 8, 9, 10])
        }
        """
        return self._static_scatter_nd(
            self,
            updates,
            shape=shape,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_scatter_flat(
        indices: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        updates: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        size: Optional[Union[int, aikit.Container]] = None,
        reduction: Union[str, aikit.Container] = "sum",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.scatter_flat. This method
        simply wraps the function, and so the docstring for aikit.scatter_flat
        also applies to this method with minimal changes.

        Parameters
        ----------
        indices
            Index array or container.
        updates
            values to update input tensor with
        size
            The size of the result. Default is `None`, in which case tensor
            argument out must be provided.
        reduction
            The reduction method for the scatter, one of 'sum', 'min', 'max'
            or 'replace'
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
        ref
            New container of given shape, with the values updated at the indices.
        """
        return ContainerBase.cont_multi_map_in_function(
            "scatter_flat",
            indices,
            updates,
            size=size,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def scatter_flat(
        self: aikit.Container,
        updates: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        size: Optional[Union[int, aikit.Container]] = None,
        reduction: Union[str, aikit.Container] = "sum",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.scatter_flat. This
        method simply wraps the function, and so the docstring for
        aikit.scatter_flat also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Index array or container.
        updates
            values to update input tensor with
        size
            The size of the result. Default is `None`, in which case tensor
            argument out must be provided.
        reduction
            The reduction method for the scatter, one of 'sum', 'min', 'max'
            or 'replace'
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
            New container of given shape, with the values updated at the indices.

        Examples
        --------
        With :class:`aikit.Container` input:
        >>> indices = aikit.Container(a=aikit.array([1, 0, 1, 0, 2, 2, 3, 3]),
        ...                 b=aikit.array([0, 0, 1, 0, 2, 2, 3, 3]))
        >>> updates = aikit.Container(a=aikit.array([9, 2, 0, 2, 3, 2, 1, 8]),
        ...                 b=aikit.array([5, 1, 7, 2, 3, 2, 1, 3]))
        >>> size = 8
        >>> print(aikit.scatter_flat(indices, updates, size=size))
        {
            a: aikit.array([2, 0, 2, 8, 0, 0, 0, 0]),
            b: aikit.array([2, 7, 2, 3, 0, 0, 0, 0])
        }
        """
        return self._static_scatter_flat(
            self,
            updates,
            size=size,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_gather_nd(
        params: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        indices: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        batch_dims: Union[int, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """Gather slices from all container params into a arrays with shape
        specified by indices.

        Parameters
        ----------
        params
            The container from which to gather values.
        indices
            Index array.
        batch_dims
            optional int, lets you gather different items from each element of a batch.
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
            optional array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
            Container object with all sub-array dimensions gathered.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[0., 10., 20.],[30.,40.,50.]]),
        ...                   b=aikit.array([[0., 100., 200.],[300.,400.,500.]]))
        >>> y = aikit.Container(a=aikit.array([1,0]),
        ...                   b=aikit.array([0]))
        >>> print(aikit.Container.static_gather_nd(x, y))
        {
            a: aikit.array(30.),
            b: aikit.array([0., 100., 200.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "gather_nd",
            params,
            indices,
            batch_dims=batch_dims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def gather_nd(
        self: aikit.Container,
        indices: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        batch_dims: Union[int, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.gather_nd. This method
        simply wraps the function, and so the docstring for aikit.gather_nd also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            The container from which to gather values.
        indices
            Index array or container.
        batch_dims
            optional int, lets you gather different items from each element of a batch.
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
            New container of given shape, with the values gathered at the indices.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[[0., 10.], [20.,30.]],
        ...                                [[40.,50.], [60.,70.]]]),
        ...                   b=aikit.array([[[0., 100.], [200.,300.]],
        ...                                [[400.,500.],[600.,700.]]]))
        >>> y = aikit.Container(a=aikit.array([1,0]),
        ...                   b=aikit.array([0]))
        >>> z = x.gather_nd(y)
        >>> print(z)
        {
            a: aikit.array([40., 50.]),
            b: aikit.array([[0., 100.],
                        [200., 300.]])
        }
        """
        return self._static_gather_nd(
            self,
            indices,
            batch_dims=batch_dims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_einops_reduce(
        x: aikit.Container,
        pattern: Union[str, aikit.Container],
        reduction: Union[str, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
        **axes_lengths: Union[Dict[str, int], aikit.Container],
    ) -> aikit.Container:
        """Perform einops reduce operation on each sub array in the container.

        Parameters
        ----------
        x
            input container.
        pattern
            Reduction pattern.
        reduction
            One of available reductions ('min', 'max', 'sum', 'mean', 'prod'), or
            callable.
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
        axes_lengths
            Any additional specifications for dimensions.
        out
            optional array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
            aikit.Container with each array having einops.reduce applied.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[-4.47, 0.93, -3.34],
        ...                                [3.66, 24.29, 3.64]]),
        ...                   b=aikit.array([[4.96, 1.52, -10.67],
        ...                                [4.36, 13.96, 0.3]]))
        >>> reduced = aikit.Container.static_einops_reduce(x, 'a b -> a', 'mean')
        >>> print(reduced)
        {
            a: aikit.array([-2.29333329, 10.53000069]),
            b: aikit.array([-1.39666676, 6.20666695])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "einops_reduce",
            x,
            pattern,
            reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **axes_lengths,
        )

    def einops_reduce(
        self: aikit.Container,
        pattern: Union[str, aikit.Container],
        reduction: Union[str, Callable, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
        **axes_lengths: Union[Dict[str, int], aikit.Container],
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.einops_reduce. This
        method simply wraps the function, and so the docstring for
        aikit.einops_reduce also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container to be reduced.
        pattern
            Reduction pattern.
        reduction
            One of available reductions ('min', 'max', 'sum', 'mean', 'prod'), or
            callable.
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
        axes_lengths
            Any additional specifications for dimensions.

        Returns
        -------
        ret
            New container with einops.reduce having been applied.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[[5, 4, 3],
        ...                                 [11, 2, 9]],
        ...                                [[3, 5, 7],
        ...                                 [9, 7, 1]]]),
        ...                    b=aikit.array([[[9,7,6],
        ...                                  [5,2,1]],
        ...                                 [[4,1,2],
        ...                                  [2,3,6]],
        ...                                 [[1, 9, 6],
        ...                                  [0, 2, 1]]]))
        >>> reduced = x.einops_reduce('a b c -> a b', 'sum')
        >>> print(reduced)
        {
            a: aikit.array([[12, 22],
                        [15, 17]]),
            b: aikit.array([[22, 8],
                        [7, 11],
                        [16, 3]])
        }
        """
        return self._static_einops_reduce(
            self,
            pattern,
            reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **axes_lengths,
        )

    @staticmethod
    def _static_einops_repeat(
        x: aikit.Container,
        pattern: Union[str, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
        **axes_lengths: Union[Dict[str, int], aikit.Container],
    ) -> aikit.Container:
        """Perform einops repeat operation on each sub array in the container.

        Parameters
        ----------
        x
            input container.
        pattern
            Rearrangement pattern.
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
        axes_lengths
            Any additional specifications for dimensions.
        **axes_lengths

        Returns
        -------
            aikit.Container with each array having einops.repeat applied.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[30, 40], [50, 75]]),
        ...                   b=aikit.array([[1, 2], [4, 5]]))
        >>> repeated = aikit.Container.static_einops_repeat(
        ...    x, 'h w -> (tile h) w', tile=2)
        >>> print(repeated)
        {
            a: aikit.array([[30, 40],
                        [50, 75],
                        [30, 40],
                        [50, 75]]),
            b: aikit.array([[1, 2],
                        [4, 5],
                        [1, 2],
                        [4, 5]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "einops_repeat",
            x,
            pattern,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **axes_lengths,
        )

    def einops_repeat(
        self: aikit.Container,
        pattern: Union[str, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
        **axes_lengths: Union[Dict[str, int], aikit.Container],
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.einops_repeat. This
        method simply wraps the function, and so the docstring for
        aikit.einops_repeat also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array or container to be repeated.
        pattern
            Rearrangement pattern.
        axes_lengths
            Any additional specifications for dimensions.
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
            optional output container, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            New container with einops.repeat having been applied.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[30, 40], [50, 75]]),
        ...                   b=aikit.array([[1, 2], [4, 5]]))
        >>> repeated = x.einops_repeat('h w ->  h  (w tile)', tile=2)
        >>> print(repeated)
        {
            a: aikit.array([[30, 30, 40, 40],
                          [50, 50, 75, 75]]),
            b: aikit.array([[1, 1, 2, 2],
                          [4, 4, 5, 5]])
        }
        """
        return self._static_einops_repeat(
            self,
            pattern,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **axes_lengths,
        )

    @staticmethod
    def _static_value_is_nan(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        include_infs: Union[bool, aikit.Container] = True,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.value_is_nan. This method
        simply wraps the function, and so the docstring for aikit.value_is_nan
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        include_infs
            Whether to include infs and -infs in the check. Default is ``True``.
        key_chains
            The key-chains to apply or not apply the method to. Default is
            None.
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
            Boolean as to whether the input value is a nan or not.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([452]), b=aikit.array([float('inf')]))
        >>> y = aikit.Container.static_value_is_nan(x)
        >>> print(y)
        {
            a: False,
            b: True
        }

        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([float('nan')]), b=aikit.array([0]))
        >>> y = aikit.Container.static_value_is_nan(x)
        >>> print(y)
        {
            a: True,
            b: False
        }

        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([float('inf')]), b=aikit.array([22]))
        >>> y = aikit.Container.static_value_is_nan(x, include_infs=False)
        >>> print(y)
        {
            a: False,
            b: False
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "value_is_nan",
            x,
            include_infs=include_infs,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def value_is_nan(
        self: aikit.Container,
        /,
        *,
        include_infs: Union[bool, aikit.Container] = True,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.value_is_nan. This
        method simply wraps the function, and so the docstring for
        aikit.value_is_nan also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        include_infs
            Whether to include infs and -infs in the check. Default is ``True``.
        key_chains
            The key-chains to apply or not apply the method to. Default is
            None.
        to_apply
            If True, the method will be applied to key_chains, otherwise
            key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            Boolean as to whether the input value is a nan or not.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([425]), b=aikit.array([float('nan')]))
        >>> y = x.value_is_nan()
        >>> print(y)
        {
            a: False,
            b: True
        }

        >>> x = aikit.Container(a=aikit.array([float('inf')]), b=aikit.array([0]))
        >>> y = x.value_is_nan()
        >>> print(y)
        {
            a: True,
            b: False
        }

        >>> x = aikit.Container(a=aikit.array([float('inf')]), b=aikit.array([22]))
        >>> y = x.value_is_nan(include_infs=False)
        >>> print(y)
        {
            a: False,
            b: False
        }
        """
        return self._static_value_is_nan(
            self,
            include_infs=include_infs,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_to_numpy(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        copy: Union[bool, aikit.Container] = True,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.to_numpy. This method
        simply wraps the function, and so the docstring for aikit.to_numpy also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        copy
            Whether to copy the input. Default is ``True``.
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

        Returns
        -------
        ret
            a container of numpy arrays copying all the element of the container
            ``self``.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([1, 0, 1, 1]),
        ...                   b=aikit.array([1, -1, 0, 0]))
        >>> y = aikit.Container.static_to_numpy(x)
        >>> print(y)
        {
            a: array([1, 0, 1, 1], dtype=int32),
            b: array([1, -1, 0, 0], dtype=int32)
        }

        >>> x = aikit.Container(a=aikit.array([1., 0., 0., 1.]),
        ...                   b=aikit.native_array([1, 1, -1, 0]))
        >>> y = aikit.Container.static_to_numpy(x)
        >>> print(y)
        {
            a: array([1., 0., 0., 1.], dtype=float32),
            b: array([1, 1, -1, 0], dtype=int32)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "to_numpy",
            x,
            copy=copy,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def to_numpy(
        self: aikit.Container,
        /,
        *,
        copy: Union[bool, aikit.Container] = True,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.to_numpy. This method
        simply wraps the function, and so the docstring for aikit.to_numpy also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        copy
            Whether to copy the input. Default is ``True``.
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

        Returns
        -------
        ret
            a container of numpy arrays copying all the element of the container
            ``self``.

        Examples
        --------
        With one :class:`aikit.Container` instances:

        >>> x = aikit.Container(a=aikit.array([-1, 0, 1]), b=aikit.array([1, 0, 1, 1]))
        >>> y = x.to_numpy()
        >>> print(y)
        {
            a: array([-1, 0, 1], dtype=int32),
            b: array([1, 0, 1, 1], dtype=int32)
        }

        >>> x = aikit.Container(a=aikit.native_array([[-1, 0, 1], [-1, 0, 1], [1, 0, -1]]),
        ...                   b=aikit.native_array([[-1, 0, 0], [1, 0, 1], [1, 1, 1]]))
        >>> y = x.to_numpy()
        >>> print(y)
        {
            a: array([[-1, 0, 1],
                    [-1, 0, 1],
                    [1, 0, -1]], dtype=int32),
            b: array([[-1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 1]], dtype=int32)
        }
        """
        return self._static_to_numpy(
            self,
            copy=copy,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_to_scalar(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.to_scalar. This method
        simply wraps the function, and so the docstring for aikit.to_scalar also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
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

        Returns
        -------
        ret
            a container of scalar values copying all the element of the container
            ``x``.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([-1]), b=aikit.array([3]))
        >>> y = aikit.Container.static_to_scalar(x)
        >>> print(y)
        {
            a: -1,
            b: 3
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "to_scalar",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def to_scalar(
        self: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.to_scalar. This method
        simply wraps the function, and so the docstring for aikit.to_scalar also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
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

        Returns
        -------
        ret
            a container of scalar values copying all the element of the container
            ``self``.

        Examples
        --------
        With one :class:`aikit.Container` instance:


        >>> x = aikit.Container(a=aikit.array([1]), b=aikit.array([0]),
        ...                   c=aikit.array([-1]))
        >>> y = x.to_scalar()
        >>> print(y)
        {
            a: 1,
            b: 0,
            c: -1
        }
        """
        return self._static_to_scalar(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_to_list(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.to_list. This method
        simply wraps the function, and so the docstring for aikit.to_list also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
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

        Returns
        -------
        ret
            A container with list representation of the leave arrays.

        Examples
        --------
        With one :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([0, 1, 2]))
        >>> y = aikit.Container.static_to_list(x)
        >>> print(y)
        {a:[0,1,2]}
        """
        return ContainerBase.cont_multi_map_in_function(
            "to_list",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def to_list(
        self: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.to_list. This method
        simply wraps the function, and so the docstring for aikit.to_list also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
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

        Returns
        -------
        ret
            A container with list representation of the leave arrays.

        Examples
        --------
        With one :class:`aikit.Container` instances:


        >>> x = aikit.Container(a=aikit.array([0, 1, 2]))
        >>> y = x.to_list()
        >>> print(y)
        {a:[0,1,2]}
        """
        return self._static_to_list(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_stable_divide(
        numerator: aikit.Container,
        denominator: Union[Number, aikit.Array, aikit.Container],
        /,
        *,
        min_denominator: Optional[
            Union[Number, aikit.Array, aikit.NativeArray, aikit.Container]
        ] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.stable_divide. This
        method simply wraps the function, and so the docstring for
        aikit.stable_divide also applies to this method with minimal changes.

        Parameters
        ----------
        numerator
            Container of the numerators of the division.
        denominator
            Container of the denominators of the division.
        min_denominator
            Container of the minimum denominator to use,
            use global aikit.min_denominator by default.
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

        Returns
        -------
        ret
            A container of elements containing the new items following the numerically
            stable division.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.asarray([10., 15.]), b=aikit.asarray([20., 25.]))
        >>> y = aikit.Container.stable_divide(x, 0.5)
        >>> print(y)
        {
            a: aikit.array([20., 30.]),
            b: aikit.array([40., 50.])
        }

        >>> x = aikit.Container(a=1, b=10)
        >>> y = aikit.asarray([4, 5])
        >>> z = aikit.Container.stable_divide(x, y)
        >>> print(z)
        {
            a: aikit.array([0.25, 0.2]),
            b: aikit.array([2.5, 2.])
        }

        >>> x = aikit.Container(a=1, b=10)
        >>> y = np.array((4.5, 9))
        >>> z = aikit.Container.stable_divide(x, y)
        >>> print(z)
        {
            a: array([0.22222222, 0.11111111]),
            b: array([2.22222222, 1.11111111])
        }


        >>> x = aikit.Container(a=aikit.asarray([1., 2.]), b=aikit.asarray([3., 4.]))
        >>> y = aikit.Container(a=aikit.asarray([0.5, 2.5]), b=aikit.asarray([3.5, 0.4]))
        >>> z = aikit.Container.stable_divide(x, y)
        >>> print(z)
        {
            a: aikit.array([2., 0.8]),
            b: aikit.array([0.857, 10.])
        }

        >>> x = aikit.Container(a=aikit.asarray([1., 2.], [3., 4.]),
        ...                   b=aikit.asarray([5., 6.], [7., 8.]))
        >>> y = aikit.Container(a=aikit.asarray([0.5, 2.5]), b=aikit.asarray([3.5, 0.4]))
        >>> z = aikit.Container.stable_divide(x, y, min_denominator=2)
        >>> print(z)
        {
            a: aikit.array([0.4, 0.444]),
            b: aikit.array([0.909, 2.5])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "stable_divide",
            numerator,
            denominator,
            min_denominator=min_denominator,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def stable_divide(
        self,
        denominator: Union[Number, aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        min_denominator: Optional[
            Union[Number, aikit.Array, aikit.NativeArray, aikit.Container]
        ] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.stable_divide. This
        method simply wraps the function, and so the docstring for
        aikit.stable_divide also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        denominator
            Container of the denominators of the division.
        min_denominator
            Container of the minimum denominator to use,
            use global aikit.min_denominator by default.
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

        Returns
        -------
        ret
            a container of numpy arrays copying all the element of the container
            ``self``.
            A container of elements containing the new items following the numerically
            stable division, using ``self`` as the numerator.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.asarray([3., 6.]), b=aikit.asarray([9., 12.]))
        >>> y = x.stable_divide(5)
        >>> print(y)
        {
            a: aikit.array([0.6, 1.2]),
            b: aikit.array([1.8, 2.4])
        }

        >>> x = aikit.Container(a=aikit.asarray([[2., 4.], [6., 8.]]),
        ...                   b=aikit.asarray([[10., 12.], [14., 16.]]))
        >>> z = x.stable_divide(2, min_denominator=2)
        >>> print(z)
        {
            a: aikit.array([[0.5, 1.],
                  [1.5, 2.]]),
            b: aikit.array([[2.5, 3.],
                  [3.5, 4.]])
        }

        >>> x = aikit.Container(a=aikit.asarray([3., 6.]), b=aikit.asarray([9., 12.]))
        >>> y = aikit.Container(a=aikit.asarray([6., 9.]), b=aikit.asarray([12., 15.]))
        >>> z = x.stable_divide(y)
        >>> print(z)
        {
            a: aikit.array([0.5, 0.667]),
            b: aikit.array([0.75, 0.8])
        }
        """
        return self._static_stable_divide(
            self,
            denominator,
            min_denominator=min_denominator,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_stable_pow(
        base: aikit.Container,
        exponent: Union[Number, aikit.Array, aikit.Container],
        /,
        *,
        min_base: Optional[Union[float, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.stable_pow. This method
        simply wraps the function, and so the docstring for aikit.stable_pow also
        applies to this method with minimal changes.

        Parameters
        ----------
        base
            Container of the base.
        exponent
            Container of the exponent.
        min_base
            The minimum base to use, use global aikit.min_base by default.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise
            key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is
            False.

        Returns
        -------
        ret
            A container of elements containing the new items following the
            numerically stable power.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.asarray([2, 4]), b=aikit.asarray([6, 8]))
        >>> y = aikit.Container.stable_pow(x, 2)
        >>> print(y)
        {
            a: aikit.array([4.00004, 16.00008]),
            b: aikit.array([36.00012, 64.00016])
        }

        >>> x = aikit.Container(a=4, b=8)
        >>> y = aikit.Container.stable_pow(x, 2)
        >>> print(y)
        {
            a: aikit.array(16.00008),
            b: aikit.array(64.00016)
        }

        >>> x = aikit.Container(a=4, b=8)
        >>> y = aikit.asarray([1, 2])
        >>> z = aikit.Container.stable_pow(x, y)
        >>> print(z)
        {
            a: aikit.array([4.00001, 16.00008]),
            b: aikit.array([8.00001, 64.00016])
        }

        >>> x = aikit.Container(a=aikit.asarray([2, 4]), b=aikit.asarray([6, 8]))
        >>> y = aikit.Container(a=4, b=8)
        >>> z = aikit.Container.stable_pow(x, y)
        >>> print(z)
        {
            a: aikit.array([16.00032, 256.00256]),
            b: aikit.array([1679638.395, 16777383.77])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "stable_pow",
            base,
            exponent,
            min_base=min_base,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def stable_pow(
        self,
        exponent: Union[Number, aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        min_base: Optional[Union[float, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.stable_pow. This method
        simply wraps the function, and so the docstring for aikit.stable_pow also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Container of the base.
        exponent
            Container of the exponent.
        min_base
            The minimum base to use, use global aikit.min_base by default.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise
            key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is
            False.

        Returns
        -------
        ret
            A container of elements containing the new items following the
            numerically stable power.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.asarray([2, 4]), b=aikit.asarray([6, 8]))
        >>> y = x.stable_pow(2)
        >>> print(y)
        {
            a: aikit.array([4.00004, 16.00008]),
            b: aikit.array([36.00012, 64.00016])
        }

        >>> x = aikit.Container(a=4, b=8)
        >>> y = x.stable_pow(2)
        >>> print(y)
        {
            a: aikit.array(16.00008),
            b: aikit.array(64.00016)
        }

        >>> x = aikit.Container(a=4, b=8)
        >>> y = aikit.asarray([1, 2])
        >>> z = x.stable_pow(y)
        >>> print(z)
        {
            a: aikit.array([4.00001, 16.00008]),
            b: aikit.array([8.00001, 64.00016])
        }

        >>> x = aikit.Container(a=aikit.asarray([2, 4]), b=aikit.asarray([6, 8]))
        >>> y = aikit.Container(a=4, b=8)
        >>> z = x.stable_pow(y)
        >>> print(z)
        {
            a: aikit.array([16.00032, 256.00256]),
            b: aikit.array([1679638.395, 16777383.77])
        }
        """
        return self._static_stable_pow(
            self,
            exponent,
            min_base=min_base,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_einops_rearrange(
        x: aikit.Container,
        pattern: Union[str, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
        **axes_lengths: Union[Dict[str, int], aikit.Container],
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.einops_rearrange. This
        method simply wraps the function, and so the docstring for
        aikit.einops_rearrange also applies to this method with minimal changes.

        Parameters
        ----------
        pattern
            Rearrangement pattern.
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
        axes_lengths
            Any additional specifications for dimensions.


        Returns
        -------
            aikit.Container with each array having einops.rearrange applied.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([[1, 2, 3],
        ...                                [-4, -5, -6]]),
        ...                 b=aikit.array([[7, 8, 9],
        ...                             [10, 11, 12]]))
        >>> y = aikit.static_einops_rearrange(x, "height width -> width height")
        >>> print(y)
        {
            a: aikit.array([[1, -4],
                        [2, -5],
                        [3, -6]]),
            b: aikit.array([[7, 10],
                        [8, 11],
                        [9, 12]])
        }

        >>> x = aikit.Container(a=aikit.array([[[ 1,  2,  3],
        ...                  [ 4,  5,  6]],
        ...               [[ 7,  8,  9],
        ...                  [10, 11, 12]]]))
        >>> y = aikit.static_einops_rearrange(x, "c h w -> c (h w)")
        >>> print(y)
        {
            a: (<class aikit.array.array.Array> shape=[2, 6])
        }

        >>> x = aikit.Container(a=aikit.array([[1, 2, 3, 4, 5, 6],
        ...               [7, 8, 9, 10, 11, 12]]))
        >>> y = aikit.static_einops_rearrange(x, "c (h w) -> (c h) w", h=2, w=3)
        {
            a: (<class aikit.array.array.Array> shape=[4, 3])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "einops_rearrange",
            x,
            pattern,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **axes_lengths,
        )

    def einops_rearrange(
        self: aikit.Container,
        pattern: Union[str, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
        **axes_lengths: Union[Dict[str, int], aikit.Container],
    ):
        """aikit.Container instance method variant of aikit.einops_rearrange. This
        method simply wraps the function, and so the docstring for
        aikit.einops_rearrange also applies to this method with minimal changes.

        Parameters
        ----------
        pattern
            Rearrangement pattern.
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
        axes_lengths
            Any additional specifications for dimensions.
        **axes_lengths


        Returns
        -------
            aikit.Container with each array having einops.rearrange applied.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[1, 2, 3],
        ...                                [-4, -5, -6]]),
        ...                 b=aikit.array([[7, 8, 9],
        ...                              [10, 11, 12]]))
        >>> y = x.einops_rearrange("height width -> width height")
        >>> print(y)
        {
            a: aikit.array([[1, -4],
                        [2, -5],
                        [3, -6]]),
            b: aikit.array([[7, 10],
                        [8, 11],
                        [9, 12]])
        }

        >>> x = aikit.Container(a=aikit.array([[[ 1,  2,  3],
        ...                  [ 4,  5,  6]],
        ...               [[ 7,  8,  9],
        ...                  [10, 11, 12]]]))
        >>> y = x.einops_rearrange("c h w -> c (h w)")
        >>> print(y)
        {
            a: (<class aikit.data_classes.array.array.Array> shape=[2, 6])
        }

        >>> x = aikit.Container(a=aikit.array([[1, 2, 3, 4, 5, 6],
        ...               [7, 8, 9, 10, 11, 12]]))
        >>> y = x.einops_rearrange("c (h w) -> (c h) w", h=2, w=3)
        >>> print(y)
        {
            a: (<class aikit.data_classes.array.array.Array> shape=[4, 3])
        }
        """
        return self._static_einops_rearrange(
            self,
            pattern,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **axes_lengths,
        )

    @staticmethod
    def _static_clip_matrix_norm(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        max_norm: Union[float, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        p: Union[float, aikit.Container] = 2.0,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.clip_matrix_norm. This
        method simply wraps the function, and so the docstring for
        aikit.clip_matrix_norm also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array containing elements to clip.
        max_norm
            The maximum value of the array norm.
        p
            The p-value for computing the p-norm. Default is 2.
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
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the matrix norm downscaled to the max norm if needed.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([[0., 1., 2.]]),
        ...                   b=aikit.array([[3., 4., 5.]]))
        >>> y = aikit.Container.static_clip_matrix_norm(x, 2.0)
        >>> print(y)
        {
            a: aikit.array([[0., 0.894, 1.79]]),
            b: aikit.array([[0.849, 1.13, 1.41]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "clip_matrix_norm",
            x,
            max_norm,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            p=p,
            out=out,
        )

    def clip_matrix_norm(
        self: aikit.Container,
        max_norm: Union[float, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        p: Union[float, aikit.Container] = 2.0,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.clip_matrix_norm. This
        method simply wraps the function, and so the docstring for
        aikit.clip_matrix_norm also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array containing elements to clip.
        max_norm
            The maximum value of the array norm.
        p
            The p-value for computing the p-norm. Default is 2.
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
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the matrix norm downscaled to the max norm if needed.

        Examples
        --------
        With :class:`aikit.Container` instance method:

        >>> x = aikit.Container(a=aikit.array([[0., 1., 2.]]),
        ...                   b=aikit.array([[3., 4., 5.]]))
        >>> y = x.clip_matrix_norm(2.0, p=1.0)
        >>> print(y)
        {
            a: aikit.array([[0., 1., 2.]]),
            b: aikit.array([[1.2, 1.6, 2.]])
        }
        """
        return self._static_clip_matrix_norm(
            self,
            max_norm,
            p=p,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_supports_inplace_updates(
        x: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.supports_inplace_updates.
        This method simply wraps the function, and so the docstring for
        aikit.supports_inplace_updates also applies to this method with minimal
        changes.

        Parameters
        ----------
        x
            An aikit.Container.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped.
            Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            An aikit.Container instance of bool values.
            True if nodes of x support in-place operations. False otherwise.
        """
        return ContainerBase.cont_multi_map_in_function(
            "supports_inplace_updates",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def supports_inplace_updates(
        self: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of
        aikit.supports_inplace_updates. This method simply wraps the static
        function, and so the docstring for the static variant also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            An aikit.Container whose elements are data types supported by Aikit.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped.
            Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            An aikit.Container instance of bool values.
            True if nodes of the Container support in-place operations. False otherwise.

        Examples
        --------
        With :class:`aikit.Container` input and backend set as `torch`:

        >>> x = aikit.Container(a=aikit.array([5., 6.]), b=aikit.array([7., 8.]))
        >>> ret = x.supports_inplace_updates()
        >>> print(ret)
        {
            a: True,
            b: True
        }

        With :class:`aikit.Container` input and backend set as `jax`:

        >>> x = aikit.Container(a=aikit.array([5.]), b=aikit.array([7.]))
        >>> ret = x.supports_inplace_updates()
        >>> print(ret)
        {
            a: False,
            b: False
        }
        """
        return _ContainerWithGeneral._static_supports_inplace_updates(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_get_num_dims(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        as_array: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.get_num_dims. This
        method simply wraps the function, and so the docstring for
        aikit.get_num_dims also applies to this method with minimal changes.

        Parameters
        ----------
        x
            aikit.Container to infer the number of dimensions for
        as_array
            Whether to return the shape as a array, default False.
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


        Returns
        -------
        ret
            Shape of the array

        Examples
        --------
        >>> x = aikit.Container(b = aikit.asarray([[0.,1.,1.],[1.,0.,0.],[8.,2.,3.]]))
        >>> aikit.Container.static_get_num_dims(x)
        {
            b: 2
        }
        >>> x = aikit.Container(b = aikit.array([[[0,0,0],[0,0,0],[0,0,0]]
        ...                                    [[0,0,0],[0,0,0],[0,0,0]],
        ...                                    [[0,0,0],[0,0,0],[0,0,0]]]))
        >>> aikit.Container.static_get_num_dims(x)
        {
            b: 3
        }
        >>> x = aikit.Container(b = aikit.array([[[0,0,0],[0,0,0],[0,0,0]],
        ...                                    [[0,0,0],[0,0,0],[0,0,0]]]),
        ...                                    c = aikit.asarray([[0.,1.,1.],[8.,2.,3.]]))
        >>> aikit.Container.static_get_num_dims(x)
        {
            b: 3,
            c: 2
        }
        >>> aikit.Container.static_get_num_dims(x, as_array=True)
        {
            b: aikit.array(3),
            c: aikit.array(2)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "get_num_dims",
            x,
            as_array=as_array,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def get_num_dims(
        self: aikit.Container,
        /,
        *,
        as_array: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.get_num_dims. This
        method simply wraps the function, and so the docstring for
        aikit.get_num_dims also applies to this method with minimal changes.

        Parameters
        ----------
        self
            aikit.Container to infer the number of dimensions for
        as_array
            Whether to return the shape as a array, default False.
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


        Returns
        -------
        ret
            Shape of the array

        Examples
        --------
        >>> a = aikit.Container(b = aikit.asarray([[0.,1.,1.],[1.,0.,0.],[8.,2.,3.]]))
        >>> a.get_num_dims()
        {
            b: 2
        }
        >>> a = aikit.Container(b = aikit.array([[[0,0,0],[0,0,0],[0,0,0]],
        ...                                    [[0,0,0],[0,0,0],[0,0,0]],
        ...                                    [[0,0,0],[0,0,0],[0,0,0]]]))
        >>> a.get_num_dims()
        {
            b: 3
        }
        >>> a = aikit.Container(b = aikit.array([[[0,0,0],[0,0,0],[0,0,0]],
        ...                                    [[0,0,0],[0,0,0],[0,0,0]]]),
        ...                                    c = aikit.asarray([[0.,1.,1.],[8.,2.,3.]]))
        >>> a.get_num_dims()
        {
            b: 3,
            c: 2
        }
        >>> a.get_num_dims(as_array=True)
        {
            b: aikit.array(3),
            c: aikit.array(2)
        }
        """
        return _ContainerWithGeneral._static_get_num_dims(
            self,
            as_array=as_array,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_array_equal(
        x0: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        x1: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.array_equal. This
        method simply wraps the function, and so the docstring for
        aikit.array_equal also applies to this method with minimal changes.

        Parameters
        ----------
        x0
            The first input container to compare.
        x1
            The second input container to compare.
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


        Returns
        -------
        ret
            A boolean container indicating whether the two containers are
            equal at each level.

        Examples
        --------
        >>> a = aikit.array([[0., 1.], [1. ,0.]])
        >>> b = aikit.array([[-2., 1.], [1. ,2.]])
        >>> c = aikit.array([[0., 1.], [1. ,0.]])
        >>> d = aikit.array([[2., 1.], [1. ,2.]])
        >>> a0 = aikit.Container(a = a, b = b)
        >>> a1 = aikit.Container(a = c, b = d)
        >>> y = aikit.Container.static_array_equal(a0, a1)
        >>> print(y)
        {
            a: true,
            b: false
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "array_equal",
            x0,
            x1,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def array_equal(
        self: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.array_equal. This
        method simply wraps the function, and so the docstring for
        aikit.array_equal also applies to this method with minimal changes.

        Parameters
        ----------
        self
            The first input container to compare.
        x
            The second input container to compare.
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


        Returns
        -------
        ret
            A boolean container indicating whether the two containers are
            equal at each level.

        Examples
        --------
        >>> a = aikit.array([[0., 1.], [1. ,0.]])
        >>> b = aikit.array([[-2., 1.], [1. ,2.]])
        >>> c = aikit.array([[0., 1.], [1. ,0.]])
        >>> d = aikit.array([[2., 1.], [1. ,2.]])
        >>> a1 = aikit.Container(a = a, b = b)
        >>> a2 = aikit.Container(a = c, b = d)
        >>> y = a1.array_equal(a2)
        >>> print(y)
        {
            a: True,
            b: False
        }

        >>> x1 = aikit.Container(a=aikit.native_array([1, 0, 0]),
                               b=aikit.array([1, 2, 3]))
        >>> x2 = aikit.Container(a=aikit.native_array([1, 0, 1]),
                               b=aikit.array([1, 2, 3]))
        >>> y = x1.array_equal(x2)
        >>> print(y)
        {
            a: False,
            b: True
        }
        """
        return _ContainerWithGeneral._static_array_equal(
            self,
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_isin(
        element: aikit.Container,
        test_elements: aikit.Container,
        /,
        *,
        assume_unique: Union[bool, aikit.Container] = False,
        invert: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """Container instance method variant of aikit.isin. This method simply
        wraps the function, and so the docstring for aikit.isin also applies to
        this method with minimal changes.

        Parameters
        ----------
        element
            input container
        test_elements
            values against which to test for each input element
        assume_unique
            If True, assumes both elements and test_elements contain unique elements,
            which can speed up the calculation. Default value is False.
        invert
            If True, inverts the boolean return array, resulting in True values for
            elements not in test_elements. Default value is False.

        Returns
        -------
        ret
            output a boolean container of the same shape as elements that is True for
            elements in test_elements and False otherwise.

        Examples
        --------
        >>> x = aikit.Container(a=[[10, 7, 4], [3, 2, 1]],\
                              b=[3, 2, 1, 0])
        >>> y = aikit.Container(a=[1, 2, 3],\
                              b=[1, 0, 3])
        >>> aikit.Container.static_isin(x, y)
        aikit.Container(a=[[False, False, False], [ True,  True,  True]],\
                      b=[ True, False,  True])

        >>> aikit.Container.static_isin(x, y, invert=True)
        aikit.Container(a=[[ True,  True,  True], [False, False, False]],\
                      b=[False,  True, False])
        """
        return ContainerBase.cont_multi_map_in_function(
            "isin", element, test_elements, assume_unique=assume_unique, invert=invert
        )

    def isin(
        self: aikit.Container,
        test_elements: aikit.Container,
        /,
        *,
        assume_unique: Union[bool, aikit.Container] = False,
        invert: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """Container instance method variant of aikit.isin. This method simply
        wraps the function, and so the docstring for aikit.isin also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array
        test_elements
            values against which to test for each input element
        assume_unique
            If True, assumes both elements and test_elements contain unique elements,
            which can speed up the calculation. Default value is False.
        invert
            If True, inverts the boolean return array, resulting in True values for
            elements not in test_elements. Default value is False.

        Returns
        -------
        ret
            output a boolean array of the same shape as elements that is True for
            elements in test_elements and False otherwise.

        Examples
        --------
        >>> x = aikit.Container(a=[[10, 7, 4], [3, 2, 1]],\
                                b=[3, 2, 1, 0])
        >>> y = aikit.Container(a=[1, 2, 3],\
                                b=[1, 0, 3])
        >>> x.isin(y)
        aikit.Container(a=[[False, False, False], [ True,  True,  True]],\
                        b=[ True, False,  True])
        """
        return self.static_isin(
            self, test_elements, assume_unique=assume_unique, invert=invert
        )

    @staticmethod
    def static_itemsize(
        x: aikit.Container,
        /,
    ) -> aikit.Container:
        """Container instance method variant of aikit.itemsize. This method
        simply wraps the function, and so the docstring for aikit.itemsize also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
           The input container.

        Returns
        -------
        ret
            Integers specifying the element size in bytes.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1,2,3], dtype=aikit.float64),\
                                b=aikit.array([1,2,3], dtype=aikit.complex128))
        >>> aikit.itemsize(x)
        aikit.Container(a=8, b=16)
        """
        return ContainerBase.cont_multi_map_in_function("itemsize", x)

    def itemsize(
        self: aikit.Container,
        /,
    ) -> aikit.Container:
        """Container instance method variant of aikit.itemsize. This method
        simply wraps the function, and so the docstring for aikit.itemsize also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
           The input container.

        Returns
        -------
        ret
            Integers specifying the element size in bytes.
        """
        return self.static_itemsize(self)

    @staticmethod
    def static_strides(
        x: aikit.Container,
        /,
    ) -> aikit.Container:
        """Container instance method variant of aikit.strides. This method simply
        wraps the function, and so the docstring for aikit.strides also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
           The input container.

        Returns
        -------
        ret
            A tuple containing the strides.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[1, 5, 9], [2, 6, 10]]),\
                                b=aikit.array([[1, 2, 3, 4], [5, 6, 7, 8]]))
        >>> aikit.strides(x)
        aikit.Container(a=(4, 12), b=(16, 4))
        """
        return ContainerBase.cont_multi_map_in_function("strides", x)

    def strides(
        self: aikit.Container,
        /,
    ) -> aikit.Container:
        """Container instance method variant of aikit.strides. This method simply
        wraps the function, and so the docstring for aikit.strides also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
           The input container.

        Returns
        -------
        ret
            A tuple containing the strides.
        """
        return self.static_strides(self)

    @staticmethod
    def _static_exists(
        x: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.exists. This method
        simply wraps the function, and so the docstring for aikit.exists also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            The input container.
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

        Returns
        -------
        ret
            A boolean container detailing if any of the leaf nodes are None.
            True if not None, False if None.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0,4,5]), b=aikit.array([2,2,0]))
        >>> y = x._static_exists(x)
        >>> print(y)
        { a: True, b: True }

        >>> x = aikit.Container(a=[1,2], b=None)
        >>> y = x._static_exists(x)
        >>> print(y)
        { a: True, b: False }

        >>> x = aikit.Container(a={"d": 1, "c": 3}, b={"d": 20, "c": None})
        >>> y = x._static_exists(x)
        >>> print(y)
        { a: { c: True, d: True }, b: { c: False, d: True } }
        """
        return ContainerBase.cont_multi_map_in_function(
            "exists",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def exists(
        self: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.exists. This method
        simply wraps the function, and so the docstring for aikit.exists also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            The input container.
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

        Returns
        -------
        ret
            A boolean container detailing if any of the leaf nodes are None.
            True if not None, False if None.

        Examples
        --------
        >>> x = aikit.Container(a=[1,2,3,4], b=[])
        >>> y = x.exists()
        >>> print(y)
        { a: True, b: True }

        >>> x = aikit.Container(a=None, b=[1,2])
        >>> y = x.exists()
        >>> print(y)
        { a: False, b: True }

        >>> x = aikit.Container(a={"d": 1, "c": 3}, b=None)
        >>> y = x.exists()
        >>> print(y)
        { a: { c: True, d: True }, b: False }
        """
        return self._static_exists(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )
