# global
from typing import Optional, Union, List, Dict, Tuple, Sequence
from numbers import Number

# local
import aikit
from aikit.data_classes.container.base import ContainerBase


class _ContainerWithElementWiseExperimental(ContainerBase):
    @staticmethod
    def static_amax(
        x: aikit.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], aikit.Container]] = None,
        keepdims: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.amax. This method simply
        wraps the function, and so the docstring for aikit.amax also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued data type.
        axis
            axis or axes along which maximum values must be computed.
            By default, the maximum value must be computed over the
            entire array. If a tuple of integers, maximum values must
            be computed over multiple axes. Default: ``None``.
        keepdims
            optional boolean, if ``True``, the reduced axes
            (dimensions) must be included in the result as singleton
            dimensions, and, accordingly, the result must be
            compatible with the input array
            (see `broadcasting<https://data-apis.org/array-api/
            latest/API_specification/
            broadcasting.html#broadcasting>`_).
            Otherwise, if ``False``, the reduced axes (dimensions)
            must not be included in the result.
            Default: ``False``.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was
            not applied. Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            container, if the maximum value was computed over the entire array,
            a zero-dimensional array containing the maximum value;
            otherwise, a non-zero-dimensional array containing the
            maximum values. The returned array must have the same data type
            as ``x``.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container.static_amax(x)
        >>> print(y)
        {
            a: aikit.array(3),
            b: aikit.array(4)
        }

        >>> x = aikit.Container(a=aikit.array([[1, 2, 3], [-1, 0, 2]]),
        ...                   b=aikit.array([[2, 3, 4], [0, 1, 2]]))
        >>> y = aikit.Container.static_amax(x, axis=1)
        >>> print(y)
        {
            a:aikit.array([3, 2]),
            b:aikit.array([4, 2])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "amax",
            x,
            axis=axis,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def amax(
        self: aikit.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], aikit.Container]] = None,
        keepdims: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.amax. This method
        simply wraps the function, and so the docstring for aikit.amax also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued data type.
        axis
            axis or axes along which maximum values must be computed.
            By default, the maximum value must be computed over the
            entire array. If a tuple of integers, maximum values must
            be computed over multiple axes. Default: ``None``.
        keepdims
            optional boolean, if ``True``, the reduced axes
            (dimensions) must be included in the result as singleton
            dimensions, and, accordingly, the result must be
            compatible with the input array
            (see `broadcasting<https://data-apis.org/array-api/
            latest/API_specification/
            broadcasting.html#broadcasting>`_).
            Otherwise, if ``False``, the reduced axes (dimensions)
            must not be included in the result.
            Default: ``False``.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was
            not applied. Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            container, if the maximum value was computed over the entire array,
            a zero-dimensional array containing the maximum value;
            otherwise, a non-zero-dimensional array containing the
            maximum values. The returned array must have the same data type
            as ``x``.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = x.amax()
        >>> print(y)
        {
            a: aikit.array(3),
            b: aikit.array(4)
        }

        >>> x = aikit.Container(a=aikit.array([[1, 2, 3], [-1, 0, 2]]),
        ...                   b=aikit.array([[2, 3, 4], [0, 1, 2]]))
        >>> y = x.amax(axis=1)
        >>> print(y)
        {
            a:aikit.array([3, 2]),
            b:aikit.array([4, 2])
        }
        """
        return self.static_amax(
            self,
            axis=axis,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_amin(
        x: aikit.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], aikit.Container]] = None,
        keepdims: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.amin. This method simply
        wraps the function, and so the docstring for aikit.amin also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued data type.
        axis
            axis or axes along which minimum values must be computed.
            By default, the minimum value must be computed over the
            entire array. If a tuple of integers, minimum values must
            be computed over multiple axes. Default: ``None``.
        keepdims
            optional boolean, if ``True``, the reduced axes
            (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the
            result must be compatible with the input array
            (see `broadcasting<https://data-apis.org/array-api/latest/
            API_specification/broadcasting.html#broadcasting>`_). Otherwise,
            if ``False``, the reduced axes (dimensions)
            must not be included in the result.
            Default: ``False``.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was
            not applied. Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            container, if the minimum value was computed over the entire array,
            a zero-dimensional array containing the minimum value;
            otherwise, a non-zero-dimensional array containing the
            minimum values. The returned array must have the same data type
            as ``x``.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container.static_amin(x)
        >>> print(y)
        {
            a: aikit.array(1),
            b: aikit.array(2)
        }

        >>> x = aikit.Container(a=aikit.array([[1, 2, 3], [-1, 0, 2]]),
        ...                   b=aikit.array([[2, 3, 4], [0, 1, 2]]))
        >>> y = aikit.Container.static_amin(x, axis=1)
        >>> print(y)
        {
            a:aikit.array([1, -1]),
            b:aikit.array([2, 0])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "amin",
            x,
            axis=axis,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def amin(
        self: aikit.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], aikit.Container]] = None,
        keepdims: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.amin. This method
        simply wraps the function, and so the docstring for aikit.amin also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued data type.
        axis
            axis or axes along which minimum values must be computed.
            By default, the minimum value must be computed over the
            entire array. If a tuple of integers, minimum values must
            be computed over multiple axes. Default: ``None``.
        keepdims
            optional boolean, if ``True``, the reduced axes
            (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the
            result must be compatible with the input array
            (see `broadcasting<https://data-apis.org/array-api/latest/
            API_specification/broadcasting.html#broadcasting>`_). Otherwise,
            if ``False``, the reduced axes (dimensions)
            must not be included in the result.
            Default: ``False``.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was
            not applied. Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            container, if the minimum value was computed over the entire array,
            a zero-dimensional array containing the minimum value;
            otherwise, a non-zero-dimensional array containing the
            minimum values. The returned array must have the same data type
            as ``x``.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = x.amin()
        >>> print(y)
        {
            a: aikit.array(1),
            b: aikit.array(2)
        }

        >>> x = aikit.Container(a=aikit.array([[1, 2, 3], [-1, 0, 2]]),
        ...                   b=aikit.array([[2, 3, 4], [0, 1, 2]]))
        >>> y = x.amin(axis=1)
        >>> print(y)
        {
            a:aikit.array([1, -1]),
            b:aikit.array([2, 0])
        }
        """
        return self.static_amin(
            self,
            axis=axis,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_sinc(
        x: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.sinc. This method simply
        wraps the function, and so the docstring for aikit.sinc also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container whose elements are each expressed in radians.
            Should have a floating-point data type.
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
            a container containing the sinc of each element in ``x``. The returned
            container must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0.5, 1.5, 2.5]),
        ...                   b=aikit.array([3.5, 4.5, 5.5]))
        >>> y = aikit.Container.static_sinc(x)
        >>> print(y)
        {
            a: aikit.array([0.636, -0.212, 0.127]),
            b: aikit.array([-0.090, 0.070, -0.057])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "sinc",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sinc(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.sinc. This method
        simply wraps the function, and so the docstring for aikit.sinc also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements are each expressed in radians.
            Should have a floating-point data type.
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
            a container containing the sinc of each element in ``self``.
            The returned container must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0.5, 1.5, 2.5]),
        ...                   b=aikit.array([3.5, 4.5, 5.5]))
        >>> y = x.sinc()
        >>> print(y)
        {
            a: aikit.array([0.637,-0.212,0.127]),
            b: aikit.array([-0.0909,0.0707,-0.0579])
        }
        """
        return self.static_sinc(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_fmod(
        x1: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        x2: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.fmod. This method simply
        wraps the function, and so the docstring for aikit.fmod also applies to
        this method with minimal changes.

        Parameters
        ----------
        x1
            container with the first input arrays.
        x2
            container with the second input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise remainder of divisions.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([2, 3, 4]),\
                               b=aikit.array([aikit.nan, 0, aikit.nan]))
        >>> x2 = aikit.Container(a=aikit.array([1, 5, 2]),\
                               b=aikit.array([0, aikit.nan, aikit.nan]))
        >>> aikit.Container.static_fmod(x1, x2)
        {
            a: aikit.array([ 0,  3,  0])
            b: aikit.array([ nan,  nan,  nan])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "fmod",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def fmod(
        self: aikit.Container,
        x2: aikit.Container,
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.fmod. This method
        simply wraps the function, and so the docstring for aikit.fmod also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            container with the first input arrays.
        x2
            container with the second input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise remainder of divisions.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([2, 3, 4]),\
                               b=aikit.array([aikit.nan, 0, aikit.nan]))
        >>> x2 = aikit.Container(a=aikit.array([1, 5, 2]),\
                               b=aikit.array([0, aikit.nan, aikit.nan]))
        >>> x1.fmod(x2)
        {
            a: aikit.array([ 0,  3,  0])
            b: aikit.array([ nan,  nan,  nan])
        }
        """
        return self.static_fmod(self, x2, out=out)

    @staticmethod
    def static_fmax(
        x1: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        x2: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.fmax. This method simply
        wraps the function, and so the docstring for aikit.fmax also applies to
        this method with minimal changes.

        Parameters
        ----------
        x1
            container with the first input arrays.
        x2
            container with the second input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise maximums.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([2, 3, 4]),\
                               b=aikit.array([aikit.nan, 0, aikit.nan]))
        >>> x2 = aikit.Container(a=aikit.array([1, 5, 2]),\
                               b=aikit.array([0, aikit.nan, aikit.nan]))
        >>> aikit.Container.static_fmax(x1, x2)
        {
            a: aikit.array([ 2.,  5.,  4.])
            b: aikit.array([ 0,  0,  nan])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "fmax",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def fmax(
        self: aikit.Container,
        x2: aikit.Container,
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.fmax. This method
        simply wraps the function, and so the docstring for aikit.fmax also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            container with the first input arrays.
        x2
            container with the second input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise maximums.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([2, 3, 4]),\
                               b=aikit.array([aikit.nan, 0, aikit.nan]))
        >>> x2 = aikit.Container(a=aikit.array([1, 5, 2]),\
                               b=aikit.array([0, aikit.nan, aikit.nan]))
        >>> x1.fmax(x2)
        {
            a: aikit.array([ 2.,  5.,  4.])
            b: aikit.array([ 0,  0,  nan])
        }
        """
        return self.static_fmax(self, x2, out=out)

    @staticmethod
    def static_float_power(
        x1: Union[aikit.Array, aikit.NativeArray, aikit.Container, float, list, tuple],
        x2: Union[aikit.Array, aikit.NativeArray, aikit.Container, float, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.float_power. This method
        simply wraps the function, and so the docstring for aikit.float_power
        also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            container with the base input arrays.
        x2
            container with the exponent input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with base arrays raised to the powers
            of exponents arrays, element-wise .

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([1, 2, 3]),\
                               b=aikit.array([2, 10]))
        >>> x2 = aikit.Container(a=aikit.array([1, 3, 1]), b=0)
        >>> aikit.Container.static_float_power(x1, x2)
        {
            a: aikit.array([1,  8,  3])
            b: aikit.array([1, 1])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "float_power",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def float_power(
        self: aikit.Container,
        x2: aikit.Container,
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.float_power. This
        method simply wraps the function, and so the docstring for
        aikit.float_power also applies to this method with minimal changes.

        Parameters
        ----------
        self
            container with the base input arrays.
        x2
            container with the exponent input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with base arrays raised to the powers
            of exponents arrays, element-wise .

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([1, 2, 3]),\
                               b=aikit.array([2, 10]))
        >>> x2 = aikit.Container(a=aikit.array([1, 3, 1]), b=0)
        >>> x1.float_power(x2)
        {
            a: aikit.array([1,  8,  3])
            b: aikit.array([1, 1])
        }
        """
        return self.static_float_power(self, x2, out=out)

    @staticmethod
    def static_copysign(
        x1: Union[aikit.Array, aikit.NativeArray, aikit.Container, Number],
        x2: Union[aikit.Array, aikit.NativeArray, aikit.Container, Number],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.copysign. This method
        simply wraps the function, and so the docstring for aikit.copysign also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            Container, Array, or scalar to change the sign of
        x2
            Container, Array, or scalar from which the new signs are applied
            Unsigned zeroes are considered positive.
        out
            optional output Container, for writing the result to.

        Returns
        -------
        ret
            x1 with the signs of x2.
            This is a scalar if both x1 and x2 are scalars.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([0,1,2]), b=aikit.array(-1))
        >>> x2 = aikit.Container(a=-1, b=aikit.array(10))
        >>> aikit.Container.static_copysign(x1, x2)
        {
            a: aikit.array([-0., -1., -2.]),
            b: aikit.array(1.)
        }
        >>> aikit.Container.static_copysign(23, x1)
        {
            a: aikit.array([23., 23., 23.]),
            b: aikit.array(-23.)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "copysign",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def copysign(
        self: aikit.Container,
        x2: aikit.Container,
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.copysign. This method
        simply wraps the function, and so the docstring for aikit.copysign also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Container to change the sign of
        x2
            Container from which the new signs are applied
            Unsigned zeroes are considered positive.
        out
            optional output Container, for writing the result to.

        Returns
        -------
        ret
            x1 with the signs of x2.
            This is a scalar if both x1 and x2 are scalars.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([0,1,2]), b=aikit.array(-1))
        >>> x2 = aikit.Container(a=-1, b=aikit.array(10))
        >>> x1.copysign(x2)
        {
            a: aikit.array([-0., -1., -2.]),
            b: aikit.array(1.)
        }
        >>> x1.copysign(-1)
        {
            a: aikit.array([-0., -1., -2.]),
            b: aikit.array(-1.)
        }
        """
        return self.static_copysign(self, x2, out=out)

    @staticmethod
    def static_count_nonzero(
        a: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...], aikit.Container]] = None,
        keepdims: Union[bool, aikit.Container] = False,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.count_nonzero. This
        method simply wraps the function, and so the docstring for
        aikit.count_nonzero also applies to this method with minimal changes.

        Parameters
        ----------
        a
            container with the base input arrays.
        axis
            optional axis or tuple of axes along which to count non-zeros. Default is
            None, meaning that non-zeros will be counted along a flattened
            version of the input array.
        keepdims
            optional, if this is set to True, the axes that are counted are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the input array.
        dtype
            optional output dtype. Default is of type integer.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including number of non-zero values in the array along a
            given axis. Otherwise, container with the total number of non-zero
            values in the array is returned.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[0, 1, 2, 3],[4, 5, 6, 7]]),\
                        b=aikit.array([[[0,1],[2,3]],[[4,5],[6,7]]]))
        >>> aikit.Container.static_count_nonzero(x)
        {
            a: aikit.array(7),
            b: aikit.array(7)
        }
        >>> x = aikit.Container(a=aikit.array([[0, 1, 2, 3],[4, 5, 6, 7]]),\
                        b=aikit.array([[[0,1],[2,3]],[[4,5],[6,7]]]))
        >>> aikit.Container.static_count_nonzero(x, axis=0)
        {
            a: aikit.array([1, 2, 2, 2]),
            b: aikit.array([[1, 2],
                          [2, 2]])
        }
        >>> x = aikit.Container(a=aikit.array([[0, 1, 2, 3],[4, 5, 6, 7]]),\
                        b=aikit.array([[[0,1],[2,3]],[[4,5],[6,7]]]))
        >>> aikit.Container.static_count_nonzero(x, axis=(0,1), keepdims=True)
        {
            a: aikit.array([[7]]),
            b: aikit.array([[[3, 4]]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "count_nonzero",
            a,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def count_nonzero(
        self: aikit.Container,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...], aikit.Container]] = None,
        keepdims: Union[bool, aikit.Container] = False,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.count_nonzero. This
        method simply wraps the function, and so the docstring for
        aikit.count_nonzero also applies to this method with minimal changes.

        Parameters
        ----------
        self
            container with the base input arrays.
        axis
            optional axis or tuple of axes along which to count non-zeros. Default is
            None, meaning that non-zeros will be counted along a flattened
            version of the input array.
        keepdims
            optional, if this is set to True, the axes that are counted are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the input array.
        dtype
            optional output dtype. Default is of type integer.
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
            Default is ``False``
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including number of non-zero values in the array along a
            given axis. Otherwise, container with the total number of non-zero
            values in the array is returned.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[0, 1, 2, 3],[4, 5, 6, 7]]),\
                        b=aikit.array([[[0,1],[2,3]],[[4,5],[6,7]]]))
        >>> x.count_nonzero()
        {
            a: aikit.array(7),
            b: aikit.array(7)
        }
        >>> x = aikit.Container(a=aikit.array([[0, 1, 2, 3],[4, 5, 6, 7]]),\
                        b=aikit.array([[[0,1],[2,3]],[[4,5],[6,7]]]))
        >>> x.count_nonzero(axis=0)
        {
            a: aikit.array([1, 2, 2, 2]),
            b: aikit.array([[1, 2],
                          [2, 2]])
        }
        >>> x = aikit.Container(a=aikit.array([[0, 1, 2, 3],[4, 5, 6, 7]]),\
                        b=aikit.array([[[0,1],[2,3]],[[4,5],[6,7]]]))
        >>> x.count_nonzero(axis=(0,1), keepdims=True)
        {
            a: aikit.array([[7]]),
            b: aikit.array([[[3, 4]]])
        }
        """
        return self.static_count_nonzero(
            self,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_nansum(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        axis: Optional[Union[tuple, int, aikit.Container]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        keepdims: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[Union[aikit.Array, aikit.Container]] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.nansum. This method
        simply wraps the function, and so the docstring for aikit.nansum also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array.
        axis
            Axis or axes along which the sum is computed.
            The default is to compute the sum of the flattened array.
        dtype
            The type of the returned array and of the accumulator in
            which the elements are summed. By default, the dtype of input is used.
        keepdims
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one.
        out
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            A new array holding the result is returned unless out is specified,
            in which it is returned.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> x = aikit.Container(a=aikit.array([[10, 7, 4], [3, 2, 1]]),\
                b=aikit.array([[1, 4, 2], [aikit.nan, aikit.nan, 0]]))
        >>> aikit.Container.static_nansum(x)
        {
            a: 27,
            b: 7.0
        }
        >>> aikit.Container.static_nansum(x, axis=0)
        {
            a: aikit.array([13, 9, 5]),
            b: aikit.array([1., 4., 2.])
        }
        >>> aikit.Container.static_nansum(x, axis=1)
        {
            a: aikit.array([21, 6]),
            b: aikit.array([7., 0.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "nansum",
            x,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def nansum(
        self: aikit.Container,
        /,
        *,
        axis: Optional[Union[tuple, int, aikit.Container]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        keepdims: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.nansum. This method
        simply wraps the function, and so the docstring for aikit.nansum also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container including arrays.
        axis
            Axis or axes along which the sum is computed.
            The default is to compute the sum of the flattened array.
        dtype
            The type of the returned array and of the accumulator in
            which the elements are summed. By default, the dtype of input is used.
        keepdims
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one.
        out
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            A new array holding the result is returned unless out is specified,
            in which it is returned.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> x = aikit.Container(a=aikit.array([[10, 7, 4], [3, 2, 1]]),\
                b=aikit.array([[1, 4, 2], [aikit.nan, aikit.nan, 0]]))
        >>> x.nansum(axis=0)
        {
            a: aikit.array([13, 9, 5]),
            b: aikit.array([1., 4., 2.])
        }
        >>> x.nansum(axis=1)
        {
            a: aikit.array([21, 6]),
            b: aikit.array([7., 0.])
        }
        """
        return self.static_nansum(
            self, axis=axis, dtype=dtype, keepdims=keepdims, out=out
        )

    @staticmethod
    def static_isclose(
        a: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        b: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        rtol: Union[float, aikit.Container] = 1e-05,
        atol: Union[float, aikit.Container] = 1e-08,
        equal_nan: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[Union[aikit.Array, aikit.Container]] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.isclose. This method
        simply wraps the function, and so the docstring for aikit.isclose also
        applies to this method with minimal changes.

        Parameters
        ----------
        a
            Input container containing first input array.
        b
            Input container containing second input array.
        rtol
            The relative tolerance parameter.
        atol
            The absolute tolerance parameter.
        equal_nan
            Whether to compare NaN's as equal. If True, NaN's in a will be
            considered equal to NaN's in b in the output array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            A new array holding the result is returned unless out is specified,
            in which it is returned.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> x = aikit.Container(a=aikit.array([1.0, aikit.nan]),\
                b=aikit.array([1.0, aikit.nan]))
        >>> y = aikit.Container(a=aikit.array([1.0, aikit.nan]),\
                b=aikit.array([1.0, aikit.nan]))
        >>> aikit.Container.static_isclose(x, y)
        {
            a: aikit.array([True, False]),
            b: aikit.array([True, False])
        }
        >>> aikit.Container.static_isclose(x, y, equal_nan=True)
        {
            a: aikit.array([True, True]),
            b: aikit.array([True, True])
        }
        >>> x = aikit.Container(a=aikit.array([1.0, 2.0]),\
                b=aikit.array([1.0, 2.0]))
        >>> y = aikit.Container(a=aikit.array([1.0, 2.001]),\
                b=aikit.array([1.0, 2.0]))
        >>> aikit.Container.static_isclose(x, y, atol=0.0)
        {
            a: aikit.array([True, False]),
            b: aikit.array([True, True])
        }
        >>> aikit.Container.static_isclose(x, y, rtol=0.01, atol=0.0)
        {
            a: aikit.array([True, True]),
            b: aikit.array([True, True])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "isclose",
            a,
            b,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def isclose(
        self: aikit.Container,
        b: aikit.Container,
        /,
        *,
        rtol: Union[float, aikit.Container] = 1e-05,
        atol: Union[float, aikit.Container] = 1e-08,
        equal_nan: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.isclose. This method
        simply wraps the function, and so the docstring for aikit.isclose also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container containing first input array.
        b
            Input container containing second input array.
        rtol
            The relative tolerance parameter.
        atol
            The absolute tolerance parameter.
        equal_nan
            Whether to compare NaN's as equal. If True, NaN's in a will be
            considered equal to NaN's in b in the output array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            A new array holding the result is returned unless out is specified,
            in which it is returned.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> x = aikit.Container(a=aikit.array([1.0, aikit.nan]),\
                b=aikit.array([1.0, aikit.nan]))
        >>> y = aikit.Container(a=aikit.array([1.0, aikit.nan]),\
                b=aikit.array([1.0, aikit.nan]))
        >>> x.isclose(y)
        {
            a: aikit.array([True, False]),
            b: aikit.array([True, False])
        }
        >>> x.isclose(y, equal_nan=True)
        {
            a: aikit.array([True, True]),
            b: aikit.array([True, True])
        }
        >>> x = aikit.Container(a=aikit.array([1.0, 2.0]),\
                b=aikit.array([1.0, 2.0]))
        >>> y = aikit.Container(a=aikit.array([1.0, 2.001]),\
                b=aikit.array([1.0, 2.0]))
        >>> x.isclose(y, atol=0.0)
        {
            a: aikit.array([True, False]),
            b: aikit.array([True, True])
        }
        >>> x.isclose(y, rtol=0.01, atol=0.0)
        {
            a: aikit.array([True, True]),
            b: aikit.array([True, True])
        }
        """
        return self.static_isclose(
            self,
            b,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_signbit(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container, float, int, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.signbit. This method
        simply wraps the function, and so the docstring for aikit.signbit also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container with array-like items.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise signbit of input arrays.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, -2, 3]),\
                               b=-5)
        >>> aikit.Container.static_signbit(x)
        {
            a: aikit.array([False, True, False])
            b: aikit.array([True])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "signbit",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def signbit(
        self: aikit.Container,
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.signbit. This method
        simply wraps the function, and so the docstring for aikit.signbit also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container with array-like items.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise signbit of input arrays.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, -2, 3]),\
                               b=-5)
        >>> x.signbit()
        {
            a: aikit.array([False, True, False])
            b: aikit.array([True])
        }
        """
        return self.static_signbit(self, out=out)

    @staticmethod
    def static_hypot(
        x1: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[Union[aikit.Array, aikit.Container]] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.hypot. This method simply
        wraps the function, and so the docstring for aikit.hypot also applies to
        this method with minimal changes.

        Parameters
        ----------
        x1
            Input container containing first input array.
        x2
            Input container containing second input array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the hypot function computed element-wise

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([2.0]),\
        ...                         b=aikit.array([3.0]))
        >>> y = aikit.Container(a=aikit.array([3.0]),\
                                    b=aikit.array([4.0]))
        >>> aikit.Container.static_hypot(x, y)
        {
            a: aikit.array([3.6055]),
            b: aikit.array([5.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "hypot",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def hypot(
        self: aikit.Container,
        x2: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.hypot. This method
        simply wraps the function, and so the docstring for aikit.hypot also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container containing first input array.
        x2
            Input container containing second input array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the hypot function computed element-wise

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([2.0]),\
        ...                         b=aikit.array([3.0]))
        >>> y = aikit.Container(a=aikit.array([3.0]),\
                                    b=aikit.array([4.0]))
        >>> x.hypot(y)
        {
            a: aikit.array([3.6055]),
            b: aikit.array([5.])
        }
        """
        return self.static_hypot(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_allclose(
        x1: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        rtol: Union[float, aikit.Container] = 1e-05,
        atol: Union[float, aikit.Container] = 1e-08,
        equal_nan: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[Union[aikit.Array, aikit.Container]] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.allclose. This method
        simply wraps the function, and so the docstring for aikit.allclose also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            Input container containing first input array.
        x2
            Input container containing second input array.
        rtol
            The relative tolerance parameter.
        atol
            The absolute tolerance parameter.
        equal_nan
            Whether to compare NaN's as equal. If True, NaN's in x1 will be
            considered equal to NaN's in x2 in the output array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            A new container holding the result is returned unless out is specified,
            in which it is returned.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([1., 2., 3.]),\
        ...                         b=aikit.array([1., 2., 3.]))
        >>> x2 = aikit.Container(a=aikit.array([1., 2., 3.]),\
        ...                         b=aikit.array([1., 2., 3.]))
        >>> y = aikit.Container.static_allclose(x1, x2)
        >>> print(y)
        {
            a: aikit.array(True),
            b: aikit.array(True)
        }

        >>> x1 = aikit.Container(a=aikit.array([1., 2., 3.]),\
        ...                         b=aikit.array([1., 2., 3.]))
        >>> x2 = aikit.Container(a=aikit.array([1., 2., 3.0003]),\
        ...                         b=aikit.array([1.0006, 2., 3.]))
        >>> y = aikit.Container.static_allclose(x1, x2, rtol=1e-3)
        >>> print(y)
        {
            a: aikit.array(True),
            b: aikit.array(True)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "allclose",
            x1,
            x2,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def allclose(
        self: aikit.Container,
        x2: aikit.Container,
        /,
        *,
        rtol: Union[float, aikit.Container] = 1e-05,
        atol: Union[float, aikit.Container] = 1e-08,
        equal_nan: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.allclose. This method
        simply wraps the function, and so the docstring for aikit.allclose also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container containing first input array.
        x2
            Input container containing second input array.
        rtol
            The relative tolerance parameter.
        atol
            The absolute tolerance parameter.
        equal_nan
            Whether to compare NaN's as equal. If True, NaN's in x1 will be
            considered equal to NaN's in x2 in the output array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            A new container holding the result is returned unless out is specified,
            in which it is returned.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([1., 2., 3.]), b=aikit.array([1., 2., 3.]))
        >>> x2 = aikit.Container(a=aikit.array([1., 2., 3.]), b=aikit.array([1., 2., 3.]))
        >>> y = x1.allclose(x2)
        >>> print(y)
        {
            a: aikit.array(True),
            b: aikit.array(True)
        }

        >>> x1 = aikit.Container(a=aikit.array([1., 2., 3.]),
        ...                         b=aikit.array([1., 2., 3.]))
        >>> x2 = aikit.Container(a=aikit.array([1., 2., 3.0003]),
        ...                         b=aikit.array([1.0006, 2., 3.]))
        >>> y = x1.allclose(x2, rtol=1e-3)
        >>> print(y)
        {
            a: aikit.array(True),
            b: aikit.array(True)
        }
        """
        return self.static_allclose(
            self,
            x2,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_diff(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        n: Union[int, aikit.Container] = 1,
        axis: Union[int, aikit.Container] = -1,
        prepend: Optional[
            Union[aikit.Array, aikit.NativeArray, int, list, tuple, aikit.Container]
        ] = None,
        append: Optional[
            Union[aikit.Array, aikit.NativeArray, int, list, tuple, aikit.Container]
        ] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.diff. This method simply
        wraps the function, and so the docstring for aikit.diff also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container with array-like items.
        n
            The number of times values are differenced. If zero, the input is returned
            as-is.
        axis
            The axis along which the difference is taken, default is the last axis.
        prepend,append
            Values to prepend/append to x along given axis prior to performing the
            difference. Scalar values are expanded to arrays with length 1 in the
            direction of axis and the shape of the input array in along all other
            axes. Otherwise the dimension and shape must match x except along axis.
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
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with the n-th discrete difference along
            the given axis.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 2, 4, 7, 0]),
                              b=aikit.array([1, 2, 4, 7, 0]))
        >>> aikit.Container.static_diff(x)
        {
            a: aikit.array([ 1,  2,  3, -7]),
            b: aikit.array([ 1,  2,  3, -7])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "diff",
            x,
            n=n,
            axis=axis,
            prepend=prepend,
            append=append,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def diff(
        self: aikit.Container,
        /,
        *,
        n: Union[int, aikit.Container] = 1,
        axis: Union[int, aikit.Container] = -1,
        prepend: Optional[
            Union[aikit.Array, aikit.NativeArray, int, list, tuple, aikit.Container]
        ] = None,
        append: Optional[
            Union[aikit.Array, aikit.NativeArray, int, list, tuple, aikit.Container]
        ] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.diff. This method
        simply wraps the function, and so the docstring for aikit.diff also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container with array-like items.
        n
            The number of times values are differenced. If zero, the input is returned
            as-is.
        axis
            The axis along which the difference is taken, default is the last axis.
        prepend,append
            Values to prepend/append to x along given axis prior to performing the
            difference. Scalar values are expanded to arrays with length 1 in the
            direction of axis and the shape of the input array in along all other
            axes. Otherwise the dimension and shape must match x except along axis.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with the n-th discrete difference along the
            given axis.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 2, 4, 7, 0]),
                              b=aikit.array([1, 2, 4, 7, 0]))
        >>> x.diff()
        {
            a: aikit.array([1, 2, 3, -7]),
            b: aikit.array([1, 2, 3, -7])
        }
        """
        return self.static_diff(
            self, n=n, axis=axis, prepend=prepend, append=append, out=out
        )

    @staticmethod
    def static_fix(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.fix. This method simply
        wraps the function, and so the docstring for aikit.fix also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container with array items.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise rounding of
            input arrays elements.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([2.1, 2.9, -2.1]),\
                               b=aikit.array([3.14]))
        >>> aikit.Container.static_fix(x)
        {
            a: aikit.array([ 2.,  2., -2.])
            b: aikit.array([ 3.0 ])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "fix",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def fix(
        self: aikit.Container,
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.fix. This method simply
        wraps the function, and so the docstring for aikit.fix also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container with array items.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise rounding of
            input arrays elements.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([2.1, 2.9, -2.1]),\
                               b=aikit.array([3.14]))
        >>> x.fix()
        {
            a: aikit.array([ 2.,  2., -2.])
            b: aikit.array([ 3.0 ])
        }
        """
        return self.static_fix(self, out=out)

    @staticmethod
    def static_nextafter(
        x1: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[Union[aikit.Array, aikit.Container]] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.nextafter. This method
        simply wraps the function, and so the docstring for aikit.nextafter also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            Input container containing first input arrays.
        x2
            Input container containing second input arrays.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the next representable values of
            input container's arrays, element-wise

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([1.0e-50, 2.0e+50]),\
        ...                         b=aikit.array([2.0, 1.0])
        >>> x2 = aikit.Container(a=aikit.array([5.5e-30]),\
        ...                         b=aikit.array([-2.0]))
        >>> aikit.Container.static_nextafter(x1, x2)
        {
            a: aikit.array([1.4013e-45., 3.4028e+38]),
            b: aikit.array([5.5e-30])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "nextafter",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def nextafter(
        self: aikit.Container,
        x2: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.nextafter. This method
        simply wraps the function, and so the docstring for aikit.nextafter also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container containing first input array.
        x2
            Input container containing second input array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the next representable values of
            input container's arrays, element-wise

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([1.0e-50, 2.0e+50]),\
        ...                         b=aikit.array([2.0, 1.0])
        >>> x2 = aikit.Container(a=aikit.array([5.5e-30]),\
        ...                         b=aikit.array([-2.0]))
        >>> x1.nextafter(x2)
        {
            a: aikit.array([1.4013e-45., 3.4028e+38]),
            b: aikit.array([5.5e-30])
        }
        """
        return self.static_nextafter(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_zeta(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        q: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[Union[aikit.Array, aikit.Container]] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.zeta. This method simply
        wraps the function, and so the docstring for aikit.zeta also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            Input container containing first input arrays.
        q
            Input container containing second input arrays.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the zeta function computed element-wise

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([5.0, 3.0]),\
        ...                         b=aikit.array([2.0, 1.0])
        >>> q = aikit.Container(a=aikit.array([2.0]),\
        ...                         b=aikit.array([5.0]))
        >>> aikit.Container.static_zeta(x1, x2)
        {
            a: aikit.array([0.0369, 0.2021]),
            b: aikit.array([0.0006, 0.0244])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "zeta",
            x,
            q,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def zeta(
        self: aikit.Container,
        q: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.zeta. This method
        simply wraps the function, and so the docstring for aikit.zeta also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container containing first input array.
        q
            Input container containing second input array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the zeta function computed element-wise

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([5.0, 3.0]),\
        ...                         b=aikit.array([2.0, 1.0])
        >>> q = aikit.Container(a=aikit.array([2.0]),\
        ...                         b=aikit.array([5.0]))
        >>> x.zeta(q)
        {
            a: aikit.array([0.0369, 0.2021]),
            b: aikit.array([0.0006, 0.0244])
        }
        """
        return self.static_zeta(
            self,
            q,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_gradient(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        spacing: Union[int, list, tuple, aikit.Container] = 1,
        edge_order: Union[int, aikit.Container] = 1,
        axis: Optional[Union[int, list, tuple, aikit.Container]] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "gradient",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            spacing=spacing,
            edge_order=edge_order,
            axis=axis,
        )

    def gradient(
        self: aikit.Container,
        /,
        *,
        spacing: Union[int, list, tuple, aikit.Container] = 1,
        edge_order: Union[int, aikit.Container] = 1,
        axis: Optional[Union[int, list, tuple, aikit.Container]] = None,
    ) -> aikit.Container:
        """Calculate gradient of x with respect to (w.r.t.) spacing.

        Parameters
        ----------
            x
                input array representing outcomes of the function
                spacing
                if not given, indices of x will be used
                if scalar indices of x will be scaled with this value
                if array gradient of x w.r.t. spacing
            edge_order
                1 or 2, for 'frist order' and 'second order' estimation
                of boundary values of gradient respectively.
            axis
                dimension(s) to approximate the gradient over.
                By default, partial gradient is computed in every dimension


        Returns
        -------
        ret
            Array with values computed from gradient function from
            inputs

        Examples
        --------
        >>> coordinates = aikit.Container(
        >>>     a=(aikit.array([-2., -1., 1., 4.]),),
        >>>     b=(aikit.array([2., 1., -1., -4.]),)
        >>> )
        >>> values = aikit.Container(
        >>>     a=aikit.array([4., 1., 1., 16.]),
        >>>     b=aikit.array([4., 1., 1., 16.])
        >>> )
        >>> aikit.gradient(values, spacing=coordinates)
        {
            a: aikit.array([-3., -2., 2., 5.]),
            b: aikit.array([3., 2., -2., -5.])
        }

        >>> values = aikit.Container(
        >>>     a=aikit.array([[1, 2, 4, 8], [10, 20, 40, 80]]),
        >>>     b=aikit.array([[-1, -2, -4, -8], [-10, -20, -40, -80]])
        >>> )
        >>> aikit.gradient(values)
        [{
            a: aikit.array([[9., 18., 36., 72.],
                          [9., 18., 36., 72.]]),
            b: aikit.array([[-9., -18., -36., -72.],
                          [-9., -18., -36., -72.]])
        }, {
            a: aikit.array([[1., 1.5, 3., 4.],
                          [10., 15., 30., 40.]]),
            b: aikit.array([[-1., -1.5, -3., -4.],
                          [-10., -15., -30., -40.]])
        }]

        >>> values = aikit.Container(
        >>>     a=aikit.array([[1, 2, 4, 8], [10, 20, 40, 80]]),
        >>>     b=aikit.array([[-1, -2, -4, -8], [-10, -20, -40, -80]])
        >>> )
        >>> aikit.gradient(values, spacing=2.0)
        [{
            a: aikit.array([[4.5, 9., 18., 36.],
                          [4.5, 9., 18., 36.]]),
            b: aikit.array([[-4.5, -9., -18., -36.],
                          [-4.5, -9., -18., -36.]])
        }, {
            a: aikit.array([[0.5, 0.75, 1.5, 2.],
                          [5., 7.5, 15., 20.]]),
            b: aikit.array([[-0.5, -0.75, -1.5, -2.],
                          [-5., -7.5, -15., -20.]])
        }]

        >>> values = aikit.Container(
        >>>     a=aikit.array([[1, 2, 4, 8], [10, 20, 40, 80]]),
        >>>     b=aikit.array([[-1, -2, -4, -8], [-10, -20, -40, -80]])
        >>> )
        >>> aikit.gradient(values, axis=1)
        {
            a: aikit.array([[1., 1.5, 3., 4.],
                          [10., 15., 30., 40.]]),
            b: aikit.array([[-1., -1.5, -3., -4.],
                          [-10., -15., -30., -40.]])
        }

        >>> values = aikit.Container(
        >>>     a=aikit.array([[1, 2, 4, 8], [10, 20, 40, 80]]),
        >>>     b=aikit.array([[-1, -2, -4, -8], [-10, -20, -40, -80]])
        >>> )
        >>> aikit.gradient(values, spacing = [3., 2.])
        [{
            a: aikit.array([[3., 6., 12., 24.],
                          [3., 6., 12., 24.]]),
            b: aikit.array([[-3., -6., -12., -24.],
                          [-3., -6., -12., -24.]])
        }, {
            a: aikit.array([[0.5, 0.75, 1.5, 2.],
                          [5., 7.5, 15., 20.]]),
            b: aikit.array([[-0.5, -0.75, -1.5, -2.],
                          [-5., -7.5, -15., -20.]])
        }]

        >>> coords = aikit.Container(
        >>>    a=(aikit.array([0, 2]), aikit.array([0, 3, 6, 9])),
        >>>    b=(aikit.array([0, -2]), aikit.array([0, -3, -6, -9]))
        >>>)
        >>> values = aikit.Container(
        >>>     a=aikit.array([[1, 2, 4, 8], [10, 20, 40, 80]]),
        >>>     b=aikit.array([[-1, -2, -4, -8], [-10, -20, -40, -80]])
        >>>)
        >>> aikit.gradient(values, spacing = coords)
        [{
            a: aikit.array([[4.5, 9., 18., 36.],
                          [4.5, 9., 18., 36.]]),
            b: aikit.array([[4.5, 9., 18., 36.],
                          [4.5, 9., 18., 36.]])
        }, {
            a: aikit.array([[0.33333333, 0.5, 1., 1.33333333],
                          [3.33333333, 5., 10., 13.33333333]]),
            b: aikit.array([[0.33333333, 0.5, 1., 1.33333333],
                          [3.33333333, 5., 10., 13.33333333]])
        }]
        """
        return self.static_gradient(
            self, spacing=spacing, edge_order=edge_order, axis=axis
        )

    @staticmethod
    def static_xlogy(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        y: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[Union[aikit.Array, aikit.Container]] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.xlogy. This method simply
        wraps the function, and so the docstring for aikit.xlogy also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            Input container containing first input arrays.
        y
            Input container containing second input arrays.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the next representable values of
            input container's arrays, element-wise

        Examples
        --------
        >>> x = aikit.Container(a=aikit.zeros(3)),\
        ...                         b=aikit.array([1.0, 2.0, 3.0]))
        >>> y = aikit.Container(a=aikit.array([-1.0, 0.0, 1.0]),\
        ...                         b=aikit.array([3.0, 2.0, 1.0]))
        >>> aikit.Container.static_xlogy(x, y)
        {
            a: aikit.array([0.0, 0.0, 0.0]),
            b: aikit.array([1.0986, 1.3863, 0.0000])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "xlogy",
            x,
            y,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def xlogy(
        self: aikit.Container,
        y: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.xlogy. This method
        simply wraps the function, and so the docstring for aikit.xlogy also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container containing first input array.
        y
            Input container containing second input array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the next representable values of
            input container's arrays, element-wise

        Examples
        --------
        >>> x = aikit.Container(a=aikit.zeros(3)),\
        ...                         b=aikit.array([1.0, 2.0, 3.0]))
        >>> y = aikit.Container(a=aikit.array([-1.0, 0.0, 1.0]),\
        ...                         b=aikit.array([3.0, 2.0, 1.0]))
        >>> x.xlogy(y)
        {
            a: aikit.array([0.0, 0.0, 0.0]),
            b: aikit.array([1.0986, 1.3863, 0.0000])
        }
        """
        return self.static_xlogy(
            self,
            y,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_binarizer(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        threshold: Union[float, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """Map the values of the input tensor to either 0 or 1, element-wise,
        based on the outcome of a comparison against a threshold value.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
        threshold
            Values greater than this are
            mapped to 1, others to 0.
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
            Binarized output data
        """
        return ContainerBase.cont_multi_map_in_function(
            "binarizer",
            x,
            threshold=threshold,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def binarizer(
        self: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        *,
        threshold: Union[float, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """Map the values of the input tensor to either 0 or 1, element-wise,
        based on the outcome of a comparison against a threshold value.

        Parameters
        ----------
        threshold
            Values greater than this are
            mapped to 1, others to 0.
        key_chains
            The keychains to apply or not apply the method to. Default is ``None``.
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
            Binarized output data
        """
        return self.static_binarizer(
            self,
            threshold=threshold,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_conj(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.conj. This method simply
        wraps the function, and so the docstring for aikit.conj also applies to
        this method with minimal changes.

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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing output array(s) of the same
            dtype as the input array(s) with the complex conjugates of
            the complex values present in the input array. If x is a
            container of scalar(s) then a container of scalar(s)
            will be returned.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([-1+5j, 0-0j, 1.23j]),
        ...                   b=aikit.array([7.9, 0.31+3.3j, -4.2-5.9j]))
        >>> z = aikit.Container.static_conj(x)
        >>> print(z)
        {
            a: aikit.array([-1-5j, 0+0j, -1.23j]),
            b: aikit.array([7.9, 0.31-3.3j, -4.2+5.9j])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "conj",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def conj(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.conj. This method
        simply wraps the function, and so the docstring for aikit.conj also
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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing output array(s) of the same dtype
            as the input array(s) with the complex conjugates of the
            complex values present in the input array.
            If x is a container of scalar(s) then a container of
            scalar(s) will be returned.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([-1j, 0.335+2.345j, 1.23+7j]),\
                          b=aikit.array([0.0, 1.2+3.3j, 1+0j]))
        >>> x.conj()
        {
            a: aikit.array([1j, 0.335-2345j, 1.23-7j]),
            b: aikit.array([0.0, 1.2-3.3j, 1-0j])
        }
        """
        return self.static_conj(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_ldexp(
        x1: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        x2: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.ldexp. This method simply
        wraps the function, and so the docstring for aikit.ldexp also applies to
        this method with minimal changes.

        Parameters
        ----------
        x1
            The container whose arrays should be multiplied by 2**i.
        x2
            The container whose arrays should be used to multiply x by 2**i.
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
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including x1 * 2**x2.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> x1 = aikit.Container(a=aikit.array([1, 2, 3]), b=aikit.array([1, 5, 10]))
        >>> x2 = aikit.Container(a=aikit.array([1, 2, 3]), b=aikit.array([1, 5, 10]))
        >>> aikit.Container.static_ldexp(x1, x2)
        {
            a: aikit.array([2, 8, 24]),
            b: aikit.array([2, 160, 10240])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "ldexp",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def ldexp(
        self: aikit.Container,
        x2: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.ldexp. This method
        simply wraps the function, and so the docstring for aikit.ldexp also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            The container whose arrays should be multiplied by 2**x2.
        x2
            The container whose arrays should be used to multiply x1 by 2**x2.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including x1 * 2**x2.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> x1 = aikit.Container(a=aikit.array([1, 2, 3]), b=aikit.array([1, 5, 10]))
        >>> x2 = aikit.Container(a=aikit.array([1, 2, 3]), b=aikit.array([1, 5, 10]))
        >>> x1.ldexp(x2)
        {
            a: aikit.array([2, 8, 24]),
            b: aikit.array([2, 160, 10240])
        }
        """
        return self.static_ldexp(self, x2, out=out)

    @staticmethod
    def static_lerp(
        input: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        end: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        weight: Union[aikit.Array, aikit.NativeArray, float, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.lerp. This method simply
        wraps the function, and so the docstring for aikit.lerp also applies to
        this method with minimal changes.

        Parameters
        ----------
        input
            The container whose arrays should be used as parameter: input
        end
            The container whose arrays should be used as parameter: end
        weight
            The container whose arrays or scalar should be used as parameter: weight
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
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including  input + ((end - input) * weight)

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> input = aikit.Container(a=aikit.array([0., 1., 2.]), b=aikit.array([3., 4., 5.]))
        >>> end = aikit.array([10.])
        >>> weight = 1.1
        >>> y = aikit.Container.static_lerp(input, end, weight)
        >>> print(y)
        {
            a: aikit.array([11., 10.90000057, 10.80000019]),
            b: aikit.array([10.70000076, 10.60000038, 10.5])
        }
        >>> input = aikit.Container(a=aikit.array([10.1, 11.1]), b=aikit.array([10, 11]))
        >>> end = aikit.Container(a=aikit.array([5]))
        >>> weight = aikit.Container(a=0.5)
        >>> y = aikit.Container.static_lerp(input, end, weight)
        >>> print(y)
        {
            a: aikit.array([7.55000019, 8.05000019]),
            b: {
                a: aikit.array([7.5, 8.])
            }
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "lerp",
            input,
            end,
            weight,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def lerp(
        self: aikit.Container,
        end: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        weight: Union[aikit.Array, aikit.NativeArray, float, aikit.Container],
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.lerp. This method
        simply wraps the function, and so the docstring for aikit.lerp also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            The container whose arrays should be used as parameter: input
        end
            The container whose arrays should be used as parameter: end
        weight
            The container whose arrays or scalar should be used as parameter: weight
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including  input + ((end - input) * weight)

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> input = aikit.Container(a=aikit.array([1, 2, 3]), b=aikit.array([1, 5, 10]))
        >>> end = aikit.Container(a=aikit.array([10, 10, 10]), b=aikit.array([20, 20, 20]))
        >>> weight = aikit.Container(a=aikit.array(0.5), b=aikit.array([0.4, 0.5, 0.6]))
        >>> input.lerp(end, weight)
        {
            a: aikit.array([5.5, 6., 6.5]),
            b: aikit.array([8.60000038, 12.5, 16.])
        }
        """
        return self.static_lerp(self, end, weight, out=out)

    @staticmethod
    def static_frexp(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.frexp. This method simply
        wraps the function, and so the docstring for aikit.frexp also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            The container whose arrays should be split into mantissa and exponent.
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
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including the mantissa and exponent of x.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> x = aikit.Container(a=aikit.array([1, 2, 3]), b=aikit.array([1, 5, 10]))
        >>> aikit.Container.static_frexp(x)
        {
            a: (aikit.array([0.5, 0.5, 0.75]), aikit.array([1, 1, 2])),
            b: (aikit.array([0.5, 0.625, 0.625]), aikit.array([1, 3, 4]))
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "frexp",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def frexp(
        self: aikit.Container,
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.frexp. This method
        simply wraps the function, and so the docstring for aikit.frexp also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            The container whose arrays should be split into mantissa and exponent.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including the mantissa and exponent of x.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> x = aikit.Container(a=aikit.array([1, 2, 3]),\
                                            b=aikit.array([1, 5, 10]))
        >>> x.frexp()
        {
            a: (aikit.array([0.5, 0.5, 0.75]), aikit.array([1, 1, 2])),
            b: (aikit.array([0.5, 0.625, 0.625]), aikit.array([1, 3, 4]))
        }
        """
        return self.static_frexp(self, out=out)

    @staticmethod
    def static_modf(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.modf. This method simply
        wraps the function, and so the docstring for aikit.modf also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            The container whose arrays should be split into
            the fractional and integral parts.
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
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including the fractional and integral parts of x.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> x = aikit.Container(a=aikit.array([1.2, 2.7, 3.9]),
        >>> b = aikit.array([-1.5, 5.3, -10.7]))
        >>> aikit.Container.static_modf(x)
        {
            a: (aikit.array([0.2, 0.7, 0.9]), aikit.array([1.0, 2.0, 3.0])),
            b: (aikit.array([-0.5, 0.3, -0.7]), aikit.array([-1.0, 5.0, -10.0]))
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "modf",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def modf(
        self: aikit.Container,
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        r"""aikit.Container instance method variant of aikit.modf. This method
        simply wraps the function, and so the docstring for aikit.modf also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            The container whose arrays should be split into
            the fractional and integral parts.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including the fractional and integral parts of x.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> x = aikit.Container(a=aikit.array([1.2, 2.7, 3.9]),
        >>> b = aikit.array([-1.5, 5.3, -10.7]))
        >>> x.modf()
        {
            a: (aikit.array([0.2, 0.7, 0.9]), aikit.array([1.0, 2.0, 3.0])),
            b: (aikit.array([-0.5, 0.3, -0.7]), aikit.array([-1.0, 5.0, -10.0]))
        }
        """
        return self.static_modf(self, out=out)

    @staticmethod
    def static_digamma(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.digamma. This method
        simply wraps the function, and so the docstring for aikit.digamma also
        applies to this method with minimal changes.

        Note
        ----
        The Aikit version only accepts real-valued inputs.

        Parameters
        ----------
        x
            Input container containing input arrays.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the digamma function computed element-wise

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 0.5]),\
        ...                         b=aikit.array([-2.0, 3.0]))
        >>> aikit.Container.static_digamma(x)
        {
            a: aikit.array([-0.57721537, -1.96351004]),
            b: aikit.array([nan, 0.92278427])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "digamma",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def digamma(
        self: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.digamma. This method
        simply wraps the function, and so the docstring for aikit.digamma also
        applies to this method with minimal changes.

        Note
        ----
        The Aikit version only accepts real-valued inputs.

        Parameters
        ----------
        self
            Input container containing input arrays.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the digamma function computed element-wise

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 0.5]), b=aikit.array([2.0, 3.0])
        >>> x.digamma()
        {
            a: aikit.array([-0.5772, -1.9635]),
            b: aikit.array([0.4228, 0.9228])
        }
        """
        return self.static_digamma(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_sparsify_tensor(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        card: Union[int, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.sparsify_tensor. This
        method simply wraps the function, and so the docstring for
        aikit.sparsify_tensor also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container containing input arrays.
        card
            The number of values to keep in each tensor.
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
            Alternate output container in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the sparsified tensor computed element-wise

        Examples
        --------
        >>> x = aikit.Container(
                a=aikit.reshape(aikit.arange(100), (10, 10)),
                b=aikit.reshape(aikit.arange(100), (10, 10)),
            )
        >>> aikit.Container.static_sparsify_tensor(x, 10)
            {
                a: (<class aikit.data_classes.array.array.Array> shape=[10, 10]),
                b: (<class aikit.data_classes.array.array.Array> shape=[10, 10])
            }
        """
        return ContainerBase.cont_multi_map_in_function(
            "sparsify_tensor",
            x,
            card,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sparsify_tensor(
        self: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        card: Union[int, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.sparsify_tensor.

        This method simply wraps the function, and so the docstring for
        aikit.sparsify_tensor also applies to this method with minimal
        changes.
        """
        return self.static_sparsify_tensor(
            self,
            card,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_erfc(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.erfc. This method simply
        wraps the function, and so the docstring for aikit.erfc also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            The container whose array contains real or complex valued argument.
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
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container with values of the complementary error function.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1., 2.]), b=aikit.array([-3., -4.]))
        >>> aikit.Container.static_erfc(x)
        {
            a: aikit.array([0.15729921, 0.00467773]),
            b: aikit.array([1.99997795, 2.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "erfc",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def erfc(
        self: aikit.Container,
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.erfc. This method
        simply wraps the function, and so the docstring for aikit.erfc also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            The container whose array contains real or complex valued argument.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container with values of the complementary error function.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> x = aikit.Container(a=aikit.array([1., 2., 3.]), b=aikit.array([-1., -2., -3.]))
        >>> x.erfc()
        {
            a: aikit.array([1.57299206e-01, 4.67773480e-03, 2.20904985e-05]),
            b: aikit.array([1.84270084, 1.99532223, 1.99997795])
        }
        """
        return self.static_erfc(self, out=out)
