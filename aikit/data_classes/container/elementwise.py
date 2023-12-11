# global
from typing import Optional, Union, List, Dict, Literal

# local
import aikit
from aikit.data_classes.container.base import ContainerBase


class _ContainerWithElementwise(ContainerBase):
    @staticmethod
    def _static_abs(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:  # noqa
        """aikit.Container static method variant of aikit.abs. This method simply
        wraps the function, and so the docstring for aikit.abs also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a numeric data type.
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
            a container containing the absolute value of each element in ``x``. The
            returned container must have the same data type as ``x``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 2.6, -3.5]),
        ...                   b=aikit.array([4.5, -5.3, -0, -2.3]))
        >>> y = aikit.Container.static_abs(x)
        >>> print(y)
        {
            a: aikit.array([0., 2.6, 3.5]),
            b: aikit.array([4.5, 5.3, 0, 2.3])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "abs",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def abs(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.abs. This method simply
        wraps the function, and so the docstring for aikit.abs also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a numeric data type.
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
            a container containing the absolute value of each element in ``self``. The
            returned container must have the same data type as ``self``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([-1.6, 2.6, -3.5]),
        ...                   b=aikit.array([4.5, -5.3, -2.3]))
        >>> y = x.abs()
        >>> print(y)
        {
            a: aikit.array([1.6, 2.6, 3.5]),
            b: aikit.array([4.5, 5.3, 2.3])
        }
        """
        return self._static_abs(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_acosh(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.cosh. This method simply
        wraps the function, and so the docstring for aikit.cosh also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container whose elements each represent the area of a hyperbolic
            sector. Should have a floating-point data type.
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
            a container containing the inverse hyperbolic cosine of each element
            in ``x``. The returned container must have a floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1., 2., 3, 4]),
        ...                   b=aikit.array([1., 3., 10.0, 6]))
        >>> y = aikit.Container.static_acosh(x)
        >>> print(y)
        {
            a: aikit.array([0., 1.32, 1.76, 2.06]),
            b: aikit.array([0., 1.76, 2.99, 2.48])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "acosh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def acosh(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.acosh. This method
        simply wraps the function, and so the docstring for aikit.acosh also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements each represent the area of a hyperbolic
            sector. Should have a floating-point data type.
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
            a container containing the inverse hyperbolic cosine of each element in
            ``self``. The returned container must have a floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1., 2., 3, 4]),
        ...                   b=aikit.array([1., 3., 10.0, 6]))
        >>> y = x.acosh()
        >>> print(y)
        {
            a: aikit.array([0., 1.32, 1.76, 2.06]),
            b: aikit.array([0., 1.76, 2.99, 2.48])
        }
        """
        return self._static_acosh(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_acos(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.acos. This method simply
        wraps the function, and so the docstring for aikit.acos also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
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
            a container containing the inverse cosine of each element in ``x``.
            The returned container must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., -1, 1]), b=aikit.array([1., 0., -1.]))
        >>> y = aikit.Container.static_acos(x)
        >>> print(y)
        {
            a: aikit.array([1.57, 3.14, 0.]),
            b: aikit.array([0., 1.57, 3.14])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "acos",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_add(
        x1: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        x2: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        alpha: Optional[Union[int, float, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.add. This method simply
        wraps the function, and so the docstring for aikit.add also applies to
        this method with minimal changes.

        Parameters
        ----------
        x1
            first input array or container. Should have a numeric data type.
        x2
            second input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`). Should have a numeric data type.
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
        alpha
            scalar multiplier for ``x2``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise sums.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x = aikit.array([[1.1, 2.3, -3.6]])
        >>> y = aikit.Container(a=aikit.array([[4.], [5.], [6.]]),
        ...                   b=aikit.array([[5.], [6.], [7.]]))
        >>> z = aikit.Container.static_add(x, y)
        >>> print(z)
        {
            a: aikit.array([[5.1, 6.3, 0.4],
                          [6.1, 7.3, 1.4],
                          [7.1, 8.3, 2.4]]),
            b: aikit.array([[6.1, 7.3, 1.4],
                          [7.1, 8.3, 2.4],
                          [8.1, 9.3, 3.4]])
        }

        With multiple :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container(a=aikit.array([4, 5, 6]),
        ...                   b=aikit.array([5, 6, 7]))
        >>> z = aikit.Container.static_add(x, y)
        >>> print(z)
        {
            a: aikit.array([5, 7, 9]),
            b: aikit.array([7, 9, 11])
        }

        >>> x = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container(a=aikit.array([4, 5, 6]),
        ...                   b=aikit.array([5, 6, 7]))
        >>> z = aikit.Container.static_add(x, y, alpha=2)
        >>> print(z)
        {
            a: aikit.array([9, 12, 15]),
            b: aikit.array([12, 15, 18])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "add",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            alpha=alpha,
            out=out,
        )

    def acos(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.acos. This method
        simply wraps the function, and so the docstring for aikit.acos also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
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
            a container containing the inverse cosine of each element in ``self``.
            The returned container must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., -1, 1]), b=aikit.array([1., 0., -1.]))
        >>> y = x.acos()
        >>> print(y)
        {
            a: aikit.array([1.57, 3.14, 0.]),
            b: aikit.array([0., 1.57, 3.14])
        }
        """
        return self._static_acos(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def add(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        alpha: Optional[Union[int, float, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.add. This method simply
        wraps the function, and so the docstring for aikit.add also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            first input container. Should have a numeric data type.
        x2
            second input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.
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
        alpha
            scalar multiplier for ``x2``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise sums.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container(a=aikit.array([4, 5, 6]),
        ...                   b=aikit.array([5, 6, 7]))

        >>> z = x.add(y)
        >>> print(z)
        {
            a: aikit.array([5, 7, 9]),
            b: aikit.array([7, 9, 11])
        }

        >>> z = x.add(y, alpha=3)
        >>> print(z)
        {
            a: aikit.array([13, 17, 21]),
            b: aikit.array([17, 21, 25])
        }
        """
        return self._static_add(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            alpha=alpha,
            out=out,
        )

    @staticmethod
    def _static_asin(
        x: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.asin. This method simply
        wraps the function, and so the docstring for aikit.asin also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
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
            a container containing the inverse sine of each element in ``x``.
            The returned container must have a floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., -0.5, -1.]),
        ...                   b=aikit.array([0.1, 0.8, 2.]))
        >>> y = aikit.Container.static_asin()
        >>> print(y)
        {
            a: aikit.array([0., -0.524, -1.57]),
            b: aikit.array([0.1, 0.927, nan])
        }

        >>> x = aikit.Container(a=aikit.array([0.4, 0.9, -0.9]),
        ...                   b=aikit.array([[4, -3, -0.2]))
        >>> y = aikit.Container(a=aikit.zeros(3), b=aikit.zeros(3))
        >>> aikit.Container.static_asin(out=y)
        >>> print(y)
        {
            a: aikit.array([0.412, 1.12, -1.12]),
            b: aikit.array([nan, nan, -0.201])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "asin",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def asin(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.asin. This method
        simply wraps the function, and so the docstring for aikit.asin also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
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
            a container containing the inverse sine of each element in ``self``.
            The returned container must have a floating-point
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 0.5, 1.]),
        ...                   b=aikit.array([-4., 0.8, 2.]))
        >>> y = x.asin()
        >>> print(y)
        {
            a: aikit.array([0., 0.524, 1.57]),
            b: aikit.array([nan, 0.927, nan])
        }

        >>> x = aikit.Container(a=aikit.array([12., 1.5, 0.]),
        ...                   b=aikit.array([-0.85, 0.6, 0.3]))
        >>> y = aikit.Container(a=aikit.zeros(3), b=aikit.zeros(3))
        >>> x.asin(out=y)
        >>> print(y)
        {
            a: aikit.array([nan, nan, 0.]),
            b: aikit.array([-1.02, 0.644, 0.305])
        }
        """
        return self._static_asin(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_asinh(
        x: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.asinh. This method simply
        wraps the function, and so the docstring for aikit.asinh also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container whose elements each represent the area of a hyperbolic
            sector. Should have a floating-point data type.
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
            a container containing the inverse hyperbolic sine of each element
            in ``x``. The returned container must have a floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1.5, 0., -3.5]),
        ...                   b=aikit.array([3.4, -5.3, -0, -2.8]))
        >>> y = aikit.Container.static_asinh(x)
        >>> print(y)
        {
            a: aikit.array([1.19, 0., -1.97]),
            b: aikit.array([1.94, -2.37, 0., -1.75])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "asinh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def asinh(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.asinh. This method
        simply wraps the function, and so the docstring for aikit.asinh also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements each represent the area of a hyperbolic
            sector. Should have a floating-point data type.
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
            a container containing the inverse hyperbolic sine of each element in
            ``self``. The returned container must have a floating-point
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([-1, 3.7, -5.1]),
        ...                   b=aikit.array([4.5, -2.4, -1.5]))
        >>> y = x.asinh()
        >>> print(y)
        {
            a: aikit.array([-0.881, 2.02, -2.33]),
            b: aikit.array([2.21, -1.61, -1.19])
        }
        """
        return self._static_asinh(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_atan(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.atan. This method simply
        wraps the function, and so the docstring for aikit.atan also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
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
            a container containing the inverse tangent of each element in ``x``.
            The returned container must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., -1, 1]), b=aikit.array([1., 0., -6]))
        >>> y = aikit.Container.static_atan(x)
        >>> print(y)
        {
            a: aikit.array([0., -0.785, 0.785]),
            b: aikit.array([0.785, 0., -1.41])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "atan",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def atan(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.atan. This method
        simply wraps the function, and so the docstring for aikit.atan also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
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
            a container containing the inverse tangent of each element in ``x``.
            The returned container must have a floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., -1, 1]), b=aikit.array([1., 0., -6]))
        >>> y = x.atan()
        >>> print(y)
        {
            a: aikit.array([0., -0.785, 0.785]),
            b: aikit.array([0.785, 0., -1.41])
        }
        """
        return self._static_atan(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_atan2(
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
        """aikit.Container static method variant of aikit.atan2. This method simply
        wraps the function, and so the docstring for aikit.atan2 also applies to
        this method with minimal changes.

        Parameters
        ----------
        x1
            first input array or container corresponding to the y-coordinates.
            Should have a real-valued floating-point data type.
        x2
            second input array or container corresponding to the x-coordinates.
            Must be compatible with ``x1``
            (see :ref:`broadcasting`). Should have a real-valued
            floating-point data type.
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
            a container containing the inverse tangent of the quotient ``x1/x2``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 2.6, -3.5]),
        ...                   b=aikit.array([4.5, -5.3, -0]))
        >>> y = aikit.array([3.0, 2.0, 1.0])
        >>> aikit.Container.static_atan2(x, y)
        {
            a: aikit.array([0., 0.915, -1.29]),
            b: aikit.array([0.983, -1.21, 0.])
        }

        >>> x = aikit.Container(a=aikit.array([0., 2.6, -3.5]),
        ...                   b=aikit.array([4.5, -5.3, -0, -2.3]))
        >>> y = aikit.Container(a=aikit.array([-2.5, 1.75, 3.5]),
        ...                   b=aikit.array([2.45, 6.35, 0, 1.5]))
        >>> z = aikit.Container.static_atan2(x, y)
        >>> print(z)
        {
            a: aikit.array([3.14, 0.978, -0.785]),
            b: aikit.array([1.07, -0.696, 0., -0.993])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "atan2",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def atan2(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.atan2. This method
        simply wraps the function, and so the docstring for aikit.atan2 also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array or container corresponding to the y-coordinates.
            Should have a real-valued floating-point data type.
        x2
            second input array or container corresponding to the x-coordinates.
            Must be compatible with ``self`` (see :ref:`broadcasting`).
            Should have a real-valued floating-point data type.
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
            a container containing the inverse tangent of the quotient ``self/x2``.
            The returned array must have a real-valued floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 2.6, -3.5]),
        ...                   b=aikit.array([4.5, -5.3, -0]))
        >>> y = aikit.array([3.0, 2.0, 1.0])
        >>> x.atan2(y)
        {
            a: aikit.array([0., 0.915, -1.29]),
            b: aikit.array([0.983, -1.21, 0.])
        }

        >>> x = aikit.Container(a=aikit.array([0., 2.6, -3.5]),
        ...                   b=aikit.array([4.5, -5.3, -0, -2.3]))
        >>> y = aikit.Container(a=aikit.array([-2.5, 1.75, 3.5]),
        ...                   b=aikit.array([2.45, 6.35, 0, 1.5]))
        >>> z = x.atan2(y)
        >>> print(z)
        {
            a: aikit.array([3.14, 0.978, -0.785]),
            b: aikit.array([1.07, -0.696, 0., -0.993])
        }
        """
        return self._static_atan2(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_atanh(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.atanh. This method simply
        wraps the function, and so the docstring for aikit.atanh also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container whose elements each represent the area of a hyperbolic
            sector. Should have a floating-point data type.
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
            a container containing the inverse hyperbolic tangent of each
            element in ``x``. The returned container must have a floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0, 0.5, -0.5]), b=aikit.array([0., 0.2, 0.9]))
        >>> y = aikit.Container.static_atanh(x)
        >>> print(y)
        {
            a: aikit.array([0., 0.549, -0.549]),
            b: aikit.array([0., 0.203, 1.47])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "atanh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def atanh(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.atanh. This method
        simply wraps the function, and so the docstring for aikit.atanh also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements each represent the area of a
            hyperbolic sector. Should have a floating-point data type.
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
            a container containing the inverse hyperbolic tangent of each element
            in ``self``. The returned container must have a floating-point
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0, 0.5, -0.5]), b=aikit.array([0., 0.2, 0.9]))
        >>> y = x.atanh()
        >>> print(y)
        {
            a: aikit.array([0., 0.549, -0.549]),
            b: aikit.array([0., 0.203, 1.47])
        }
        """
        return self._static_atanh(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_bitwise_and(
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
        """aikit.Container static method variant of aikit.bitwise_and. This method
        simply wraps the function, and so the docstring for aikit.bitwise_and
        also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            first input array or container. Should have an integer or boolean
            data type.
        x2
            second input array or container Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
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
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x = aikit.array([1, 2, 3])
        >>> y = aikit.Container(a=aikit.array([4, 5, 6]),
        ...                   b=aikit.array([5, 6, 7]))
        >>> z = aikit.Container.static_bitwise_and(x, y)
        >>> print(z)
        {
                a: aikit.array([0, 0, 2]),
                b: aikit.array([1, 2, 3])
        }

        With multiple :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container(a=aikit.array([4, 5, 6]),
        ...                   b=aikit.array([5, 6, 7]))
        >>> z = aikit.Container.static_bitwise_and(x, y)
        >>> print(z)
        {
                a: aikit.array([0, 0, 2]),
                b: aikit.array([0, 2, 4])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "bitwise_and",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_and(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.bitwise_and. This
        method simply wraps the function, and so the docstring for
        aikit.bitwise_and also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array or container. Should have an integer or boolean
            data type.
        x2
            second input array or container Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
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
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([True, True]), b=aikit.array([False, True]))
        >>> y = aikit.Container(a=aikit.array([False, True]), b=aikit.array([False, True]))
        >>> x.bitwise_and(y, out=y)
        >>> print(y)
        {
            a: aikit.array([False, True]),
            b: aikit.array([False, True])
        }
        """
        return self._static_bitwise_and(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_bitwise_left_shift(
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
        """aikit.Container static method variant of aikit.bitwise_left_shift. This
        method simply wraps the function, and so the docstring for
        aikit.bitwise_left_shift also applies to this method with minimal
        changes.

        Parameters
        ----------
        x1
            first input array or container. Should have an integer or boolean
            data type.
        x2
            second input array or container Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
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
            a container containing the element-wise results.
            The returned container must have a data type determined by
            :ref:`type-promotion`.
        """
        return ContainerBase.cont_multi_map_in_function(
            "bitwise_left_shift",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_left_shift(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.bitwise_left_shift.
        This method simply wraps the function, and so the docstring for
        aikit.bitwise_left_shift also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            first input array or container. Should have an integer or boolean
            data type.
        x2
            second input array or container Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
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
            a container containing the element-wise results. The returned container
            must have a data type determined by :ref:`type-promotion`.
        """
        return self._static_bitwise_left_shift(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_bitwise_invert(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.bitwise_invert. This
        method simply wraps the function, and so the docstring for
        aikit.bitwise_invert also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have an integer or boolean data type.
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
            a container containing the element-wise results.
            The returned array must have the same data type as ``x``.

        Examples
        --------
        >>> x = aikit.Container(a=[False, True, False], b=[True, True, False])
        >>> y = aikit.Container.static_bitwise_invert(x)
        >>> print(y)
        {
            a: aikit.array([True, False, True]),
            b: aikit.array([False, False, True])
        }

        >>> x = aikit.Container(a=[1, 2, 3], b=[4, 5, 6])
        >>> y = aikit.Container.static_bitwise_invert(x)
        >>> print(y)
        {
            a: aikit.array([-2, -3, -4]),
            b: aikit.array([-5, -6, -7])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "bitwise_invert",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_invert(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.bitwise_invert. This
        method simply wraps the function, and so the docstring for
        aikit.bitwise_invert also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have an integer or boolean data type.
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
            a container containing the element-wise results.
            The returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = aikit.Container(a = aikit.array([False, True, False]),
        ...                   b = aikit.array([True, True, False]))
        >>> y = x.bitwise_invert()
        >>> print(y)
        {
            a: aikit.array([True, False, True]),
            b: aikit.array([False, False, True])
        }

        >>> x = aikit.Container(a = aikit.array([1, 2, 3]),
        ...                   b = aikit.array([4, 5, 6]))
        >>> y = x.bitwise_invert()
        >>> print(y)
        {
            a: aikit.array([-2, -3, -4]),
            b: aikit.array([-5, -6, -7])
        }
        """
        return self._static_bitwise_invert(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_cos(
        x: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.cos. This method simply
        wraps the function, and so the docstring for aikit.cos also applies to
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
            a container containing the cosine of each element in ``x``. The returned
            container must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([0., -1, 1]), b=aikit.array([1., 0., -6]))
        >>> y = aikit.Container.static_cos(x)
        >>> print(y)
        {
            a: aikit.array([1., 0.54, 0.54]),
            b: aikit.array([0.54, 1., 0.96])
        }
        """
        return aikit.ContainerBase.cont_multi_map_in_function(
            "cos",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def cos(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.cos. This method simply
        wraps the function, and so the docstring for aikit.cos also applies to
        this method with minimal changes.

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
            a container containing the cosine of each element in ``self``.
            The returned container must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([0., -1, 1]), b=aikit.array([1., 0., -6]))
        >>> y = x.cos()
        >>> print(y)
        {
            a: aikit.array([1., 0.54, 0.54]),
            b: aikit.array([0.54, 1., 0.96])
        }
        """
        return self._static_cos(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_bitwise_or(
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
        """aikit.Container static method variant of aikit.bitwise_or. This method
        simply wraps the function, and so the docstring for aikit.bitwise_or also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have an integer or boolean data type.
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
            a container containing the element-wise results.
            The returned array must have the same data type as ``x``.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> y = aikit.array([1, 2, 3])
        >>> x = aikit.Container(a=aikit.array([4, 5, 6]))
        >>> z = aikit.Container.static_bitwise_or(x, y)
        >>> print(z)
        {
            a: aikit.array([5, 7, 7]),
        }

        With multiple :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container(a=aikit.array([4, 5, 6]),
        ...                   b=aikit.array([5, 6, 7]))
        >>> z = aikit.Container.static_bitwise_or(x, y)
        >>> print(z)
        {
            a: aikit.array([5, 7, 7]),
            b: aikit.array([7, 7, 7])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "bitwise_or",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_or(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.bitwise_or. This method
        simply wraps the function, and so the docstring for aikit.bitwise_or also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have an integer or boolean data type.
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
            a container containing the element-wise results.
            The returned array must have the same data type as ``self``.

        Examples
        --------
        Using :class:`aikit.Container` instance method:

        >>> x = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container(a=aikit.array([4, 5, 6]),
        ...                   b=aikit.array([5, 6, 7]))
        >>> z = x.bitwise_or(y)
        >>> print(z)
        {
            a: aikit.array([5, 7, 7]),
            b: aikit.array([7, 7, 7])
        }
        """
        return self._static_bitwise_or(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_bitwise_right_shift(
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
        """aikit.Container static method variant of aikit.bitwise_right_shift. This
        method simply wraps the function, and so the docstring for
        aikit.bitwise_right_shift also applies to this method with minimal
        changes.

        Parameters
        ----------
        x1
            first input array or container. Should have an integer or boolean data type.
        x2
            second input array or container Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
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
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> a = aikit.Container(a = aikit.array([2, 3, 4]), b = aikit.array([5, 10, 64]))
        >>> b = aikit.array([0, 1, 2])
        >>> y = aikit.Container.static_bitwise_right_shift(a, b)
        >>> print(y)
        {
            a: aikit.array([2, 1, 1]),
            b: aikit.array([5, 5, 16])
        }

        With multiple :class:`aikit.Container` inputs:

        >>> a = aikit.Container(a = aikit.array([2, 3, 4]), b = aikit.array([5, 10, 64]))
        >>> b = aikit.Container(a = aikit.array([0, 1, 2]), b = aikit.array([2]))
        >>> y = aikit.Container.static_bitwise_right_shift(a, b)
        >>> print(y)
        {
            a: aikit.array([2, 1, 1]),
            b: aikit.array([1, 2, 16])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "bitwise_right_shift",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_right_shift(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.bitwise_right_shift.
        This method simply wraps the function, and so the docstring for
        aikit.bitwise_right_shift also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            first input array or container. Should have an integer or boolean data type.
        x2
            second input array or container Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
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
            a container containing the element-wise results. The returned container
            must have a data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> a = aikit.Container(a = aikit.array([2, 3, 4]), b = aikit.array([5, 10, 64]))
        >>> b = aikit.Container(a = aikit.array([0, 1, 2]), b = aikit.array([2]))
        >>> y = a.bitwise_right_shift(b)
        >>> print(y)
        {
            a: aikit.array([2, 1, 1]),
            b: aikit.array([1, 2, 16])
        }
        """
        return self._static_bitwise_right_shift(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_bitwise_xor(
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
        """aikit.Container static method variant of aikit.bitwise_xor. This method
        simply wraps the function, and so the docstring for aikit.bitwise_xor
        also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            first input array or container. Should have an integer or boolean
            data type.
        x2
            second input array or container Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
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
            a container containing the element-wise results.
            The returned container must have a data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a = aikit.array([89]), b = aikit.array([3]))
        >>> y = aikit.Container(a = aikit.array([12]), b = aikit.array([5]))
        >>> z = aikit.Container.static_bitwise_xor(x, y)
        >>> print(z)
        {
            a: aikit.array([85]),
            b: aikit.array([6])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "bitwise_xor",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_xor(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.bitwise_xor. This
        method simply wraps the function, and so the docstring for
        aikit.bitwise_xor also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array or container. Should have an integer or
            boolean data type.
        x2
            second input array or container Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
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
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a = aikit.array([89]), b = aikit.array([3]))
        >>> y = aikit.Container(a = aikit.array([12]), b = aikit.array([5]))
        >>> z = x.bitwise_xor(y)
        >>> print(z)
        {
            a: aikit.array([85]),
            b: aikit.array([6])
        }
        """
        return self._static_bitwise_xor(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_ceil(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.ceil. This method simply
        wraps the function, and so the docstring for aikit.ceil also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a numeric data type.
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
            an container containing the rounded result for each element in ``x``.
            The returned array must have the same data type as ``x``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([2.5, 0.5, -1.4]),
        ...                   b=aikit.array([5.4, -3.2, 5.2]))
        >>> y = aikit.Container.static_ceil(x)
        >>> print(y)
        {
            a: aikit.array([3., 1., -1.]),
            b: aikit.array([6., -3., 6.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "ceil",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def ceil(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.ceil. This method
        simply wraps the function, and so the docstring for aikit.ceil also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a numeric data type.
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
            an container containing the rounded result for each element in ``self``.
            The returned container must have the same data type as ``self``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([2.5, 0.5, -1.4]),
        ...                   b=aikit.array([5.4, -3.2, 5.2]))
        >>> y = x.ceil()
        >>> print(y)
        {
            a: aikit.array([3., 1., -1.]),
            b: aikit.array([6., -3., 6.])
        }
        """
        return self._static_ceil(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_cosh(
        x: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.cosh. This method simply
        wraps the function, and so the docstring for aikit.cosh also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container whose elements each represent a hyperbolic angle. Should
            have a floating-point data type.
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
            an container containing the hyperbolic cosine of each element in ``x``. The
            returned container must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([-1, 0.23, 1.12]), b=aikit.array([1, -2, 0.76]))
        >>> y = aikit.Container.static_cosh(x)
        >>> print(y)
        {
            a: aikit.array([1.54, 1.03, 1.7]),
            b: aikit.array([1.54, 3.76, 1.3])
        }

        >>> x = aikit.Container(a=aikit.array([-3, 0.34, 2.]),
        ...                   b=aikit.array([0.67, -0.98, -3]))
        >>> y = aikit.Container(a=aikit.zeros(1), b=aikit.zeros(1))
        >>> aikit.Container.static_cosh(x, out=y)
        >>> print(y)
        {
            a: aikit.array([10.1, 1.06, 3.76]),
            b: aikit.array([1.23, 1.52, 10.1])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "cosh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def cosh(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.cosh. This method
        simply wraps the function, and so the docstring for aikit.cosh also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements each represent a hyperbolic angle. Should
            have a floating-point data type.
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
            an container containing the hyperbolic cosine of each element in ``self``.
            The returned container must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([-1, 0.23, 1.12]), b=aikit.array([1, -2, 0.76]))
        >>> y = x.cosh()
        >>> print(y)
        {
            a: aikit.array([1.54, 1.03, 1.7]),
            b: aikit.array([1.54, 3.76, 1.3])
        }

        >>> x = aikit.Container(a=aikit.array([-3, 0.34, 2.]),
        ...                   b=aikit.array([0.67, -0.98, -3]))
        >>> y = aikit.Container(a=aikit.zeros(3), b=aikit.zeros(3))
        >>> x.cosh(out=y)
        >>> print(y)
        {
            a: aikit.array([10.1, 1.06, 3.76]),
            b: aikit.array([1.23, 1.52, 10.1])
        }
        """
        return self._static_cosh(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_divide(
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
        """aikit.Container static method variant of aikit.divide. This method
        simply wraps the function, and so the docstring for aikit.divide also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            dividend input array or container. Should have a real-valued data type.
        x2
            divisor input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x1 = aikit.Container(a=aikit.array([12., 3.5, 6.3]), b=aikit.array([3., 1., 0.9]))
        >>> x2 = aikit.Container(a=aikit.array([1., 2.3, 3]), b=aikit.array([2.4, 3., 2.]))
        >>> y = aikit.Container.static_divide(x1, x2)
        >>> print(y)
        {
            a: aikit.array([12., 1.52, 2.1]),
            b: aikit.array([1.25, 0.333, 0.45])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "divide",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def divide(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.divide. This method
        simply wraps the function, and so the docstring for aikit.divide also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            dividend input array or container. Should have a real-valued
            data type.
        x2
            divisor input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x1 = aikit.Container(a=aikit.array([12., 3.5, 6.3]), b=aikit.array([3., 1., 0.9]))
        >>> x2 = aikit.Container(a=aikit.array([1., 2.3, 3]), b=aikit.array([2.4, 3., 2.]))
        >>> y = x1.divide(x2)
        >>> print(y)
        {
            a: aikit.array([12., 1.52, 2.1]),
            b: aikit.array([1.25, 0.333, 0.45])
        }

        With :code:`Number` instances at the leaves:

        >>> x = aikit.Container(a=1, b=2)
        >>> y = aikit.Container(a=5, b=4)
        >>> z = x.divide(y)
        >>> print(z)
        {
            a: 0.2,
            b: 0.5
        }
        """
        return self._static_divide(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_equal(
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
        """aikit.Container static method variant of aikit.equal. This method simply
        wraps the function, and so the docstring for aikit.equal also applies to
        this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. May have any data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            May have any data type.
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
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x1 = aikit.Container(a=aikit.array([12, 3.5, 6.3]), b=aikit.array([3., 1., 0.9]))
        >>> x2 = aikit.Container(a=aikit.array([12, 2.3, 3]), b=aikit.array([2.4, 3., 2.]))
        >>> y = aikit.Container.static_equal(x1, x2)
        >>> print(y)
        {
            a: aikit.array([True, False, False]),
            b: aikit.array([False, False, False])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "equal",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def equal(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.equal. This method
        simply wraps the function, and so the docstring for aikit.equal also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. May have any data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            May have any data type.
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
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        With :class:`aikit.Container` inputs:

        >>> x1 = aikit.Container(a=aikit.array([12, 3.5, 6.3]), b=aikit.array([3., 1., 0.9]))
        >>> x2 = aikit.Container(a=aikit.array([12, 2.3, 3]), b=aikit.array([2.4, 3., 2.]))
        >>> y = x1.equal(x2)
        >>> print(y)
        {
            a: aikit.array([True, False, False]),
            b: aikit.array([False, False, False])
        }

        With mixed :class:`aikit.Container` and :class:`aikit.Array` inputs:

        >>> x1 = aikit.Container(a=aikit.array([12., 3.5, 6.3]), b=aikit.array([3., 1., 0.9]))
        >>> x2 = aikit.array([3., 1., 0.9])
        >>> y = x1.equal(x2)
        >>> print(y)
        {
            a: aikit.array([False, False, False]),
            b: aikit.array([True, True, True])
        }
        """
        return self._static_equal(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_nan_to_num(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        copy: Union[bool, aikit.Container] = True,
        nan: Union[float, int, aikit.Container] = 0.0,
        posinf: Optional[Union[float, int, aikit.Container]] = None,
        neginf: Optional[Union[float, int, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.nan_to_num. This method
        simply wraps the function, and so the docstring for aikit.nan_to_num also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container with array items.
        copy
            Whether to create a copy of x (True) or to replace values in-place (False).
            The in-place operation only occurs if casting to an array does not require
            a copy. Default is True.
        nan
            Value to be used to fill NaN values. If no value is passed then NaN values
            will be replaced with 0.0.
        posinf
            Value to be used to fill positive infinity values. If no value is passed
            then positive infinity values will be replaced with a very large number.
        neginf
            Value to be used to fill negative infinity values.
            If no value is passed then negative infinity values
            will be replaced with a very small (or negative) number.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with replaced non-finite elements.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 2, 3, nan]),\
                               b=aikit.array([1, 2, 3, inf]))
        >>> aikit.Container.static_nan_to_num(x, posinf=5e+100)
        {
            a: aikit.array([1.,  1.,  3.,  0.0])
            b: aikit.array([1., 2., 1.,  5e+100])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "nan_to_num",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            copy=copy,
            nan=nan,
            posinf=posinf,
            neginf=neginf,
            out=out,
        )

    def nan_to_num(
        self: aikit.Container,
        /,
        *,
        copy: Union[bool, aikit.Container] = True,
        nan: Union[float, int, aikit.Container] = 0.0,
        posinf: Optional[Union[float, int, aikit.Container]] = None,
        neginf: Optional[Union[float, int, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.nan_to_num. This method
        simply wraps the function, and so the docstring for aikit.nan_to_num also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container with array items.
        copy
            Whether to create a copy of x (True) or to replace values in-place (False).
            The in-place operation only occurs if casting to an array does not require
            a copy. Default is True.
        nan
            Value to be used to fill NaN values. If no value is passed then NaN values
            will be replaced with 0.0.
        posinf
            Value to be used to fill positive infinity values. If no value is passed
            then positive infinity values will be replaced with a very large number.
        neginf
            Value to be used to fill negative infinity values.
            If no value is passed then negative infinity values
            will be replaced with a very small (or negative) number.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with replaced non-finite elements.

        Examples
        --------
        >>> a = aikit.array([1., 2, 3, aikit.nan], dtype="float64")
        >>> b = aikit.array([1., 2, 3, aikit.inf], dtype="float64")
        >>> x = aikit.Container(a=a, b=b)
        >>> ret = x.nan_to_num(posinf=5e+100)
        >>> print(ret)
        {
            a: aikit.array([1., 2., 3., 0.]),
            b: aikit.array([1.e+000, 2.e+000, 3.e+000, 5.e+100])
        }
        """
        return self.static_nan_to_num(
            self, copy=copy, nan=nan, posinf=posinf, neginf=neginf, out=out
        )

    @staticmethod
    def static_imag(
        val: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.imag. This method simply
        wraps the function, and so the docstring for aikit.imag also applies to
        this method with minimal changes.

        Parameters
        ----------
        val
            Array-like input.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Returns an Container including arrays with the imaginary part
            of complex numbers.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array(np.array([1+2j, 3+4j, 5+6j])),
                                b=aikit.array(np.array([-2.25 + 4.75j, 3.25 + 5.75j])))
        >>> x
        {
            a: aikit.array([1.+2.j, 3.+4.j, 5.+6.j]),
            b: aikit.array([-2.25+4.75j, 3.25+5.75j])
        }
        >>> aikit.Container.static_imag(x)
        {
            a: aikit.array([2., 4., 6.]),
            b: aikit.array([4.75, 5.75])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "imag",
            val,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def imag(
        self: aikit.Container,
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.imag. This method
        simply wraps the function, and so the docstring for aikit.imag also
        applies to this method with minimal changes.

        Parameters
        ----------
        val
            Array-like input.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Returns an Container including arrays with the imaginary part
            of complex numbers.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array(np.array([1+2j, 3+4j, 5+6j])),
                                b=aikit.array(np.array([-2.25 + 4.75j, 3.25 + 5.75j])))
        >>> x
        {
            a: aikit.array([1.+2.j, 3.+4.j, 5.+6.j]),
            b: aikit.array([-2.25+4.75j, 3.25+5.75j])
        }
        >>> x.imag()
        {
            a: aikit.array([2., 4., 6.]),
            b: aikit.array([4.75, 5.75])
        }
        """
        return self.static_imag(self, out=out)

    @staticmethod
    def static_angle(
        z: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        deg: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.angle. This method simply
        wraps the function, and so the docstring for aikit.angle also applies to
        this method with minimal changes.

        Parameters
        ----------
        z
            Array-like input.
        deg
            optional bool.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Returns an array of angles for each complex number in the input.
            If def is False(default), angle is calculated in radian and if
            def is True, then angle is calculated in degrees.

        Examples
        --------
        >>> aikit.set_backend('tensorflow')
        >>> x = aikit.Container(a=aikit.array([-2.25 + 4.75j, 3.25 + 5.75j]),
                                b=aikit.array([-2.25 + 4.75j, 3.25 + 5.75j]))
        >>> x
        {
            a: aikit.array([-2.25+4.75j, 3.25+5.75j]),
            b: aikit.array([-2.25+4.75j, 3.25+5.75j])
        }
        >>> aikit.Container.static_angle(x)
        {
            a: aikit.array([2.01317055, 1.05634501]),
            b: aikit.array([2.01317055, 1.05634501])
        }
        >>> aikit.set_backend('numpy')
        >>> aikit.Container.static_angle(x,deg=True)
        {
            a: aikit.array([115.3461759, 60.524111]),
            b: aikit.array([115.3461759, 60.524111])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "angle",
            z,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            deg=deg,
            out=out,
        )

    def angle(
        self: aikit.Container,
        /,
        *,
        deg: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.angle. This method
        simply wraps the function, and so the docstring for aikit.angle also
        applies to this method with minimal changes.

        Parameters
        ----------
        z
            Array-like input.
        deg
            optional bool.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Returns an array of angles for each complex number in the input.
            If def is False(default), angle is calculated in radian and if
            def is True, then angle is calculated in degrees.

        Examples
        --------
        >>> aikit.set_backend('tensorflow')
        >>> x = aikit.Container(a=aikit.array([-2.25 + 4.75j, 3.25 + 5.75j]),
                                b=aikit.array([-2.25 + 4.75j, 3.25 + 5.75j]))
        >>> x
        {
            a: aikit.array([-2.25+4.75j, 3.25+5.75j]),
            b: aikit.array([-2.25+4.75j, 3.25+5.75j])
        }
        >>> x.angle()
        {
            a: aikit.array([2.01317055, 1.05634501]),
            b: aikit.array([2.01317055, 1.05634501])
        }
        >>> aikit.set_backend('numpy')
        >>> x.angle(deg=True)
        {
            a: aikit.array([115.3461759, 60.524111]),
            b: aikit.array([115.3461759, 60.524111])
        }
        """
        return self.static_angle(self, deg=deg, out=out)

    @staticmethod
    def static_gcd(
        x1: Union[aikit.Array, aikit.NativeArray, aikit.Container, int, list, tuple],
        x2: Union[aikit.Array, aikit.NativeArray, aikit.Container, int, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.gcd. This method simply
        wraps the function, and so the docstring for aikit.gcd also applies to
        this method with minimal changes.

        Parameters
        ----------
        x1
            first input container with array-like items.
        x2
            second input container with array-like items.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise gcd of input arrays.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([1, 2, 3]),\
                               b=aikit.array([1, 2, 3]))
        >>> x2 = aikit.Container(a=aikit.array([5, 6, 7]),\
                               b=10)
        >>> aikit.Container.static_gcd(x1, x2)
        {
            a: aikit.array([1.,  1.,  3.])
            b: aikit.array([1., 2., 1.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "gcd",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def gcd(
        self: aikit.Container,
        x2: aikit.Container,
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.gcd. This method simply
        wraps the function, and so the docstring for aikit.gcd also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            first input container with array-like items.
        x2
            second input container with array-like items.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise gcd of input arrays.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([1, 2, 3]),\
                               b=aikit.array([1, 2, 3]))
        >>> x2 = aikit.Container(a=aikit.array([5, 6, 7]),\
                               b=10)
        >>> x1.gcd(x2)
        {
            a: aikit.array([1.,  1.,  3.])
            b: aikit.array([1., 2., 1.])
        }
        """
        return self.static_gcd(self, x2, out=out)

    @staticmethod
    def static_exp2(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container, float, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.exp2. This method simply
        wraps the function, and so the docstring for aikit.exp2 also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            container with the base input arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise 2 to the power
            of input arrays elements.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 2, 3]),\
                               b=[5, 6, 7])
        >>> aikit.Container.static_exp2(x)
        {
            a: aikit.array([2.,  4.,  8.])
            b: aikit.array([32., 64., 128.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "exp2",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def exp2(
        self: aikit.Container,
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.exp2. This method
        simply wraps the function, and so the docstring for aikit.exp2 also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            container with the base input arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise 2 to the power
            of input array elements.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 2, 3]),\
                               b=[5, 6, 7])
        >>> x.exp2()
        {
            a: aikit.array([2.,  4.,  8.])
            b: aikit.array([32., 64., 128.])
        }
        """
        return self.static_exp2(self, out=out)

    @staticmethod
    def _static_exp(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.exp. This method simply
        wraps the function, and so the docstring for aikit.exp also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a floating-point data type.
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
            a container containing the evaluated result for each element in ``x``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1., 2.,]), b=aikit.array([4., 5.]))
        >>> y = aikit.Container.static_exp(x)
        >>> print(y)
        {
            a: aikit.array([2.71828198, 7.38905573]),
            b: aikit.array([54.59814835, 148.4131622])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "exp",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def exp(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.exp. This method simply
        wraps the function, and so the docstring for aikit.exp also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a floating-point data type.
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
            a container containing the evaluated result for each element in ``self``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1., 2., 3.]), b=aikit.array([4., 5., 6.]))
        >>> y = x.exp()
        >>> print(y)
        {
            a: aikit.array([2.71828198, 7.38905573, 20.08553696]),
            b: aikit.array([54.59814835, 148.4131622, 403.428772])
        }
        """
        return self._static_exp(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_expm1(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.expm1. This method simply
        wraps thefunction, and so the docstring for aikit.expm1 also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a floating-point data type.
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
            a container containing the evaluated result for each element in ``x``.
            The returned array must have areal-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Container` static method:

        >>> x = aikit.Container(a=aikit.array([1, 2]), b=aikit.array([3, 4]))
        >>> print(aikit.Container.static_expm1(x))
        {
            a: aikit.array([1.71828175, 6.38905621]),
            b: aikit.array([19.08553696, 53.59815216])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "expm1",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def expm1(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.expm1. This method
        simply wraps the function, and so the docstring for aikit.expm1 also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a floating-point data type.
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
            a container containing the evaluated result for each element in ``self``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([2.5, 0.5]),
        ...                   b=aikit.array([5.4, -3.2]))
        >>> y = x.expm1()
        >>> print(y)
        {
            a: aikit.array([11.2, 0.649]),
            b: aikit.array([220., -0.959])
        }

        >>> y = aikit.Container(a=aikit.array([0., 0.]))
        >>> x = aikit.Container(a=aikit.array([4., -2.]))
        >>> x.expm1(out=y)
        >>> print(y)
        {
            a: aikit.array([53.6, -0.865])
        }
        """
        return self._static_expm1(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_floor(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.floor. This method simply
        wraps thefunction, and so the docstring for aikit.floor also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a numeric data type.
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
            a container containing the rounded result for each element in ``x``. The
            returned array must have the same data type as ``x``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([2.5, 0.5, -1.4]),
        ...                   b=aikit.array([5.4, -3.2, 5.2]))
        >>> y = aikit.Container.static_floor(x)
        >>> print(y)
        {
            a: aikit.array([2., 0., -2.]),
            b: aikit.array([5., -4., 5.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "floor",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def floor(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.floor. This method
        simply wraps the function, and so the docstring for aikit.floor also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a numeric data type.
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
            a container containing the rounded result for each element in ``self``.
            The returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([2.5, 0.5, -1.4]),
        ...                   b=aikit.array([5.4, -3.2, 5.2]))
        >>> y = x.floor()
        >>> print(y)
        {
            a: aikit.array([2., 0., -2.]),
            b: aikit.array([5., -4., 5.])
        }
        """
        return self._static_floor(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_floor_divide(
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
        """aikit.Container static method variant of aikit.floor_divide. This method
        simply wraps the function, and so the docstring for aikit.floor_divide
        also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            dividend input array or container. Should have a real-valued data type.
        x2
            divisor input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x1 = aikit.Container(a=aikit.array([4., 5., 6.]), b=aikit.array([7., 8., 9.]))
        >>> x2 = aikit.Container(a=aikit.array([5., 4., 2.5]), b=aikit.array([2.3, 3.7, 5]))
        >>> y = aikit.Container.static_floor_divide(x1, x2)
        >>> print(y)
        {
            a: aikit.array([0., 1., 2.]),
            b: aikit.array([3., 2., 1.])
        }

        With mixed :class:`aikit.Container` and :class:`aikit.Array` inputs:

        >>> x1 = aikit.Container(a=aikit.array([4., 5., 6.]), b=aikit.array([7., 8., 9.]))
        >>> x2 = aikit.array([2, 3, 4])
        >>> y = aikit.Container.static_floor_divide(x1, x2)
        >>> print(y)
        {
            a: aikit.array([2., 1., 1.]),
            b: aikit.array([3., 2., 2.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "floor_divide",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def floor_divide(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.floor_divide. This
        method simply wraps the function, and so the docstring for
        aikit.floor_divide also applies to this method with minimal changes.

        Parameters
        ----------
        self
            dividend input array or container. Should have a real-valued
            data type.
        x2
            divisor input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x1 = aikit.Container(a=aikit.array([4., 5., 6.]), b=aikit.array([7., 8., 9.]))
        >>> x2 = aikit.Container(a=aikit.array([5., 4., 2.5]), b=aikit.array([2.3, 3.7, 5]))
        >>> y = x1.floor_divide(x2)
        >>> print(y)
        {
            a: aikit.array([0., 1., 2.]),
            b: aikit.array([3., 2., 1.])
        }

        With mixed :class:`aikit.Container` and :class:`aikit.Array` inputs:

        >>> x1 = aikit.Container(a=aikit.array([4., 5., 6.]), b=aikit.array([7., 8., 9.]))
        >>> x2 = aikit.array([2, 3, 4])
        >>> y = x1.floor_divide(x2)
        >>> print(y)
        {
            a: aikit.array([2., 1., 1.]),
            b: aikit.array([3., 2., 2.])
        }
        """
        return self._static_floor_divide(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_fmin(
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
        """aikit.Container static method variant of aikit.fmin. This method simply
        wraps the function, and so the docstring for aikit.fmin also applies to
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
            Container including arrays with element-wise minimums.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([2, 3, 4]),\
                               b=aikit.array([aikit.nan, 0, aikit.nan]))
        >>> x2 = aikit.Container(a=aikit.array([1, 5, 2]),\
                               b=aikit.array([0, aikit.nan, aikit.nan]))
        >>> aikit.Container.static_fmin(x1, x2)
        {
            a: aikit.array([1, 3, 2]),
            b: aikit.array([0., 0., nan])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "fmin",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def fmin(
        self: aikit.Container,
        x2: aikit.Container,
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.fmin. This method
        simply wraps the function, and so the docstring for aikit.fmin also
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
            Container including arrays with element-wise minimums.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([2, 3, 4]),\
                               b=aikit.array([aikit.nan, 0, aikit.nan]))
        >>> x2 = aikit.Container(a=aikit.array([1, 5, 2]),\
                               b=aikit.array([0, aikit.nan, aikit.nan]))
        >>> x1.fmin(x2)
        {
            a: aikit.array([1, 3, 2]),
            b: aikit.array([0., 0., nan])
        }
        """
        return self.static_fmin(self, x2, out=out)

    @staticmethod
    def _static_greater(
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
        """aikit.Container static method variant of aikit.greater. This method
        simply wraps the function, and so the docstring for aikit.greater also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a real-valued data type.
        x2
            divisor input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results. The returned array must
            have a data type of ``bool``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([4, 5, 6]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([5, 6, 7]))
        >>> z = aikit.Container.static_greater(y,x)
        >>> print(z)
        {
            a: aikit.array([False, False, False]),
            b: aikit.array([True, True, True])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "greater",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def greater(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.greater. This method
        simply wraps the function, and so the docstring for aikit.greater also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a real-valued data type.
        x2
            divisor input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results. The returned array must
            have a data type of ``bool``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([4, 5, 6]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([5, 6, 7]))
        >>> z = x.greater(y)
        >>> print(z)
        {
            a: aikit.array([True, True, True]),
            b: aikit.array([False, False, False])
        }
        """
        return self._static_greater(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_greater_equal(
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
        """aikit.Container static method variant of aikit.greater_equal. This
        method simply wraps the function, and so the docstring for
        aikit.greater_equal also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([4, 5, 6]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([5, 6, 7]))
        >>> z = aikit.Container.static_greater_equal(y)
        >>> print(z)
        {
            a:aikit.array([True,True,True]),
            b:aikit.array([False,False,False])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "greater_equal",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def greater_equal(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.greater_equal. This
        method simply wraps the function, and so the docstring for
        aikit.greater_equal also applies to this metho with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([4, 5, 6]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([5, 6, 7]))
        >>> z = x.greater_equal(y)
        >>> print(z)
        {
            a:aikit.array([True,True,True]),
            b:aikit.array([False,False,False])
        }
        """
        return self._static_greater_equal(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_isfinite(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.isfinite. This method
        simply wraps the function, and so the docstring for aikit.isfinite also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued data type.
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
            a container containing the test result. An element ``out_i`` is ``True``
            if ``x_i`` is finite and ``False`` otherwise.
            The returned array must have a data type of ``bool``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 999999999999]),
        ...                   b=aikit.array([float('-0'), aikit.nan]))
        >>> y = aikit.Container.static_isfinite(x)
        >>> print(y)
        {
            a: aikit.array([True, True]),
            b: aikit.array([True, False])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "isfinite",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def isfinite(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.isfinite. This method
        simply wraps the function, and so the docstring for aikit.isfinite also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued data type.
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
            a container containing the test result. An element ``out_i`` is ``True``
            if ``self_i`` is finite and ``False`` otherwise.
            The returned array must have a data type of ``bool``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 999999999999]),
        ...                   b=aikit.array([float('-0'), aikit.nan]))
        >>> y = x.isfinite()
        >>> print(y)
        {
            a: aikit.array([True, True]),
            b: aikit.array([True, False])
        }
        """
        return self._static_isfinite(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_isinf(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        detect_positive: Union[bool, aikit.Container] = True,
        detect_negative: Union[bool, aikit.Container] = True,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.isinf. This method simply
        wraps the function, and so the docstring for aikit.isinf also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued data type.
        detect_positive
            if ``True``, positive infinity is detected.
        detect_negative
            if ``True``, negative infinity is detected.
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
            a container containing the test result. An element ``out_i`` is ``True``
            if ``x_i`` is either positive or negative infinity and ``False``
            otherwise. The returned array must have a data type of ``bool``.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([-1, -float('inf'), 1.23]),
        ...                   b=aikit.array([float('inf'), 3.3, -4.2]))
        >>> z = aikit.Container.static_isinf(x)
        >>> print(z)
        {
            a: aikit.array([False, True, False]),
            b: aikit.array([True, False, False])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "isinf",
            x,
            detect_positive=detect_positive,
            detect_negative=detect_negative,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def isinf(
        self: aikit.Container,
        *,
        detect_positive: Union[bool, aikit.Container] = True,
        detect_negative: Union[bool, aikit.Container] = True,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.isinf. This method
        simply wraps the function, and so the docstring for aikit.isinf also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued data type.
        detect_positive
            if ``True``, positive infinity is detected.
        detect_negative
            if ``True``, negative infinity is detected.
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
            a container containing the test result. An element ``out_i`` is ``True``
            if ``self_i`` is either positive or negative infinity and ``False``
            otherwise. The returned array must have a data type of ``bool``.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([-1, -float('inf'), 1.23]),
        ...                   b=aikit.array([float('inf'), 3.3, -4.2]))
        >>> z = x.isinf()
        >>> print(z)
        {
            a: aikit.array([False, True, False]),
            b: aikit.array([True, False, False])
        }
        """
        return self._static_isinf(
            self,
            detect_positive=detect_positive,
            detect_negative=detect_negative,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_isnan(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.isnan. This method simply
        wraps the function, and so the docstring for aikit.isnan also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued data type.
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
            a container containing the test result. An element ``out_i`` is ``True``
            if ``x_i`` is ``NaN`` and ``False`` otherwise.
            The returned array should have a data type of ``bool``.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([-1, -float('nan'), 1.23]),
        ...                   b=aikit.array([float('nan'), 3.3, -4.2]))
        >>> z = aikit.Container.static_isnan(x)
        >>> print(z)
        {
            a: aikit.array([False, True, False]),
            b: aikit.array([True, False, False])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "isnan",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def isnan(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.isnan. This method
        simply wraps the function, and so the docstring for aikit.isnan also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued data type.
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
            a container containing the test result. An element ``out_i`` is ``True``
            if ``self_i`` is ``NaN`` and ``False`` otherwise.
            The returned array should have a data type of ``bool``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([-1, -float('nan'), 1.23]),
        ...                   b=aikit.array([float('nan'), 3.3, -4.2]))
        >>> y = x.isnan()
        >>> print(y)
        {
            a: aikit.array([False, True, False]),
            b: aikit.array([True, False, False])
        }
        """
        return self._static_isnan(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_less(
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
        """aikit.Container static method variant of aikit.less. This method simply
        wraps the function, and so the docstring for aikit.less also applies to
        this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([4, 5, 6]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([5, 6, 7]))
        >>> z = aikit.Container.static_less(y,x)
        >>> print(z)
        {
            a: aikit.array([True, True, True]),
            b: aikit.array([False, False, False])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "less",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def less(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.less. This method
        simply wraps the function, and so the docstring for aikit.less also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([4, 5, 6]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([5, 6, 7]))
        >>> z = x.less(y)
        >>> print(z)
        {
            a: aikit.array([False, False, False]),
            b: aikit.array([True, True, True])
        }
        """
        return self._static_less(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_less_equal(
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
        """aikit.Container static method variant of aikit.less_equal. This method
        simply wraps the function, and so the docstring for aikit.less_equal also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        With :code:'aikit.Container' inputs:

        >>> x1 = aikit.Container(a=aikit.array([12, 3.5, 9.2]), b=aikit.array([2., 1.1, 5.5]))
        >>> x2 = aikit.Container(a=aikit.array([12, 2.2, 4.1]), b=aikit.array([1, 0.7, 3.8]))
        >>> y = aikit.Container.static_less_equal(x1, x2)
        >>> print(y)
        {
            a: aikit.array([True, False, False]),
            b: aikit.array([False, False, False])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "less_equal",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def less_equal(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.less_equal. This method
        simply wraps the function, and so the docstring for aikit.less_equal also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        With :code:'aikit.Container' inputs:

        >>> x1 = aikit.Container(a=aikit.array([12, 3.5, 9.2]), b=aikit.array([2., 1.1, 5.5]))
        >>> x2 = aikit.Container(a=aikit.array([12, 2.2, 4.1]), b=aikit.array([1, 0.7, 3.8]))
        >>> y = x1.less_equal(x2)
        >>> print(y)
        {
            a: aikit.array([True, False, False]),
            b: aikit.array([False, False, False])
        }

        With mixed :code:'aikit.Container' and :code:'aikit.Array' inputs:

        >>> x1 = aikit.Container(a=aikit.array([12., 3.5, 9.2]), b=aikit.array([2., 1., 5.5]))
        >>> x2 = aikit.array([2., 1.1, 5.5])
        >>> y = x1.less_equal(x2)
        >>> print(y)
        {
            a: aikit.array([False, False, False]),
            b: aikit.array([True, True, True])
        }
        """
        return self._static_less_equal(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_log(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.log. This method simply
        wraps the function, and so the docstring for aikit.log also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
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
            a container containing the log for each element in ``x``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        Using :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([0.0, float('nan')]),
        ...                   b=aikit.array([-0., -3.9, float('+inf')]),
        ...                   c=aikit.array([7.9, 1.1, 1.]))
        >>> y = aikit.Container.static_log(x)
        >>> print(y)
        {
            a: aikit.array([-inf, nan]),
            b: aikit.array([-inf, nan, inf]),
            c: aikit.array([2.07, 0.0953, 0.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "log",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def log(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.log. This method simply
        wraps the function, and so the docstring for aikit.log also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
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
            a container containing the log for each element in ``self``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        Using :class:`aikit.Container` instance method:

        >>> x = aikit.Container(a=aikit.array([0.0, float('nan')]),
        ...                   b=aikit.array([-0., -3.9, float('+inf')]),
        ...                   c=aikit.array([7.9, 1.1, 1.]))
        >>> y = x.log()
        >>> print(y)
        {
            a: aikit.array([-inf, nan]),
            b: aikit.array([-inf, nan, inf]),
            c: aikit.array([2.07, 0.0953, 0.])
        }
        """
        return self._static_log(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_log1p(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.log1p. This method simply
        wraps the function, and so the docstring for aikit.log1p also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
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
            a container containing the evaluated result for each element in ``x``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]), b=aikit.array([3., 4., 5.1]))
        >>> y = aikit.Container.static_log1p(x)
        >>> print(y)
        {
            a: aikit.array([0., 0.693, 1.1]),
            b: aikit.array([1.39, 1.61, 1.81])
        }

        >>> x = aikit.Container(a=aikit.array([0., 2.]), b=aikit.array([ 4., 5.1]))
        >>> aikit.Container.static_log1p(x, out = x)
        >>> print(y)
        {
            a: aikit.array([0., 0.693, 1.1]),
            b: aikit.array([1.39, 1.61, 1.81])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "log1p",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def log1p(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.log1p. This method
        simply wraps the function, and so the docstring for aikit.log1p also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
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
            a container containing the evaluated result for each element in ``self``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1.6, 2.6, 3.5]),
        ...                   b=aikit.array([4.5, 5.3, 2.3]))
        >>> y = x.log1p()
        >>> print(y)
        {
            a: aikit.array([0.956, 1.28, 1.5]),
            b: aikit.array([1.7, 1.84, 1.19])
        }
        """
        return self._static_log1p(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_log2(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.log2. This method simply
        wraps the function, and so the docstring for aikit.log2 also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
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
            a container containing the evaluated base ``2`` logarithm for
            each element in ``x``. The returned array must have a real-valued
            floating-point data type determined by :ref:`type-promotion`.

        Examples
        --------
        Using :code:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([0.0, float('nan')]),\
                              b=aikit.array([-0., -4.9, float('+inf')]),\
                              c=aikit.array([8.9, 2.1, 1.]))
        >>> y = aikit.Container.static_log2(x)
        >>> print(y)
        {
            a: aikit.array([-inf, nan]),
            b: aikit.array([-inf, nan, inf]),
            c: aikit.array([3.15, 1.07, 0.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "log2",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def log2(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.log2. This method
        simply wraps the function, and so the docstring for aikit.log2 also
        applies to this metho with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
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
            a container containing the evaluated base ``2`` logarithm for each
            element in ``self``. The returned array must have a real-valued
            floating-point data type determined by :ref:`type-promotion`.

        Examples
        --------
        Using :code:`aikit.Container` instance method:

        >>> x = aikit.Container(a=aikit.array([0.0, float('nan')]),
        ...                   b=aikit.array([-0., -5.9, float('+inf')]),
        ...                   c=aikit.array([8.9, 2.1, 1.]))
        >>> y = aikit.log2(x)
        >>> print(y)
        {
            a: aikit.array([-inf, nan]),
            b: aikit.array([-inf, nan, inf]),
            c: aikit.array([3.15, 1.07, 0.])
        }
        """
        return self._static_log2(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_log10(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.log10. This method simply
        wraps the function, and so the docstring for aikit.log10 also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
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
            a container containing the evaluated base ``10`` logarithm for each
            element in ``x``. The returned array must have a real-valued
            floating-point data type determined by :ref:`type-promotion`.

        Examples
        --------
        Using :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([0.0, float('nan')]),
        ...                   b=aikit.array([-0., -3.9, float('+inf')]),
        ...                   c=aikit.array([7.9, 1.1, 1.]))
        >>> y = aikit.Container.static_log10(x)
        >>> print(y)
        {
            a: aikit.array([-inf, nan]),
            b: aikit.array([-inf, nan, inf]),
            c: aikit.array([0.898, 0.0414, 0.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "log10",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def log10(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.log10. This method
        simply wraps the function, and so the docstring for aikit.log10 also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
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
            a container containing the evaluated base ``10`` logarithm for
            each element in ``self``. The returned array must have a real-valued
            floating-point data type determined by :ref:`type-promotion`.

        Examples
        --------
        Using :class:`aikit.Container` instance method:

        >>> x = aikit.Container(a=aikit.array([0.0, float('nan')]),
        ...                   b=aikit.array([-0., -3.9, float('+inf')]),
        ...                   c=aikit.array([7.9, 1.1, 1.]))
        >>> y = x.log10()
        >>> print(y)
        {
            a: aikit.array([-inf, nan]),
            b: aikit.array([-inf, nan, inf]),
            c: aikit.array([0.898, 0.0414, 0.])
        }
        """
        return self._static_log10(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_logaddexp(
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
        """aikit.Container static method variant of aikit.greater_equal. This
        method simply wraps the function, and so the docstring for
        aikit.greater_equal also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results. The returned container
            must have a real-valued floating-point data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        Using :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([4., 5., .]),
        ...                   b=aikit.array([2., 3., 4.]))
        >>> y = aikit.Container(a=aikit.array([1., 2., 3.]),
        ...                   b=aikit.array([5., 6., 7.]))
        >>> z = aikit.Container.static_logaddexp(y,x)
        >>> print(z)
        {
            a: aikit.array([4.05, 5.05, 6.05]),
            b: aikit.array([5.05, 6.05, 7.05])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "logaddexp",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logaddexp(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.greater_equal. This
        method simply wraps the function, and so the docstring for
        aikit.greater_equal also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results. The returned container
            must have a real-valued floating-point data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        Using :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([4., 5., 6.]),
        ...                   b=aikit.array([2., 3., 4.]))
        >>> y = aikit.Container(a=aikit.array([1., 2., 3.]),
        ...                   b=aikit.array([5., 6., 7.]))
        >>> z = aikit.logaddexp(y,x)
        >>> print(z)
        {
            a: aikit.array([4.05, 5.05, 6.05]),
            b: aikit.array([5.05, 6.05, 7.05])
        }
        """
        return self._static_logaddexp(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_logaddexp2(
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
        """aikit.Container static method variant of aikit.logaddexp2. This method
        simply wraps the function, and so the docstring for aikit.logaddexp2 also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            first input container with array-like items.
        x2
            second input container with array-like items.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise logaddexp2 of input arrays.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([1, 2, 3]),\
                               b=aikit.array([1, 2, 3]))
        >>> x2 = aikit.Container(a=aikit.array([4, 5, 6]),\
                               b=5)
        >>> aikit.Container.static_logaddexp2(x1, x2)
        {
            a: aikit.array([4.169925, 5.169925, 6.169925])
            b: aikit.array([5.08746284, 5.169925  , 5.32192809])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "logaddexp2",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logaddexp2(
        self: aikit.Container,
        x2: aikit.Container,
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.logaddexp2. This method
        simply wraps the function, and so the docstring for aikit.logaddexp2 also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input container with array-like items.
        x2
            second input container with array-like items.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise logaddexp2 of input arrays.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([1, 2, 3]),\
                               b=aikit.array([1, 2, 3]))
        >>> x2 = aikit.Container(a=aikit.array([4, 5, 6]),\
                               b=5)
        >>> x1.logaddexp2(x2)
        {
            a: aikit.array([4.169925, 5.169925, 6.169925])
            b: aikit.array([5.08746284, 5.169925  , 5.32192809])
        }
        """
        return self.static_logaddexp2(self, x2, out=out)

    @staticmethod
    def _static_logical_and(
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
        """aikit.Container static method variant of aikit.logical_and. This method
        simply wraps the function, and so the docstring for aikit.logical_and
        also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a boolean data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a boolean data type.
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
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        Using 'aikit.Container' instance

        >>> a = aikit.Container(a=aikit.array([True, False, True, False]))
        >>> b = aikit.Container(a=aikit.array([True, True, False, False]))
        >>> w = aikit.Container.static_logical_and(a, b)
        >>> print(w)
        {
            a:aikit.array([True,False,False,False])
        }

        >>> j = aikit.Container(a=aikit.array([True, True, False, False]))
        >>> m = aikit.array([False, True, False, True])
        >>> x = aikit.Container.static_logical_and(j, m)
        >>> print(x)
        {
            a:aikit.array([False,True,False,False])
        }


        >>> k = aikit.Container(a=aikit.array([True, False, True]),
        ...                   b=aikit.array([True, False, False]))
        >>> l = aikit.Container(a=aikit.array([True, True, True]),
        ...                   b=aikit.array([False, False, False]))
        >>> z = aikit.Container.static_logical_and(k, l)
        >>> print(z)
        {
            a:aikit.array([True,False,True]),
            b:aikit.array([False,False,False])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "logical_and",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logical_and(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.logical_and. This
        method simply wraps the function, and so the docstring for
        aikit.logical_and also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a boolean data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a boolean data type.
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
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        Using 'aikit.Container' instance

        >>> a = aikit.Container(a=aikit.array([True, False, True, False]))
        >>> b = aikit.Container(a=aikit.array([True, True, False, False]))
        >>> w = a.logical_and(b)
        >>> print(w)
        {
            a:aikit.array([True,False,False,False])
        }

        >>> j = aikit.Container(a=aikit.array([True, True, False, False]))
        >>> m = aikit.array([False, True, False, True])
        >>> x = j.logical_and(m)
        >>> print(x)
        {
            a:aikit.array([False,True,False,False])
        }

        >>> k = aikit.Container(a=aikit.array([True, False, True]),
        ...                   b=aikit.array([True, False, False]))
        >>> l = aikit.Container(a=aikit.array([True, True, True]),
        ...                   b=aikit.array([False, False, False]))
        >>> z = k.logical_and(l)
        >>> print(z)
        {
            a:aikit.array([True,False,True]),
            b:aikit.array([False,False,False])
        }
        """
        return self._static_logical_and(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_logical_not(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.logical_not. This method
        simply wraps the function, and so the docstring for aikit.logical_not
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a boolean data type.
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
            a container containing the evaluated result for each element in ``x``.
            The returned container must have a data type of ``bool``.

        Examples
        --------
        Using 'aikit.Container' instance

        >>> x=aikit.Container(a=aikit.array([1,0,0,1]), b=aikit.array([3,1,7,0]))
        >>> aikit.Container.static_logical_not(x)
        {
            a: aikit.array([False, True, True, False]),
            b: aikit.array([False, False, False, True])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "logical_not",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logical_not(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.logical_not. This
        method simply wraps the function, and so the docstring for
        aikit.logical_not also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a boolean data type.
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
            a container containing the evaluated result for each element in ``self``.
            The returned container must have a data type of ``bool``.

        Examples
        --------
        Using 'aikit.Container' instance

        >>> x=aikit.Container(a=aikit.array([1,0,0,1]), b=aikit.array([3,1,7,0]))
        >>> y = x.logical_not()
        >>> print(y)
        {
            a: aikit.array([False, True, True, False]),
            b: aikit.array([False, False, False, True])
        }

        >>> x=aikit.Container(a=aikit.array([1,0,1,0]), b=aikit.native_array([5,2,0,3]))
        >>> y = x.logical_not()
        >>> print(y)
        {
            a: aikit.array([False, True, False, True]),
            b: aikit.array([False, False, True, False])
        }
        """
        return self._static_logical_not(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_logical_or(
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
        """aikit.Container static method variant of aikit.logical_or. This method
        simply wraps the function, and so the docstring for aikit.logical_or also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a boolean data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a boolean data type.
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
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([False, False, True]),
        ...                   b=aikit.array([True, False, True]))
        >>> y = aikit.Container(a=aikit.array([False, True, False]),
        ...                   b=aikit.array([True, True, False]))
        >>> z = aikit.Container.static_logical_or(x, y)
        >>> print(z)
        {
            a: aikit.array([False, True, True]),
            b: aikit.array([True, True, True])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "logical_or",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logical_or(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.logical_or. This method
        simply wraps the function, and so the docstring for aikit.logical_or also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a boolean data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a boolean data type.
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
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        This function conforms to the `Array API Standard
        <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of
        the `docstring <https://data-apis.org/array-api/latest/
        API_specification/generated/array_api.logical_or.html>`_
        in the standard.

        Both the description and the type hints above assumes an array input for
        simplicity, but this function is *nestable*, and therefore also
        accepts :class:`aikit.Container` instances in place of any of the arguments.

        Examples
        --------
        Using :class:`aikit.Container` instance method:

        >>> x = aikit.Container(a=aikit.array([False,True,True]),
        ...                   b=aikit.array([3.14, 2.718, 1.618]))
        >>> y = aikit.Container(a=aikit.array([0, 5.2, 0.8]), b=aikit.array([0.2, 0, 0.9]))
        >>> z = x.logical_or(y)
        >>> print(z)
        {
            a: aikit.array([False, True, True]),
            b: aikit.array([True, True, True])
        }
        """
        return self._static_logical_or(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_logical_xor(
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
        """aikit.Container static method variant of aikit.logical_xor. This method
        simply wraps the function, and so the docstring for aikit.logical_xor
        also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a boolean data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a boolean data type.
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
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x = aikit.array([0,0,1,1,0])
        >>> y = aikit.Container(a=aikit.array([1,0,0,1,0]), b=aikit.array([1,0,1,0,0]))
        >>> z = aikit.Container.static_logical_xor(x, y)
        >>> print(z)
        {
            a: aikit.array([True, False, True, False, False]),
            b: aikit.array([True, False, False, True, False])
        }

        With multiple :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([1,0,0,1,0]), b=aikit.array([1,0,1,0,0]))
        >>> y = aikit.Container(a=aikit.array([0,0,1,1,0]), b=aikit.array([1,0,1,1,0]))
        >>> z = aikit.Container.static_logical_xor(x, y)
        >>> print(z)
        {
            a: aikit.array([True, False, True, False, False]),
            b: aikit.array([False, False, False, True, False])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "logical_xor",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logical_xor(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.logical_xor. This
        method simply wraps the function, and so the docstring for
        aikit.logical_xor also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a boolean data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a boolean data type.
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
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1,0,0,1,0]), b=aikit.array([1,0,1,0,0]))
        >>> y = aikit.Container(a=aikit.array([0,0,1,1,0]), b=aikit.array([1,0,1,1,0]))
        >>> z = x.logical_xor(y)
        >>> print(z)
        {
            a: aikit.array([True, False, True, False, False]),
            b: aikit.array([False, False, False, True, False])
        }
        """
        return self._static_logical_xor(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_multiply(
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
        """aikit.Container static method variant of aikit.multiply. This method
        simply wraps the function, and so the docstring for aikit.multiply also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :code:`aikit.Container` inputs:

        >>> x1 = aikit.Container(a=aikit.array([15., 4.5, 6.5]),\
                               b=aikit.array([3.2, 5., 7.5]))
        >>> x2 = aikit.Container(a=aikit.array([1.7, 2.8, 3.]),\
                               b=aikit.array([5.6, 1.2, 4.2]))
        >>> y =aikit.Container.static_multiply(x1, x2)
        >>> print(y)
        {
            a: aikit.array([25.5, 12.6, 19.5]),
            b: aikit.array([17.9, 6., 31.5])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "multiply",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def multiply(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.multiply. This method
        simply wraps the function, and so the docstring for aikit.multiply also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array or container. Should have a numeric data type.
        x2
            second input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a nuneric data type.
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
            a container containing the element-wise products.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :code:`aikit.Container` inputs:

        >>> x1 = aikit.Container(a=aikit.array([15., 4.5, 6.5]),\
                               b=aikit.array([3.2, 5., 7.5]))
        >>> x2 = aikit.Container(a=aikit.array([1.7, 2.8, 3.]),\
                               b=aikit.array([5.6, 1.2, 4.2]))
        >>> y = aikit.Container.multiply(x1, x2)
        >>> print(y)
        {
            a: aikit.array([25.5, 12.6, 19.5]),
            b: aikit.array([17.9, 6., 31.5])
        }

        With mixed :code:`aikit.Container` and :code:`aikit.Array` inputs:

        >>> x1 = aikit.Container(a=aikit.array([6.2, 4.8, 2.3]),\
                               b=aikit.array([5., 1.7, 0.1]))
        >>> x2 = aikit.array([8.3, 3.2, 6.5])
        >>> y = aikit.Container.multiply(x1, x2)
        >>> print(y)
        {
            a: aikit.array([51.5, 15.4, 14.9]),
            b: aikit.array([41.5, 5.44, 0.65])
        }
        """
        return self._static_multiply(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_negative(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.negative. This method
        simply wraps the function, and so the docstring for aikit.negative also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a numeric data type.
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
            a container containing the evaluated result for each element in ``x``.
            The returned container must have the same data type as ``x``.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),
        ...                   b=aikit.array([3., 4., -5.]))
        >>> y = aikit.Container.static_negative(x)
        >>> print(y)
        {
            a: aikit.array([-0., -1., -2.]),
            b: aikit.array([-3., -4., 5.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "negative",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def negative(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.negative. This method
        simply wraps the function, and so the docstring for aikit.negative also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a numeric data type.
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
            a container containing the evaluated result for each element in ``self``.
            The returned container must have the same data type as ``self``.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),
        ...                   b=aikit.array([3., 4., -5.]))
        >>> y = x.negative()
        >>> print(y)
        {
            a: aikit.array([-0., -1., -2.]),
            b: aikit.array([-3., -4., 5.])
        }
        """
        return self._static_negative(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_not_equal(
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
        """aikit.Container static method variant of aikit.not_equal. This method
        simply wraps the function, and so the docstring for aikit.not_equal also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. May have any data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            May have any data type.
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
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x1 = aikit.Container(a=aikit.array([12, 3.5, 6.3]), b=aikit.array([3., 1., 0.9]))
        >>> x2 = aikit.Container(a=aikit.array([12, 2.3, 3]), b=aikit.array([2.4, 3., 2.]))
        >>> y = aikit.Container.static_not_equal(x1, x2)
        >>> print(y)
        {
            a: aikit.array([False, True, True]),
            b: aikit.array([True, True, True])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "not_equal",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def not_equal(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.not_equal. This method
        simply wraps the function, and so the docstring for aikit.not_equal also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. May have any data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            May have any data type.
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
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x1 = aikit.Container(a=aikit.array([12, 3.5, 6.3]), b=aikit.array([3., 1., 0.9]))
        >>> x2 = aikit.Container(a=aikit.array([12, 2.3, 3]), b=aikit.array([2.4, 3., 2.]))
        >>> y = x1.not_equal(x2)
        >>> print(y)
        {
            a: aikit.array([False, True, True]),
            b: aikit.array([True, True, True])
        }

        With mixed :class:`aikit.Container` and :class:`aikit.Array` inputs:

        >>> x1 = aikit.Container(a=aikit.array([12., 3.5, 6.3]), b=aikit.array([3., 1., 0.9]))
        >>> x2 = aikit.array([3., 1., 0.9])
        >>> y = x1.not_equal(x2)
        >>> print(y)
        {
            a: aikit.array([True, True, True]),
            b: aikit.array([False, False, False])
        }
        """
        return self._static_not_equal(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_positive(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.positive. This method
        simply wraps the function, and so the docstring for aikit.positive also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a numeric data type.
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
            a container containing the evaluated result for each element in ``x``.
            The returned container must have the same data type as ``x``.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),
        ...                   b=aikit.array([3., 4., -5.]))
        >>> y = aikit.Container.static_positive(x)
        >>> print(y)
        {
            a: aikit.array([0., 1., 2.]),
            b: aikit.array([3., 4., -5.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "positive",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def positive(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.positive. This method
        simply wraps the function, and so the docstring for aikit.positive also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a numeric data type.
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
            a container containing the evaluated result for each element in ``self``.
            The returned container must have the same data type as ``self``.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),
        ...                   b=aikit.array([3., 4., -5.]))
        >>> y = aikit.positive(x)
        >>> print(y)
        {
            a: aikit.array([0., 1., 2.]),
            b: aikit.array([3., 4., -5.])
        }
        """
        return self._static_positive(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_pow(
        x1: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        x2: Union[int, float, aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.pow. This method simply
        wraps the function, and so the docstring for aikit.pow also applies to
        this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results. The returned container
            must have a data type determined by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([0, 1]), b=aikit.array([2, 3]))
        >>> y = aikit.Container.static_pow(x, 2)
        >>> print(y)
        {
            a: aikit.array([0, 1]),
            b: aikit.array([4, 9])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "pow",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def pow(
        self: aikit.Container,
        x2: Union[int, float, aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.pow. This method simply
        wraps the function, and so the docstring for aikit.pow also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results. The returned container
            must have a data type determined by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([0, 1]), b=aikit.array([2, 3]))
        >>> y = x.pow(3)
        >>> print(y)
        {
            a:aikit.array([0,1]),
            b:aikit.array([8,27])
        }
        """
        return self._static_pow(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_real(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.real. This method simply
        wraps the function, and so the docstring for aikit.real also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
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
            a container containing the test result. An element ``out_i`` is ``out_i``
            if ``x_i`` is real number part only else ``real number part``,
            if it contains real and complex part both.
            The returned array should have a data type of ``float``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([-1+5j, 0-0j, 1.23j]),
        ...                   b=aikit.array([7.9, 0.31+3.3j, -4.2-5.9j]))
        >>> z = aikit.Container.static_real(x)
        >>> print(z)
        {
            a: aikit.array([-1., 0., 0.]),
            b: aikit.array([7.9, 0.31, -4.2])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "real",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def real(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.real. This method
        simply wraps the function, and so the docstring for aikit.real also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
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
            a container containing the test result.
            An element ``out_i`` is ``self_i`` if ``self_i`` is real number
            else ``took real number part only`` if ``self_i``
            contains real number and complex number both.
            The returned array should have a data type of ``float``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([-1j, 0.335+2.345j, 1.23+7j]),\
                          b=aikit.array([0.0, 1.2+3.3j, 1+0j]))
        >>> x.real()
        {
            a: aikit.array([0., 0.335, 1.23]),
            b: aikit.array([0.0, 1.2, 1.])
        }
        """
        return self.static_real(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_remainder(
        x1: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        x2: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        modulus: Union[bool, aikit.Container] = True,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.remainder. This method
        simply wraps the function, and so the docstring for aikit.remainder also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        modulus
            whether to compute the modulus instead of the remainder.
            Default is ``True``.
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
            a container containing the element-wise results. The returned container
            must have the same sign as the respective element ``x2_i``.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x1 = aikit.Container(a=aikit.array([2., 3., 5.]), b=aikit.array([2., 2., 4.]))
        >>> x2 = aikit.Container(a=aikit.array([1., 3., 4.]), b=aikit.array([1., 3., 3.]))
        >>> y = aikit.Container.static_remainder(x1, x2)
        >>> print(y)
        {
            a: aikit.array([0., 0., 1.]),
            b: aikit.array([0., 2., 1.])
        }

        With mixed :class:`aikit.Container` and `aikit.Array` inputs:

        >>> x1 = aikit.Container(a=aikit.array([2., 3., 5.]), b=aikit.array([2., 2., 4.]))
        >>> x2 = aikit.array([1., 2., 3.])
        >>> y = aikit.Container.static_remainder(x1, x2)
        >>> print(y)
        {
            a: aikit.array([0., 1., 2.]),
            b: aikit.array([0., 0., 1.])
        }

        With mixed :class:`aikit.Container` and `aikit.NativeArray` inputs:

        >>> x1 = aikit.Container(a=aikit.array([2., 3., 5.]), b=aikit.array([2., 2., 4.]))
        >>> x2 = aikit.native_array([1., 2., 3.])
        >>> y = aikit.Container.static_remainder(x1, x2)
        >>> print(y)
        {
            a: aikit.array([0., 1., 2.]),
            b: aikit.array([0., 0., 1.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "remainder",
            x1,
            x2,
            modulus=modulus,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def remainder(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        modulus: Union[bool, aikit.Container] = True,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.remainder. This method
        simply wraps the function, and so the docstring for aikit.remainder also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        modulus
            whether to compute the modulus instead of the remainder.
            Default is ``True``.
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
            a container containing the element-wise results. The returned container
            must have the same sign as the respective element ``x2_i``.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x1 = aikit.Container(a=aikit.array([2., 3., 5.]), b=aikit.array([2., 2., 4.]))
        >>> x2 = aikit.Container(a=aikit.array([1., 3., 4.]), b=aikit.array([1., 3., 3.]))
        >>> y = x1.remainder(x2)
        >>> print(y)
        {
            a: aikit.array([0., 0., 1.]),
            b: aikit.array([0., 2., 1.])
        }

        With mixed :class:`aikit.Container` and `aikit.Array` inputs:

        >>> x1 = aikit.Container(a=aikit.array([2., 3., 5.]), b=aikit.array([2., 2., 4.]))
        >>> x2 = aikit.array([1., 2., 3.])
        >>> y = x1.remainder(x2)
        >>> print(y)
        {
            a: aikit.array([0., 1., 2.]),
            b: aikit.array([0., 0., 1.])
        }

        With mixed :class:`aikit.Container` and `aikit.NativeArray` inputs:

        >>> x1 = aikit.Container(a=aikit.array([2., 3., 5.]), b=aikit.array([2., 2., 4.]))
        >>> x2 = aikit.native_array([1., 2., 3.])
        >>> y = x1.remainder(x2)
        >>> print(y)
        {
            a: aikit.array([0., 1., 2.]),
            b: aikit.array([0., 0., 1.])
        }
        """
        return self._static_remainder(
            self,
            x2,
            modulus=modulus,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_round(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        decimals: Union[int, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.round. This method simply
        wraps thevfunction, and so the docstring for aikit.round also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a numeric data type.
        decimals
            number of decimal places to round to. Default is ``0``.
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
            a container containing the rounded result for each element in ``x``.
            The returned container must have the same data type as ``x``.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([4.20, 8.6, 6.90, 0.0]),
        ...                   b=aikit.array([-300.9, -527.3, 4.5]))
        >>> y = aikit.Container.static_round(x)
        >>> print(y)
        {
            a: aikit.array([4., 9., 7., 0.]),
            b: aikit.array([-301., -527., 4.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "round",
            x,
            decimals=decimals,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def round(
        self: aikit.Container,
        *,
        decimals: Union[int, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.round. This method
        simply wraps the function, and so the docstring for aikit.round also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a numeric data type.
        decimals
            number of decimal places to round to. Default is ``0``.
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
            a container containing the rounded result for each element in ``self``.
            The returned container must have the same data type as ``self``.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([4.20, 8.6, 6.90, 0.0]),
        ...                   b=aikit.array([-300.9, -527.3, 4.5]))
        >>> y = x.round()
        >>> print(y)
        {
            a: aikit.array([4., 9., 7., 0.]),
            b: aikit.array([-301., -527., 4.])
        }
        """
        return self._static_round(
            self,
            decimals=decimals,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_sign(
        x: Union[float, aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        np_variant: Optional[Union[bool, aikit.Container]] = True,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.sign. This method simply
        wraps the function, and so the docstring for aikit.sign also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a numeric data type.
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
            a container containing the evaluated result for each element in ``x``.
            The returned container must have the same data type as ``x``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0, -1., 6.6]),
        ...                   b=aikit.array([-14.2, 8.3, 0.1, -0]))
        >>> y = aikit.Container.static_sign(x)
        >>> print(y)
        {
            a: aikit.array([0., -1., 1.]),
            b: aikit.array([-1., 1., 1., 0.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "sign",
            x,
            np_variant=np_variant,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sign(
        self: aikit.Container,
        *,
        np_variant: Optional[Union[bool, aikit.Container]] = True,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.sign. This method
        simply wraps the function, and so the docstring for aikit.sign also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a numeric data type.
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
            a container containing the evaluated result for each element in ``self``.
            The returned container must have the same data type as ``self``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([-6.7, 2.4, -8.5]),
        ...                   b=aikit.array([1.5, -0.3, 0]),
        ...                   c=aikit.array([-4.7, -5.4, 7.5]))
        >>> y = x.sign()
        >>> print(y)
        {
            a: aikit.array([-1., 1., -1.]),
            b: aikit.array([1., -1., 0.]),
            c: aikit.array([-1., -1., 1.])
        }
        """
        return self._static_sign(
            self,
            np_variant=np_variant,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_sin(
        x: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.sin. This method simply
        wraps the function, and so the docstring for aikit.sin also applies to
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
            a container containing the sine of each element in ``x``. The returned
            container must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([-1., -2., -3.]),
        ...                   b=aikit.array([4., 5., 6.]))
        >>> y = aikit.Container.static_sin(x)
        >>> print(y)
        {
            a: aikit.array([-0.841, -0.909, -0.141]),
            b: aikit.array([-0.757, -0.959, -0.279])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "sin",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sin(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.sin. This method simply
        wraps the function, and so the docstring for aikit.sin also applies to
        this method with minimal changes.

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
            a container containing the sine of each element in ``self``.
            The returned container must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1., 2., 3.]),
        ...                   b=aikit.array([-4., -5., -6.]))
        >>> y = x.sin()
        >>> print(y)
        {
            a: aikit.array([0.841, 0.909, 0.141]),
            b: aikit.array([0.757, 0.959, 0.279])
        }
        """
        return self._static_sin(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_sinh(
        x: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.sinh. This method simply
        wraps the function, and so the docstring for aikit.sinh also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container whose elements each represent a hyperbolic angle.
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
            an container containing the hyperbolic sine of each element in ``x``.
            The returned container must have a floating-point data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([-1, 0.23, 1.12]), b=aikit.array([1, -2, 0.76]))
        >>> y = aikit.Container.static_sinh(x)
        >>> print(y)
        {
            a: aikit.array([-1.18, 0.232, 1.37]),
            b: aikit.array([1.18, -3.63, 0.835])
        }

        >>> x = aikit.Container(a=aikit.array([-3, 0.34, 2.]),
        ...                   b=aikit.array([0.67, -0.98, -3]))
        >>> y = aikit.Container(a=aikit.zeros(1), b=aikit.zeros(1))
        >>> aikit.Container.static_sinh(x, out=y)
        >>> print(y)
        {
            a: aikit.array([-10., 0.347, 3.63]),
            b: aikit.array([0.721, -1.14, -10.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "sinh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sinh(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.sinh. This method
        simply wraps the function, and so the docstring for aikit.sinh also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements each represent a hyperbolic angle.
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
            an container containing the hyperbolic sine of each element in ``self``.
            The returned container must have a floating-point data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([-1, 0.23, 1.12]), b=aikit.array([1, -2, 0.76]))
        >>> y = x.sinh()
        >>> print(y)
        {
            a: aikit.array([-1.18, 0.232, 1.37]),
            b: aikit.array([1.18, -3.63, 0.835])
        }

        >>> x = aikit.Container(a=aikit.array([-3, 0.34, 2.]),
        ...                   b=aikit.array([0.67, -0.98, -3]))
        >>> y = aikit.Container(a=aikit.zeros(3), b=aikit.zeros(3))
        >>> x.sinh(out=y)
        >>> print(y)
        {
            a: aikit.array([-10., 0.347, 3.63]),
            b: aikit.array([0.721, -1.14, -10.])
        }
        """
        return self._static_sinh(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_square(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.square. This method
        simply wraps the function, and so the docstring for aikit.square also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
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
            a container containing the square of each element in ``x``.
            The returned container must have a real-valued floating-point
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0, 1]), b=aikit.array([2, 3]))
        >>> y = aikit.Container.static_square(x)
        >>> print(y)
        {
            a:aikit.array([0,1]),
            b:aikit.array([4,9])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "square",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def square(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.square. This method
        simply wraps the function, and so the docstring for aikit.square also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
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
            a container containing the square of each element in ``self``.
            The returned container must have a real-valued floating-point
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0, 1]), b=aikit.array([2, 3]))
        >>> y = x.square()
        >>> print(y)
        {
            a:aikit.array([0,1]),
            b:aikit.array([4,9])
        }
        """
        return self._static_square(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_sqrt(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.sqrt. This method simply
        wraps the function, and so the docstring for aikit.sqrt also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
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
            a container containing the square root of each element in ``x``.
            The returned container must have a real-valued floating-point
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        with :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([0., 100., 27.]),
        ...                   b=aikit.native_array([93., 54., 25.]))
        >>> y = aikit.Container.static_sqrt(x)
        >>> print(y)
        {
            a: aikit.array([0., 10., 5.2]),
            b: aikit.array([9.64, 7.35, 5.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "sqrt",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sqrt(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.sqrt. This method
        simply wraps the function, and so the docstring for aikit.sqrt also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
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
            a container containing the square root of each element in
            ``self``. The returned container must have a real-valued
            floating-point data type determined by :ref:`type-promotion`.

        Examples
        --------
        with :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([0., 100., 27.]),
        ...                   b=aikit.native_array([93., 54., 25.]))
        >>> y = x.sqrt()
        >>> print(y)
        {
            a: aikit.array([0., 10., 5.2]),
            b: aikit.array([9.64, 7.35, 5.])
        }
        """
        return self._static_sqrt(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_subtract(
        x1: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        x2: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        alpha: Optional[Union[int, float, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.subtract. This method
        simply wraps the function, and so the docstring for aikit.subtract also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            first input array or container. Should have a numeric data type.
        x2
            second input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`). Should have a numeric data type.
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
        alpha
            optional scalar multiplier for ``x2``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise sums.
            The returned container must have a data type determined
            by :ref:`type-promotion`.


        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container(a=aikit.array([4, 1, 3]),
        ...                   b=aikit.array([1, -1, 0]))
        >>> z = aikit.Container.static_subtract(x, y)
        >>> print(z)
        {
            a: aikit.array([-3, 1, 0]),
            b: aikit.array([1, 4, 4])
        }

        >>> z = aikit.Container.static_subtract(x, y, alpha=3)
        >>> print(z)
        {
            a: aikit.array([-11, -1, -6]),
            b: aikit.array([-1, 6, 4])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "subtract",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            alpha=alpha,
            out=out,
        )

    def subtract(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        alpha: Optional[Union[int, float, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.subtract. This method
        simply wraps the function, and so the docstring for aikit.subtract also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array or container. Should have a numeric data type.
        x2
            second input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.
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
        alpha
            optional scalar multiplier for ``x2``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise sums.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 2, 3]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container(a=aikit.array([4, 1, 3]),
        ...                   b=aikit.array([1, -1, 0]))
        >>> z = x.subtract(y)
        >>> print(z)
        {
            a: aikit.array([-3, 1, 0]),
            b: aikit.array([1, 4, 4])
        }

        >>> z = x.subtract(y, alpha=3)
        >>> print(z)
        {
            a: aikit.array([-11, -1, -6]),
            b: aikit.array([-1, 6, 4])
        }
        """
        return self._static_subtract(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            alpha=alpha,
            out=out,
        )

    @staticmethod
    def _static_tan(
        x: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.tan. This method simply
        wraps the function, and so the docstring for aikit.tan also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input array whose elements are expressed in radians. Should have a
            floating-point data type.
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
            an array containing the tangent of each element in ``x``.
            The return must have a floating-point data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]), b=aikit.array([3., 4., 5.]))
        >>> y = aikit.Container.static_tan(x)
        >>> print(y)
        {
            a: aikit.array([0., 1.56, -2.19]),
            b: aikit.array([-0.143, 1.16, -3.38])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "tan",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def tan(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.tan. This method simply
        wraps the function, and so the docstring for aikit.tan also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array whose elements are expressed in radians. Should have a
            floating-point data type.
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
            a container containing the tangent of each element in ``self``.
            The return must have a floating-point data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]), b=aikit.array([3., 4., 5.]))
        >>> y = x.tan()
        >>> print(y)
        {
            a:aikit.array([0., 1.56, -2.19]),
            b:aikit.array([-0.143, 1.16, -3.38])
        }
        """
        return self._static_tan(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_tanh(
        x: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.tanh. This method simply
        wraps the function, and so the docstring for aikit.tanh also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container whose elements each represent a hyperbolic angle.
            Should have a real-valued floating-point data type.
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
        complex_mode
            optional specifier for how to handle complex data types. See
            ``aikit.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the hyperbolic tangent of each element in ``x``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]), b=aikit.array([3., 4., 5.]))
        >>> y = aikit.Container.static_tanh(x)
        >>> print(y)
        {
            a: aikit.array([0., 0.76, 0.96]),
            b: aikit.array([0.995, 0.999, 0.9999])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "tanh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            complex_mode=complex_mode,
            out=out,
        )

    def tanh(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.tanh. This method
        simply wraps the function, and so the docstring for aikit.tanh also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements each represent a hyperbolic angle.
            Should have a real-valued floating-point data type.
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
        complex_mode
            optional specifier for how to handle complex data types. See
            ``aikit.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the hyperbolic tangent of each element in
            ``self``. The returned container must have a real-valued floating-point
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),
        ...                   b=aikit.array([3., 4., 5.]))
        >>> y = x.tanh()
        >>> print(y)
        {
            a:aikit.array([0., 0.762, 0.964]),
            b:aikit.array([0.995, 0.999, 1.])
        }
        """
        return self._static_tanh(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            complex_mode=complex_mode,
            out=out,
        )

    @staticmethod
    def _static_trunc(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.trunc. This method simply
        wraps the function, and so the docstring for aikit.trunc also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued data type.
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
            a container containing the rounded result for each element in ``x``.
            The returned container must have the same data type as ``x``.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([-0.25, 4, 1.3]),
        ...                   b=aikit.array([12, -3.5, 1.234]))
        >>> y = aikit.Container.static_trunc(x)
        >>> print(y)
        {
            a: aikit.array([-0., 4., 1.]),
            b: aikit.array([12., -3., 1.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "trunc",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def trunc(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.trunc. This method
        simply wraps the function, and so the docstring for aikit.trunc also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued data type.
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
            a container containing the rounded result for each element in ``self``.
            The returned container must have the same data type as ``self``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([-0.25, 4, 1.3]),
        ...                   b=aikit.array([12, -3.5, 1.234]))
        >>> y = x.trunc()
        >>> print(y)
        {
            a: aikit.array([-0., 4., 1.]),
            b: aikit.array([12., -3., 1.])
        }
        """
        return self._static_trunc(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_erf(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.erf. This method simply
        wraps the function, and so the docstring for aikit.erf also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container to compute exponential for.
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
            a container containing the Gauss error of ``x``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([-0.25, 4, 1.3]),
        ...                   b=aikit.array([12, -3.5, 1.234]))
        >>> y = aikit.Container.static_erf(x)
        >>> print(y)
        {
            a: aikit.array([-0.27632612, 1., 0.934008]),
            b: aikit.array([1., -0.99999928, 0.91903949])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "erf",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def erf(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.erf. This method simply
        wraps thefunction, and so the docstring for aikit.erf also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container to compute exponential for.
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
            a container containing the Gauss error of ``self``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([-0.25, 4, 1.3]),
        ...                   b=aikit.array([12, -3.5, 1.234]))
        >>> y = x.erf()
        >>> print(y)
        {
            a: aikit.array([-0.27632612, 1., 0.934008]),
            b: aikit.array([1., -0.99999928, 0.91903949])
        }
        """
        return self._static_erf(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_minimum(
        x1: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        use_where: Union[bool, aikit.Container] = True,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.minimum. This method
        simply wraps the function, and so the docstring for aikit.minimum also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            Input array containing elements to minimum threshold.
        x2
            The other container or number to compute the minimum against.
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
        use_where
            Whether to use :func:`where` to calculate the minimum. If ``False``, the
            minimum is calculated using the ``(x + y - |x - y|)/2`` formula. Default is
            ``True``.
        out
            optional output container, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
            Container object with all sub-arrays having the minimum values computed.

        Examples
        --------
        With multiple :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([1, 3, 1]),
        ...                   b=aikit.array([2, 8, 5]))
        >>> y = aikit.Container(a=aikit.array([1, 5, 6]),
        ...                   b=aikit.array([5, 9, 7]))
        >>> z = aikit.Container.static_minimum(x, y)
        >>> print(z)
        {
            a: aikit.array([1, 3, 1]),
            b: aikit.array([2, 8, 5])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "minimum",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            use_where=use_where,
            out=out,
        )

    def minimum(
        self: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        use_where: Union[bool, aikit.Container] = True,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.minimum. This method
        simply wraps the function, and so the docstring for aikit.minimum also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array containing elements to minimum threshold.
        x2
            The other container or number to compute the minimum against.
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
        use_where
            Whether to use :func:`where` to calculate the minimum. If ``False``, the
            minimum is calculated using the ``(x + y - |x - y|)/2`` formula. Default is
            ``True``.
        out
            optional output container, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
            Container object with all sub-arrays having the minimum values computed.

        Examples
        --------
        With multiple :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([1, 3, 1]),
        ...                   b=aikit.array([2, 8, 5]))
        >>> y = aikit.Container(a=aikit.array([1, 5, 6]),
        ...                   b=aikit.array([5, 9, 7]))
        >>> z = x.minimum(y)
        >>> print(z)
        {
            a: aikit.array([1, 3, 1]),
            b: aikit.array([2, 8, 5])
        }
        """
        return self._static_minimum(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            use_where=use_where,
            out=out,
        )

    @staticmethod
    def _static_maximum(
        x1: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        use_where: Union[bool, aikit.Container] = True,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.maximum. This method
        simply wraps the function, and so the docstring for aikit.maximum also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            Input array containing elements to maximum threshold.
        x2
            Tensor containing maximum values, must be broadcastable to x1.
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
        use_where
            Whether to use :func:`where` to calculate the maximum. If ``False``, the
            maximum is calculated using the ``(x + y + |x - y|)/2`` formula. Default is
            ``True``.
        out
            optional output container, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the elements of x1, but clipped to not be lower than the x2
            values.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x = aikit.array([[1, 3], [2, 4], [3, 7]])
        >>> y = aikit.Container(a=aikit.array([1, 0,]),
        ...                   b=aikit.array([-5, 9]))
        >>> z = aikit.Container.static_maximum(x, y)
        >>> print(z)
        {
            a: aikit.array([[1, 3],
                          [2, 4],
                          [3, 7]]),
            b: aikit.array([[1, 9],
                          [2, 9],
                          [3, 9]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "maximum",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            use_where=use_where,
            out=out,
        )

    def maximum(
        self: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        use_where: Union[bool, aikit.Container] = True,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.maximum. This method
        simply wraps the function, and so the docstring for aikit.maximum also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array containing elements to maximum threshold.
        x2
            Tensor containing maximum values, must be broadcastable to x1.
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
        use_where
            Whether to use :func:`where` to calculate the maximum. If ``False``, the
            maximum is calculated using the ``(x + y + |x - y|)/2`` formula. Default is
            ``True``.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            An container with the elements of x1, but clipped to not be
            lower than the x2 values.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x = aikit.array([[1, 3], [2, 4], [3, 7]])
        >>> y = aikit.Container(a=aikit.array([1, 0,]),
        ...                   b=aikit.array([-5, 9]))
        >>> z = x.maximum(y)
        >>> print(z)
        {
            a: aikit.array([[1, 3],
                          [2, 4],
                          [3, 7]]),
            b: aikit.array([[1, 9],
                          [2, 9],
                          [3, 9]])
        }
        """
        return self._static_maximum(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            use_where=use_where,
            out=out,
        )

    @staticmethod
    def _static_reciprocal(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.reciprocal. This method
        simply wraps the function, and so the docstring for aikit.reciprocal also
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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container with the element-wise recirpocal of ``x``

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 2]), b=aikit.array([4, 5]))
        >>> y = aikit.Container.static_reciprocal(x)
        >>> print(y)
        {
            a: aikit.array([1, 0.5]),
            b: aikit.array([0.25, 0.2])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "reciprocal",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def reciprocal(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.reciprocal. This method
        simply wraps the function, and so the docstring for aikit.reciprocal also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container to compute the element-wise reciprocal for.
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
            a container with the element-wise recirpocal of ``x``

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 2]), b=aikit.array([4, 5]))
        >>> y = x.reciprocal()
        >>> print(y)
        {
            a: aikit.array([1, 0.5]),
            b: aikit.array([0.25, 0.2])
        }
        """
        return self._static_reciprocal(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_deg2rad(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.deg2rad. This method
        simply wraps the function, and so the docstring for aikit.deg2rad also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. to be converted from degrees to radians.
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
            a container with each element in ``x`` converted from degrees to radians.

        Examples
        --------
        >>> x=aikit.Container(a=aikit.array([0,90,180,270,360]),
        ...                 b=aikit.native_array([0,-1.5,-50,aikit.nan]))
        >>> y=aikit.Container.static_deg2rad(x)
        >>> print(y)
        {
            a: aikit.array([0., 1.57, 3.14, 4.71, 6.28]),
            b: aikit.array([0., -0.0262, -0.873, nan])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "deg2rad",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def deg2rad(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.deg2rad. This method
        simply wraps the function, and so the docstring for aikit.deg2rad also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. to be converted from degrees to radians.
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
            a container with each element in ``x`` converted from degrees to radians.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x=aikit.Container(a=aikit.array([0., 0.351, -0.881, aikit.nan]),
        ...                 b=aikit.native_array([0,-1.5,-50,aikit.nan]))
        >>> y=x.deg2rad()
        >>> print(y)
        {
            a: aikit.array([0., 0.00613, -0.0154, nan]),
            b: aikit.array([0., -0.0262, -0.873, nan])
        }
        """
        return self._static_deg2rad(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_rad2deg(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.rad2deg. This method
        simply wraps the function, and so the docstring for aikit.rad2deg also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. to be converted from radians to degrees.
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
            a container with each element in ``x`` converted from radians to degrees.

        Examples
        --------
        >>> x=aikit.Container(a=aikit.array([0,90,180,270,360]),
        ...                 b=aikit.native_array([0,-1.5,-50,aikit.nan]))
        >>> y=aikit.Container.static_rad2deg(x)
        >>> print(y)
        {
            a: aikit.array([0., 5160., 10300., 15500., 20600.]),
            b: aikit.array([0., -85.9, -2860., nan])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "rad2deg",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def rad2deg(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.rad2deg. This method
        simply wraps the function, and so the docstring for aikit.rad2deg also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. to be converted from radians to degrees.
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
            a container with each element in ``x`` converted from radians to degrees.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x=aikit.Container(a=aikit.array([0., 0.351, -0.881, aikit.nan]),
        ...                 b=aikit.native_array([0,-1.5,-50,7.2]))
        >>> y=x.rad2deg()
        >>> print(y)
        {
            a: aikit.array([0., 20.1, -50.5, nan]),
            b: aikit.array([0., -85.9, -2860., 413.])
        }
        """
        return self._static_rad2deg(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_trunc_divide(
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
        """aikit.Container static method variant of aikit.trunc_divide. This method
        simply wraps the function, and so the docstring for aikit.trunc_divide
        also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            dividend input array or container. Should have a real-valued data type.
        x2
            divisor input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x1 = aikit.Container(a=aikit.array([12., 3.5, 6.3]), b=aikit.array([3., 1., 9.]))
        >>> x2 = aikit.Container(a=aikit.array([1., 2.3, -3]), b=aikit.array([2.4, 3., -2.]))
        >>> y = aikit.Container.static_divide(x1, x2)
        >>> print(y)
        {
            a: aikit.array([12., 1., -2.]),
            b: aikit.array([1., 0., -4.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "trunc_divide",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def trunc_divide(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.trunc_divide. This
        method simply wraps the function, and so the docstring for
        aikit.trunc_divide also applies to this method with minimal changes.

        Parameters
        ----------
        self
            dividend input array or container. Should have a real-valued
            data type.
        x2
            divisor input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
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
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x1 = aikit.Container(a=aikit.array([12., 3.5, 6.3]), b=aikit.array([3., 1., 9.]))
        >>> x2 = aikit.Container(a=aikit.array([1., 2.3, -3]), b=aikit.array([2.4, 3., -2.]))
        >>> y = x1.trunc_divide(x2)
        >>> print(y)
        {
            a: aikit.array([12., 1., -2.]),
            b: aikit.array([1., 0., -4.])
        }
        """
        return self._static_trunc_divide(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_isreal(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.isreal. This method
        simply wraps the function, and so the docstring for aikit.isreal also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued data type.
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
            a container containing the test result. An element ``out_i`` is ``True``
            if ``x_i`` is real number and ``False`` otherwise.
            The returned array should have a data type of ``bool``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([-1+5j, 0-0j, 1.23j]),
        ...                   b=aikit.array([7.9, 3.3j, -4.2-5.9j]))
        >>> z = aikit.Container.static_isreal(x)
        >>> print(z)
        {
            a: aikit.array([False, True, False]),
            b: aikit.array([True, False, False])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "isreal",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def isreal(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.isreal. This method
        simply wraps the function, and so the docstring for aikit.isreal also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued data type.
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
            a container containing the test result. An element ``out_i`` is ``True``
            if ``self_i`` is real number and ``False`` otherwise.
            The returned array should have a data type of ``bool``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([-1j, -np.inf, 1.23+7j]),\
                          b=aikit.array([0.0, 3.3j, 1+0j]))
        >>> y = x.isreal()
        >>> print(y)
        {
            a: aikit.array([False, True, False]),
            b: aikit.array([True, False, True])
        }
        """
        return self._static_isreal(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_trapz(
        y: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        x: Optional[Union[aikit.Array, aikit.NativeArray, aikit.Container]] = None,
        dx: Union[float, aikit.Container] = 1.0,
        axis: Union[int, aikit.Container] = -1,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.trapz. This method simply
        wraps the function, and so the docstring for aikit.trapz also applies to
        this method with minimal changes.

        Parameters
        ----------
        y
            The container whose arrays should be integrated.
        x
            The sample points corresponding to the input array values.
            If x is None, the sample points are assumed to be evenly spaced
            dx apart. The default is None.
        dx
            The spacing between sample points when x is None. The default is 1.
        axis
            The axis along which to integrate.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including definite integrals of n-dimensional arrays
            as approximated along a single axis by the trapezoidal rule.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> y = aikit.Container(a=aikit.array((1, 2, 3)), b=aikit.array((1, 5, 10)))
        >>> aikit.Container.static_trapz(y)
        {
            a: 4.0
            b: 10.5
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "trapz",
            y,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            x=x,
            dx=dx,
            axis=axis,
            out=out,
        )

    def trapz(
        self: aikit.Container,
        /,
        *,
        x: Optional[Union[aikit.Array, aikit.NativeArray, aikit.Container]] = None,
        dx: Union[float, aikit.Container] = 1.0,
        axis: Union[int, aikit.Container] = -1,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.trapz. This method
        simply wraps the function, and so the docstring for aikit.trapz also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            The container whose arrays should be integrated.
        x
            The sample points corresponding to the input array values.
            If x is None, the sample points are assumed to be evenly spaced
            dx apart. The default is None.
        dx
            The spacing between sample points when x is None. The default is 1.
        axis
            The axis along which to integrate.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including definite integrals of n-dimensional arrays
            as approximated along a single axis by the trapezoidal rule.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> y = aikit.Container(a=aikit.array((1, 2, 3)), b=aikit.array((1, 5, 10)))
        >>> y.trapz()
        {
            a: 4.0
            b: 10.5
        }
        """
        return self._static_trapz(self, x=x, dx=dx, axis=axis, out=out)

    @staticmethod
    def _static_lcm(
        x1: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.lcm. This method simply
        wraps the function, and so the docstring for aikit.lcm also applies to
        this method with minimal changes.

        Parameters
        ----------
        x1
            first input container.
        x2
            second input container.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            a container containing the element-wise least common multiples
            of the arrays contained in x1 and x2.

        Examples
        --------
        >>> x1=aikit.Container(a=aikit.array([2, 3, 4]),
        ...                  b=aikit.array([6, 54, 62, 10]))
        >>> x2=aikit.Container(a=aikit.array([5, 8, 15]),
        ...                  b=aikit.array([32, 40, 25, 13]))
        >>> aikit.Container.lcm(x1, x2)
        {
            a: aikit.array([10, 21, 60]),
            b: aikit.array([96, 1080, 1550, 130])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "lcm",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def lcm(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.lcm. This method simply
        wraps the function, and so the docstring for aikit.lcm also applies to
        this method with minimal changes.

        Parameters
        ----------
        x1
            first input container.
        x2
            second input container.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            a container containing the the element-wise least common multiples
            of the arrays contained in x1 and x2.

        Examples
        --------
        >>> x1=aikit.Container(a=aikit.array([2, 3, 4]),
        ...                  b=aikit.array([6, 54, 62, 10]))
        >>> x2=aikit.Container(a=aikit.array([5, 8, 15]),
        ...                  b=aikit.array([32, 40, 25, 13]))
        >>> x1.lcm(x2)
        {
            a: aikit.array([10, 24, 60]),
            b: aikit.array([96, 1080, 1550, 130])
        }
        """
        return self._static_lcm(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
