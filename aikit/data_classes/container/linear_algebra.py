# global

from typing import Union, Optional, Tuple, Literal, List, Dict, Sequence

# local
from aikit.data_classes.container.base import ContainerBase
import aikit

inf = float("inf")


# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor,PyMethodParameters
class _ContainerWithLinearAlgebra(ContainerBase):
    @staticmethod
    def _static_matmul(
        x1: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        x2: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        transpose_a: Union[bool, aikit.Container] = False,
        transpose_b: Union[bool, aikit.Container] = False,
        adjoint_a: Union[bool, aikit.Container] = False,
        adjoint_b: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.matmul. This method
        simply wraps the function, and so the docstring for aikit.matul also
        applies to this method with minimal changes.

        Parameters
        ----------
        x1
            first input array
        x2
            second input array
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
            the matrix multiplication result of x1 and x2

        Examples
        --------
        >>> x = aikit.Container(a = aikit.array([[3., -1.], [-1., 3.]]) ,
        ...                   b = aikit.array([[2., 1.], [1., 1.]]))
        >>> y = aikit.Container.static_matmul(x, x)
        >>> print(y)
        {
            a: aikit.array([[10., -6.],
                          [-6., 10.]]),
            b: aikit.array([[5., 3.],
                          [3., 2.]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "matmul",
            x1,
            x2,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            adjoint_a=adjoint_a,
            adjoint_b=adjoint_b,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def matmul(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        transpose_a: Union[bool, aikit.Container] = False,
        transpose_b: Union[bool, aikit.Container] = False,
        adjoint_a: Union[bool, aikit.Container] = False,
        adjoint_b: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.matmul. This method
        simply wraps the function, and so the docstring for aikit.matmul also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array
        x2
            second input array
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
            the matrix multiplication result of self and x2

        Examples
        --------
        >>> x = aikit.Container(a = aikit.array([[3., -1.], [-1., 3.]]) ,
        ...                   b = aikit.array([[2., 1.], [1., 1.]]))
        >>> y = x.matmul(x)
        >>> print(y)
        {
            a: aikit.array([[10., -6.],
                          [-6., 10.]]),
            b: aikit.array([[5., 3.],
                          [3., 2.]])
        }
        """
        return self._static_matmul(
            self,
            x2,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            adjoint_a=adjoint_a,
            adjoint_b=adjoint_b,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_cholesky(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        upper: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.cholesky. This method
        simply wraps the function, and so the docstring for aikit.cholesky also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array or container having shape (..., M, M) and whose innermost two
            dimensions form square symmetric positive-definite matrices. Should have a
            floating-point data type.
        upper
            If True, the result must be the upper-triangular Cholesky factor U. If
            False, the result must be the lower-triangular Cholesky factor L.
            Default: ``False``.
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
            a container containing the Cholesky factors for each square matrix. If upper
            is False, the returned container must contain lower-triangular matrices;
            otherwise, the returned container must contain upper-triangular matrices.
            The returned container must have a floating-point data type determined by
            Type Promotion Rules and must have the same shape as self.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> x = aikit.Container(a=aikit.array([[3., -1.], [-1., 3.]]),
        ...                      b=aikit.array([[2., 1.], [1., 1.]]))
        >>> y = aikit.Container.static_cholesky(x, upper='false')
        >>> print(y)
        {
            a: aikit.array([[1.73, -0.577],
                            [0., 1.63]]),
            b: aikit.array([[1.41, 0.707],
                            [0., 0.707]])
         }
        With multiple :class:`aikit.Container` inputs:
        >>> x = aikit.Container(a=aikit.array([[3., -1], [-1., 3.]]),
        ...                      b=aikit.array([[2., 1.], [1., 1.]]))
        >>> upper = aikit.Container(a=1, b=-1)
        >>> y = aikit.Container.static_roll(x, upper=False)
        >>> print(y)
        {
            a: aikit.array([[3., 3.],
                         [-1., -1.]]),
            b: aikit.array([[1., 1.],
                          [1., 2.]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "cholesky",
            x,
            upper=upper,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def cholesky(
        self: aikit.Container,
        /,
        *,
        upper: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.cholesky. This method
        simply wraps the function, and so the docstring for aikit.cholesky also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container having shape (..., M, M) and whose innermost two dimensions
            form square symmetric positive-definite matrices. Should have a
            floating-point data type.
        upper
            If True, the result must be the upper-triangular Cholesky factor U. If
            False, the result must be the lower-triangular Cholesky factor L.
            Default: ``False``.
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
            a container containing the Cholesky factors for each square matrix. If upper
            is False, the returned container must contain lower-triangular matrices;
            otherwise, the returned container must contain upper-triangular matrices.
            The returned container must have a floating-point data type determined by
            Type Promotion Rules and must have the same shape as self.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[3., -1],[-1., 3.]]),
        ...                      b=aikit.array([[2., 1.],[1., 1.]]))
        >>> y = x.cholesky(upper='false')
        >>> print(y)
        {
            a: aikit.array([[1.73, -0.577],
                            [0., 1.63]]),
            b: aikit.array([[1.41, 0.707],
                            [0., 0.707]])
        }
        """
        return self._static_cholesky(
            self,
            upper=upper,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_cross(
        x1: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        axis: Union[int, aikit.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.cross. This method simply
        wraps the function, and so the docstring for aikit.cross also applies to
        this method with minimal changes.

        Parameters
        ----------
        x1
            first input array. Should have a numeric data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.
        axis
            the axis (dimension) of x1 and x2 containing the vectors for which to
            compute the cross product.vIf set to -1, the function computes the
            cross product for vectors defined by the last axis (dimension).
            Default: ``-1``.
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
            an array containing the element-wise products. The returned array must have
            a data type determined by :ref:`type-promotion`.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x = aikit.array([9., 0., 3.])
        >>> y = aikit.Container(a=aikit.array([1., 1., 0.]), b=aikit.array([1., 0., 1.]))
        >>> z = aikit.Container.static_cross(x, y)
        >>> print(z)
        {
            a: aikit.array([-3., 3., 9.]),
            b: aikit.array([0., -6., 0.])
        }

        With multiple :class:`aikit.Container` inputs:

        >>> x = x = aikit.Container(a=aikit.array([5., 0., 0.]), b=aikit.array([0., 0., 2.]))
        >>> y = aikit.Container(a=aikit.array([0., 7., 0.]), b=aikit.array([3., 0., 0.]))
        >>> z = aikit.Container.static_cross(x, y)
        >>> print(z)
        {
            a: aikit.array([0., 0., 35.]),
            b: aikit.array([0., 6., 0.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "cross",
            x1,
            x2,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def cross(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        axis: Union[int, aikit.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.cross. This method
        simply wraps the function, and so the docstring for aikit.cross also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a numeric data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.
        axis
            the axis (dimension) of x1 and x2 containing the vectors for which to
            compute (default: -1) the cross product.vIf set to -1, the function
            computes the cross product for vectors defined by the last axis (dimension).
            Default: ``-1``.
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
            an array containing the element-wise products. The returned array must have
            a data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([5., 0., 0.]), b=aikit.array([0., 0., 2.]))
        >>> y = aikit.Container(a=aikit.array([0., 7., 0.]), b=aikit.array([3., 0., 0.]))
        >>> z = x.cross(y)
        >>> print(z)
        {
            a: aikit.array([0., 0., 35.]),
            b: aikit.array([0., 6., 0.])
        }
        """
        return self._static_cross(
            self,
            x2,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_det(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "det",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def det(
        self: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """
        Examples
        --------
        >>> x = aikit.Container(a = aikit.array([[3., -1.], [-1., 3.]]) ,
        ...                   b = aikit.array([[2., 1.], [1., 1.]]))
        >>> y = x.det()
        >>> print(y)
        {a:aikit.array(8.),b:aikit.array(1.)}
        """
        return self._static_det(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_diagonal(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        offset: Union[int, aikit.Container] = 0,
        axis1: Union[int, aikit.Container] = -2,
        axis2: Union[int, aikit.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.diagonal. This method
        simply wraps the function, and so the docstring for aikit.diagonal also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input Container with leave arrays having shape
             ``(..., M, N)`` and whose innermost two dimensions form
            ``MxN`` matrices.
        offset
            offset specifying the off-diagonal relative to the main diagonal.
            - ``offset = 0``: the main diagonal.
            - ``offset > 0``: off-diagonal above the main diagonal.
            - ``offset < 0``: off-diagonal below the main diagonal.
            Default: `0`.
        axis1
            axis to be used as the first axis of the 2-D sub-arrays from
            which the diagonals should be taken. Defaults to first axis (-2).
        axis2
            axis to be used as the second axis of the 2-D sub-arrays from which the
            diagonals should be taken. Defaults to second axis (-1).
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the diagonals. More details can be found in
            the docstring for aikit.diagonal.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([[1., 2.], [3., 4.]],
        ...                   b=aikit.array([[5., 6.], [7., 8.]])))
        >>> d = aikit.Container.static_diagonal(x)
        >>> print(d)
        {
            a:aikit.array([1., 4.]),
            b:aikit.array([5., 8.])
        }

        >>> a = aikit.array([[0, 1, 2],
        ...                [3, 4, 5],
        ...                [6, 7, 8]])
        >>> b = aikit.array([[-1., -2., -3.],
        ...                 [-3., 4., 5.],
        ...                 [5., 6., 7.]])],
        >>> x = aikit.Container(a=a, b=b)
        >>> d = aikit.Container.static_diagonal(offset=-1, axis1=0)
        >>> print(d)
        {
            a:aikit.array([3., 7.]),
            b:aikit.array([-3., 6.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "diagonal",
            x,
            offset=offset,
            axis1=axis1,
            axis2=axis2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def diagonal(
        self: aikit.Container,
        /,
        *,
        offset: Union[int, aikit.Container] = 0,
        axis1: Union[int, aikit.Container] = -2,
        axis2: Union[int, aikit.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.diagonal. This method
        simply wraps the function, and so the docstring for aikit.diagonal also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input Container with leave arrays having shape
             ``(..., M, N)`` and whose innermost two dimensions form
            ``MxN`` matrices.
        offset
            offset specifying the off-diagonal relative to the main diagonal.
            - ``offset = 0``: the main diagonal.
            - ``offset > 0``: off-diagonal above the main diagonal.
            - ``offset < 0``: off-diagonal below the main diagonal.
            Default: `0`.
        axis1
            axis to be used as the first axis of the 2-D sub-arrays from
            which the diagonals should be taken. Defaults to first axis (-2).
        axis2
            axis to be used as the second axis of the 2-D sub-arrays from which the
            diagonals should be taken. Defaults to second axis (-1).
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the diagonals. More details can be found in
            the docstring for aikit.diagonal.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([[1., 2.], [3., 4.]]),
        ...                   b=aikit.array([[5., 6.], [7., 8.]]))
        >>> d = x.diagonal()
        >>> print(d)
        {
            a:aikit.array([1., 4.]),
            b:aikit.array([5., 8.])
        }

        >>> a = aikit.array([[0, 1, 2],
        ...                [3, 4, 5],
        ...                [6, 7, 8]])
        >>> b = aikit.array([[-1., -2., -3.],
        ...                 [-3., 4., 5.],
        ...                 [5., 6., 7.]]),
        >>> x = aikit.Container(a=a, b=b)
        >>> d = x.diagonal(offset=-1)
        >>> print(d)
        {
            a: aikit.array([3, 7]),
            b: aikit.array([[-3., 6.]])
        }
        """
        return self._static_diagonal(
            self,
            offset=offset,
            axis1=axis1,
            axis2=axis2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_diag(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        k: Union[int, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "diag",
            x,
            k=k,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def diag(
        self: aikit.Container,
        /,
        *,
        k: Union[int, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.diag. This method
        simply wraps the function, and so the docstring for aikit.diag also
        applies to this method with minimal changes.

        Examples
        --------
        >>> x = aikit.Container(a=[[0, 1, 2],
        >>>                      [3, 4, 5],
        >>>                      [6, 7, 8]])
        >>> aikit.diag(x, k=1)
        {
            a: aikit.array([1, 5])
        }
        """
        return self._static_diag(
            self,
            k=k,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_eigh(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        UPLO: Union[str, aikit.Container] = "L",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "eigh",
            x,
            UPLO=UPLO,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def eigh(
        self: aikit.Container,
        /,
        *,
        UPLO: Union[str, aikit.Container] = "L",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.eigh. This method
        simply wraps the function, and so the docstring for aikit.eigh also
        applies to this method with minimal changes.

        Parameters
        ----------
        self : aikit.Container
            Aikit container having shape `(..., M, M)` and whose
            innermost two dimensions form square matrices.
            Should have a floating-point data type.
        UPLO : str, optional
            Specifies whether the upper or lower triangular part of the
            Hermitian matrix should be
            used for the eigenvalue decomposition. Default is 'L'.
        key_chains : Union[List[str], Dict[str, str]], optional
            The key-chains to apply or not apply the method to. Default is `None`.
        to_apply : bool, optional
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is `True`.
        prune_unapplied : bool, optional
            Whether to prune key_chains for which the function was not applied.
            Default is `False`.
        map_sequences : bool, optional
            Whether to also map method to sequences (lists, tuples).
            Default is `False`.
        out : aikit.Container, optional
            Optional output container, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        aikit.Container
            A container containing the computed eigenvalues.
            The returned array must have shape `(..., M)` and have the same
            data type as `self`.

        Examples
        --------
        With `aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([[[1.,2.],[2.,1.]]]),
        ...                   b=aikit.array([[[2.,4.],[4.,2.]]]))
        >>> y = x.eigh()
        >>> print(y)
        {
            a: aikit.array([[-1., 3.]]),
            b: aikit.array([[-2., 6.]])
        }
        """
        return self._static_eigh(
            self,
            UPLO=UPLO,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_eigvalsh(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        UPLO: Union[str, aikit.Container] = "L",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.eigvalsh. This method
        simply wraps the function, and so the docstring for aikit.eigvalsh also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            Aikit container having shape ``(..., M, M)`` and whose
            innermost two dimensions form square matrices.
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
            optional output container, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the computed eigenvalues.
            The returned array must have shape
            (..., M) and have the same data type as x.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([[[1.,2.,3.],[2.,4.,5.],[3.,5.,6.]]]),
        ...                   b=aikit.array([[[1.,1.,2.],[1.,2.,1.],[2.,1.,1.]]]),
        ...                   c=aikit.array([[[2.,2.,2.],[2.,3.,3.],[2.,3.,3.]]]))
        >>> e = aikit.Container.static_eigvalsh(x)
        >>> print(e)
        {
            a: aikit.array([[-0.51572949, 0.17091519, 11.3448143]]),
            b: aikit.array([[-1., 1., 4.]]),
            c: aikit.array([[-8.88178420e-16, 5.35898387e-01, 7.46410179e+00]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "eigvalsh",
            x,
            UPLO=UPLO,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def eigvalsh(
        self: aikit.Container,
        /,
        *,
        UPLO: Union[str, aikit.Container] = "L",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.eigvalsh. This method
        simply wraps the function, and so the docstring for aikit.eigvalsh also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Aikit container having shape ``(..., M, M)`` and whose
            innermost two dimensions form square matrices.
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
            optional output container, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the computed eigenvalues.
            The returned array must have shape
            (..., M) and have the same data type as x.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([[[1.,2.],[2.,1.]]]),
        ...                   b=aikit.array([[[2.,4.],[4.,2.]]]))
        >>> y = aikit.eigvalsh(x)
        >>> print(y)
        {
            a: aikit.array([[-1., 3.]]),
            b: aikit.array([[-2., 6.]])
        }
        """
        return self._static_eigvalsh(
            self,
            UPLO=UPLO,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_inner(
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
        """aikit.Container static method variant of aikit.inner. This method simply
        wraps the function, and so the docstring for aikit.inner also applies to
        this method with minimal changes.

        Return the inner product of two vectors ``x1`` and ``x2``.

        Parameters
        ----------
        x1
            first one-dimensional input array of size N.
            Should have a numeric data type.
            a(N,) array_like
            First input vector. Input is flattened if not already 1-dimensional.
        x2
            second one-dimensional input array of size M.
            Should have a numeric data type.
            b(M,) array_like
            Second input vector. Input is flattened if not already 1-dimensional.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            a two-dimensional array containing the inner product and whose
            shape is (N, M).
            The returned array must have a data type determined by Type Promotion Rules.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([[1, 2], [3, 4]]))
        >>> x2 = aikit.Container(a=aikit.array([5, 6]))
        >>> y = aikit.Container.static_inner(x1, x2)
        >>> print(y)
        {
            a: aikit.array([17, 39])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "inner",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def inner(
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
        """aikit.Container instance method variant of aikit.inner. This method
        simply wraps the function, and so the docstring for aikit.inner also
        applies to this method with minimal changes.

        Return the inner product of two vectors ``self`` and ``x2``.

        Parameters
        ----------
        self
            input container of size N. Should have a numeric data type.
            a(N,) array_like
            First input vector. Input is flattened if not already 1-dimensional.
        x2
            one-dimensional input array of size M. Should have a numeric data type.
            b(M,) array_like
            Second input vector. Input is flattened if not already 1-dimensional.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            a new container representing the inner product and whose
            shape is (N, M).
            The returned array must have a data type determined by Type Promotion Rules.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([[1, 2], [3, 4]]))
        >>> x2 = aikit.Container(a=aikit.array([5, 6]))
        >>> y = aikit.Container.inner(x1, x2)
        >>> print(y)
        {
            a: aikit.array([17, 39])
        }
        """
        return self._static_inner(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_inv(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        adjoint: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.inv. This method simply
        wraps the function, and so the docstring for aikit.inv also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            Aikit container having shape ``(..., M, M)`` and whose
            innermost two dimensions form square matrices.
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
            optional output container, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            A container containing the multiplicative inverses.
            The returned array must have a floating-point data type
            determined by :ref:`type-promotion` and must have the
            same shape as ``x``.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([[0., 1.], [4., 4.]]),
        ...                      b=aikit.array([[4., 4.], [2., 1.]]))
        >>> y = aikit.Container.static_inv(x)
        >>> print(y)
        {
            a: aikit.array([[-1, 0.25], [1., 0.]]),
            b: aikit.array([-0.25, 1.], [0.5, -1.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "inv",
            x,
            adjoint=adjoint,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def inv(
        self: aikit.Container,
        /,
        *,
        adjoint: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.inv. This method simply
        wraps the function, and so the docstring for aikit.inv also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Aikit container having shape ``(..., M, M)`` and whose
            innermost two dimensions form square matrices.
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
            optional output container, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            A container containing the multiplicative inverses.
            The returned array must have a floating-point data type
            determined by :ref:`type-promotion` and must have the
            same shape as ``x``.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([[0., 1.], [4., 4.]]),
        ...                      b=aikit.array([[4., 4.], [2., 1.]]))
        >>> y = x.inv()
        >>> print(y)
        {
            a: aikit.array([[-1, 0.25], [1., 0.]]),
            b: aikit.array([-0.25, 1.], [0.5, -1.])
        }
        """
        return self._static_inv(
            self,
            adjoint=adjoint,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_pinv(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        rtol: Optional[Union[float, Tuple[float], aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container special method variant of aikit.pinv. This method simply
        wraps the function, and so the docstring for aikit.pinv also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input array having shape ``(..., M, N)`` and whose innermost two
            dimensions form``MxN`` matrices. Should have a floating-point
            data type.
        rtol
            relative tolerance for small singular values approximately less
            than or equal to ``rtol * largest_singular_value`` are set to zero.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the pseudo-inverses. The returned array must have a
            floating-point data type determined by :ref:`type-promotion` and
            must have shape ``(..., N, M)`` (i.e., must have the same shape as
            ``x``, except the innermost two dimensions must be transposed).

        Examples
        --------
        >>> x = aikit.Container(a= aikit.array([[1., 2.], [3., 4.]]))
        >>> y = aikit.Container.static_pinv(x)
        >>> print(y)
        {
            a: aikit.array([[-2., 1.],
                          [1.5, -0.5]])
        }

        >>> x = aikit.Container(a=aikit.array([[1., 2.], [3., 4.]]))
        >>> out = aikit.Container(a=aikit.zeros((2, 2)))
        >>> aikit.Container.static_pinv(x, rtol=1e-1, out=out)
        >>> print(out)
        {
            a: aikit.array([[0.0426, 0.0964],
                          [0.0605, 0.1368]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "pinv",
            x,
            rtol=rtol,
            out=out,
        )

    def pinv(
        self: aikit.Container,
        /,
        *,
        rtol: Optional[Union[float, Tuple[float], aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.pinv. This method
        simply wraps the function, and so the docstring for aikit.pinv also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array having shape ``(..., M, N)`` and whose innermost
            two dimensions form``MxN`` matrices. Should have a floating-point
            data type.
        rtol
            relative tolerance for small singular values approximately less
            than or equal to ``rtol * largest_singular_value`` are set to zero.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the pseudo-inverses. The returned array must have a
            floating-point data type determined by :ref:`type-promotion` and
            must have shape ``(..., N, M)`` (i.e., must have the same shape as
            ``x``, except the innermost two dimensions must be transposed).


        Examples
        --------
        >>> x = aikit.Container(a= aikit.array([[1., 2.], [3., 4.]]))
        >>> y = x.pinv()
        >>> print(y)
        {
            a: aikit.array([[-1.99999988, 1.],
                          [1.5, -0.5]])
        }

        >>> x = aikit.Container(a = aikit.array([[1., 2.], [3., 4.]]))
        >>> out = aikit.Container(a = aikit.zeros(x["a"].shape))
        >>> x.pinv(out=out)
        >>> print(out)
        {
            a: aikit.array([[-1.99999988, 1.],
                          [1.5, -0.5]])
        }
        """
        return self._static_pinv(
            self,
            rtol=rtol,
            out=out,
        )

    @staticmethod
    def _static_matrix_norm(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        ord: Union[int, float, Literal[inf, -inf, "fro", "nuc"], aikit.Container] = "fro",
        axis: Tuple[int, int, aikit.Container] = (-2, -1),
        keepdims: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.matrix_norm. This method
        simply wraps the function, and so the docstring for aikit.matrix_norm
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array having shape (..., M, N) and whose innermost two deimensions
            form MxN matrices. Should have a floating-point data type.
        ord
            Order of the norm. Default is "fro".
        axis
            specifies the axes that hold 2-D matrices. Default: (-2, -1).
        keepdims
            If this is set to True, the axes which are normed over are left in the
            result as dimensions with size one. With this option the result will
            broadcast correctly against the original x. Default is ``False``.
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
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            Matrix norm of the array at specified axes.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[1.1, 2.2], [1., 2.]]), \
                              b=aikit.array([[1., 2.], [3., 4.]]))
        >>> y = aikit.Container.static_matrix_norm(x, ord=1)
        >>> print(y)
        {
            a: aikit.array(4.2),
            b: aikit.array(6.)
        }

        >>> x = aikit.Container(a=aikit.arange(12, dtype=float).reshape((3, 2, 2)), \
                              b=aikit.arange(8, dtype=float).reshape((2, 2, 2)))
        >>> ord = aikit.Container(a=1, b=float('inf'))
        >>> axis = aikit.Container(a=(1, 2), b=(2, 1))
        >>> k = aikit.Container(a=False, b=True)
        >>> y = aikit.Container.static_matrix_norm(x, ord=ord, axis=axis, keepdims=k)
        >>> print(y)
        {
            a: aikit.array([4.24, 11.4, 19.2]),
            b: aikit.array([[[3.7]],
                          [[11.2]]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "matrix_norm",
            x,
            ord=ord,
            axis=axis,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def matrix_norm(
        self: aikit.Container,
        /,
        *,
        ord: Union[int, float, Literal[inf, -inf, "fro", "nuc"], aikit.Container] = "fro",
        axis: Tuple[int, int, aikit.Container] = (-2, -1),
        keepdims: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.matrix_norm. This
        method simply wraps the function, and so the docstring for
        aikit.matrix_norm also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Container having shape (..., M, N) and whose innermost two dimensions
            form MxN matrices. Should have a floating-point data type.
        ord
            Order of the norm. Default is "fro".
        axis
            specifies the axes that hold 2-D matrices. Default: (-2, -1).
        keepdims
            If this is set to True, the axes which are normed over are left in the
            result as dimensions with size one. With this option the result will
            broadcast correctly against the original x. Default is ``False``.
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
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            Matrix norm of the array at specified axes.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[1.1, 2.2], [1., 2.]]), \
                              b=aikit.array([[1., 2.], [3., 4.]]))
        >>> y = x.matrix_norm(ord=1)
        >>> print(y)
        {
            a: aikit.array(4.2),
            b: aikit.array(6.)
        }

        >>> x = aikit.Container(a=aikit.arange(12, dtype=float).reshape((3, 2, 2)), \
                              b=aikit.arange(8, dtype=float).reshape((2, 2, 2)))
        >>> ord = aikit.Container(a="nuc", b=aikit.inf)
        >>> axis = aikit.Container(a=(1, 2), b=(2, 1))
        >>> k = aikit.Container(a=True, b=False)
        >>> y = x.matrix_norm(ord=ord, axis=axis, keepdims=k)
        >>> print(y)
        {
            a: aikit.array([[[4.24]],
                         [[11.4]],
                         [[19.2]]]),
            b: aikit.array([4., 12.])
        }
        """
        return self._static_matrix_norm(
            self,
            ord=ord,
            axis=axis,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_matrix_power(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        n: Union[int, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "matrix_power",
            x,
            n,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def matrix_power(
        self: aikit.Container,
        n: Union[int, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return self._static_matrix_power(
            self,
            n,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_matrix_rank(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        atol: Optional[Union[float, Tuple[float], aikit.Container]] = None,
        rtol: Optional[Union[float, Tuple[float], aikit.Container]] = None,
        hermitian: Optional[Union[bool, aikit.Container]] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.matrix_rank. This method
        returns the rank (i.e., number of non-zero singular values) of a matrix
        (or a stack of matrices).

        Parameters
        ----------
        x
            input array or container having shape ``(..., M, N)`` and whose innermost
            two dimensions form ``MxN`` matrices. Should have a floating-point data
            type.

        atol
            absolute tolerance. When None it’s considered to be zero.

        rtol
            relative tolerance for small singular values. Singular values
            approximately less than or equal to ``rtol * largest_singular_value`` are
            set to zero. If a ``float``, the value is equivalent to a zero-dimensional
            array having a floating-point data type determined by :ref:`type-promotion`
            (as applied to ``x``) and must be broadcast against each matrix. If an
            ``array``, must have a floating-point data type and must be compatible with
            ``shape(x)[:-2]`` (see:ref:`broadcasting`). If ``None``, the default value
            is ``max(M, N) * eps``, where ``eps`` must be the machine epsilon associated
            with the floating-point data type determined by :ref:`type-promotion`
            (as applied to ``x``).
            Default: ``None``.

        hermitian
            indicates whether ``x`` is Hermitian. When ``hermitian=True``, ``x`` is
            assumed to be Hermitian, enabling a more efficient method for finding
            eigenvalues, but x is not checked inside the function. Instead, We just use
            the lower triangular of the matrix to compute.
            Default: ``False``.
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
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the ranks. The returned array must have a
            floating-point data type determined by :ref:`type-promotion` and must have
            shape ``(...)`` (i.e., must have a shape equal to ``shape(x)[:-2]``).

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([[1., 0.], [0., 1.]]),
        ...                   b=aikit.array([[1., 0.], [0., 0.]]))
        >>> y = aikit.Container.static_matrix_rank(x)
        >>> print(y)
        {
            a: aikit.array(2.),
            b: aikit.array(1.)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "matrix_rank",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            atol=atol,
            rtol=rtol,
            hermitian=hermitian,
            out=out,
        )

    def matrix_rank(
        self: aikit.Container,
        /,
        *,
        atol: Optional[Union[float, Tuple[float], aikit.Container]] = None,
        rtol: Optional[Union[float, Tuple[float], aikit.Container]] = None,
        hermitian: Optional[Union[bool, aikit.Container]] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.matrix_rank. This
        method returns the rank (i.e., number of non-zero singular values) of a
        matrix (or a stack of matrices).

        Parameters
        ----------
        self
            input container having shape ``(..., M, N)`` and whose innermost two
            dimensions form ``MxN`` matrices. Should have a floating-point data type.

        atol
            absolute tolerance. When None it’s considered to be zero.

        rtol
            relative tolerance for small singular values. Singular values approximately
            less than or equal to ``rtol * largest_singular_value`` are set to zero. If
            a ``float``, the value is equivalent to a zero-dimensional array having a
            floating-point data type determined by :ref:`type-promotion` (as applied to
            ``x``) and must be broadcast against each matrix. If an ``array``, must have
            a floating-point data type and must be compatible with ``shape(x)[:-2]``
            (see :ref:`broadcasting`). If ``None``, the default value is
            ``max(M, N) * eps``, where ``eps`` must be the machine epsilon associated
            with the floating-point data type determined by :ref:`type-promotion`
            (as applied to ``x``). Default: ``None``.

        hermitian
            indicates whether ``x`` is Hermitian. When ``hermitian=True``, ``x`` is
            assumed to be Hermitian, enabling a more efficient method for finding
            eigenvalues, but x is not checked inside the function. Instead, We just use
            the lower triangular of the matrix to compute.
            Default: ``False``.
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
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the ranks. The returned array must have a
            floating-point data type determined by :ref:`type-promotion` and must have
            shape ``(...)`` (i.e., must have a shape equal to ``shape(x)[:-2]``).

        Examples
        --------
        With :class:`aikit.Container` input:
        >>> x = aikit.Container(a=aikit.array([[1., 0.], [0., 1.]]),
        ...                   b=aikit.array([[1., 0.], [0., 0.]]))
        >>> y = x.matrix_rank()
        >>> print(y)
        {
            a: aikit.array(2),
            b: aikit.array(1)
        }
        """
        return self._static_matrix_rank(
            self,
            atol=atol,
            rtol=rtol,
            hermitian=hermitian,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_matrix_transpose(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        conjugate: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """Transpose a matrix (or a stack of matrices) ``x``.

        Parameters
        ----------
        x
            input Container which will have arrays with shape ``(..., M, N)``
            and whose innermost two dimensions form ``MxN`` matrices.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the transposes for each matrix and having shape
            ``(..., N, M)``. The returned array must have the same data
            type as ``x``.


        Examples
        --------
        With :code:`aikit.Container` instance method:

        >>> x = aikit.Container(a=aikit.array([[1., 1.], [0., 3.]]), \
                        b=aikit.array([[0., 4.], [3., 1.]]))
        >>> y = aikit.Container.static_matrix_transpose(x)
        >>> print(y)
        {
            a: aikit.array([[1., 0.],
                          [1., 3.]]),
            b: aikit.array([[0., 3.],
                          [4., 1.]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "matrix_transpose",
            x,
            conjugate=conjugate,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def matrix_transpose(
        self: aikit.Container,
        /,
        *,
        conjugate: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """Transpose a matrix (or a stack of matrices) ``x``.

        Parameters
        ----------
        self
            input Container which will have arrays with shape ``(..., M, N)``
            and whose innermost two dimensions form ``MxN`` matrices.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the transposes for each matrix and having shape
            ``(..., N, M)``. The returned array must have the same data
            type as ``x``.

        Examples
        --------
        With :code:`aikit.Container` instance method:

        >>> x = aikit.Container(a=aikit.array([[1., 1.], [0., 3.]]), \
                      b=aikit.array([[0., 4.], [3., 1.]]))
        >>> y = x.matrix_transpose()
        >>> print(y)
        {
            a: aikit.array([[1., 0.],
                          [1., 3.]]),
            b: aikit.array([[0., 3.],
                          [4., 1.]])
        }
        """
        return self._static_matrix_transpose(
            self,
            conjugate=conjugate,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_outer(
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
        """aikit.Container static method variant of aikit.outer. This method simply
        wraps the function, and so the docstring for aikit.outer also applies to
        this method with minimal changes.

        Computes the outer product of two arrays, x1 and x2,
        by computing the tensor product along the last dimension of both arrays.

        Parameters
        ----------
        x1
            first input array having shape (..., N1)
        x2
            second input array having shape (..., N2)
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to.
            The container must have shape (..., N1, N2). The first x1.ndim-1
            dimensions must have the same size as those of the input array x1
            and the first x2.ndim-1 dimensions must have the same
            size as those of the input array x2.

        Returns
        -------
        ret
            an aikit container whose shape is (..., N1, N2).
            The first x1.ndim-1 dimensions have the same size as those
            of the input array x1 and the first x2.ndim-1
            dimensions have the same size as those of the input array x2.

        Example
        -------
        >>> x1 =aikit.Container( a=aikit.array([[1, 2, 3], [4, 5, 6]]))
        >>> x2 = aikit.Container(a=aikit.array([1, 2, 3]))
        >>> y = aikit.Container.static_outer(x1, x2)
        >>> print(y)
        aikit.array([[[ 1.,  2.,  3.],
                    [ 2.,  4.,  6.],
                    [ 3.,  6.,  9.]],
                   [[ 4.,  8., 12.],
                    [ 5., 10., 15.],
                    [ 6., 12., 18.]]])
        """
        return ContainerBase.cont_multi_map_in_function(
            "outer",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def outer(
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
        """Return the outer product of two arrays or containers.

        The instance method implementation of the static method static_outer of the
        aikit.Container class. It calculates the outer product of two input arrays or
        containers along the last dimension and returns the resulting container. The
        input arrays should be either aikit.Container, aikit.Array, or aikit.NativeArray. The
        output container shape is the concatenation of the shapes of the input
        containers along the last dimension.

        Parameters
        ----------
        self : aikit.Container
            Input container of shape (...,B) where the last dimension
            represents B elements.
        x2 : Union[aikit.Container, aikit.Array, aikit.NativeArray]
            Second input array or container of shape (..., N)
            where the last dimension represents N elements.
        key_chains : Optional[Union[List[str], Dict[str, str]]]
            The key-chains to apply or not apply the method to. Default is None.
        to_apply : bool
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped.Default is True.
        prune_unapplied : bool
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences : bool
            Whether to also map the method to sequences (lists, tuples).
            Default is False.
        out : Optional[aikit.Container]
            Optional output container to write the result to.
            If not provided, a new container will be created.

        Returns
        -------
        aikit.Container
            A new container of shape (..., M, N) representing
            the outer product of the input arrays or containers
            along the last dimension.

        Examples
        --------
        >>> x = aikit.array([[1., 2.],[3., 4.]])
        >>> y = aikit.array([[5., 6.],[7., 8.]])
        >>> d = aikit.outer(x,y)
        >>> print(d)
        aikit.array([[ 5.,  6.,  7.,  8.],
                    [10., 12., 14., 16.],
                    [15., 18., 21., 24.],
                    [20., 24., 28., 32.]])
        """
        return self._static_outer(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_qr(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        mode: Union[str, aikit.Container] = "reduced",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[Tuple[aikit.Container, aikit.Container]] = None,
    ) -> Tuple[aikit.Container, aikit.Container]:
        """aikit.Container static method variant of aikit.qr. This method simply
        wraps the function, and so the docstring for aikit.qr also applies to
        this method with minimal changes.

        Returns the qr decomposition x = QR of a full column rank matrix (or a stack of
        matrices), where Q is an orthonormal matrix (or a stack of matrices) and R is an
        upper-triangular matrix (or a stack of matrices).

        Parameters
        ----------
        x
            input container having shape (..., M, N) and whose innermost two dimensions
            form MxN matrices of rank N. Should have a floating-point data type.
        mode
            decomposition mode. Should be one of the following modes:
            - 'reduced': compute only the leading K columns of q, such that q and r have
            dimensions (..., M, K) and (..., K, N), respectively, and where
            K = min(M, N).
            - 'complete': compute q and r with dimensions (..., M, M) and (..., M, N),
            respectively.
            Default: 'reduced'.
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
            optional output tuple of containers, for writing the result to. The arrays
            must have shapes that the inputs broadcast to.

        Returns
        -------
        ret
            a namedtuple (Q, R) whose
            - first element must have the field name Q and must be an container whose
            shape depends on the value of mode and contain matrices with orthonormal
            columns. If mode is 'complete', the container must have shape (..., M, M).
            If mode is 'reduced', the container must have shape (..., M, K), where
            K = min(M, N). The first x.ndim-2 dimensions must have the same size as
            those of the input container x.
            - second element must have the field name R and must be an container whose
            shape depends on the value of mode and contain upper-triangular matrices. If
            mode is 'complete', the container must have shape (..., M, N). If mode is
            'reduced', the container must have shape (..., K, N), where K = min(M, N).
            The first x.ndim-2 dimensions must have the same size as those of the input
            x.

        Examples
        --------
        >>> x = aikit.Container(a = aikit.native_array([[1., 2.], [3., 4.]]),
        ...                   b = aikit.array([[2., 3.], [4. ,5.]]))
        >>> q,r = aikit.Container.static_qr(x, mode='complete')
        >>> print(q)
        {
            a: aikit.array([[-0.31622777, -0.9486833],
                        [-0.9486833, 0.31622777]]),
            b: aikit.array([[-0.4472136, -0.89442719],
                        [-0.89442719, 0.4472136]])
        }
        >>> print(r)
        {
            a: aikit.array([[-3.16227766, -4.42718872],
                        [0., -0.63245553]]),
            b: aikit.array([[-4.47213595, -5.81377674],
                        [0., -0.4472136]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "qr",
            x,
            mode=mode,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def qr(
        self: aikit.Container,
        /,
        *,
        mode: Union[str, aikit.Container] = "reduced",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[Tuple[aikit.Container, aikit.Container]] = None,
    ) -> Tuple[aikit.Container, aikit.Container]:
        """aikit.Container instance method variant of aikit.qr. This method simply
        wraps the function, and so the docstring for aikit.qr also applies to
        this method with minimal changes.

        Returns the qr decomposition x = QR of a full column rank matrix (or a stack of
        matrices), where Q is an orthonormal matrix (or a stack of matrices) and R is an
        upper-triangular matrix (or a stack of matrices).

        Parameters
        ----------
        self
            input container having shape (..., M, N) and whose innermost two dimensions
            form MxN matrices of rank N. Should have a floating-point data type.
        mode
            decomposition mode. Should be one of the following modes:
            - 'reduced': compute only the leading K columns of q, such that q and r have
            dimensions (..., M, K) and (..., K, N), respectively, and where
            K = min(M, N).
            - 'complete': compute q and r with dimensions (..., M, M) and (..., M, N),
            respectively.
            Default: 'reduced'.
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
            optional output tuple of containers, for writing the result to. The arrays
            must have shapes that the inputs broadcast to.

        Returns
        -------
        ret
            a namedtuple (Q, R) whose
            - first element must have the field name Q and must be an container whose
            shape depends on the value of mode and contain matrices with orthonormal
            columns. If mode is 'complete', the container must have shape (..., M, M).
            If mode is 'reduced', the container must have shape (..., M, K), where
            K = min(M, N). The first x.ndim-2 dimensions must have the same size as
            those of the input container x.
            - second element must have the field name R and must be an container whose
            shape depends on the value of mode and contain upper-triangular matrices. If
            mode is 'complete', the container must have shape (..., M, N). If mode is
            'reduced', the container must have shape (..., K, N), where K = min(M, N).
            The first x.ndim-2 dimensions must have the same size as those of the input
            x.

        Examples
        --------
        >>> x = aikit.Container(a = aikit.native_array([[1., 2.], [3., 4.]]),
        ...                   b = aikit.array([[2., 3.], [4. ,5.]]))
        >>> q,r = x.qr(mode='complete')
        >>> print(q)
        {
            a: aikit.array([[-0.31622777, -0.9486833],
                        [-0.9486833, 0.31622777]]),
            b: aikit.array([[-0.4472136, -0.89442719],
                        [-0.89442719, 0.4472136]])
        }
        >>> print(r)
        {
            a: aikit.array([[-3.16227766, -4.42718872],
                        [0., -0.63245553]]),
            b: aikit.array([[-4.47213595, -5.81377674],
                        [0., -0.4472136]])
        }
        """
        return self._static_qr(
            self,
            mode=mode,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_slogdet(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.slogdet. This method
        simply wraps the function, and so the docstring for aikit.slogdet also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array or container having shape (..., M, M) and whose innermost two
            dimensions form square matrices. Should have a floating-point data type.
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
            This function returns a container containing NamedTuples.
            Each NamedTuple of output will have -
                sign:
                An array containing a number representing the sign of the determinant
                for each square matrix.

                logabsdet:
                An array containing natural log of the absolute determinant of each
                square matrix.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[1.0, 2.0],
        ...                                [3.0, 4.0]]),
        ...                   b=aikit.array([[1.0, 2.0],
        ...                                [2.0, 1.0]]))
        >>> y = aikit.Container.static_slogdet(x)
        >>> print(y)
        {
            a: [
                sign = aikit.array(-1.),
                logabsdet = aikit.array(0.6931472)
            ],
            b: [
                sign = aikit.array(-1.),
                logabsdet = aikit.array(1.0986123)
            ]
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "slogdet",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def slogdet(
        self: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.slogdet. This method
        simply wraps the function, and so the docstring for aikit.slogdet also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container having shape (..., M, M) and whose innermost two dimensions
            form square matrices. Should have a floating-point data type.
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
            This function returns container containing NamedTuples.
            Each NamedTuple of output will have -
                sign:
                An array of a number representing the sign of the determinant of each
                square.

                logabsdet:
                An array of the natural log of the absolute value of the determinant of
                each square.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[1.0, 2.0],
        ...                                [3.0, 4.0]]),
        ...                   b=aikit.array([[1.0, 2.0],
        ...                                [2.0, 1.0]]))
        >>> y = x.slogdet()
        >>> print(y)
        [{
            a: aikit.array(-1.),
            b: aikit.array(-1.)
        }, {
            a: aikit.array(0.69314718),
            b: aikit.array(1.09861231)
        }]
        """
        return self._static_slogdet(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_solve(
        x1: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        x2: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        adjoint: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "solve",
            x1,
            x2,
            adjoint=adjoint,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def solve(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        adjoint: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return self._static_solve(
            self,
            x2,
            adjoint=adjoint,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_svd(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        compute_uv: Union[bool, aikit.Container] = True,
        full_matrices: Union[bool, aikit.Container] = True,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> Union[aikit.Container, Tuple[aikit.Container, ...]]:
        """aikit.Container static method variant of aikit.svd. This method simply
        wraps the function, and so the docstring for aikit.svd also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container with array leaves having shape ``(..., M, N)`` and whose
            innermost two dimensions form matrices on which to perform singular value
            decomposition. Should have a floating-point data type.
        full_matrices
            If ``True``, compute full-sized ``U`` and ``Vh``, such that ``U`` has
            shape ``(..., M, M)`` and ``Vh`` has shape ``(..., N, N)``. If ``False``,
            compute on             the leading ``K`` singular vectors, such that ``U``
            has shape ``(..., M, K)`` and ``Vh`` has shape ``(..., K, N)`` and where
            ``K = min(M, N)``. Default: ``True``.
        compute_uv
            If ``True`` then left and right singular vectors will be computed and
            returned in ``U`` and ``Vh``, respectively. Otherwise, only the singular
            values will be computed, which can be significantly faster.
        .. note::
            with backend set as torch, svd with still compute left and right singular
            vectors irrespective of the value of compute_uv, however Aikit will
            still only return the
            singular values.

        Returns
        -------
        .. note::
            once complex numbers are supported, each square matrix must be Hermitian.

        ret
            A container of a namedtuples ``(U, S, Vh)``. More details in aikit.svd.


        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.random_normal(shape = (9, 6))
        >>> y = aikit.random_normal(shape = (2, 4))
        >>> z = aikit.Container(a=x, b=y)
        >>> ret = aikit.Container.static_svd(z)
        >>> aU, aS, aVh = ret.a
        >>> bU, bS, bVh = ret.b
        >>> print(aU.shape, aS.shape, aVh.shape, bU.shape, bS.shape, bVh.shape)
        (9, 9) (6,) (6, 6) (2, 2) (2,) (4, 4)
        """
        return ContainerBase.cont_multi_map_in_function(
            "svd",
            x,
            compute_uv=compute_uv,
            full_matrices=full_matrices,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def svd(
        self: aikit.Container,
        /,
        *,
        compute_uv: Union[bool, aikit.Container] = True,
        full_matrices: Union[bool, aikit.Container] = True,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.svd. This method simply
        wraps the function, and so the docstring for aikit.svd also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container with array leaves having shape ``(..., M, N)`` and whose
            innermost two dimensions form matrices on which to perform singular value
            decomposition. Should have a floating-point data type.
        full_matrices
            If ``True``, compute full-sized ``U`` and ``Vh``, such that ``U`` has
            shape ``(..., M, M)`` and ``Vh`` has shape ``(..., N, N)``. If ``False``,
            compute on             the leading ``K`` singular vectors, such that ``U``
            has shape ``(..., M, K)`` and ``Vh`` has shape ``(..., K, N)`` and where
            ``K = min(M, N)``. Default: ``True``.
        compute_uv
            If ``True`` then left and right singular vectors will be computed and
            returned in ``U`` and ``Vh``, respectively. Otherwise, only the singular
            values will be computed, which can be significantly faster.
        .. note::
            with backend set as torch, svd with still compute left and right singular
            vectors irrespective of the value of compute_uv, however Aikit will
            still only return the
            singular values.

        Returns
        -------
        .. note::
            once complex numbers are supported, each square matrix must be Hermitian.

        ret
            A container of a namedtuples ``(U, S, Vh)``. More details in aikit.svd.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.random_normal(shape = (9, 6))
        >>> y = aikit.random_normal(shape = (2, 4))
        >>> z = aikit.Container(a=x, b=y)
        >>> ret = z.svd()
        >>> print(ret[0], ret[1], ret[2])
        {
            a: (<class aikit.data_classes.array.array.Array> shape=[9, 9]),
            b: aikit.array([[-0.3475602, -0.93765765],
                          [-0.93765765, 0.3475602]])
        } {
            a: aikit.array([3.58776021, 3.10416126, 2.80644298, 1.87024701, 1.48127627,
                          0.79101127]),
            b: aikit.array([1.98288572, 0.68917423])
        } {
            a: (<class aikit.data_classes.array.array.Array> shape=[6, 6]),
            b: (<class aikit.data_classes.array.array.Array> shape=[4, 4])
        }
        """
        return self._static_svd(
            self,
            compute_uv=compute_uv,
            full_matrices=full_matrices,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_svdvals(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "svdvals",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def svdvals(
        self: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return self._static_svdvals(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_tensordot(
        x1: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        x2: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        axes: Union[int, Tuple[List[int], List[int]], aikit.Container] = 2,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "tensordot",
            x1,
            x2,
            axes=axes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def tensordot(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        axes: Union[int, Tuple[List[int], List[int]], aikit.Container] = 2,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return self._static_tensordot(
            self,
            x2,
            axes=axes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_tensorsolve(
        x1: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        x2: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        axes: Optional[Union[int, Tuple[List[int], List[int]], aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "tensorsolve",
            x1,
            x2,
            axes=axes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def tensorsolve(
        self: aikit.Container,
        x2: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        axes: Optional[Union[int, Tuple[List[int], List[int]], aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return self._static_tensorsolve(
            self,
            x2,
            axes=axes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_trace(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        offset: Union[int, aikit.Container] = 0,
        axis1: Union[int, aikit.Container] = 0,
        axis2: Union[int, aikit.Container] = 1,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.trace. This method
        Returns the sum along the specified diagonals of a matrix (or a stack
        of matrices).

        Parameters
        ----------
        x
            input container having shape ``(..., M, N)`` and whose innermost two
            dimensions form ``MxN`` matrices. Should have a floating-point data type.
        offset
            Offset of the diagonal from the main diagonal. Can be both positive and
            negative. Defaults to 0.
        axis1
            axis to be used as the first axis of the 2-D sub-arrays from which the
            diagonals should be taken.
            Defaults to ``0.`` .
        axis2
            axis to be used as the second axis of the 2-D sub-arrays from which the
            diagonals should be taken.
            Defaults to ``1.`` .
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
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the traces and whose shape is determined by removing
            the last two dimensions and storing the traces in the last array dimension.
            For example, if ``x`` has rank ``k`` and shape ``(I, J, K, ..., L, M, N)``,
            then an output array has rank ``k-2`` and shape ``(I, J, K, ..., L)`` where

            ::

            out[i, j, k, ..., l] = trace(a[i, j, k, ..., l, :, :])

            The returned array must have the same data type as ``x``.

        Examples
        --------
        With :class:`aikit.Container` input:
        >>> x = aikit.Container(
        ...    a = aikit.array([[7, 1, 2],
        ...                   [1, 3, 5],
        ...                   [0, 7, 4]]),
        ...    b = aikit.array([[4, 3, 2],
        ...                   [1, 9, 5],
        ...                   [7, 0, 6]])
        )
        >>> y = x.Container.static_trace(x)
        >>> print(y)
        {
            a: aikit.array(14),
            b: aikit.array(19)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "trace",
            x,
            offset=offset,
            axis1=axis1,
            axis2=axis2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def trace(
        self: aikit.Container,
        /,
        *,
        offset: Union[int, aikit.Container] = 0,
        axis1: Union[int, aikit.Container] = 0,
        axis2: Union[int, aikit.Container] = 1,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.trace. This method
        Returns the sum along the specified diagonals of a matrix (or a stack
        of matrices).

        Parameters
        ----------
        self
            input container having shape ``(..., M, N)`` and whose innermost two
            dimensions form ``MxN`` matrices. Should have a floating-point data type.
        offset
            Offset of the diagonal from the main diagonal. Can be both positive and
            negative. Defaults to 0.
        axis1
            axis to be used as the first axis of the 2-D sub-arrays from which the
            diagonals should be taken.
            Defaults to ``0.`` .
        axis2
            axis to be used as the second axis of the 2-D sub-arrays from which the
            diagonals should be taken.
            Defaults to ``1.`` .
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
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the traces and whose shape is determined by removing
            the last two dimensions and storing the traces in the last array dimension.
            For example, if ``x`` has rank ``k`` and shape ``(I, J, K, ..., L, M, N)``,
            then an output array has rank ``k-2`` and shape ``(I, J, K, ..., L)`` where

            ::

            out[i, j, k, ..., l] = trace(a[i, j, k, ..., l, :, :])

            The returned array must have the same data type as ``x``.

        Examples
        --------
        With :class:`aikit.Container` input:
        >>> x = aikit.Container(
        ...    a = aikit.array([[7, 1, 2],
        ...                   [1, 3, 5],
        ...                   [0, 7, 4]]),
        ...    b = aikit.array([[4, 3, 2],
        ...                   [1, 9, 5],
        ...                   [7, 0, 6]]))
        >>> y = x.trace()
        >>> print(y)
        {
            a: aikit.array(14),
            b: aikit.array(19)
        }
        """
        return self._static_trace(
            self,
            offset=offset,
            axis1=axis1,
            axis2=axis2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_vecdot(
        x1: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        axis: Union[int, aikit.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "vecdot",
            x1,
            x2,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def vecdot(
        self: aikit.Container,
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        axis: Union[int, aikit.Container] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return self._static_vecdot(
            self,
            x2,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_vector_norm(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        axis: Optional[Union[int, Sequence[int], aikit.Container]] = None,
        keepdims: Union[bool, aikit.Container] = False,
        ord: Union[int, float, Literal[inf, -inf], aikit.Container] = 2,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.vector_norm. This method
        simply wraps the function, and so the docstring for aikit.vector_norm
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array. Should have a floating-point data type.
        axis
            If an integer, ``axis`` specifies the axis (dimension)
            along which to compute vector norms. If an n-tuple,
            ``axis`` specifies the axes (dimensions) along
            which to compute batched vector norms. If ``None``, the
             vector norm must be computed over all array values
             (i.e., equivalent to computing the vector norm of
            a flattened array). Negative indices must be
            supported. Default: ``None``.
        keepdims
            If ``True``, the axes (dimensions) specified by ``axis``
            must be included in the result as singleton dimensions,
            and, accordingly, the result must be compatible
            with the input array (see :ref:`broadcasting`). Otherwise,
            if ``False``, the axes (dimensions) specified by ``axis`` must
            not be included in the result. Default: ``False``.
        ord
            order of the norm. The following mathematical norms must be supported:

            +------------------+----------------------------+
            | ord              | description                |
            +==================+============================+
            | 1                | L1-norm (Manhattan)        |
            +------------------+----------------------------+
            | 2                | L2-norm (Euclidean)        |
            +------------------+----------------------------+
            | inf              | infinity norm              |
            +------------------+----------------------------+
            | (int,float >= 1) | p-norm                     |
            +------------------+----------------------------+

            The following non-mathematical "norms" must be supported:

            +------------------+--------------------------------+
            | ord              | description                    |
            +==================+================================+
            | 0                | sum(a != 0)                    |
            +------------------+--------------------------------+
            | -1               | 1./sum(1./abs(a))              |
            +------------------+--------------------------------+
            | -2               | 1./sqrt(sum(1./abs(a)/*/*2))   | # noqa
            +------------------+--------------------------------+
            | -inf             | min(abs(a))                    |
            +------------------+--------------------------------+
            | (int,float < 1)  | sum(abs(a)/*/*ord)/*/*(1./ord) |
            +------------------+--------------------------------+

            Default: ``2``.
        dtype
            data type that may be used to perform the computation more precisely. The
            input array ``x`` gets cast to ``dtype`` before the function's computations.
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the vector norms. If ``axis`` is
            ``None``, the returned array must be a zero-dimensional
            array containing a vector norm. If ``axis`` is
            a scalar value (``int`` or ``float``), the returned array
            must have a rank which is one less than the rank of ``x``.
            If ``axis`` is a ``n``-tuple, the returned array must have
             a rank which is ``n`` less than the rank of ``x``. The returned
            array must have a floating-point data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a = [1., 2., 3.], b = [-2., 0., 3.2])
        >>> y = aikit.Container.static_vector_norm(x)
        >>> print(y)
        {
            a: aikit.array([3.7416575]),
            b: aikit.array([3.77359247])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "vector_norm",
            x,
            axis=axis,
            keepdims=keepdims,
            ord=ord,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def vector_norm(
        self: aikit.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], aikit.Container]] = None,
        keepdims: Union[bool, aikit.Container] = False,
        ord: Union[int, float, Literal[inf, -inf], aikit.Container] = 2,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        r"""aikit.Container instance method variant of aikit.vector_norm. This
        method simply wraps the function, and so the docstring for
        aikit.vector_norm also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a floating-point data type.
        axis
            If an integer, ``axis`` specifies the axis (dimension)
            along which to compute vector norms. If an n-tuple, ``axis``
            specifies the axes (dimensions) along which to compute
            batched vector norms. If ``None``, the vector norm must be
            computed over all array values (i.e., equivalent to computing
            the vector norm of a flattened array). Negative indices must
            be supported. Default: ``None``.
        keepdims
            If ``True``, the axes (dimensions) specified by ``axis`` must
            be included in the result as singleton dimensions, and, accordingly,
            the result must be compatible with the input array
            (see :ref:`broadcasting`).Otherwise, if ``False``, the axes
            (dimensions) specified by ``axis`` must not be included in
            the result. Default: ``False``.
        ord
            order of the norm. The following mathematical norms must be supported:

            +------------------+----------------------------+
            | ord              | description                |
            +==================+============================+
            | 1                | L1-norm (Manhattan)        |
            +------------------+----------------------------+
            | 2                | L2-norm (Euclidean)        |
            +------------------+----------------------------+
            | inf              | infinity norm              |
            +------------------+----------------------------+
            | (int,float >= 1) | p-norm                     |
            +------------------+----------------------------+

            The following non-mathematical "norms" must be supported:

            +------------------+--------------------------------+
            | ord              | description                    |
            +==================+================================+
            | 0                | sum(a != 0)                    |
            +------------------+--------------------------------+
            | -1               | 1./sum(1./abs(a))              |
            +------------------+--------------------------------+
            | -2               | 1./sqrt(sum(1./abs(a)/*/*2))   | # noqa
            +------------------+--------------------------------+
            | -inf             | min(abs(a))                    |
            +------------------+--------------------------------+
            | (int,float < 1)  | sum(abs(a)/*/*ord)/*/*(1./ord) |
            +------------------+--------------------------------+

            Default: ``2``.
        dtype
            data type that may be used to perform the computation more precisely. The
            input array ``x`` gets cast to ``dtype`` before the function's computations.
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the vector norms. If ``axis`` is ``None``,
            the returned array must be a zero-dimensional array containing
            a vector norm. If ``axis`` is a scalar value (``int`` or ``float``),
            the returned array must have a rank which is one less than the
            rank of ``x``. If ``axis`` is a ``n``-tuple, the returned
            array must have a rank which is ``n`` less than the rank of
            ``x``. The returned array must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.Container(a = [1., 2., 3.], b = [-2., 0., 3.2])
        >>> y = x.vector_norm()
        >>> print(y)
        {
            a: aikit.array([3.7416575]),
            b: aikit.array([3.77359247])
        }
        """
        return self._static_vector_norm(
            self,
            axis=axis,
            keepdims=keepdims,
            ord=ord,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_vector_to_skew_symmetric_matrix(
        vector: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "vector_to_skew_symmetric_matrix",
            vector,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def vector_to_skew_symmetric_matrix(
        self: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return self._static_vector_to_skew_symmetric_matrix(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_vander(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        N: Optional[Union[int, aikit.Container]] = None,
        increasing: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.vander. This method
        simply wraps the function, and so the docstring for aikit.vander also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            aikit container that contains 1-D arrays.
        N
            Number of columns in the output. If N is not specified,
            a square array is returned (N = len(x))
        increasing
            Order of the powers of the columns. If True, the powers increase
            from left to right, if False (the default) they are reversed.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container that contains the Vandermonde matrix of the arrays included
            in the input container.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x = aikit.Container(
                a = aikit.array([1, 2, 3, 5])
                b = aikit.array([6, 7, 8, 9])
            )
        >>> aikit.Container.static_vander(x)
        {
            a: aikit.array(
                    [[  1,   1,   1,   1],
                    [  8,   4,   2,   1],
                    [ 27,   9,   3,   1],
                    [125,  25,   5,   1]]
                    ),
            b: aikit.array(
                    [[216,  36,   6,   1],
                    [343,  49,   7,   1],
                    [512,  64,   8,   1],
                    [729,  81,   9,   1]]
                    )
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "vander",
            x,
            N=N,
            increasing=increasing,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def vander(
        self: aikit.Container,
        /,
        *,
        N: Optional[Union[int, aikit.Container]] = None,
        increasing: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.vander. This method
        Returns the Vandermonde matrix of the input array.

        Parameters
        ----------
        self
            1-D input array.
        N
            Number of columns in the output. If N is not specified,
            a square array is returned (N = len(x))
        increasing
            Order of the powers of the columns. If True, the powers increase
            from left to right, if False (the default) they are reversed.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            an container containing the Vandermonde matrices of the arrays
            included in the input container.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x = aikit.Container(
                a = aikit.array([1, 2, 3, 5])
                b = aikit.array([6, 7, 8, 9])
            )
        >>> x.vander()
        {
            a: aikit.array(
                    [[  1,   1,   1,   1],
                    [  8,   4,   2,   1],
                    [ 27,   9,   3,   1],
                    [125,  25,   5,   1]]
                    ),
            b: aikit.array(
                    [[216,  36,   6,   1],
                    [343,  49,   7,   1],
                    [512,  64,   8,   1],
                    [729,  81,   9,   1]]
                    )
        }
        """
        return self._static_vander(
            self,
            N=N,
            increasing=increasing,
            out=out,
        )

    @staticmethod
    def static_general_inner_product(
        x1: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        n_modes: Optional[Union[int, aikit.Container]] = None,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.general_inner_product.
        This method simply wraps the function, and so the docstring for
        aikit.general_inner_product also applies to this method with minimal
        changes.

        Parameters
        ----------
        x1
            First input container containing input array.
        x2
            First input container containing input array.
        n_modes
            int, default is None. If None, the traditional inner product is returned
            (i.e. a float) otherwise, the product between the `n_modes` last modes of
            `x1` and the `n_modes` first modes of `x2` is returned. The resulting
            tensor's order is `len(x1) - n_modes`.
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
            Container including the inner product tensor.

        Examples
        --------
        >>> x = aikit.Container(
                a=aikit.reshape(aikit.arange(4), (2, 2)),
                b=aikit.reshape(aikit.arange(8), (2, 4)),
            )
        >>> aikit.Container.general_inner_product(x, 1)
            {
                a: aikit.array(6),
                b: aikit.array(28)
            }
        """
        return ContainerBase.cont_multi_map_in_function(
            "general_inner_product",
            x1,
            x2,
            n_modes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def general_inner_product(
        self: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        n_modes: Optional[Union[int, aikit.Container]] = None,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.general_inner_product.

        This method simply wraps the function, and so the docstring for
        aikit.general_inner_product also applies to this method with
        minimal changes.
        """
        return self.static_general_inner_product(
            self,
            x2,
            n_modes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
