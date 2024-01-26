# For Review
# global
from typing import (
    Optional,
    Union,
    List,
    Tuple,
    Dict,
    Iterable,
    Sequence,
)
from numbers import Number

# local
import aikit
from aikit.data_classes.container.base import ContainerBase


class _ContainerWithManipulation(ContainerBase):
    @staticmethod
    def _static_concat(
        xs: Union[
            Tuple[Union[aikit.Array, aikit.NativeArray, aikit.Container], ...],
            List[Union[aikit.Array, aikit.NativeArray, aikit.Container]],
        ],
        /,
        *,
        axis: Union[int, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.concat.

        This method simply wraps the function, and so the docstring for
        aikit.concat also applies to this method with minimal changes.
        """
        return ContainerBase.cont_multi_map_in_function(
            "concat",
            xs,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def concat(
        self: aikit.Container,
        /,
        xs: Union[
            Tuple[Union[aikit.Array, aikit.NativeArray, aikit.Container], ...],
            List[Union[aikit.Array, aikit.NativeArray, aikit.Container]],
        ],
        *,
        axis: Union[int, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.concat.

        This method simply wraps the function, and so the docstring for
        aikit.concat also applies to this method with minimal changes.
        """
        new_xs = xs.cont_copy() if aikit.is_aikit_container(xs) else xs.copy()
        new_xs.insert(0, self.cont_copy())
        return self._static_concat(
            new_xs,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_expand_dims(
        x: aikit.Container,
        /,
        *,
        copy: Optional[Union[bool, aikit.Container]] = None,
        axis: Union[int, Sequence[int], aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.expand_dims. This method
        simply wraps the function, and so the docstring for aikit.expand_dims
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        axis
            position where a new axis (dimension) of size one will be added. If an
            element of the container has the rank of ``N``, then the ``axis`` needs
            to be between ``[-N-1, N]``. Default: ``0``.
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
            A container with the elements of ``x``, but with the dimensions of
            its elements added by one in a given ``axis``.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([0., 1.]),
        ...                   b=aikit.array([3., 4.]),
        ...                   c=aikit.array([6., 7.]))
        >>> y = aikit.Container.static_expand_dims(x, axis=1)
        >>> print(y)
        {
            a: aikit.array([[0.],
                          [1.]]),
            b: aikit.array([[3.],
                          [4.]]),
            c: aikit.array([[6.],
                          [7.]])
        }

        With multiple :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),
        ...                   b=aikit.array([3., 4., 5.]),
        ...                   c=aikit.array([6., 7., 8.]))
        >>> container_axis = aikit.Container(a=0, b=-1, c=(0,1))
        >>> y = aikit.Container.static_expand_dims(x, axis=container_axis)
        >>> print(y)
        {
            a: aikit.array([[0., 1., 2.]]),
            b: aikit.array([[3.],
                          [4.],
                          [5.]]),
            c: aikit.array([[[6., 7., 8.]]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "expand_dims",
            x,
            copy=copy,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def expand_dims(
        self: aikit.Container,
        /,
        *,
        copy: Optional[Union[bool, aikit.Container]] = None,
        axis: Union[int, Sequence[int], aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.expand_dims. This
        method simply wraps the function, and so the docstring for
        aikit.expand_dims also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        axis
            position where a new axis (dimension) of size one will be added. If an
            element of the container has the rank of ``N``, the ``axis`` needs to
            be between ``[-N-1, N]``. Default: ``0``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the elements of ``self``, but with the dimensions of
            its elements added by one in a given ``axis``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[0., 1.],
        ...                                [2., 3.]]),
        ...                   b=aikit.array([[4., 5.],
        ...                                [6., 7.]]))
        >>> y = x.expand_dims(axis=1)
        >>> print(y)
        {
            a: aikit.array([[[0., 1.]],
                          [[2., 3.]]]),
            b: aikit.array([[[4., 5.]],
                          [[6., 7.]]])
        }
        """
        return self._static_expand_dims(
            self,
            copy=copy,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_split(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        copy: Optional[Union[bool, aikit.Container]] = None,
        num_or_size_splits: Optional[
            Union[int, Sequence[int], aikit.Array, aikit.NativeArray, aikit.Container]
        ] = None,
        axis: Union[int, aikit.Container] = 0,
        with_remainder: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> List[aikit.Container]:
        """aikit.Container static method variant of aikit.split. This method simply
        wraps the function, and so the docstring for aikit.split also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            array to be divided into sub-arrays.
        num_or_size_splits
            Number of equal arrays to divide the array into along the given axis if an
            integer. The size of each split element if a sequence of integers
            or 1-D array. Default is to divide into as many 1-dimensional arrays
            as the axis dimension.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        axis
            The axis along which to split, default is ``0``.
        with_remainder
            If the tensor does not split evenly, then store the last remainder entry.
            Default is ``False``.
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
            list of containers of sub-arrays.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([2, 1, 5, 9]), b=aikit.array([3, 7, 2, 11]))
        >>> y = aikit.Container.static_split(x, num_or_size_splits=2)
        >>> print(y)
        [{
            a: aikit.array([2, 1]),
            b: aikit.array([3, 7])
        }, {
            a: aikit.array([5, 9]),
            b: aikit.array([2, 11])
        }]
        """
        return ContainerBase.cont_multi_map_in_function(
            "split",
            x,
            copy=copy,
            num_or_size_splits=num_or_size_splits,
            axis=axis,
            with_remainder=with_remainder,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def split(
        self: aikit.Container,
        /,
        *,
        copy: Optional[Union[bool, aikit.Container]] = None,
        num_or_size_splits: Optional[
            Union[int, Sequence[int], aikit.Array, aikit.NativeArray, aikit.Container]
        ] = None,
        axis: Union[int, aikit.Container] = 0,
        with_remainder: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> List[aikit.Container]:
        """aikit.Container instance method variant of aikit.split. This method
        simply wraps the function, and so the docstring for aikit.split also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            array to be divided into sub-arrays.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        num_or_size_splits
            Number of equal arrays to divide the array into along the given axis if an
            integer. The size of each split element if a sequence of integers
            or 1-D array. Default is to divide into as many 1-dimensional arrays
            as the axis dimension.
        axis
            The axis along which to split, default is ``0``.
        with_remainder
            If the tensor does not split evenly, then store the last remainder entry.
            Default is ``False``.
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
            list of containers of sub-arrays.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([2, 1, 5, 9]), b=aikit.array([3, 7, 2, 11]))
        >>> y = x.split(num_or_size_splits=2)
        >>> print(y)
        [{
            a: aikit.array([2, 1]),
            b: aikit.array([3, 7])
        }, {
            a: aikit.array([5, 9]),
            b: aikit.array([2, 11])
        }]
        """
        return self._static_split(
            self,
            copy=copy,
            num_or_size_splits=num_or_size_splits,
            axis=axis,
            with_remainder=with_remainder,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_permute_dims(
        x: aikit.Container,
        /,
        axes: Union[Tuple[int, ...], aikit.Container],
        *,
        copy: Optional[Union[bool, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.permute_dims. This method
        simply wraps the function, and so the docstring for aikit.permute_dims
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        axes
            tuple containing a permutation of (0, 1, ..., N-1) where N is the number
            of axes (dimensions) of x.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the elements of ``self`` permuted along the given axes.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[0., 1., 2.]]), b=aikit.array([[3., 4., 5.]]))
        >>> y = aikit.Container.static_permute_dims(x, axes=(1, 0))
        >>> print(y)
        {
            a:aikit.array([[0.],[1.],[2.]]),
            b:aikit.array([[3.],[4.],[5.]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "permute_dims",
            x,
            axes,
            copy=copy,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def permute_dims(
        self: aikit.Container,
        /,
        axes: Union[Tuple[int, ...], aikit.Container],
        *,
        copy: Optional[Union[bool, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.permute_dims. This
        method simply wraps the function, and so the docstring for
        aikit.permute_dims also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        axes
            tuple containing a permutation of (0, 1, ..., N-1) where N is the number
            of axes (dimensions) of x.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the elements of ``self`` permuted along the given axes.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[0., 1., 2.]]), b=aikit.array([[3., 4., 5.]]))
        >>> y = x.permute_dims(axes=(1, 0))
        >>> print(y)
        {
            a:aikit.array([[0.],[1.],[2.]]),
            b:aikit.array([[3.],[4.],[5.]])
        }
        """
        return self._static_permute_dims(
            self,
            axes,
            copy=copy,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_flip(
        x: aikit.Container,
        /,
        *,
        copy: Optional[Union[bool, aikit.Container]] = None,
        axis: Optional[Union[int, Sequence[int], aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.flip. This method simply
        wraps the function, and so the docstring for aikit.flip also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        axis
            axis (or axes) along which to flip. If axis is None,
            all input array axes are flipped. If axis is negative,
            axis is counted from the last dimension. If provided more
            than one axis, only the specified axes. Default: None.
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
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an output container having the same data type and
            shape as ``x`` and whose elements, relative to ``x``, are reordered.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([-1, 0, 1]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container.static_flip(x)
        >>> print(y)
        {
            a: aikit.array([1, 0, -1]),
            b: aikit.array([4, 3, 2])
        }

        >>> x = aikit.Container(a=aikit.array([-1, 0, 1]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = aikit.Container.static_flip(x, axis=0)
        >>> print(y)
        {
            a: aikit.array([1, 0, -1]),
            b: aikit.array([4, 3, 2])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "flip",
            x,
            copy=copy,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def flip(
        self: aikit.Container,
        /,
        *,
        copy: Optional[Union[bool, aikit.Container]] = None,
        axis: Optional[Union[int, Sequence[int], aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.flip. This method
        simply wraps the function, and so the docstring for aikit.flip also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        axis
            axis (or axes) along which to flip. If axis is None,
            all input array axes are flipped. If axis is negative,
            axis is counted from the last dimension. If provided
            more than one axis, only the specified axes. Default: None.
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
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an output container having the same data type and
            shape as ``self`` and whose elements, relative to ``self``, are reordered.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([-1, 0, 1]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = x.flip()
        >>> print(y)
        {
            a: aikit.array([1, 0, -1]),
            b: aikit.array([4, 3, 2])
        }

        >>> x = aikit.Container(a=aikit.array([-1, 0, 1]),
        ...                   b=aikit.array([2, 3, 4]))
        >>> y = x.flip(axis=0)
        >>> print(y)
        {
            a: aikit.array([1, 0, -1]),
            b: aikit.array([4, 3, 2])
        }
        """
        return self._static_flip(
            self,
            copy=copy,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_reshape(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        shape: Union[aikit.Shape, aikit.NativeShape, Sequence[int], aikit.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        copy: Optional[Union[bool, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
        order: Union[str, aikit.Container] = "C",
        allowzero: Union[bool, aikit.Container] = True,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.reshape. This method
        simply wraps the function, and so the docstring for aikit.reshape also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.

        shape
            The new shape should be compatible with the original shape.
            One shape dimension can be -1. In this case, the value is
            inferred from the length of the array and remaining dimensions.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
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
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.
        order
            Read the elements of x using this index order, and place the elements into
            the reshaped array using this index order.
            ‘C’ means to read / write the elements using C-like index order,
            with the last axis index changing fastest, back to the first axis index
            changing slowest.
            ‘F’ means to read / write the elements using Fortran-like index order, with
            the first index changing fastest, and the last index changing slowest.
            Note that the ‘C’ and ‘F’ options take no account of the memory layout
            of the underlying array, and only refer to the order of indexing.
            Default order is 'C'

        Returns
        -------
        ret
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([0, 1, 2, 3, 4, 5]),
        ...                   b=aikit.array([0, 1, 2, 3, 4, 5]))
        >>> y = aikit.Container.static_reshape(x, (3,2))
        >>> print(y)
        {
            a: aikit.array([[0, 1],
                          [2, 3],
                          [4, 5]]),
            b: aikit.array([[0, 1],
                          [2, 3],
                          [4, 5]])
        }

        >>> x = aikit.Container(a=aikit.array([0, 1, 2, 3, 4, 5]),
        ...                   b=aikit.array([0, 1, 2, 3, 4, 5]))
        >>> y = aikit.Container.static_reshape(x, (3,2), order='F')
        >>> print(y)
        {
            a: aikit.array([[0, 3],
                          [1, 4],
                          [2, 5]]),
            b: aikit.array([[0, 3],
                          [1, 4],
                          [2, 5]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "reshape",
            x,
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            copy=copy,
            allowzero=allowzero,
            out=out,
            order=order,
        )

    def reshape(
        self: aikit.Container,
        /,
        shape: Union[aikit.Shape, aikit.NativeShape, Sequence[int], aikit.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        copy: Optional[Union[bool, aikit.Container]] = None,
        order: Union[str, aikit.Container] = "C",
        allowzero: Union[bool, aikit.Container] = True,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.reshape. This method
        simply wraps the function, and so the docstring for aikit.reshape also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        shape
            The new shape should be compatible with the original shape.
            One shape dimension can be -1. In this case, the value is
            inferred from the length of the array and remaining dimensions.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
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
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        order
            Read the elements of the input container using this index order,
            and place the elements into the reshaped array using this index order.
            ‘C’ means to read / write the elements using C-like index order,
            with the last axis index changing fastest, back to the first axis index
            changing slowest.
            ‘F’ means to read / write the elements using Fortran-like index order, with
            the first index changing fastest, and the last index changing slowest.
            Note that the ‘C’ and ‘F’ options take no account of the memory layout
            of the underlying array, and only refer to the order of indexing.
            Default order is 'C'
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an output container having the same data type as ``self``
            and elements as ``self``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0, 1, 2, 3, 4, 5]),
        ...                   b=aikit.array([0, 1, 2, 3, 4, 5]))
        >>> y = x.reshape((2,3))
        >>> print(y)
        {
            a: aikit.array([[0, 1, 2],
                          [3, 4, 5]]),
            b: aikit.array([[0, 1, 2],
                          [3, 4, 5]])
        }

        >>> x = aikit.Container(a=aikit.array([0, 1, 2, 3, 4, 5]),
        ...                   b=aikit.array([0, 1, 2, 3, 4, 5]))
        >>> y = x.reshape((2,3), order='F')
        >>> print(y)
        {
            a: aikit.array([[0, 2, 4],
                          [1, 3, 5]]),
            b: aikit.array([[0, 2, 4],
                          [1, 3, 5]])
        }
        """
        return self._static_reshape(
            self,
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            copy=copy,
            allowzero=allowzero,
            out=out,
            order=order,
        )

    @staticmethod
    def _static_roll(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        shift: Union[int, Tuple[int, ...], aikit.Container],
        *,
        axis: Optional[Union[int, Tuple[int, ...], aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.roll. This method simply
        wraps the function, and so the docstring for aikit.roll also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        shift
            number of places by which the elements are shifted. If ``shift`` is a tuple,
            then ``axis`` must be a tuple of the same size, and each of the given axes
            must be shifted by the corresponding element in ``shift``. If ``shift`` is
            an ``int`` and ``axis`` a tuple, then the same ``shift`` must be used for
            all specified axes. If a shift is positivclipe, then array elements must be
            shifted positively (toward larger indices) along the dimension of ``axis``.
            If a shift is negative, then array elements must be shifted negatively
            (toward smaller indices) along the dimension of ``axis``.
        axis
            axis (or axes) along which elements to shift. If ``axis`` is ``None``, the
            array must be flattened, shifted, and then restored to its original shape.
            Default ``None``.
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
            an output container having the same data type as ``x`` and whose elements,
            relative to ``x``, are shifted.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),
        ...                   b=aikit.array([3., 4., 5.]))
        >>> y = aikit.Container.static_roll(x, 1)
        >>> print(y)
        {
            a: aikit.array([2., 0., 1.]),
            b: aikit.array([5., 3., 4.])
        }

        With multiple :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),
        ...                   b=aikit.array([3., 4., 5.]))
        >>> shift = aikit.Container(a=1, b=-1)
        >>> y = aikit.Container.static_roll(x, shift)
        >>> print(y)
        {
            a: aikit.array([2., 0., 1.]),
            b: aikit.array([4., 5., 3.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "roll",
            x,
            shift,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def roll(
        self: aikit.Container,
        /,
        shift: Union[int, Sequence[int], aikit.Container],
        *,
        axis: Optional[Union[int, Sequence[int], aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.roll. This method
        simply wraps the function, and so the docstring for aikit.roll also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        shift
            number of places by which the elements are shifted. If ``shift`` is a tuple,
            then ``axis`` must be a tuple of the same size, and each of the given axes
            must be shifted by the corresponding element in ``shift``. If ``shift`` is
            an ``int`` and ``axis`` a tuple, then the same ``shift`` must be used for
            all specified axes. If a shift is positive, then array elements must be
            shifted positively (toward larger indices) along the dimension of ``axis``.
            If a shift is negative, then array elements must be shifted negatively
            (toward smaller indices) along the dimension of ``axis``.
        axis
            axis (or axes) along which elements to shift. If ``axis`` is ``None``, the
            array must be flattened, shifted, and then restored to its original shape.
            Default ``None``.
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
            an output container having the same data type as ``self`` and whose
            elements, relative to ``self``, are shifted.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]), b=aikit.array([3., 4., 5.]))
        >>> y = x.roll(1)
        >>> print(y)
        {
            a: aikit.array([2., 0., 1.]),
            b: aikit.array([5., 3., 4.])
        }
        """
        return self._static_roll(
            self,
            shift,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_squeeze(
        x: aikit.Container,
        /,
        axis: Union[int, Sequence[int], aikit.Container],
        *,
        copy: Optional[Union[bool, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.squeeze. This method
        simply wraps the function, and so the docstring for aikit.squeeze also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        axis
            axis (or axes) to squeeze.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
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
            an output container with the results.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[[10.], [11.]]]),
        ...                   b=aikit.array([[[11.], [12.]]]))
        >>> y = aikit.Container.static_squeeze(x, 0)
        >>> print(y)
        {
            a: aikit.array([[10., 11.]]),
            b: aikit.array([[11., 12.]])
        }

        >>> x = aikit.Container(a=aikit.array([[[10.], [11.]]]),
        ...                   b=aikit.array([[[11.], [12.]]]))
        >>> y = aikit.Container.static_squeeze(x, [0, 2])
        >>> print(y)
        {
            a: aikit.array([[10.], [11.]]),
            b: aikit.array([[11.], [12.]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "squeeze",
            x,
            axis=axis,
            copy=copy,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def squeeze(
        self: aikit.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], aikit.Container]],
        copy: Optional[Union[bool, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.squeeze. This method
        simply wraps the function, and so the docstring for aikit.squeeze also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        axis
            axis (or axes) to squeeze.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
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
            an output container with the results.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[[10.], [11.]]]),
        ...                   b=aikit.array([[[11.], [12.]]]))
        >>> y = x.squeeze(axis=2)
        >>> print(y)
        {
            a: aikit.array([[10., 11.]]),
            b: aikit.array([[11., 12.]])
        }

        >>> x = aikit.Container(a=aikit.array([[[10.], [11.]]]),
        ...                   b=aikit.array([[[11.], [12.]]]))
        >>> y = x.squeeze(axis=0)
        >>> print(y)
        {
            a: aikit.array([[10.],
                          [11.]]),
            b: aikit.array([[11.],
                          [12.]])
        }
        """
        return self._static_squeeze(
            self,
            axis=axis,
            copy=copy,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_stack(
        xs: Union[
            Tuple[Union[aikit.Array, aikit.NativeArray, aikit.Container]],
            List[Union[aikit.Array, aikit.NativeArray, aikit.Container]],
        ],
        /,
        *,
        axis: Union[int, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.stack. This method simply
        wraps the function, and so the docstring for aikit.stack also applies to
        this method with minimal changes.

        Parameters
        ----------
        xs
            Container with leaves to join. Each array leavve must have the same shape.
        axis
            axis along which the array leaves will be joined. More details can be found
            in the docstring for aikit.stack.

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
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an output container with the results.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[0, 1], [2,3]]), b=aikit.array([[4, 5]]))
        >>> z = aikit.Container.static_stack(x,axis = 1)
        >>> print(z)
        {
            a: aikit.array([[0, 2],
                        [1, 3]]),
            b: aikit.array([[4],
                        [5]])
        }

        >>> x = aikit.Container(a=aikit.array([[0, 1], [2,3]]), b=aikit.array([[4, 5]]))
        >>> y = aikit.Container(a=aikit.array([[3, 2], [1,0]]), b=aikit.array([[1, 0]]))
        >>> z = aikit.Container.static_stack([x,y])
        >>> print(z)
        {
            a: aikit.array([[[0, 1],
                        [2, 3]],
                        [[3, 2],
                        [1, 0]]]),
            b: aikit.array([[[4, 5]],
                        [[1, 0]]])
        }

        >>> x = aikit.Container(a=aikit.array([[0, 1], [2,3]]), b=aikit.array([[4, 5]]))
        >>> y = aikit.Container(a=aikit.array([[3, 2], [1,0]]), b=aikit.array([[1, 0]]))
        >>> z = aikit.Container.static_stack([x,y],axis=1)
        >>> print(z)
        {
            a: aikit.array([[[0, 1],
                        [3, 2]],
                        [[2, 3],
                        [1, 0]]]),
            b: aikit.array([[[4, 5],
                        [1, 0]]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "stack",
            xs,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def stack(
        self: aikit.Container,
        /,
        xs: Union[
            Tuple[Union[aikit.Array, aikit.NativeArray, aikit.Container]],
            List[Union[aikit.Array, aikit.NativeArray, aikit.Container]],
        ],
        *,
        axis: Union[int, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.stack. This method
        simply wraps the function, and so the docstring for aikit.stack also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Container with leaves to join with leaves of other arrays/containers.
             Each array leave must have the same shape.
        xs
            Container with other leaves to join.
            Each array leave must have the same shape.
        axis
            axis along which the array leaves will be joined. More details can be found
            in the docstring for aikit.stack.
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
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an output container with the results.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[0, 1], [2,3]]), b=aikit.array([[4, 5]]))
        >>> y = aikit.Container(a=aikit.array([[3, 2], [1,0]]), b=aikit.array([[1, 0]]))
        >>> x.stack([y])
        {
            a: aikit.array([[[0, 1],
                        [2, 3]],
                        [[3, 2],
                        [1, 0]]]),
            b: aikit.array([[[4, 5]],
                        [[1, 0]]])
        }
        """
        new_xs = xs.cont_copy() if aikit.is_aikit_container(xs) else xs.copy()
        new_xs.insert(0, self.cont_copy())
        return self._static_stack(
            new_xs,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_repeat(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        repeats: Union[int, Iterable[int], aikit.Container],
        *,
        axis: Optional[Union[int, Sequence[int], aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.repeat. This method
        simply wraps the function, and so the docstring for aikit.repeat also
        applies to this method with minimal changes.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]), b=aikit.array([3., 4., 5.]))
        >>> y = aikit.Container.static_repeat(2)
        >>> print(y)
        {
            a: aikit.array([0., 0., 1., 1., 2., 2.]),
            b: aikit.array([3., 3., 4., 4., 5., 5.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "repeat",
            x,
            repeats,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def repeat(
        self: aikit.Container,
        /,
        repeats: Union[int, Iterable[int], aikit.Container],
        *,
        axis: Optional[Union[int, Sequence[int], aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.repeat. This method
        simply wraps the function, and so the docstring for aikit.repeat also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container.
        repeats
            The number of repetitions for each element. repeats is broadcast to fit the
            shape of the given axis.
        axis
            The axis along which to repeat values. By default, use the flattened input
            array, and return a flat output array.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The output container with repreated leaves.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]), b=aikit.array([3., 4., 5.]))
        >>> y = x.repeat(2)
        >>> print(y)
        {
            a: aikit.array([0., 0., 1., 1., 2., 2.]),
            b: aikit.array([3., 3., 4., 4., 5., 5.])
        }
        """
        return self._static_repeat(
            self,
            repeats,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_tile(
        x: aikit.Container,
        /,
        repeats: Union[Iterable[int], aikit.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.tile. This method simply
        wraps the function, and so the docstring for aikit.tile also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            Input Container.
        repeats
            The number of repetitions of x along each axis.
        out
            optional output array, for writing the result to. It must have
            a shape that the inputs broadcast to.

        Returns
        -------
        ret
            The container output with tiled leaves.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[0, 1], [2,3]]), b=aikit.array([[4, 5]]))
        >>> y = aikit.Container.static_tile((2,3))
        >>> print(y)
        {
            a: aikit.array([[0,1,0,1,0,1],
                          [2,3,2,3,2,3],
                          [0,1,0,1,0,1],
                          [2,3,2,3,2,3]]),
            b: aikit.array([[4,5,4,5,4,5],
                          [4,5,4,5,4,5]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "tile",
            x,
            repeats,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def tile(
        self: aikit.Container,
        /,
        repeats: Union[Iterable[int], aikit.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.tile. This method
        simply wraps the function, and so the docstring for aikit.tile also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container.
        repeats
            The number of repetitions of x along each axis.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            The container output with tiled leaves.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[0, 1], [2,3]]), b=aikit.array([[4, 5]]))
        >>> y = x.tile((2,3))
        >>> print(y)
        {
            a: (<class aikit.data_classes.array.array.Array> shape=[4, 6]),
            b: (<class aikit.data_classes.array.array.Array> shape=[2, 6])
        }
        """
        return self._static_tile(
            self,
            repeats,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_constant_pad(
        x: aikit.Container,
        /,
        pad_width: Union[Iterable[Tuple[int]], aikit.Container],
        *,
        value: Union[Number, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.constant_pad. This method
        simply wraps the function, and so the docstring for aikit.constant_pad
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container with leaves to pad.
        pad_width
            Number of values padded to the edges of each axis.
            Specified as ((before_1, after_1), … (before_N, after_N)), where N
            is number of axes of x.
        value
            The constant value to pad the array with.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            Output container with padded array leaves of rank equal to x with
            shape increased according to pad_width.

        Examples
        --------
        >>> x = aikit.Container(a = aikit.array([1, 2, 3]), b = aikit.array([4, 5, 6]))
        >>> y = aikit.Container.static_constant_pad(x, pad_width = [[2, 3]])
        >>> print(y)
        {
            a: aikit.array([0, 0, 1, 2, 3, 0, 0, 0]),
            b: aikit.array([0, 0, 4, 5, 6, 0, 0, 0])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "constant_pad",
            x,
            pad_width,
            value=value,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def constant_pad(
        self: aikit.Container,
        /,
        pad_width: Union[Iterable[Tuple[int]], aikit.Container],
        *,
        value: Union[Number, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.constant_pad. This
        method simply wraps the function, and so the docstring for
        aikit.constant_pad also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container with leaves to pad.
        pad_width
            Number of values padded to the edges of each axis.
            Specified as ((before_1, after_1), … (before_N, after_N)), where N
            is number of axes of x.
        value
            The constant value to pad the array with.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            Output container with padded array leaves of rank equal to x with
            shape increased according to pad_width.

        Examples
        --------
        >>> x = aikit.Container(a = aikit.array([1, 2, 3]), b = aikit.array([4, 5, 6]))
        >>> y = x.constant_pad(pad_width = [[2, 3]])
        >>> print(y)
        {
            a: aikit.array([0, 0, 1, 2, 3, 0, 0, 0]),
            b: aikit.array([0, 0, 4, 5, 6, 0, 0, 0])
        }
        """
        return self._static_constant_pad(
            self,
            pad_width,
            value=value,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_zero_pad(
        x: aikit.Container,
        /,
        pad_width: Union[Iterable[Tuple[int]], aikit.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.zero_pad. This method
        simply wraps the function, and so the docstring for aikit.zero_pad also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array to pad.
        pad_width
            Number of values padded to the edges of each axis. Specified as
            ((before_1, after_1), … (before_N, after_N)),
            where N is number of axes of x.
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
            Padded array of rank equal to x with shape increased according to pad_width.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a = aikit.array([1., 2., 3.]), b = aikit.array([3., 4., 5.]))
        >>> y = aikit.zero_pad(x, pad_width = [[2, 3]])
        >>> print(y)
        {
            a: aikit.array([0., 0., 1., 2., 3., 0., 0., 0.]),
            b: aikit.array([0., 0., 3., 4., 5., 0., 0., 0.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "zero_pad",
            x,
            pad_width,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def zero_pad(
        self: aikit.Container,
        /,
        pad_width: Union[Iterable[Tuple[int]], aikit.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.zero_pad. This method
        simply wraps the function, and so the docstring for aikit.zero_pad also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array to pad.
        pad_width
            Number of values padded to the edges of each axis. Specified as
            ((before_1, after_1), … (before_N, after_N)),
            where N is number of axes of x.
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
            Padded array of rank equal to x with shape increased according to pad_width.

        Examples
        --------
        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a = aikit.array([1., 2., 3.]), b = aikit.array([3., 4., 5.]))
        >>> y = x.zero_pad(pad_width = [[2, 3]])
        >>> print(y)
        {
            a: aikit.array([0., 0., 1., 2., 3., 0., 0., 0.]),
            b: aikit.array([0., 0., 3., 4., 5., 0., 0., 0.])
        }
        """
        return self._static_zero_pad(
            self,
            pad_width,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_swapaxes(
        x: aikit.Container,
        axis0: Union[int, aikit.Container],
        axis1: Union[int, aikit.Container],
        /,
        *,
        copy: Optional[Union[bool, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.swapaxes. This method
        simply wraps the function, and so the docstring for aikit.swapaxes also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container
        axis0
            First axis to be swapped.
        axis1
            Second axis to be swapped.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            x with its axes permuted.

        >>> a = aikit.array([[1, 2, 3], [4, 5, 6]])
        >>> b = aikit.array([[7, 8, 9], [10, 11, 12]])
        >>> x = aikit.Container(a = a, b = b)
        >>> y = x.swapaxes(0, 1)
        >>> print(y)
        {
            a: aikit.array([[1, 4],
                          [2, 5],
                          [3, 6]]),
            b: aikit.array([[7, 10],
                          [8, 11],
                          [9, 12]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "swapaxes",
            x,
            axis0,
            axis1,
            copy=copy,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def swapaxes(
        self: aikit.Container,
        axis0: Union[int, aikit.Container],
        axis1: Union[int, aikit.Container],
        /,
        *,
        copy: Optional[Union[bool, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.swapaxes. This method
        simply wraps the function, and so the docstring for aikit.swapaxes also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container.
        axis0
            First axis to be swapped.
        axis1
            Second axis to be swapped.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            x with its axes permuted.

        Examples
        --------
        >>> a = aikit.array([[1, 2, 3], [4, 5, 6]])
        >>> b = aikit.array([[7, 8, 9], [10, 11, 12]])
        >>> x = aikit.Container(a = a, b = b)
        >>> y = x.swapaxes(0, 1)
        >>> print(y)
        {
            a: aikit.array([[1, 4],
                          [2, 5],
                          [3, 6]]),
            b: aikit.array([[7, 10],
                          [8, 11],
                          [9, 12]])
        }
        """
        return self._static_swapaxes(
            self,
            axis0,
            axis1,
            copy=copy,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_unstack(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        copy: Optional[Union[bool, aikit.Container]] = None,
        axis: Union[int, aikit.Container] = 0,
        keepdims: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.unstack. This method
        simply wraps the function, and so the docstring for aikit.unstack also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array or container to unstack.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        axis
            Axis for which to unpack the array.
        keepdims
            Whether to keep dimension 1 in the unstack dimensions. Default is ``False``.
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
            List of arrays, unpacked along specified dimensions, or containers
            with arrays unpacked at leaves

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
                            b=aikit.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]))
        >>> y = aikit.Container.static_unstack(x, axis=0)
        >>> print(y)
        [{
            a: aikit.array([[1, 2],
                         [3, 4]]),
            b: aikit.array([[9, 10],
                         [11, 12]])
        }, {
            a: aikit.array([[5, 6],
                         [7, 8]]),
             b: aikit.array([[13, 14],
                          [15, 16]])
        }]

        >>> x = aikit.Container(a=aikit.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
                            b=aikit.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]))
        >>> y = aikit.Container.static_unstack(x, axis=1, keepdims=True)
        >>> print(y)
        [{
            a: aikit.array([[[1, 2]],
                         [[5, 6]]]),
            b: aikit.array([[[9, 10]],
                         [[13, 14]]])
        }, {
            a: aikit.array([[[3, 4]],
                         [[7, 8]]]),
            b: aikit.array([[[11, 12]],
                         [[15, 16]]])
        }]
        """
        return ContainerBase.cont_multi_map_in_function(
            "unstack",
            x,
            copy=copy,
            axis=axis,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def unstack(
        self: aikit.Container,
        /,
        *,
        copy: Optional[Union[bool, aikit.Container]] = None,
        axis: Union[int, aikit.Container] = 0,
        keepdims: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.unstack. This method
        simply wraps the function, and so the docstring for aikit.unstack also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container to unstack at leaves.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        axis
            Axis for which to unpack the array.
        keepdims
            Whether to keep dimension 1 in the unstack dimensions. Default is ``False``.
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
            Containers with arrays unpacked at leaves

        Examples
        --------
        With one :class:`aikit.Container` instances:

        >>> x = aikit.Container(a=aikit.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
                            b=aikit.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]))
        >>> x.unstack(axis=0)
        [{
            a: aikit.array([[1, 2],
                         [3, 4]]),
            b: aikit.array([[9, 10],
                          [11, 12]])
        }, {
            a: aikit.array([[5, 6],
                          [7, 8]]),
            b: aikit.array([[13, 14],
                          [15, 16]])
        }]
        """
        return self._static_unstack(
            self,
            copy=copy,
            axis=axis,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_clip(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        x_min: Optional[
            Union[Number, aikit.Array, aikit.NativeArray, aikit.Container]
        ] = None,
        x_max: Optional[
            Union[Number, aikit.Array, aikit.NativeArray, aikit.Container]
        ] = None,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.clip. This method simply
        wraps the function, and so the docstring for aikit.clip also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            Input array or container containing elements to clip.
        x_min
            Minimum value.
        x_max
            Maximum value.
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
            A container with the elements of x, but where values < x_min are replaced
            with x_min, and those > x_max with x_max.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),
        ...                   b=aikit.array([3., 4., 5.]))
        >>> y = aikit.Container.static_clip(x, 1., 5.)
        >>> print(y)
        {
            a: aikit.array([1., 1., 2.]),
            b: aikit.array([3., 4., 5.])
        }

        With multiple :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),
        ...                   b=aikit.array([3., 4., 5.]))
        >>> x_min = aikit.Container(a=0, b=0)
        >>> x_max = aikit.Container(a=1, b=1)
        >>> y = aikit.Container.static_clip(x, x_min, x_max)
        >>> print(y)
        {
            a: aikit.array([0., 1., 1.]),
            b: aikit.array([1., 1., 1.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "clip",
            x,
            x_min,
            x_max,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def clip(
        self: aikit.Container,
        /,
        x_min: Optional[
            Union[Number, aikit.Array, aikit.NativeArray, aikit.Container]
        ] = None,
        x_max: Optional[
            Union[Number, aikit.Array, aikit.NativeArray, aikit.Container]
        ] = None,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.clip. This method
        simply wraps the function, and so the docstring for aikit.clip also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container containing elements to clip.
        x_min
            Minimum value.
        x_max
            Maximum value.
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
            A container with the elements of x, but where values < x_min are replaced
            with x_min, and those > x_max with x_max.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]), b=aikit.array([3., 4., 5.]))
        >>> y = x.clip(1,2)
        >>> print(y)
        {
            a: aikit.array([1., 1., 2.]),
            b: aikit.array([2., 2., 2.])
        }
        """
        return self._static_clip(
            self,
            x_min,
            x_max,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
