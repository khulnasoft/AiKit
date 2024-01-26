# global
from numbers import Number
from typing import Optional, Union, List, Dict

# local
import aikit
from aikit.data_classes.container.base import ContainerBase


# noinspection PyMissingConstructor
class _ContainerWithSearching(ContainerBase):
    @staticmethod
    def _static_argmax(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        axis: Optional[Union[int, aikit.Container]] = None,
        keepdims: Union[bool, aikit.Container] = False,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        select_last_index: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.argmax. This method
        simply wraps the function, and so the docstring for aikit.argmax also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array or container. Should have a numeric data type.
        axis
            axis along which to search. If None, the function must return the index of
            the maximum value of the flattened array. Default: ``None``.
        keepdims
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the array.
        dtype
             Optional data type of the output array.
        out
            If provided, the result will be inserted into this array. It should be of
            the appropriate shape and dtype.

        Returns
        -------
        ret
            a container containing the indices of the maximum values across the
            specified axis.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[4., 0., -1.], [2., -3., 6]]),\
        ...                   b=aikit.array([[1., 2., 3.], [1., 1., 1.]])
        >>> y = aikit.Container.static_argmax(x, axis=1, keepdims=True)
        >>> print(y)
        {
            a: aikit.array([[0],
                          [2]]),
            b: aikit.array([[2],
                          [0]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "argmax",
            x,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            select_last_index=select_last_index,
            out=out,
        )

    def argmax(
        self: aikit.Container,
        /,
        *,
        axis: Optional[Union[int, aikit.Container]] = None,
        keepdims: Union[bool, aikit.Container] = False,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        select_last_index: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.argmax. This method
        simply wraps the function, and so the docstring for aikit.argmax also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a numeric data type.
        axis
            axis along which to search. If None, the function must return the index of
            the maximum value of the flattened array. Default: ``None``.
        keepdims
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the array.
        dtype
            Optional output dtype of the container.
        out
            If provided, the result will be inserted into this array. It should be of
            the appropriate shape and dtype.

        Returns
        -------
        ret
            a container containing the indices of the maximum values across the
            specified axis.

        Examples
        --------
        >>> a = aikit.array([[4., 0., -1.], [2., -3., 6]])
        >>> b = aikit.array([[1., 2., 3.], [1., 1., 1.]])
        >>> x = aikit.Container(a=a, b=b)
        >>> y = x.argmax(axis=1, keepdims=True)
        >>> print(y)
        {
            a: aikit.array([[0],
                          [2]]),
            b: aikit.array([[2],
                          [0]])
        }
        """
        return self._static_argmax(
            self,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            select_last_index=select_last_index,
            out=out,
        )

    @staticmethod
    def _static_argmin(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        axis: Optional[Union[int, aikit.Container]] = None,
        keepdims: Union[bool, aikit.Container] = False,
        dtype: Optional[Union[aikit.int32, aikit.int64, aikit.Container]] = None,
        select_last_index: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.argmin. This method
        simply wraps the function, and so the docstring for aikit.argmin also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array or container. Should have a numeric data type.
        axis
            axis along which to search. If None, the function must return the index of
            the minimum value of the flattened array. Default = None.
        keepdims
            if True, the reduced axes (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the result must be compatible with
            the input array (see Broadcasting). Otherwise, if False, the reduced axes
            (dimensions) must not be included in the result. Default = False.
        dtype
            An optional output_dtype from: int32, int64. Defaults to int64.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the indices of the minimum values across the
            specified axis.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[4., 0., -1.], [2., -3., 6]]),\
        ...                   b=aikit.array([[1., 2., 3.], [1., 1., 1.]])
        >>> y = aikit.Container.static_argmin(axis=1, keepdims=True)
        >>> print(y)
        {
            a: aikit.array([[2],
                          [1]]),
            b: aikit.array([[0],
                          [0]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "argmin",
            x,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            select_last_index=select_last_index,
            out=out,
        )

    def argmin(
        self: aikit.Container,
        /,
        *,
        axis: Optional[Union[int, aikit.Container]] = None,
        keepdims: Union[bool, aikit.Container] = False,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        select_last_index: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.argmin. This method
        simply wraps the function, and so the docstring for aikit.argmin also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a numeric data type.
        axis
            axis along which to search. If None, the function must return the index of
            the minimum value of the flattened array. Default = None.
        keepdims
            if True, the reduced axes (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the result must be compatible with
            the input array (see Broadcasting). Otherwise, if False, the reduced axes
            (dimensions) must not be included in the result. Default = False.
        dtype
            An optional output_dtype from: int32, int64. Defaults to int64.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the indices of the minimum values across the
            specified axis.

        Examples
        --------
        Using :class:`aikit.Container` instance method:

        >>> x = aikit.Container(a=aikit.array([0., -1., 2.]), b=aikit.array([3., 4., 5.]))
        >>> y = x.argmin()
        >>> print(y)
        {
            a: aikit.array(1),
            b: aikit.array(0)
        }

        >>> x = aikit.Container(a=aikit.array([[4., 0., -1.], [2., -3., 6]]),
        ...                   b=aikit.array([[1., 2., 3.], [1., 1., 1.]]))
        >>> y = x.argmin(axis=1, keepdims=True)
        >>> print(y)
        {
            a: aikit.array([[2],
                          [1]]),
            b: aikit.array([[0],
                          [0]])
        }
        """
        return self._static_argmin(
            self,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            select_last_index=select_last_index,
            out=out,
        )

    @staticmethod
    def _static_nonzero(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        as_tuple: Union[bool, aikit.Container] = True,
        size: Optional[Union[int, aikit.Container]] = None,
        fill_value: Union[Number, aikit.Container] = 0,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.nonzero. This method
        simply wraps the function, and so the docstring for aikit.nonzero also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array or container. Should have a numeric data type.
        as_tuple
            if True, the output is returned as a tuple of indices, one for each
            dimension of the input, containing the indices of the true elements in that
            dimension. If False, the coordinates are returned in a (N, ndim) array,
            where N is the number of true elements. Default = True.
        size
            if specified, the function will return an array of shape (size, ndim).
            If the number of non-zero elements is fewer than size, the remaining
            elements will be filled with fill_value. Default = None.
        fill_value
            when size is specified and there are fewer than size number of elements,
            the remaining elements in the output array will be filled with fill_value.
            Default = 0.

        Returns
        -------
        ret
            a container containing the indices of the nonzero values.
        """
        return ContainerBase.cont_multi_map_in_function(
            "nonzero", x, as_tuple=as_tuple, size=size, fill_value=fill_value
        )

    def nonzero(
        self: aikit.Container,
        /,
        *,
        as_tuple: Union[bool, aikit.Container] = True,
        size: Optional[Union[int, aikit.Container]] = None,
        fill_value: Union[Number, aikit.Container] = 0,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.nonzero. This method
        simply wraps the function, and so the docstring for aikit.nonzero also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a numeric data type.
        as_tuple
            if True, the output is returned as a tuple of indices, one for each
            dimension of the input, containing the indices of the true elements in that
            dimension. If False, the coordinates are returned in a (N, ndim) array,
            where N is the number of true elements. Default = True.
        size
            if specified, the function will return an array of shape (size, ndim).
            If the number of non-zero elements is fewer than size, the remaining
            elements will be filled with fill_value. Default = None.
        fill_value
            when size is specified and there are fewer than size number of elements,
            the remaining elements in the output array will be filled with fill_value.
            Default = 0.

        Returns
        -------
        ret
            a container containing the indices of the nonzero values.
        """
        return self._static_nonzero(
            self, as_tuple=as_tuple, size=size, fill_value=fill_value
        )

    @staticmethod
    def _static_where(
        condition: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        x1: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.where. This method simply
        wraps the function, and so the docstring for aikit.where also applies to
        this method with minimal changes.

        Parameters
        ----------
        condition
            input array or container. Should have a boolean data type.
        x1
            input array or container. Should have a numeric data type.
        x2
            input array or container. Should have a numeric data type.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the values of x1 where condition is True, and x2
            where condition is False.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([3, 1, 5]), b=aikit.array([2, 4, 6]))
        >>> x2 = aikit.Container(a=aikit.array([0, 7, 2]), b=aikit.array([3, 8, 5]))
        >>> res = aikit.Container.static_where((x1.a > x2.a), x1, x2)
        >>> print(res)
        {
            a: aikit.array([3, 7, 5]),
            b: aikit.array([2, 8, 6])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "where", condition, x1, x2, out=out
        )

    def where(
        self: aikit.Container,
        x1: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        x2: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.where. This method
        simply wraps the function, and so the docstring for aikit.where also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a boolean data type.
        x1
            input array or container. Should have a numeric data type.
        x2
            input array or container. Should have a numeric data type.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the values of x1 where condition is True, and x2
            where condition is False.

        Examples
        --------
        >>> x1 = aikit.Container(a=aikit.array([3, 1, 5]), b=aikit.array([2, 4, 6]))
        >>> x2 = aikit.Container(a=aikit.array([0, 7, 2]), b=aikit.array([3, 8, 5]))
        >>> res = x1.where((x1.a > x2.a), x2)
        >>> print(res)
        {
            a: aikit.array([1, 0, 1]),
            b: aikit.array([1, 0, 1])
        }
        """
        return self._static_where(self, x1, x2, out=out)

    # Extra #
    # ----- #

    @staticmethod
    def _static_argwhere(
        x: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.argwhere. This method
        simply wraps the function, and so the docstring for aikit.argwhere also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            Boolean array, for which indices are desired.
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
            Indices for where the boolean array is True.

        Examples
        --------
        Using :class:`aikit.Container` instance method

        >>> x = aikit.Container(a=aikit.array([1, 2]), b=aikit.array([3, 4]))
        >>> res = aikit.Container.static_argwhere(x)
        >>> print(res)
        {
            a: aikit.array([[0], [1]]),
            b: aikit.array([[0], [1]])
        }

        >>> x = aikit.Container(a=aikit.array([1, 0]), b=aikit.array([3, 4]))
        >>> res = aikit.Container.static_argwhere(x)
        >>> print(res)
        {
            a: aikit.array([[0]]),
            b: aikit.array([[0], [1]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "argwhere",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def argwhere(
        self: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ):
        """aikit.Container instance method variant of aikit.argwhere. This method
        simply wraps the function, and so the docstring for aikit.argwhere also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Boolean array, for which indices are desired.
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
            Indices for where the boolean array is True.

        Examples
        --------
        Using :class:`aikit.Container` instance method

        >>> x = aikit.Container(a=aikit.array([1, 2]), b=aikit.array([3, 4]))
        >>> res = x.argwhere()
        >>> print(res)
        {
            a: aikit.array([[0], [1]]),
            b: aikit.array([[0], [1]])
        }

        >>> x = aikit.Container(a=aikit.array([1, 0]), b=aikit.array([3, 4]))
        >>> res = x.argwhere()
        >>> print(res)
        {
            a: aikit.array([[0]]),
            b: aikit.array([[0], [1]])
        }
        """
        return self._static_argwhere(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
