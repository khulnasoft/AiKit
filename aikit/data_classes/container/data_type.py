# global
from typing import Optional, Union, List, Dict, Tuple, Callable

# local
import aikit
from aikit.data_classes.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class _ContainerWithDataTypes(ContainerBase):
    @staticmethod
    def _static_astype(
        x: aikit.Container,
        dtype: Union[aikit.Dtype, aikit.Container],
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        copy: Union[bool, aikit.Container] = True,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """Copy an array to a specified data type irrespective of :ref:`type-
        promotion` rules.

        .. note::
        Casting floating-point ``NaN`` and ``infinity`` values to integral data types
        is not specified and is implementation-dependent.

        .. note::
        When casting a boolean input array to a numeric data type, a value of ``True``
        must cast to a numeric value equal to ``1``, and a value of ``False`` must cast
        to a numeric value equal to ``0``.

        When casting a numeric input array to ``bool``, a value of ``0`` must cast to
        ``False``, and a non-zero value must cast to ``True``.

        Parameters
        ----------
        x
            array to cast.
        dtype
            desired data type.
        copy
            specifies whether to copy an array when the specified ``dtype`` matches
            the data type of the input array ``x``. If ``True``, a newly allocated
            array must always be returned. If ``False`` and the specified ``dtype``
            matches the data type of the input array, the input array must be returned;
            otherwise, a newly allocated must be returned. Default: ``True``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array having the specified data type. The returned array must have
            the same shape as ``x``.

        Examples
        --------
        >>> c = aikit.Container(a=aikit.array([False,True,True]),
        ...                   b=aikit.array([3.14, 2.718, 1.618]))
        >>> aikit.Container.static_astype(c, aikit.int32)
        {
            a: aikit.array([0, 1, 1]),
            b: aikit.array([3, 2, 1])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "astype",
            x,
            dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            copy=copy,
            out=out,
        )

    def astype(
        self: aikit.Container,
        dtype: Union[aikit.Dtype, aikit.Container],
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        copy: Union[bool, aikit.Container] = True,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """Copy an array to a specified data type irrespective of :ref:`type-
        promotion` rules.

        .. note::
        Casting floating-point ``NaN`` and ``infinity`` values to integral data types
        is not specified and is implementation-dependent.

        .. note::
        When casting a boolean input array to a numeric data type, a value of ``True``
        must cast to a numeric value equal to ``1``, and a value of ``False`` must cast
        to a numeric value equal to ``0``.

        When casting a numeric input array to ``bool``, a value of ``0`` must cast to
        ``False``, and a non-zero value must cast to ``True``.

        Parameters
        ----------
        self
            array to cast.
        dtype
            desired data type.
        copy
            specifies whether to copy an array when the specified ``dtype`` matches
            the data type of the input array ``x``. If ``True``, a newly allocated
            array must always be returned. If ``False`` and the specified ``dtype``
            matches the data type of the input array, the input array must be returned;
            otherwise, a newly allocated must be returned. Default: ``True``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array having the specified data type. The returned array must have
            the same shape as ``x``.

        Examples
        --------
        Using :class:`aikit.Container` instance method:

        >>> x = aikit.Container(a=aikit.array([False,True,True]),
        ...                   b=aikit.array([3.14, 2.718, 1.618]))
        >>> print(x.astype(aikit.int32))
        {
            a: aikit.array([0, 1, 1]),
            b: aikit.array([3, 2, 1])
        }
        """
        return self._static_astype(
            self,
            dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            copy=copy,
            out=out,
        )

    @staticmethod
    def _static_broadcast_arrays(
        *arrays: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """`aikit.Container` static method variant of `aikit.broadcast_arrays`.
        This method simply wraps the function, and so the docstring for
        `aikit.broadcast_arrays` also applies to this method with minimal
        changes.

        Parameters
        ----------
        arrays
            an arbitrary number of arrays to-be broadcasted.
            Each array must have the same shape.
            And Each array must have the same dtype as its
            corresponding input array.
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
            A list of containers containing broadcasted arrays

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x1 = aikit.Container(a=aikit.array([1, 2]), b=aikit.array([3, 4]))
        >>> x2 = aikit.Container(a=aikit.array([-1.2, 0.4]), b=aikit.array([0, 1]))
        >>> y = aikit.Container.static_broadcast_arrays(x1, x2)
        >>> print(y)
        [{
            a: aikit.array([1, 2]),
            b: aikit.array([3, 4])
        }, {
            a: aikit.array([-1.2, 0.4]),
            b: aikit.array([0, 1])
        }]

        With mixed :class:`aikit.Container` and :class:`aikit.Array` inputs:

        >>> x1 = aikit.Container(a=aikit.array([4, 5]), b=aikit.array([2, -1]))
        >>> x2 = aikit.array([0.2, 3.])
        >>> y = aikit.Container.static_broadcast_arrays(x1, x2)
        >>> print(y)
        [{
            a: aikit.array([4, 5]),
            b: aikit.array([2, -1])
        }, {
            a: aikit.array([0.2, 3.]),
            b: aikit.array([0.2, 3.])
        }]
        """
        return ContainerBase.cont_multi_map_in_function(
            "broadcast_arrays",
            *arrays,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def broadcast_arrays(
        self: aikit.Container,
        *arrays: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """`aikit.Container` instance method variant of `aikit.broadcast_arrays`.
        This method simply wraps the function, and so the docstring for
        `aikit.broadcast_arrays` also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            A container to be broadcatsed against other input arrays.
        arrays
            an arbitrary number of containers having arrays to-be broadcasted.
            Each array must have the same shape.
            Each array must have the same dtype as its corresponding input array.
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


        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x1 = aikit.Container(a=aikit.array([1, 2]), b=aikit.array([3, 4]))
        >>> x2 = aikit.Container(a=aikit.array([-1.2, 0.4]), b=aikit.array([0, 1]))
        >>> y = x1.broadcast_arrays(x2)
        >>> print(y)
        [{
            a: aikit.array([1, 2]),
            b: aikit.array([3, 4])
        }, {
            a: aikit.array([-1.2, 0.4]),
            b: aikit.array([0, 1])
        }]

        With mixed :class:`aikit.Container` and :class:`aikit.Array` inputs:

        >>> x1 = aikit.Container(a=aikit.array([4, 5]), b=aikit.array([2, -1]))
        >>> x2 = aikit.zeros(2)
        >>> y = x1.broadcast_arrays(x2)
        >>> print(y)
        [{
            a: aikit.array([4, 5]),
            b: aikit.array([2, -1])
        }, {
            a: aikit.array([0., 0.]),
            b: aikit.array([0., 0.])
        }]
        """
        return self._static_broadcast_arrays(
            self,
            *arrays,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_broadcast_to(
        x: aikit.Container,
        /,
        shape: Union[Tuple[int, ...], aikit.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """`aikit.Container` static method variant of `aikit.broadcast_to`. This
        method simply wraps the function, and so the docstring for
        `aikit.broadcast_to` also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array to be broadcasted.
        shape
            desired shape to be broadcasted to.
        out
            Optional array to store the broadcasted array.

        Returns
        -------
        ret
            Returns the broadcasted array of shape 'shape'

        Examples
        --------
        With :class:`aikit.Container` static method:

        >>> x = aikit.Container(a=aikit.array([1]),
        ...                   b=aikit.array([2]))
        >>> y = aikit.Container.static_broadcast_to(x,(3, 1))
        >>> print(y)
        {
            a: aikit.array([1],
                         [1],
                         [1]),
            b: aikit.array([2],
                         [2],
                         [2])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "broadcast_to",
            x,
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def broadcast_to(
        self: aikit.Container,
        /,
        shape: Union[Tuple[int, ...], aikit.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """`aikit.Container` instance method variant of `aikit.broadcast_to`. This
        method simply wraps the function, and so the docstring for
        `aikit.broadcast_to` also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array to be broadcasted.
        shape
            desired shape to be broadcasted to.
        out
            Optional array to store the broadcasted array.

        Returns
        -------
        ret
            Returns the broadcasted array of shape 'shape'

        Examples
        --------
        With :class:`aikit.Container` instance method:

        >>> x = aikit.Container(a=aikit.array([0, 0.5]),
        ...                   b=aikit.array([4, 5]))
        >>> y = x.broadcast_to((3,2))
        >>> print(y)
        {
            a: aikit.array([[0., 0.5],
                          [0., 0.5],
                          [0., 0.5]]),
            b: aikit.array([[4, 5],
                          [4, 5],
                          [4, 5]])
        }
        """
        return self._static_broadcast_to(
            self,
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_can_cast(
        from_: aikit.Container,
        to: Union[aikit.Dtype, aikit.Container],
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """`aikit.Container` static method variant of `aikit.can_cast`. This method
        simply wraps the function, and so the docstring for `aikit.can_cast` also
        applies to this method with minimal changes.

        Parameters
        ----------
        from_
            input container from which to cast.
        to
            desired data type.
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
            ``True`` if the cast can occur according to :ref:`type-promotion` rules;
            otherwise, ``False``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),
        ...                   b=aikit.array([3, 4, 5]))
        >>> print(x.a.dtype, x.b.dtype)
        float32 int32

        >>> print(aikit.Container.static_can_cast(x, 'int64'))
        {
            a: false,
            b: true
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "can_cast",
            from_,
            to,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def can_cast(
        self: aikit.Container,
        to: Union[aikit.Dtype, aikit.Container],
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """`aikit.Container` instance method variant of `aikit.can_cast`. This
        method simply wraps the function, and so the docstring for
        `aikit.can_cast` also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container from which to cast.
        to
            desired data type.
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
            ``True`` if the cast can occur according to :ref:`type-promotion` rules;
            otherwise, ``False``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]),
        ...                   b=aikit.array([3, 4, 5]))
        >>> print(x.a.dtype, x.b.dtype)
        float32 int32

        >>> print(x.can_cast('int64'))
        {
            a: False,
            b: True
        }
        """
        return self._static_can_cast(
            self, to, key_chains, to_apply, prune_unapplied, map_sequences
        )

    @staticmethod
    def _static_dtype(
        x: aikit.Container,
        *,
        as_native: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "dtype",
            x,
            as_native=as_native,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def dtype(
        self: aikit.Container,
        *,
        as_native: Union[bool, aikit.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """
        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 2, 3]), b=aikit.array([2, 3, 4]))
        >>> y = x.dtype()
        >>> print(y)
        {
            a: int32,
            b: int32
        }
        """
        return self._static_dtype(
            self,
            as_native=as_native,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_default_float_dtype(
        *,
        input: Optional[Union[aikit.Array, aikit.NativeArray, aikit.Container]] = None,
        float_dtype: Optional[
            Union[aikit.FloatDtype, aikit.NativeDtype, aikit.Container]
        ] = None,
        as_native: Optional[Union[bool, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "default_float_dtype",
            input=input,
            float_dtype=float_dtype,
            as_native=as_native,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_default_complex_dtype(
        *,
        input: Optional[Union[aikit.Array, aikit.NativeArray, aikit.Container]] = None,
        complex_dtype: Optional[
            Union[aikit.FloatDtype, aikit.NativeDtype, aikit.Container]
        ] = None,
        as_native: Optional[Union[bool, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "default_complex_dtype",
            input=input,
            complex_dtype=complex_dtype,
            as_native=as_native,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_function_supported_dtypes(
        fn: Union[Callable, aikit.Container],
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "function_supported_dtypes",
            fn,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_function_unsupported_dtypes(
        fn: Union[Callable, aikit.Container],
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "function_unsupported_dtypes",
            fn,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_finfo(
        type: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """`aikit.Container` static method variant of `aikit.finfo`.

        Parameters
        ----------
        type
            input container with leaves to inquire information about.

        Returns
        -------
        ret
            container of the same structure as `self`, with each element
            as a finfo object for the corresponding dtype of
            leave in`self`.

        Examples
        --------
        >>> c = aikit.Container(x=aikit.array([-9.5,1.8,-8.9], dtype=aikit.float16),
        ...                   y=aikit.array([7.6,8.1,1.6], dtype=aikit.float64))
        >>> y = aikit.Container.static_finfo(c)
        >>> print(y)
        {
            x: finfo(resolution=0.001, min=-6.55040e+04, max=6.55040e+04,\
                    dtype=float16),
            y: finfo(resolution=1e-15, min=-1.7976931348623157e+308, \
                max=1.7976931348623157e+308, dtype=float64)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "finfo",
            type,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def finfo(
        self: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """`aikit.Container` instance method variant of `aikit.finfo`.

        Parameters
        ----------
        self
            input container with leaves to inquire information about.

        Returns
        -------
        ret
            container of the same structure as `self`, with each element
            as a finfo object for the corresponding dtype of
            leave in`self`.

        Examples
        --------
        >>> c = aikit.Container(x=aikit.array([-9.5,1.8,-8.9], dtype=aikit.float16),
        ...                   y=aikit.array([7.6,8.1,1.6], dtype=aikit.float64))
        >>> print(c.finfo())
        {
            x: finfo(resolution=0.001, min=-6.55040e+04, max=6.55040e+04,\
                    dtype=float16),
            y: finfo(resolution=1e-15, min=-1.7976931348623157e+308, \
                max=1.7976931348623157e+308, dtype=float64)
        }
        """
        return self._static_finfo(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_iinfo(
        type: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """`aikit.Container` static method variant of `aikit.iinfo`. This method
        simply wraps the function, and so the docstring for `aikit.iinfo` also
        applies to this method with minimal changes.

        Parameters
        ----------
        type
            input container with leaves to inquire information about.

        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.

        to_apply
            Boolean indicating whether to apply the
            method to the key-chains. Default is ``True``.

        prune_unapplied
            Boolean indicating whether to prune the
            key-chains that were not applied. Default is ``False``.

        map_sequences
            Boolean indicating whether to map method
            to sequences (list, tuple). Default is ``False``.

        Returns
        -------
        ret
            container of the same structure as `type`, with each element
            as an iinfo object for the corresponding dtype of
            leave in`type`.

        Examples
        --------
        >>> c = aikit.Container(x=aikit.array([12,-1800,1084], dtype=aikit.int16),
        ...                   y=aikit.array([-40000,99,1], dtype=aikit.int32))
        >>> y = aikit.Container.static_iinfo(c)
        >>> print(y)
        {
            x: iinfo(min=-32768, max=32767, dtype=int16),
            y: iinfo(min=-2147483648, max=2147483647, dtype=int32)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "iinfo",
            type,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def iinfo(
        self: aikit.Container,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """`aikit.Container` instance method variant of `aikit.iinfo`. This method
        simply wraps the function, and so the docstring for `aikit.iinfo` also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container with leaves to inquire information about.

        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.

        to_apply
            Boolean indicating whether to apply the
            method to the key-chains. Default is ``True``.

        prune_unapplied
            Boolean indicating whether to prune the
            key-chains that were not applied. Default is ``False``.

        map_sequences
            Boolean indicating whether to map method
            to sequences (list, tuple). Default is ``False``.

        Returns
        -------
        ret
            container of the same structure as `self`, with each element
            as an iinfo object for the corresponding dtype of
            leave in`self`.

        Examples
        --------
        >>> c = aikit.Container(x=aikit.array([-9,1800,89], dtype=aikit.int16),
        ...                   y=aikit.array([76,-81,16], dtype=aikit.int32))
        >>> c.iinfo()
        {
            x: iinfo(min=-32768, max=32767, dtype=int16),
            y: iinfo(min=-2147483648, max=2147483647, dtype=int32)
        }

        >>> c = aikit.Container(x=aikit.array([-12,123,4], dtype=aikit.int8),
        ...                   y=aikit.array([76,-81,16], dtype=aikit.int16))
        >>> c.iinfo()
        {
            x: iinfo(min=-128, max=127, dtype=int8),
            y: iinfo(min=-32768, max=32767, dtype=int16)
        }
        """
        return self._static_iinfo(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_is_bool_dtype(
        dtype_in: aikit.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "is_bool_dtype",
            dtype_in,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def is_bool_dtype(
        self: aikit.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        return self._static_is_bool_dtype(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_is_float_dtype(
        dtype_in: aikit.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """`aikit.Container` static method variant of `is_float_dtype`. This
        method simply wraps this function, so the docstring of `is_float_dtype`
        roughly applies to this method.

        Parameters
        ----------
        dtype_in : aikit.Container
            The input to check for float dtype.

        key_chains : Optional[Union[List[str], Dict[str, str]]]
            The key chains to use when mapping over the input.

        to_apply : bool
            Whether to apply the mapping over the input.

        prune_unapplied : bool
            Whether to prune the keys that were not applied.

        map_sequences : bool
            Boolean indicating whether to map method
            to sequences (list, tuple). Default is ``False``.

        Returns
        -------
        ret : bool
            Boolean indicating whether the input has float dtype.

        Examples
        --------
        >>> x = aikit.static_is_float_dtype(aikit.float32)
        >>> print(x)
        True

        >>> x = aikit.static_is_float_dtype(aikit.int64)
        >>> print(x)
        False

        >>> x = aikit.static_is_float_dtype(aikit.int32)
        >>> print(x)
        False

        >>> x = aikit.static_is_float_dtype(aikit.bool)
        >>> print(x)
        False

        >>> arr = aikit.array([1.2, 3.2, 4.3], dtype=aikit.float32)
        >>> print(arr.is_float_dtype())
        True

        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]), b=aikit.array([3, 4, 5]))
        >>> print(x.a.dtype, x.b.dtype)
        float32 int32
        """
        return ContainerBase.cont_multi_map_in_function(
            "is_float_dtype",
            dtype_in,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def is_float_dtype(
        self: aikit.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """`aikit.Container` instance method variant of `aikit.is_float_dtype`.
        This method simply wraps the function, and so the docstring for
        `aikit.is_float_dtype` also applies to this method with minimal changes.

        Parameters
        ----------
        self : aikit.Container
            The `aikit.Container` instance to call `aikit.is_float_dtype` on.

        key_chains : Union[List[str], Dict[str, str]]
            The key-chains to apply or not apply the method to.
            Default is ``None``.

        to_apply : bool
            Boolean indicating whether to apply the
            method to the key-chains. Default is ``False``.

        prune_unapplied : bool
            Boolean indicating whether to prune the
            key-chains that were not applied. Default is ``False``.

        map_sequences : bool
            Boolean indicating whether to map method
            to sequences (list, tuple). Default is ``False``.

        Returns
        -------
        ret : bool
            Boolean of whether the input is of a float dtype.

        Examples
        --------
        >>> x = aikit.is_float_dtype(aikit.float32)
        >>> print(x)
        True

        >>> x = aikit.is_float_dtype(aikit.int64)
        >>> print(x)
        False

        >>> x = aikit.is_float_dtype(aikit.int32)
        >>> print(x)
        False

        >>> x = aikit.is_float_dtype(aikit.bool)
        >>> print(x)
        False

        >>> arr = aikit.array([1.2, 3.2, 4.3], dtype=aikit.float32)
        >>> print(arr.is_float_dtype())
        True

        >>> x = aikit.Container(a=aikit.array([0., 1., 2.]), b=aikit.array([3, 4, 5]))
        >>> print(x.a.dtype, x.b.dtype)
        float32 int32
        """
        return self._static_is_float_dtype(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_is_int_dtype(
        dtype_in: aikit.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "is_int_dtype",
            dtype_in,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def is_int_dtype(
        self: aikit.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        return self._static_is_int_dtype(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_is_uint_dtype(
        dtype_in: aikit.Container,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "is_uint_dtype",
            dtype_in,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def is_uint_dtype(
        self: aikit.Container,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        return self._static_is_uint_dtype(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_is_complex_dtype(
        dtype_in: aikit.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """`aikit.Container` static method variant of `is_complex_dtype`. This
        method simply wraps this function, so the docstring of
        `is_complex_dtype` roughly applies to this method.

        Parameters
        ----------
        dtype_in : aikit.Container
            The input to check for complex dtype.

        key_chains : Optional[Union[List[str], Dict[str, str]]]
            The key chains to use when mapping over the input.

        to_apply : bool
            Whether to apply the mapping over the input.

        prune_unapplied : bool
            Whether to prune the keys that were not applied.

        map_sequences : bool
            Boolean indicating whether to map method
            to sequences (list, tuple). Default is ``False``.

        Returns
        -------
        ret : bool
            Boolean indicating whether the input has float dtype.

        Examples
        --------
        >>> x = aikit.Container.static_is_complex_dtype(aikit.complex64)
        >>> print(x)
        True

        >>> x = aikit.Container.static_is_complex_dtype(aikit.int64)
        >>> print(x)
        False

        >>> x = aikit.Container.static_is_complex_dtype(aikit.float32)
        >>> print(x)
        False
        """
        return ContainerBase.cont_multi_map_in_function(
            "is_complex_dtype",
            dtype_in,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def is_complex_dtype(
        self: aikit.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """`aikit.Container` instance method variant of `aikit.is_complex_dtype`.
        This method simply wraps the function, and so the docstring for
        `aikit.is_complex_dtype` also applies to this method with minimal
        changes.

        Parameters
        ----------
        self : aikit.Container
            The `aikit.Container` instance to call `aikit.is_complex_dtype` on.

        key_chains : Union[List[str], Dict[str, str]]
            The key-chains to apply or not apply the method to.
            Default is ``None``.

        to_apply : bool
            Boolean indicating whether to apply the
            method to the key-chains. Default is ``False``.

        prune_unapplied : bool
            Boolean indicating whether to prune the
            key-chains that were not applied. Default is ``False``.

        map_sequences : bool
            Boolean indicating whether to map method
            to sequences (list, tuple). Default is ``False``.

        Returns
        -------
        ret : bool
            Boolean of whether the input is of a complex dtype.

        Examples
        --------
        >>> x = aikit.is_complex_dtype(aikit.complex64)
        >>> print(x)
        True

        >>> x = aikit.is_complex_dtype(aikit.int64)
        >>> print(x)
        False

        >>> x = aikit.is_complex_dtype(aikit.float32)
        >>> print(x)
        False
        """
        return self._static_is_complex_dtype(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_result_type(
        *arrays_and_dtypes: aikit.Container,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """`aikit.Container` static method variant of `aikit.result_type`. This
        method simply wraps the function, and so the docstring for
        `aikit.result_type` also applies to this method with minimal changes.

        Parameters
        ----------
        arrays_and_dtypes
            an arbitrary number of input arrays and/or dtypes.
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
            the dtype resulting from an operation involving the input arrays and dtypes.

        Examples
        --------
        >>> x = aikit.Container(a = aikit.array([0, 1, 2]),
        ...                   b = aikit.array([3., 4., 5.]))
        >>> print(x.a.dtype, x.b.dtype)
        int32 float32

        >>> print(aikit.Container.static_result_type(x, aikit.float64))
        {
            a: float64,
            b: float32
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "result_type",
            *arrays_and_dtypes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def result_type(
        self: aikit.Container,
        *arrays_and_dtypes: aikit.Container,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """`aikit.Container` instance method variant of `aikit.result_type`. This
        method simply wraps the function, and so the docstring for
        `aikit.result_type` also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container from which to cast.
        arrays_and_dtypes
            an arbitrary number of input arrays and/or dtypes.
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
            the dtype resulting from an operation involving the input arrays and dtypes.

        Examples
        --------
        >>> x = aikit.Container(a = aikit.array([3, 3, 3]))
        >>> print(x.a.dtype)
        int32

        >>> y = aikit.Container(b = aikit.float64)
        >>> print(x.result_type(y))
        {
            a: {
                b: float64
            }
        }
        """
        return self._static_result_type(
            self,
            *arrays_and_dtypes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )
