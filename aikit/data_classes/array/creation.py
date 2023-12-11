# global
import abc
from numbers import Number
from typing import Optional, Union, List

# local
import aikit


# Array API Standard #
# -------------------#


class _ArrayWithCreation(abc.ABC):
    def asarray(
        self: aikit.Array,
        /,
        *,
        copy: Optional[bool] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.asarray. This method simply
        wraps the function, and so the docstring for aikit.asarray also applies
        to this method with minimal changes.

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
        With list of lists as input:

        >>> aikit.asarray([[1,2],[3,4]])
        aikit.array([[1, 2],
               [3, 4]])

        With tuple of lists as input:

        >>> aikit.asarray(([1.4,5.6,5.5],[3.1,9.1,7.5]))
        aikit.array([[1.39999998, 5.5999999 , 5.5       ],
               [3.0999999 , 9.10000038, 7.5       ]])

        With ndarray as input:

        >>> x = aikit.np.ndarray(shape=(2,2), order='C')
        >>> x
        array([[6.90786433e-310, 6.90786433e-310],
               [6.90786433e-310, 6.90786433e-310]])
        >>> aikit.asarray(x)
        aikit.array([[6.90786433e-310, 6.90786433e-310],
               [6.90786433e-310, 6.90786433e-310]])
        """
        return aikit.asarray(self._data, copy=copy, dtype=dtype, device=device, out=out)

    def full_like(
        self: aikit.Array,
        /,
        fill_value: float,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.full_like. This method
        simply wraps the function, and so the docstring for aikit.full_like also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array from which to derive the output array shape.
        fill_value
            Scalar fill value
        dtype
            output array data type. If ``dtype`` is `None`, the output array data type
            must be inferred from ``self``. Default: ``None``.
        device
            device on which to place the created array. If ``device`` is ``None``, the
            output array device must be inferred from ``self``. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array having the same shape as ``self`` and where every element is equal
            to ``fill_value``.

        Examples
        --------
        With :code:`int` datatype:

        >>> x = aikit.array([1,2,3])
        >>> fill_value = 0
        >>> x.full_like(fill_value)
        aikit.array([0, 0, 0])

        With float datatype:

        >>> fill_value = 0.000123
        >>> x = aikit.array(aikit.ones(5))
        >>> y = x.full_like(fill_value)
        >>> print(y)
        aikit.array([0.000123, 0.000123, 0.000123, 0.000123, 0.000123])

        With :class:`aikit.Array` input:

        >>> x = aikit.array([1, 2, 3, 4, 5, 6])
        >>> fill_value = 1
        >>> y = x.full_like(fill_value)
        >>> print(y)
        aikit.array([1, 1, 1, 1, 1, 1])
        """
        return aikit.full_like(
            self._data, fill_value=fill_value, dtype=dtype, device=device, out=out
        )

    def ones_like(
        self: aikit.Array,
        /,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.ones_like. This method
        simply wraps the function, and so the docstring for aikit.ones_like also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array from which to derive the output array shape.
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
            an array having the same shape as ``self`` and filled with ones.
        """
        return aikit.ones_like(self._data, dtype=dtype, device=device, out=out)

    def zeros_like(
        self: aikit.Array,
        /,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.zeros_like. This method
        simply wraps the function, and so the docstring for aikit.zeros_like also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array from which to derive the output array shape.
        dtype
            output array data type. If ``dtype`` is ``None``, the output array data type
            must be inferred from ``self``. Default: ``None``.
        device
            device on which to place the created array. If ``device`` is ``None``, the
            output array device must be inferred from ``self``. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array having the same shape as ``self`` and filled with ``zeros``.
        """
        return aikit.zeros_like(self._data, dtype=dtype, device=device, out=out)

    def tril(
        self: aikit.Array, /, *, k: int = 0, out: Optional[aikit.Array] = None
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.tril. This method simply
        wraps the function, and so the docstring for aikit.tril also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array having shape (..., M, N) and whose innermost two dimensions form
            MxN matrices.
        k
            diagonal above which to zero elements. If k = 0, the diagonal is the main
            diagonal. If k < 0, the diagonal is below the main diagonal. If k > 0, the
            diagonal is above the main diagonal. Default: ``0``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the lower triangular part(s). The returned array must
            have the same shape and data type as ``self``. All elements above the
            specified diagonal k must be zeroed. The returned array should be allocated
            on the same device as ``self``.
        """
        return aikit.tril(self._data, k=k, out=out)

    def triu(
        self: aikit.Array, /, *, k: int = 0, out: Optional[aikit.Array] = None
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.triu. This method simply
        wraps the function, and so the docstring for aikit.triu also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array having shape (..., M, N) and whose innermost two dimensions form
            MxN matrices.    *,
        k
            diagonal below which to zero elements. If k = 0, the diagonal is the main
            diagonal. If k < 0, the diagonal is below the main diagonal. If k > 0, the
            diagonal is above the main diagonal. Default: ``0``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the upper triangular part(s). The returned array must
            have the same shape and data type as ``self``. All elements below the
            specified diagonal k must be zeroed. The returned array should be allocated
            on the same device as ``self``.
        """
        return aikit.triu(self._data, k=k, out=out)

    def empty_like(
        self: aikit.Array,
        /,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.empty_like. This method
        simply wraps the function, and so the docstring for aikit.empty_like also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array from which to derive the output array shape.
        dtype
            output array data type. If dtype is None, the output array data type must be
            inferred from ``self``. Default: ``None``.
        device
            device on which to place the created array. If device is None, the output
            array device must be inferred from ``self``. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array having the same shape as ``self`` and containing uninitialized
            data.
        """
        return aikit.empty_like(self._data, dtype=dtype, device=device, out=out)

    def meshgrid(
        self: aikit.Array,
        /,
        *arrays: Union[aikit.Array, aikit.NativeArray],
        sparse: bool = False,
        indexing: str = "xy",
    ) -> List[aikit.Array]:
        """aikit.Array instance method variant of aikit.meshgrid. This method
        simply wraps the function, and so the docstring for aikit.meshgrid also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            one-dimensional input array.
        arrays
            an arbitrary number of one-dimensional arrays representing grid coordinates.
            Each array should have the same numeric data type.
        sparse
            if True, a sparse grid is returned in order to conserve memory. Default:
            ``False``.
        indexing
            Cartesian ``'xy'`` or matrix ``'ij'`` indexing of output. If provided zero
            or one one-dimensional vector(s) (i.e., the zero- and one-dimensional cases,
            respectively), the ``indexing`` keyword has no effect and should be ignored.
            Default: ``'xy'``.

        Returns
        -------
        ret
            list of N arrays, where ``N`` is the number of provided one-dimensional
            input arrays. Each returned array must have rank ``N``. For ``N``
            one-dimensional arrays having lengths ``Ni = len(xi)``.
        """
        return aikit.meshgrid(self, *arrays, sparse=sparse, indexing=indexing)

    def from_dlpack(
        self: aikit.Array,
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.from_dlpack. This method
        simply wraps the function, and so the docstring for aikit.from_dlpack
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the data in ``self``.
        """
        return aikit.from_dlpack(self._data, out=out)

    # Extra #
    # ----- #

    def copy_array(
        self: aikit.Array,
        /,
        *,
        to_aikit_array: bool = True,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.copy_array. This method
        simply wraps the function, and so the docstring for aikit.copy_array also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array
        to_aikit_array
            boolean, if True the returned array will be an aikit.Array object otherwise
            returns an aikit.NativeArray object (i.e. a torch.tensor, np.array, etc.,
            depending on the backend), defaults to True.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            a copy of the input array ``x``.
        """
        return aikit.copy_array(self, to_aikit_array=to_aikit_array, out=out)

    def native_array(
        self: aikit.Array,
        /,
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    ) -> aikit.NativeArray:
        """aikit.Array instance method variant of aikit.native_array. This method
        simply wraps the function, and so the docstring for aikit.native_array
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        dtype
            datatype, optional. Datatype is inferred from the input data.
        device
            device on which to place the created array. Default: ``None``.

        Returns
        -------
        ret
            A native array interpretation of ``self``.
        """
        return aikit.native_array(self._data, dtype=dtype, device=device)

    def one_hot(
        self: aikit.Array,
        depth: int,
        /,
        *,
        on_value: Optional[Number] = None,
        off_value: Optional[Number] = None,
        axis: Optional[int] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.one_hot. This method simply
        wraps the function, and so the docstring for aikit.one_hot also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array containing the indices for which the ones should be scattered
        depth
            Scalar defining the depth of the one-hot dimension.
        on_value
            Value to fill in output when ``indices[j] == i``. Default 1.
        off_value
            Value to fill in output when ``indices[j] != i``. Default 0.
        axis
            The axis to scatter on. The default is ``-1`` which is the last axis.
        dtype
            The data type of the output array. If None, the data type of the on_value is
            used, or if that is None, the data type of the off_value is used. Default
            float32.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            Same as x if None.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Tensor of zeros with the same shape and type as a, unless dtype provided
            which overrides.

        Examples
        --------
        With :class:`aikit.Array` inputs:

        >>> x = aikit.array([3, 1])
        >>> y = 5
        >>> z = x.one_hot(5)
        >>> print(z)
        aikit.array([[0., 0., 0., 1., 0.],
        ...    [0., 1., 0., 0., 0.]])

        >>> x = aikit.array([0])
        >>> y = 5
        >>> aikit.one_hot(x, y)
        aikit.array([[1., 0., 0., 0., 0.]])

        >>> x = aikit.array([0])
        >>> y = 5
        >>> aikit.one_hot(x, 5, out=z)
        aikit.array([[1., 0., 0., 0., 0.]])
        >>> print(z)
        aikit.array([[1., 0., 0., 0., 0.]])
        """
        return aikit.one_hot(
            self,
            depth,
            on_value=on_value,
            off_value=off_value,
            axis=axis,
            dtype=dtype,
            device=device,
            out=out,
        )

    def linspace(
        self: aikit.Array,
        stop: Union[aikit.Array, aikit.NativeArray, float],
        /,
        num: int,
        *,
        axis: Optional[int] = None,
        endpoint: bool = True,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Array:
        return aikit.linspace(
            self,
            stop,
            num=num,
            axis=axis,
            endpoint=endpoint,
            dtype=dtype,
            device=device,
            out=out,
        )

    def logspace(
        self: aikit.Array,
        stop: Union[aikit.Array, aikit.NativeArray, float],
        /,
        num: int,
        *,
        base: float = 10.0,
        axis: int = 0,
        endpoint: bool = True,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.logspace. This method
        simply wraps the function, and so the docstring for aikit.logspace also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            First value in the range in log space. base ** start is the starting value
            in the sequence. Can be an array or a float.
        stop
            Last value in the range in log space. base ** stop is the final value in the
            sequence. Can be an array or a float.
        num
            Number of values to generate.
        base
            The base of the log space. Default is 10.0
        axis
            Axis along which the operation is performed. Relevant only if start or stop
            are array-like. Default is 0.
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
            Tensor of evenly-spaced values in log space.

        Both the description and the type hints above assumes an array input for
        simplicity, but this function is *nestable*, and therefore also accepts
        :class:`aikit.Container` instances in place of any of the arguments.

        Examples
        --------
        With float input:

        >>> x = aikit.array([1, 2])
        >>> y = aikit.array([4, 5])
        >>> x.logspace(y, 4)
        aikit.array([[1.e+01, 1.e+02],
                   [1.e+02, 1.e+03],
                   [1.e+03, 1.e+04],
                   [1.e+04, 1.e+05])

        >>> x.logspace(y, 4, axis = 1)
        aikit.array([[[1.e+01, 1.e+02, 1.e+03, 1.e+04],
                   [1.e+02, 1.e+03, 1.e+04, 1.e+05]]])

        >>> x = aikit.array([1, 2])
        >>> y = aikit.array([4])      # Broadcasting example
        >>> x.logspace(y, 4)
        aikit.array([[10., 100.]
                   [100., 464.15888336]
                   [1000., 2154.43469003]
                   [10000., 10000.]])
        """
        return aikit.logspace(
            self,
            stop,
            num=num,
            base=base,
            axis=axis,
            endpoint=endpoint,
            dtype=dtype,
            device=device,
            out=out,
        )
