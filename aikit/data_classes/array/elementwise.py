# global
import abc
from typing import Optional, Union, Literal

# local
import aikit


# noinspection PyUnresolvedReferences
class _ArrayWithElementwise(abc.ABC):
    def abs(
        self: Union[float, aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:  # noqa
        """aikit.Array instance method variant of aikit.abs. This method simply
        wraps the function, and so the docstring for aikit.abs also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the absolute value of each element in ``self``. The
            returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = aikit.array([2.6, -6.6, 1.6, -0])
        >>> y = x.abs()
        >>> print(y)
        aikit.array([ 2.6, 6.6, 1.6, 0.])
        """
        return aikit.abs(self, out=out)

    def acosh(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.acosh. This method simply
        wraps the function, and so the docstring for aikit.acosh also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array whose elements each represent the area of a hyperbolic sector.
            Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the inverse hyperbolic cosine
            of each element in ``self``.
            The returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = aikit.array([2., 10.0, 1.0])
        >>> y = x.acosh()
        >>> print(y)
        aikit.array([1.32, 2.99, 0.  ])
        """
        return aikit.acosh(self._data, out=out)

    def acos(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.acos. This method simply
        wraps the function, and so the docstring for aikit.acos also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the inverse cosine of each element in ``self``.
            The  returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = aikit.array([1.0, 0.0, -0.9])
        >>> y = x.acos()
        >>> print(y)
        aikit.array([0.  , 1.57, 2.69])
        """
        return aikit.acos(self._data, out=out)

    def add(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        alpha: Optional[Union[int, float]] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.add. This method simply
        wraps the function, and so the docstring for aikit.add also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a numeric data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.
        alpha
            optional scalar multiplier for ``x2``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise sums. The returned array must have a
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.array([1, 2, 3])
        >>> y = aikit.array([4, 5, 6])
        >>> z = x.add(y)
        >>> print(z)
        aikit.array([5, 7, 9])

        >>> x = aikit.array([1, 2, 3])
        >>> y = aikit.array([4, 5, 6])
        >>> z = x.add(y, alpha=2)
        >>> print(z)
        aikit.array([9, 12, 15])
        """
        return aikit.add(self._data, x2, alpha=alpha, out=out)

    def asin(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.asin. This method simply
        wraps the function, and so the docstring for aikit.asin also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the inverse sine of each element in ``self``. The
            returned array must have the same data type as ``self``.

        Examples
        --------
        Using :class:`aikit.Array` instance method:

        >>> x = aikit.array([-1., 1., 4., 0.8])
        >>> y = x.asin()
        >>> print(y)
        aikit.array([-1.57, 1.57, nan, 0.927])

        >>> x = aikit.array([-3., -0.9, 1.5, 2.8])
        >>> y = aikit.zeros(4)
        >>> x.asin(out=y)
        >>> print(y)
        aikit.array([nan, -1.12, nan, nan])
        """
        return aikit.asin(self._data, out=out)

    def asinh(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.asinh. This method simply
        wraps the function, and so the docstring for aikit.asinh also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array whose elements each represent the area of a hyperbolic sector.
            Should have a floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the inverse hyperbolic sine of each element in ``self``.
            The returned array must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.array([-1., 0., 3.])
        >>> y = x.asinh()
        >>> print(y)
        aikit.array([-0.881,  0.   ,  1.82 ])
        """
        return aikit.asinh(self._data, out=out)

    def atan(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.atan. This method simply
        wraps the function, and so the docstring for aikit.atan also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the inverse tangent of each element in ``self``. The
            returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = aikit.array([1.0, 0.5, -0.5])
        >>> y = x.atan()
        >>> print(y)
        aikit.array([ 0.785,  0.464, -0.464])
        """
        return aikit.atan(self._data, out=out)

    def atan2(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.atan2. This method simply
        wraps the function, and so the docstring for aikit.atan2 also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            first input array corresponding to the y-coordinates.
            Should have a real-valued floating-point data type.
        x2
            second input array corresponding to the x-coordinates.
            Must be compatible with ``self``(see :ref:`broadcasting`).
            Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the inverse tangent of the quotient ``self/x2``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.array([1.0, 0.5, 0.0, -0.5, 0.0])
        >>> y = aikit.array([1.0, 2.0, -1.5, 0, 1.0])
        >>> z = x.atan2(y)
        >>> print(z)
        aikit.array([ 0.785,  0.245,  3.14 , -1.57 ,  0.   ])

        >>> x = aikit.array([1.0, 2.0])
        >>> y = aikit.array([-2.0, 3.0])
        >>> z = aikit.zeros(2)
        >>> x.atan2(y, out=z)
        >>> print(z)
        aikit.array([2.68 , 0.588])

        >>> nan = float("nan")
        >>> x = aikit.array([nan, 1.0, 1.0, -1.0, -1.0])
        >>> y = aikit.array([1.0, +0, -0, +0, -0])
        >>> x.atan2(y)
        aikit.array([  nan,  1.57,  1.57, -1.57, -1.57])

        >>> x = aikit.array([+0, +0, +0, +0, -0, -0, -0, -0])
        >>> y = aikit.array([1.0, +0, -0, -1.0, 1.0, +0, -0, -1.0])
        >>> x.atan2(y)
        aikit.array([0.  , 0.  , 0.  , 3.14, 0.  , 0.  , 0.  , 3.14])
        >>> y.atan2(x)
        aikit.array([ 1.57,  0.  ,  0.  , -1.57,  1.57,  0.  ,  0.  , -1.57])

        >>> inf = float("infinity")
        >>> x = aikit.array([inf, -inf, inf, inf, -inf, -inf])
        >>> y = aikit.array([1.0, 1.0, inf, -inf, inf, -inf])
        >>> z = x.atan2(y)
        >>> print(z)
        aikit.array([ 1.57 , -1.57 ,  0.785,  2.36 , -0.785, -2.36 ])

        >>> x = aikit.array([2.5, -1.75, 3.2, 0, -1.0])
        >>> y = aikit.array([-3.5, 2, 0, 0, 5])
        >>> z = x.atan2(y)
        >>> print(z)
        aikit.array([ 2.52 , -0.719,  1.57 ,  0.   , -0.197])

        >>> x = aikit.array([[1.1, 2.2, 3.3], [-4.4, -5.5, -6.6]])
        >>> y = x.atan2(x)
        >>> print(y)
        aikit.array([[ 0.785,  0.785,  0.785],
            [-2.36 , -2.36 , -2.36 ]])
        """
        return aikit.atan2(self._data, x2, out=out)

    def atanh(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.atanh. This method simply
        wraps the function, and so the docstring for aikit.atanh also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array whose elements each represent the area of a hyperbolic sector.
            Should have a floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the inverse hyperbolic tangent of each element
            in ``self``. The returned array must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.array([0.0, 0.5, -0.9])
        >>> y = x.atanh()
        >>> print(y)
        aikit.array([ 0.   ,  0.549, -1.47 ])
        """
        return aikit.atanh(self._data, out=out)

    def bitwise_and(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.bitwise_and. This method
        simply wraps the function, and so the docstring for aikit.bitwise_and
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have an integer or boolean data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have an integer or boolean data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.array([True, False])
        >>> y = aikit.array([True, True])
        >>> x.bitwise_and(y, out=y)
        >>> print(y)
        aikit.array([ True, False])

        >>> x = aikit.array([[7],[8],[9]])
        >>> y = aikit.native_array([[10],[11],[12]])
        >>> z = x.bitwise_and(y)
        >>> print(z)
        aikit.array([[2],[8],[8]])
        """
        return aikit.bitwise_and(self._data, x2, out=out)

    def bitwise_left_shift(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.bitwise_left_shift. This
        method simply wraps the function, and so the docstring for
        aikit.bitwise_left_shift also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            first input array. Should have an integer or boolean data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a data type determined
            by :ref:`type-promotion`.
        """
        return aikit.bitwise_left_shift(self._data, x2, out=out)

    def bitwise_invert(
        self: aikit.Array, *, out: Optional[aikit.Array] = None
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.bitwise_invert. This method
        simply wraps the function, and so the docstring for aikit.bitiwse_invert
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have an integer or boolean data type.

        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = aikit.array([1, 6, 9])
        >>> y = x.bitwise_invert()
        >>> print(y)
        aikit.array([-2, -7, -10])

        >>> x = aikit.array([False, True])
        >>> y = x.bitwise_invert()
        >>> print(y)
        aikit.array([True, False])
        """
        return aikit.bitwise_invert(self._data, out=out)

    def bitwise_or(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.bitwise_or. This method
        simply wraps the function, and so the docstring for aikit.bitwise_or also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have an integer or boolean data type.
        x2
            second input array. Must be compatible with ``self``

        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = aikit.array([1, 2, 3])
        >>> y = aikit.array([4, 5, 6])
        >>> z = x.bitwise_or(y)
        >>> print(z)
        aikit.array([5, 7, 7])
        """
        return aikit.bitwise_or(self._data, x2, out=out)

    def bitwise_right_shift(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.bitwise_right_shift. This
        method simply wraps the function, and so the docstring for
        aikit.bitwise_right_shift also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            first input array. Should have an integer or boolean data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have an integer or boolean data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> a = aikit.array([[2, 3, 4], [5, 10, 64]])
        >>> b = aikit.array([0, 1, 2])
        >>> y = a.bitwise_right_shift(b)
        >>> print(y)
        aikit.array([[ 2,  1,  1],
                    [ 5,  5, 16]])
        """
        return aikit.bitwise_right_shift(self._data, x2, out=out)

    def bitwise_xor(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.bitwise_xor. This method
        simply wraps the function, and so the docstring for aikit.bitwise_xor
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have an integer or boolean data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> a = aikit.array([[89, 51, 32], [14, 18, 19]])
        >>> b = aikit.array([[[19, 26, 27], [22, 23, 20]]])
        >>> y = a.bitwise_xor(b)
        >>> print(y)
        aikit.array([[[74,41,59],[24,5,7]]])
        """
        return aikit.bitwise_xor(self._data, x2, out=out)

    def ceil(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.ceil. This method simply
        wraps the function, and so the docstring for aikit.ceil also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the rounded result for each element in ``self``. The
            returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = aikit.array([5.5, -2.5, 1.5, -0])
        >>> y = x.ceil()
        >>> print(y)
        aikit.array([ 6., -2.,  2.,  0.])
        """
        return aikit.ceil(self._data, out=out)

    def cos(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.cos. This method simply
        wraps the function, and so the docstring for aikit.cos also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array whose elements are each expressed in radians. Should have a
            floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the cosine of each element in ``self``. The returned
            array must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Array` input:

        >>> x = aikit.array([1., 0., 2.,])
        >>> y = x.cos()
        >>> print(y)
        aikit.array([0.54, 1., -0.416])

        >>> x = aikit.array([-3., 0., 3.])
        >>> y = aikit.zeros(3)
        >>> x.cos(out=y)
        >>> print(y)
        aikit.array([-0.99,  1.  , -0.99])

        >>> x = aikit.array([[0., 1.,], [2., 3.]])
        >>> y = x.cos()
        >>> print(y)
        aikit.array([[1., 0.540], [-0.416, -0.990]])
        """
        return aikit.cos(self._data, out=out)

    def cosh(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.cosh. This method simply
        wraps the function, and so the docstring for aikit.cosh also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array whose elements each represent a hyperbolic angle.
            Should have a floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the hyperbolic cosine of each element in ``self``.
            The returned array must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.array([1., 2., 3.])
        >>> print(x.cosh())
            aikit.array([1.54, 3.76, 10.1])

        >>> x = aikit.array([0.23, 3., -1.2])
        >>> y = aikit.zeros(3)
        >>> print(x.cosh(out=y))
            aikit.array([1.03, 10.1, 1.81])
        """
        return aikit.cosh(self._data, out=out)

    def divide(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.divide. This method simply
        wraps the function, and so the docstring for aikit.divide also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            dividend input array. Should have a real-valued data type.
        x2
            divisor input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Array` inputs:

        >>> x1 = aikit.array([2., 7., 9.])
        >>> x2 = aikit.array([2., 2., 2.])
        >>> y = x1.divide(x2)
        >>> print(y)
        aikit.array([1., 3.5, 4.5])

        With mixed :class:`aikit.Array` and `aikit.NativeArray` inputs:

        >>> x1 = aikit.array([2., 7., 9.])
        >>> x2 = aikit.native_array([2., 2., 2.])
        >>> y = x1.divide(x2)
        >>> print(y)
        aikit.array([1., 3.5, 4.5])
        """
        return aikit.divide(self._data, x2, out=out)

    def equal(
        self: aikit.Array,
        x2: Union[float, aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.equal. This method simply
        wraps the function, and so the docstring for aikit.equal also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            first input array. May have any data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            May have any data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array
            must have a data type of ``bool``.

        Examples
        --------
        With :class:`aikit.Array` inputs:

        >>> x1 = aikit.array([2., 7., 9.])
        >>> x2 = aikit.array([1., 7., 9.])
        >>> y = x1.equal(x2)
        >>> print(y)
        aikit.array([False, True, True])

        With mixed :class:`aikit.Array` and :class:`aikit.NativeArray` inputs:

        >>> x1 = aikit.array([2.5, 7.3, 9.375])
        >>> x2 = aikit.native_array([2.5, 2.9, 9.375])
        >>> y = x1.equal(x2)
        >>> print(y)
        aikit.array([True, False,  True])

        With mixed :class:`aikit.Array` and `float` inputs:

        >>> x1 = aikit.array([2.5, 7.3, 9.375])
        >>> x2 = 7.3
        >>> y = x1.equal(x2)
        >>> print(y)
        aikit.array([False, True, False])

        With mixed :class:`aikit.Container` and :class:`aikit.Array` inputs:

        >>> x1 = aikit.array([3., 1., 0.9])
        >>> x2 = aikit.Container(a=aikit.array([12., 3.5, 6.3]), b=aikit.array([3., 1., 0.9]))
        >>> y = x1.equal(x2)
        >>> print(y)
        {
            a: aikit.array([False, False, False]),
            b: aikit.array([True, True, True])
        }
        """
        return aikit.equal(self._data, x2, out=out)

    def exp(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.exp. This method simply
        wraps the function, and so the docstring for aikit.exp also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the evaluated exponential function result for
            each element in ``self``. The returned array must have a floating-point
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.array([1., 2., 3.])
        >>> print(x.exp())
        aikit.array([ 2.71828198,  7.38905573, 20.08553696])
        """
        return aikit.exp(self._data, out=out)

    def expm1(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.expm1. This method simply
        wraps the function, and so the docstring for aikit.expm1 also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        out
            optional output array, for writing the result to. It must have
            a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the evaluated result for each element in ``x``.
            The returned array must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.array([5.5, -2.5, 1.5, -0])
        >>> y = x.expm1()
        >>> print(y)
        aikit.array([244.   ,  -0.918,   3.48 ,   0.   ])

        >>> y = aikit.array([0., 0.])
        >>> x = aikit.array([5., 0.])
        >>> _ = x.expm1(out=y)
        >>> print(y)
        aikit.array([147.,   0.])
        """
        return aikit.expm1(self._data, out=out)

    def floor(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.floor. This method simply
        wraps the function, and so the docstring for aikit.floor also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the rounded result for each element in ``self``. The
            returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = aikit.array([5.5, -2.5, 1.5, -0])
        >>> y = x.floor()
        >>> print(y)
        aikit.array([ 5., -3.,  1.,  0.])
        """
        return aikit.floor(self._data, out=out)

    def floor_divide(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.floor_divide. This method
        simply wraps the function, and so the docstring for aikit.floor_divide
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            dividend input array. Should have a real-valued data type.
        x2
            divisor input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Array` inputs:

        >>> x1 = aikit.array([13., 7., 8.])
        >>> x2 = aikit.array([3., 2., 7.])
        >>> y = x1.floor_divide(x2)
        >>> print(y)
        aikit.array([4., 3., 1.])

        With mixed :class:`aikit.Array` and :class:`aikit.NativeArray` inputs:

        >>> x1 = aikit.array([13., 7., 8.])
        >>> x2 = aikit.native_array([3., 2., 7.])
        >>> y = x1.floor_divide(x2)
        >>> print(y)
        aikit.array([4., 3., 1.])
        """
        return aikit.floor_divide(self._data, x2, out=out)

    def fmin(
        self: aikit.Array,
        x2: aikit.Array,
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.fmin. This method simply
        wraps the function, and so the docstring for aikit.fmin also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            First input array.
        x2
            Second input array
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Array with element-wise minimums.

        Examples
        --------
        >>> x1 = aikit.array([2, 3, 4])
        >>> x2 = aikit.array([1, 5, 2])
        >>> aikit.fmin(x1, x2)
        aikit.array([1, 3, 2])

        >>> x1 = aikit.array([aikit.nan, 0, aikit.nan])
        >>> x2 = aikit.array([0, aikit.nan, aikit.nan])
        >>> x1.fmin(x2)
        aikit.array([ 0.,  0., nan])
        """
        return aikit.fmin(self._data, x2, out=out)

    def greater(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.greater. This method simply
        wraps the function, and so the docstring for aikit.greater also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a real-valued data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array must
            have a data type of ``bool``.

        Examples
        --------
        >>> x1 = aikit.array([2., 5., 15.])
        >>> x2 = aikit.array([3., 2., 4.])
        >>> y = x1.greater(x2)
        >>> print(y)
        aikit.array([False,  True,  True])
        """
        return aikit.greater(self._data, x2, out=out)

    def greater_equal(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.greater_equal. This method
        simply wraps the function, and so the docstring for aikit.greater_equal
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a real-valued data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array
            must have a data type of ``bool``.

        Examples
        --------
        >>> x = aikit.array([1, 2, 3])
        >>> y = aikit.array([4, 5, 6])
        >>> z = x.greater_equal(y)
        >>> print(z)
        aikit.array([False,False,False])
        """
        return aikit.greater_equal(self._data, x2, out=out)

    def isfinite(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.isfinite. This method
        simply wraps the function, and so the docstring for aikit.isfinite also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing test results. An element ``out_i`` is ``True``
            if ``self_i`` is finite and ``False`` otherwise.
            The returned array must have a data type of ``bool``.

        Examples
        --------
        >>> x = aikit.array([0, aikit.nan, -aikit.inf, float('inf')])
        >>> y = x.isfinite()
        >>> print(y)
        aikit.array([ True, False, False, False])
        """
        return aikit.isfinite(self._data, out=out)

    def isinf(
        self: aikit.Array,
        *,
        detect_positive: bool = True,
        detect_negative: bool = True,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.isinf. This method simply
        wraps the function, and so the docstring for aikit.isinf also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued data type.
        detect_positive
            if ``True``, positive infinity is detected.
        detect_negative
            if ``True``, negative infinity is detected.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing test results. An element ``out_i`` is ``True``
            if ``self_i`` is either positive or negative infinity and ``False``
            otherwise. The returned array must have a data type of ``bool``.

        Examples
        --------
        With :class:`aikit.Array` inputs:

        >>> x = aikit.array([1, 2, 3])
        >>> x.isinf()
        aikit.array([False, False, False])

        >>> x = aikit.array([[1.1, 2.3, -3.6]])
        >>> x.isinf()
        aikit.array([[False, False, False]])

        >>> x = aikit.array([[[1.1], [float('inf')], [-6.3]]])
        >>> x.isinf()
        aikit.array([[[False],[True],[False]]])

        >>> x = aikit.array([[-float('inf'), float('inf'), 0.0]])
        >>> x.isinf()
        aikit.array([[ True, True, False]])

        >>> x = aikit.zeros((3, 3))
        >>> x.isinf()
        aikit.array([[False, False, False],
            [False, False, False],
            [False, False, False]])
        """
        return aikit.isinf(
            self._data,
            detect_positive=detect_positive,
            detect_negative=detect_negative,
            out=out,
        )

    def isnan(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.isnan. This method simply
        wraps the function, and so the docstring for aikit.isnan also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing test results. An element ``out_i`` is ``True``
            if ``self_i`` is ``NaN`` and ``False`` otherwise.
            The returned array should have a data type of ``bool``.

        Examples
        --------
        With :class:`aikit.Array` inputs:

        >>> x = aikit.array([1, 2, 3])
        >>> x.isnan()
        aikit.array([False, False, False])

        >>> x = aikit.array([[1.1, 2.3, -3.6]])
        >>> x.isnan()
        aikit.array([[False, False, False]])

        >>> x = aikit.array([[[1.1], [float('inf')], [-6.3]]])
        >>> x.isnan()
        aikit.array([[[False],
                [False],
                [False]]])

        >>> x = aikit.array([[-float('nan'), float('nan'), 0.0]])
        >>> x.isnan()
        aikit.array([[ True, True, False]])

        >>> x = aikit.array([[-float('nan'), float('inf'), float('nan'), 0.0]])
        >>> x.isnan()
        aikit.array([[ True, False,  True, False]])

        >>> x = aikit.zeros((3, 3))
        >>> x.isnan()
        aikit.array([[False, False, False],
            [False, False, False],
            [False, False, False]])
        """
        return aikit.isnan(self._data, out=out)

    def less(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.less. This method simply
        wraps the function, and so the docstring for aikit.less also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a real-valued data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array
            must have a data type of ``bool``.

        Examples
        --------
        >>> x1 = aikit.array([2., 5., 15.])
        >>> x2 = aikit.array([3., 2., 4.])
        >>> y = x1.less(x2)
        >>> print(y)
        aikit.array([ True, False, False])
        """
        return aikit.less(self._data, x2, out=out)

    def less_equal(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.less_equal. This method
        simply wraps the function, and so the docstring for aikit.less_equal also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a real-valued data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array
            must have a data type of ``bool``.

        Examples
        --------
        With :code:'aikit.Array' inputs:

        >>> x1 = aikit.array([1, 2, 3])
        >>> x2 = aikit.array([2, 2, 1])
        >>> y = x1.less_equal(x2)
        >>> print(y)
        aikit.array([True, True, False])

        With mixed :code:'aikit.Array' and :code:'aikit.NativeArray' inputs:

        >>> x1 = aikit.array([2.5, 3.3, 9.24])
        >>> x2 = aikit.native_array([2.5, 1.1, 9.24])
        >>> y = x1.less_equal(x2)
        >>> print(y)
        aikit.array([True, False, True])

        With mixed :code:'aikit.Container' and :code:'aikit.Array' inputs:

        >>> x1 = aikit.array([3., 1., 0.8])
        >>> x2 = aikit.Container(a=aikit.array([2., 1., 0.7]), b=aikit.array([3., 0.6, 1.2]))
        >>> y = x1.less_equal(x2)
        >>> print(y)
        {
            a: aikit.array([False, True, False]),
            b: aikit.array([True, False, True])
        }
        """
        return aikit.less_equal(self._data, x2, out=out)

    def log(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.log. This method simply
        wraps the function, and so the docstring for aikit.log also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the evaluated result for each element in ``self``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        Using :class:`aikit.Array` instance method:

        >>> x = aikit.array([4.0, 1, -0.0, -5.0])
        >>> y = x.log()
        >>> print(y)
        aikit.array([1.39, 0., -inf, nan])

        >>> x = aikit.array([float('nan'), -5.0, -0.0, 1.0, 5.0, float('+inf')])
        >>> y = x.log()
        >>> print(y)
        aikit.array([nan, nan, -inf, 0., 1.61, inf])

        >>> x = aikit.array([[float('nan'), 1, 5.0, float('+inf')],
        ...                [+0, -1.0, -5, float('-inf')]])
        >>> y = x.log()
        >>> print(y)
        aikit.array([[nan, 0., 1.61, inf],
                   [-inf, nan, nan, nan]])
        """
        return aikit.log(self._data, out=out)

    def log1p(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.log1p. This method simply
        wraps the function, and so the docstring for aikit.log1p also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the evaluated result for each element in ``self``.
            The returned array must have a real-valued floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.array([1., 2., 3.])
        >>> y = x.log1p()
        >>> print(y)
        aikit.array([0.693, 1.1  , 1.39 ])

        >>> x = aikit.array([0.1 , .001 ])
        >>> x.log1p(out = x)
        >>> print(x)
        aikit.array([0.0953, 0.001 ])
        """
        return aikit.log1p(self._data, out=out)

    def log2(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.log2. This method simply
        wraps the function, and so the docstring for aikit.log2 also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the evaluated base ``2`` logarithm for each element
            in ``self``. The returned array must have a real-valued floating-point
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        Using :code:`aikit.Array` instance method:

        >>> x = aikit.array([5.0, 1, -0.0, -6.0])
        >>> y = aikit.log2(x)
        >>> print(y)
        aikit.array([2.32, 0., -inf, nan])

        >>> x = aikit.array([float('nan'), -5.0, -0.0, 1.0, 5.0, float('+inf')])
        >>> y = x.log2()
        >>> print(y)
        aikit.array([nan, nan, -inf, 0., 2.32, inf])

        >>> x = aikit.array([[float('nan'), 1, 5.0, float('+inf')],\
                            [+0, -2.0, -5, float('-inf')]])
        >>> y = x.log2()
        >>> print(y)
        aikit.array([[nan, 0., 2.32, inf],
                   [-inf, nan, nan, nan]])
        """
        return aikit.log2(self._data, out=out)

    def log10(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.log10. This method simply
        wraps the function, and so the docstring for aikit.log10 also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the evaluated base ``10`` logarithm for each element
            in ``self``. The returned array must have a real-valued
            floating-point data type determined by :ref:`type-promotion`.

        Examples
        --------
        Using :class:`aikit.Array` instance method:

        >>> x = aikit.array([4.0, 1, -0.0, -5.0])
        >>> y = x.log10()
        >>> print(y)
        aikit.array([0.602, 0., -inf, nan])

        >>> x = aikit.array([float('nan'), -5.0, -0.0, 1.0, 5.0, float('+inf')])
        >>> y = x.log10()
        >>> print(y)
        aikit.array([nan, nan, -inf, 0., 0.699, inf])

        >>> x = aikit.array([[float('nan'), 1, 5.0, float('+inf')],
        ...                [+0, -1.0, -5, float('-inf')]])
        >>> y = x.log10()
        >>> print(y)
        aikit.array([[nan, 0., 0.699, inf],
                   [-inf, nan, nan, nan]])
        """
        return aikit.log10(self._data, out=out)

    def logaddexp(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.logaddexp. This method
        simply wraps the function, and so the docstring for aikit.logaddexp also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a real-valued data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a real-valued floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.array([2., 5., 15.])
        >>> y = aikit.array([3., 2., 4.])
        >>> z = x.logaddexp(y)
        >>> print(z)
        aikit.array([ 3.31,  5.05, 15.  ])
        """
        return aikit.logaddexp(self._data, x2, out=out)

    def logaddexp2(
        self: Union[aikit.Array, float, list, tuple],
        x2: Union[aikit.Array, float, list, tuple],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.logaddexp2. This method
        simply wraps the function, and so the docstring for aikit.logaddexp2 also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            First array-like input.
        x2
            Second array-like input
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Element-wise logaddexp2 of x1 and x2.

        Examples
        --------
        >>> x1 = aikit.array([1, 2, 3])
        >>> x2 = aikit.array([4, 5, 6])
        >>> x1.logaddexp2(x2)
        aikit.array([4.169925, 5.169925, 6.169925])
        """
        return aikit.logaddexp2(self._data, x2, out=out)

    def logical_and(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.logical_and. This method
        simply wraps the function, and so the docstring for aikit.logical_and
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a boolean data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a boolean data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array
            must have a data type of ``bool``.

        Examples
        --------
        Using 'aikit.Array' instance:

        >>> x = aikit.array([True, False, True, False])
        >>> y = aikit.array([True, True, False, False])
        >>> z = x.logical_and(y)
        >>> print(z)
        aikit.array([True, False, False, False])
        """
        return aikit.logical_and(self._data, x2, out=out)

    def logical_not(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.logical_not. This method
        simply wraps the function, and so the docstring for aikit.logical_not
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a boolean data type.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a data type of ``bool``.

        Examples
        --------
        With :class:`aikit.Array` input:

        >>> x=aikit.array([0,1,1,0])
        >>> x.logical_not()
        aikit.array([ True, False, False,  True])

        >>> x=aikit.array([2,0,3,9])
        >>> x.logical_not()
        aikit.array([False,  True, False, False])
        """
        return aikit.logical_not(self._data, out=out)

    def logical_or(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.logical_or. This method
        simply wraps the function, and so the docstring for aikit.logical_or also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a boolean data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a boolean data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array
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
        Using :class:`aikit.Array` instance method:

        >>> x = aikit.array([False, 3, 0])
        >>> y = aikit.array([2, True, False])
        >>> z = x.logical_or(y)
        >>> print(z)
        aikit.array([ True,  True, False])
        """
        return aikit.logical_or(self._data, x2, out=out)

    def logical_xor(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.logical_xor. This method
        simply wraps the function, and so the docstring for aikit.logical_xor
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a boolean data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array
            must have a data type of ``bool``.

        Examples
        --------
        >>> x = aikit.array([True, False, True, False])
        >>> y = aikit.array([True, True, False, False])
        >>> z = x.logical_xor(y)
        >>> print(z)
        aikit.array([False,  True,  True, False])
        """
        return aikit.logical_xor(self._data, x2, out=out)

    def multiply(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.multiply. This method
        simply wraps the function, and so the docstring for aikit.multiply also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a real-valued data type.
        x2
            second input array. Must be compatible with the first input array.
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise products.
            The returned array must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :code:`aikit.Array` inputs:

        >>> x1 = aikit.array([3., 5., 7.])
        >>> x2 = aikit.array([4., 6., 8.])
        >>> y = x1.multiply(x2)
        >>> print(y)
        aikit.array([12., 30., 56.])

        With mixed :code:`aikit.Array` and `aikit.NativeArray` inputs:

        >>> x1 = aikit.array([8., 6., 7.])
        >>> x2 = aikit.native_array([1., 2., 3.])
        >>> y = x1.multiply(x2)
        >>> print(y)
        aikit.array([ 8., 12., 21.])
        """
        return aikit.multiply(self._data, x2, out=out)

    def maximum(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        use_where: bool = True,
        out: Optional[aikit.Array] = None,
    ):
        """aikit.Array instance method variant of aikit.maximum. This method simply
        wraps the function, and so the docstring for aikit.maximum also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input array containing elements to maximum threshold.
        x2
            Tensor containing maximum values, must be broadcastable to x1.
        use_where
            Whether to use :func:`where` to calculate the maximum. If ``False``, the
            maximum is calculated using the ``(x + y + |x - y|)/2`` formula. Default is
            ``True``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            An array with the elements of x1, but clipped to not be lower than the x2
            values.

        Examples
        --------
        With :class:`aikit.Array` inputs:
        >>> x = aikit.array([7, 9, 5])
        >>> y = aikit.array([9, 3, 2])
        >>> z = x.maximum(y)
        >>> print(z)
        aikit.array([9, 9, 5])

        >>> x = aikit.array([1, 5, 9, 8, 3, 7])
        >>> y = aikit.array([[9], [3], [2]])
        >>> z = aikit.zeros((3, 6))
        >>> x.maximum(y, out=z)
        >>> print(z)
        aikit.array([[9.,9.,9.,9.,9.,9.],
                   [3.,5.,9.,8.,3.,7.],
                   [2.,5.,9.,8.,3.,7.]])

        >>> x = aikit.array([[7, 3]])
        >>> y = aikit.array([0, 7])
        >>> x.maximum(y, out=x)
        >>> print(x)
        aikit.array([[7, 7]])
        """
        return aikit.maximum(self, x2, use_where=use_where, out=out)

    def minimum(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        use_where: bool = True,
        out: Optional[aikit.Array] = None,
    ):
        """aikit.Array instance method variant of aikit.minimum. This method simply
        wraps the function, and so the docstring for aikit.minimum also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input array containing elements to minimum threshold.
        x2
            Tensor containing minimum values, must be broadcastable to x1.
        use_where
            Whether to use :func:`where` to calculate the minimum. If ``False``, the
            minimum is calculated using the ``(x + y - |x - y|)/2`` formula. Default is
            ``True``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            An array with the elements of x1, but clipped to not exceed the x2 values.

        Examples
        --------
        With :class:`aikit.Array` inputs:

        >>> x = aikit.array([7, 9, 5])
        >>> y = aikit.array([9, 3, 2])
        >>> z = x.minimum(y)
        >>> print(z)
        aikit.array([7, 3, 2])

        >>> x = aikit.array([1, 5, 9, 8, 3, 7])
        >>> y = aikit.array([[9], [3], [2]])
        >>> z = aikit.zeros((3, 6))
        >>> x.minimum(y, out=z)
        >>> print(z)
        aikit.array([[1.,5.,9.,8.,3.,7.],
                   [1.,3.,3.,3.,3.,3.],
                   [1.,2.,2.,2.,2.,2.]])

        >>> x = aikit.array([[7, 3]])
        >>> y = aikit.array([0, 7])
        >>> x.minimum(y, out=x)
        >>> print(x)
        aikit.array([[0, 3]])
        """
        return aikit.minimum(self, x2, use_where=use_where, out=out)

    def negative(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.negative. This method
        simply wraps the function, and so the docstring for aikit.negative also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the evaluated result for each element in ``self``.
            The returned array must have the same data type as ``self``.

        Examples
        --------
        With :class:`aikit.Array` input:

        >>> x = aikit.array([2, 3 ,5, 7])
        >>> y = x.negative()
        >>> print(y)
        aikit.array([-2, -3, -5, -7])

        >>> x = aikit.array([0,-1,-0.5,2,3])
        >>> y = aikit.zeros(5)
        >>> x.negative(out=y)
        >>> print(y)
        aikit.array([-0. ,  1. ,  0.5, -2. , -3. ])

        >>> x = aikit.array([[1.1, 2.2, 3.3],
        ...                [-4.4, -5.5, -6.6]])
        >>> x.negative(out=x)
        >>> print(x)
        aikit.array([[ -1.1, -2.2, -3.3],
        [4.4, 5.5, 6.6]])
        """
        return aikit.negative(self._data, out=out)

    def not_equal(
        self: aikit.Array,
        x2: Union[float, aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.not_equal. This method
        simply wraps the function, and so the docstring for aikit.not_equal also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. May have any data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned
            array must have a data type of ``bool``.

        Examples
        --------
        With :class:`aikit.Array` inputs:

        >>> x1 = aikit.array([2., 7., 9.])
        >>> x2 = aikit.array([1., 7., 9.])
        >>> y = x1.not_equal(x2)
        >>> print(y)
        aikit.array([True, False, False])

        With mixed :class:`aikit.Array` and :class:`aikit.NativeArray` inputs:

        >>> x1 = aikit.array([2.5, 7.3, 9.375])
        >>> x2 = aikit.native_array([2.5, 2.9, 9.375])
        >>> y = x1.not_equal(x2)
        >>> print(y)
        aikit.array([False, True,  False])

        With mixed :class:`aikit.Array` and `float` inputs:

        >>> x1 = aikit.array([2.5, 7.3, 9.375])
        >>> x2 = 7.3
        >>> y = x1.not_equal(x2)
        >>> print(y)
        aikit.array([True, False, True])

        With mixed :class:`aikit.Container` and :class:`aikit.Array` inputs:

        >>> x1 = aikit.array([3., 1., 0.9])
        >>> x2 = aikit.Container(a=aikit.array([12., 3.5, 6.3]), b=aikit.array([3., 1., 0.9]))
        >>> y = x1.not_equal(x2)
        >>> print(y)
        {
            a: aikit.array([True, True, True]),
            b: aikit.array([False, False, False])
        }
        """
        return aikit.not_equal(self._data, x2, out=out)

    def positive(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.positive. This method
        simply wraps the function, and so the docstring for aikit.positive also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the evaluated result for each element in ``self``.
            The returned array must have the same data type as ``self``.

        Examples
        --------
        With :class:`aikit.Array` input:

        >>> x = aikit.array([2, 3 ,5, 7])
        >>> y = x.positive()
        >>> print(y)
        aikit.array([2, 3, 5, 7])

        >>> x = aikit.array([0, -1, -0.5, 2, 3])
        >>> y = aikit.zeros(5)
        >>> x.positive(out=y)
        >>> print(y)
        aikit.array([0., -1., -0.5,  2.,  3.])

        >>> x = aikit.array([[1.1, 2.2, 3.3],
        ...                [-4.4, -5.5, -6.6]])
        >>> x.positive(out=x)
        >>> print(x)
        aikit.array([[ 1.1,  2.2,  3.3],
        [-4.4, -5.5, -6.6]])
        """
        return aikit.positive(self._data, out=out)

    def pow(
        self: aikit.Array,
        x2: Union[int, float, aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.pow. This method simply
        wraps the function, and so the docstring for aikit.pow also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            first input array whose elements correspond to the exponentiation base.
            Should have a real-valued data type.
        x2
            second input array whose elements correspond to the exponentiation
            exponent. Must be compatible with ``self`` (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Array` input:

        >>> x = aikit.array([1, 2, 3])
        >>> y = x.pow(3)
        >>> print(y)
        aikit.array([1, 8, 27])

        >>> x = aikit.array([1.5, -0.8, 0.3])
        >>> y = aikit.zeros(3)
        >>> x.pow(2, out=y)
        >>> print(y)
        aikit.array([2.25, 0.64, 0.09])
        """
        return aikit.pow(self._data, x2, out=out)

    def real(self: aikit.Array, /, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.real. This method simply
        wraps the function, and so the docstring for aikit.real also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing test results. If input in an
            array is real then, it is returned unchanged. on the
            other hand, if it is complex then, it returns real part from it

        Examples
        --------
        >>> x = aikit.array([4+3j, 6+2j, 1-6j])
        >>> x.real()
        aikit.array([4., 6., 1.])
        """
        return aikit.real(self._data, out=out)

    def remainder(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        modulus: bool = True,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.remainder. This method
        simply wraps the function, and so the docstring for aikit.remainder also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            dividend input array. Should have a real-valued data type.
        x2
            divisor input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        modulus
            whether to compute the modulus instead of the remainder.
            Default is ``True``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            Each element-wise result must have the same sign as the respective
            element ``x2_i``. The returned array must have a data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Array` inputs:

        >>> x1 = aikit.array([2., 5., 15.])
        >>> x2 = aikit.array([3., 2., 4.])
        >>> y = x1.remainder(x2)
        >>> print(y)
        aikit.array([2., 1., 3.])

        With mixed :class:`aikit.Array` and :class:`aikit.NativeArray` inputs:

        >>> x1 = aikit.array([11., 4., 18.])
        >>> x2 = aikit.native_array([2., 5., 8.])
        >>> y = x1.remainder(x2)
        >>> print(y)
        aikit.array([1., 4., 2.])
        """
        return aikit.remainder(self._data, x2, modulus=modulus, out=out)

    def round(
        self: aikit.Array, *, decimals: int = 0, out: Optional[aikit.Array] = None
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.round. This method simply
        wraps the function, and so the docstring for aikit.round also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        decimals
            number of decimal places to round to. Default is ``0``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the rounded result for each element in ``self``. The
            returned array must have the same data type as ``self``.

        Examples
        --------
        Using :class:`aikit.Array` instance method:

        >>> x = aikit.array([6.3, -8.1, 0.5, -4.2, 6.8])
        >>> y = x.round()
        >>> print(y)
        aikit.array([ 6., -8.,  0., -4.,  7.])

        >>> x = aikit.array([-94.2, 256.0, 0.0001, -5.5, 36.6])
        >>> y = x.round()
        >>> print(y)
        aikit.array([-94., 256., 0., -6., 37.])

        >>> x = aikit.array([0.23, 3., -1.2])
        >>> y = aikit.zeros(3)
        >>> x.round(out=y)
        >>> print(y)
        aikit.array([ 0.,  3., -1.])

        >>> x = aikit.array([[ -1., -67.,  0.,  15.5,  1.], [3, -45, 24.7, -678.5, 32.8]])
        >>> y = x.round()
        >>> print(y)
        aikit.array([[-1., -67., 0., 16., 1.],
        [3., -45., 25., -678., 33.]])
        """
        return aikit.round(self._data, decimals=decimals, out=out)

    def sign(
        self: aikit.Array,
        *,
        np_variant: Optional[bool] = True,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.sign. This method simply
        wraps the function, and so the docstring for aikit.sign also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the evaluated result for each element in ``self``. The
            returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = aikit.array([5.7, -7.1, 0, -0, 6.8])
        >>> y = x.sign()
        >>> print(y)
        aikit.array([ 1., -1.,  0.,  0.,  1.])

        >>> x = aikit.array([-94.2, 256.0, 0.0001, -0.0001, 36.6])
        >>> y = x.sign()
        >>> print(y)
        aikit.array([-1.,  1.,  1., -1.,  1.])

        >>> x = aikit.array([[ -1., -67.,  0.,  15.5,  1.], [3, -45, 24.7, -678.5, 32.8]])
        >>> y = x.sign()
        >>> print(y)
        aikit.array([[-1., -1.,  0.,  1.,  1.],
        [ 1., -1.,  1., -1.,  1.]])
        """
        return aikit.sign(self._data, np_variant=np_variant, out=out)

    def sin(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.sin. This method simply
        wraps the function, and so the docstring for aikit.sin also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array whose elements are each expressed in radians. Should have a
            floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the sine of each element in ``self``. The returned
            array must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.array([0., 1., 2., 3.])
        >>> y = x.sin()
        >>> print(y)
        aikit.array([0., 0.841, 0.909, 0.141])
        """
        return aikit.sin(self._data, out=out)

    def sinh(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.sinh. This method simply
        wraps the function, and so the docstring for aikit.sinh also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array whose elements each represent a hyperbolic angle.
            Should have a floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the hyperbolic sine of each element in ``self``. The
            returned array must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.array([1., 2., 3.])
        >>> print(x.sinh())
            aikit.array([1.18, 3.63, 10.])

        >>> x = aikit.array([0.23, 3., -1.2])
        >>> y = aikit.zeros(3)
        >>> print(x.sinh(out=y))
            aikit.array([0.232, 10., -1.51])
        """
        return aikit.sinh(self._data, out=out)

    def square(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.square. This method simply
        wraps the function, and so the docstring for aikit.square also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the square of each element in ``self``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Array` instance method:

        >>> x = aikit.array([1, 2, 3])
        >>> y = x.square()
        >>> print(y)
        aikit.array([1, 4, 9])

        >>> x = aikit.array([[1.2, 2, 3.1], [-1, -2.5, -9]])
        >>> x.square(out=x)
        >>> print(x)
        aikit.array([[1.44,4.,9.61],[1.,6.25,81.]])
        """
        return aikit.square(self._data, out=out)

    def sqrt(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.sqrt. This method simply
        wraps the function, and so the docstring for aikit.sqrt also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued floating-point data type.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the square root of each element in ``self``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        Using :class:`aikit.Array` instance method:

        >>> x = aikit.array([[1., 2.],  [3., 4.]])
        >>> y = x.sqrt()
        >>> print(y)
        aikit.array([[1.  , 1.41],
                   [1.73, 2.  ]])
        """
        return aikit.sqrt(self._data, out=out)

    def subtract(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        alpha: Optional[Union[int, float]] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.subtract. This method
        simply wraps the function, and so the docstring for aikit.subtract also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a real-valued data type.
        x2
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        alpha
            optional scalar multiplier for ``x2``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise differences. The returned array
            must have a data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.array([5, 2, 3])
        >>> y = aikit.array([1, 2, 6])
        >>> z = x.subtract(y)
        >>> print(z)
        aikit.array([4, 0, -3])

        >>> x = aikit.array([5., 5, 3])
        >>> y = aikit.array([4, 5, 6])
        >>> z = x.subtract(y, alpha=2)
        >>> print(z)
        aikit.array([-3., -5., -9.])
        """
        return aikit.subtract(self._data, x2, alpha=alpha, out=out)

    def trapz(
        self: aikit.Array,
        /,
        *,
        x: Optional[aikit.Array] = None,
        dx: float = 1.0,
        axis: int = -1,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.trapz. This method simply
        wraps the function, and so the docstring for aikit.trapz also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            The array that should be integrated.
        x
            The sample points corresponding to the input array values.
            If x is None, the sample points are assumed to be evenly spaced
            dx apart. The default is None.
        dx
            The spacing between sample points when x is None. The default is 1.
        axis
            The axis along which to integrate.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Definite integral of n-dimensional array as approximated along
            a single axis by the trapezoidal rule. If the input array is a
            1-dimensional array, then the result is a float. If n is greater
            than 1, then the result is an n-1 dimensional array.

        Examples
        --------
        >>> y = aikit.array([1, 2, 3])
        >>> aikit.trapz(y)
        4.0
        >>> y = aikit.array([1, 2, 3])
        >>> x = aikit.array([4, 6, 8])
        >>> aikit.trapz(y, x=x)
        8.0
        >>> y = aikit.array([1, 2, 3])
        >>> aikit.trapz(y, dx=2)
        8.0
        """
        return aikit.trapz(self._data, x=x, dx=dx, axis=axis, out=out)

    def tan(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.tan. This method simply
        wraps the function, and so the docstring for aikit.tan also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array whose elements are expressed in radians. Should have a
            floating-point data type.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the tangent of each element in ``self``.
            The return must have a floating-point data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.array([0., 1., 2.])
        >>> y = x.tan()
        >>> print(y)
        aikit.array([0., 1.56, -2.19])
        """
        return aikit.tan(self._data, out=out)

    def tanh(
        self: aikit.Array,
        *,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.tanh. This method simply
        wraps the function, and so the docstring for aikit.tanh also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array whose elements each represent a hyperbolic angle.
            Should have a real-valued floating-point data type.
        complex_mode
            optional specifier for how to handle complex data types. See
            ``aikit.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the hyperbolic tangent of each element in ``self``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.array([0., 1., 2.])
        >>> y = x.tanh()
        >>> print(y)
        aikit.array([0., 0.762, 0.964])
        """
        return aikit.tanh(self._data, complex_mode=complex_mode, out=out)

    def trunc(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.trunc. This method simply
        wraps the function, and so the docstring for aikit.trunc also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued data type.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the rounded result for each element in ``self``.
            The returned array must have the same data type as ``self``

        Examples
        --------
        >>> x = aikit.array([-1, 0.54, 3.67, -0.025])
        >>> y = x.trunc()
        >>> print(y)
        aikit.array([-1.,  0.,  3., -0.])
        """
        return aikit.trunc(self._data, out=out)

    def erf(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.erf. This method simply
        wraps the function, and so the docstring for aikit.erf also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array to compute exponential for.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the Gauss error of ``self``.

        Examples
        --------
        >>> x = aikit.array([0, 0.3, 0.7, 1.0])
        >>> x.erf()
        aikit.array([0., 0.328, 0.677, 0.842])
        """
        return aikit.erf(self._data, out=out)

    def exp2(
        self: Union[aikit.Array, float, list, tuple],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.exp2. This method simply
        wraps the function, and so the docstring for aikit.exp2 also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Array-like input.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Element-wise 2 to the power x. This is a scalar if x is a scalar.

        Examples
        --------
        >>> x = aikit.array([1, 2, 3])
        >>> x.exp2()
        aikit.array([2.,    4.,   8.])
        >>> x = [5, 6, 7]
        >>> x.exp2()
        aikit.array([32.,   64.,  128.])
        """
        return aikit.exp2(self._data, out=out)

    def gcd(
        self: Union[aikit.Array, int, list, tuple],
        x2: Union[aikit.Array, int, list, tuple],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.gcd. This method simply
        wraps the function, and so the docstring for aikit.gcd also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            First array-like input.
        x2
            Second array-like input
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Element-wise gcd of |x1| and |x2|.

        Examples
        --------
        >>> x1 = aikit.array([1, 2, 3])
        >>> x2 = aikit.array([4, 5, 6])
        >>> x1.gcd(x2)
        aikit.array([1.,    1.,   3.])
        >>> x1 = aikit.array([1, 2, 3])
        >>> x1.gcd(10)
        aikit.array([1.,   2.,  1.])
        """
        return aikit.gcd(self._data, x2, out=out)

    def nan_to_num(
        self: aikit.Array,
        /,
        *,
        copy: bool = True,
        nan: Union[float, int] = 0.0,
        posinf: Optional[Union[float, int]] = None,
        neginf: Optional[Union[float, int]] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.nan_to_num. This method
        simply wraps the function, and so the docstring for aikit.nan_to_num also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Array input.
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
            optional output array, for writing the result to.

        Returns
        -------
        ret
            Array with the non-finite values replaced.
            If copy is False, this may be x itself.

        Examples
        --------
        >>> x = aikit.array([1, 2, 3, nan])
        >>> x.nan_to_num()
        aikit.array([1.,    1.,   3.,   0.0])
        >>> x = aikit.array([1, 2, 3, inf])
        >>> x.nan_to_num(posinf=5e+100)
        aikit.array([1.,   2.,   3.,   5e+100])
        """
        return aikit.nan_to_num(
            self._data, copy=copy, nan=nan, posinf=posinf, neginf=neginf, out=out
        )

    def angle(
        self: aikit.Array,
        /,
        *,
        deg: bool = False,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.angle. This method simply
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
        >>> z = aikit.array([-1 + 1j, -2 + 2j, 3 - 3j])
        >>> z
        aikit.array([-1.+1.j, -2.+2.j,  3.-3.j])
        >>> aikit.angle(z)
        aikit.array([ 2.35619449,  2.35619449, -0.78539816])
        >>> aikit.set_backend('numpy')
        >>> aikit.angle(z,deg=True)
        aikit.array([135., 135., -45.])
        """
        return aikit.angle(self._data, deg=deg, out=out)

    def reciprocal(
        self: aikit.Array,
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.reciprocal.This method
        simply wraps the function, and so the docstring for aikit.reciprocal also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array to compute the element-wise reciprocal for.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise reciprocal of ``self``.

        Examples
        --------
        >>> x = aikit.array([1, 2, 3])
        >>> y = x.reciprocal()
        >>> print(y)
        aikit.array([1., 0.5, 0.333])
        """
        return aikit.reciprocal(self._data, out=out)

    def deg2rad(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.deg2rad. This method simply
        wraps the function, and so the docstring for aikit.deg2rad also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array. to be converted from degrees to radians.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise conversion from degrees to radians.

        Examples
        --------
        With :class:`aikit.Array` input:

        >>> x=aikit.array([90,180,270,360])
        >>> y=x.deg2rad()
        >>> print(y)
        aikit.array([1.57, 3.14, 4.71, 6.28])
        """
        return aikit.deg2rad(self._data, out=out)

    def rad2deg(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.rad2deg. This method simply
        wraps the function, and so the docstring for aikit.rad2deg also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array. to be converted from degrees to radians.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise conversion from radians to degrees.

        Examples
        --------
        With :class:`aikit.Array` input:

        >>> x=aikit.array([1., 5., 8., 10.])
        >>> y=x.rad2deg()
        >>> print(y)
        aikit.array([ 57.3, 286. , 458. , 573. ])
        """
        return aikit.rad2deg(self._data, out=out)

    def trunc_divide(
        self: aikit.Array,
        x2: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.trunc_divide. This method
        simply wraps the function, and so the docstring for aikit.trunc_divide
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            dividend input array. Should have a real-valued data type.
        x2
            divisor input array. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results.
            The returned array must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Array` inputs:

        >>> x1 = aikit.array([2., 7., 9.])
        >>> x2 = aikit.array([2., -2., 2.])
        >>> y = x1.trunc_divide(x2)
        >>> print(y)
        aikit.array([ 1., -3.,  4.])
        """
        return aikit.trunc_divide(self._data, x2, out=out)

    def isreal(self: aikit.Array, *, out: Optional[aikit.Array] = None) -> aikit.Array:
        """aikit.Array instance method variant of aikit.isreal. This method simply
        wraps the function, and so the docstring for aikit.isreal also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a real-valued data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing test results. An element ``out_i`` is ``True``
            if ``self_i`` is real number and ``False`` otherwise.
            The returned array should have a data type of ``bool``.

        Examples
        --------
        >>> x = aikit.array([1j, 2+5j, 3.7-6j])
        >>> x.isreal()
        aikit.array([False, False, False])
        """
        return aikit.isreal(self._data, out=out)

    def lcm(
        self: aikit.Array, x2: aikit.Array, *, out: Optional[aikit.Array] = None
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.lcm. This method simply
        wraps the function, and so the docstring for aikit.lcm also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            first input array.
        x2
            second input array
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            an array that includes the element-wise least common multiples
            of 'self' and x2

        Examples
        --------
        >>> x1=aikit.array([2, 3, 4])
        >>> x2=aikit.array([5, 8, 15])
        >>> x1.lcm(x2)
        aikit.array([10, 21, 60])
        """
        return aikit.lcm(self, x2, out=out)
