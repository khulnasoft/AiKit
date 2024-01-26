# global
from typing import Union, Optional, Sequence

# local
import aikit
from aikit.utils.backend import current_backend
from aikit.func_wrapper import (
    handle_array_function,
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
    handle_device,
    handle_backend_invalid,
)
from aikit.utils.exceptions import handle_exceptions


# Helpers #
# --------#


def _get_promoted_type_of_operands(operands):
    dtype = None
    for operand in operands:
        operand_dtype = aikit.as_aikit_dtype(operand.dtype)
        if dtype is None:
            dtype = operand_dtype
        else:
            dtype = aikit.promote_types(dtype, operand_dtype)
    return aikit.as_native_dtype(dtype)


# Array API Standard #
# -------------------#


@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def min(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float, complex]] = None,
    where: Optional[aikit.Array] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Calculate the minimum value of the input array ``x``.

    .. note::
       When the number of elements over which to compute the minimum value is zero, the
       minimum value is implementation-defined. Specification-compliant libraries may
       choose to raise an error, return a sentinel value (e.g., if ``x`` is a
       floating-point input array, return ``NaN``), or return the maximum possible value
       for the input array ``x`` data type (e.g., if ``x`` is a floating-point array,
       return ``+infinity``).

    **Special Cases**

    For floating-point operands,

    -   If ``x_i`` is ``NaN``, the minimum value is ``NaN``
        (i.e., ``NaN`` values propagate).

    Parameters
    ----------
    x
        Input array. Should have a real-valued data type.
    axis
        axis or axes along which minimum values must be computed. By default, the
        minimum value must be computed over the entire array. If a tuple of integers,
        minimum values must be computed over multiple axes. Default: ``None``.

    keepdims
        optional boolean, if ``True``, the reduced axes (dimensions) must be included
        in the result as singleton dimensions, and, accordingly, the result must be
        compatible with the input array (see :ref:`broadcasting`). Otherwise,
        if ``False``, the reduced axes (dimensions) must not be included in the result.
        Default: ``False``.
    initial
        The maximum value of an output element.
        Must be present to allow computation on empty slice.
    where
        Elements to compare for minimum
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        if the minimum value was computed over the entire array, a zero-dimensional
        array containing the minimum value; otherwise, a non-zero-dimensional array
        containing the minimum values. The returned array must have the same data type
        as ``x``.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.min.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([1, 2, 3])
    >>> z = aikit.min(x)
    >>> print(z)
    aikit.array(1)

    >>> x = aikit.array([0, 1, 2])
    >>> z = aikit.array([0, 0, 0])
    >>> y = aikit.min(x, out=z)
    >>> print(z)
    aikit.array(0)

    >>> x = aikit.array([[0, 1, 2], [4, 6, 10]])
    >>> y = aikit.min(x, axis=0, keepdims=True)
    >>> print(y)
    aikit.array([[0, 1, 2]])

    >>> x = aikit.native_array([[0, 1, 2], [4, 6, 10]])
    >>> y = aikit.min(x)
    >>> print(y)
    aikit.array(0)

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([1, 2, 3]), b=aikit.array([2, 3, 4]))
    >>> z = aikit.min(x)
    >>> print(z)
    {
        a: aikit.array(1),
        b: aikit.array(2)
    }
    """
    return current_backend(x).min(
        x, axis=axis, keepdims=keepdims, initial=initial, where=where, out=out
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def max(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Calculate the maximum value of the input array ``x``.

    .. note::
       When the number of elements over which to compute the maximum value is zero, the
       maximum value is implementation-defined. Specification-compliant libraries may
       choose to raise an error, return a sentinel value (e.g., if ``x`` is a
       floating-point input array, return ``NaN``), or return the minimum possible
       value for the input array ``x`` data type (e.g., if ``x`` is a floating-point
       array, return ``-infinity``).

    **Special Cases**

    For floating-point operands,

    -   If ``x_i`` is ``NaN``, the maximum value is ``NaN`` (i.e., ``NaN`` values
        propagate).

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    axis
        axis or axes along which maximum values must be computed. By default, the
        maximum value must be computed over the entire array. If a tuple of integers,
        maximum values must be computed over multiple axes. Default: ``None``.
    keepdims
        if ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes
        (dimensions) must not be included in the result. Default: ``False``.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        if the maximum value was computed over the entire array, a zero-dimensional
        array containing the maximum value; otherwise, a non-zero-dimensional array
        containing the maximum values. The returned array must have the same data type
        as ``x``.


    This method conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.max.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([1, 2, 3])
    >>> z = aikit.max(x)
    >>> print(z)
    aikit.array(3)

    >>> x = aikit.array([0, 1, 2])
    >>> z = aikit.array(0)
    >>> y = aikit.max(x, out=z)
    >>> print(z)
    aikit.array(2)

    >>> x = aikit.array([[0, 1, 2], [4, 6, 10]])
    >>> y = aikit.max(x, axis=0, keepdims=True)
    >>> print(y)
    aikit.array([[4, 6, 10]])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([0., 1., 2.]), b=aikit.array([3., 4., 5.]))
    >>> y = aikit.max(x)
    >>> print(y)
    {
        a: aikit.array(2.),
        b: aikit.array(5.)
    }

    >>> x = aikit.Container(a=aikit.array([[1, 2, 3],[-1,0,2]]),
    ...                   b=aikit.array([[2, 3, 4], [0, 1, 2]]))
    >>> z = aikit.max(x, axis=1)
    >>> print(z)
    {
        a: aikit.array([3, 2]),
        b: aikit.array([4, 2])
    }
    """
    return current_backend(x).max(x, axis=axis, keepdims=keepdims, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def mean(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Calculate the arithmetic mean of the input array ``x``.

    **Special Cases**

    Let ``N`` equal the number of elements over which to compute the arithmetic mean.
    -   If ``N`` is ``0``, the arithmetic mean is ``NaN``.
    -   If ``x_i`` is ``NaN``, the arithmetic mean is ``NaN`` (i.e., ``NaN`` values
        propagate).

    Parameters
    ----------
    x
        input array. Should have a floating-point data type.
    axis
        axis or axes along which arithmetic means must be computed. By default, the mean
        must be computed over the entire array. If a Sequence of integers, arithmetic
        means must be computed over multiple axes. Default: ``None``.
    keepdims
        bool, if ``True``, the reduced axes (dimensions) must be included in the result
        as singleton dimensions, and, accordingly, the result must be compatible with
        the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced
        axes (dimensions) must not be included in the result. Default: ``False``.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        array, if the arithmetic mean was computed over the entire array, a
        zero-dimensional array containing the arithmetic mean; otherwise, a
        non-zero-dimensional array containing the arithmetic means. The returned
        array must have the same data type as ``x``.
        .. note::
           While this specification recommends that this function only accept input
           arrays having a floating-point data type, specification-compliant array
           libraries may choose to accept input arrays having an integer data type.
           While mixed data type promotion is implementation-defined, if the input
           array ``x`` has an integer data type, the returned array must have the
           default floating-point data type.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/
    signatures.statistical_functions.mean.html>`_ in the standard.

    Both the description and the type hints above assumes an array input for
    simplicity, but this function is *nestable*, and therefore also accepts
    :class:`aikit.Container` instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([3., 4., 5.])
    >>> y = aikit.mean(x)
    >>> print(y)
    aikit.array(4.)

    >>> x = aikit.array([0., 1., 2.])
    >>> y = aikit.array(0.)
    >>> aikit.mean(x, out=y)
    >>> print(y)
    aikit.array(1.)

    >>> x = aikit.array([[-1., -2., -3., 0., -1.], [1., 2., 3., 0., 1.]])
    >>> y = aikit.array([0., 0.])
    >>> aikit.mean(x, axis=1, out=y)
    >>> print(y)
    aikit.array([-1.4,  1.4])


    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([-1., 0., 1.]), b=aikit.array([1.1, 0.2, 1.4]))
    >>> y = aikit.mean(x)
    >>> print(y)
    {
        a: aikit.array(0.),
        b: aikit.array(0.90000004)
    }

    >>> x = aikit.Container(a=aikit.array([[0., 1., 2.], [3., 4., 5.]]),
    ...                   b=aikit.array([[3., 4., 5.], [6., 7., 8.]]))
    >>> y = aikit.Container(a = aikit.zeros(3), b = aikit.zeros(3))
    >>> aikit.mean(x, axis=0, out=y)
    >>> print(y)
    {
        a: aikit.array([1.5, 2.5, 3.5]),
        b: aikit.array([4.5, 5.5, 6.5])
    }
    """
    return current_backend(x).mean(x, axis=axis, keepdims=keepdims, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def prod(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    keepdims: bool = False,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Calculate the product of input array x elements.

    **Special Cases**

    Let ``N`` equal the number of elements over which to compute the product.

    -  If ``N`` is ``0``, the product is ``1`` (i.e., the empty product).

    For both both real-valued and complex floating-point operands, special
    cases must be handled as the operation is implemented by successive application
    of :func:`aikit.multiply`:

    Parameters
    ----------
    x
        input array. Should have a numeric data type.
    axis
        axis or axes along which products must be computed. By default, the product must
        be computed over the entire array. If a tuple of integers, products must be
        computed over multiple axes. Default: ``None``.
    keepdims
        bool, if True, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see Broadcasting). Otherwise, if False, the reduced axes
        (dimensions) must not be included in the result. Default: ``False``.
    dtype
        data type of the returned array. If None,
        if the default data type corresponding to the data type “kind” (integer or
        floating-point) of x has a smaller range of values than the data type of x
        (e.g., x has data type int64 and the default data type is int32, or x has data
        type uint64 and the default data type is int64), the returned array must have
        the same data type as x. if x has a floating-point data type, the returned array
        must have the default floating-point data type. if x has a signed integer data
        type (e.g., int16), the returned array must have the default integer data type.
        if x has an unsigned integer data type (e.g., uint16), the returned array must
        have an unsigned integer data type having the same number of bits as the default
        integer data type (e.g., if the default integer data type is int32, the returned
        array must have a uint32 data type). If the data type (either specified or
        resolved) differs from the data type of x, the input array should be cast to the
        specified data type before computing the product. Default: ``None``.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        array,  if the product was computed over the entire array, a zero-dimensional
        array containing the product; otherwise, a non-zero-dimensional array containing
        the products. The returned array must have a data type as described by the dtype
        parameter above.


    This method conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.prod.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([1, 2, 3])
    >>> z = aikit.prod(x)
    >>> print(z)
    aikit.array(6)

    >>> x = aikit.array([1, 0, 3])
    >>> z = aikit.prod(x)
    >>> print(z)
    aikit.array(0)

    >>> x = aikit.array([[3., 4., 5.]])
    >>> y = aikit.prod(x, keepdims=True)
    >>> print(y)
    aikit.array([60.])

    >>> x = aikit.array([2., 1.])
    >>> y = aikit.array(0.)
    >>> aikit.prod(x, out=y)
    >>> print(y)
    aikit.array(2.)

    >>> x = aikit.array([[-1., -2.], [3., 3.]])
    >>> y = aikit.prod(x, axis=1)
    >>> print(y)
    aikit.array([2., 9.])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([-1., 0., 1.]), b=aikit.array([1.1, 0.2, 1.4]))
    >>> y = aikit.prod(x)
    >>> print(y)
    {
        a: aikit.array(-0.),
        b: aikit.array(0.30800003)
    }

    >>> x = aikit.Container(a=aikit.array([[1., 2.], [3., 4.]]),
    ...                   b=aikit.array([[ 4., 5.], [5., 6.]]))
    >>> y = aikit.prod(x, axis=1, keepdims=True)
    >>> print(y)
    {
        a: aikit.array([[2.],
                      [12.]]),
        b: aikit.array([[20.],
                      [30.]])
    }
    """
    return current_backend(x).prod(
        x, axis=axis, dtype=dtype, keepdims=keepdims, out=out
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def std(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Calculate the standard deviation of the input array ``x``.

    **Special Cases**

    Let ``N`` equal the number of elements over which to compute the standard deviation.

    -   If ``N - correction`` is less than or equal to ``0``,
        the standard deviation is ``NaN``.
    -   If ``x_i`` is ``NaN``, the standard deviation is ``NaN``
        (i.e., ``NaN`` values propagate).

    Parameters
    ----------
    x
        input array.
    axis
        axis or axes along which standard deviations must be computed. By default, the
        standard deviation must be computed over the entire array. If a tuple of
        integers, standard deviations must be computed over multiple axes.
        Default: ``None``.
    correction
        degrees of freedom adjustment. Setting this parameter to a value other
        than ``0`` has the effect of adjusting the divisor during the calculation of the
        standard deviation according to ``N-c`` where ``N`` corresponds to the total
        number of elements over which the standard deviation is computed and ``c``
        corresponds to the provided degrees of freedom adjustment. When computing the
        standard deviation of a population, setting this parameter to ``0`` is the
        standard choice (i.e., the provided array contains data constituting an
        entire population). When computing the corrected sample standard deviation,
        setting this parameter to ``1`` is the standard choice (i.e., the provided array
        contains data sampled from a larger population; this is commonly referred to as
        Bessel's correction).
        Default: ``0``.
    keepdims
        if ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes
        (dimensions) must not be included in the result. Default: ``False``.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        if the standard deviation was computed over the entire array, a zero-dimensional
        array containing the standard deviation; otherwise, a non-zero-dimensional array
        containing the standard deviations. The returned array must have the same data
        type as ``x``.

        .. note::
           While this specification recommends that this function only accept input
           arrays having a real-valued floating-point data type, specification-compliant
           array libraries may choose to accept input arrays having an integer data
           type. While mixed data type promotion is implementation-defined, if the input
           array ``x`` has an integer data type, the returned array must have
           the default real-valued floating-point data type.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.std.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    >>> x = aikit.array([-1., 0., 1.])
    >>> y = aikit.std(x)
    >>> print(y)
    aikit.array(0.81649661)

    >>> x = aikit.array([-1., 0., 1.])
    >>> z = aikit.std(x, correction=1)
    >>> print(z)
    aikit.array(1.)

    >>> x = aikit.array([[0., 4.]])
    >>> y = aikit.std(x, keepdims=True)
    >>> print(y)
    aikit.array([[2.]])

    >>> x = aikit.array([2., 1.])
    >>> y = aikit.array(0.)
    >>> aikit.std(x, out=y)
    >>> print(y)
    aikit.array(0.5)

    >>> x = aikit.array([[-1., -2.], [3., 3.]])
    >>> y = aikit.std(x, axis=1)
    >>> print(y)
    aikit.array([0.5, 0. ])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([-1., 0., 1.]), b=aikit.array([1.1, 0.2, 1.4]))
    >>> y = x.std()
    >>> print(y)
    {
        a: aikit.array(0.81649661),
        b: aikit.array(0.509902)
    }

    >>> x = aikit.Container(a=aikit.array([[1., 3.], [3., 6.]]),
    ...                   b=aikit.array([[ 4., 2.], [2., 1.]]))
    >>> y = x.std(axis=1, keepdims=True)
    >>> print(y)
    {
        a: aikit.array([[1.],
                      [1.5]]),
        b: aikit.array([[1.],
                      [0.5]])
    }
    """
    return current_backend(x).std(
        x, axis=axis, correction=correction, keepdims=keepdims, out=out
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def sum(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Calculate the sum of the input array x.

    **Special Cases**

    Let ``N`` equal the number of elements over which to compute the sum.
    -   If ``N`` is ``0``, the sum is ``0`` (i.e., the empty sum).

    For floating-point operands,
    -   If ``x_i`` is ``NaN``, the sum is ``NaN`` (i.e., ``NaN`` values propagate).

    For both real-valued and complex floating-point operands, special cases must
    be handled as if the operation is implemented by successive application of
    :func:`aikit.add`:

    Parameters
    ----------
    x
        Input array. Should have a numeric data type.
    axis
        Axis or axes along which sums must be computed. By default, the sum must be
        computed over the entire array. If a tuple of integers, sums must be computed
        over multiple axes. Default: ``None``.
    dtype
        Data type of the returned array. If ``None``,
            If the default data type corresponding to the data type "kind" (integer or
            floating-point) of ``x`` has a smaller range of values than the data type of
            ``x`` (e.g., ``x`` has data type ``int64`` and the default data type is
            ``int32``, or ``x`` has data type ``uint64`` and the default data type is
            ``int64``), the returned array must have the same data type as ``x``.
            If ``x`` has a floating-point data type, the returned array must have the
            default floating-point data type.
            If ``x`` has a signed integer data type (e.g., ``int16``), the returned
            array must have the default integer data type.
            If ``x`` has an unsigned integer data type (e.g., ``uint16``), the returned
            array must have an unsigned integer data type having the same number of bits
            as the default integer data type (e.g., if the default integer data type is
            ``int32``, the returned array must have a ``uint32`` data type).

        If the data type (either specified or resolved) differs from the data type of
        ``x``, the input array should be cast to the specified data type before
        computing the sum. Default: ``None``.

        .. note::
            keyword argument is intended to help prevent data type overflows.

    keepdims
        If ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes
        (dimensions) must not be included in the result. Default: ``False``.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        If the sum was computed over the entire array, a zero-dimensional array
        containing the sum; otherwise, an array containing the sums. The returned array
        must have a data type as described by the ``dtype`` parameter above.


    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.sum.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([0.41, 0.89])
    >>> y = aikit.sum(x)
    >>> print(y)
    aikit.array(1.3)

    >>> x = aikit.array([0.5, 0.7, 2.4])
    >>> y = aikit.array(0.0)
    >>> aikit.sum(x, out=y)
    >>> print(y)
    aikit.array(3.6)

    >>> x = aikit.array([[0, 1, 2], [4, 6, 10]])
    >>> y = aikit.sum(x, axis = 1, keepdims = False)
    >>> print(y)
    aikit.array([3, 20])

    >>> x = aikit.array([[0, 1, 2], [4, 6, 10]])
    >>> y = aikit.array([0,0,0])
    >>> aikit.sum(x, axis = 0, keepdims = False, out = y)
    >>> print(y)
    aikit.array([4, 7, 12])

    With :class:`aikit.NativeArray` input:

    >>> x = aikit.native_array([0.1, 0.2, 0.3, 0.3, 0.9, 0.10])
    >>> y = aikit.sum(x)
    >>> print(y)
    aikit.array(1.9)

    >>> x = aikit.native_array([1.0, 2.0, 2.0, 3.0])
    >>> y = aikit.array(0.0)
    >>> aikit.sum(x, out=y)
    >>> print(y)
    aikit.array(8.)

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([0., 1., 2.]), b=aikit.array([3., 4., 5.]))
    >>> y = aikit.sum(x)
    >>> print(y)
    {
        a: aikit.array(3.),
        b: aikit.array(12.)
    }
    """
    return current_backend(x).sum(x, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def var(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Calculate the variance of the input array x.

    **Special Cases**

    Let N equal the number of elements over which to compute the variance.

    If N - correction is less than or equal to 0, the variance is NaN.

    If x_i is NaN, the variance is NaN (i.e., NaN values propagate).

    Parameters
    ----------
    x
        input array. Should have a floating-point data type.
    axis
        axis or axes along which variances must be computed. By default, the variance
        must be computed over the entire array. If a tuple of integers, variances must
        be computed over multiple axes. Default: ``None``.
    correction
        degrees of freedom adjustment. Setting this parameter to a value other than 0
        has the effect of adjusting the divisor during the calculation of the variance
        according to N-c where N corresponds to the total number of elements over which
        the variance is computed and c corresponds to the provided degrees of freedom
        adjustment. When computing the variance of a population, setting this parameter
        to 0 is the standard choice (i.e., the provided array contains data constituting
        an entire population). When computing the unbiased sample variance, setting this
        parameter to 1 is the standard choice (i.e., the provided array contains data
        sampled from a larger population; this is commonly referred to as Bessel's
        correction). Default: ``0``.
    keepdims
        if True, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see Broadcasting). Otherwise, if False, the reduced axes
        (dimensions) must not be included in the result. Default: ``False``.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        if the variance was computed over the entire array, a zero-dimensional array
        containing the variance; otherwise, a non-zero-dimensional array containing the
        variances. The returned array must have the same data type as x.


    This method conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.var.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([0.1, 0.2, 0.3, 0.3, 0.9, 0.10])
    >>> y = aikit.var(x)
    >>> print(y)
    aikit.array(0.07472222)

    >>> x = aikit.array([0.1, 0.2, 0.3, 0.3, 0.9, 0.10])
    >>> y = aikit.array(0.0)
    >>> aikit.var(x, out=y)
    >>> print(y)
    aikit.array(0.07472222)

    >>> x = aikit.array([[0.1, 0.2, 0.3], [0.3, 0.9, 0.10]])
    >>> print(aikit.var(x, axis=1, keepdims=True))
    aikit.array([[0.00666667],
       [0.11555555]])

    >>> x = aikit.array([[0.1, 0.2, 0.3], [0.3, 0.9, 0.10]])
    >>> y = aikit.var(x, correction=1)
    >>> print(y)
    aikit.array(0.08966666)

    With :class:`aikit.Container` input:
    >>> x = aikit.Container(a=aikit.array([0.1, 0.2, 0.9]),
    ...                   b=aikit.array([0.7, 0.1, 0.9]))
    >>> y = aikit.var(x)
    >>> print(y)
    {
        a: aikit.array(0.12666667),
        b: aikit.array(0.11555555)
    }
    """
    return current_backend(x).var(
        x, axis=axis, correction=correction, keepdims=keepdims, out=out
    )


# Extra #
# ------#


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def cumsum(
    x: Union[aikit.Array, aikit.NativeArray],
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    *,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    x
        Input array.
    axis
        Axis along which the cumulative sum is computed. Default is ``0``.
    exclusive
        Whether to perform cumsum exclusively. Default is ``False``.
    reverse
        Whether to perform the cumsum from last to first element in the selected
        axis. Default is ``False`` (from first to last element)
    dtype
        Data type of the returned array. Default is ``None``.
        If None, if the default data type corresponding to the data type “kind”
        (integer or floating-point) of x has a smaller range of values than the
        data type of x (e.g., x has data type int64 and the default data type
        is int32, or x has data type uint64 and the default data type is int64),
        the returned array must have the same data type as x.
        If x has a floating-point data type, the returned array must have the
        default floating-point data type.
        If x has a signed integer data type (e.g., int16), the returned array
        must have the default integer data type.
        If x has an unsigned integer data type (e.g., uint16), the returned
        array must have an unsigned integer data type having the same number of
        bits as the default integer data type (e.g., if the default integer data
        type is int32, the returned array must have a uint32 data type).
        If the data type (either specified or resolved) differs from the data type
        of x, the input array should be cast to the specified data type before
        computing the product.
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Array which holds the result of applying cumsum at each
        original array elements along the specified axis.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([1, 5, 2, 0])
    >>> y = aikit.cumsum(x, exclusive= True, reverse=False)
    >>> print(y)
    aikit.array([0, 1, 6, 8])

    >>> x = aikit.array([[6, 4, 2],
    ...                [1, 3, 0]])
    >>> y = aikit.zeros((2,3))
    >>> aikit.cumsum(x, axis=0, exclusive=False, reverse=True, out=y)
    >>> print(y)
    aikit.array([[7, 7, 2],
               [1, 3, 0]])

    >>> x = aikit.array([[1, 5, 2],
    ...                [4, 3, 0]])
    >>> y = aikit.cumsum(x, axis=0, exclusive=True, reverse=True)
    >>> print(y)
    aikit.array([[4, 3, 0],
               [0, 0, 0]])

    >>> x = aikit.array([[2, 4, 5],
    ...                [3, 6, 5],
    ...                [1, 3, 10]])
    >>> aikit.cumsum(x,axis=1,reverse=True, dtype='int64', out=x)
    >>> print(x)
    aikit.array([[11,  9,  5],
               [14, 11,  5],
               [14, 13, 10]])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([[1, 3, 5]]),
    ...                   b=aikit.array([[3, 5, 7]]))
    >>> y = aikit.cumsum(x, axis= 0)
    >>> print(y)
    {
        a: aikit.array([[1, 3, 5]]),
        b: aikit.array([[3, 5, 7]])
    }

    >>> x = aikit.Container(a=aikit.array([[1, 3, 4]]),
    ...                   b=aikit.array([[3, 5, 8],
    ...                                [5, 6, 5]]),
    ...                   c=aikit.array([[2, 4, 1],
    ...                                [3, 6, 9],
    ...                                [0, 2, 3]]))
    >>> y = aikit.Container(a = aikit.zeros((1, 3)),
    ...                   b = aikit.zeros((2, 3)),
    ...                   c = aikit.zeros((3,3)))
    >>> aikit.cumsum(x,axis=1,reverse=True, out=y)
    >>> print(y)
    {
        a: aikit.array([[8, 7, 4]]),
        b: aikit.array([[16, 13, 8],
                      [16, 11, 5]]),
        c: aikit.array([[7, 5, 1],
                      [18, 15, 9],
                      [5, 5, 3]])
    }

    >>> x = aikit.Container(a=aikit.array([[0],
    ...                                [5]]),
    ...                   b=aikit.array([[6, 8, 7],
    ...                                [4, 2, 3]]),
    ...                   c=aikit.array([[1, 2],
    ...                                [3, 4],
    ...                                [6, 4]]))
    >>> aikit.cumsum(x,axis=0,out=x)
    >>> print(x)
    {
        a: aikit.array([[0],
                      [5]]),
        b: aikit.array([[6, 8, 7],
                      [10, 10, 10]]),
        c: aikit.array([[1, 2],
                      [4, 6],
                      [10, 10]])
    }
    """
    return current_backend(x).cumsum(x, axis, exclusive, reverse, dtype=dtype, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def cumprod(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Return the cumulative product of the elements along a given axis.

    Parameters
    ----------
    x
        Input array.
    axis
        int , axis along which the cumulative product is computed. By default 0.
    exclusive
        optional bool, Whether to perform the cumprod exclusively. Defaults is False.
    reverse
        Whether to perform the cumprod from last to first element in the selected
        axis. Default is ``False`` (from first to last element)
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Input array with cumulatively multiplied elements along axis.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([2, 3, 4])
    >>> y = aikit.cumprod(x)
    >>> print(y)
    aikit.array([2, 6, 24])

    >>> x = aikit.array([2, 3, 4])
    >>> y = aikit.cumprod(x, exclusive=True)
    >>> print(y)
    aikit.array([1, 2, 6])

    >>> x = aikit.array([[2, 3],
                       [5, 7],
                       [11, 13]])
    >>> y = aikit.zeros((3, 2))
    >>> aikit.cumprod(x, axis=1, exclusive=True, out=y)
    >>> print(y)
    aikit.array([[ 1.,  2.],
               [ 1.,  5.],
               [ 1., 11.]])

    >>> x = aikit.array([[2, 3],[5, 7],[11, 13]])
    >>> aikit.cumprod(x, axis=0, exclusive=True, out=x)
    >>> print(x)
    aikit.array([[1,  1],
               [2,  3],
               [10, 21]])

    >>> x = aikit.array([[2, 3],[5, 7],[11, 13]])
    >>> y = aikit.zeros((3, 2))
    >>> x.cumprod(axis=0, exclusive=True, out=y)
    >>> print(x)
    aikit.array([[1.,  1.],
                [2.,  3.],
                [10., 21.]])

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([2, 3, 4]), b=aikit.array([3, 4, 5]))
    >>> y = aikit.cumprod(x)
    >>> print(y)
    {
        a: aikit.array([2, 6, 24]),
        b: aikit.array([3, 12, 60])
    }

    >>> x = aikit.Container(a=aikit.array([2, 3, 4]), b=aikit.array([3, 4, 5]))
    >>> y = aikit.cumprod(x, exclusive=True)
    >>> print(y)
    {
        a: aikit.array([1, 2, 6]),
        b: aikit.array([1, 3, 12])
    }

    >>> x = aikit.Container(a=aikit.array([[2, 3],
                                       [5, 7],
                                       [11, 13]]),
                          b=aikit.array([[3, 4],
                                       [4, 5],
                                       [5, 6]]))
    >>> y = aikit.Container(a = aikit.zeros((3, 2)), b = aikit.zeros((3, 2)))
    >>> aikit.cumprod(x, axis=1, exclusive=True, out=y)
    >>> print(y)
    {
        a: aikit.array([[1, 2],
                      [1, 5],
                      [1, 11]]),
        b: aikit.array([[1, 3],
                      [1, 4],
                      [1, 5]])
    }

    >>> x = aikit.Container(a=aikit.array([[2, 3],
                                        [5, 7],
                                        [11, 13]]),
                            b=aikit.array([[3, 4],
                                        [4, 5],
                                        [5, 6]]))
    >>> x.cumprod(axis=0, exclusive=True, out=x)
    >>> print(x)
    {
        a: aikit.array([[1, 1],
                      [2, 3],
                      [10, 21]]),
        b: aikit.array([[1, 1],
                      [3, 4],
                      [15, 42]])
    }
    """
    return current_backend(x).cumprod(
        x, axis=axis, exclusive=exclusive, reverse=reverse, dtype=dtype, out=out
    )


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device
def einsum(
    equation: str,
    *operands: Union[aikit.Array, aikit.NativeArray],
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Sum the product of the elements of the input operands along dimensions
    specified using a notation based on the Einstein summation convention.

    Parameters
    ----------
    equation
        A str describing the contraction, in the same format as numpy.einsum.
    operands
        seq of arrays, the inputs to contract (each one an aikit.Array), whose shapes
        should be consistent with equation.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        The array with sums computed.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> y = aikit.einsum('ii', x)
    >>> print(y)
    aikit.array(12)

    >>> x = aikit.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> z = aikit.einsum('ij -> j', x)
    >>> print(z)
    aikit.array([ 9, 12, 15])

    >>> A = aikit.array([0, 1, 2])
    >>> B = aikit.array([[ 0,  1,  2,  3],
    ...                [ 4,  5,  6,  7],
    ...                [ 8,  9, 10, 11]])
    >>> C = aikit.einsum('i,ij->i', A, B)
    >>> print(C)
    aikit.array([ 0, 22, 76])

    >>> A = aikit.array([[1, 1, 1],
    ...                [2, 2, 2],
    ...                [5, 5, 5]])
    >>> B = aikit.array([[0, 1, 0],
    ...                [1, 1, 0],
    ...                [1, 1, 1]])
    >>> C = aikit.einsum('ij,jk->ik', A, B)
    >>> print(C)
    aikit.array([[ 2,  3,  1],
           [ 4,  6,  2],
           [10, 15,  5]])

    >>> A = aikit.arange(10)
    >>> B = aikit.arange(5, 15)
    >>> C = aikit.einsum('i->', A)
    >>> print(C)
    aikit.array(45)

    >>> A = aikit.arange(10)
    >>> B = aikit.arange(5, 15)
    >>> C = aikit.einsum('i,i->i', A, B)
    >>> print(C)
    aikit.array([  0,   6,  14,  24,  36,  50,  66,  84, 104, 126])

    >>> A = aikit.arange(10)
    >>> B = aikit.arange(5, 15)
    >>> C = aikit.einsum('i,i->', A, B) # or just use 'i,i'
    >>> print(C)
    aikit.array(510)

    >>> A = aikit.arange(10)
    >>> B = aikit.arange(5, 15)
    >>> C = aikit.einsum('i,j->ij', A, B)
    >>> print(C)
    aikit.array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
           [  5,   6,   7,   8,   9,  10,  11,  12,  13,  14],
           [ 10,  12,  14,  16,  18,  20,  22,  24,  26,  28],
           [ 15,  18,  21,  24,  27,  30,  33,  36,  39,  42],
           [ 20,  24,  28,  32,  36,  40,  44,  48,  52,  56],
           [ 25,  30,  35,  40,  45,  50,  55,  60,  65,  70],
           [ 30,  36,  42,  48,  54,  60,  66,  72,  78,  84],
           [ 35,  42,  49,  56,  63,  70,  77,  84,  91,  98],
           [ 40,  48,  56,  64,  72,  80,  88,  96, 104, 112],
           [ 45,  54,  63,  72,  81,  90,  99, 108, 117, 126]])

    With a mix of :class:`aikit.Array` and :class:`aikit.Container` inputs:

    >>> x = aikit.array([0, 1, 2])
    >>> y = aikit.Container(a=aikit.array([[ 0,  1,  2,  3],
    ...                                [ 4,  5,  6,  7],
    ...                                [ 8,  9, 10, 11]]),
    ...                   b=aikit.array([[ 0,  1,  2],
    ...                                [ 4,  5,  6],
    ...                                [ 8,  9, 10]]))
    >>> z = aikit.einsum('i,ij->i', x, y)
    >>> print(z)
    {
        a: aikit.array([0, 22, 76]),
        b: aikit.array([0, 15, 54])
    }

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([[0, 1, 0],[1, 1, 0],[1, 1, 1]]),
    ...                   b=aikit.array([[0, 1, 2],[4, 5, 6],[8, 9, 10]]))
    >>> y = aikit.einsum('ii', x)
    >>> print(y)
    {
        a: aikit.array(2),
        b: aikit.array(15)
    }
    """
    return current_backend(operands[0]).einsum(equation, *operands, out=out)
