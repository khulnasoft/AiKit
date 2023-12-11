# global
from typing import Union, Optional, Sequence

# local
import aikit
from aikit.func_wrapper import (
    handle_array_function,
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
    handle_backend_invalid,
)
from aikit.utils.exceptions import handle_exceptions


# Array API Standard #
# -------------------#


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def all(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Test whether all input array elements evaluate to ``True`` along a
    specified axis.

    .. note::
       Positive infinity, negative infinity, and NaN must evaluate to ``True``.

    .. note::
       If ``x`` is an empty array or the size of the axis (dimension) along which to
       evaluate elements is zero, the test result must be ``True``.

    Parameters
    ----------
    x
        input array.
    axis
        axis or axes along which to perform a logical AND reduction. By default, a
        logical AND reduction must be performed over the entire array. If a tuple of
        integers, logical AND reductions must be performed over multiple axes. A valid
        ``axis`` must be an integer on the interval ``[-N, N)``, where ``N`` is the rank
        (number of dimensions) of ``x``. If an ``axis`` is specified as a negative
        integer, the function must determine the axis along which to perform a reduction
        by counting backward from the last dimension (where ``-1`` refers to the last
        dimension). If provided an invalid ``axis``, the function must raise an
        exception. Default ``None``.
    keepdims
        If ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes
        (dimensions) must not be included in the result. Default: ``False``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        if a logical AND reduction was performed over the entire array, the returned
        array must be a zero-dimensional array containing the test result; otherwise,
        the returned array must be a non-zero-dimensional array containing the test
        results. The returned array must have a data type of ``bool``.


    This method conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.all.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicit
    y,but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([1, 2, 3])
    >>> y = aikit.all(x)
    >>> print(y)
    aikit.array(True)

    >>> x = aikit.array([[0],[1]])
    >>> y = aikit.zeros((1,1), dtype='bool')
    >>> a = aikit.all(x, axis=0, out = y, keepdims=True)
    >>> print(a)
    aikit.array([[False]])

    >>> x = aikit.array(False)
    >>> y = aikit.all(aikit.array([[0, 4],[1, 5]]), axis=(0,1), out=x, keepdims=False)
    >>> print(y)
    aikit.array(False)

    >>> x = aikit.array(False)
    >>> y = aikit.all(aikit.array([[[0], [1]], [[1], [1]]]), axis=(0,1,2), out=x,
    ...             keepdims=False)
    >>> print(y)
    aikit.array(False)

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([0, 1, 2]), b=aikit.array([3, 4, 5]))
    >>> y = aikit.all(x)
    >>> print(y)
    {
        a: aikit.array(False),
        b: aikit.array(True)
    }

    >>> x = aikit.Container(a=aikit.native_array([0, 1, 2]),b=aikit.array([3, 4, 5]))
    >>> y = aikit.all(x)
    >>> print(y)
    {
        a: aikit.array(False),
        b: aikit.array(True)
    }
    """
    return aikit.current_backend(x).all(x, axis=axis, keepdims=keepdims, out=out)


@handle_exceptions
@handle_backend_invalid
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
def any(
    x: Union[aikit.Array, aikit.NativeArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[aikit.Array] = None,
) -> aikit.Array:
    """Test whether any input array element evaluates to ``True`` along a
    specified axis.

    .. note::
       Positive infinity, negative infinity, and NaN must evaluate to ``True``.

    .. note::
       If ``x`` is an empty array or the size of the axis (dimension) along which to
       evaluate elements is zero, the test result must be ``False``.

    Parameters
    ----------
    x
        input array.
    axis
        axis or axes along which to perform a logical OR reduction. By default, a
        logical OR reduction must be performed over the entire array. If a tuple of
        integers, logical OR reductions must be performed over multiple axes. A valid
        ``axis`` must be an integer on the interval ``[-N, N)``, where ``N`` is the rank
        (number of dimensions) of ``x``. If an ``axis`` is specified as a negative
        integer, the function must determine the axis along which to perform a reduction
        by counting backward from the last dimension (where ``-1`` refers to the last
        dimension). If provided an invalid ``axis``, the function must raise an
        exception. Default: ``None``.
    keepdims
        If ``True``, the reduced axes (dimensions) must be included in the result as
        singleton dimensions, and, accordingly, the result must be compatible with the
        input array (see :ref:`broadcasting`). Otherwise, if ``False``, the reduced axes
        (dimensions) must not be included in the result. Default: ``False``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        if a logical OR reduction was performed over the entire array, the returned
        array must be a zero-dimensional array containing the test result; otherwise,
        the returned array must be a non-zero-dimensional array containing the test
        results. The returned array must have a data type of ``bool``.


    This method conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/
    API_specification/generated/array_api.any.html>`_
    in the standard.

    Both the description and the type hints above assumes an array input for simplicit
    y,but this function is *nestable*, and therefore also accepts :class:`aikit.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`aikit.Array` input:

    >>> x = aikit.array([2, 3, 4])
    >>> y = aikit.any(x)
    >>> print(y)
    aikit.array(True)

    >>> x = aikit.array([[0],[1]])
    >>> y = aikit.zeros((1,1), dtype='bool')
    >>> a = aikit.any(x, axis=0, out = y, keepdims=True)
    >>> print(a)
    aikit.array([[True]])

    >>> x=aikit.array(False)
    >>> y=aikit.any(aikit.array([[0, 3],[1, 4]]), axis=(0,1), out=x, keepdims=False)
    >>> print(y)
    aikit.array(True)

    >>> x=aikit.array(False)
    >>> y=aikit.any(aikit.array([[[0],[1]],[[1],[1]]]),axis=(0,1,2), out=x, keepdims=False)
    >>> print(y)
    aikit.array(True)

    With :class:`aikit.Container` input:

    >>> x = aikit.Container(a=aikit.array([0, 1, 2]), b=aikit.array([3, 4, 5]))
    >>> y = aikit.any(x)
    >>> print(y)
    {
        a: aikit.array(True),
        b: aikit.array(True)
    }
    """
    return aikit.current_backend(x).any(x, axis=axis, keepdims=keepdims, out=out)


# Extra #
# ----- #


def save(item, filepath, format=None):
    if isinstance(item, aikit.Container):
        if format is not None:
            item.cont_save(filepath, format=format)
        else:
            item.cont_save(filepath)
    elif isinstance(item, aikit.Module):
        item.save(filepath)
    else:
        raise aikit.utils.exceptions.AikitException("Unsupported item type for saving.")


@staticmethod
def load(filepath, format=None, type="module"):
    if type == "module":
        return aikit.Module.load(filepath)
    elif type == "container":
        if format is not None:
            return aikit.Container.cont_load(filepath, format=format)
        else:
            return aikit.Container.cont_load(filepath)
    else:
        raise aikit.utils.exceptions.AikitException("Unsupported item type for loading.")
