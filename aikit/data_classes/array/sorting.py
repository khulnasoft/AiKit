# global
import abc
from typing import Optional, Union, Literal, List

# local

import aikit


class _ArrayWithSorting(abc.ABC):
    def argsort(
        self: aikit.Array,
        /,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.argsort. This method simply
        wraps the function, and so the docstring for aikit.argsort also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        axis
            axis along which to sort. If set to ``-1``, the function
            must sort along the last axis. Default: ``-1``.
        descending
            sort order. If ``True``, the returned indices sort ``x`` in descending order
            (by value). If ``False``, the returned indices sort ``x`` in ascending order
            (by value). Default: ``False``.
        stable
            sort stability. If ``True``, the returned indices
            must maintain the relative order of ``x`` values
            which compare as equal. If ``False``, the returned
            indices may or may not maintain the relative order
            of ``x`` values which compare as equal (i.e., the
            relative order of ``x`` values which compare as
            equal is implementation-dependent). Default: ``True``.
        out
            optional output array, for writing the result to. It must have the same
            shape as input.

        Returns
        -------
        ret
            an array of indices. The returned array must have the same shape as ``x``.
            The returned array must have the default array index data type.

        Examples
        --------
        >>> x = aikit.array([1, 5, 2])
        >>> y = x.argsort(axis=-1, descending=True, stable=False)
        >>> print(y)
        aikit.array([1, 2, 0])

        >>> x = aikit.array([9.6, 2.7, 5.2])
        >>> y = x.argsort(axis=-1, descending=True, stable=False)
        >>> print(y)
        aikit.array([0, 2, 1])
        """
        return aikit.argsort(
            self._data, axis=axis, descending=descending, stable=stable, out=out
        )

    def sort(
        self: aikit.Array,
        /,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.sort. This method simply
        wraps the function, and so the docstring for aikit.sort also applies to
        this method with minimal changes.

        Examples
        --------
        >>> x = aikit.array([7, 8, 6])
        >>> y = x.sort(axis=-1, descending=True, stable=False)
        >>> print(y)
        aikit.array([8, 7, 6])

        >>> x = aikit.array([8.5, 8.2, 7.6])
        >>> y = x.sort(axis=-1, descending=True, stable=False)
        >>> print(y)
        aikit.array([8.5, 8.2, 7.6])
        """
        return aikit.sort(
            self._data, axis=axis, descending=descending, stable=stable, out=out
        )

    def msort(
        self: aikit.Array,
        /,
        *,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.msort. This method simply
        wraps the function, and so the docstring for aikit.msort also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            sorted array of the same type and shape as a

        Examples
        --------
        >>> a = aikit.asarray([[8, 9, 6],[6, 2, 6]])
        >>> a.msort()
        aikit.array(
            [[6, 2, 6],
            [8, 9, 6]]
            )
        """
        return aikit.msort(self._data, out=out)

    def searchsorted(
        self: aikit.Array,
        v: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        side: Literal["left", "right"] = "left",
        sorter: Optional[Union[aikit.Array, aikit.NativeArray, List[int]]] = None,
        ret_dtype: Union[aikit.Dtype, aikit.NativeDtype] = aikit.int64,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.searchsorted.

        This method simply wraps the function, and so the docstring for
        aikit.searchsorted also applies to this method with minimal
        changes.
        """
        return aikit.searchsorted(
            self.data, v, side=side, sorter=sorter, ret_dtype=ret_dtype, out=out
        )
