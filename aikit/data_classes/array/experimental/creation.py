# global
import abc
from typing import Optional, Union

# local
import aikit


class _ArrayWithCreationExperimental(abc.ABC):
    def eye_like(
        self: aikit.Array,
        /,
        *,
        k: int = 0,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.eye_like. This method
        simply wraps the function, and so the docstring for aikit.eye_like also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array from which to derive the output array shape.
        k
            index of the diagonal. A positive value refers to an upper diagonal,
            a negative value to a lower diagonal, and 0 to the main diagonal.
            Default: ``0``.
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
            an array having the same shape as ``self`` and filled with ``ones``
            in diagonal ``k`` and ``zeros`` elsewhere.

        Examples
        --------
        >>> x = aikit.array([[2, 3, 8],[1, 2, 1]])
        >>> y = x.eye_like()
        >>> print(y)
        aikit.array([[1., 0., 0.],
                    0., 1., 0.]])
        """
        return aikit.eye_like(self._data, k=k, dtype=dtype, device=device, out=out)

    def unsorted_segment_min(
        self: aikit.Array,
        segment_ids: aikit.Array,
        num_segments: Union[int, aikit.Array],
    ) -> aikit.Array:
        r"""aikit.Array instance method variant of aikit.unsorted_segment_min. This
        method simply wraps the function, and so the docstring for
        aikit.unsorted_segment_min also applies to this method with minimal
        changes.

        Note
        ----
        If the given segment ID `i` is negative, then the corresponding
        value is dropped, and will not be included in the result.

        Parameters
        ----------
        self
            The array from which to gather values.

        segment_ids
            Must be in the same size with the first dimension of `self`. Has to be
            of integer data type. The index-th element of `segment_ids` array is
            the segment identifier for the index-th element of `self`.

        num_segments
            An integer or array representing the total number of distinct segment IDs.

        Returns
        -------
        ret
            The output array, representing the result of a segmented min operation.
            For each segment, it computes the min value in `self` where `segment_ids`
            equals to segment ID.
        """
        return aikit.unsorted_segment_min(self._data, segment_ids, num_segments)

    def unsorted_segment_sum(
        self: aikit.Array,
        segment_ids: aikit.Array,
        num_segments: Union[int, aikit.Array],
    ) -> aikit.Array:
        r"""aikit.Array instance method variant of aikit.unsorted_segment_sum. This
        method simply wraps the function, and so the docstring for
        aikit.unsorted_segment_sum also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            The array from which to gather values.

        segment_ids
            Must be in the same size with the first dimension of `self`. Has to be
            of integer data type. The index-th element of `segment_ids` array is
            the segment identifier for the index-th element of `self`.

        num_segments
            An integer or array representing the total number of distinct segment IDs.

        Returns
        -------
        ret
            The output array, representing the result of a segmented sum operation.
            For each segment, it computes the sum of values in `self` where
            `segment_ids` equals to segment ID.
        """
        return aikit.unsorted_segment_sum(self._data, segment_ids, num_segments)

    def blackman_window(
        self: aikit.Array,
        /,
        *,
        periodic: bool = True,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.blackman_window. This
        method simply wraps the function, and so the docstring for
        aikit.blackman_window also applies to this method with minimal changes.

        Parameters
        ----------
        self
            int.
        periodic
            If True, returns a window to be used as periodic function.
            If False, return a symmetric window.
            Default: ``True``.
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
            The array containing the window.

        Examples
        --------
        >>> aikit.blackman_window(4, periodic = True)
        aikit.array([-1.38777878e-17,  3.40000000e-01,  1.00000000e+00,  3.40000000e-01])
        >>> aikit.blackman_window(7, periodic = False)
        aikit.array([-1.38777878e-17,  1.30000000e-01,  6.30000000e-01,  1.00000000e+00,
        6.30000000e-01,  1.30000000e-01, -1.38777878e-17])
        """
        return aikit.blackman_window(self._data, periodic=periodic, dtype=dtype, out=out)

    def trilu(
        self: aikit.Array,
        /,
        *,
        k: int = 0,
        upper: bool = True,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.trilu. This method simply
        wraps the function, and so the docstring for aikit.trilu also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array having shape (..., M, N) and whose innermost two dimensions form
            MxN matrices.    *,
        k
            diagonal below or above which to zero elements. If k = 0, the diagonal is
            the main diagonal. If k < 0, the diagonal is below the main diagonal. If
            k > 0, the diagonal is above the main diagonal. Default: ``0``.
        upper
            indicates whether upper or lower part of matrix is retained.
            Default: ``True``.
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
        return aikit.trilu(self._data, k=k, upper=upper, out=out)

    @staticmethod
    def mel_weight_matrix(
        num_mel_bins: Union[int, aikit.Array],
        dft_length: Union[int, aikit.Array],
        sample_rate: Union[int, aikit.Array],
        lower_edge_hertz: Optional[Union[float, aikit.Array]] = 0.0,
        upper_edge_hertz: Optional[Union[float, aikit.Array]] = 3000.0,
    ):
        """Generate a MelWeightMatrix that can be used to re-weight a Tensor
        containing a linearly sampled frequency spectra (from DFT or STFT) into
        num_mel_bins frequency information based on the [lower_edge_hertz,
        upper_edge_hertz]

        range on the mel scale. This function defines the mel scale
        in terms of a frequency in hertz according to the following
        formula: mel(f) = 2595 * log10(1 + f/700)

        Parameters
        ----------
        num_mel_bins
            The number of bands in the mel spectrum.
        dft_length
            The size of the original DFT obtained from (n_fft / 2 + 1).
        sample_rate
            Samples per second of the input signal.
        lower_edge_hertz
            Lower bound on the frequencies to be included in the mel spectrum.
        upper_edge_hertz
            The desired top edge of the highest frequency band.

        Returns
        -------
        ret
            MelWeightMatrix of shape:  [frames, num_mel_bins].

        Examples
        --------
        >>> x = aikit.array([[1, 2, 3],
        >>>                [1, 1, 1],
        >>>                [5,6,7  ]])
        >>> x.mel_weight_matrix(3, 3, 8000)
        aikit.array([[0.        ,0.        , 0.],
                  [0.        ,0. , 0.75694758],
                  [0.        ,0. , 0.       ]])
        """
        return aikit.mel_weight_matrix(
            num_mel_bins,
            dft_length,
            sample_rate,
            lower_edge_hertz,
            upper_edge_hertz,
        )

    def unsorted_segment_mean(
        self: aikit.Array,
        segment_ids: aikit.Array,
        num_segments: Union[int, aikit.Array],
    ) -> aikit.Array:
        """Compute the mean of values in the array 'self' based on segment
        identifiers.

        Parameters
        ----------
        self : aikit.Array
            The array from which to gather values.
        segment_ids : aikit.Array
            Must be in the same size with the first dimension of `self`. Has to be
            of integer data type. The index-th element of `segment_ids` array is
            the segment identifier for the index-th element of `self`.
        num_segments : Union[int, aikit.Array]
            An integer or array representing the total number of distinct segment IDs.

        Returns
        -------
        ret : aikit.Array
            The output array, representing the result of a segmented mean operation.
            For each segment, it computes the mean of values in `self` where
            `segment_ids` equals to segment ID.

        Examples
        --------
        >>> data = aikit.array([1.0, 2.0, 3.0, 4.0])
        >>> segment_ids = aikit.array([0, 0, 0, 0])
        >>> num_segments = 1
        >>> result = aikit.unsorted_segment_mean(data, segment_ids, num_segments)
        >>> result
        aikit.array([2.5])

        >>> data = aikit.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> segment_ids = aikit.array([0, 0, 1, 1, 2, 2])
        >>> num_segments = 3
        >>> result = aikit.unsorted_segment_mean(data, segment_ids, num_segments)
        >>> result
        aikit.array([[1.5, 3.5, 5.5],[1.5, 3.5, 5.5],[1.5, 3.5, 5.5]])
        """
        return aikit.unsorted_segment_mean(self._data, segment_ids, num_segments)


def polyval(
    coeffs=aikit.Array,
    x=Union[aikit.Array, aikit.NativeArray, int, float],
    /,
    *,
    dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
    device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
) -> aikit.Array:
    """aikit.Array instance method of polyval. This method simply wraps the
    function, and so the docstring for aikit.polyval also applies to this method
    with minimal changes.

    Evaluate and return a polynomial at specific given values.

    Parameters
    ----------
    coeffs
        Input array containing polynomial coefficients (including zero)
        from highest degree to constant term.
    x
        The value of the indeterminate variable at which to evaluate the polynomial.

    Returns
    -------
    ret
        Simplified result of substituting x in the coefficients - final value of
        polynomial.

    Examples
    --------
    >>> x = aikit.array([[0, 0, 0])
    >>> x.polyval([3, 0, 1], 5)
    aikit.array(76)
    """
    return aikit.polyval(
        coeffs,
        x,
    )
