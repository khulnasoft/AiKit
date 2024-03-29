# global
from typing import Optional, Union, List, Dict

# local
import aikit
from aikit.data_classes.container.base import ContainerBase


class _ContainerWithCreationExperimental(ContainerBase):
    @staticmethod
    def static_hann_window(
        window_length: Union[int, aikit.Container],
        periodic: Union[bool, aikit.Container] = True,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.hann_window. This method
        simply wraps the function, and so the docstring for aikit.hann_window
        also applies to this method with minimal changes.

        Parameters
        ----------
        window_length
            container including multiple window sizes.
        periodic
            If True, returns a window to be used as periodic function.
            If False, return a symmetric window.
        dtype
            The data type to produce. Must be a floating point type.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that contains the Hann windows.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x = aikit.Container(a=3, b=5)
        >>> aikit.Container.static_hann(x)
        {
            a: aikit.array([0.0000, 0.7500, 0.7500])
            b: aikit.array([0.0000, 0.3455, 0.9045, 0.9045, 0.3455])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "hann_window",
            window_length,
            periodic,
            dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def hann_window(
        self: aikit.Container,
        periodic: Union[bool, aikit.Container] = True,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.hann_window. This
        method simply wraps the function, and so the docstring for
        aikit.hann_window also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container with window sizes.
        periodic
            If True, returns a window to be used as periodic function.
            If False, return a symmetric window.
        dtype
            The data type to produce. Must be a floating point type.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container containing the Hann windows.

        Examples
        --------
        With one :class:`aikit.Container` input:

        >>> x = aikit.Container(a=3, b=5)
        >>> aikit.hann_window(x)
        {
            a: aikit.array([0.0000, 0.7500, 0.7500])
            b: aikit.array([0.0000, 0.3455, 0.9045, 0.9045, 0.3455])
        }
        """
        return self.static_hann_window(self, periodic, dtype, out=out)

    @staticmethod
    def static_kaiser_window(
        window_length: Union[int, aikit.Container],
        periodic: Union[bool, aikit.Container] = True,
        beta: Union[float, aikit.Container] = 12.0,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        dtype: Optional[Union[aikit.Array, aikit.NativeArray, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.kaiser_window. This
        method simply wraps the function, and so the docstring for
        aikit.kaiser_window also applies to this method with minimal changes.

        Parameters
        ----------
        window_length
            input container including window lengths.
        periodic
            If True, returns a periodic window suitable for use in spectral analysis.
            If False, returns a symmetric window suitable for use in filter design.
        beta
            a float used as shape parameter for the window.
        dtype
            data type of the returned array.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the Kaiser windows.

        Examples
        --------
        >>> x = aikit.Container(a=3, b=5)
        >>> aikit.Container.static_kaiser_window(x, True, 5)
        {
            a: aikit.array([0.2049, 0.8712, 0.8712]),
            a: aikit.array([0.0367, 0.7753, 0.7753]),
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "kaiser_window",
            window_length,
            periodic,
            beta,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    def kaiser_window(
        self: aikit.Container,
        periodic: Union[bool, aikit.Container] = True,
        beta: Union[float, aikit.Container] = 12.0,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        dtype: Optional[Union[aikit.Array, aikit.NativeArray, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.kaiser_window. This
        method simply wraps the function, and so the docstring for
        aikit.kaiser_window also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container including window lengths.
        periodic
            If True, returns a periodic window suitable for use in spectral analysis.
            If False, returns a symmetric window suitable for use in filter design.
        beta
            a float used as shape parameter for the window.
        dtype
            data type of the returned array.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the Kaiser windows.

        Examples
        --------
        >>> x = aikit.Container(a=3, b=5)
        >>> aikit.Container.static_kaiser_window(x, True, 5)
        {
            a: aikit.array([0.2049, 0.8712, 0.8712]),
            a: aikit.array([0.0367, 0.7753, 0.7753]),
        }
        """
        return self.static_kaiser_window(
            self,
            periodic,
            beta,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    @staticmethod
    def static_kaiser_bessel_derived_window(
        x: Union[int, aikit.Array, aikit.NativeArray, aikit.Container],
        periodic: Union[bool, aikit.Container] = True,
        beta: Union[float, aikit.Container] = 12.0,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        dtype: Optional[Union[aikit.Array, aikit.NativeArray, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of
        aikit.kaiser_bessel_derived_window. This method simply wraps the
        function, and so the docstring for aikit.kaiser_bessel_derived_window
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container including window lengths.
        periodic
            If True, returns a periodic window suitable for use in spectral analysis.
            If False, returns a symmetric window suitable for use in filter design.
        beta
            a float used as shape parameter for the window.
        dtype
            data type of the returned array.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the Kaiser Bessel Derived windows.

        Examples
        --------
        >>> x = aikit.Container(a=3, b=5)
        >>> aikit.Container.static_kaiser_bessel_derived_window(x, True, 5)
        {
            a: aikit.array([0.70710677, 0.70710677]),
            b: aikit.array([0.18493208, 0.9827513 , 0.9827513 , 0.18493208]),
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "kaiser_bessel_derived_window",
            x,
            periodic,
            beta,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    def kaiser_bessel_derived_window(
        self: aikit.Container,
        periodic: Union[bool, aikit.Container] = True,
        beta: Union[float, aikit.Container] = 12.0,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        dtype: Optional[Union[aikit.Array, aikit.NativeArray, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of
        aikit.kaiser_bessel_derived_window. This method simply wraps the
        function, and so the docstring for aikit.kaiser_bessel_derived_window
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container including window lengths.
        periodic
            If True, returns a periodic window suitable for use in spectral analysis.
            If False, returns a symmetric window suitable for use in filter design.
        beta
            a float used as shape parameter for the window.
        dtype
            data type of the returned array.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the Kaiser Bessel Derived windows.

        Examples
        --------
        >>> x = aikit.Container(a=3, b=5))
        >>> x.kaiser_bessel_derived_window(True, 5)
        {
            a: aikit.array([0.70710677, 0.70710677]),
            b: aikit.array([0.18493208, 0.9827513 , 0.9827513 , 0.18493208]),
        }
        """
        return self.static_kaiser_bessel_derived_window(
            self,
            periodic,
            beta,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    @staticmethod
    def static_hamming_window(
        x: Union[int, aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        periodic: Union[bool, aikit.Container] = True,
        alpha: Union[float, aikit.Container] = 0.54,
        beta: Union[float, aikit.Container] = 0.46,
        dtype: Optional[Union[aikit.Array, aikit.NativeArray, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.hamming_window. This
        method simply wraps the function, and so the docstring for
        aikit.hamming_window also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container including window lengths.
        periodic
            If True, returns a window to be used as periodic function.
            If False, return a symmetric window.
        alpha
            The coefficient alpha in the hamming window equation
        beta
            The coefficient beta in the hamming window equation
        dtype
            data type of the returned arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the Hamming windows.

        Examples
        --------
        >>> x = aikit.Container(a=3, b=5)
        >>> aikit.Container.static_hamming_window(x, periodic=True, alpha=0.2, beta=2)
        {
            a: aikit.array([-1.8000,  1.2000,  1.2000]),
            b: aikit.array([-1.8000, -0.4180,  1.8180,  1.8180, -0.4180])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "hamming_window",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            periodic=periodic,
            alpha=alpha,
            beta=beta,
            dtype=dtype,
            out=out,
        )

    def hamming_window(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        periodic: Union[bool, aikit.Container] = True,
        alpha: Union[float, aikit.Container] = 0.54,
        beta: Union[float, aikit.Container] = 0.46,
        dtype: Optional[Union[aikit.Array, aikit.NativeArray, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.hamming_window. This
        method simply wraps the function, and so the docstring for
        aikit.hamming_window also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container including window lengths.
        periodic
            If True, returns a window to be used as periodic function.
            If False, return a symmetric window.
        alpha
            The coefficient alpha in the hamming window equation
        beta
            The coefficient beta in the hamming window equation
        dtype
            data type of the returned arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the Hamming windows.

        Examples
        --------
        >>> x = aikit.Container(a=3, b=5))
        >>> x.hamming_window(periodic=True, alpha=0.2, beta=2)
        {
            a: aikit.array([-1.8000,  1.2000,  1.2000]),
            b: aikit.array([-1.8000, -0.4180,  1.8180,  1.8180, -0.4180])
        }
        """
        return self.static_hamming_window(
            self, periodic=periodic, alpha=alpha, beta=beta, dtype=dtype, out=out
        )

    @staticmethod
    def static_vorbis_window(
        x: Union[int, aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        dtype: Optional[Union[aikit.Array, aikit.NativeArray, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.vorbis_window. This
        method simply wraps the function, and so the docstring for
        aikit.vorbis_window also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container including window lengths.

        dtype
            data type of the returned arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the vorbis windows.

        Examples
        --------
        >>> x = aikit.Container(a=3, b=5)
        >>> aikit.Container.static_vorbis_window(x)
        {
            a: aikit.array([0., 0.38268343, 0.92387953, 1., 0.92387953,
                          0.38268343]),
            b: aikit.array([0., 0.14943586, 0.51644717, 0.85631905, 0.98877142,
                          1., 0.98877142, 0.85631905, 0.51644717, 0.14943586])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "vorbis_window",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    def vorbis_window(
        self: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        dtype: Optional[Union[aikit.Array, aikit.NativeArray, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.vorbis_window. This
        method simply wraps the function, and so the docstring for
        aikit.vorbis_window also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container including window lengths.
        dtype
            data type of the returned arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that includes the vorbis windows.

        Examples
        --------
        >>> x = aikit.Container(a=3, b=5))
        >>> x.vorbis_window()
        {
            a: aikit.array([0., 0.38268343, 0.92387953, 1., 0.92387953,
                          0.38268343]),
            b: aikit.array([0., 0.14943586, 0.51644717, 0.85631905, 0.98877142,
                          1., 0.98877142, 0.85631905, 0.51644717, 0.14943586])
        }
        """
        return self.static_vorbis_window(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    @staticmethod
    def static_tril_indices(
        n_rows: Union[int, aikit.Container],
        n_cols: Optional[Union[int, aikit.Container]] = None,
        k: Union[int, aikit.Container] = 0,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
    ) -> aikit.Container:
        return ContainerBase.multi_map_in_static_method(
            "tril_indices",
            n_rows,
            n_cols,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            device=device,
        )

    def tril_indices(
        self: aikit.Container,
        n_rows: Union[int, aikit.Container],
        n_cols: Optional[Union[int, aikit.Container]] = None,
        k: Union[int, aikit.Container] = 0,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
    ) -> aikit.Container:
        return self.static_tril_indices(
            n_rows,
            n_cols,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            device=device,
        )

    @staticmethod
    def static_eye_like(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        k: Union[int, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        out: Optional[aikit.Container] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.eye_like. This method
        simply wraps the function, and so the docstring for aikit.eye_like also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array or container from which to derive the output container shape.
        k
            index of the diagonal. A positive value refers to an upper diagonal,
            a negative value to a lower diagonal, and 0 to the main diagonal.
            Default: ``0``.
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
        dtype
            output array data type. If ``dtype`` is ``None``, the output container
            data type must be inferred from ``self``. Default  ``None``.
        device
            device on which to place the created array. If device is ``None``, the
            output container device must be inferred from ``self``. Default: ``None``.
        out
            optional output container, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            a container having the same shape as ``x`` and filled with ``ones``
            in diagonal ``k`` and ``zeros`` elsewhere.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([0., 2.6, -3.5]),
                              b=aikit.array([4.5, -5.3, -0, -2.3]))
        >>> y = aikit.Container.static_eye_like(x)
        >>> print(y)
        {
            a: aikit.array([[1.]]),
            b: aikit.array([[1.]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "eye_like",
            x,
            k=k,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    def eye_like(
        self: aikit.Container,
        /,
        k: Union[int, aikit.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        out: Optional[aikit.Container] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.eye_like. This method
        simply wraps the function, and so the docstring for aikit.eye_like also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container from which to derive the output container shape.
        k
            index of the diagonal. A positive value refers to an upper diagonal,
            a negative value to a lower diagonal, and 0 to the main diagonal.
            Default: ``0``.
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
        dtype
            output array data type. If ``dtype`` is ``None``, the output container
            data type must be inferred from ``self``. Default: ``None``.
        device
            device on which to place the created array. If device is ``None``, the
            output container device must be inferred from ``self``. Default: ``None``.
        out
            optional output container, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            a container having the same shape as ``x`` and filled with ``ones``
            in diagonal ``k`` and ``zeros`` elsewhere.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([3., 8.]), b=aikit.array([2., 2.]))
        >>> y = x.eye_like()
        >>> print(y)
        {
            a: aikit.array([[1.],
                          [0.]]),
            b: aikit.array([[1.],
                          [0.]])
        }
        """
        return self.static_eye_like(
            self,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def static_unsorted_segment_min(
        data: aikit.Container,
        segment_ids: aikit.Container,
        num_segments: Union[int, aikit.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        r"""aikit.Container instance method variant of aikit.unsorted_segment_min.
        This method simply wraps the function, and so the docstring for
        aikit.unsorted_segment_min also applies to this method with minimal
        changes.

        Note
        ----
        If the given segment ID `i` is negative, then the corresponding
        value is dropped, and will not be included in the result.

        Parameters
        ----------
        data
            input array or container from which to gather the input.
        segment_ids
            Must be in the same size with the first dimension of `data`. Has to be
            of integer data type. The index-th element of `segment_ids` array is
            the segment identifier for the index-th element of `data`.
        num_segments
            An integer or array representing the total number of distinct segment IDs.
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
            A container, representing the result of a segmented min operation.
            For each segment, it computes the min value in `data` where `segment_ids`
            equals to segment ID.
        """
        return ContainerBase.cont_multi_map_in_function(
            "unsorted_segment_min",
            data,
            segment_ids,
            num_segments,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def unsorted_segment_min(
        self: aikit.Container,
        segment_ids: aikit.Container,
        num_segments: Union[int, aikit.Container],
    ):
        r"""aikit.Container instance method variant of aikit.unsorted_segment_min.
        This method simply wraps the function, and so the docstring for
        aikit.unsorted_segment_min also applies to this method with minimal
        changes.

        Note
        ----
        If the given segment ID `i` is negative, then the corresponding
        value is dropped, and will not be included in the result.

        Parameters
        ----------
        self
            input array or container from which to gather the input.
        segment_ids
            Must be in the same size with the first dimension of `self`. Has to be
            of integer data type. The index-th element of `segment_ids` array is
            the segment identifier for the index-th element of `self`.
        num_segments
            An integer or array representing the total number of distinct segment IDs.

        Returns
        -------
        ret
            A container, representing the result of a segmented min operation.
            For each segment, it computes the min value in `self` where `segment_ids`
            equals to segment ID.
        """
        return self.static_unsorted_segment_min(
            self,
            segment_ids,
            num_segments,
        )

    @staticmethod
    def static_unsorted_segment_sum(
        data: aikit.Container,
        segment_ids: aikit.Container,
        num_segments: Union[int, aikit.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        r"""aikit.Container instance method variant of aikit.unsorted_segment_sum.
        This method simply wraps the function, and so the docstring for
        aikit.unsorted_segment_sum also applies to this method with minimal
        changes.

        Parameters
        ----------
        data
            input array or container from which to gather the input.
        segment_ids
            Must be in the same size with the first dimension of `data`. Has to be
            of integer data type. The index-th element of `segment_ids` array is
            the segment identifier for the index-th element of `data`.
        num_segments
            An integer or array representing the total number of distinct segment IDs.
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
            A container, representing the result of a segmented sum operation.
            For each segment, it computes the sum of values in `data` where
            `segment_ids` equals to segment ID.
        """
        return ContainerBase.cont_multi_map_in_function(
            "unsorted_segment_sum",
            data,
            segment_ids,
            num_segments,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def unsorted_segment_sum(
        self: aikit.Container,
        segment_ids: aikit.Container,
        num_segments: Union[int, aikit.Container],
    ):
        r"""aikit.Container instance method variant of aikit.unsorted_segment_sum.
        This method simply wraps the function, and so the docstring for
        aikit.unsorted_segment_sum also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            input array or container from which to gather the input.
        segment_ids
            Must be in the same size with the first dimension of `self`. Has to be
            of integer data type. The index-th element of `segment_ids` array is
            the segment identifier for the index-th element of `self`.
        num_segments
            An integer or array representing the total number of distinct segment IDs.

        Returns
        -------
        ret
            A container, representing the result of a segmented sum operation.
            For each segment, it computes the sum of values in `self` where
            `segment_ids` equals to segment ID.
        """
        return self.static_unsorted_segment_sum(
            self,
            segment_ids,
            num_segments,
        )

    @staticmethod
    def static_blackman_window(
        window_length: Union[int, aikit.Container],
        periodic: bool = True,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.blackman_window. This
        method simply wraps the function, and so the docstring for
        aikit.blackman_window also applies to this method with minimal changes.

        Parameters
        ----------
        window_length
            container including multiple window sizes.
        periodic
            If True, returns a window to be used as periodic function.
            If False, return a symmetric window.
        dtype
            The data type to produce. Must be a floating point type.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that contains the Blackman windows.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> x = aikit.Container(a=3, b=5)
        >>> aikit.Container.static_blackman_window(x)
        {
            a: aikit.array([-1.38777878e-17,  6.30000000e-01,  6.30000000e-01])
            b: aikit.array([-1.38777878e-17,  2.00770143e-01,  8.49229857e-01,
                        8.49229857e-01, 2.00770143e-01])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "blackman_window",
            window_length,
            periodic,
            dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def blackman_window(
        self: aikit.Container,
        periodic: bool = True,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        *,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.blackman_window. This
        method simply wraps the function, and so the docstring for
        aikit.blackman_window also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container with window sizes.
        periodic
            If True, returns a window to be used as periodic function.
            If False, return a symmetric window.
        dtype
            The data type to produce. Must be a floating point type.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container containing the Blackman windows.

        Examples
        --------
        With one :class:`aikit.Container` input:
        >>> x = aikit.Container(a=3, b=5)
        >>> aikit.blackman_window(x)
        {
            a: aikit.array([-1.38777878e-17,  6.30000000e-01,  6.30000000e-01])
            b: aikit.array([-1.38777878e-17,  2.00770143e-01,  8.49229857e-01,
                            8.49229857e-01, 2.00770143e-01])
        }
        """
        return self.static_blackman_window(self, periodic, dtype, out=out)

    @staticmethod
    def _static_trilu(
        x: Union[aikit.Array, aikit.NativeArray, aikit.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        k: Union[int, aikit.Container] = 0,
        upper: bool = True,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return ContainerBase.cont_multi_map_in_function(
            "trilu",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            k=k,
            upper=upper,
            out=out,
        )

    def trilu(
        self: aikit.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        k: Union[int, aikit.Container] = 0,
        upper: bool = True,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        return self._static_trilu(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            k=k,
            upper=upper,
            out=out,
        )

    @staticmethod
    def static_mel_weight_matrix(
        num_mel_bins: Union[int, aikit.Container],
        dft_length: Union[int, aikit.Container],
        sample_rate: Union[int, aikit.Container],
        lower_edge_hertz: Optional[Union[float, aikit.Container]] = 0.0,
        upper_edge_hertz: Optional[Union[float, aikit.Container]] = 3000.0,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        r"""aikit.Container instance method variant of aikit.mel_weight_matrix. This
        method simply wraps the function, and so the docstring for
        aikit.mel_weight_matrix also applies to this method with minimal changes.

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
            MelWeightMatrix of shape:  [frames, num_mel_bins]
        """
        return ContainerBase.cont_multi_map_in_function(
            "mel_weight_matrix",
            num_mel_bins,
            dft_length,
            sample_rate,
            lower_edge_hertz,
            upper_edge_hertz,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def mel_weight_matrix(
        self: aikit.Container,
        num_mel_bins: int,
        dft_length: int,
        sample_rate: int,
        lower_edge_hertz: Optional[float] = 0.0,
        upper_edge_hertz: Optional[float] = 3000.0,
    ):
        r"""aikit.Container instance method variant of aikit.mel_weight_matrix. This
        method simply wraps the function, and so the docstring for
        aikit.mel_weight_matrix also applies to this method with minimal changes.

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
            MelWeightMatrix of shape:  [frames, num_mel_bins]
        """
        return self.static_mel_weight_matrix(
            num_mel_bins,
            dft_length,
            sample_rate,
            lower_edge_hertz,
            upper_edge_hertz,
        )

    @staticmethod
    def static_unsorted_segment_mean(
        data: aikit.Container,
        segment_ids: Union[aikit.Array, aikit.Container],
        num_segments: Union[int, aikit.Container],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        """Compute the mean of values in the input data based on segment
        identifiers.

        Parameters
        ----------
        data : aikit.Container
            Input array or container from which to gather the input.
        segment_ids : aikit.Container
            An array of integers indicating the segment identifier for each element in
            'data'.
        num_segments : Union[int, aikit.Container]
            An integer or array representing the total number of distinct segment IDs.
        key_chains : Optional[Union[List[str], Dict[str, str], aikit.Container]], optional
            The key-chains to apply or not apply the method to. Default is None.
        to_apply : Union[bool, aikit.Container], optional
            If True, the method will be applied to key-chains, otherwise key-chains will
            be skipped. Default is True.
        prune_unapplied : Union[bool, aikit.Container], optional
            Whether to prune key-chains for which the function was not applied.
            Default is False.
        map_sequences : Union[bool, aikit.Container], optional
            Whether to also map method to sequences (lists, tuples). Default is False.

        Returns
        -------
        aikit.Container
            A container representing the result of a segmented mean operation.
            For each segment, it computes the mean of values in 'data' where
            'segment_ids' equals the segment ID.
        """
        return ContainerBase.cont_multi_map_in_function(
            "unsorted_segment_mean",
            data,
            segment_ids,
            num_segments,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def unsorted_segment_mean(
        self: aikit.Container,
        segment_ids: Union[aikit.Array, aikit.Container],
        num_segments: Union[int, aikit.Container],
    ) -> aikit.Container:
        """Compute the mean of values in the input array or container based on
        segment identifiers.

        Parameters
        ----------
        self : aikit.Container
            Input array or container from which to gather the input.
        segment_ids : aikit.Container
            An array of integers indicating the segment identifier for each element
            in 'self'.
        num_segments : Union[int, aikit.Container]
            An integer or array representing the total number of distinct segment IDs.

        Returns
        -------
        aikit.Container
            A container representing the result of a segmented mean operation.
            For each segment, it computes the mean of values in 'self' where
            'segment_ids' equals the segment ID.

        Example
        --------
        >>> data = aikit.Container(a=aikit.array([0., 1., 2., 4.]),
        ...                      b=aikit.array([3., 4., 5., 6.]))
        >>> segment_ids = aikit.array([0, 0, 1, 1])
        >>> num_segments = 2
        >>> result = aikit.unsorted_segment_mean(data, segment_ids, num_segments)
        >>> print(result)
        {
            a: aikit.array([0.5, 3.0]),
            b: aikit.array([3.5, 5.5])
        }

        >>> data = aikit.Container(a=aikit.array([0., 1., 2., 4., 5., 6.]),
        ...                      b=aikit.array([3., 4., 5., 6., 7., 8.]))
        >>> segment_ids = aikit.array([0, 0, 1, 1, 2, 2])
        >>> num_segments = 3
        >>> result = aikit.unsorted_segment_mean(data, segment_ids, num_segments)
        >>> print(result)
        {
            a: aikit.array([0.5, 3.0, 5.5]),
            b: aikit.array([3.5, 5.5, 7.5])
        }
        """
        return self.static_unsorted_segment_mean(
            self,
            segment_ids,
            num_segments,
        )

    @staticmethod
    def static_polyval(
        coeffs: aikit.Container,
        x: Union[aikit.Container, int, float],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> aikit.Container:
        r"""aikit.Container static method variant of aikit.polyval. This method
        simply wraps the function, and so the docstring for aikit.polyval also
        applies to this method with minimal changes.

        Evaluate and return a polynomial at specific given values.

        Parameters
        ----------
        coeffs
            Polynomial coefficients (including zero) from highest degree
            to constant term.
        x
            The value of the indeterminate variable at which to evaluate the polynomial.
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
            Output container containing simplified result of substituting x in the
            coefficients - final value of polynomial.
        """
        return ContainerBase.cont_multi_map_in_function(
            "polyval",
            coeffs,
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def polyval(
        self: aikit.Container,
        coeffs: aikit.Container,
        x: aikit.Container,
    ) -> aikit.Container:
        r"""aikit.Container instance method variant of aikit.polyval. This method
        simply wraps the function, and so the docstring for aikit.polyval also
        applies to this method with minimal changes.

        Evaluate and return a polynomial at specific given values.

        Parameters
        ----------
        self
            Arbitrary input container
        coeffs
            Polynomial coefficients (including zero) from highest degree to
            constant term.
        x
            The value of the indeterminate variable at which to
            evaluate the polynomial.

        Returns
        -------
        ret
            Output container containing simplified result of substituting x in the
            coefficients - final value of polynomial.
        """
        return self.static_polyval(coeffs, x)
