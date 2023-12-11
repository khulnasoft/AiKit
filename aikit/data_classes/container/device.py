# local
from typing import Union, Optional, Any, List, Dict

import aikit
from aikit.data_classes.container.base import ContainerBase


# ToDo: implement all methods here as public instance methods


class _ContainerWithDevice(ContainerBase):
    @staticmethod
    def _static_dev(
        x: aikit.Container, /, *, as_native: Union[bool, aikit.Container] = False
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.dev. This method simply
        wraps the function, and so the docstring for aikit.dev also applies to
        this method with minimal changes.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[2, 3], [3, 5]]),
        ...                   b=aikit.native_array([1, 2, 4, 5, 7]))
        >>> as_native = aikit.Container(a=True, b=False)
        >>> y = aikit.Container.static_dev(x, as_native=as_native)
        >>> print(y)
        {
            a: device(type=cpu),
            b: cpu
        }
        """
        return ContainerBase.cont_multi_map_in_function("dev", x, as_native=as_native)

    def dev(
        self: aikit.Container, as_native: Union[bool, aikit.Container] = False
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.dev. This method simply
        wraps the function, and so the docstring for aikit.dev also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            contaioner of arrays for which to get the device handle.
        as_native
            Whether or not to return the dev in native format. Default is ``False``.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[2, 3, 1], [3, 5, 3]]),
        ...                   b=aikit.native_array([[1, 2], [4, 5]]))
        >>> as_native = aikit.Container(a=False, b=True)
        >>> y = x.dev(as_native=as_native)
        >>> print(y)
        {
            a:cpu,
            b:cpu
        }
        """
        return self._static_dev(self, as_native=as_native)

    @staticmethod
    def _static_to_device(
        x: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        device: Union[aikit.Device, aikit.NativeDevice, aikit.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        stream: Optional[Union[int, Any, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.to_device. This method
        simply wraps the function, and so the docstring for aikit.to_device also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
           input array to be moved to the desired device
        device
            device to move the input array `x` to
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
        stream
            stream object to use during copy. In addition to the types supported
            in array.__dlpack__(), implementations may choose to support any
            library-specific stream object with the caveat that any code using
            such an object would not be portable.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            input array x placed on the desired device

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[2, 3, 1], [3, 5, 3]]),
        ...                   b=aikit.native_array([[1, 2], [4, 5]]))
        >>> y = aikit.Container.static_to_device(x, 'cpu')
        >>> print(y.a.device, y.b.device)
        cpu cpu
        """
        return ContainerBase.cont_multi_map_in_function(
            "to_device",
            x,
            device,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            stream=stream,
            out=out,
        )

    def to_device(
        self: aikit.Container,
        device: Union[aikit.Device, aikit.NativeDevice, aikit.Container],
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        *,
        stream: Optional[Union[int, Any, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.to_device. This method
        simply wraps the function, and so the docstring for aikit.to_device also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
           input array to be moved to the desired device
        device
            device to move the input array `x` to
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
        stream
            stream object to use during copy. In addition to the types supported
            in array.__dlpack__(), implementations may choose to support any
            library-specific stream object with the caveat that any code using
            such an object would not be portable.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            input array x placed on the desired device

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[2, 3, 1], [3, 5, 3]]),
        ...                   b=aikit.native_array([[1, 2], [4, 5]]))
        >>> y = x.to_device('cpu')
        >>> print(y.a.device, y.b.device)
        cpu cpu
        """
        return self._static_to_device(
            self,
            device,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            stream=stream,
            out=out,
        )
