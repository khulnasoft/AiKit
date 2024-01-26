# global
from typing import Optional, Union, List, Dict

# local
import aikit
from aikit.data_classes.container.base import ContainerBase


class _ContainerWithLosses(ContainerBase):
    @staticmethod
    def _static_cross_entropy(
        true: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        pred: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        axis: Union[int, aikit.Container] = -1,
        epsilon: Union[float, aikit.Container] = 1e-7,
        reduction: Union[str, aikit.Container] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.cross_entropy. This
        method simply wraps the function, and so the docstring for
        aikit.cross_entropy also applies to this method with minimal changes.

        Parameters
        ----------
        true
            input array or container containing true labels.
        pred
            input array or container containing the predicted labels.
        axis
            the axis along which to compute the cross-entropy. If axis is ``-1``,
            the cross-entropy will be computed along the last dimension.
            Default: ``-1``.
        epsilon
            a float in [0.0, 1.0] specifying the amount of smoothing when calculating
            the loss. If epsilon is ``0``, no smoothing will be applied.
            Default: ``1e-7``.
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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The cross-entropy loss between the given distributions.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([0, 0, 1]), b=aikit.array([1, 1, 0]))
        >>> y = aikit.Container(a=aikit.array([0.6, 0.2, 0.3]),b=aikit.array([0.8, 0.2, 0.2]))
        >>> z = aikit.Container.static_cross_entropy(x, y)
        >>> print(z)
        {
            a: aikit.array(1.20397282),
            b: aikit.array(1.83258148)
        }

        With a mix of :class:`aikit.Array` and :class:`aikit.Container` inputs:

        >>> x = aikit.array([0, 0, 1])
        >>> y = aikit.Container(a=aikit.array([0.6, 0.2, 0.3]),b=aikit.array([0.8, 0.2, 0.2]))
        >>> z = aikit.Container.static_cross_entropy(x, y)
        >>> print(z)
        {
            a: aikit.array(1.20397282),
            b: aikit.array(1.60943794)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "cross_entropy",
            true,
            pred,
            axis=axis,
            epsilon=epsilon,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def cross_entropy(
        self: aikit.Container,
        pred: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        axis: Union[int, aikit.Container] = -1,
        epsilon: Union[float, aikit.Container] = 1e-7,
        reduction: Union[str, aikit.Container] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.cross_entropy. This
        method simply wraps the function, and so the docstring for
        aikit.cross_entropy also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container containing true labels.
        pred
            input array or container containing the predicted labels.
        axis
            the axis along which to compute the cross-entropy. If axis is ``-1``,
            the cross-entropy will be computed along the last dimension.
            Default: ``-1``.
        epsilon
            a float in [0.0, 1.0] specifying the amount of smoothing when calculating
            the loss. If epsilon is ``0``, no smoothing will be applied.
            Default: ``1e-7``.
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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The cross-entropy loss between the given distributions.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 0, 0]),b=aikit.array([0, 0, 1]))
        >>> y = aikit.Container(a=aikit.array([0.6, 0.2, 0.3]),b=aikit.array([0.8, 0.2, 0.2]))
        >>> z = x.cross_entropy(y)
        >>> print(z)
        {
            a:aikit.array(0.5108256),
            b:aikit.array(1.609438)
        }
        """
        return self._static_cross_entropy(
            self,
            pred,
            axis=axis,
            epsilon=epsilon,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_binary_cross_entropy(
        true: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        pred: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        from_logits: Union[bool, aikit.Container] = False,
        epsilon: Union[float, aikit.Container] = 0.0,
        reduction: Union[str, aikit.Container] = "mean",
        pos_weight: Optional[Union[aikit.Container, aikit.Array, aikit.NativeArray]] = None,
        axis: Optional[Union[int, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.binary_cross_entropy.
        This method simply wraps the function, and so the docstring for
        aikit.binary_cross_entropy also applies to this method with minimal
        changes.

        Parameters
        ----------
        true
            input array or container containing true labels.
        pred
            input array or container containing Predicted labels.
        from_logits
            Whether `pred` is expected to be a logits tensor. By
            default, we assume that `pred` encodes a probability distribution.
        epsilon
            a float in [0.0, 1.0] specifying the amount of smoothing when calculating
            the loss. If epsilon is ``0``, no smoothing will be applied. Default: ``0``.
        reduction
            ``'none'``: No reduction will be applied to the output.
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'none'``.
        pos_weight
            a weight for positive examples. Must be an array with length equal
            to the number of classes.
        axis
            Axis along which to compute crossentropy.
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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The binary cross entropy between the given distributions.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([1, 0, 0]),b=aikit.array([0, 0, 1]))
        >>> y = aikit.Container(a=aikit.array([0.6, 0.2, 0.3]),b=aikit.array([0.8, 0.2, 0.2]))
        >>> z = aikit.Container.static_binary_cross_entropy(x, y)
        >>> print(z)
        {
            a: aikit.array([0.511, 0.223, 0.357]),
            b: aikit.array([1.61, 0.223, 1.61])
        }

        With a mix of :class:`aikit.Array` and :class:`aikit.Container` inputs:

        >>> x = aikit.array([1 , 1, 0])
        >>> y = aikit.Container(a=aikit.array([0.7, 0.8, 0.2]),b=aikit.array([0.2, 0.6, 0.7]))
        >>> z = aikit.Container.static_binary_cross_entropy(x, y)
        >>> print(z)
        {
            a: aikit.array([0.357, 0.223, 0.223]),
            b: aikit.array([1.61, 0.511, 1.2])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "binary_cross_entropy",
            true,
            pred,
            epsilon=epsilon,
            from_logits=from_logits,
            reduction=reduction,
            pos_weight=pos_weight,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def binary_cross_entropy(
        self: aikit.Container,
        pred: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        from_logits: Union[bool, aikit.Container] = False,
        epsilon: Union[float, aikit.Container] = 0.0,
        reduction: Union[str, aikit.Container] = "mean",
        pos_weight: Optional[Union[aikit.Container, aikit.Array, aikit.NativeArray]] = None,
        axis: Optional[Union[int, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.binary_cross_entropy.
        This method simply wraps the function, and so the docstring for
        aikit.binary_cross_entropy also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            input container containing true labels.
        pred
            input array or container containing Predicted labels.
         from_logits
            Whether `pred` is expected to be a logits tensor. By
            default, we assume that `pred` encodes a probability distribution.
        epsilon
            a float in [0.0, 1.0] specifying the amount of smoothing when
            calculating the loss. If epsilon is ``0``, no smoothing will be applied.
            Default: ``0``.
        reduction
            ``'none'``: No reduction will be applied to the output.
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'none'``.
        pos_weight
            a weight for positive examples. Must be an array with length equal
            to the number of classes.
        axis
            Axis along which to compute crossentropy.
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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.


        Returns
        -------
        ret
            The binary cross entropy between the given distributions.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 0, 0]),b=aikit.array([0, 0, 1]))
        >>> y = aikit.Container(a=aikit.array([0.6, 0.2, 0.3]),b=aikit.array([0.8, 0.2, 0.2]))
        >>> z = x.binary_cross_entropy(y)
        >>> print(z)
        {
            a: aikit.array([0.511, 0.223, 0.357]),
            b: aikit.array([1.61, 0.223, 1.61])
        }
        """
        return self._static_binary_cross_entropy(
            self,
            pred,
            epsilon=epsilon,
            from_logits=from_logits,
            reduction=reduction,
            pos_weight=pos_weight,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_sparse_cross_entropy(
        true: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        pred: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        axis: Union[int, aikit.Container] = -1,
        epsilon: Union[float, aikit.Container] = 1e-7,
        reduction: Union[str, aikit.Container] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.sparse_cross_entropy.
        This method simply wraps the function, and so the docstring for
        aikit.sparse_cross_entropy also applies to this method with minimal
        changes.

        Parameters
        ----------
        true
            input array or container containing the true labels as logits.
        pred
            input array or container containing the predicted labels as logits.
        axis
            the axis along which to compute the cross-entropy. If axis is ``-1``, the
            cross-entropy will be computed along the last dimension. Default: ``-1``.
            epsilon
            a float in [0.0, 1.0] specifying the amount of smoothing when calculating
            the loss. If epsilon is ``0``, no smoothing will be applied.
            Default: ``1e-7``.
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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The sparse cross-entropy loss between the given distributions.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([1, 0, 0]),b=aikit.array([0, 0, 1]))
        >>> y = aikit.Container(a=aikit.array([0.6, 0.2, 0.3]),b=aikit.array([0.8, 0.2, 0.2]))
        >>> z = aikit.Container.static_sparse_cross_entropy(x, y)
        >>> print(z)
        {
            a: aikit.array([1.61, 0.511, 0.511]),
            b: aikit.array([0.223, 0.223, 1.61])
        }

        With a mix of :class:`aikit.Array` and :class:`aikit.Container` inputs:

        >>> x = aikit.array([1 , 1, 0])
        >>> y = aikit.Container(a=aikit.array([0.7, 0.8, 0.2]),b=aikit.array([0.2, 0.6, 0.7]))
        >>> z = aikit.Container.static_sparse_cross_entropy(x, y)
        >>> print(z)
        {
            a: aikit.array([0.223, 0.223, 0.357]),
            b: aikit.array([0.511, 0.511, 1.61])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "sparse_cross_entropy",
            true,
            pred,
            axis=axis,
            epsilon=epsilon,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sparse_cross_entropy(
        self: aikit.Container,
        pred: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        axis: Union[int, aikit.Container] = -1,
        epsilon: Union[float, aikit.Container] = 1e-7,
        reduction: Union[str, aikit.Container] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.sparse_cross_entropy.
        This method simply wraps the function, and so the docstring for
        aikit.sparse_cross_entropy also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            input container containing the true labels as logits.
        pred
            input array or container containing the predicted labels as logits.
        axis
            the axis along which to compute the cross-entropy. If axis is ``-1``, the
            cross-entropy will be computed along the last dimension. Default: ``-1``.
            epsilon
            a float in [0.0, 1.0] specifying the amount of smoothing when calculating
            the loss. If epsilon is ``0``, no smoothing will be applied.
            Default: ``1e-7``.
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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The sparse cross-entropy loss between the given distributions.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 0, 0]),b=aikit.array([0, 0, 1]))
        >>> y = aikit.Container(a=aikit.array([0.6, 0.2, 0.3]),b=aikit.array([0.8, 0.2, 0.2]))
        >>> z = x.sparse_cross_entropy(y)
        >>> print(z)
        {
            a: aikit.array([1.61, 0.511, 0.511]),
            b: aikit.array([0.223, 0.223, 1.61])
        }
        """
        return self._static_sparse_cross_entropy(
            self,
            pred,
            axis=axis,
            epsilon=epsilon,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
