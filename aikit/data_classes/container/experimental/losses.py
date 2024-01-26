# global
from typing import Optional, Union, List, Dict

# local
import aikit
from aikit.data_classes.container.base import ContainerBase


class _ContainerWithLossesExperimental(ContainerBase):
    @staticmethod
    def _static_l1_loss(
        input: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        target: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        reduction: Optional[Union[str, aikit.Container]] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.l1_loss. This method
        simply wraps the function, and so the docstring for aikit.l1_loss also
        applies to this method with minimal changes.

        Parameters
        ----------
        input
            input array or container.
        target
            input array or container containing the targeted values.
        reduction
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed.
            ``'none'``: No reduction will be applied to the output. Default: ``'mean'``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If input, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``input``.
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
            The L1 loss between the input array and the targeted values.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([1, 2, 3]), b=aikit.array([4, 5, 6]))
        >>> y = aikit.Container(a=aikit.array([2, 2, 2]), b=aikit.array([5, 5, 5]))
        >>> z = aikit.Container.static_l1_loss(x, y)
        >>> print(z)
        {
            a: aikit.array(1.),
            b: aikit.array(0.)
        }

        With a mix of :class:`aikit.Array` and :class:`aikit.Container` inputs:

        >>> x = aikit.array([1, 2, 3])
        >>> y = aikit.Container(a=aikit.array([2, 2, 2]), b=aikit.array([5, 5, 5]))
        >>> z = aikit.Container.static_l1_loss(x, y)
        >>> print(z)
        {
            a: aikit.array(1.),
            b: aikit.array(4.)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "l1_loss",
            input,
            target,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def l1_loss(
        self: aikit.Container,
        target: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        reduction: Optional[Union[str, aikit.Container]] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.l1_loss. This method
        simply wraps the function, and so the docstring for aikit.l1_loss also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        target
            input array or container containing the targeticted values.
        reduction
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed.
            ``'none'``: No reduction will be applied to the output. Default: ``'mean'``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If input, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``input``.
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
            The L1 loss between the input array and the targeticted values.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 2, 3]), b=aikit.array([4, 5, 6]))
        >>> y = aikit.Container(a=aikit.array([2, 2, 2]), b=aikit.array([5, 5, 5]))
        >>> z = x.l1_loss(y)
        >>> print(z)
        {
            a: aikit.array(1.),
            b: aikit.array(0.)
        }
        """
        return self._static_l1_loss(
            self,
            target,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_log_poisson_loss(
        input: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        target: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        compute_full_loss: bool = False,
        axis: int = -1,
        reduction: Optional[Union[str, aikit.Container]] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.log_poisson_loss. This
        method simply wraps the function, and so the docstring for
        aikit.log_poisson_loss also applies to this method with minimal changes.

        Parameters
        ----------
        input
            input array or container.
        target
            input array or container containing the targeted values.
        compute_full_loss
            whether to compute the full loss. If false, a constant term is dropped
            in favor of more efficient optimization. Default: ``False``.
        axis
            the axis along which to compute the log-likelihood loss. If axis is ``-1``,
            the log-likelihood loss will be computed along the last dimension.
            Default: ``-1``.
        reduction
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed.
            ``'none'``: No reduction will be applied to the output. Default: ``'none'``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If input, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``input``.
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
            The L1 loss between the input array and the targeted values.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([1, 2, 3]), b=aikit.array([4, 5, 6]))
        >>> y = aikit.Container(a=aikit.array([2, 2, 2]), b=aikit.array([5, 5, 5]))
        >>> z = aikit.Container.static_log_poisson_loss(x, y, reduction='mean')
        >>> print(z)
        {
            a: aikit.array(1.),
            b: aikit.array(0.)
        }

        With a mix of :class:`aikit.Array` and :class:`aikit.Container` inputs:

        >>> x = aikit.array([1, 2, 3])
        >>> y = aikit.Container(a=aikit.array([2, 2, 2]), b=aikit.array([5, 5, 5]))
        >>> z = aikit.Container.static_log_poisson_loss(x, y, reduction='mean')
        >>> print(z)
        {
            a: aikit.array(1.),
            b: aikit.array(4.)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "log_poisson_loss",
            input,
            target,
            compute_full_loss=compute_full_loss,
            axis=axis,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def log_poisson_loss(
        self: aikit.Container,
        target: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        compute_full_loss: bool = False,
        axis: int = -1,
        reduction: Optional[Union[str, aikit.Container]] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.log_poisson_loss. This
        method simply wraps the function, and so the docstring for
        aikit.log_poisson_loss also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        target
            input array or container containing the targeticted values.
        compute_full_loss
            whether to compute the full loss. If false, a constant term is dropped
            in favor of more efficient optimization. Default: ``False``.
        axis
            the axis along which to compute the log-likelihood loss. If axis is ``-1``,
            the log-likelihood loss will be computed along the last dimension.
            Default: ``-1``.
        reduction
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed.
            ``'none'``: No reduction will be applied to the output. Default: ``'none'``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If input, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``input``.
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
            The L1 loss between the input array and the targeticted values.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 2, 3]), b=aikit.array([4, 5, 6]))
        >>> y = aikit.Container(a=aikit.array([2, 2, 2]), b=aikit.array([5, 5, 5]))
        >>> z = x.log_poisson_loss(y)
        >>> print(z)
        {
            a: aikit.array(1.),
            b: aikit.array(0.)
        }
        """
        return self._static_log_poisson_loss(
            self,
            target,
            compute_full_loss=compute_full_loss,
            axis=axis,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_smooth_l1_loss(
        input: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        target: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        beta: Optional[Union[float, aikit.Container]] = 1.0,
        reduction: Optional[Union[str, aikit.Container]] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.smooth_l1_loss. This
        method simply wraps the function, and so the docstring for aikit.
        smooth_l1_loss also applies to this method with minimal changes.

        Parameters
        ----------
        input
            input array or container containing input labels.
        target
            input array or container containing the targeticted labels.
        beta
            a positive float value that sets the smoothness threshold.
            Default: ``1.0``.
        reduction
            ``'none'``: No reduction will be applied to the output.
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'mean'``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If input, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``input``.
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
            The smooth L1 loss between the input array and the targeticted labels.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([1, 0, 2]), b=aikit.array([3, 2, 1]))
        >>> y = aikit.Container(a=aikit.array([0.6, 0.2, 0.3]),
        b=aikit.array([0.8, 0.2, 0.2]))
        >>> z = aikit.Container.static_smooth_l1_loss(x, y)
        >>> print(z)
        {
            a: aikit.array(0.9),
            b: aikit.array(0.25)
        }

        With a mix of :class:`aikit.Array` and :class:`aikit.Container` inputs:

        >>> x = aikit.array([1 , 0, 2])
        >>> y = aikit.Container(a=aikit.array([0.6, 0.2, 0.3]),
        b=aikit.array([0.8, 0.2, 0.2]))
        >>> z = aikit.Container.static_smooth_l1_loss(x, y)
        >>> print(z)
        {
            a: aikit.array(0.9),
            b: aikit.array(0.25)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "smooth_l1_loss",
            input,
            target,
            beta=beta,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def smooth_l1_loss(
        self: aikit.Container,
        target: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        beta: Optional[Union[float, aikit.Container]] = 1.0,
        reduction: Optional[Union[str, aikit.Container]] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.smooth_l1_loss. This
        method simply wraps the function, and so the docstring for aikit.
        smooth_l1_loss also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container containing input labels.
        target
            input array or container containing the targeticted labels.
        beta
            a positive float value that sets the smoothness threshold.
            Default: ``1.0``.
        reduction
            ``'none'``: No reduction will be applied to the output.
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'mean'``.
        key_chains
            The key-chains to apply or not apply the method to. Default is
            ``None``.
        to_apply
            If input, the method will be applied to key_chains, otherwise
            key_chains
            will be skipped. Default is ``input``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to.
            It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The smooth L1 loss between the input array and the targeticted labels.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 0, 2]), b=aikit.array([3, 2, 1]))
        >>> y = aikit.Container(a=aikit.array([0.6, 0.2, 0.3]),
        b=aikit.array([0.8, 0.2, 0.2]))
        >>> z = x.smooth_l1_loss(y)
        >>> print(z)
        {
            a: aikit.array(0.9),
            b: aikit.array(0.25)
        }
        """
        return self._static_smooth_l1_loss(
            self,
            target,
            beta=beta,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_huber_loss(
        true: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        pred: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        delta: Optional[Union[float, aikit.Container]] = 1.0,
        reduction: Optional[Union[str, aikit.Container]] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of huber_loss. This method
        simply wraps the function, and so the docstring for huber_loss also
        applies to this method with minimal changes.

        Parameters
        ----------
        true
            true array or container containing true labels.
        pred
            true array or container containing the predicted labels.
        delta
            The threshold parameter that determines the point where the loss transitions
            from squared error to absolute error. Default is 1.0.
        reduction : str, optional
            The type of reduction to apply to the loss.
            Possible values are "mean" (default)
            and "sum".
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If true, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``true``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the true broadcast to.

        Returns
        -------
        ret
            The Huber loss between the true and predicted values.

        Examples
        --------
        With :class:`aikit.Container` trues:

        >>> x = aikit.Container(a=aikit.array([1, 0, 3]), b=aikit.array([0, 0, 2]))
        >>> y = aikit.Container(a=aikit.array([1.5, 0.2, 2.8]), b=aikit.array([0.5, 0.2, 1.9])
        )
        >>> z = aikit.Container.static_huber_loss(x, y, delta=1.0)
        >>> print(z)
        {
            a: aikit.array(0.0575),
            b: aikit.array(0.005)
        }

        With a mix of :class:`aikit.Array` and :class:`aikit.Container` trues:

        >>> x = aikit.array([1, 0, 3])
        >>> y = aikit.Container(a=aikit.array([1.5, 0.2, 2.8]), b=aikit.array([0.5, 0.2, 1.9])
        )
        >>> z = aikit.Container.static_huber_loss(x, y, delta=1.0)
        >>> print(z)
        {
            a: aikit.array(0.0575),
            b: aikit.array(0.005)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "huber_loss",
            true,
            pred,
            delta=delta,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def huber_loss(
        self: aikit.Container,
        pred: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        delta: Optional[Union[float, aikit.Container]] = 1.0,
        reduction: Optional[Union[str, aikit.Container]] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of huber_loss. This method
        simply wraps the function, and so the docstring for huber_loss also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            true container containing true labels.
        pred
            true array or container containing the predicted labels.
        delta
            The threshold parameter that determines the point where the loss transitions
            from squared error to absolute error. Default is 1.0.
        reduction : str, optional
            The type of reduction to apply to the loss.
            Possible values are "mean" (default)
            and "sum".
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If true, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``true``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the trues broadcast to.

        Returns
        -------
        ret
            The Huber loss between the true and predicted values.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([1, 0, 3]), b=aikit.array([0, 0, 2]))
        >>> y = aikit.Container(a=aikit.array([1.5, 0.2, 2.8]), b=aikit.array([0.5, 0.2, 1.9])
        )
        >>> z = x.huber_loss(y, delta=1.0)
        >>> print(z)
        {
            a: aikit.array(0.0575),
            b: aikit.array(0.005)
        }
        """
        return self._static_huber_loss(
            self,
            pred,
            delta=delta,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_soft_margin_loss(
        input: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        target: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        reduction: Optional[Union[str, aikit.Container]] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.soft_margin_loss. This
        method simply wraps the function, and so the docstring for
        aikit.soft_margin_loss also applies to this method with minimal changes.

        # Insert the docstring here

        Parameters
        ----------
        input
            input array or container containing input labels.
        target
            input array or container containing the targeticted labels.
        reduction
            the reduction method. Default: "mean".
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If input, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is input.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The soft margin loss between the given distributions.
        """
        return ContainerBase.cont_multi_map_in_function(
            "soft_margin_loss",
            input,
            target,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def soft_margin_loss(
        self: aikit.Container,
        target: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        reduction: Optional[Union[str, aikit.Container]] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.soft_margin_loss. This
        method simply wraps the function, and so the docstring for
        aikit.soft_margin_loss also applies to this method with minimal changes.

        # Insert the docstring here

        Parameters
        ----------
        self
            input container containing input labels.
        target
            input array or container containing the targeticted labels.
        reduction
            the reduction method. Default: "mean".
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If input, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is input.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The soft margin loss between the given distributions.
        """
        return self._static_soft_margin_loss(
            self,
            target,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_kl_div(
        input: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        target: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        reduction: Optional[Union[str, aikit.Container]] = "mean",
        log_target=False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.kl_div. This method
        simply wraps the function, and so the docstring for aikit.kl_div also
        applies to this method with minimal changes.

        Parameters
        ----------
        input
            input array or container containing input distribution.
        target
            input array or container containing target distribution.
        reduction
            the reduction method. Default: "mean".
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If input, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is input.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The Kullback-Leibler divergence loss between the given distributions.
        """
        return ContainerBase.cont_multi_map_in_function(
            "kl_div",
            input,
            target,
            reduction=reduction,
            log_target=log_target,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def kl_div(
        self: aikit.Container,
        target: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        reduction: Optional[Union[str, aikit.Container]] = "mean",
        log_target=False,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.kl_div. This method
        simply wraps the function, and so the docstring for aikit.kl_div also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container containing input distribution.
        target
            input array or container containing target distribution.
        reduction
            the reduction method. Default: "mean".
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If input, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is input.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The Kullback-Leibler divergence loss between the given distributions.
        """
        return self._static_kl_div(
            self,
            target,
            reduction=reduction,
            log_target=log_target,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_poisson_nll_loss(
        input: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        target: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        *,
        log_input: [Union[bool, aikit.Container]] = True,
        full: [Union[bool, aikit.Container]] = False,
        eps: [Union[float, aikit.Container]] = 1e-8,
        reduction: [Union[str, aikit.Container]] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        r"""aikit.Container static method variant of aikit.poisson_nll_loss. This
        method simplywraps the function, and so the docstring for
        aikit.poisson_nll_loss also applies to this method with minimal changes.

        Parameters
        ----------
        input
            input array or container containing input labels.
        target
            input array or container containing the target labels.
        log_input
            If `True`, the loss is computed as
            :math:`exp(input) - target * input`. If `False`, the loss is computed as
            :math:`input - target * log(input + eps)`. Default is `True`.
        full
            Whether to compute the full loss, i.e.,
            to add the Stirling approximation term
            :math:`target * log(target) - target + 0.5 * log(2 * pi * target)`.
            Default is `False`.
        eps
            Small value to prevent evaluation of `log(0)` when `log_input` is `False`.
            Default is 1e-8.
        reduction
            Specifies the reduction applied to the output.
            Options are 'none', 'mean', or 'sum'.
            'none': no reduction will be applied. 'mean': the output will be averaged.
            'sum': the output will be summed.
            Default is 'mean'.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If input, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``input``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            An array of the same shape as `input` representing
            the Poisson Negative Log Likelihood Loss.

        Raises
        ------
        ValueError
            If the `input` and `target` tensors do not have the same shape.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([[0.6, 0.2, 0.3]], dtype=aikit.float32),
        ...                   b=aikit.array([[0.8, 0.2, 0.2]], dtype=aikit.float32))
        >>> y = aikit.Container(a=aikit.array([[1, 0, 2]], dtype=aikit.float32),
        ...                   b=aikit.array([[3, 2, 1]], dtype=aikit.float32))
        >>> z = aikit.Container._static_poisson_nll_loss(x,y)
        >>> print(z)
        {
            a: aikit.array(1.06446016),
            b: aikit.array(0.55611551)
        }

        With a mix of :class:`aikit.Array` and :class:`aikit.Container` inputs:

        >>> x = aikit.array([[1, 0, 2]], dtype=aikit.float32)
        >>> y = aikit.Container(a=aikit.array([[0.6, 0.2, 0.3]], dtype=aikit.float32),
        ...             b=aikit.array([[0.8, 0.2, 0.2]], dtype=aikit.float32))
        >>> z = aikit.Container._static_poisson_nll_loss(x, y)
        >>> print(z)
        {
            a: aikit.array(3.30244565),
            b: aikit.array(3.30244565)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "poisson_nll_loss",
            input,
            target,
            log_input=log_input,
            full=full,
            eps=eps,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def poisson_nll_loss(
        self: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        target: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        *,
        log_input: [Union[bool, aikit.Container]] = True,
        full: [Union[bool, aikit.Container]] = False,
        eps: [Union[float, aikit.Container]] = 1e-8,
        reduction: [Union[str, aikit.Container]] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        r"""aikit.Container instance method variant of aikit.poisson_nll_loss. This
        method simply wraps the function, and so the docstring for aikit.
        poisson_nll_loss also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container containing input labels.
        target
            input array or container containing the target labels.
        log_input
            If `True`, the loss is computed as
            :math:`exp(input) - target * input`. If `False`, the loss is computed as
            :math:`input - target * log(input + eps)`. Default is `True`.
        full
            Whether to compute the full loss, i.e.,
            to add the Stirling approximation term
            :math:`target * log(target) - target + 0.5 * log(2 * pi * target)`.
            Default is `False`.
        eps
            Small value to prevent evaluation of `log(0)` when `log_input` is `False`.
            Default is 1e-8.
        reduction
            Specifies the reduction applied to the output.
            Options are 'none', 'mean', or 'sum'.
            'none': no reduction will be applied. 'mean': the output will be averaged.
            'sum': the output will be summed.
            Default is 'mean'.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If input, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``input``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            An array of the same shape as `input` representing
            the Poisson Negative Log Likelihood Loss.

        Raises
        ------
        ValueError
            If the `input` and `target` tensors do not have the same shape.

        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[1, 0, 2]], dtype=aikit.float32),
        ...              b=aikit.array([[3, 2, 1]], dtype=aikit.float32))
        >>> y = aikit.Container(a=aikit.array([[0.6, 0.2, 0.3]], dtype=aikit.float32),
        ...              b=aikit.array([[0.8, 0.2, 0.2]], dtype=aikit.float32))
        >>> z = x.poisson_nll_loss(y)
        >>> print(z)
        {
            a: aikit.array(3.30244565),
            b: aikit.array(9.06429195)
        }
        """
        return self._static_poisson_nll_loss(
            self,
            target,
            log_input=log_input,
            full=full,
            eps=eps,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_hinge_embedding_loss(
        input: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        target: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        *,
        margin: [Union[float, aikit.Container]] = 1.0,
        reduction: [Union[str, aikit.Container]] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        r"""aikit.Container static method variant of aikit.hinge_embedding_loss.
        This method simplywraps the function, and so the docstring for
        aikit.hinge_embedding_loss also applies to this method with minimal
        changes.

        Parameters
        ----------
        input
            input array or container containing input labels.
        target
            input array or container containing the target labels.
        margin
            Sets the hyperparameter margin. Determines the necessary input size
            for hinge_embedding_loss calculations when label is -1. Inputs smaller
            than the margin are minimized with hinge_embedding_loss.
            Default is 1.0.
        reduction
            Specifies how to aggregate the loss across the batch. Options are:
            - ``'none'``: Returns the unreduced loss.
            - ``'mean'``: Returns the mean loss.
            - ``'sum'``: Returns the summed loss.
            Default is ``'mean'``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If input, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``input``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Shape
        -----
            - Input: :math:`(*)` where :math:`*` means, any number of dimensions. \
            The sum operation operates over all the elements.
            - Target: :math:`(*)`, same shape as the input
            - Output: scalar. If :attr:`reduction` is ``'none'``,
            then same shape as the input

        Returns
        -------
        ret
            Hinge embedding loss calculated from the input and label,
            shaped based on the reduction method.

        Examples
        --------
        With :class:`aikit.Container` inputs:

        >>> x = aikit.Container(a=aikit.array([[1, 0, 2]], dtype=aikit.float32),
        ...             b=aikit.array([[-1, 1, 1]], dtype=aikit.float32))
        >>> y = aikit.Container(a=aikit.array([[0.6, 0.2, 0.3]], dtype=aikit.float32),
        ...            b=aikit.array([[1, 1, 1]], dtype=aikit.float32))
        >>> z = aikit.Container._static_hinge_embedding_loss(x, y, reduction="none")
        >>> z
        {
            a: aikit.array([[0., 0., 0.]]),
            b: aikit.array([[-1., 1., 1.]])
        }

        With a mix of :class:`aikit.Array` and :class:`aikit.Container` inputs:

        >>> x = aikit.array([[10, 20, 32]], dtype=aikit.float32)
        >>> y = aikit.Container(a=aikit.array([[-1, -1, -1]], dtype=aikit.float32),
        ...           b=aikit.array([[1, 1, 1]], dtype=aikit.float32))
        >>> z = aikit.Container._static_hinge_embedding_loss(x, y,
        ...                             reduction="sum", margin=2.0)
        >>> z
        {
            a: aikit.array(0.),
            b: aikit.array(62.)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "hinge_embedding_loss",
            input,
            target,
            margin=margin,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def hinge_embedding_loss(
        self: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        target: Union[aikit.Container, aikit.Array, aikit.NativeArray],
        *,
        margin: [Union[float, aikit.Container]] = 1.0,
        reduction: [Union[str, aikit.Container]] = "mean",
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
    ) -> aikit.Container:
        r"""aikit.Container instance method variant of aikit.hinge_embedding_loss.
        This method simply wraps the function, and so the docstring for
        aikit.hinge_embedding_loss also applies to this method with minimal
        changes.

        Parameters
        ----------
        input
            input array or container containing input labels.
        target
            input array or container containing the target labels.
        margin
            Sets the hyperparameter margin. Determines the necessary input size
            for hinge_embedding_loss calculations when label is -1. Inputs smaller
            than the margin are minimized with hinge_embedding_loss.
            Default is 1.0.
        reduction
            Specifies how to aggregate the loss across the batch. Options are:
            - ``'none'``: Returns the unreduced loss.
            - ``'mean'``: Returns the mean loss.
            - ``'sum'``: Returns the summed loss.
            Default is ``'mean'``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If input, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``input``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Shape
        -----
            - Input: :math:`(*)` where :math:`*` means, any number of dimensions. \
            The sum operation operates over all the elements.
            - Target: :math:`(*)`, same shape as the input
            - Output: scalar. If :attr:`reduction` is ``'none'``,
            then same shape as the input

        Returns
        -------
        ret
            Hinge embedding loss calculated from the input and label,
            shaped based on the reduction method.


        Examples
        --------
        >>> x = aikit.Container(a=aikit.array([[1, 0, 2]], dtype=aikit.float32),
        ...              b=aikit.array([[3, 2, 1]], dtype=aikit.float32))
        >>> y = aikit.Container(a=aikit.array([[-1, -1, -1]], dtype=aikit.float32),
        ...              b=aikit.array([[1, 1, 1]], dtype=aikit.float32))
        >>> x.hinge_embedding_loss(y, reduction="none", margin=0.5)
        {
            a: aikit.array([[0., 0.5, 0.]]),
            b: aikit.array([[3., 2., 1.]])
        }
        """
        return self._static_hinge_embedding_loss(
            self,
            target,
            margin=margin,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )
