# global
import abc
from typing import Union, Optional

# local
import aikit

# ToDo: implement all methods here as public instance methods


class _ArrayWithGradients(abc.ABC):
    def stop_gradient(
        self: aikit.Array,
        /,
        *,
        preserve_type: bool = True,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.stop_gradient. This method
        simply wraps the function, and so the docstring for aikit.stop_gradient
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Array for which to stop the gradient.
        preserve_type
            Whether to preserve gradient computation on aikit.Array instances. Default is
            True.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            The same array x, but with no gradient information.

        Examples
        --------
        >>> x = aikit.array([1., 2., 3.])
        >>> y = x.stop_gradient(preserve_type=True)
        >>> print(y)
        aikit.array([1., 2., 3.])
        """
        return aikit.stop_gradient(self, preserve_type=preserve_type, out=out)

    def adam_step(
        self: aikit.Array,
        mw: Union[aikit.Array, aikit.NativeArray],
        vw: Union[aikit.Array, aikit.NativeArray],
        step: Union[int, float],
        /,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.adam_step. This method
        simply wraps the function, and so the docstring for aikit.adam_step also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
        mw
            running average of the gradients.
        vw
            running average of second moments of the gradients.
        step
            training step.
        beta1
            gradient forgetting factor (Default value = 0.9).
        beta2
            second moment of gradient forgetting factor (Default value = 0.999).
        epsilon
            divisor during adam update, preventing division by zero
            (Default value = 1e-7).
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The adam step delta.

        Examples
        --------
        With :class:`aikit.Array` inputs:

        >>> dcdw = aikit.array([1, 2, 3])
        >>> mw = aikit.ones(3)
        >>> vw = aikit.ones(1)
        >>> step = aikit.array(3)
        >>> adam_step_delta = dcdw.adam_step(mw, vw, step)
        >>> print(adam_step_delta)
        (aikit.array([0.2020105,0.22187898,0.24144873]),
            aikit.array([1.,1.10000002,1.20000005]),
            aikit.array([1.,1.00300002,1.00800002]))
        """
        return aikit.adam_step(
            self, mw, vw, step, beta1=beta1, beta2=beta2, epsilon=epsilon, out=out
        )

    def optimizer_update(
        self: aikit.Array,
        effective_grad: Union[aikit.Array, aikit.NativeArray],
        lr: Union[float, aikit.Array, aikit.NativeArray],
        /,
        *,
        stop_gradients: bool = True,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.optimizer_update. This
        method simply wraps the function, and so the docstring for
        aikit.optimizer_update also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Weights of the function to be updated.
        effective_grad
            Effective gradients of the cost c with respect to the weights ws,
            [dc/dw for w in ws].
        lr
            Learning rate(s), the rate(s) at which the weights should be updated
            relative to the gradient.
        stop_gradients
            Whether to stop the gradients of the variables after each gradient step.
            Default is ``True``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The new function weights ws_new, following the optimizer updates.

        Examples
        --------
        >>> w = aikit.array([1., 2., 3.])
        >>> effective_grad = aikit.zeros(3)
        >>> lr = 3e-4
        >>> ws_new = w.optimizer_update(effective_grad, lr)
        >>> print(ws_new)
        aikit.array([1., 2., 3.])
        """
        return aikit.optimizer_update(
            self, effective_grad, lr, stop_gradients=stop_gradients, out=out
        )

    def gradient_descent_update(
        self: aikit.Array,
        dcdw: Union[aikit.Array, aikit.NativeArray],
        lr: Union[float, aikit.Array, aikit.NativeArray],
        /,
        *,
        stop_gradients: bool = True,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.gradient_descent_update.
        This method simply wraps the function, and so the docstring for
        aikit.gradient_descent_update also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            Weights of the function to be updated.
        dcdw
            Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
        lr
            Learning rate(s), the rate(s) at which the weights should be
            updated relative to the gradient.
        stop_gradients
            Whether to stop the gradients of the variables after each gradient step.
            Default is ``True``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The new weights, following the gradient descent updates.

        Examples
        --------
        With :class:`aikit.Array` inputs:

        >>> w = aikit.array([[1., 2, 3],
        ...                [4, 6, 1],
        ...                [1, 0, 7]])
        >>> dcdw = aikit.array([[0.5, 0.2, 0.1],
        ...                   [0.3, 0.6, 0.4],
        ...                   [0.4, 0.7, 0.2]])
        >>> lr = aikit.array(0.1)
        >>> new_weights = w.gradient_descent_update(dcdw, lr, stop_gradients = True)
        >>> print(new_weights)
        aikit.array([[ 0.95,  1.98,  2.99],
        ...        [ 3.97,  5.94,  0.96],
        ...        [ 0.96, -0.07,  6.98]])
        """
        return aikit.gradient_descent_update(
            self, dcdw, lr, stop_gradients=stop_gradients, out=out
        )

    def lars_update(
        self: aikit.Array,
        dcdw: Union[aikit.Array, aikit.NativeArray],
        lr: Union[float, aikit.Array, aikit.NativeArray],
        /,
        *,
        decay_lambda: float = 0,
        stop_gradients: bool = True,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.lars_update. This method
        simply wraps the function, and so the docstring for aikit.lars_update
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Weights of the function to be updated.
        dcdw
            Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
        lr
            Learning rate, the rate at which the weights should be updated relative to
            the gradient.
        decay_lambda
            The factor used for weight decay. Default is zero.
        stop_gradients
            Whether to stop the gradients of the variables after each gradient step.
            Default is ``True``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The new function weights ws_new, following the LARS updates.

        Examples
        --------
        With :class:`aikit.Array` inputs:

        >>> w = aikit.array([[3., 1, 5],
        ...                [7, 2, 9]])
        >>> dcdw = aikit.array([[0.3, 0.1, 0.2],
        ...                   [0.1, 0.2, 0.4]])
        >>> lr = aikit.array(0.1)
        >>> new_weights = w.lars_update(dcdw, lr, stop_gradients = True)
        >>> print(new_weights)
        aikit.array([[2.34077978, 0.78025991, 4.56051969],
        ...        [6.78026009, 1.56051981, 8.12103939]])
        """
        return aikit.lars_update(
            self,
            dcdw,
            lr,
            decay_lambda=decay_lambda,
            stop_gradients=stop_gradients,
            out=out,
        )

    def adam_update(
        self: aikit.Array,
        dcdw: Union[aikit.Array, aikit.NativeArray],
        lr: Union[float, aikit.Array, aikit.NativeArray],
        mw_tm1: Union[aikit.Array, aikit.NativeArray],
        vw_tm1: Union[aikit.Array, aikit.NativeArray],
        step: int,
        /,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7,
        stop_gradients: bool = True,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.adam_update. This method
        simply wraps the function, and so the docstring for aikit.adam_update
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Weights of the function to be updated.
        dcdw
            Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
        lr
            Learning rate(s), the rate(s) at which the weights should be updated
            relative to the gradient.
        mw_tm1
            running average of the gradients, from the previous time-step.
        vw_tm1
            running average of second moments of the gradients, from the previous
            time-step.
        step
            training step.
        beta1
            gradient forgetting factor (Default value = 0.9).
        beta2
            second moment of gradient forgetting factor (Default value = 0.999).
        epsilon
            divisor during adam update, preventing division by zero
            (Default value = 1e-7).
        stop_gradients
            Whether to stop the gradients of the variables after each gradient step.
            Default is ``True``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The new function weights ws_new, and also new mw and vw, following the adam
            updates.

        Examples
        --------
        With :class:`aikit.Array` inputs:

        >>> w = aikit.array([1., 2, 3.])
        >>> dcdw = aikit.array([0.2,0.1,0.3])
        >>> lr = aikit.array(0.1)
        >>> vw_tm1 = aikit.zeros(1)
        >>> mw_tm1 = aikit.zeros(3)
        >>> step = 2
        >>> updated_weights = w.adam_update(dcdw, lr, mw_tm1, vw_tm1, step)
        >>> print(updated_weights)
        (aikit.array([0.92558753, 1.92558873, 2.92558718]),
        aikit.array([0.02, 0.01, 0.03]),
        aikit.array([4.00000063e-05, 1.00000016e-05, 9.00000086e-05]))
        """
        return aikit.adam_update(
            self,
            dcdw,
            lr,
            mw_tm1,
            vw_tm1,
            step,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            stop_gradients=stop_gradients,
            out=out,
        )

    def lamb_update(
        self: aikit.Array,
        dcdw: Union[aikit.Array, aikit.NativeArray],
        lr: Union[float, aikit.Array, aikit.NativeArray],
        mw_tm1: Union[aikit.Array, aikit.NativeArray],
        vw_tm1: Union[aikit.Array, aikit.NativeArray],
        step: int,
        /,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7,
        max_trust_ratio: Union[int, float] = 10,
        decay_lambda: float = 0,
        stop_gradients: bool = True,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.lamb_update. This method
        simply wraps the function, and so the docstring for aikit.lamb_update
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Weights of the function to be updated.
        dcdw
            Derivates of the cost c with respect to the weights ws, [dc/dw for w in ws].
        lr
            Learning rate(s), the rate(s) at which the weights should be updated
            relative to the gradient.
        mw_tm1
            running average of the gradients, from the previous time-step.
        vw_tm1
            running average of second moments of the gradients, from the previous
            time-step.
        step
            training step.
        beta1
            gradient forgetting factor (Default value = 0.9).
        beta2
            second moment of gradient forgetting factor (Default value = 0.999).
        epsilon
            divisor during adam update, preventing division by zero
            (Default value = 1e-7).
        max_trust_ratio
            The maximum value for the trust ratio. Default is 10.
        decay_lambda
            The factor used for weight decay. Default is zero.
        stop_gradients
            Whether to stop the gradients of the variables after each gradient step.
            Default is ``True``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The new function weights ws_new, following the LAMB updates.

        Examples
        --------
        With :class:`aikit.Array` inputs:

        >>> w = aikit.array([1., 2, 3])
        >>> dcdw = aikit.array([0.5,0.2,0.1])
        >>> lr = aikit.array(0.1)
        >>> vw_tm1 = aikit.zeros(1)
        >>> mw_tm1 = aikit.zeros(3)
        >>> step = aikit.array(1)
        >>> new_weights = w.lamb_update(dcdw, lr, mw_tm1, vw_tm1, step)
        >>> print(new_weights)
        (aikit.array([0.784, 1.78 , 2.78 ]),
        aikit.array([0.05, 0.02, 0.01]),
        aikit.array([2.5e-04, 4.0e-05, 1.0e-05]))
        """
        return aikit.lamb_update(
            self,
            dcdw,
            lr,
            mw_tm1,
            vw_tm1,
            step,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            max_trust_ratio=max_trust_ratio,
            decay_lambda=decay_lambda,
            stop_gradients=stop_gradients,
            out=out,
        )
