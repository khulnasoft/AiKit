# global
import abc
from typing import Optional, Union, Literal

# local
import aikit


# ToDo: implement all methods here as public instance methods


class _ArrayWithActivations(abc.ABC):
    def relu(
        self: aikit.Array,
        /,
        *,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.relu. This method simply
        wraps the function, and so the docstring for aikit.relu also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        complex_mode
            optional specifier for how to handle complex data types. See
            ``aikit.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array with the relu activation function applied element-wise.

        Examples
        --------
        >>> x = aikit.array([-1., 0., 1.])
        >>> y = x.relu()
        >>> print(y)
        aikit.array([0., 0., 1.])
        """
        return aikit.relu(self._data, complex_mode=complex_mode, out=out)

    def leaky_relu(
        self: aikit.Array,
        /,
        *,
        alpha: float = 0.2,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.leaky_relu. This method
        simply wraps the function, and so the docstring for aikit.leaky_relu also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        alpha
            the slope of the negative section.
        complex_mode
            optional specifier for how to handle complex data types. See
            ``aikit.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array with the leaky relu activation function applied element-wise.

        Examples
        --------
        >>> x = aikit.array([0.39, -0.85])
        >>> y = x.leaky_relu()
        >>> print(y)
        aikit.array([ 0.39, -0.17])
        """
        return aikit.leaky_relu(
            self._data, alpha=alpha, complex_mode=complex_mode, out=out
        )

    def gelu(
        self: aikit.Array,
        /,
        *,
        approximate: bool = False,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.gelu. This method simply
        wraps the function, and so the docstring for aikit.gelu also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        approximate
            whether to use the approximate version of the gelu function.
        complex_mode
            optional specifier for how to handle complex data types. See
            ``aikit.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array with the gelu activation function applied element-wise.

        Examples
        --------
        >>> x = aikit.array([-1.2, -0.6, 1.5])
        >>> y = x.gelu()
        >>> print(y)
        aikit.array([-0.138, -0.165, 1.4])
        """
        return aikit.gelu(
            self._data, approximate=approximate, complex_mode=complex_mode, out=out
        )

    def sigmoid(
        self: aikit.Array,
        /,
        *,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.sigmoid.

        This method simply wraps the function, and so the docstring for aikit.sigmoid also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array
        complex_mode
            optional specifier for how to handle complex data types. See
            ``aikit.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array for writing the result to. It must have the same shape
            the input broadcast to default: None

        Returns
        -------
        ret
            an array with the sigmoid activation function applied element-wise.


        Examples
        --------
        >>> x = aikit.array([-1., 1., 2.])
        >>> y = x.sigmoid()
        >>> print(y)
        aikit.array([0.269, 0.731, 0.881])
        """
        return aikit.sigmoid(self._data, complex_mode=complex_mode, out=out)

    def softmax(
        self: aikit.Array,
        /,
        *,
        axis: Optional[int] = None,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.softmax. This method simply
        wraps the function, and so the docstring for aikit.softmax also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        axis
            the axis or axes along which the softmax should be computed
        complex_mode
            optional specifier for how to handle complex data types. See
            ``aikit.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array with the softmax activation function applied element-wise.

        Examples
        --------
        >>> x = aikit.array([1.0, 0, 1.0])
        >>> y = x.softmax()
        >>> print(y)
        aikit.array([0.422, 0.155, 0.422])
        """
        return aikit.softmax(self._data, axis=axis, complex_mode=complex_mode, out=out)

    def softplus(
        self: aikit.Array,
        /,
        *,
        beta: Optional[Union[int, float]] = None,
        threshold: Optional[Union[int, float]] = None,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.softplus. This method
        simply wraps the function, and so the docstring for aikit.softplus also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        beta
            the beta parameter of the softplus function.
        threshold
            the threshold parameter of the softplus function.
        complex_mode
           optional specifier for how to handle complex data types. See
            ``aikit.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array, for writing the result to. It must have a shape

        Returns
        -------
        ret
            an array with the softplus activation function applied element-wise.

        Examples
        --------
        >>> x = aikit.array([-0.3461, -0.6491])
        >>> y = x.softplus()
        >>> print(y)
        aikit.array([0.535,0.42])

        >>> x = aikit.array([-0.3461, -0.6491])
        >>> y = x.softplus(beta=0.5)
        >>> print(y)
        aikit.array([1.22, 1.09])

        >>> x = aikit.array([1.31, 2., 2.])
        >>> y = x.softplus(threshold=2, out=x)
        >>> print(x)
        aikit.array([1.55, 2.13, 2.13])
        """
        return aikit.softplus(
            self._data,
            beta=beta,
            threshold=threshold,
            complex_mode=complex_mode,
            out=out,
        )

    def log_softmax(
        self: aikit.Array,
        /,
        *,
        axis: Optional[int] = -1,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.log_softmax. This method
        simply wraps the function, and so the docstring for aikit.log_softmax
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        axis
            the axis or axes along which the log_softmax should be computed
        complex_mode
            optional specifier for how to handle complex data types. See
            ``aikit.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array with the log_softmax activation function applied element-wise.

        Examples
        --------
        >>> x = aikit.array([-1.0, -0.98, 2.3])
        >>> y = x.log_softmax()
        >>> print(y)
        aikit.array([-3.37, -3.35, -0.0719])

        >>> x = aikit.array([2.0, 3.4, -4.2])
        >>> y = x.log_softmax(x)
        aikit.array([-1.62, -0.221, -7.82 ])
        """
        return aikit.log_softmax(
            self._data,
            axis=axis,
            complex_mode=complex_mode,
            out=out,
        )

    def mish(
        self: aikit.Array,
        /,
        *,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.mish. This method simply
        wraps the function, and so the docstring for aikit.mish also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        complex_mode
            optional specifier for how to handle complex data types. See
            ``aikit.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Examples
        --------
        >>> x = aikit.array([-1., 0., 1.])
        >>> y = x.mish()
        >>> print(y)
        aikit.array([-0.30340147,  0.        ,  0.86509842])
        """
        return aikit.mish(self._data, complex_mode=complex_mode, out=out)

    def hardswish(
        self: aikit.Array,
        /,
        *,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """Apply the hardswish activation function element-wise.

        Parameters
        ----------
        x
            input array
        complex_mode
            optional specifier for how to handle complex data types. See
            ``aikit.func_wrapper.handle_complex_input`` for more detail.
        out
            optional output array, for writing the result to. It must have
            a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the hardswish activation of each element in ``x``.

        Examples
        --------
        With :class:`aikit.Array` input:

        >>> x = aikit.array([0., 0., 4.])
        >>> y = aikit.hardswish(x)
        >>> y
        aikit.array([0., 0., 4.])

        With :class:`aikit.Container` input:

        >>> x = aikit.Container(a=aikit.array([-3., 4., 5.]), b=aikit.array([0., 5.]))
        >>> x = aikit.hardswish(x, out=x)
        >>> x
        {
            a: aikit.array([-0.,  4.,  5.]),
            b: aikit.array([0., 5.])
        }
        """
        return aikit.hardswish(self._data, complex_mode=complex_mode, out=out)
