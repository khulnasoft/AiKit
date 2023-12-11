# global
import abc
from typing import Optional, Union

# local
import aikit


class _ArrayWithRandomExperimental(abc.ABC):
    def dirichlet(
        self: aikit.Array,
        /,
        *,
        size: Optional[Union[aikit.Shape, aikit.NativeShape]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        seed: Optional[int] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.dirichlet. This method
        simply wraps the function, and so the docstring for aikit.shuffle also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Sequence of floats of length k
        size
            optional int or tuple of ints, Output shape. If the given shape is,
            e.g., (m, n), then m * n * k samples are drawn. Default is None,
            in which case a vector of length k is returned.
        dtype
            output array data type. If ``dtype`` is ``None``, the output array data
            type will be the default floating-point data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            The drawn samples, of shape (size, k).

        Examples
        --------
        >>> alpha = aikit.array([1.0, 2.0, 3.0])
        >>> alpha.dirichlet()
        aikit.array([0.10598304, 0.21537054, 0.67864642])

        >>> alpha = aikit.array([1.0, 2.0, 3.0])
        >>> alpha.dirichlet(size = (2,3))
        aikit.array([[[0.48006698, 0.07472073, 0.44521229],
            [0.55479872, 0.05426367, 0.39093761],
            [0.19531053, 0.51675832, 0.28793114]],

        [[0.12315625, 0.29823365, 0.5786101 ],
            [0.15564976, 0.50542368, 0.33892656],
            [0.1325352 , 0.44439589, 0.42306891]]])
        """
        return aikit.dirichlet(self, size=size, dtype=dtype, seed=seed, out=out)

    def beta(
        self: aikit.Array,
        beta: Union[int, aikit.Array, aikit.NativeArray],
        /,
        *,
        shape: Optional[Union[aikit.Shape, aikit.NativeShape]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        seed: Optional[int] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.beta. This method simply
        wraps the function, and so the docstring for aikit.beta also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Input Array.
        alpha
            The first parameter of the beta distribution.
        beta
            The second parameter of the beta distribution.
        device
            device on which to create the array.
        dtype
             output array data type. If ``dtype`` is ``None``, the output array data
             type will be the default data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the parameterized beta distribution with the shape of
            the array.
        """
        return aikit.beta(
            self,
            beta,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )

    def gamma(
        self: aikit.Array,
        beta: Union[int, aikit.Array, aikit.NativeArray],
        /,
        *,
        shape: Optional[Union[aikit.Shape, aikit.NativeShape]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        seed: Optional[int] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.gamma. This method simply
        wraps the function, and so the docstring for aikit.gamma also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Input Array and the first parameter of the gamma distribution.
        beta
            The second parameter of the gamma distribution.
        shape
            If the given shape is, e.g '(m, n, k)', then 'm * n * k' samples are drawn.
            (Default value = 'None', where 'aikit.shape(logits)' samples are drawn)
        device
            device on which to create the array.
        dtype
             output array data type. If ``dtype`` is ``None``, the output array data
             type will be the default data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the parameterized gamma distribution with the shape of
            the input array.
        """
        return aikit.gamma(
            self,
            beta,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )

    def poisson(
        self: aikit.Array,
        *,
        shape: Optional[Union[aikit.Shape, aikit.NativeShape]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        seed: Optional[int] = None,
        fill_value: Optional[Union[float, int]] = 0,
        out: Optional[aikit.Array] = None,
    ):
        """
        Parameters
        ----------
        self
            Input Array of rate parameter(s). It must have a shape that is broadcastable
            to the requested shape
        shape
            If the given shape is, e.g '(m, n, k)', then 'm * n * k' samples are drawn.
            (Default value = 'None', where 'aikit.shape(lam)' samples are drawn)
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            (Default value = None).
        dtype
            output array data type. If ``dtype`` is ``None``, the output array data
            type will be the default floating-point data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        fill_value
            if lam is negative, fill the output array with this value
            on that specific dimension.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the parameterized poisson distribution.

        Examples
        --------
        >>> lam = aikit.array([1.0, 2.0, 3.0])
        >>> lam.poisson()
        aikit.array([1., 4., 4.])

        >>> lam = aikit.array([1.0, 2.0, 3.0])
        >>> lam.poisson(shape=(2,3))
        aikit.array([[0., 2., 2.],
                   [1., 2., 3.]])
        """
        return aikit.poisson(
            self,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            fill_value=fill_value,
            out=out,
        )

    def bernoulli(
        self: aikit.Array,
        *,
        logits: Optional[Union[float, aikit.Array, aikit.NativeArray]] = None,
        shape: Optional[Union[aikit.Shape, aikit.NativeShape]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        seed: Optional[int] = None,
        out: Optional[aikit.Array] = None,
    ):
        """

        Parameters
        ----------
        self
             An N-D Array representing the probability of a 1 event.
             Each entry in the Array parameterizes an independent Bernoulli
             distribution. Only one of logits or probs should be passed in
        logits
            An N-D Array representing the log-odds of a 1 event.
            Each entry in the Array parameterizes an independent Bernoulli
            distribution where the probability of an event is sigmoid
            (logits). Only one of logits or probs should be passed in.

        shape
            If the given shape is, e.g '(m, n, k)', then 'm * n * k' samples are drawn.
            (Default value = 'None', where 'aikit.shape(logits)' samples are drawn)
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            (Default value = None).

        dtype
            output array data type. If ``dtype`` is ``None``, the output array data
            type will be the default floating-point data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the Bernoulli distribution
        """
        return aikit.bernoulli(
            self,
            logits=logits,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )
