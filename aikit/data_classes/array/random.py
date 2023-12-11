# global
import abc
from typing import Optional, Union

# local
import aikit


class _ArrayWithRandom(abc.ABC):
    def random_uniform(
        self: aikit.Array,
        /,
        *,
        high: Union[float, aikit.Array, aikit.NativeArray] = 1.0,
        shape: Optional[Union[aikit.Array, aikit.Shape, aikit.NativeShape]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        seed: Optional[int] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.random_uniform. This method
        simply wraps the function, and so the docstring for aikit.random_uniform
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Lower boundary of the output interval. All values generated will be
            greater than or equal to ``low``. If array, must have same shape as
            ``high``.
        high
            Upper boundary of the output interval. All the values generated will be
            less than ``high``. If array, must have same shape as ``low``.
        shape
            If the given shape is, e.g ``(m, n, k)``, then ``m * n * k`` samples
            are drawn. Can only be specified when ``low`` and ``high`` are numeric
            values, else exception will be raised.
            Default is ``None``, where a single value is returned.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            (Default value = None).
        dtype
            output array data type. If ``dtype`` is ``None``, the output array data
            type will be the default floating-point data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the parameterized uniform distribution.

        Examples
        --------
        >>> x = aikit.array([[9.8, 3.4], [5.8, 7.2]])
        >>> x.random_uniform(high=10.2)
        aikit.array([[9.86, 4.89],
                   [7.06, 7.47]])

        >>> x.random_uniform(high=10.2, device='cpu')
        aikit.array([[9.86, 4.89],
                   [7.06, 7.47]])

        >>> x.random_uniform(high=14.2, dtype='float16')
        aikit.array([[9.86, 4.89],
                   [7.06, 7.47]])

        >>> x.random_uniform(high=10.8, device='cpu', dtype='float64')
        aikit.array([[9.86, 4.89],
                   [7.06, 7.47]])

        >>> z = aikit.ones((2,2))
        >>> x.random_uniform(high=11.2, device='cpu', dtype='float64', out=z)
        aikit.array([[10.1 ,  6.53],
                   [ 7.94,  8.85]])

        >>> x = aikit.array([8.7, 9.3])
        >>> y = aikit.array([12.8, 14.5])
        >>> x.random_uniform(y)
        aikit.array([12.1, 14. ])

        >>> x.random_uniform(high=y, device='cpu')
        aikit.array([12.1, 14. ])

        >>> x.random_uniform(high=y, dtype='float16')
        aikit.array([12.1, 14. ])

        >>> x.random_uniform(high=y, device='cpu', dtype='float64')
        aikit.array([12.1, 14. ])

        >>> z = aikit.ones((2,))
        >>> x.random_uniform(high=y, device='cpu', dtype='float64', out=z)
        aikit.array([12.1, 14. ])
        """
        return aikit.random_uniform(
            low=self._data,
            high=high,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )

    def random_normal(
        self: aikit.Array,
        /,
        *,
        std: Union[float, aikit.Array, aikit.NativeArray] = 1.0,
        shape: Optional[Union[aikit.Shape, aikit.NativeShape]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        seed: Optional[int] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.random_normal. This method
        simply wraps the function, and so the docstring for aikit.random_normal
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            The mean of the normal distribution to sample from. Default is ``0.0``.
        std
            The standard deviation of the normal distribution to sample from.
            Must be non-negative. Default is ``1.0``.
        shape
            If the given shape is, e.g ``(m, n, k)``, then ``m * n * k`` samples
            are drawn. Can only be specified when ``mean`` and ``std`` are numeric
            values, else exception will be raised.
            Default is ``None``, where a single value is returned.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            (Default value = None).
        dtype
             output array data type. If ``dtype`` is ``None``, the output array data
             type will be the default floating-point data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the parameterized normal distribution.

        Examples
        --------
        >>> x = aikit.array([[9.8, 3.4], [5.8, 7.2]])
        >>> x.random_normal(std=10.2)
        aikit.array([[19.   , -6.44 ],
                   [ 5.72 ,  0.235]])

        >>> x.random_normal(std=10.2, device='cpu')
        aikit.array([[18.7 , 25.2 ],
                   [27.5 , -3.22]])

        >>> x.random_normal(std=14.2, dtype='float16')
        aikit.array([[26.6 , 12.1 ],
                   [ 4.56,  5.49]])

        >>> x.random_normal(std=10.8, device='cpu', dtype='float64')
        aikit.array([[ 1.02, -1.39],
                   [14.2 , -1.  ]])

        >>> z = aikit.ones((2,2))
        >>> x.random_normal(std=11.2, device='cpu', dtype='float64', out=z)
        aikit.array([[ 7.72, -8.32],
                   [ 4.95, 15.8 ]])

        >>> x = aikit.array([8.7, 9.3])
        >>> y = aikit.array([12.8, 14.5])
        >>> x.random_normal(std=y)
        aikit.array([-10.8,  12.1])

        >>> x.random_normal(std=y, device='cpu')
        aikit.array([ 13. , -26.9])

        >>> x.random_normal(std=y, dtype='float16')
        aikit.array([14.3  , -0.807])

        >>> x.random_normal(std=y, device='cpu', dtype='float64')
        aikit.array([21.3 ,  3.85])

        >>> z = aikit.ones((2,))
        >>> x.random_normal(std=y, device='cpu', dtype='float64', out=z)
        aikit.array([ 4.32, 42.2 ])
        """
        return aikit.random_normal(
            mean=self._data,
            std=std,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )

    def multinomial(
        self: aikit.Array,
        population_size: int,
        num_samples: int,
        /,
        *,
        batch_size: int = 1,
        replace: bool = True,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        seed: Optional[int] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.multinomial. This method
        simply wraps the function, and so the docstring for aikit.multinomial
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            The unnormalized probabilities for all elements in population,
            default is uniform *[batch_shape, population_size]*
        population_size
            The size of the population from which to draw samples.
        num_samples
            Number of independent samples to draw from the population.
        batch_size
            Number of tensors to generate. Default is 1.
        replace
            Whether to replace samples once they've been drawn. Default is ``True``.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            (Default value = None)
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the parameterized normal distribution.
        """
        return aikit.multinomial(
            population_size,
            num_samples,
            batch_size=batch_size,
            probs=self._data,
            replace=replace,
            device=device,
            seed=seed,
            out=out,
        )

    def randint(
        self: aikit.Array,
        high: Union[int, aikit.Array, aikit.NativeArray],
        /,
        *,
        shape: Optional[Union[aikit.Shape, aikit.NativeShape]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        seed: Optional[int] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.randint. This method simply
        wraps the function, and so the docstring for aikit.randint also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Lowest integer that can be drawn from the distribution.
        high
            One above the highest integer that can be drawn from the distribution.
        shape
            If the given shape is, e.g ``(m, n, k)``, then ``m * n * k`` samples
            are drawn. Can only be specified when ``low`` and ``high`` are numeric
            values, else exception will be raised.
            Default is ``None``, where a single value is returned.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            (Default value = None).
        dtype
             output array data type. If ``dtype`` is ``None``, the output array data
             type will be the default integer data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Returns an array with the given shape filled with integers from
            the uniform distribution in the “half-open” interval [low, high)

        Examples
        --------
        >>> x = aikit.array([[1, 2], [0, 5]])
        >>> x.randint(10)
        aikit.array([[1, 5],
                   [9, 7]])

        >>> x.randint(8, device='cpu')
        aikit.array([[6, 5],
                   [0, 5]])

        >>> x.randint(9, dtype='int8')
        aikit.array([[1, 2],
                   [7, 7]])

        >>> x.randint(14, device='cpu', dtype='int16')
        aikit.array([[6, 5],
                   [0, 5]])

        >>> z = aikit.ones((2,2))
        >>> x.randint(16, device='cpu', dtype='int64', out=z)
        aikit.array([[1, 2],
                   [7, 7]])

        >>> x = aikit.array([1, 2, 3])
        >>> y = aikit.array([23, 25, 98])
        >>> x.randint(y)
        aikit.array([ 5, 14, 18])

        >>> x.randint(y, device='cpu')
        aikit.array([20, 13, 46])

        >>> x.randint(y, dtype='int32')
        aikit.array([ 9, 18, 33])

        >>> x.randint(y, device='cpu', dtype='int16')
        aikit.array([ 9, 20, 85])

        >>> z = aikit.ones((3,))
        >>> x.randint(y, device='cpu', dtype='int64', out=z)
        aikit.array([20, 13, 46])
        """
        return aikit.randint(
            self._data,
            high,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )

    def shuffle(
        self: aikit.Array,
        axis: Optional[int] = 0,
        /,
        *,
        seed: Optional[int] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """aikit.Array instance method variant of aikit.shuffle. This method simply
        wraps the function, and so the docstring for aikit.shuffle also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input array. Should have a numeric data type.
        axis
            The axis which x is shuffled along. Default is 0.
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array object, shuffled along the first dimension.

        Examples
        --------
        >>> x = aikit.array([5, 2, 9])
        >>> y = x.shuffle()
        >>> print(y)
        aikit.array([2, 5, 9])
        """
        return aikit.shuffle(self, axis, seed=seed, out=out)
