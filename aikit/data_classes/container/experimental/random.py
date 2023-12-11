# global
from typing import Optional, Union, List, Dict

# local
import aikit
from aikit.data_classes.container.base import ContainerBase


class _ContainerWithRandomExperimental(ContainerBase):
    # dirichlet
    @staticmethod
    def static_dirichlet(
        alpha: aikit.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        size: Optional[Union[aikit.Shape, aikit.NativeShape, aikit.Container]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        seed: Optional[Union[int, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.dirichlet. This method
        simply wraps the function, and so the docstring for aikit.dirichlet also
        applies to this method with minimal changes.

        Parameters
        ----------
        alpha
            Sequence of floats of length k
        size
            optional container including ints or tuple of ints,
            Output shape for the arrays in the input container.
        dtype
            output container array data type. If ``dtype`` is ``None``, the output data
            type will be the default floating-point data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including the drawn samples.

        Examples
        --------
        >>> alpha = aikit.Container(a=aikit.array([7,6,5]), \
                                  b=aikit.array([8,9,4]))
        >>> size = aikit.Container(a=3, b=5)
        >>> aikit.Container.static_dirichlet(alpha, size)
        {
            a: aikit.array(
                [[0.43643127, 0.32325703, 0.24031169],
                 [0.34251311, 0.31692529, 0.3405616 ],
                 [0.5319725 , 0.22458365, 0.24344385]]
                ),
            b: aikit.array(
                [[0.26588406, 0.61075421, 0.12336174],
                 [0.51142915, 0.25041268, 0.23815817],
                 [0.64042903, 0.25763214, 0.10193883],
                 [0.31624692, 0.46567987, 0.21807321],
                 [0.37677699, 0.39914594, 0.22407707]]
                )
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "dirichlet",
            alpha,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            size=size,
            dtype=dtype,
            out=out,
        )

    def dirichlet(
        self: aikit.Container,
        /,
        *,
        size: Optional[Union[aikit.Shape, aikit.NativeShape, aikit.Container]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        seed: Optional[Union[int, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.dirichlet. This method
        simply wraps the function, and so the docstring for aikit.shuffle also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Sequence of floats of length k
        size
            optional container including ints or tuple of ints,
            Output shape for the arrays in the input container.
        dtype
            output container array data type. If ``dtype`` is ``None``, the output data
            type will be the default floating-point data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including the drawn samples.

        Examples
        --------
        >>> alpha = aikit.Container(a=aikit.array([7,6,5]), \
                                  b=aikit.array([8,9,4]))
        >>> size = aikit.Container(a=3, b=5)
        >>> alpha.dirichlet(size)
        {
            a: aikit.array(
                [[0.43643127, 0.32325703, 0.24031169],
                 [0.34251311, 0.31692529, 0.3405616 ],
                 [0.5319725 , 0.22458365, 0.24344385]]
                ),
            b: aikit.array(
                [[0.26588406, 0.61075421, 0.12336174],
                 [0.51142915, 0.25041268, 0.23815817],
                 [0.64042903, 0.25763214, 0.10193883],
                 [0.31624692, 0.46567987, 0.21807321],
                 [0.37677699, 0.39914594, 0.22407707]]
                )
        }
        """
        return self.static_dirichlet(
            self,
            size=size,
            dtype=dtype,
            out=out,
        )

    @staticmethod
    def static_beta(
        alpha: aikit.Container,
        beta: Union[int, float, aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        shape: Optional[Union[aikit.Shape, aikit.NativeShape, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        device: Optional[Union[str, aikit.Container]] = None,
        dtype: Optional[Union[str, aikit.Container]] = None,
        seed: Optional[Union[int, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.beta. This method simply
        wraps the function, and so the docstring for aikit.beta also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            Input array or container. Should have a numeric data type.
        alpha
            The alpha parameter of the distribution.
        beta
            The beta parameter of the distribution.
        shape
            The shape of the output array. Default is ``None``.
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
        device
            The device to place the output array on. Default is ``None``.
        dtype
            The data type of the output array. Default is ``None``.
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container object, with values drawn from the beta distribution.
        """
        return ContainerBase.cont_multi_map_in_function(
            "beta",
            alpha,
            beta,
            shape=shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )

    def beta(
        self: aikit.Container,
        beta: Union[int, float, aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        shape: Optional[Union[aikit.Shape, aikit.NativeShape, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        device: Optional[Union[str, aikit.Container]] = None,
        dtype: Optional[Union[str, aikit.Container]] = None,
        seed: Optional[Union[int, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.beta. This method
        simply wraps the function, and so the docstring for aikit.beta also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container. Should have a numeric data type.
        alpha
            The alpha parameter of the distribution.
        beta
            The beta parameter of the distribution.
        shape
            The shape of the output array. Default is ``None``.
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
        device
            The device to place the output array on. Default is ``None``.
        dtype
            The data type of the output array. Default is ``None``.
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container object, with values drawn from the beta distribution.
        """
        return self.static_beta(
            self,
            beta,
            shape=shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )

    @staticmethod
    def static_poisson(
        lam: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        shape: Optional[Union[aikit.Shape, aikit.NativeShape, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        seed: Optional[Union[int, aikit.Container]] = None,
        fill_value: Optional[Union[float, int, aikit.Container]] = 0,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container static method variant of aikit.poisson. This method
        simply wraps the function, and so the docstring for aikit.poisson also
        applies to this method with minimal changes.

        Parameters
        ----------
        lam
            Input container with rate parameter(s) describing the poisson
            distribution(s) to sample.
        shape
            optional container including ints or tuple of ints,
            Output shape for the arrays in the input container.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            (Default value = None).
        dtype
            output container array data type. If ``dtype`` is ``None``, the output data
            type will be the default floating-point data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution.
        fill_value
            if lam is negative, fill the output array with this value
            on that specific dimension.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including the drawn samples.

        Examples
        --------
        >>> lam = aikit.Container(a=aikit.array([7,6,5]), \
                                b=aikit.array([8,9,4]))
        >>> shape = aikit.Container(a=(2,3), b=(1,1,3))
        >>> aikit.Container.static_poisson(lam, shape=shape)
        {
            a: aikit.array([[5, 4, 6],
                          [12, 4, 5]]),
            b: aikit.array([[[8, 13, 3]]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "poisson",
            lam,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            fill_value=fill_value,
            out=out,
        )

    def poisson(
        self: aikit.Container,
        /,
        *,
        shape: Optional[Union[aikit.Shape, aikit.NativeShape, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        seed: Optional[Union[int, aikit.Container]] = None,
        fill_value: Optional[Union[float, int, aikit.Container]] = 0,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """aikit.Container instance method variant of aikit.poisson. This method
        simply wraps the function, and so the docstring for aikit.poisson also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container with rate parameter(s) describing the poisson
            distribution(s) to sample.
        shape
            optional container including ints or tuple of ints,
            Output shape for the arrays in the input container.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            (Default value = None).
        dtype
            output container array data type. If ``dtype`` is ``None``, the output data
            type will be the default floating-point data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution.
        fill_value
            if lam is negative, fill the output array with this value
            on that specific dimension.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including the drawn samples.

        Examples
        --------
        >>> lam = aikit.Container(a=aikit.array([7,6,5]), \
                                b=aikit.array([8,9,4]))
        >>> shape = aikit.Container(a=(2,3), b=(1,1,3))
        >>> lam.poisson(shape=shape)
        {
            a: aikit.array([[5, 4, 6],
                          [12, 4, 5]]),
            b: aikit.array([[[8, 13, 3]]])
        }
        """
        return self.static_poisson(
            self,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            fill_value=fill_value,
            out=out,
        )

    @staticmethod
    def static_bernoulli(
        probs: aikit.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        logits: Optional[
            Union[float, aikit.Array, aikit.NativeArray, aikit.Container]
        ] = None,
        shape: Optional[Union[aikit.Shape, aikit.NativeShape, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        seed: Optional[Union[int, aikit.Container]] = None,
        out: Optional[Union[aikit.Array, aikit.Container]] = None,
    ) -> aikit.Container:
        """

        Parameters
        ----------
        probs
             An N-D Array representing the probability of a 1 event.
             Each entry in the Array parameterizes an independent Bernoulli
             distribution. Only one of logits or probs should be passed in
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
        logits
            An N-D Array representing the log-odds of a 1 event.
            Each entry in the Array parameterizes an independent Bernoulli
            distribution where the probability of an event is sigmoid
            (logits). Only one of logits or probs should be passed in.
        shape
            If the given shape is, e.g '(m, n, k)', then 'm * n * k' samples are drawn.
            (Default value = 'None', where 'aikit.shape(logits)' samples are drawn)
        device
            The device to place the output array on. Default is ``None``.
        dtype
            The data type of the output array. Default is ``None``.
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the Bernoulli distribution

        """
        return ContainerBase.cont_multi_map_in_function(
            "bernoulli",
            probs,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            logits=logits,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )

    def bernoulli(
        self: aikit.Container,
        /,
        *,
        logits: Optional[
            Union[float, aikit.Array, aikit.NativeArray, aikit.Container]
        ] = None,
        shape: Optional[Union[aikit.Shape, aikit.NativeShape, aikit.Container]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice, aikit.Container]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype, aikit.Container]] = None,
        seed: Optional[Union[int, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ) -> aikit.Container:
        """

        Parameters
        ----------
        self
             An N-D Array representing the probability of a 1 event.
             Each entry in the Array parameterizes an independent
             Bernoulli distribution. Only one of logits or probs should
             be passed in.
        logits
            An N-D Array representing the log-odds of a 1 event.
            Each entry in the Array parameterizes an independent Bernoulli
            distribution where the probability of an event is
            sigmoid(logits). Only one of logits or probs should be passed in.

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
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the Bernoulli distribution

        """
        return self.static_bernoulli(
            self,
            logits=logits,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )

    @staticmethod
    def static_gamma(
        alpha: aikit.Container,
        beta: Union[int, float, aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        shape: Optional[Union[aikit.Shape, aikit.NativeShape, aikit.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], aikit.Container]] = None,
        to_apply: Union[bool, aikit.Container] = True,
        prune_unapplied: Union[bool, aikit.Container] = False,
        map_sequences: Union[bool, aikit.Container] = False,
        device: Optional[Union[str, aikit.Container]] = None,
        dtype: Optional[Union[str, aikit.Container]] = None,
        seed: Optional[Union[int, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ):
        """aikit.Container static method variant of aikit.gamma. This method simply
        wraps the function, and so the docstring for aikit.gamma also applies to
        this method with minimal changes.

        Parameters
        ----------
        alpha
            First parameter of the distribution.
        beta
            Second parameter of the distribution.
        shape
            If the given shape is, e.g '(m, n, k)', then 'm * n * k' samples are drawn.
            (Default value = 'None', where 'aikit.shape(logits)' samples are drawn)
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
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc.
            (Default value = None).

        dtype
            output array data type. If ``dtype`` is ``None``, the output array data
            type will be the default floating-point data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution

        out
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the parameterized gamma distribution with the shape of
            the input Container.
        """
        return ContainerBase.cont_multi_map_in_function(
            "gamma",
            alpha,
            beta,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )

    def gamma(
        self: aikit.Container,
        beta: Union[int, float, aikit.Container, aikit.Array, aikit.NativeArray],
        /,
        *,
        shape: Optional[Union[aikit.Shape, aikit.NativeShape, aikit.Container]] = None,
        device: Optional[Union[str, aikit.Container]] = None,
        dtype: Optional[Union[str, aikit.Container]] = None,
        seed: Optional[Union[int, aikit.Container]] = None,
        out: Optional[aikit.Container] = None,
    ):
        """aikit.Container method variant of aikit.gamma. This method simply wraps
        the function, and so the docstring for aikit.gamma also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            First parameter of the distribution.
        beta
            Second parameter of the distribution.
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
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Drawn samples from the parameterized gamma distribution with the shape of
            the input Container.
        """
        return self.static_gamma(
            self,
            beta,
            shape=shape,
            device=device,
            dtype=dtype,
            seed=seed,
            out=out,
        )
