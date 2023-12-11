Building Blocks
===============

Here we explain the components of Aikit which are fundamental to its usage either as a code converter or as a fully-fledged framework-agnostic ML framework.
These are the 4 parts labelled as (a) in the image below:

.. image:: https://github.com/khulnasoft/khulnasoft.github.io/blob/main/img/externally_linked/design/submodule_dependency_graph.png?raw=true
   :align: center
   :width: 100%

Backend Functional APIs ✅
--------------------------

The first important point to make is that, Aikit does not implement it’s own C++ or CUDA backend.
Instead, Aikit **wraps** the functional APIs of existing frameworks, bringing them into syntactic and semantic alignment.
Let’s take the function :func:`aikit.stack` as an example.

There are separate backend modules for JAX, TensorFlow, PyTorch, and NumPy, and so we implement the :code:`stack` method once for each backend, each in separate backend files like so:

.. code-block:: python

   # aikit/functional/backends/jax/manipulation.py:
    def stack(
        arrays: Union[Tuple[JaxArray], List[JaxArray]],
        /,
        *,
        axis: int = 0,
        out: Optional[JaxArray] = None,
    ) -> JaxArray:
        return jnp.stack(arrays, axis=axis)

.. code-block:: python

   # aikit/functional/backends/numpy/manipulation.py:
    def stack(
        arrays: Union[Tuple[np.ndarray], List[np.ndarray]],
        /,
        *,
        axis: int = 0,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return np.stack(arrays, axis, out=out)


    stack.support_native_out = True

.. code-block:: python

   # aikit/functional/backends/tensorflow/manipulation.py:
    def stack(
        arrays: Union[Tuple[tf.Tensor], List[tf.Tensor]],
        /,
        *,
        axis: int = 0,
        out: Optional[Union[tf.Tensor, tf.Variable]] = None,
    ) -> Union[tf.Tensor, tf.Variable]:
        return tf.experimental.numpy.stack(arrays, axis)

.. code-block:: python

   # aikit/functional/backends/torch/manipulation.py:
    def stack(
        arrays: Union[Tuple[torch.Tensor], List[torch.Tensor]],
        /,
        *,
        axis: int = 0,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.stack(arrays, axis, out=out)


    stack.support_native_out = True

There were no changes required for this function, however NumPy and PyTorch both had to be marked as supporting the :ref:`overview/deep_dive/inplace_updates:out argument` natively.

For more complicated functions, we need to do more than simply wrap and maybe change the name.
For functions with differing behavior then we must modify the function to fit the unified in-out behavior of Aikit’s API.
For example, the APIs of JAX, PyTorch, and NumPy all have a :code:`logspace` method, but TensorFlow does not at the time of writing.
Therefore, we need to construct it using a composition of existing TensorFlow ops like so:

.. code-block:: python

   # aikit/functional/backends/tensorflow/creation.py:
    def logspace(
        start: Union[tf.Tensor, tf.Variable, int],
        stop: Union[tf.Tensor, tf.Variable, int],
        num: int,
        base: float = 10.0,
        axis: Optional[int] = None,
        *,
        dtype: tf.DType,
        device: str,
    ) -> Union[tf.Tensor, tf.Variable]:
        power_seq = aikit.linspace(start, stop, num, axis, dtype=dtype, device=device)
        return base**power_seq

Aikit Functional API ✅
---------------------

Calling the different backend files explicitly would work okay, but it would mean we need to :code:`import aikit.functional.backends.torch as aikit` to use a PyTorch backend or :code:`import aikit.functional.backends.tensorflow as aikit` to use a TensorFlow backend.
Instead, we allow these backends to be bound to the single shared namespace aikit.
The backend can then be changed by calling :code:`aikit.set_backend('torch')` for example.

:mod:`aikit.functional.aikit` is the submodule where all the doc strings and argument typing reside for the functional Aikit API.
For example, the function :func:`prod`  is shown below:

.. code-block:: python

   # aikit/functional/aikit/elementwise.py:
    @to_native_arrays_and_back
    @handle_out_argument
    @handle_nestable
    def prod(
        x: Union[aikit.Array, aikit.NativeArray],
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        keepdims: bool = False,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        """Calculate the product of input array x elements.

        x
            input array. Should have a numeric data type.
        axis
            axis or axes along which products must be computed. By default, the product must
            be computed over the entire array. If a tuple of integers, products must be
            computed over multiple axes. Default: ``None``.
        keepdims
            bool, if True, the reduced axes (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the result must be compatible with the
            input array (see Broadcasting). Otherwise, if False, the reduced axes
            (dimensions) must not be included in the result. Default: ``False``.
        dtype
            data type of the returned array. If None,
            if the default data type corresponding to the data type “kind” (integer or
            floating-point) of x has a smaller range of values than the data type of x
            (e.g., x has data type int64 and the default data type is int32, or x has data
            type uint64 and the default data type is int64), the returned array must have
            the same data type as x. if x has a floating-point data type, the returned array
            must have the default floating-point data type. if x has a signed integer data
            type (e.g., int16), the returned array must have the default integer data type.
            if x has an unsigned integer data type (e.g., uint16), the returned array must
            have an unsigned integer data type having the same number of bits as the default
            integer data type (e.g., if the default integer data type is int32, the returned
            array must have a uint32 data type). If the data type (either specified or
            resolved) differs from the data type of x, the input array should be cast to the
            specified data type before computing the product. Default: ``None``.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            array,  if the product was computed over the entire array, a zero-dimensional
            array containing the product; otherwise, a non-zero-dimensional array containing
            the products. The returned array must have a data type as described by the dtype
            parameter above.

        >>> x = aikit.array([1, 2, 3])
        >>> z = aikit.prod(x)
        >>> print(z)
        aikit.array(6)

        >>> x = aikit.array([1, 0, 3])
        >>> z = aikit.prod(x)
        >>> print(z)
        aikit.array(0)

        """
        return current_backend(x).prod(
            x, axis=axis, dtype=dtype, keepdims=keepdims, out=out
        )

Implicitly, Aikit sets numpy as the default backend or operates with the backend corresponding to the specified data inputs
until the user explicitly sets a different backend.
The examples can be seen below:


+----------------------------------------+----------------------------------------------------+
|                                        |                                                    |
|.. code-block:: python                  |.. code-block:: python                              |
|                                        |                                                    |
|   # implicit                           |   # explicit                                       |
|   import aikit                           |   import aikit                                       |
|   x = aikit.array([1, 2, 3])             |   aikit.set_backend("jax")                           |
|   (type(aikit.to_native(x)))             |                                                    |
|   # -> <class 'numpy.ndarray'>         |   z = aikit.array([1, 2, 3]))                        |
|                                        |   type(aikit.to_native(z))                           |
|   import torch                         |   # ->  <class 'jaxlib.xla_extension.DeviceArray'> |
|   t = torch.tensor([23,42, -1])        |                                                    |
|   type(aikit.to_native(aikit.sum(t)))      |                                                    |
|   # -> <class 'torch.Tensor'>          |                                                    |
+----------------------------------------+----------------------------------------------------+

This implicit backend selection, and the use of a shared global aikit namespace for all backends, are both made possible via the backend handler.

Backend Handler ✅
------------------

All code for setting and unsetting the backend resides in the submodule at :mod:`aikit/utils/backend/handler.py`, and the front facing function is :func:`aikit.current_backend`.
The contents of this function are as follows:

.. code-block:: python

   # aikit/utils/backend/handler.py
    def current_backend(*args, **kwargs):
        global implicit_backend
        # if a global backend has been set with set_backend then this will be returned
        if backend_stack:
            f = backend_stack[-1]
            if verbosity.level > 0:
                verbosity.cprint(f"Using backend from stack: {f}")
            return f

        # if no global backend exists, we try to infer the backend from the arguments
        f = _determine_backend_from_args(list(args) + list(kwargs.values()))
        if f is not None:
            if verbosity.level > 0:
                verbosity.cprint(f"Using backend from type: {f}")
            implicit_backend = f.current_backend_str()
            return f
        return importlib.import_module(_backend_dict[implicit_backend])

If a global backend framework has been previously set using for example :code:`aikit.set_backend('tensorflow')`, then this globally set backend is returned.
Otherwise, the input arguments are type-checked to infer the backend, and this is returned from the function as a callable module with all bound functions adhering to the specific backend.

The functions in this returned module are populated by iterating through the global :attr:`aikit.__dict__` (or a non-global copy of :attr:`aikit.__dict__` if non-globally-set), and overwriting every function which is also directly implemented in the backend-specific namespace.
The following is a slightly simplified version of this code for illustration, which updates the global :attr:`aikit.__dict__` directly:

.. code-block:: python

   # aikit/utils/backend/handler.py
   def set_backend(backend: str):

       # un-modified aikit.__dict__
       global aikit_original_dict
       if not backend_stack:
           aikit_original_dict = aikit.__dict__.copy()

       # add the input backend to the global stack
       backend_stack.append(backend)

       # iterate through original aikit.__dict__
       for k, v in aikit_original_dict.items():

           # if method doesn't exist in the backend
           if k not in backend.__dict__:
               # add the original aikit method to backend
               backend.__dict__[k] = v
           # update global aikit.__dict__ with this method
           aikit.__dict__[k] = backend.__dict__[k]

       # maybe log to the terminal
       if verbosity.level > 0:
           verbosity.cprint(
               f'Backend stack: {backend_stack}'
            )

The functions implemented by the backend-specific backend such as :code:`aikit.functional.backends.torch` only constitute a subset of the full Aikit API.
This is because many higher level functions are written as a composition of lower level Aikit functions.
These functions therefore do not need to be written independently for each backend framework.
A good example is :func:`aikit.lstm_update`, as shown:

.. code-block:: python

    # aikit/functional/aikit/layers.py
    @to_native_arrays_and_back
    @handle_nestable
    def lstm_update(
        x: Union[aikit.Array, aikit.NativeArray],
        init_h: Union[aikit.Array, aikit.NativeArray],
        init_c: Union[aikit.Array, aikit.NativeArray],
        kernel: Union[aikit.Array, aikit.NativeArray],
        recurrent_kernel: Union[aikit.Array, aikit.NativeArray],
        bias: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
        recurrent_bias: Optional[Union[aikit.Array, aikit.NativeArray]] = None,
    ) -> Tuple[aikit.Array, aikit.Array]:
        """Perform long-short term memory update by unrolling time dimension of the input array.
        Parameters
        ----------
        x
            input tensor of LSTM layer *[batch_shape, t, in]*.
        init_h
            initial state tensor for the cell output *[batch_shape, out]*.
        init_c
            initial state tensor for the cell hidden state *[batch_shape, out]*.
        kernel
            weights for cell kernel *[in, 4 x out]*.
        recurrent_kernel
            weights for cell recurrent kernel *[out, 4 x out]*.
        bias
            bias for cell kernel *[4 x out]*. (Default value = None)
        recurrent_bias
            bias for cell recurrent kernel *[4 x out]*. (Default value = None)
        Returns
        -------
        ret
            hidden state for all timesteps *[batch_shape,t,out]* and cell state for last
            timestep *[batch_shape,out]*
        """
        # get shapes
        x_shape = list(x.shape)
        batch_shape = x_shape[:-2]
        timesteps = x_shape[-2]
        input_channels = x_shape[-1]
        x_flat = aikit.reshape(x, (-1, input_channels))

        # input kernel
        Wi = kernel
        Wi_x = aikit.reshape(
            aikit.matmul(x_flat, Wi) + (bias if bias is not None else 0),
            batch_shape + [timesteps, -1],
        )
        Wii_x, Wif_x, Wig_x, Wio_x = aikit.split(Wi_x, 4, -1)

        # recurrent kernel
        Wh = recurrent_kernel

        # lstm states
        ht = init_h
        ct = init_c

        # lstm outputs
        hts_list = []

        # unrolled time dimension with lstm steps
        for Wii_xt, Wif_xt, Wig_xt, Wio_xt in zip(
            aikit.unstack(Wii_x, axis=-2),
            aikit.unstack(Wif_x, axis=-2),
            aikit.unstack(Wig_x, axis=-2),
            aikit.unstack(Wio_x, axis=-2),
        ):
            htm1 = ht
            ctm1 = ct

            Wh_htm1 = aikit.matmul(htm1, Wh) + (
                recurrent_bias if recurrent_bias is not None else 0
            )
            Whi_htm1, Whf_htm1, Whg_htm1, Who_htm1 = aikit.split(
                Wh_htm1, num_or_size_splits=4, axis=-1
            )

            it = aikit.sigmoid(Wii_xt + Whi_htm1)
            ft = aikit.sigmoid(Wif_xt + Whf_htm1)
            gt = aikit.tanh(Wig_xt + Whg_htm1)
            ot = aikit.sigmoid(Wio_xt + Who_htm1)
            ct = ft * ctm1 + it * gt
            ht = ot * aikit.tanh(ct)

            hts_list.append(aikit.expand_dims(ht, -2))

        return aikit.concat(hts_list, -2), ct

We *could* find and wrap the functional LSTM update methods for each backend framework which might bring a small performance improvement, but in this case there are no functional LSTM methods exposed in the official functional APIs of the backend frameworks, and therefore the functional LSTM code which does exist for the backends is much less stable and less reliable for wrapping into Aikit.
Generally, we have made decisions so that Aikit is as stable and scalable as possible, minimizing dependencies to backend framework code where possible with minimal sacrifices in performance.

Tracer 🚧
-----------------

“What about performance?” I hear you ask.
This is a great point to raise!

With the design as currently presented, there would be a small performance hit every time we call an Aikit function by virtue of the added Python wrapping.
One reason we created the tracer was to address this issue.

The tracer takes in any Aikit function, backend function, or composition, and returns the computation graph using the backend functional API only.
The dependency graph for this process looks like this:

.. image:: https://github.com/khulnasoft/khulnasoft.github.io/blob/main/img/externally_linked/design/compiler_dependency_graph.png?raw=true
   :align: center
   :width: 75%

Let's look at a few examples, and observe the traced graph of the Aikit code against the native backend code.
First, let's set our desired backend as PyTorch.
When we trace the three functions below, despite the fact that each
has a different mix of Aikit and PyTorch code, they all trace to the same graph:

+----------------------------------------+-----------------------------------------+-----------------------------------------+
|.. code-block:: python                  |.. code-block:: python                   |.. code-block:: python                   |
|                                        |                                         |                                         |
| def pure_aikit(x):                       | def pure_torch(x):                      | def mix(x):                             |
|     y = aikit.mean(x)                    |     y = torch.mean(x)                   |     y = aikit.mean(x)                     |
|     z = aikit.sum(x)                     |     z = torch.sum(x)                    |     z = torch.sum(x)                    |
|     f = aikit.var(y)                     |     f = torch.var(y)                    |     f = aikit.var(y)                      |
|     k = aikit.cos(z)                     |     k = torch.cos(z)                    |     k = torch.cos(z)                    |
|     m = aikit.sin(f)                     |     m = torch.sin(f)                    |     m = aikit.sin(f)                      |
|     o = aikit.tan(y)                     |     o = torch.tan(y)                    |     o = torch.tan(y)                    |
|     return aikit.concatenate(            |     return torch.cat(                   |     return aikit.concatenate(             |
|         [k, m, o], -1)                 |         [k, m, o], -1)                  |         [k, m, o], -1)                  |
|                                        |                                         |                                         |
| # input                                | # input                                 | # input                                 |
| x = aikit.array([[1., 2., 3.]])          | x = torch.tensor([[1., 2., 3.]])        | x = aikit.array([[1., 2., 3.]])           |
|                                        |                                         |                                         |
| # create graph                         | # create graph                          | # create graph                          |
| graph = aikit.trace_graph(               | graph = aikit.trace_graph(                | graph = aikit.trace_graph(                |
|     pure_aikit, x)                       |     pure_torch, x)                      |     mix, x)                             |
|                                        |                                         |                                         |
| # call graph                           | # call graph                            | # call graph                            |
| ret = graph(x)                         | ret = graph(x)                          | ret = graph(x)                          |
+----------------------------------------+-----------------------------------------+-----------------------------------------+

.. image:: https://github.com/khulnasoft/khulnasoft.github.io/blob/main/img/externally_linked/design/compiled_graph_a.png?raw=true
   :align: center
   :width: 75%

For all existing ML frameworks, the functional API is the backbone that underpins all higher level functions and classes.
This means that under the hood, any code can be expressed as a composition of ops in the functional API.
The same is true for Aikit.
Therefore, when compiling the graph with Aikit, any higher-level classes or extra code which does not directly contribute towards the computation graph is excluded.
For example, the following 3 pieces of code all result in the exact same computation graph when traced as shown:

+----------------------------------------+-----------------------------------------+-----------------------------------------+
|.. code-block:: python                  |.. code-block:: python                   |.. code-block:: python                   |
|                                        |                                         |                                         |
| class Network(aikit.module)              | def clean(x, w, b):                     | def unclean(x, w, b):                   |
|                                        |     return w*x + b                      |     y = b + w + x                       |
|     def __init__(self):                |                                         |     print('message')                    |
|         self._layer = aikit.Linear(3, 3) |                                         |     wx = w * x                          |
|         super().__init__()             |                                         |     ret = wx + b                        |
|                                        |                                         |     temp = y * wx                       |
|     def _forward(self, x):             |                                         |     return ret                          |
|         return self._layer(x)          |                                         |                                         |
|                                        | # input                                 | # input                                 |
| # build network                        | x = aikit.array([1., 2., 3.])             | x = aikit.array([1., 2., 3.])             |
| net = Network()                        | w = aikit.random_uniform(                 | w = aikit.random_uniform(                 |
|                                        |     -1, 1, (3, 3))                      |     -1, 1, (3, 3))                      |
| # input                                | b = aikit.zeros((3,))                     | b = aikit.zeros((3,))                     |
| x = aikit.array([1., 2., 3.])            |                                         |                                         |
|                                        | # trace graph                           | # trace graph                           |
| # trace graph                          | graph = aikit.trace_graph(                | graph = aikit.trace_graph(                |
| net.trace_graph(x)                     |     clean, x, w, b)                     |     unclean, x, w, b)                   |
|                                        |                                         |                                         |
| # execute graph                        | # execute graph                         | # execute graph                         |
| net(x)                                 | graph(x, w, b)                          | graph(x, w, b)                          |
+----------------------------------------+-----------------------------------------+-----------------------------------------+

.. image:: https://github.com/khulnasoft/khulnasoft.github.io/blob/main/img/externally_linked/design/compiled_graph_b.png?raw=true
   :align: center
   :width: 75%

This tracing is not restricted to just PyTorch.
Let's take another example, but trace to Tensorflow, NumPy, and JAX:

+------------------------------------+
|.. code-block:: python              |
|                                    |
| def aikit_func(x, y):                |
|     w = aikit.diag(x)                |
|     z = aikit.matmul(w, y)           |
|     return z                       |
|                                    |
| # input                            |
| x = aikit.array([[1., 2., 3.]])      |
| y = aikit.array([[2., 3., 4.]])      |
| # create graph                     |
| graph = aikit.trace_graph(           |
|     aikit_func, x, y)                |
|                                    |
| # call graph                       |
| ret = graph(x, y)                  |
+------------------------------------+

Converting this code to a graph, we get a slightly different graph for each backend:

Tensorflow:

.. image:: https://github.com/khulnasoft/khulnasoft.github.io/blob/main/img/externally_linked/design/compiled_graph_tf.png?raw=true
   :align: center
   :width: 75%

|

Numpy:

.. image:: https://github.com/khulnasoft/khulnasoft.github.io/blob/main/img/externally_linked/design/compiled_graph_numpy.png?raw=true
   :align: center
   :width: 75%

|

Jax:

.. image:: https://github.com/khulnasoft/khulnasoft.github.io/blob/main/img/externally_linked/design/compiled_graph_jax.png?raw=true
   :align: center
   :width: 75%
|

The example above further emphasizes that the tracer creates a computation graph consisting of backend functions, not Aikit functions.
Specifically, the same Aikit code is traced to different graphs depending on the selected backend.
However, when compiling native framework code, we are only able to trace a graph for that same framework.
For example, we cannot take torch code and trace this into tensorflow code.
However, we can transpile torch code into tensorflow code (see `Aikit as a Transpiler <aikit_as_a_transpiler.rst>`_ for more details).

The tracer is not a compiler and does not compile to C++, CUDA, or any other lower level language.
It simply traces the backend functional methods in the graph, stores this graph, and then efficiently traverses this graph at execution time, all in Python.
Compiling to lower level languages (C++, CUDA, TorchScript etc.) is supported for most backend frameworks via :func:`aikit.compile`, which wraps backend-specific compilation code, for example:

.. code-block:: python

    # aikit/functional/backends/tensorflow/compilation.py
    compile = lambda fn, dynamic=True, example_inputs=None,\
    static_argnums=None, static_argnames=None:\
        tf.function(fn)

.. code-block:: python

    # aikit/functional/backends/torch/compilation.py
    def compile(fn, dynamic=True, example_inputs=None,
            static_argnums=None, static_argnames=None):
    if dynamic:
        return torch.jit.script(fn)
    return torch.jit.trace(fn, example_inputs)

.. code-block:: python

    # aikit/functional/backends/jax/compilation.py
    compile = lambda fn, dynamic=True, example_inputs=None,\
                static_argnums=None, static_argnames=None:\
    jax.jit(fn, static_argnums=static_argnums,
            static_argnames=static_argnames)

Therefore, the backend code can always be run with maximal efficiency by compiling into an efficient low-level backend-specific computation graph.

**Round Up**

Hopefully, this has painted a clear picture of the fundamental building blocks underpinning the Aikit framework, being the Backend functional APIs, Aikit functional API, Backend handler, and Tracer 😄

Please reach out on `discord <https://discord.gg/sXyFF8tDtm>`_ if you have any questions!
