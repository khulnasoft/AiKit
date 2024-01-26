Devices
=======

.. _`backend setting`: https://github.com/khulnasoft/aikit/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/aikit/backend_handler.py#L204
.. _`infer_device`: https://github.com/khulnasoft/aikit/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/aikit/func_wrapper.py#L286
.. _`aikit.Device`: https://github.com/khulnasoft/aikit/blob/0b89c7fa050db13ef52b0d2a3e1a5fb801a19fa2/aikit/__init__.py#L42
.. _`empty class`: https://github.com/khulnasoft/aikit/blob/0b89c7fa050db13ef52b0d2a3e1a5fb801a19fa2/aikit/__init__.py#L34
.. _`device class`: https://github.com/khulnasoft/aikit/blob/0b89c7fa050db13ef52b0d2a3e1a5fb801a19fa2/aikit/functional/backends/torch/__init__.py#L13
.. _`device.py`: https://github.com/khulnasoft/aikit/blob/08ebc4d6d5e200dcbb8498b213538ffd550767f3/aikit/functional/aikit/device.py
.. _`aikit.total_mem_on_dev`: https://github.com/khulnasoft/aikit/blob/08ebc4d6d5e200dcbb8498b213538ffd550767f3/aikit/functional/aikit/device.py#L460
.. _`aikit.dev_util`: https://github.com/khulnasoft/aikit/blob/08ebc4d6d5e200dcbb8498b213538ffd550767f3/aikit/functional/aikit/device.py#L600
.. _`aikit.num_cpu_cores`: https://github.com/khulnasoft/aikit/blob/08ebc4d6d5e200dcbb8498b213538ffd550767f3/aikit/functional/aikit/device.py#L659
.. _`aikit.default_device`: https://github.com/khulnasoft/aikit/blob/08ebc4d6d5e200dcbb8498b213538ffd550767f3/aikit/functional/aikit/device.py#L720
.. _`aikit.set_soft_device_mode`: https://github.com/khulnasoft/aikit/blob/afca97b95d7101c45fa647b308fc8c41f97546e3/aikit/functional/aikit/device.py#L292
.. _`@handle_device_shifting`: https://github.com/khulnasoft/aikit/blob/afca97b95d7101c45fa647b308fc8c41f97546e3/aikit/func_wrapper.py#L797
.. _`aikit.functional.aikit`: https://github.com/khulnasoft/aikit/tree/afca97b95d7101c45fa647b308fc8c41f97546e3/aikit/functional/aikit
.. _`tensorflow soft device handling function`: https://github.com/khulnasoft/aikit/blob/afca97b95d7101c45fa647b308fc8c41f97546e3/aikit/functional/backends/tensorflow/device.py#L102
.. _`numpy soft device handling function`: https://github.com/khulnasoft/aikit/blob/afca97b95d7101c45fa647b308fc8c41f97546e3/aikit/functional/backends/numpy/device.py#L88
.. _`aikit implementation`: https://github.com/khulnasoft/aikit/blob/afca97b95d7101c45fa647b308fc8c41f97546e3/aikit/functional/aikit/device.py#L138
.. _`tf.device`: https://www.tensorflow.org/api_docs/python/tf/device
.. _`aikit.DefaultDevice`: https://github.com/khulnasoft/aikit/blob/afca97b95d7101c45fa647b308fc8c41f97546e3/aikit/functional/aikit/device.py#L52
.. _`__enter__`: https://github.com/khulnasoft/aikit/blob/afca97b95d7101c45fa647b308fc8c41f97546e3/aikit/functional/aikit/device.py#L76
.. _`__exit__`: https://github.com/khulnasoft/aikit/blob/afca97b95d7101c45fa647b308fc8c41f97546e3/aikit/functional/aikit/device.py#L98
.. _`aikit.unset_soft_device_mode()`: https://github.com/khulnasoft/aikit/blob/2f90ce7b6a4c8ddb7227348d58363cd2a3968602/aikit/functional/aikit/device.py#L317
.. _`aikit.unset_default_device()`: https://github.com/khulnasoft/aikit/blob/2f90ce7b6a4c8ddb7227348d58363cd2a3968602/aikit/functional/aikit/device.py#L869
.. _`repo`: https://github.com/khulnasoft/aikit
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`devices thread`: https://discord.com/channels/799879767196958751/1189906353653817354

The devices currently supported by Aikit are as follows:

* cpu
* gpu:idx
* tpu:idx

In a similar manner to the :class:`aikit.Dtype` and :class:`aikit.NativeDtype` classes (see `Data Types <data_types.rst>`_), there is both an `aikit.Device`_ class and an :class:`aikit.NativeDevice` class, with :class:`aikit.NativeDevice` initially set as an `empty class`_.
The :class:`aikit.Device` class derives from :code:`str`, and has simple logic in the constructor to verify that the string formatting is correct.
When a backend is set, the :class:`aikit.NativeDevice` is replaced with the backend-specific `device class`_.

Device Module
-------------

The `device.py`_ module provides a variety of functions for working with devices.
A few examples include :func:`aikit.get_all_aikit_arrays_on_dev` which gets all arrays which are currently alive on the specified device, :func:`aikit.dev` which gets the device for input array, and :func:`aikit.num_gpus` which determines the number of available GPUs for use with the backend framework.

Many functions in the :mod:`device.py` module are *convenience* functions, which means that they do not directly modify arrays, as explained in the `Function Types <function_types.rst>`_ section.

For example, the following are all convenience functions: `aikit.total_mem_on_dev`_, which gets the total amount of memory for a given device, `aikit.dev_util`_, which gets the current utilization (%) for a given device, `aikit.num_cpu_cores`_, which determines the number of cores available in the CPU, and `aikit.default_device`_, which returns the correct device to use.

`aikit.default_device`_ is arguably the most important function.
Any function in the functional API that receives a :code:`device` argument will make use of this function, as explained below.

Arguments in other Functions
----------------------------

Like with :code:`dtype`, all :code:`device` arguments are also keyword-only.
All creation functions include the :code:`device` argument, for specifying the device on which to place the created array.
Some other functions outside of the :code:`creation.py` submodule also support the :code:`device` argument, such as :func:`aikit.random_uniform` which is located in :mod:`random.py`, but this is simply because of dual categorization.
:func:`aikit.random_uniform` is also essentially a creation function, despite not being located in :mod:`creation.py`.

The :code:`device` argument is generally not included for functions which accept arrays in the input and perform operations on these arrays.
In such cases, the device of the output arrays is the same as the device for the input arrays.
In cases where the input arrays are located on different devices, an error will generally be thrown, unless the function is specific to distributed training.

The :code:`device` argument is handled in `infer_device`_ for all functions which have the :code:`@infer_device` decorator, similar to how :code:`dtype` is handled.
This function calls `aikit.default_device`_ in order to determine the correct device.
As discussed in the `Function Wrapping <function_wrapping.rst>`_ section, this is applied to all applicable functions dynamically during `backend setting`_.

Overall, `aikit.default_device`_ infers the device as follows:

#. if the :code:`device` argument is provided, use this directly
#. otherwise, if an array is present in the arguments (very rare if the :code:`device` argument is present), set :code:`arr` to this array.
   This will then be used to infer the device by calling :func:`aikit.dev` on the array
#. otherwise, if no arrays are present in the arguments (by far the most common case if the :code:`device` argument is present), then use the global default device, which currently can either be :code:`cpu`, :code:`gpu:idx` or :code:`tpu:idx`.
   The default device is settable via :func:`aikit.set_default_device`.

For the majority of functions which defer to `infer_device`_ for handling the device, these steps will have been followed and the :code:`device` argument will be populated with the correct value before the backend-specific implementation is even entered into.
Therefore, whereas the :code:`device` argument is listed as optional in the aikit API at :mod:`aikit/functional/aikit/category_name.py`, the argument is listed as required in the backend-specific implementations at :mod:`aikit/functional/backends/backend_name/category_name.py`.

This is exactly the same as with the :code:`dtype` argument, as explained in the `Data Types <data_types.rst>`_ section.

Let's take a look at the function :func:`aikit.zeros` as an example.

The implementation in :mod:`aikit/functional/aikit/creation.py` has the following signature:

.. code-block:: python

    @outputs_to_aikit_arrays
    @handle_out_argument
    @infer_dtype
    @infer_device
    def zeros(
        shape: Union[int, Sequence[int]],
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
    ) -> aikit.Array:

Whereas the backend-specific implementations in :mod:`aikit/functional/backends/backend_name/creation.py` all list :code:`device` as required.

Jax:

.. code-block:: python

    def zeros(
        shape: Union[int, Sequence[int]],
        *,
        dtype: jnp.dtype,
        device: jaxlib.xla_extension.Device,
    ) -> JaxArray:

NumPy:

.. code-block:: python

    def zeros(
        shape: Union[int, Sequence[int]],
        *,
        dtype: np.dtype,
        device: str,
    ) -> np.ndarray:

TensorFlow:

.. code-block:: python

    def zeros(
        shape: Union[int, Sequence[int]],
        *,
        dtype: tf.DType,
        device: str,
    ) -> Tensor:

PyTorch:

.. code-block:: python

    def zeros(
        shape: Union[int, Sequence[int]],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:

This makes it clear that these backend-specific functions are only entered into once the correct :code:`device` has been determined.

However, the :code:`device` argument for functions without the :code:`@infer_device` decorator is **not** handled by `infer_device`_, and so these defaults must be handled by the backend-specific implementations themselves, by calling :func:`aikit.default_device` internally.

Device handling
---------------

Different frameworks handle devices differently while performing an operation. For example, torch expects
all the tensors to be on the same device while performing an operation, or else, it throws a device exception. On the other hand, tensorflow
doesn't care about this, it moves all the tensors to the same device before performing an operation.

**Controlling Device Handling Behaviour**

In Aikit, users can control the device on which the operation is to be executed using `aikit.set_soft_device_mode`_ flag. There are two cases for this,
either the soft device mode is set to :code:`True` or :code:`False`.

**When aikit.set_soft_device_mode(True)**:

a. All the input arrays are moved to :code:`aikit.default_device()` while performing an operation. If the array is already present
in the default device, no device shifting is done.

In the example below, even though the input arrays :code:`x` and :code:`y` are created on different devices('cpu' and 'gpu:0'), the arrays
are moved to :code:`aikit.default_device()` while performing :code:`aikit.add` operation, and the output array will be on this device.

.. code-block:: python

    aikit.set_backend("torch")
    aikit.set_soft_device_mode(True)
    x = aikit.array([1], device="cpu")
    y = aikit.array([34], device="gpu:0")
    aikit.add(x, y)

The priority of device shifting is the following in this mode:

#. The ``device`` argument.
#. device the arrays are on.
#. :code:`default_device`


**When aikit.set_soft_device_mode(False)**:

a. If any of the input arrays are on a different device, a device exception is raised.

In the example below, since the input arrays are on different devices('cpu' and 'gpu:0'), an :code:`AikitBackendException` is raised while performing :code:`aikit.add`.

.. code-block:: python

    aikit.set_backend("torch")
    aikit.set_soft_device_mode(False)
    x = aikit.array([1], device="cpu")
    y = aikit.array([34], device="gpu:0")
    aikit.add(x, y)

This is the exception you will get while running the code above:

.. code-block:: python

    AikitBackendException: torch: add:   File "/content/aikit/aikit/utils/exceptions.py", line 210, in _handle_exceptions
        return fn(*args, **kwargs)
    File "/content/aikit/aikit/func_wrapper.py", line 1013, in _handle_nestable
        return fn(*args, **kwargs)
    File "/content/aikit/aikit/func_wrapper.py", line 905, in _handle_out_argument
        return fn(*args, out=out, **kwargs)
    File "/content/aikit/aikit/func_wrapper.py", line 441, in _inputs_to_native_arrays
        return fn(*new_args, **new_kwargs)
    File "/content/aikit/aikit/func_wrapper.py", line 547, in _outputs_to_aikit_arrays
        ret = fn(*args, **kwargs)
    File "/content/aikit/aikit/func_wrapper.py", line 358, in _handle_array_function
        return fn(*args, **kwargs)
    File "/content/aikit/aikit/func_wrapper.py", line 863, in _handle_device_shifting
        raise aikit.utils.exceptions.AikitException(
    During the handling of the above exception, another exception occurred:
    Expected all input arrays to be on the same device, but found at least two devices - ('cpu', 'gpu:0'),
    set `aikit.set_soft_device_mode(True)` to handle this problem.

b. If all the input arrays are on the same device, the operation is executed without raising any device exceptions.

The example below runs without issues since both the input arrays are on 'gpu:0' device:

.. code-block:: python

    aikit.set_backend("torch")
    aikit.set_soft_device_mode(False)
    x = aikit.array([1], device="gpu:0")
    y = aikit.array([34], device="gpu:0")
    aikit.add(x, y)

The code to handle all these cases are present inside `@handle_device_shifting`_ decorator, which is wrapped around
all the functions that accept at least one array as input(except mixed and compositional functions) in `aikit.functional.aikit`_ submodule. The decorator calls
:code:`aikit.handle_soft_device_variable` function under the hood to handle device shifting for each backend.

The priority of device shifting is following in this mode:

#. The ``device`` argument.
#. :code:`default_device`

**Soft Device Handling Function**

This is a function which plays a crucial role in the :code:`handle_device_shifting` decorator. The purpose of this function is to ensure that the function :code:`fn` passed to it is executed on the device passed in :code:`device_shifting_dev` argument. If it is passed as :code:`None`, then the function will be executed on the default device.

Most of the backend implementations are very similar, first they move all the arrays to the desired device using :code:`aikit.nested_map` and then execute the function inside the device handling context manager from that native framework. The purpose of executing the function inside the context manager is to handle the functions that do not accept any arrays, the only way in that case to let the native framework know on which device we want the function to be executed on is through the context manager. This approach is used in most backend implementations with the exception being tensorflow, where we don't have to move all the tensors to the desired device because just using its context manager is enough, it moves all the tensors itself internally, and numpy, since it only accepts `cpu` as a device.

**Forcing Operations on User Specified Device**

The `aikit.DefaultDevice`_ context manager can be used to force the operations to be performed on to a specific device. For example,
in the code below, both :code:`x` and :code:`y` will be moved from 'gpu:0' to 'cpu' device and :code:`aikit.add` operation will be performed on 'cpu' device:

.. code-block:: python

    x = aikit.array([1], device="gpu:0")
    y = aikit.array([34], device="gpu:0")
    with aikit.DefaultDevice("cpu"):
        z = aikit.add(x, y)

On entering :code:`aikit.DefaultDevice("cpu")` context manager, under the hood, the default device is set to 'cpu' and soft device
mode is turned on. All these happens under the `__enter__`_ method of the
context manager. So from now on, all the operations will be executed on 'cpu' device.

On exiting the context manager(`__exit__`_ method), the default device and soft device mode is reset to the previous state using `aikit.unset_default_device()`_ and
`aikit.unset_soft_device_mode()`_ respectively, to move back to the previous state.

There are some functions(mostly creation function) which accept a :code:`device` argument. This is for specifying on which device the function is executed on and the device of the returned array. :code:`handle_device_shifting` deals with this argument by first checking if it exists and then setting :code:`device_shifting_dev` to that which is then passed to the :code:`handle_soft_device_variable` function depending on the :code:`soft_device` mode.


**Round Up**

This should have hopefully given you a good feel for devices, and how these are handled in Aikit.

If you have any questions, please feel free to reach out on `discord`_ in the `devices thread`_!


**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/RZmTUwTYhKI" class="video">
    </iframe>
