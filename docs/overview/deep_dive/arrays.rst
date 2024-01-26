Arrays
======

.. _`inputs_to_native_arrays`: https://github.com/khulnasoft/aikit/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/aikit/func_wrapper.py#L149
.. _`outputs_to_aikit_arrays`: https://github.com/khulnasoft/aikit/blob/1eb841cdf595e2bb269fce084bd50fb79ce01a69/aikit/func_wrapper.py#L209
.. _`empty class`: https://github.com/khulnasoft/aikit/blob/529c8c0f128ff28331da7c8f52912d777d786cbe/aikit/__init__.py#L8
.. _`overwritten`: https://github.com/khulnasoft/aikit/blob/529c8c0f128ff28331da7c8f52912d777d786cbe/aikit/functional/backends/torch/__init__.py#L11
.. _`self._data`: https://github.com/khulnasoft/aikit/blob/529c8c0f128ff28331da7c8f52912d777d786cbe/aikit/array/__init__.py#L89
.. _`ArrayWithElementwise`: https://github.com/khulnasoft/aikit/blob/529c8c0f128ff28331da7c8f52912d777d786cbe/aikit/array/elementwise.py#L12
.. _`aikit.Array.add`: https://github.com/khulnasoft/aikit/blob/63d9c26acced9ef40e34f7b4fc1c1a75017f9c69/aikit/array/elementwise.py#L22
.. _`programmatically`: https://github.com/khulnasoft/aikit/blob/529c8c0f128ff28331da7c8f52912d777d786cbe/aikit/__init__.py#L148
.. _`backend type hints`: https://github.com/khulnasoft/aikit/blob/8605c0a50171bb4818d0fb3e426cec874de46baa/aikit/functional/backends/torch/elementwise.py#L219
.. _`Aikit type hints`: https://github.com/khulnasoft/aikit/blob/8605c0a50171bb4818d0fb3e426cec874de46baa/aikit/functional/aikit/elementwise.py#L1342
.. _`__setitem__`: https://github.com/khulnasoft/aikit/blob/8605c0a50171bb4818d0fb3e426cec874de46baa/aikit/array/__init__.py#L234
.. _`function wrapping`: https://github.com/khulnasoft/aikit/blob/0f131178be50ea08ec818c73078e6e4c88948ab3/aikit/func_wrapper.py#L170
.. _`inherits`: https://github.com/khulnasoft/aikit/blob/8cbffbda9735cf16943f4da362ce350c74978dcb/aikit/array/__init__.py#L44
.. _`is the case`: https://data-apis.org/array-api/latest/API_specification/array_object.html
.. _`__add__`: https://github.com/khulnasoft/aikit/blob/e4d9247266f5d99faad59543923bb24b88a968d9/aikit/array/__init__.py#L291
.. _`__sub__`: https://github.com/khulnasoft/aikit/blob/e4d9247266f5d99faad59543923bb24b88a968d9/aikit/array/__init__.py#L299
.. _`__mul__`: https://github.com/khulnasoft/aikit/blob/e4d9247266f5d99faad59543923bb24b88a968d9/aikit/array/__init__.py#L307
.. _`__truediv__`: https://github.com/khulnasoft/aikit/blob/e4d9247266f5d99faad59543923bb24b88a968d9/aikit/array/__init__.py#L319
.. _`repo`: https://github.com/khulnasoft/aikit
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`arrays thread`: https://discord.com/channels/799879767196958751/1189905906905919609
.. _`wrapped logic`: https://github.com/khulnasoft/aikit/blob/6a729004c5e0db966412b00aa2fce174482da7dd/aikit/func_wrapper.py#L95
.. _`NumPy's`: https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch
.. _`PyTorch's`: https://pytorch.org/docs/stable/notes/extending.html#extending-torch
There are two types of arrays in Aikit, there is the :class:`aikit.NativeArray` and also the :class:`aikit.Array`.

Native Array
------------

The :class:`aikit.NativeArray` is simply a placeholder class for a backend-specific array class, such as :class:`np.ndarray`, :class:`tf.Tensor`, :class:`torch.Tensor` or :class:`jaxlib.xla_extension.DeviceArray`.

When no framework is set, this is an `empty class`_.
When a framework is set, this is `overwritten`_ with the backend-specific array class.

Aikit Array
---------

The :class:`aikit.Array` is a simple wrapper class, which wraps around the :class:`aikit.NativeArray`, storing it in `self._data`_.

All functions in the Aikit functional API which accept *at least one array argument* in the input are implemented as instance methods in the :class:`aikit.Array` class.
The only exceptions to this are functions in the `nest <https://github.com/khulnasoft/aikit/blob/906ddebd9b371e7ae414cdd9b4bf174fd860efc0/aikit/functional/aikit/nest.py>`_ module and the `meta <https://github.com/khulnasoft/aikit/blob/906ddebd9b371e7ae414cdd9b4bf174fd860efc0/aikit/functional/aikit/meta.py>`_ module, which have no instance method implementations.

The organization of these instance methods follows the same organizational structure as the files in the functional API.
The :class:`aikit.Array` class `inherits`_ from many category-specific array classes, such as `ArrayWithElementwise`_, each of which implements the category-specific instance methods.

Each instance method simply calls the functional API function internally, but passes in :code:`self._data` as the first *array* argument.
`aikit.Array.add`_ is a good example.
However, it's important to bear in mind that this is *not necessarily the first argument*, although in most cases it will be.
We also **do not** set the :code:`out` argument to :code:`self` for instance methods.
If the only array argument is the :code:`out` argument, then we do not implement this instance method.
For example, we do not implement an instance method for `aikit.zeros <https://github.com/khulnasoft/aikit/blob/1dba30aae5c087cd8b9ffe7c4b42db1904160873/aikit/functional/aikit/creation.py#L116>`_.

Given the simple set of rules which underpin how these instance methods should all be implemented, if a source-code implementation is not found, then this instance method is added `programmatically`_.
This serves as a helpful backup in cases where some methods are accidentally missed out.

The benefit of the source code implementations is that this makes the code much more readable, with important methods not being entirely absent from the code.
It also enables other helpful perks, such as auto-completions in the IDE etc.

Most special methods also simply wrap a corresponding function in the functional API, as `is the case`_ in the Array API Standard.
Examples include `__add__`_, `__sub__`_, `__mul__`_ and `__truediv__`_ which directly call :func:`aikit.add`, :func:`aikit.subtract`, :func:`aikit.multiply` and :func:`aikit.divide` respectively.
However, for some special methods such as `__setitem__`_, there are substantial differences between the backend frameworks which must be addressed in the :class:`aikit.Array` implementation.

Array Handling
--------------

When calling backend-specific functions such as :func:`torch.sin`, we must pass in :class:`aikit.NativeArray` instances.
For example, :func:`torch.sin` will throw an error if we try to pass in an :class:`aikit.Array` instance.
It must be provided with a :class:`torch.Tensor`, and this is reflected in the `backend type hints`_.

However, all Aikit functions must return :class:`aikit.Array` instances, which is reflected in the `Aikit type hints`_.
The reason we always return :class:`aikit.Array` instances from Aikit functions is to ensure that any subsequent Aikit code is fully framework-agnostic, with all operators performed on the returned array being handled by the special methods of the :class:`aikit.Array` class, and not the special methods of the backend :class:`aikit.NativeArray` class.

For example, calling any of (:code:`+`, :code:`-`, :code:`*`, :code:`/` etc.) on the array will result in (:meth:`__add__`, :meth:`__sub__`, :meth:`__mul__`, :meth:`__truediv__` etc.) being called on the array class.

For most special methods, calling them on the :class:`aikit.NativeArray` would not be a problem because all backends are generally quite consistent, but as explained above, for some functions such as `__setitem__`_ there are substantial differences which must be addressed in the :class:`aikit.Array` implementation in order to guarantee unified behaviour.

Given that all Aikit functions return :class:`aikit.Array` instances, all Aikit functions must also support :class:`aikit.Array` instances in the input, otherwise it would be impossible to chain functions together!

Therefore, most functions in Aikit must adopt the following pipeline:

#. convert all :class:`aikit.Array` instances in the input arguments to :class:`aikit.NativeArray` instances
#. call the backend-specific function, passing in these :class:`aikit.NativeArray` instances
#. convert all of the :class:`aikit.NativeArray` instances which are returned from the backend function back into :class:`aikit.Array` instances, and return

Given the repeating nature of these steps, this is all entirely handled in the `inputs_to_native_arrays`_ and `outputs_to_aikit_arrays`_ wrappers, as explained in the `Function Wrapping <function_wrapping.rst>`_ section.

All Aikit functions *also* accept :class:`aikit.NativeArray` instances in the input.
This is for a couple of reasons.
Firstly, :class:`aikit.Array` instances must be converted to :class:`aikit.NativeArray` instances anyway, and so supporting them in the input is not a problem.
Secondly, this makes it easier to combine backend-specific code with Aikit code, without needing to explicitly wrap any arrays before calling sections of Aikit code.

Therefore, all input arrays to Aikit functions have type :code:`Union[aikit.Array, aikit.NativeArray]`, whereas the output arrays have type :class:`aikit.Array`.
This is further explained in the `Function Arguments <function_arguments.rst>`_ section.

However, :class:`aikit.NativeArray` instances are not permitted for the :code:`out` argument, which is used in most functions.
This is because the :code:`out` argument dictates the array to which the result should be written, and so it effectively serves the same purpose as the function return.
This is further explained in the `Inplace Updates <inplace_updates.rst>`_ section.

As a final point, extra attention is required for *compositional* functions, as these do not directly defer to a backend implementation.
If the first line of code in a compositional function performs operations on the input array, then this will call the special methods on an :class:`aikit.NativeArray` and not on an :class:`aikit.Array`.
For the reasons explained above, this would be a problem.

Therefore, all compositional functions have a separate piece of `wrapped logic`_ to ensure that all :class:`aikit.NativeArray` instances are converted to :class:`aikit.Array` instances before entering into the compositional function.

Integrating custom classes with Aikit
-----------------------------------

Aikit's functional API and its functions can easily be integrated with non-Aikit classes. Whether these classes are ones that inherit from Aikit or completely standalone custom classes, using Aikit's :code:`__aikit_array_function__`, Aikit's functions can handle inputs of those types.

To make use of that feature, the class must contain an implementation for these functions and it must contain an implementation for the function :code:`__aikit_array_function__`. If a non-Aikit class is passed to an Aikit function, a call to this class's :code:`__aikit_array_function__` is made which directs Aikit's function to handle that input type correctly. This allows users to define custom implementations for any of the functions that can be found in Aikit's functional API which would further make it easy to integrate those classes with other Aikit projects.

**Note**
This functionality is inspired by `NumPy's`_ :code:`__aikit_array_function__` and `PyTorch's`_ :code:`__torch_function__`.

As an example, consider the following class :code:`MyArray` with the following definition:

.. code-block:: python

    class MyArray:
	    def __init__(self, data=None):
		    self.data = data

Running any of Aikit’s functions using a :code:`MyArray` object as input will throw an :code:`AikitBackendException` since Aikit’s functions do not support this class type as input. This is where :code:`__aikit_array_function__` comes into play. Let’s add the method to our :code:`MyArray` class to see how it works.

There are different ways to do so. One way is to use a global dict :code:`HANDLED_FUNCTIONS` which will map Aikit’s functions to the custom variant functions:

.. code-block:: python

    HANDLED_FUNCTIONS = {}
    class MyArray:
        def __init__(self, data=None):
    		self.data = data
    	def __aikit_array_function__(self, func, types, args, kwargs):
    		if func not in HANDLED_FUNCTIONS:
    			return NotImplemented
    		if not all(issubclass(t, (MyArray, aikit.Array, aikit.NativeArray)) for t in types):
    			return NotImplemented
    		return HANDLED_FUNCTIONS[func](*args, **kwargs)

:code:`__aikit_array_function__` accepts four parameters: :code:`func` representing a reference to the array API function being
overridden, :code:`types` a list of the types of objects implementing :code:`__aikit_array_function__`, :code:`args` a tuple of arguments supplied to the function, and :code:`kwargs` being a dictionary of keyword arguments passed to the function.
While this class contains an implementation for :code:`__aikit_array_function__`, it is still not enough as it is necessary to implement any needed Aikit functions with the new :code:`MyArray` class as input(s) for the code to run successfully.
We will define a decorator function :code:`implements` that can be used to add functions to :code:`HANDLED_FUNCTIONS`:

.. code-block:: python

    def implements(aikit_function):
        def decorator(func):
            HANDLED_FUNCTIONS[aikit_function] = func
            return func
        return decorator

Lastly, we need to apply that decorator to the override function. Let’s consider for example a function that overrides :code:`aikit.abs`:

.. code-block:: python

    @implements(aikit.abs)
    def my_abs(my_array, aikit_array):
     	my_array.data = abs(my_array.data)

Now that we have added the function to :code:`HANDLED_FUNCTIONS`, we can now use :code:`aikit.abs` with :code:`MyArray` objects:

.. code-block:: python

    X = MyArray(-3)
    X = aikit.abs(X)

Of course :code:`aikit.abs` is an example of a function that is easy to override since it only requires one operand. The same approach can be used to override functions with multiple operands, including arrays or array-like objects that define :code:`__aikit_array_function__`.

It is relevant to mention again that any function not stored inside the dict :code:`HANDLED_FUNCTIONS` will not work and it is also important to notice that the operands passed to the function must match that of the function stored in the dict. For instance :code:`my_abs` takes only one parameter which is a :code:`MyArray` object. So, passing any other operands to the function will result in an exception :code:`AikitBackendException` being thrown. Lastly, for a custom class to be covered completely with Aikit's functional API, it is necessary to create an implementation for all the relevant functions within the API that will be used by this custom class. That can be all the functions in the API or only a subset of them.

**Round Up**

This should have hopefully given you a good feel for the different types of arrays, and how these are handled in Aikit.

If you have any questions, please feel free to reach out on `discord`_ in the `arrays thread`_!


**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/tAlDPnWcLDE" class="video">
    </iframe>
