Function Arguments
==================

.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`spec/API_specification/signatures`: https://github.com/data-apis/array-api/tree/main/spec/2022.12/API_specification
.. _`repo`: https://github.com/khulnasoft/aikit
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`function arguments thread`: https://discord.com/channels/799879767196958751/1190247823275470978
.. _`Array API Standard convention`: https://data-apis.org/array-api/2021.12/API_specification/array_object.html#api-specification-array-object--page-root

Here, we explain how the function arguments differ between the placeholder implementation at :mod:`aikit/functional/aikit/category_name.py`, and the backend-specific implementation at :mod:`aikit/functional/backends/backend_name/category_name.py`.

Many of these points are already addressed in the previous sections: `Arrays <arrays.rst>`_, `Data Types <data_types.rst>`_, `Devices <devices.rst>`_ and `Inplace Updates <inplace_updates.rst>`_.
However, we thought it would be convenient to revisit all of these considerations in a single section, dedicated to function arguments.

As for type-hints, all functions in the Aikit API at :mod:`aikit/functional/aikit/category_name.py` should have full and thorough type-hints.
Likewise, all backend implementations at :mod:`aikit/functional/backends/backend_name/category_name.py` should also have full and thorough type-hints.

In order to understand the various requirements for function arguments, it's useful to first look at some examples.

Examples
--------

For the purposes of explanation, we will use four functions as examples: :func:`aikit.tan`, :func:`aikit.roll`, :func:`aikit.add` and :func:`aikit.zeros`.

We present both the Aikit API signature and also a backend-specific signature for each function:

.. code-block:: python

    # Aikit
    @handle_exceptions
    @handle_nestable
    @handle_array_like_without_promotion
    @handle_out_argument
    @to_native_arrays_and_back
    @handle_array_function
    def tan(
        x: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        out: Optional[aikit.Array] = None
    ) -> aikit.Array:

    # PyTorch
    @handle_numpy_arrays_in_specific_backend
    def tan(
        x: torch.Tensor,
        /,
        *,
        out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

.. code-block:: python

    # Aikit
    @handle_exceptions
    @handle_nestable
    @handle_array_like_without_promotion
    @handle_out_argument
    @to_native_arrays_and_back
    @handle_array_function
    def roll(
        x: Union[aikit.Array, aikit.NativeArray],
        /,
        shift: Union[int, Sequence[int]],
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:

    # NumPy
    def roll(
        x: np.ndarray,
        /,
        shift: Union[int, Sequence[int]],
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:

.. code-block:: python

    # Aikit
    @handle_exceptions
    @handle_nestable
    @handle_out_argument
    @to_native_arrays_and_back
    @handle_array_function
    def add(
        x1: Union[float, aikit.Array, aikit.NativeArray],
        x2: Union[float, aikit.Array, aikit.NativeArray],
        /,
        *,
        alpha: Optional[Union[int, float]] = None,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:

    # TensorFlow
    def add(
        x1: Union[float, tf.Tensor, tf.Variable],
        x2: Union[float, tf.Tensor, tf.Variable],
        /,
        *,
        alpha: Optional[Union[int, float]] = None,
        out: Optional[Union[tf.Tensor, tf.Variable]] = None,
    ) -> Union[tf.Tensor, tf.Variable]:

.. code-block:: python

    # Aikit
    @handle_nestable
    @handle_array_like_without_promotion
    @handle_out_argument
    @inputs_to_native_shapes
    @outputs_to_aikit_arrays
    @handle_array_function
    @infer_dtype
    @infer_device
    def zeros(
        shape: Union[aikit.Shape, aikit.NativeShape],
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        out: Optional[aikit.Array] = None
    ) -> aikit.Array:

    # JAX
    def zeros(
        shape:  Union[aikit.NativeShape, Sequence[int]],
        *,
        dtype: jnp.dtype,
        device: jaxlib.xla_extension.Device,
        out: Optional[JaxArray] = None,
    ) -> JaxArray:


Positional and Keyword Arguments
--------------------------------
In both signatures, we follow the `Array API Standard convention`_ about positional and keyword arguments.

* Positional parameters must be positional-only parameters.
  Positional-only parameters have no externally-usable name.
  When a method accepting positional-only parameters is called, positional arguments are mapped to these parameters based solely on their order.
  This is indicated with an :code:`/` after all the position-only arguments.
* Optional parameters must be keyword-only arguments.
  A :code:`*` must be added before any of the keyword-only arguments.

Nearly all the functions in the `Array API Standard convention`_ have strictly positional-only and keyword-only arguments, with an exception of few :code:`creation` functions such as :code:`ones(shape, *, dtype=None, device=None)` , :code:`linspace(start, stop, /, num, *, dtype=None, device=None, endpoint=True)` etc.
The rationale behind this is purely a convention.
The :code:`shape` argument is often passed as a keyword, while the :code:`num` argument in :code:`linspace` is often passed as a keyword for improved understandability of the code.
Therefore, given that Aikit fully adheres to the Array API Standard, Aikit also adopts these same exceptions to the general rule for the :code:`shape` and :code:`num` arguments in these functions.


Input Arrays
------------

In each example, we can see that the input arrays have type :code:`Union[aikit.Array, aikit.NativeArray]` whereas the output arrays have type :class:`aikit.Array`.
This is the case for all functions in the Aikit API.
We always return an :class:`aikit.Array` instance to ensure that any subsequent Aikit code is fully framework-agnostic, with all operators performed on the returned array now handled by the special methods of the :class:`aikit.Array` class, and not the special methods of the backend array class (:class:`aikit.NativeArray`).
For example, calling any of (:code:`+`, :code:`-`, :code:`*`, :code:`/` etc.) on the array will result in (:code:`__add__`, :code:`__sub__`, :code:`__mul__`, :code:`__div__` etc.) being called on the array class.

:class:`aikit.NativeArray` instances are also not permitted for the :code:`out` argument, which is used in many functions.
This is because the :code:`out` argument dictates the array to which the result should be written, and so it effectively serves the same purpose as the function return when no :code:`out` argument is specified.
This is all explained in more detail in the `Arrays <arrays.rst>`_ section.

out Argument
------------

The :code:`out` argument should always be provided as a keyword-only argument, and it should be added to all functions in the Aikit API and backend API which support inplace updates, with a default value of :code:`None` in all cases.
The :code:`out` argument is explained in more detail in the `Inplace Updates <inplace_updates.rst>`_ section.

dtype and device arguments
--------------------------

In the Aikit API at :mod:`aikit/functional/aikit/category_name.py`, the :code:`dtype` and :code:`device` arguments should both always be provided as keyword-only arguments, with a default value of :code:`None`.
In contrast, these arguments should both be added as required arguments in the backend implementation at :mod:`aikit/functional/backends/backend_name/category_name.py`.
In a nutshell, by the time the backend implementation is entered, the correct :code:`dtype` and :code:`device` to use have both already been correctly handled by code which is wrapped around the backend implementation.
This is further explained in the `Data Types <data_types.rst>`_ and `Devices <devices.rst>`_ sections respectively.

Numbers in Operator Functions
-----------------------------

All operator functions (which have a corresponding such as :code:`+`, :code:`-`, :code:`*`, :code:`/`) must also be fully compatible with numbers (float or :code:`int`) passed into any of the array inputs, even in the absence of any arrays.
For example, :code:`aikit.add(1, 2)`, :code:`aikit.add(1.5, 2)` and :code:`aikit.add(1.5, aikit.array([2]))` should all run without error.
Therefore, the type hints for :func:`aikit.add` include float as one of the types in the :code:`Union` for the array inputs, and also as one of the types in the :code:`Union` for the output.
`PEP 484 Type Hints <https://peps.python.org/pep-0484/#the-numeric-tower>`_ states that "when an argument is annotated as having type float, an argument of type int is acceptable".
Therefore, we only include float in the type hints.

Integer Sequences
-----------------

For sequences of integers, generally the `Array API Standard`_ dictates that these should be of type :code:`Tuple[int]`, and not :code:`List[int]`.
However, in order to make Aikit code less brittle, we accept arbitrary integer sequences :code:`Sequence[int]` for such arguments (which includes :code:`list`, :code:`tuple` etc.).
This does not break the standard, as the standard is only intended to define a subset of required behaviour.
The standard can be freely extended, as we are doing here.
Good examples of this are the :code:`axis` argument of :func:`aikit.roll` and the :code:`shape` argument of :func:`aikit.zeros`, as shown above.

Nestable Functions
------------------

Most functions in the Aikit API can also consume and return :class:`aikit.Container` instances in place of the **any** of the function arguments.
If an :class:`aikit.Container` is passed, then the function is mapped across all of the leaves of this container.
Because of this feature, we refer to these functions as *nestable* functions.
However, because so many functions in the Aikit API are indeed *nestable* functions, and because this flexibility applies to **every** argument in the function, every type hint for these functions should technically be extended like so: :code:`Union[original_type, aikit.Container]`.

However, this would be very cumbersome, and would only serve to hinder the readability of the docs.
Therefore, we simply omit these :class:`aikit.Container` type hints from *nestable* functions, and instead mention in the docstring whether the function is *nestable* or not.

**Round Up**

These examples should hopefully give you a good understanding of what is required when adding function arguments.

If you have any questions, please feel free to reach out on `discord`_ in the `function arguments thread`_!


**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/5cAbryXza18" class="video">
    </iframe>
