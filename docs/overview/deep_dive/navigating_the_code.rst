Navigating the Code
===================

.. _`Array API Standard`: https://data-apis.org/array-api/latest/
.. _`repo`: https://github.com/khulnasoft/aikit
.. _`discord`: https://discord.gg/sXyFF8tDtm
.. _`navigating the code thread`: https://discord.com/channels/799879767196958751/1189905165373935616
.. _`Array API Standard convention`: https://data-apis.org/array-api/2021.12/API_specification/array_object.html#api-specification-array-object--page-root
.. _`flake8`: https://flake8.pycqa.org/en/latest/index.html

Categorization
--------------

Aikit uses the following categories taken from the `Array API Standard`_:

* `constants <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/constants.py>`_
* `creation <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/creation.py>`_
* `data_type <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/data_type.py>`_
* `elementwise <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/elementwise.py>`_
* `linear_algebra <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/linear_algebra.py>`_
* `manipulation <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/manipulation.py>`_
* `searching <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/searching.py>`_
* `set <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/set.py>`_
* `sorting <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/sorting.py>`_
* `statistical <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/statistical.py>`_
* `utility <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/utility.py>`_

In addition to these, we also add the following categories, used for additional functions in Aikit that are not in the `Array API Standard`_:

* `activations <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/activations.py>`_
* `compilation <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/compilation.py>`_
* `device <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/device.py>`_
* `general <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/general.py>`_
* `gradients <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/gradients.py>`_
* `layers <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/layers.py>`_
* `losses <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/losses.py>`_
* `meta <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/meta.py>`_
* `nest <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/nest.py>`_
* `norms <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/norms.py>`_
* `random <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/random.py>`_

Some functions that you're considering adding might overlap several of these categorizations, and in such cases you should look at the other functions included in each file, and use your best judgement for which categorization is most suitable.

We can always suggest a more suitable location when reviewing your pull request if needed 🙂

Submodule Design
----------------

Aikit is designed so that all methods are called directly from the :mod:`aikit` namespace, such as :func:`aikit.matmul`, and not :func:`aikit.some_namespace.matmul`.
Therefore, inside any of the folders :mod:`aikit.functional.aikit`, :mod:`aikit.functional.backends.some_backend`, :mod:`aikit.functional.backends.another_backend` the functions can be moved to different files or folders without breaking anything at all.
This makes it very simple to refactor and re-organize parts of the code structure in an ongoing manner.

The :code:`__init__.py` inside each of the subfolders are very similar, importing each function via :code:`from .file_name import *` and also importing each file as a submodule via :code:`from . import file_name`.
For example, an extract from `aikit/aikit/functional/aikit/__init__.py <https://github.com/khulnasoft/aikit/blob/40836963a8edfe23f00a375b63bbb5c878bfbaac/aikit/functional/aikit/__init__.py>`_ is given below:

.. code-block:: python

    from . import elementwise
    from .elementwise import *
    from . import general
    from .general import *
    # etc.


Aikit API
-------

All function signatures for the Aikit API are defined in the :mod:`aikit.functional.aikit` submodule.
Functions written here look something like the following, (explained in much more detail in the following sections):


.. code-block:: python


    def my_func(
        x: Union[aikit.Array, aikit.NativeArray],
        /,
        axes: Union[int, Sequence[int]],
        *,
        dtype: Optional[Union[aikit.Dtype, aikit.NativeDtype]] = None,
        device: Optional[Union[aikit.Device, aikit.NativeDevice]] = None,
        out: Optional[aikit.Array] = None
    ) -> aikit.Array:
        """
        Explanation of the function.

        .. note::
            This is an important note.

        **Special Cases**

        For this particular case,

        - If ``x`` is ``NaN``, do something
        - If ``y`` is ``-0``, do something else
        - etc.

        Parameters
        ----------
        x
            input array. Should have a numeric data type.
        axes
            the axes along which to perform the op.
        dtype
            array data type.
        device
            the device on which to place the new array.
        out
            optional output array, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array.

        Examples
        --------

        Some examples go here
        """
        return aikit.current_backend(x).my_func(x, axes, dtype=dtype, device=device, out=out)

We follow the `Array API Standard convention`_ about positional and keyword arguments.

* Positional parameters must be positional-only parameters.
  Positional-only parameters have no externally-usable name.
  When a method accepting positional-only parameters is called, positional arguments are mapped to these parameters based solely on their order.
* Optional parameters must be keyword-only arguments.

This convention makes it easier for us to modify functions in the future.
Keyword-only parameters will mandate the use of argument names when calling functions, and this will increase our flexibility for extending function behaviour in future releases without breaking forward compatibility.
Similar arguments can be kept together in the argument list, rather than us needing to add these at the very end to ensure positional argument behaviour remains the same.

The :code:`dtype`, :code:`device` and :code:`out` arguments are always keyword-only.
Arrays always have a type hint :code:`Union[aikit.Array, aikit.NativeArray]` in the input and :class:`aikit.Array` in the output.
All functions which produce a single array include the :code:`out` argument.
The reasons for each of these features are explained in the following sections.

Backend API
-----------

Code in the backend submodules such as :mod:`aikit.functional.backends.torch` should then look something like:

.. code-block:: python


    def my_func(
        x: torch.Tensor,
        /,
        axes: Union[int, Sequence[int]],
        *,
        dtype: torch.dtype,
        device: torch.device,
        out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return torch.function_name(x, axes, dtype, device, out)

The :code:`dtype`, :code:`device` and :code:`out` arguments are again all keyword-only, but :code:`dtype` and :code:`device` are now required arguments, rather than optional as they were in the Aikit API.
All arrays also now have the same type hint :class:`torch.Tensor`, rather than :code:`Union[aikit.Array, aikit.NativeArray]` in the input and :class:`aikit.Array` in the output.
The backend methods also should not add a docstring.
Again, the reasons for these features are explained in the following sections.

Submodule Helper Functions
--------------------------

At times, helper functions specific to the submodule are required to:

* keep the code clean and readable
* be imported in their respective backend implementations

To have a better idea on this, let's look at an example!

**Helper in Aikit**

.. code-block:: python

    # in aikit/utils/assertions.py
    def check_fill_value_and_dtype_are_compatible(fill_value, dtype):
        if (
            not (
                (aikit.is_int_dtype(dtype) or aikit.is_uint_dtype(dtype))
                and isinstance(fill_value, int)
            )
            and not (
                aikit.is_complex_dtype(dtype) and isinstance(fill_value, (float, complex))
            )
            and not (
                aikit.is_float_dtype(dtype)
                and isinstance(fill_value, (float, np.float32))
                or isinstance(fill_value, bool)
            )
        ):
            raise aikit.utils.exceptions.AikitException(
                f"the fill_value: {fill_value} and data type: {dtype} are not compatible"
            )


In the :func:`full_like` function in :mod:`creation.py`, the types of :code:`fill_value` and :code:`dtype` has to be verified to avoid errors.
This check has to be applied to all backends, which means the related code is common and identical.
In this case, we can extract the code to be a helper function on its own, placed in its related submodule (:mod:`creation.py` here).
In this example, the helper function is named as :func:`check_fill_value_and_dtype_are_compatible`.

Then, we import this submodule-specific helper function to the respective backends, where examples for each backend is shown below.

**Jax**

.. code-block:: python

    # in aikit/functional/backends/jax/creation.py

    def full_like(
        x: JaxArray,
        /,
        fill_value: Number,
        *,
        dtype: jnp.dtype,
        device: jaxlib.xla_extension.Device,
        out: Optional[JaxArray] = None,
    ) -> JaxArray:
        aikit.utils.assertions.check_fill_value_and_dtype_are_compatible(fill_value, dtype)
        return _to_device(
            jnp.full_like(x, fill_value, dtype=dtype),
            device=device,
        )

**NumPy**

.. code-block:: python

    # in aikit/functional/backends/numpy/creation.py

    def full_like(
        x: np.ndarray,
        /,
        fill_value: Number,
        *,
        dtype: np.dtype,
        device: str,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        aikit.utils.assertions.check_fill_value_and_dtype_are_compatible(fill_value, dtype)
        return _to_device(np.full_like(x, fill_value, dtype=dtype), device=device)

**TensorFlow**

.. code-block:: python

    # in aikit/functional/backends/tensorflow/creation.py

    def full_like(
        x: Union[tf.Tensor, tf.Variable],
        /,
        fill_value: Number,
        *,
        dtype: tf.DType,
        device: str,
        out: Optional[Union[tf.Tensor, tf.Variable]] = None,
    ) -> Union[tf.Tensor, tf.Variable]:
        aikit.utils.assertions.check_fill_value_and_dtype_are_compatible(fill_value, dtype)
        with tf.device(device):
            return tf.experimental.numpy.full_like(x, fill_value, dtype=dtype)


.. note::
   We shouldn't be enabling numpy behaviour in tensorflow as it leads to issues with the bfloat16 datatype in tensorflow implementations


**Torch**

.. code-block:: python

    # in aikit/functional/backends/torch/creation.py

    def full_like(
        x: torch.Tensor,
        /,
        fill_value: Number,
        *,
        dtype: torch.dtype,
        device: torch.device,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        aikit.utils.assertions.check_fill_value_and_dtype_are_compatible(fill_value, dtype)
        return torch.full_like(x, fill_value, dtype=dtype, device=device)

Version Unpinning
-----------------

At any point in time, Aikit's development will be predominantly focused around the latest pypi version (and all prior versions) for each of the backend frameworks.

Earlier we had our versions pinned for each framework to provide stability but later concluded that by unpinnning the versions we would be able to account for the latest breaking changes if any and support the latest version of the framework.
Any prior version's compatibility would be tested by our multiversion testing pipeline, thus keeping us ahead and in light of the latest changes.

This helps to prevent our work from culminating over a fixed version while strides are being made in the said frameworks. Multiversion testing ensures the backward compatibility of the code while this approach ensures we support the latest changes too.


**Round Up**

This should have hopefully given you a good feel for how to navigate the Aikit codebase.

If you have any questions, please feel free to reach out on `discord`_ in the `navigating the code thread`_!


**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/67UYuLcAKbY" class="video">
    </iframe>
