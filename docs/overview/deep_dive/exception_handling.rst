Exception Handling
==================

.. _`exception handling thread`: https://discord.com/channels/799879767196958751/1189908450570928149
.. _`discord`: https://discord.gg/sXyFF8tDtm

As Aikit is unifying multiple backends, various issues are seen during exception handling:

#. each backend throws its own exceptions
#. exceptions thrown are backend-specific, therefore inconsistent

To unify the handling of exceptions and assertions, Aikit includes a custom exception class and decorator, which are explained further in the following sub-sections.


Aikit Exception Class
-------------------

Firstly, Aikit's base exception class is :code:`AikitException` class, which inherits from the Python :code:`Exception` class.

.. code-block:: python

    # in aikit/utils/exceptions.py
    class AikitException(Exception):
        def __init__(self, *messages, include_backend=False):
            self.native_error = (
                messages[0]
                if len(messages) == 1
                and isinstance(messages[0], Exception)
                and not include_backend
                else None
            )
            if self.native_error is None:
                super().__init__(
                    _combine_messages(*messages, include_backend=include_backend)
                )
            else:
                super().__init__(str(messages[0]))

In cases where an exception class for a specific purpose is required, we inherit from the :code:`AikitException` class.
For example, the :code:`AikitBackendException` class is created to unify backend exceptions.

.. code-block:: python

    # in aikit/utils/exceptions.py
    class AikitBackendException(AikitException):
        def __init__(self, *messages, include_backend=False):
            super().__init__(*messages, include_backend=include_backend)

In some Array API tests, :code:`IndexError` and :code:`ValueError` are explicitly tested to ensure that the functions are behaving correctly.
Thus, the :code:`AikitIndexError` and :code:`AikitValueError` classes unifies these special cases.
For a more general case, the :code:`AikitError` class can be used.

.. code-block:: python

    # in aikit/utils/exceptions.py
    class AikitError(AikitException):
        def __init__(self, *messages, include_backend=False):
            super().__init__(*messages, include_backend=include_backend)

More Custom Exception classes were created to unify sub-categories of errors. We try our best to ensure that the same type of
Exception is raised for the same type of Error regardless of the backend.
This will ensure that the exceptions are truly unified for all the different types of errors.
The implementations of these custom classes are exactly the same as :code:`AikitError` class.
Currently there are 5 custom exception classes in aikit.

1. :code:`AikitIndexError`: This Error is raised for anything Indexing related. For Instance, providing out of bound axis in any function.
2. :code:`AikitValueError`: This is for anything related to providing wrong values. For instance, passing :code:`high` value
                          smaller than :code:`low` value in :code:`aikit.random_uniform`.
3. :code:`AikitAttributeError`: This is raised when an undefined attribute is referenced.
4. :code:`AikitBroadcastShapeError`: This is raised whenever 2 shapes are expected to be broadcastable but are not.
5. :code:`AikitDtypePromotionError`: Similar to :code:`AikitBroadcastShapeError`, this is raised when 2 dtypes are expected to be promotable but are not.

The correct type of Exception class should be used for the corresponding type of error across the backends. This will truly unify all the exceptions raised in Aikit.

Configurable Mode for Stack Trace
---------------------------------

Aikit's transpilation nature allows users to write code in their preferred frontend
framework and then execute it with a different backend framework. For example, a
user who is comfortable with NumPy can use Aikit's NumPy frontend to run their code
with a JAX backend. However, since they may have no prior experience with JAX or
other backend frameworks, they may not want to encounter stack traces that traverse
Aikit and JAX functions. In such cases, it may be preferable for the user to avoid
encountering stack traces that extend through Aikit and JAX functions.

Therefore, options are made available for the stack traces to either truncate
at the frontend or aikit level, or in other cases, no truncation at all.

Let's look at the 3 different modes with an example of :code:`aikit.all` below!

1. Full

This is the default mode and keeps the complete stack traces. All :code:`numpy`
frontend, aikit specific, and native :code:`jax` stack traces are displayed.
The format of the error displayed in this mode is :code:`Aikit error: backend name: backend function name: native error: error message`

.. code-block:: none

    >>> aikit.functional.frontends.numpy.all(aikit.array([1,2,3]), axis=2)
      File "/aikit/aikit/utils/exceptions.py", line 198, in _handle_exceptions
        return fn(*args, **kwargs)
      File "/aikit/aikit/func_wrapper.py", line 911, in _handle_nestable
        return fn(*args, **kwargs)
      File "/aikit/aikit/func_wrapper.py", line 392, in _handle_array_like_without_promotion
        return fn(*args, **kwargs)
      File "/aikit/aikit/func_wrapper.py", line 805, in _handle_out_argument
        return fn(*args, out=out, **kwargs)
      File "/aikit/aikit/func_wrapper.py", line 432, in _inputs_to_native_arrays
        return fn(*new_args, **new_kwargs)
      File "/aikit/aikit/func_wrapper.py", line 535, in _outputs_to_aikit_arrays
        ret = fn(*args, **kwargs)
      File "/aikit/aikit/func_wrapper.py", line 349, in _handle_array_function
        return fn(*args, **kwargs)
      File "/aikit/aikit/functional/backends/jax/utility.py", line 22, in all
        raise aikit.utils.exceptions.AikitIndexError(error)

    During the handling of the above exception, another exception occurred:

      File "/aikit/other_test.py", line 22, in <module>
        aikit.functional.frontends.numpy.all(aikit.array([1,2,3]), axis=2)
      File "/aikit/aikit/functional/frontends/numpy/func_wrapper.py", line 523, in _handle_numpy_out
        return fn(*args, **kwargs)
      File "/aikit/aikit/functional/frontends/numpy/func_wrapper.py", line 396, in _outputs_to_numpy_arrays
        ret = fn(*args, **kwargs)
      File "/aikit/aikit/functional/frontends/numpy/func_wrapper.py", line 352, in _inputs_to_aikit_arrays_np
        return fn(*aikit_args, **aikit_kwargs)
      File "/aikit/aikit/functional/frontends/numpy/func_wrapper.py", line 453, in _from_zero_dim_arrays_to_scalar
        ret = fn(*args, **kwargs)
      File "/aikit/aikit/functional/frontends/numpy/logic/truth_value_testing.py", line 24, in all
        ret = aikit.all(a, axis=axis, keepdims=keepdims, out=out)
      File "/aikit/aikit/utils/exceptions.py", line 217, in _handle_exceptions
        raise aikit.utils.exceptions.AikitIndexError(

    AikitIndexError: jax: all: ValueError: axis 2 is out of bounds for an array of dimension 1


2. Frontend-only

This option displays only frontend-related stack traces. If compared with the
stack traces in the :code:`full` mode above, the :code:`jax` related traces
are pruned. Only the :code:`numpy` frontend related errors are shown.
A message is also displayed to inform that the traces are truncated and
the instructions to switch it back to the :code:`full` mode is included.
In this case, the format of the error is :code:`Aikit error: backend name: backend function name: error message`

.. code-block:: none

    >>> aikit.set_exception_trace_mode('frontend')
    >>> aikit.functional.frontends.numpy.all(aikit.array([1,2,3]), axis=2)
    <stack trace is truncated to frontend specific files, call `aikit.set_exception_trace_mode('full')` to view the full trace>

    During the handling of the above exception, another exception occurred:

    <stack trace is truncated to frontend specific files, call `aikit.set_exception_trace_mode('full')` to view the full trace>
      File "/aikit/aikit/functional/frontends/numpy/func_wrapper.py", line 523, in _handle_numpy_out
        return fn(*args, **kwargs)
      File "/aikit/aikit/functional/frontends/numpy/func_wrapper.py", line 396, in _outputs_to_numpy_arrays
        ret = fn(*args, **kwargs)
      File "/aikit/aikit/functional/frontends/numpy/func_wrapper.py", line 352, in _inputs_to_aikit_arrays_np
        return fn(*aikit_args, **aikit_kwargs)
      File "/aikit/aikit/functional/frontends/numpy/func_wrapper.py", line 453, in _from_zero_dim_arrays_to_scalar
        ret = fn(*args, **kwargs)
      File "/aikit/aikit/functional/frontends/numpy/logic/truth_value_testing.py", line 24, in all
        ret = aikit.all(a, axis=axis, keepdims=keepdims, out=out)

    AikitIndexError: jax: all: axis 2 is out of bounds for an array of dimension 1


3. Aikit specific

This option displays only aikit-related stack traces. If compared to the different
stack traces modes above, the aikit backend :code:`jax` related
traces (which were hidden in the :code:`frontend` mode) are available again
and the aikit frontend :code:`numpy` related traces remain visible.
However, the native :code:`jax` traces remain hidden because they are not
aikit-specific.
A message is also displayed to inform that the traces are truncated and the
instructions to switch it back to the :code:`full` mode is included.
The format of the error displayed is the same as the :code:`frontend` mode above.

.. code-block:: none

    >>> aikit.set_exception_trace_mode('aikit')
    >>> aikit.functional.frontends.numpy.all(aikit.array([1,2,3]), axis=2)
    <stack trace is truncated to aikit specific files, call `aikit.set_exception_trace_mode('full')` to view the full trace>
      File "/aikit/aikit/utils/exceptions.py", line 198, in _handle_exceptions
        return fn(*args, **kwargs)
      File "/aikit/aikit/func_wrapper.py", line 911, in _handle_nestable
        return fn(*args, **kwargs)
      File "/aikit/aikit/func_wrapper.py", line 392, in _handle_array_like_without_promotion
        return fn(*args, **kwargs)
      File "/aikit/aikit/func_wrapper.py", line 805, in _handle_out_argument
        return fn(*args, out=out, **kwargs)
      File "/aikit/aikit/func_wrapper.py", line 432, in _inputs_to_native_arrays
        return fn(*new_args, **new_kwargs)
      File "/aikit/aikit/func_wrapper.py", line 535, in _outputs_to_aikit_arrays
        ret = fn(*args, **kwargs)
      File "/aikit/aikit/func_wrapper.py", line 349, in _handle_array_function
        return fn(*args, **kwargs)
      File "/aikit/aikit/functional/backends/jax/utility.py", line 22, in all
        raise aikit.utils.exceptions.AikitIndexError(error)

    During the handling of the above exception, another exception occurred:

    <stack trace is truncated to aikit specific files, call `aikit.set_exception_trace_mode('full')` to view the full trace>
      File "/aikit/other_test.py", line 21, in <module>
        aikit.functional.frontends.numpy.all(aikit.array([1,2,3]), axis=2)
      File "/aikit/aikit/functional/frontends/numpy/func_wrapper.py", line 523, in _handle_numpy_out
        return fn(*args, **kwargs)
      File "/aikit/aikit/functional/frontends/numpy/func_wrapper.py", line 396, in _outputs_to_numpy_arrays
        ret = fn(*args, **kwargs)
      File "/aikit/aikit/functional/frontends/numpy/func_wrapper.py", line 352, in _inputs_to_aikit_arrays_np
        return fn(*aikit_args, **aikit_kwargs)
      File "/aikit/aikit/functional/frontends/numpy/func_wrapper.py", line 453, in _from_zero_dim_arrays_to_scalar
        ret = fn(*args, **kwargs)
      File "/aikit/aikit/functional/frontends/numpy/logic/truth_value_testing.py", line 24, in all
        ret = aikit.all(a, axis=axis, keepdims=keepdims, out=out)
      File "/aikit/aikit/utils/exceptions.py", line 217, in _handle_exceptions
        raise aikit.utils.exceptions.AikitIndexError(

    AikitIndexError: jax: all: axis 2 is out of bounds for an array of dimension 1


Aikit :code:`func_wrapper` Pruning
--------------------------------

Due to the wrapping operations in Aikit, a long list of less informative
:code:`func_wrapper` traces is often seen in the stack.
Including all of these wrapper functions in the stack trace can be very
unwieldy, thus they can be prevented entirely by setting
:code:`aikit.set_show_func_wrapper_trace_mode(False)`.
Examples are shown below to demonstrate the combination of this mode and the
3 different stack traces mode explained above.

1. Full

The :code:`func_wrapper` related traces have been hidden. All other traces
such as aikit-specific, frontend-related and the native traces remain visible.
A message is displayed as well to the user so that they are aware of the
pruning. The instructions to recover the :code:`func_wrapper` traces are
shown too.

.. code-block:: none

    >>> aikit.set_show_func_wrapper_trace_mode(False)
    >>> aikit.functional.frontends.numpy.all(aikit.array([1,2,3]), axis=2)
    <func_wrapper.py stack trace is squashed, call `aikit.set_show_func_wrapper_trace_mode(True)` in order to view this>
      File "/aikit/aikit/utils/exceptions.py", line 198, in _handle_exceptions
        return fn(*args, **kwargs)
      File "/aikit/aikit/functional/backends/jax/utility.py", line 22, in all
        raise aikit.utils.exceptions.AikitIndexError(error)

    During the handling of the above exception, another exception occurred:

    <func_wrapper.py stack trace is squashed, call `aikit.set_show_func_wrapper_trace_mode(True)` in order to view this>
      File "/aikit/other_test.py", line 22, in <module>
        aikit.functional.frontends.numpy.all(aikit.array([1,2,3]), axis=2)
      File "/aikit/aikit/functional/frontends/numpy/logic/truth_value_testing.py", line 24, in all
        ret = aikit.all(a, axis=axis, keepdims=keepdims, out=out)
      File "/aikit/aikit/utils/exceptions.py", line 217, in _handle_exceptions
        raise aikit.utils.exceptions.AikitIndexError(

    AikitIndexError: jax: all: ValueError: axis 2 is out of bounds for an array of dimension 1


2. Frontend-only

In the frontend-only stack trace mode, the aikit backend wrapping traces were
hidden but the frontend wrappers were still visible. By configuring the func
wrapper trace mode, the frontend wrappers will also be hidden. This can be
observed from the example below.

.. code-block:: none

    >>> aikit.set_exception_trace_mode('frontend')
    >>> aikit.set_show_func_wrapper_trace_mode(False)
    >>> aikit.functional.frontends.numpy.all(aikit.array([1,2,3]), axis=2)
    <stack trace is truncated to frontend specific files, call `aikit.set_exception_trace_mode('full')` to view the full trace>
    <func_wrapper.py stack trace is squashed, call `aikit.set_show_func_wrapper_trace_mode(True)` in order to view this>

    During the handling of the above exception, another exception occurred:

    <stack trace is truncated to frontend specific files, call `aikit.set_exception_trace_mode('full')` to view the full trace>
    <func_wrapper.py stack trace is squashed, call `aikit.set_show_func_wrapper_trace_mode(True)` in order to view this>
      File "/aikit/aikit/functional/frontends/numpy/logic/truth_value_testing.py", line 24, in all
        ret = aikit.all(a, axis=axis, keepdims=keepdims, out=out)

    AikitIndexError: jax: all: axis 2 is out of bounds for an array of dimension 1


3. Aikit specific

As the wrappers occur in :code:`aikit` itself, all backend and frontend wrappers
remain visible in the aikit-specific mode. By hiding the func wrapper traces,
the stack becomes cleaner and displays the aikit backend and frontend
exception messages only.

.. code-block:: none

    >>> aikit.set_exception_trace_mode('frontend')
    >>> aikit.set_show_func_wrapper_trace_mode(False)
    >>> aikit.functional.frontends.numpy.all(aikit.array([1,2,3]), axis=2)
    <stack trace is truncated to aikit specific files, call `aikit.set_exception_trace_mode('full')` to view the full trace>
    <func_wrapper.py stack trace is squashed, call `aikit.set_show_func_wrapper_trace_mode(True)` in order to view this>
      File "/aikit/aikit/utils/exceptions.py", line 198, in _handle_exceptions
        return fn(*args, **kwargs)
      File "/aikit/aikit/functional/backends/jax/utility.py", line 22, in all
        raise aikit.utils.exceptions.AikitIndexError(error)

    During the handling of the above exception, another exception occurred:

    <stack trace is truncated to aikit specific files, call `aikit.set_exception_trace_mode('full')` to view the full trace>
    <func_wrapper.py stack trace is squashed, call `aikit.set_show_func_wrapper_trace_mode(True)` in order to view this>
      File "/aikit/other_test.py", line 22, in <module>
        aikit.functional.frontends.numpy.all(aikit.array([1,2,3]), axis=2)
      File "/aikit/aikit/functional/frontends/numpy/logic/truth_value_testing.py", line 24, in all
        ret = aikit.all(a, axis=axis, keepdims=keepdims, out=out)
      File "/aikit/aikit/utils/exceptions.py", line 217, in _handle_exceptions
        raise aikit.utils.exceptions.AikitIndexError(

    AikitIndexError: jax: all: axis 2 is out of bounds for an array of dimension 1

:code:`@handle_exceptions` Decorator
----------------------------

To ensure that all backend exceptions are caught properly, a decorator is used to handle functions in the :code:`try/except` block.

.. code-block:: python

    # in aikit/utils/exceptions.py
    def handle_exceptions(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def _handle_exceptions(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            # Not to rethrow as AikitBackendException
            except AikitNotImplementedException as e:
                raise e
            except AikitError as e:
                _print_traceback_history()
                raise aikit.utils.exceptions.AikitError(fn.__name__, e, include_backend=True)
            except AikitBroadcastShapeError as e:
                _print_traceback_history()
                raise aikit.utils.exceptions.AikitBroadcastShapeError(
                    fn.__name__, e, include_backend=True
                )
            except AikitDtypePromotionError as e:
                _print_traceback_history()
                raise aikit.utils.exceptions.AikitDtypePromotionError(
                    fn.__name__, e, include_backend=True
                )
            except (IndexError, AikitIndexError) as e:
                _print_traceback_history()
                raise aikit.utils.exceptions.AikitIndexError(
                    fn.__name__, e, include_backend=True
                )
            except (AttributeError, AikitAttributeError) as e:
                _print_traceback_history()
                raise aikit.utils.exceptions.AikitAttributeError(
                    fn.__name__, e, include_backend=True
                )
            except (ValueError, AikitValueError) as e:
                _print_traceback_history()
                raise aikit.utils.exceptions.AikitValueError(
                    fn.__name__, e, include_backend=True
                )
            except (Exception, AikitBackendException) as e:
                _print_traceback_history()
                raise aikit.utils.exceptions.AikitBackendException(
                    fn.__name__, e, include_backend=True
                )

        _handle_exceptions.handle_exceptions = True
        return _handle_exceptions

The decorator is then added to each function for wrapping.
Let's look at an example of :func:`aikit.all`.

.. code-block:: python

    # in aikit/functional/aikit/utility.py
    @handle_exceptions
    def all(
        x: Union[aikit.Array, aikit.NativeArray],
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        out: Optional[aikit.Array] = None,
    ) -> aikit.Array:
        return aikit.current_backend(x).all(x, axis=axis, keepdims=keepdims, out=out)

When a backend throws an exception, it will be caught in the decorator and then the appropriate Error will be raised.
This ensures that all exceptions are consistent.

Let's look at the comparison of before and after adding the decorator.

**without decorator**

In NumPy,

.. code-block:: none

    >>> x = aikit.array([0,0,1])
    >>> aikit.all(x, axis=2)
    <error_stack>
    numpy.AxisError: axis 2 is out of bounds for an array of dimension 1

In PyTorch,

.. code-block:: none

    >>> x = aikit.array([0,0,1])
    >>> aikit.all(x, axis=2)
    <error_stack>
    IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 2)

The errors raised are different across backends, therefore confusing and inconsistent.

**with decorator**

In NumPy,

.. code-block:: none

    >>> x = aikit.array([0,0,1])
    >>> aikit.all(x, axis=2)
    <error_stack>
    AikitIndexError: numpy: all: AxisError: axis 2 is out of bounds for an array of dimension 1

In PyTorch,

    >>> x = aikit.array([0,0,1])
    >>> aikit.all(x, axis=2)
    <error_stack>
    AikitIndexError: torch: all: IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 2)

The errors are unified into an :code:`AikitIndexError`, with the current backend and function stated to provide clearer information.
The message string is inherited from the native exception.


Consistency in Errors
---------------------

For consistency, we make sure that the same type of Exception is raised for the same type of error regardless of the backend set.
Let's take an example of :func:`aikit.all` again. In Jax, :code:`ValueError` is raised when the axis is out of bounds,
and for Numpy, :code:`AxisError` is raised. To unify the behaviour, we raise :code:`AikitIndexError` for both cases.

In Numpy,

.. code-block:: python

    # in aikit/functional/backends/numpy/utility.py
    def all(
        x: np.ndarray,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        try:
            return np.asarray(np.all(x, axis=axis, keepdims=keepdims, out=out))
        except np.AxisError as e:
            raise aikit.utils.exceptions.AikitIndexError(error)

In Jax,

.. code-block:: python

    # in aikit/functional/backends/jax/utility.py
    def all(
        x: JaxArray,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        out: Optional[JaxArray] = None,
    ) -> JaxArray:
        x = jnp.array(x, dtype="bool")
        try:
            return jnp.all(x, axis, keepdims=keepdims)
        except ValueError as error:
            raise aikit.utils.exceptions.AikitIndexError(error)

In both cases, :code:`AikitIndexError` is raised, to make sure the same type of Exception is raised for this specific error.


Assertion Function
------------------

There are often conditions or limitations needed to ensure that a function is working correctly.

Inconsistency is observed such as some functions:

#. use :code:`assert` for checks and throw :code:`AssertionError`, or
#. use :code:`if/elif/else` blocks and raise :code:`Exception`, :code:`ValueError`, etc.

To unify the behaviours, our policy is to use conditional blocks and raise :code:`AikitException` whenever a check is required.
Moreover, to reduce code redundancy, conditions which are commonly used are collected as helper functions with custom parameters in :mod:`aikit/assertions.py`.
This allows them to be reused and promotes cleaner code.

Let's look at an example!

**Helper: check_less**

.. code-block:: python

    # in aikit/utils/assertions.py
    def check_less(x1, x2, allow_equal=False, message=""):
      # less_equal
      if allow_equal and aikit.any(x1 > x2):
          raise aikit.exceptions.AikitException(
              f"{x1} must be lesser than or equal to {x2}"
              if message == ""
              else message
          )
      # less
      elif not allow_equal and aikit.any(x1 >= x2):
          raise aikit.exceptions.AikitException(
              f"{x1} must be lesser than {x2}"
              if message == ""
              else message
          )

**aikit.set_split_factor**

.. code-block:: python

    # in aikit/functional/aikit/device.py
    @handle_exceptions
    def set_split_factor(
        factor: float,
        device: Union[aikit.Device, aikit.NativeDevice] = None,
        /,
    ) -> None:
        aikit.assertions.check_less(0, factor, allow_equal=True)
        global split_factors
        device = aikit.default(device, default_device())
        split_factors[device] = factor

Instead of coding a conditional block and raising an exception if the conditions are not met, a helper function is used to simplify the logic and increase code readability.

**Round Up**

This should have hopefully given you a good feel for how function wrapping is applied to functions in Aikit.

If you have any questions, please feel free to reach out on `discord`_ in the `exception handling thread`_!

**Video**

.. raw:: html

    <iframe width="420" height="315" allow="fullscreen;"
    src="https://www.youtube.com/embed/eTc24eG9P_s" class="video">
    </iframe>
