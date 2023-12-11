# flake8: noqa
# global
import copy
import functools
import numpy as np
from operator import mul
from typing import Optional

# local
import aikit
from .conversions import args_to_native, to_aikit
from .activations import _ArrayWithActivations
from .creation import _ArrayWithCreation
from .data_type import _ArrayWithDataTypes
from .device import _ArrayWithDevice
from .elementwise import _ArrayWithElementwise
from .general import _ArrayWithGeneral
from .gradients import _ArrayWithGradients
from .image import _ArrayWithImage
from .layers import _ArrayWithLayers
from .linear_algebra import _ArrayWithLinearAlgebra
from .losses import _ArrayWithLosses
from .manipulation import _ArrayWithManipulation
from .norms import _ArrayWithNorms
from .random import _ArrayWithRandom
from .searching import _ArrayWithSearching
from .set import _ArrayWithSet
from .sorting import _ArrayWithSorting
from .statistical import _ArrayWithStatistical
from .utility import _ArrayWithUtility
from aikit.func_wrapper import handle_view_indexing
from .experimental import (
    _ArrayWithSearchingExperimental,
    _ArrayWithActivationsExperimental,
    _ArrayWithConversionsExperimental,
    _ArrayWithCreationExperimental,
    _ArrayWithData_typeExperimental,
    _ArrayWithDeviceExperimental,
    _ArrayWithElementWiseExperimental,
    _ArrayWithGeneralExperimental,
    _ArrayWithGradientsExperimental,
    _ArrayWithImageExperimental,
    _ArrayWithLayersExperimental,
    _ArrayWithLinearAlgebraExperimental,
    _ArrayWithLossesExperimental,
    _ArrayWithManipulationExperimental,
    _ArrayWithNormsExperimental,
    _ArrayWithRandomExperimental,
    _ArrayWithSetExperimental,
    _ArrayWithSortingExperimental,
    _ArrayWithStatisticalExperimental,
    _ArrayWithUtilityExperimental,
)


class Array(
    _ArrayWithActivations,
    _ArrayWithCreation,
    _ArrayWithDataTypes,
    _ArrayWithDevice,
    _ArrayWithElementwise,
    _ArrayWithGeneral,
    _ArrayWithGradients,
    _ArrayWithImage,
    _ArrayWithLayers,
    _ArrayWithLinearAlgebra,
    _ArrayWithLosses,
    _ArrayWithManipulation,
    _ArrayWithNorms,
    _ArrayWithRandom,
    _ArrayWithSearching,
    _ArrayWithSet,
    _ArrayWithSorting,
    _ArrayWithStatistical,
    _ArrayWithUtility,
    _ArrayWithActivationsExperimental,
    _ArrayWithConversionsExperimental,
    _ArrayWithCreationExperimental,
    _ArrayWithData_typeExperimental,
    _ArrayWithDeviceExperimental,
    _ArrayWithElementWiseExperimental,
    _ArrayWithGeneralExperimental,
    _ArrayWithGradientsExperimental,
    _ArrayWithImageExperimental,
    _ArrayWithLayersExperimental,
    _ArrayWithLinearAlgebraExperimental,
    _ArrayWithLossesExperimental,
    _ArrayWithManipulationExperimental,
    _ArrayWithNormsExperimental,
    _ArrayWithRandomExperimental,
    _ArrayWithSearchingExperimental,
    _ArrayWithSetExperimental,
    _ArrayWithSortingExperimental,
    _ArrayWithStatisticalExperimental,
    _ArrayWithUtilityExperimental,
):
    def __init__(self, data, dynamic_backend=None):
        _ArrayWithActivations.__init__(self)
        _ArrayWithCreation.__init__(self)
        _ArrayWithDataTypes.__init__(self)
        _ArrayWithDevice.__init__(self)
        _ArrayWithElementwise.__init__(self)
        _ArrayWithGeneral.__init__(self)
        _ArrayWithGradients.__init__(self)
        _ArrayWithImage.__init__(self)
        _ArrayWithLayers.__init__(self)
        _ArrayWithLinearAlgebra.__init__(self)
        _ArrayWithLosses.__init__(self)
        _ArrayWithManipulation.__init__(self)
        _ArrayWithNorms.__init__(self)
        _ArrayWithRandom.__init__(self)
        _ArrayWithSearching.__init__(self)
        _ArrayWithSet.__init__(self)
        _ArrayWithSorting.__init__(self)
        _ArrayWithStatistical.__init__(self)
        _ArrayWithUtility.__init__(self)
        _ArrayWithActivationsExperimental.__init__(self),
        _ArrayWithConversionsExperimental.__init__(self),
        _ArrayWithCreationExperimental.__init__(self),
        _ArrayWithData_typeExperimental.__init__(self),
        _ArrayWithDeviceExperimental.__init__(self),
        _ArrayWithElementWiseExperimental.__init__(self),
        _ArrayWithGeneralExperimental.__init__(self),
        _ArrayWithGradientsExperimental.__init__(self),
        _ArrayWithImageExperimental.__init__(self),
        _ArrayWithLayersExperimental.__init__(self),
        _ArrayWithLinearAlgebraExperimental.__init__(self),
        _ArrayWithLossesExperimental.__init__(self),
        _ArrayWithManipulationExperimental.__init__(self),
        _ArrayWithNormsExperimental.__init__(self),
        _ArrayWithRandomExperimental.__init__(self),
        _ArrayWithSearchingExperimental.__init__(self),
        _ArrayWithSetExperimental.__init__(self),
        _ArrayWithSortingExperimental.__init__(self),
        _ArrayWithStatisticalExperimental.__init__(self),
        _ArrayWithUtilityExperimental.__init__(self),
        self._init(data, dynamic_backend)
        self._view_attributes(data)

    def _init(self, data, dynamic_backend=None):
        if aikit.is_aikit_array(data):
            self._data = data.data
        elif aikit.is_native_array(data):
            self._data = data
        elif isinstance(data, np.ndarray):
            self._data = aikit.asarray(data)._data
        elif aikit.is_aikit_sparse_array(data):
            self._data = data._data
        elif aikit.is_native_sparse_array(data):
            self._data = data._data
        else:
            raise aikit.utils.exceptions.AikitException(
                "data must be aikit array, native array or ndarray"
            )
        self._size = None
        self._strides = None
        self._itemsize = None
        self._dtype = None
        self._device = None
        self._dev_str = None
        self._pre_repr = None
        self._post_repr = None
        self._backend = aikit.current_backend(self._data).backend
        if dynamic_backend is not None:
            self._dynamic_backend = dynamic_backend
        else:
            self._dynamic_backend = aikit.dynamic_backend
        self.weak_type = False  # to handle 0-D jax front weak typed arrays

    def _view_attributes(self, data):
        self._base = None
        self._view_refs = []
        self._manipulation_stack = []
        self._torch_base = None
        self._torch_view_refs = []
        self._torch_manipulation = None

    # Properties #
    # ---------- #

    @property
    def backend(self):
        return self._backend

    @property
    def dynamic_backend(self):
        return self._dynamic_backend

    @dynamic_backend.setter
    def dynamic_backend(self, value):
        from aikit.functional.aikit.gradients import _variable
        from aikit.utils.backend.handler import _data_to_new_backend, _get_backend_for_arg

        if value:
            aikit_backend = aikit.with_backend(self._backend)

            if aikit_backend.gradients._is_variable(self.data):
                native_var = aikit_backend.gradients._variable_data(
                    self,
                )
                data = _data_to_new_backend(native_var, aikit_backend).data
                self._data = _variable(data).data

            else:
                self._data = _data_to_new_backend(self, aikit_backend).data

            self._backend = aikit.backend

        else:
            self._backend = _get_backend_for_arg(self.data.__class__.__module__).backend

        self._dynamic_backend = value

    @property
    def data(self) -> aikit.NativeArray:
        """The native array being wrapped in self."""
        return self._data

    @property
    def dtype(self) -> aikit.Dtype:
        """Data type of the array elements."""
        if self._dtype is None:
            self._dtype = aikit.dtype(self._data)
        return self._dtype

    @property
    def device(self) -> aikit.Device:
        """Hardware device the array data resides on."""
        if self._device is None:
            self._device = aikit.dev(self._data)
        return self._device

    @property
    def mT(self) -> aikit.Array:
        """Transpose of a matrix (or a stack of matrices).

        Returns
        -------
        ret
            array whose last two dimensions (axes) are permuted in reverse order
            relative to original array (i.e., for an array instance having shape
            ``(..., M, N)``, the returned array must have shape ``(..., N, M)``).
            The returned array must have the same data type as the original array.
        """
        aikit.utils.assertions.check_greater(
            len(self._data.shape), 2, allow_equal=True, as_array=False
        )
        return aikit.matrix_transpose(self._data)

    @property
    def ndim(self) -> int:
        """Number of array dimensions (axes)."""
        return len(tuple(self._data.shape))

    @property
    def shape(self) -> aikit.Shape:
        """Array dimensions."""
        return aikit.Shape(self._data.shape)

    @property
    def size(self) -> Optional[int]:
        """Number of elements in the array."""
        if self._size is None:
            if aikit.current_backend_str() in ["numpy", "jax"]:
                self._size = self._data.size
                return self._size
            self._size = (
                functools.reduce(mul, self._data.shape)
                if len(self._data.shape) > 0
                else 1
            )
        return self._size

    @property
    def itemsize(self) -> Optional[int]:
        """Size of array elements in bytes."""
        if self._itemsize is None:
            self._itemsize = aikit.itemsize(self._data)
        return self._itemsize

    @property
    def strides(self) -> Optional[int]:
        """Get strides across each dimension."""
        if self._strides is None:
            # for this to work consistently for non-contiguous arrays
            # we must pass self to aikit.strides, not self.data
            self._strides = aikit.strides(self)
        return self._strides

    @property
    def T(self) -> aikit.Array:
        """Transpose of the array.

        Returns
        -------
        ret
            two-dimensional array whose first and last dimensions (axes) are
            permuted in reverse order relative to original array.
        """
        aikit.utils.assertions.check_equal(len(self._data.shape), 2, as_array=False)
        return aikit.matrix_transpose(self._data)

    @property
    def base(self) -> aikit.Array:
        """Original array referenced by view."""
        return self._base

    @property
    def real(self) -> aikit.Array:
        """Real part of the array.

        Returns
        -------
        ret
            array containing the real part of each element in the array.
            The returned array must have the same shape and data type as
            the original array.
        """
        return aikit.real(self._data)

    @property
    def imag(self) -> aikit.Array:
        """Imaginary part of the array.

        Returns
        -------
        ret
            array containing the imaginary part of each element in the array.
            The returned array must have the same shape and data type as
            the original array.
        """
        return aikit.imag(self._data)

    # Setters #
    # --------#

    @data.setter
    def data(self, data):
        aikit.utils.assertions.check_true(
            aikit.is_native_array(data), "data must be native array"
        )
        self._init(data)

    # Built-ins #
    # ----------#

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs={}):
        args, kwargs = args_to_native(*args, **kwargs)
        return func(*args, **kwargs)

    def __aikit_array_function__(self, func, types, args, kwargs):
        # Cannot handle items that have __aikit_array_function__ other than those of
        # aikit arrays or native arrays.
        for t in types:
            if (
                hasattr(t, "__aikit_array_function__")
                and (t.__aikit_array_function__ is not aikit.Array.__aikit_array_function__)
                or (
                    hasattr(aikit.NativeArray, "__aikit_array_function__")
                    and (
                        t.__aikit_array_function__
                        is not aikit.NativeArray.__aikit_array_function__
                    )
                )
            ):
                return NotImplemented

        # Arguments contain no overrides, so we can safely call the
        # overloaded function again.
        return func(*args, **kwargs)

    def __array__(self, *args, **kwargs):
        args, kwargs = args_to_native(*args, **kwargs)
        return self._data.__array__(*args, dtype=self.dtype, **kwargs)

    def __array_prepare__(self, *args, **kwargs):
        args, kwargs = args_to_native(*args, **kwargs)
        return self._data.__array_prepare__(*args, **kwargs)

    def __array_ufunc__(self, *args, **kwargs):
        args, kwargs = args_to_native(*args, **kwargs)
        return self._data.__array_ufunc__(*args, **kwargs)

    def __array_wrap__(self, *args, **kwargs):
        args, kwargs = args_to_native(*args, **kwargs)
        return self._data.__array_wrap__(*args, **kwargs)

    def __array_namespace__(self, api_version=None):
        return aikit

    def __repr__(self):
        if self._dev_str is None:
            self._dev_str = aikit.as_aikit_dev(self.device)
            self._pre_repr = "aikit.array"
            if "gpu" in self._dev_str:
                self._post_repr = f", dev={self._dev_str})"
            else:
                self._post_repr = ")"
        sig_fig = aikit.array_significant_figures
        dec_vals = aikit.array_decimal_values
        backend = aikit.with_backend(self.backend)
        arr_np = backend.to_numpy(self._data)
        rep = (
            np.array(aikit.vec_sig_fig(arr_np, sig_fig))
            if self.size > 0
            else np.array(arr_np)
        )
        with np.printoptions(precision=dec_vals):
            repr = rep.__repr__()[:-1].partition(", dtype")[0].partition(", dev")[0]
            return (
                self._pre_repr
                + repr[repr.find("(") :]
                + self._post_repr.format(aikit.current_backend_str())
            )

    def __dir__(self):
        return self._data.__dir__()

    def __getattribute__(self, item):
        return super().__getattribute__(item)

    def __getattr__(self, item):
        try:
            attr = self._data.__getattribute__(item)
        except AttributeError:
            attr = self._data.__getattr__(item)
        return to_aikit(attr)

    @handle_view_indexing
    def __getitem__(self, query):
        return aikit.get_item(self._data, query)

    def __setitem__(self, query, val):
        self._data = aikit.set_item(self._data, query, val)._data

    def __contains__(self, key):
        return self._data.__contains__(key)

    def __getstate__(self):
        data_dict = {}

        # only pickle the native array
        data_dict["data"] = self.data

        # also store the local aikit framework that created this array
        data_dict["backend"] = self.backend
        data_dict["device_str"] = aikit.as_aikit_dev(self.device)

        return data_dict

    def __setstate__(self, state):
        # we can construct other details of aikit.Array
        # just by re-creating the aikit.Array using the native array

        # get the required backend
        (
            aikit.set_backend(state["backend"])
            if state["backend"] is not None and len(state["backend"]) > 0
            else aikit.current_backend(state["data"])
        )
        aikit_array = aikit.array(state["data"])
        aikit.previous_backend()

        self.__dict__ = aikit_array.__dict__

        # TODO: what about placement of the array on the right device ?
        # device = backend.as_native_dev(state["device_str"])
        # backend.to_device(self, device)

    def __pos__(self):
        return aikit.positive(self._data)

    def __neg__(self):
        return aikit.negative(self._data)

    def __pow__(self, power):
        """aikit.Array special method variant of aikit.pow. This method simply
        wraps the function, and so the docstring for aikit.pow also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Input array or float.
        power
            Array or float power. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        ret
            an array containing the element-wise sums. The returned array must have a
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Array` input:

        >>> x = aikit.array([1, 2, 3])
        >>> y = x ** 2
        >>> print(y)
        aikit.array([1, 4, 9])

        >>> x = aikit.array([1.2, 2.1, 3.5])
        >>> y = x ** 2.9
        >>> print(y)
        aikit.array([ 1.69678056,  8.59876156, 37.82660675])
        """
        return aikit.pow(self._data, power)

    def __rpow__(self, power):
        return aikit.pow(power, self._data)

    def __ipow__(self, power):
        return aikit.pow(self._data, power)

    def __add__(self, other):
        """aikit.Array special method variant of aikit.add. This method simply
        wraps the function, and so the docstring for aikit.add also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a numeric data type.
        other
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        ret
            an array containing the element-wise sums. The returned array must have a
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.array([1, 2, 3])
        >>> y = aikit.array([4, 5, 6])
        >>> z = x + y
        >>> print(z)
        aikit.array([5, 7, 9])
        """
        return aikit.add(self._data, other)

    def __radd__(self, other):
        """aikit.Array reverse special method variant of aikit.add. This method
        simply wraps the function, and so the docstring for aikit.add also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a numeric data type.
        other
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        ret
            an array containing the element-wise sums. The returned array must have a
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = 1
        >>> y = aikit.array([4, 5, 6])
        >>> z = x + y
        >>> print(z)
        aikit.array([5, 6, 7])
        """
        return aikit.add(other, self._data)

    def __iadd__(self, other):
        return aikit.add(self._data, other)

    def __sub__(self, other):
        """aikit.Array special method variant of aikit.subtract. This method simply
        wraps the function, and so the docstring for aikit.subtract also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a numeric data type.
        other
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        ret
            an array containing the element-wise differences. The returned array must have a
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Array` instances only:

        >>> x = aikit.array([1, 2, 3])
        >>> y = aikit.array([4, 5, 6])
        >>> z = x - y
        >>> print(z)
        aikit.array([-3, -3, -3])
        """
        return aikit.subtract(self._data, other)

    def __rsub__(self, other):
        """aikit.Array reverse special method variant of aikit.subtract. This
        method simply wraps the function, and so the docstring for aikit.subtract
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a numeric data type.
        other
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        ret
            an array containing the element-wise differences. The returned array must have a
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = 1
        >>> y = aikit.array([4, 5, 6])
        >>> z = x - y
        >>> print(z)
        aikit.array([-3, -4, -5])
        """
        return aikit.subtract(other, self._data)

    def __isub__(self, other):
        return aikit.subtract(self._data, other)

    def __mul__(self, other):
        return aikit.multiply(self._data, other)

    def __rmul__(self, other):
        return aikit.multiply(other, self._data)

    def __imul__(self, other):
        return aikit.multiply(self._data, other)

    def __mod__(self, other):
        return aikit.remainder(self._data, other)

    def __rmod__(self, other):
        return aikit.remainder(other, self._data)

    def __imod__(self, other):
        return aikit.remainder(self._data, other)

    def __divmod__(self, other):
        return aikit.divide(self._data, other), aikit.remainder(self._data, other)

    def __rdivmod__(self, other):
        return aikit.divide(other, self._data), aikit.remainder(other, self._data)

    def __truediv__(self, other):
        """aikit.Array reverse special method variant of aikit.divide. This method
        simply wraps the function, and so the docstring for aikit.divide also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have a numeric data type.
        other
            second input array. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array must have a
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = aikit.array([1, 2, 3])
        >>> y = aikit.array([4, 5, 6])
        >>> z = x / y
        >>> print(z)
        aikit.array([0.25      , 0.40000001, 0.5       ])
        """
        return aikit.divide(self._data, other)

    def __rtruediv__(self, other):
        return aikit.divide(other, self._data)

    def __itruediv__(self, other):
        return aikit.divide(self._data, other)

    def __floordiv__(self, other):
        return aikit.floor_divide(self._data, other)

    def __rfloordiv__(self, other):
        return aikit.floor_divide(other, self._data)

    def __ifloordiv__(self, other):
        return aikit.floor_divide(self._data, other)

    def __matmul__(self, other):
        return aikit.matmul(self._data, other)

    def __rmatmul__(self, other):
        return aikit.matmul(other, self._data)

    def __imatmul__(self, other):
        return aikit.matmul(self._data, other)

    def __abs__(self):
        """aikit.Array special method variant of aikit.abs. This method simply
        wraps the function, and so the docstring for aikit.abs also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.

        Returns
        -------
        ret
            an array containing the absolute value of each element
            in ``self``. The returned array must have the same data
            type as ``self``.

        Examples
        --------
        With :class:`aikit.Array` input:

        >>> x = aikit.array([6, -2, 0, -1])
        >>> print(abs(x))
        aikit.array([6, 2, 0, 1])

        >>> x = aikit.array([-1.2, 1.2])
        >>> print(abs(x))
        aikit.array([1.2, 1.2])
        """
        return aikit.abs(self._data)

    def __float__(self):
        if hasattr(self._data, "__float__"):
            if "complex" in self.dtype:
                res = float(self.real)
            else:
                res = self._data.__float__()
        else:
            res = float(aikit.to_scalar(self._data))
        if res is NotImplemented:
            return res
        return to_aikit(res)

    def __int__(self):
        if hasattr(self._data, "__int__"):
            if "complex" in self.dtype:
                res = int(self.real)
            else:
                res = self._data.__int__()
        else:
            res = int(aikit.to_scalar(self._data))
        if res is NotImplemented:
            return res
        return to_aikit(res)

    def __complex__(self):
        res = complex(aikit.to_scalar(self._data))
        if res is NotImplemented:
            return res
        return to_aikit(res)

    def __bool__(self):
        return self._data.__bool__()

    def __dlpack__(self, stream=None):
        # Not completely supported yet as paddle and tf
        # doesn't support __dlpack__ and __dlpack_device__ dunders right now
        # created issues
        # paddle https://github.com/PaddlePaddle/Paddle/issues/56891
        # tf https://github.com/tensorflow/tensorflow/issues/61769
        return aikit.to_dlpack(self)

    def __dlpack_device__(self):
        return self._data.__dlpack_device__()

    def __lt__(self, other):
        """aikit.Array special method variant of aikit.less. This method simply
        wraps the function, and so the docstring for aikit.less also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            first input array. May have any data type.
        other
            second input array. Must be compatible with x1 (with Broadcasting). May have any
            data type.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array must have a
            data type of bool.

        Examples
        --------
        >>> x = aikit.array([6, 2, 3])
        >>> y = aikit.array([4, 5, 3])
        >>> z = x < y
        >>> print(z)
        aikit.array([ False, True, False])
        """
        return aikit.less(self._data, other)

    def __le__(self, other):
        """aikit.Array special method variant of aikit.less_equal. This method
        simply wraps the function, and so the docstring for aikit.less_equal also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. May have any data type.
        other
            second input array. Must be compatible with x1 (with Broadcasting). May have any
            data type.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array must have a
            data type of bool.

        Examples
        --------
        >>> x = aikit.array([6, 2, 3])
        >>> y = aikit.array([4, 5, 3])
        >>> z = x <= y
        >>> print(z)
        aikit.array([ False, True, True])
        """
        return aikit.less_equal(self._data, other)

    def __eq__(self, other):
        """aikit.Array special method variant of aikit.equal. This method simply
        wraps the function, and so the docstring for aikit.equal also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            first input array. May have any data type.
        other
            second input array. Must be compatible with x1 (with Broadcasting). May have any
            data type.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array must have a
            data type of bool.

        Examples
        --------
        With :class:`aikit.Array` instances:

        >>> x1 = aikit.array([1, 0, 1, 1])
        >>> x2 = aikit.array([1, 0, 0, -1])
        >>> y = x1 == x2
        >>> print(y)
        aikit.array([True, True, False, False])

        >>> x1 = aikit.array([1, 0, 1, 0])
        >>> x2 = aikit.array([0, 1, 0, 1])
        >>> y = x1 == x2
        >>> print(y)
        aikit.array([False, False, False, False])
        """
        return aikit.equal(self._data, other)

    def __ne__(self, other):
        """aikit.Array special method variant of aikit.not_equal. This method
        simply wraps the function, and so the docstring for aikit.not_equal also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. May have any data type.
        other
            second input array. Must be compatible with x1 (with Broadcasting). May have any
            data type.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array must have a
            data type of bool.

        Examples
        --------
        With :class:`aikit.Array` instances:

        >>> x1 = aikit.array([1, 0, 1, 1])
        >>> x2 = aikit.array([1, 0, 0, -1])
        >>> y = x1 != x2
        >>> print(y)
        aikit.array([False, False, True, True])

        >>> x1 = aikit.array([1, 0, 1, 0])
        >>> x2 = aikit.array([0, 1, 0, 1])
        >>> y = x1 != x2
        >>> print(y)
        aikit.array([True, True, True, True])
        """
        return aikit.not_equal(self._data, other)

    def __gt__(self, other):
        """aikit.Array special method variant of aikit.greater. This method simply
        wraps the function, and so the docstring for aikit.greater also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. May have any data type.
        other
            second input array. Must be compatible with x1 (with Broadcasting). May have any
            data type.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array must have a
            data type of bool.

        Examples
        --------
        With :class:`aikit.Array` instances:

        >>> x = aikit.array([6, 2, 3])
        >>> y = aikit.array([4, 5, 3])
        >>> z = x > y
        >>> print(z)
        aikit.array([True,False,False])

        With mix of :class:`aikit.Array` and :class:`aikit.Container` instances:

        >>> x = aikit.array([[5.1, 2.3, -3.6]])
        >>> y = aikit.Container(a=aikit.array([[4.], [5.1], [6.]]),b=aikit.array([[-3.6], [6.], [7.]]))
        >>> z = x > y
        >>> print(z)
        {
            a: aikit.array([[True, False, False],
                          [False, False, False],
                          [False, False, False]]),
            b: aikit.array([[True, True, False],
                          [False, False, False],
                          [False, False, False]])
        }
        """
        return aikit.greater(self._data, other)

    def __ge__(self, other):
        """aikit.Array special method variant of aikit.greater_equal. This method
        simply wraps the function, and so the docstring for aikit.bitwise_xor
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. May have any data type.
        other
            second input array. Must be compatible with x1 (with Broadcasting). May have any
            data type.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array must have a
            data type of bool.

        Examples
        --------
        With :class:`aikit.Array` instances:

        >>> x = aikit.array([6, 2, 3])
        >>> y = aikit.array([4, 5, 6])
        >>> z = x >= y
        >>> print(z)
        aikit.array([True,False,False])

        With mix of :class:`aikit.Array` and :class:`aikit.Container` instances:

        >>> x = aikit.array([[5.1, 2.3, -3.6]])
        >>> y = aikit.Container(a=aikit.array([[4.], [5.1], [6.]]),b=aikit.array([[5.], [6.], [7.]]))
        >>> z = x >= y
        >>> print(z)
        {
            a: aikit.array([[True, False, False],
                          [True, False, False],
                          [False, False, False]]),
            b: aikit.array([[True, False, False],
                          [False, False, False],
                          [False, False, False]])
        }
        """
        return aikit.greater_equal(self._data, other)

    def __and__(self, other):
        return aikit.bitwise_and(self._data, other)

    def __rand__(self, other):
        return aikit.bitwise_and(other, self._data)

    def __iand__(self, other):
        return aikit.bitwise_and(self._data, other)

    def __or__(self, other):
        return aikit.bitwise_or(self._data, other)

    def __ror__(self, other):
        return aikit.bitwise_or(other, self._data)

    def __ior__(self, other):
        return aikit.bitwise_or(self._data, other)

    def __invert__(self):
        return aikit.bitwise_invert(self._data)

    def __xor__(self, other):
        """aikit.Array special method variant of aikit.bitwise_xor. This method
        simply wraps the function, and so the docstring for aikit.bitwise_xor
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array. Should have an integer or boolean data type.
        other
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
        out
            optional output array, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array must have a
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Array` instances:

        >>> a = aikit.array([1, 2, 3])
        >>> b = aikit.array([3, 2, 1])
        >>> y = a ^ b
        >>> print(y)
        aikit.array([2,0,2])

        With mix of :class:`aikit.Array` and :class:`aikit.Container` instances:

        >>> x = aikit.Container(a = aikit.array([-67, 21]))
        >>> y = aikit.array([12, 13])
        >>> z = x ^ y
        >>> print(z)
        {a: aikit.array([-79, 24])}
        """
        return aikit.bitwise_xor(self._data, other)

    def __rxor__(self, other):
        return aikit.bitwise_xor(other, self._data)

    def __ixor__(self, other):
        return aikit.bitwise_xor(self._data, other)

    def __lshift__(self, other):
        return aikit.bitwise_left_shift(self._data, other)

    def __rlshift__(self, other):
        return aikit.bitwise_left_shift(other, self._data)

    def __ilshift__(self, other):
        return aikit.bitwise_left_shift(self._data, other)

    def __rshift__(self, other):
        """aikit.Array special method variant of aikit.bitwise_right_shift. This
        method simply wraps the function, and so the docstring for
        aikit.bitwise_right_shift also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            first input array. Should have an integer data type.
        other
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`).
            Should have an integer data type. Each element must be greater than or equal
            to ``0``.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array must have
            a data type determined by :ref:`type-promotion`.

        Examples
        --------
        With :class:`aikit.Array` instances only:

        >>> a = aikit.array([2, 3, 4])
        >>> b = aikit.array([0, 1, 2])
        >>> y = a >> b
        >>> print(y)
        aikit.array([2, 1, 1])
        """
        return aikit.bitwise_right_shift(self._data, other)

    def __rrshift__(self, other):
        """aikit.Array reverse special method variant of aikit.bitwise_right_shift.
        This method simply wraps the function, and so the docstring for
        aikit.bitwise_right_shift also applies to this method with minimal
        changes.

        Parameters
        ----------
        self
            first input array. Should have an integer data type.
        other
            second input array. Must be compatible with ``x1`` (see :ref:`broadcasting`).
            Should have an integer data type. Each element must be greater than or equal
            to ``0``.

        Returns
        -------
        ret
            an array containing the element-wise results. The returned array must have
            a data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> a = 32
        >>> b = aikit.array([0, 1, 2])
        >>> y = a >> b
        >>> print(y)
        aikit.array([32, 16,  8])
        """
        return aikit.bitwise_right_shift(other, self._data)

    def __irshift__(self, other):
        return aikit.bitwise_right_shift(self._data, other)

    def __deepcopy__(self, memodict={}):
        try:
            return to_aikit(self._data.__deepcopy__(memodict))
        except AttributeError:
            # ToDo: try and find more elegant solution to jax inability to
            #  deepcopy device arrays
            if aikit.current_backend_str() == "jax":
                np_array = copy.deepcopy(self._data)
                jax_array = aikit.array(np_array)
                return to_aikit(jax_array)
            return to_aikit(copy.deepcopy(self._data))
        except RuntimeError:
            from aikit.functional.aikit.gradients import _is_variable

            # paddle and torch don't support the deepcopy protocol on non-leaf tensors
            if _is_variable(self):
                return to_aikit(copy.deepcopy(aikit.stop_gradient(self)._data))
            return to_aikit(copy.deepcopy(self._data))

    def __len__(self):
        if not len(self._data.shape):
            return 0
        try:
            return len(self._data)
        except TypeError:
            return self._data.shape[0]

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d aikit.Array not supported")
        if aikit.current_backend_str() == "paddle":
            if self.dtype in ["int8", "int16", "uint8", "float16"]:
                return iter([to_aikit(i) for i in aikit.unstack(self._data)])
            elif self.ndim == 1:
                return iter([to_aikit(i).squeeze(axis=0) for i in self._data])
        return iter([to_aikit(i) for i in self._data])
