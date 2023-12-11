import aikit
import aikit.functional.frontends.numpy as np_frontend
import numpy as np


masked = True
masked_print_options = "--"
nomask = False


# Class #
# ----- #


class MaskedArray(np_frontend.ndarray):
    def __init__(
        self,
        data,
        mask=nomask,
        dtype=None,
        copy=False,
        ndmin=0,
        fill_value=None,
        keep_mask=True,
        hard_mask=False,
        shrink=True,
        subok=True,
        order=None,
    ):
        self._init_data(data, dtype, mask, keep_mask)
        self._init_fill_value(fill_value)
        self._init_ndmin(ndmin)
        self._init_hard_mask(hard_mask)
        # shrink
        if shrink and not aikit.any(self._mask):
            self._mask = aikit.array(False)
        # copy
        if copy:
            self._data = aikit.copy_array(self._data)
            self._mask = aikit.copy_array(self._mask)

    def _init_data(self, data, dtype, mask, keep_mask):
        if _is_masked_array(data):
            self._data = (
                aikit.array(data.data, dtype=dtype)
                if aikit.exists(dtype)
                else aikit.array(data.data)
            )
            self._init_mask(mask)
            if keep_mask:
                if not isinstance(data.mask, bool):
                    aikit.utils.assertions.check_equal(
                        aikit.shape(self._mask),
                        aikit.shape(data.mask),
                        message="shapes of input mask does not match current mask",
                        as_array=False,
                    )
                self._mask = aikit.bitwise_or(self._mask, data.mask)
        else:
            self._data = (
                aikit.array(data, dtype=dtype) if aikit.exists(dtype) else aikit.array(data)
            )
            self._init_mask(mask)
        self._dtype = self._data.dtype

    def _init_mask(self, mask):
        if isinstance(mask, list) or aikit.is_array(mask):
            aikit.utils.assertions.check_equal(
                aikit.shape(self._data),
                aikit.shape(aikit.array(mask)),
                message="shapes of data and mask must match",
                as_array=False,
            )
            self._mask = aikit.array(mask)
        elif mask.all():
            self._mask = aikit.ones_like(self._data)
        else:
            self._mask = aikit.zeros_like(self._data)
        self._mask = self._mask.astype("bool")

    def _init_fill_value(self, fill_value):
        if aikit.exists(fill_value):
            self._fill_value = aikit.array(fill_value, dtype=self._dtype)
        elif aikit.is_bool_dtype(self._dtype):
            self._fill_value = aikit.array(True)
        elif aikit.is_int_dtype(self._dtype):
            self._fill_value = aikit.array(999999, dtype="int64")
        else:
            self._fill_value = aikit.array(1e20, dtype="float64")

    def _init_ndmin(self, ndmin):
        aikit.utils.assertions.check_isinstance(ndmin, int)
        if ndmin > len(aikit.shape(self._data)):
            self._data = aikit.expand_dims(self._data, axis=0)
            self._mask = aikit.expand_dims(self._mask, axis=0)

    def _init_hard_mask(self, hard_mask):
        aikit.utils.assertions.check_isinstance(hard_mask, bool)
        self._hard_mask = hard_mask

    # Properties #
    # ---------- #

    @property
    def data(self):
        return self._data

    @property
    def mask(self):
        return self._mask

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def hardmask(self):
        return self._hard_mask

    @property
    def dtype(self):
        return self._dtype

    # Setter #
    # ------ #

    @mask.setter
    def mask(self, mask):
        self._init_mask(mask)

    @fill_value.setter
    def fill_value(self, fill_value):
        self._init_fill_value(fill_value)

    # Built-ins #
    # --------- #

    def __getitem__(self, query):
        if self._mask.shape != self._data.shape:
            self._mask = aikit.ones_like(self._data, dtype=aikit.bool) * self._mask
        if self._fill_value.shape != self._data.shape:
            self._fill_value = aikit.ones_like(self._data) * self._fill_value
        if hasattr(self._mask[query], "shape"):
            return MaskedArray(
                data=self._data[query],
                mask=self._mask[query],
                fill_value=self._fill_value[query],
                hard_mask=self._hard_mask,
            )

    def __setitem__(self, query, val):
        self._data[query] = val
        if self._mask.shape != self._data.shape:
            self._mask = aikit.ones_like(self._data, dtype=aikit.bool) * self._mask
        val_mask = aikit.ones_like(self._mask[query]) * getattr(val, "_mask", False)
        if self._hard_mask:
            self._mask[query] |= val_mask
        else:
            self._mask[query] = val_mask
        return self

    def __repr__(self):
        dec_vals = aikit.array_decimal_values
        with np.printoptions(precision=dec_vals):
            return (
                "aikit.MaskedArray("
                + self._array_in_str()
                + ",\n\tmask="
                + str(self._mask.to_list())
                + ",\n\tfill_value="
                + str(self._fill_value.to_list())
                + "\n)"
            )

    def _array_in_str(self):
        # check if we have unsized array
        if self._data.shape == ():
            if self._mask:
                return masked_print_options
            return str(self._data.to_list())
        if aikit.any(self._mask):
            return str(
                [
                    masked_print_options if mask else x
                    for x, mask in zip(self._data.to_list(), self._mask.to_list())
                ]
            )
        return str(self._data.to_list())


# --- Helpers --- #
# --------------- #


def _is_masked_array(x):
    return isinstance(x, (np.ma.MaskedArray, np_frontend.ma.MaskedArray))


# Instance Methods #
# ---------------- #

# TODO


# masked_array (alias)
masked_array = MaskedArray
