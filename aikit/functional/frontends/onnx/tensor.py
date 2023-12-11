# global

# local
import aikit

# import aikit.functional.frontends.onnx as onnx_frontend


class Tensor:
    def __init__(self, array):
        self._aikit_array = (
            aikit.array(array) if not isinstance(array, aikit.Array) else array
        )

    def __len__(self):
        return len(self._aikit_array)

    def __repr__(self):
        return str(self.aikit_array.__repr__()).replace(
            "aikit.array", "aikit.frontends.onnx.Tensor"
        )

    # Properties #
    # ---------- #

    @property
    def aikit_array(self):
        return self._aikit_array

    @property
    def device(self):
        return self.aikit_array.device

    @property
    def dtype(self):
        return self.aikit_array.dtype

    @property
    def shape(self):
        return self.aikit_array.shape

    @property
    def ndim(self):
        return self.aikit_array.ndim

    # Setters #
    # --------#

    @aikit_array.setter
    def aikit_array(self, array):
        self._aikit_array = (
            aikit.array(array) if not isinstance(array, aikit.Array) else array
        )
