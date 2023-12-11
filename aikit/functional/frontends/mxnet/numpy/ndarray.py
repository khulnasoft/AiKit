# global

# local
import aikit
import aikit.functional.frontends.mxnet as mxnet_frontend


class ndarray:
    def __init__(self, array):
        self._aikit_array = (
            aikit.array(array) if not isinstance(array, aikit.Array) else array
        )

    def __repr__(self):
        return str(self.aikit_array.__repr__()).replace(
            "aikit.array", "aikit.frontends.mxnet.numpy.array"
        )

    # Properties #
    # ---------- #

    @property
    def aikit_array(self):
        return self._aikit_array

    @property
    def dtype(self):
        return self.aikit_array.dtype

    @property
    def shape(self):
        return self.aikit_array.shape

    # Instance Methods #
    # ---------------- #

    def __add__(self, other):
        return mxnet_frontend.numpy.add(self, other)
