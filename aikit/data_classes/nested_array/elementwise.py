# global
from typing import Optional, Union

# local
import aikit
from .base import NestedArrayBase


class NestedArrayElementwise(NestedArrayBase):
    @staticmethod
    def static_add(
        x1: Union[NestedArrayBase, aikit.Array, aikit.NestedArray],
        x2: Union[NestedArrayBase, aikit.Array, aikit.NestedArray],
        /,
        *,
        alpha: Optional[Union[int, float]] = None,
        out: Optional[aikit.Array] = None,
    ) -> NestedArrayBase:
        pass
        # return self._elementwise_op(other, aikit.add)
