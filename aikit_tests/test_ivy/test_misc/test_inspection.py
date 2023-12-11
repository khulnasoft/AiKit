# global
import pytest
from typing import List, Tuple, Dict, Optional, Union

# local
import aikit


# --- Helpers --- #
# --------------- #


def _fn0(xs: Optional[List[aikit.Array]] = None):
    return xs


def _fn1(
    a: Union[aikit.Array, aikit.NativeArray],
    b: str = "hello",
    c: Optional[int] = None,
    d: aikit.NativeArray = None,
):
    return a, b, c, d


def _fn2(
    a: Tuple[Union[aikit.Array, aikit.NativeArray, aikit.Container]],
    bs: Tuple[str] = ("a", "b", "c"),
    cs: Optional[Dict[str, aikit.Array]] = None,
):
    return a, bs, cs


# --- Main --- #
# ------------ #


@pytest.mark.parametrize(
    "fn_n_spec",
    [
        (_fn0, [[(0, "xs"), "optional", int]]),
        (_fn1, [[(0, "a")], [(3, "d"), "optional"]]),
        (_fn2, [[(0, "a"), int], [(2, "cs"), "optional", str]]),
    ],
)
def test_fn_array_spec(fn_n_spec, backend_fw):
    aikit.set_backend(backend_fw)
    fn, spec = fn_n_spec
    assert aikit.fn_array_spec(fn) == spec
    aikit.previous_backend()
