# Global
import pytest
import itertools
from hypothesis import strategies as st, given, settings, HealthCheck

# Local
import aikit
import numpy as np
from aikit.utils.backend.handler import _backend_dict


def test_is_local(backend_fw):
    local_aikit = aikit.with_backend(backend_fw)
    assert local_aikit.is_local()


@settings(
    # To be able to share traced_backends between examples
    suppress_health_check=[HealthCheck(9)]
)
@given(name=st.sampled_from(["add", "Array", "Container", "globals_vars"]))
def test_memory_id(name, traced_backends):
    for b in traced_backends:
        assert id(getattr(aikit, name)) != id(
            getattr(b, name)
        ), f"Shared object {name} between global Aikit and backend {b.backend}"

    for comb in itertools.combinations(traced_backends, 2):
        assert id(getattr(comb[0], name)) != id(getattr(comb[1], name)), (
            f"Shared object {name} between {comb[0].backend} and backend "
            f"{comb[1].backend}"
        )


def test_prevent_access(backend_fw):
    local_aikit = aikit.with_backend(backend_fw)
    with pytest.raises(RuntimeError):
        local_aikit.with_backend(backend_fw)

    with pytest.raises(RuntimeError):
        local_aikit.set_backend(backend_fw)


def test_with_backend_array(backend_fw):
    local_aikit = aikit.with_backend(backend_fw)
    local_x = local_aikit.array([1, 2, 3, 4])
    aikit.set_backend(backend_fw)
    x = aikit.array([1, 2, 3, 4])
    assert np.allclose(x._data, local_x._data)


def test_with_backend_cached(backend_fw):
    non_cached_local_aikit = aikit.with_backend(backend_fw)
    cached_local_aikit = aikit.with_backend(backend_fw)
    assert non_cached_local_aikit == cached_local_aikit


@pytest.fixture
def traced_backends():
    traced_backends = []
    for b in _backend_dict:
        _b = aikit.with_backend(b)
        traced_backends.append(_b)
    return traced_backends
