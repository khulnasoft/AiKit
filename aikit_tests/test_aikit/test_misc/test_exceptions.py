import sys
import os
import contextlib
import pytest
import aikit


@pytest.mark.parametrize("trace_mode", ["full", "aikit", "frontend"])
def test_get_trace_mode(trace_mode, backend_fw):
    aikit.set_backend(backend_fw)
    aikit.set_exception_trace_mode(trace_mode)
    aikit.set_exception_trace_mode("aikit")
    aikit.utils.assertions.check_equal(aikit.exception_trace_mode, "aikit", as_array=False)
    aikit.previous_backend()


@pytest.mark.parametrize("trace_mode", ["full", "aikit", "frontend"])
def test_set_trace_mode(trace_mode, backend_fw):
    aikit.set_backend(backend_fw)
    aikit.set_exception_trace_mode(trace_mode)
    aikit.utils.assertions.check_equal(
        aikit.exception_trace_mode, trace_mode, as_array=False
    )
    aikit.previous_backend()


@pytest.mark.parametrize("trace_mode", ["full", "aikit", "frontend"])
@pytest.mark.parametrize("show_func_wrapper", [True, False])
def test_trace_modes(backend_fw, trace_mode, show_func_wrapper):
    aikit.set_backend(backend_fw)
    filename = "excep_out.txt"
    orig_stdout = sys.stdout
    with open(filename, "w") as f:
        sys.stdout = f
        aikit.set_exception_trace_mode(trace_mode)
        aikit.set_show_func_wrapper_trace_mode(show_func_wrapper)
        x = aikit.array([])
        y = aikit.array([1.0, 3.0, 4.0])
        lines = ""
        try:
            aikit.divide(x, y)
        except Exception as e:
            print(e)
        sys.stdout = orig_stdout
    with open(filename) as f:
        lines += f.read()

    if trace_mode == "full" and not show_func_wrapper:
        assert "/func_wrapper.py" not in lines
        assert "/aikit/functional/backends" in lines
        if backend_fw.current_backend_str() not in ["torch", "numpy"]:
            assert "/dist-packages" in lines

    if trace_mode == "full" and show_func_wrapper:
        assert "/func_wrapper.py" in lines
        assert "/aikit/functional/backends" in lines
        if backend_fw.current_backend_str() not in ["torch", "numpy"]:
            assert "/dist-packages" in lines

    if trace_mode in ["aikit", "frontend"]:
        if not show_func_wrapper:
            assert "/func_wrapper.py" not in lines
            assert "/dist-packages" not in lines

        if show_func_wrapper:
            if trace_mode == "frontend":
                assert "/aikit/functional/backends" not in lines
            else:
                assert "/func_wrapper.py" in lines
            assert "/dist-packages" not in lines

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)
    aikit.previous_backend()


@pytest.mark.parametrize("trace_mode", ["full", "aikit", "frontend"])
def test_unset_trace_mode(trace_mode, backend_fw):
    aikit.set_backend(backend_fw)
    aikit.set_exception_trace_mode(trace_mode)
    aikit.set_exception_trace_mode("aikit")
    aikit.utils.assertions.check_equal(aikit.exception_trace_mode, "aikit", as_array=False)
    aikit.unset_exception_trace_mode()
    aikit.utils.assertions.check_equal(
        aikit.exception_trace_mode, trace_mode, as_array=False
    )
    aikit.previous_backend()
