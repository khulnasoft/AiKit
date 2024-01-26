import aikit

from aikit.functional.frontends.onnx.func_wrapper import to_aikit_arrays_and_back


@to_aikit_arrays_and_back
def MatMul(x1, x2):
    return aikit.matmul(x1, x2)
