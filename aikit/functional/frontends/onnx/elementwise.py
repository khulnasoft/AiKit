import aikit

from aikit.functional.frontends.onnx.func_wrapper import to_aikit_arrays_and_back


@to_aikit_arrays_and_back
def Abs(input):
    return aikit.abs(input)


@to_aikit_arrays_and_back
def Acos(input):
    return aikit.acos(input)


@to_aikit_arrays_and_back
def Acosh(input):
    return aikit.acosh(input)


@to_aikit_arrays_and_back
def Add(x1, x2):
    return aikit.add(x1, x2)


@to_aikit_arrays_and_back
def Asin(input):
    return aikit.asin(input)
