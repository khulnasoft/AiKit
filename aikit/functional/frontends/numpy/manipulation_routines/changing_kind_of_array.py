# local
import aikit.functional.frontends.numpy as np_frontend
import aikit


def asmatrix(data, dtype=None):
    return np_frontend.matrix(aikit.array(data), dtype=dtype, copy=False)


def asscalar(a):
    return a.item()
