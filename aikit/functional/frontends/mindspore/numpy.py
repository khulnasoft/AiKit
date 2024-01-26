import aikit


def array(obj, dtype=None, copy=True, ndmin=4):
    ret = aikit.array(obj, dtype=dtype, copy=copy)
    while ndmin > len(ret.shape):
        ret = aikit.expand_dims(ret, axis=0)
    return ret
