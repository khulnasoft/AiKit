import aikit
from aikit.functional.frontends.tensorflow.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_tf_dtype,
)
from aikit.func_wrapper import with_unsupported_dtypes


@to_aikit_arrays_and_back
def gamma(shape, alpha, beta=None, dtype=aikit.float32, seed=None, name=None):
    return aikit.gamma(alpha, beta, shape=shape, dtype=dtype, seed=seed)


@with_unsupported_dtypes(
    {"2.15.0 and below": ("int8", "int16", "int32", "int64", "unsigned")}, "tensorflow"
)
@to_aikit_arrays_and_back
def normal(shape, mean=0.0, stddev=1.0, dtype=aikit.float32, seed=None, name=None):
    return aikit.random_normal(mean=mean, std=stddev, shape=shape, dtype=dtype, seed=seed)


@with_unsupported_dtypes(
    {"2.15.0 and below": ("int8", "int16", "unsigned")}, "tensorflow"
)
@to_aikit_arrays_and_back
@handle_tf_dtype
def poisson(shape, lam, dtype=aikit.float32, seed=None, name=None):
    shape = aikit.array(shape, dtype=aikit.int32)
    lam = aikit.array(lam, dtype=aikit.float32)
    if lam.ndim > 0:
        shape = aikit.concat([shape, aikit.array(lam.shape)])
    return aikit.poisson(shape=shape, lam=lam, dtype=dtype, seed=seed, fill_value=0)


# implement random shuffle
@with_unsupported_dtypes(
    {"2.15.0 and below": ("int8", "int16", "in32", "int64", "unsigned")}, "tensorflow"
)
@to_aikit_arrays_and_back
def shuffle(value, seed=None, name=None):
    return aikit.shuffle(value, seed=seed)


@with_unsupported_dtypes(
    {"2.15.0 and below": ("int8", "int16", "unsigned")}, "tensorflow"
)
@to_aikit_arrays_and_back
def stateless_normal(
    shape, seed, mean=0.0, stddev=1.0, dtype=aikit.float32, name=None, alg="auto_select"
):
    return aikit.random_normal(
        mean=mean, std=stddev, shape=shape, dtype=dtype, seed=seed[0] + seed[1]
    )


@with_unsupported_dtypes(
    {"2.15.0 and below": ("int8", "int16", "unsigned")}, "tensorflow"
)
@to_aikit_arrays_and_back
def stateless_poisson(shape, seed, lam, dtype=aikit.int32, name=None):
    return aikit.poisson(shape=shape, lam=lam, dtype=dtype, seed=seed[0] + seed[1])


@to_aikit_arrays_and_back
def stateless_uniform(
    shape, seed, minval=0, maxval=None, dtype=aikit.float32, name=None, alg="auto_select"
):
    return aikit.random_uniform(
        shape=shape, seed=seed[0] + seed[1], low=minval, high=maxval, dtype=dtype
    )


@with_unsupported_dtypes(
    {"2.15.0 and below": ("int8", "int16", "unsigned")}, "tensorflow"
)
@to_aikit_arrays_and_back
def uniform(shape, minval=0, maxval=None, dtype=aikit.float32, seed=None, name=None):
    if maxval is None:
        if dtype != "int64":
            maxval = 1.0
        else:
            raise ValueError("maxval must be specified for int64 dtype")
    return aikit.random_uniform(
        shape=shape, low=minval, high=maxval, dtype=dtype, seed=seed
    )
