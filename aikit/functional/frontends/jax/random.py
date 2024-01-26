# global
import operator

# local
import aikit
from aikit.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from aikit.functional.frontends.jax.func_wrapper import (
    to_aikit_arrays_and_back,
    handle_jax_dtype,
)


# --- Helpers --- #
# --------------- #


def _get_seed(key):
    if "PRNGKeyArray" in repr(key):
        key = key._base_array
    key1, key2 = int(key[0]), int(key[1])
    return aikit.to_scalar(int("".join(map(str, [key1, key2]))))


def _remove_axis(shape, axis):
    return shape[:axis] + shape[axis + 1 :]


# --- Main --- #
# ------------ #


@to_aikit_arrays_and_back
def PRNGKey(seed):
    return aikit.array([0, seed % 4294967295 - (seed // 4294967295)], dtype=aikit.int64)


@handle_jax_dtype
@to_aikit_arrays_and_back
@with_supported_dtypes(
    {
        "0.4.23 and below": (
            "float32",
            "float64",
        )
    },
    "jax",
)
def ball(key, d, p=2.0, shape=(), dtype="float64"):
    seed = _get_seed(key)
    d = operator.index(d)

    g = aikit.gamma(1 / p, 1.0, shape=shape, dtype=dtype, seed=seed)
    b = aikit.bernoulli(aikit.array([0.5]), shape=shape, dtype=dtype, seed=seed)
    r = 2 * b - 1
    gn = r * g ** (1 / p)

    uniform = aikit.random_uniform(seed=seed, shape=shape, dtype=dtype)
    exp = -aikit.log(1 - uniform)

    return gn / (((aikit.abs(gn) ** p).sum(axis=-1) + exp) ** (1 / p))[..., None]


@to_aikit_arrays_and_back
def bernoulli(key, p=0.5, shape=None):
    seed = _get_seed(key)
    return aikit.bernoulli(p, shape=shape, seed=seed)


@handle_jax_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.23 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def beta(key, a, b, shape=None, dtype=None):
    seed = _get_seed(key)
    return aikit.beta(a, b, shape=shape, dtype=dtype, seed=seed)


@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.23 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def categorical(key, logits, axis, shape=None):
    logits_arr = aikit.asarray(logits)

    if axis >= 0:
        axis -= len(logits_arr.shape)
    batch_shape = tuple(_remove_axis(logits_arr.shape, axis))

    if shape is None:
        shape = batch_shape
    else:
        shape = tuple(shape)
        if shape != batch_shape:
            raise ValueError(
                +f"Shape {shape} is not compatible with reference shape {batch_shape}"
            )

    logits_shape = list(shape[len(shape) - len(batch_shape) :])
    logits_shape.insert(axis % len(logits_arr.shape), logits_arr.shape[axis])

    gumbel_noise = gumbel(key, aikit.array(logits_shape), logits_arr.dtype)
    expanded_logits = aikit.expand_dims(logits_arr, axis=axis)
    noisy_logits = gumbel_noise + expanded_logits

    # Use Aikit's argmax to get indices
    indices = aikit.argmax(noisy_logits, axis=axis)

    return indices


@handle_jax_dtype
@to_aikit_arrays_and_back
def cauchy(key, shape=(), dtype="float64"):
    seed = _get_seed(key)
    u = aikit.random_uniform(low=0.0, high=1.0, shape=shape, dtype=dtype, seed=seed)
    return aikit.tan(aikit.pi * (u - 0.5))


@handle_jax_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.23 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def dirichlet(key, alpha, shape=None, dtype="float32"):
    seed = _get_seed(key)
    alpha = aikit.astype(alpha, dtype)
    return aikit.dirichlet(alpha, size=shape, dtype=dtype, seed=seed)


@handle_jax_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {"0.4.23 and below": "uint32"},
    "jax",
)
def double_sided_maxwell(key, loc, scale, shape=(), dtype="float64"):
    params_shapes = aikit.broadcast_shapes(aikit.shape(loc), aikit.shape(scale))
    if not shape:
        shape = params_shapes

    shape = shape + params_shapes
    maxwell_rvs = maxwell(key, shape=shape, dtype=dtype)
    random_sign = rademacher(key, shape=shape, dtype=dtype)

    return random_sign * maxwell_rvs * scale + loc


@handle_jax_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.23 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def exponential(key, shape=(), dtype="float64"):
    seed = _get_seed(key)
    uniform = aikit.random_uniform(seed=seed, shape=shape, dtype=dtype)
    exp = -aikit.log(1 - uniform)
    return exp


@to_aikit_arrays_and_back
def fold_in(key, data):
    if "PRNGKeyArray" in repr(key):
        key = key._base_array
    s = aikit.bitwise_left_shift(
        aikit.asarray(data, dtype=aikit.uint32), aikit.array(32, dtype=aikit.uint32)
    )
    return aikit.bitwise_xor(key, s)


@handle_jax_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.23 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def gamma(key, a, shape=None, dtype="float64"):
    seed = _get_seed(key)
    return aikit.gamma(a, 1.0, shape=shape, dtype=dtype, seed=seed)


@handle_jax_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.23 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def generalized_normal(key, p, shape=(), dtype="float64"):
    seed = _get_seed(key)
    g = aikit.gamma(1 / p, 1.0, shape=shape, dtype=dtype, seed=seed)
    b = aikit.bernoulli(aikit.array([0.5]), shape=shape, dtype=dtype, seed=seed)
    r = 2 * b - 1
    return r * g ** (1 / p)


@handle_jax_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.23 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def gumbel(key, shape=(), dtype="float64"):
    seed = _get_seed(key)
    uniform_x = aikit.random_uniform(
        low=0.0,
        high=1.0,
        shape=shape,
        dtype=dtype,
        seed=seed,
    )
    return -aikit.log(-aikit.log(uniform_x))


# loggamma
@handle_jax_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.23 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def loggamma(key, a, shape=None, dtype="float64"):
    seed = _get_seed(key)
    return aikit.log(aikit.gamma(a, 1.0, shape=shape, dtype=dtype, seed=seed))


@handle_jax_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {"0.4.23 and below": ("float16", "bfloat16")},
    "jax",
)
def logistic(key, shape=(), dtype="float64"):
    seed = _get_seed(key)
    uniform_x = aikit.random_uniform(seed=seed, shape=shape, dtype=dtype)
    return aikit.log(aikit.divide(uniform_x, aikit.subtract(1.0, uniform_x)))


@handle_jax_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.3.14 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def maxwell(key, shape, dtype="float64"):
    seed = _get_seed(key)
    shape = shape + (3,)
    random_normal = aikit.random_normal(seed=seed, shape=shape, dtype=dtype)
    return aikit.vector_norm(random_normal, axis=-1)


@handle_jax_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.23 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def multivariate_normal(key, mean, cov, shape=None, dtype="float64", method="cholesky"):
    if shape is None:
        shape = aikit.broadcast_shapes(mean.shape[:-1], cov.shape[:-2])
    if method == "cholesky":
        cov_factor = aikit.cholesky(cov)
    elif method == "eigh":
        (w, v) = aikit.eigh(cov)
        cov_factor = v * aikit.sqrt(w[..., None, :])
    elif method == "svd":
        (u, s, _) = aikit.svd(cov)
        cov_factor = u * aikit.sqrt(s[..., None, :])

    rand_normal = normal(key=key, shape=shape + mean.shape[-1:], dtype=dtype)
    result = mean + aikit.einsum("...ij,...j->...i", cov_factor, rand_normal.aikit_array)

    return result


@handle_jax_dtype
@to_aikit_arrays_and_back
def normal(key, shape=(), dtype=None):
    seed = _get_seed(key)
    return aikit.random_normal(shape=shape, dtype=dtype, seed=seed)


@handle_jax_dtype
@to_aikit_arrays_and_back
def orthogonal(key, n, shape=(), dtype=None):
    seed = _get_seed(key)
    flat_shape = (n, n)
    if shape:
        flat_shape = shape + flat_shape

    # Generate a random matrix with the given shape and dtype
    random_matrix = aikit.random_uniform(seed=seed, shape=flat_shape, dtype=dtype)

    # Compute the QR decomposition of the random matrix
    q, _ = aikit.linalg.qr(random_matrix)

    # Reshape the resulting orthogonal matrix to the desired shape
    if shape:
        q = aikit.reshape(q, shape + (n, n))

    return q


@handle_jax_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.23 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def pareto(key, b, shape=None, dtype="float64"):
    seed = _get_seed(key)
    if shape is None:
        shape = b.shape
    # Draw samples from exponential distribution
    uniform = aikit.random_uniform(seed=seed, shape=shape, dtype=dtype)
    e = -aikit.log(1 - uniform)

    return aikit.exp(e / b)


@to_aikit_arrays_and_back
def permutation(key, x, axis=0, independent=False):
    x = aikit.array(x)
    seed = _get_seed(key)
    if not aikit.get_num_dims(x):
        r = int(x)
        return aikit.shuffle(aikit.arange(r), axis, seed=seed)
    if independent:
        return aikit.shuffle(x, axis, seed=seed)
    rand = aikit.arange(x.shape[axis])
    ind = aikit.shuffle(rand, 0, seed=seed)
    return aikit.gather(x, ind, axis=axis)


@handle_jax_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {"0.4.23 and below": ("unsigned", "int8", "int16")},
    "jax",
)
def poisson(key, lam, shape=None, dtype=None):
    seed = _get_seed(key)
    return aikit.poisson(lam, shape=shape, dtype=dtype, seed=seed, fill_value=-1)


@handle_jax_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {"0.4.23 and below": ("unsigned", "int8", "int16")},
    "jax",
)
def rademacher(key, shape, dtype="int64"):
    seed = _get_seed(key)
    prob = aikit.full(shape, 0.5, dtype="float32")
    b = aikit.bernoulli(prob, shape=shape, dtype="float32", seed=seed)
    b = aikit.astype(b, dtype)
    return 2 * b - 1


@handle_jax_dtype
@to_aikit_arrays_and_back
@with_unsupported_dtypes(
    {"0.4.23 and below": ("unsigned", "int8", "int16")},
    "jax",
)
def randint(key, shape, minval, maxval, dtype="int64"):
    seed = _get_seed(key)
    return aikit.randint(minval, maxval, shape=shape, dtype=dtype, seed=seed)


@to_aikit_arrays_and_back
def shuffle(key, x, axis=0):
    seed = _get_seed(key)
    x = aikit.flip(x, axis=axis)
    return aikit.shuffle(x, seed=seed)


@handle_jax_dtype
@to_aikit_arrays_and_back
def t(key, df, shape=(), dtype="float64"):
    seed = _get_seed(key)
    n = aikit.random_normal(shape=shape, dtype=dtype, seed=seed)
    half_df = df / 2.0
    g = aikit.gamma(half_df, 1.0, shape=shape, dtype=dtype, seed=seed)
    return n * aikit.sqrt(aikit.divide(half_df, g))


@handle_jax_dtype
@to_aikit_arrays_and_back
def uniform(key, shape=(), dtype=None, minval=0.0, maxval=1.0):
    seed = _get_seed(key)
    return aikit.random_uniform(
        low=minval, high=maxval, shape=shape, dtype=dtype, seed=seed
    )


@handle_jax_dtype
@to_aikit_arrays_and_back
def weibull_min(key, scale, concentration, shape=(), dtype="float64"):
    seed = _get_seed(key)
    uniform_x = aikit.random_uniform(seed=seed, shape=shape, dtype=dtype)
    x = 1 - uniform_x
    weibull = x ** (concentration - 1) * -aikit.log(x / scale)
    return weibull
