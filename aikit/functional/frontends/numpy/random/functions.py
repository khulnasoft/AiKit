# local

import aikit
from aikit.functional.frontends.numpy.func_wrapper import (
    to_aikit_arrays_and_back,
    from_zero_dim_arrays_to_scalar,
)
from aikit import with_supported_dtypes


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def beta(a, b, size=None):
    return aikit.beta(a, b, shape=size)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def binomial(n, p, size=None):
    if p < 0 or p > 1:
        raise ValueError("p must be in the interval (0, 1)")
    if n < 0:
        raise ValueError("n must be strictly positive")
    if size is None:
        size = 1
    else:
        size = size
    if isinstance(size, int):
        size = (size,)
    lambda_ = aikit.multiply(n, p)
    return aikit.poisson(lambda_, shape=size)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def chisquare(df, size=None):
    df = aikit.array(df)  # scalar ints and floats are also array_like
    if aikit.any(df <= 0):
        raise ValueError("df <= 0")

    # aikit.gamma() throws an error if both alpha is an array and a shape is passed
    # so this part broadcasts df into the shape of `size`` first to keep it happy.
    if size is not None:
        df = df * aikit.ones(size)

    return aikit.gamma(df / 2, 2, dtype="float64")


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def choice(a, size=None, replace=True, p=None):
    sc_size = 1
    if isinstance(size, int):
        sc_size = size
    elif size is not None:
        #  If the given shape is, e.g., (m, n, k)
        #  then m * n * k samples are drawn. As per numpy docs
        sc_size = 1
        for s in size:
            if s is not None:
                sc_size *= s
    if isinstance(a, int):
        a = aikit.arange(a)
    index = aikit.multinomial(len(a), sc_size, replace=replace, probs=p)
    return a[index]


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def dirichlet(alpha, size=None):
    return aikit.dirichlet(alpha, size=size)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def exponential(scale=1.0, size=None, dtype="float64"):
    if scale > 0:
        # Generate samples that are uniformly distributed based on given parameters
        u = aikit.random_uniform(low=0.0, high=0.0, shape=size, dtype=dtype)
        return aikit.exp(scale, out=u)
    return 0  # if scale parameter is less than or equal to 0


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def f(dfn, dfd, size=None):
    # Generate samples from the uniform distribution
    x1 = aikit.gamma(aikit.to_scalar(aikit.divide(dfn, 2)), 2.0, shape=size, dtype="float64")
    x2 = aikit.gamma(aikit.to_scalar(aikit.divide(dfd, 2)), 2.0, shape=size, dtype="float64")
    # Calculate the F-distributed samples
    samples = aikit.divide(aikit.divide(x1, aikit.array(dfn)), aikit.divide(x2, aikit.array(dfd)))
    return samples


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def gamma(shape, scale=1.0, size=None):
    return aikit.gamma(shape, scale, shape=size, dtype="float64")


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def geometric(p, size=None):
    if p < 0 or p > 1:
        raise ValueError("p must be in the interval [0, 1]")
    oneMinusP = aikit.subtract(1, p)
    sizeMinusOne = aikit.subtract(size, 1)

    return aikit.multiply(aikit.pow(oneMinusP, sizeMinusOne), p)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def gumbel(loc=0.0, scale=1.0, size=None):
    u = aikit.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    x = loc - scale * aikit.log(-aikit.log(u))
    return x


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def laplace(loc=0.0, scale=1.0, size=None):
    u = aikit.random_uniform(low=0.0, high=0.0, shape=size, dtype="float64")
    u = loc - scale * aikit.sign(u - 0.5) * aikit.log(1 - 2 * aikit.abs(u - 0.5))
    return u


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def logistic(loc=0.0, scale=1.0, size=None):
    u = aikit.random_uniform(low=0.0, high=0.0, shape=size, dtype="float64")
    x = loc + scale * aikit.log(u / (1 - u))
    return x


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def lognormal(mean=0.0, sigma=1.0, size=None):
    ret = aikit.exp(aikit.random_normal(mean=mean, std=sigma, shape=size, dtype="float64"))
    return ret


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def logseries(p=0, size=None):
    if p < 0 or p >= 1:
        raise ValueError("p value must be in the open interval (0, 1)")
    r = aikit.log(1 - p)
    u = aikit.random_uniform(low=0.0, high=1.0, shape=size)
    v = aikit.random_uniform(low=0.0, high=1.0, shape=size)
    q = 1 - aikit.exp(r * u)
    ret = 1 + aikit.log(v) / aikit.log(q)
    return ret


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def multinomial(n, pvals, size=None):
    assert not aikit.exists(size) or (len(size) > 0 and len(size) < 3)
    batch_size = 1
    if aikit.exists(size):
        if len(size) == 2:
            batch_size = size[0]
            num_samples = size[1]
        else:
            num_samples = size[0]
    else:
        num_samples = len(pvals)
    return aikit.multinomial(n, num_samples, batch_size=batch_size, probs=pvals)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def negative_binomial(n, p, size=None):
    if p <= 0 or p >= 1:
        raise ValueError("p must be in the interval (0, 1)")
    if n <= 0:
        raise ValueError("n must be strictly positive")
    # numpy implementation uses scale = (1 - p) / p
    scale = (1 - p) / p
    # poisson requires shape to be a tuple
    if isinstance(size, int):
        size = (size,)
    lambda_ = aikit.gamma(n, scale, shape=size)
    return aikit.poisson(lam=lambda_, shape=size)


@with_supported_dtypes(
    {"1.25.2 and below": ("float16", "float32")},
    "numpy",
)
@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def noncentral_chisquare(df, nonc, size=None):
    if aikit.any(df <= 0):
        raise ValueError("Degree of freedom must be greater than 0")
    if aikit.has_nans(nonc):
        return aikit.nan
    if aikit.any(nonc == 0):
        return chisquare(df, size=size)
    if aikit.any(df < 1):
        n = standard_normal() + aikit.sqrt(nonc)
        return chisquare(df - 1, size=size) + n * n
    else:
        i = poisson(nonc / 2.0, size=size)
        return chisquare(df + 2 * i, size=size)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def normal(loc=0.0, scale=1.0, size=None):
    return aikit.random_normal(mean=loc, std=scale, shape=size, dtype="float64")


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def pareto(a, size=None):
    if a < 0:
        return 0
    u = aikit.random_uniform(low=0.0, high=0.0, shape=size, dtype="float64")
    return aikit.pow(1 / (1 - u), 1 / a)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def permutation(x, /):
    if isinstance(x, int):
        x = aikit.arange(x)
    return aikit.shuffle(x)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def poisson(lam=1.0, size=None):
    return aikit.poisson(lam=lam, shape=size)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def random_sample(size=None):
    return aikit.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def rayleigh(scale, size=None):
    u = aikit.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    log_u = aikit.log(u)
    x = aikit.multiply(scale, aikit.sqrt(aikit.multiply(-2, log_u)))
    return x


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def shuffle(x, axis=0, /):
    if isinstance(x, int):
        x = aikit.arange(x)
    return aikit.shuffle(x, axis)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def standard_cauchy(size=None):
    u = aikit.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    return aikit.tan(aikit.pi * (u - 0.5))


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def standard_exponential(size=None):
    if size is None:
        size = 1
    U = aikit.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    return -aikit.log(U)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def standard_gamma(shape, size=None):
    return aikit.gamma(shape, 1.0, shape=size, dtype="float64")


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def standard_normal(size=None):
    return aikit.random_normal(mean=0.0, std=1.0, shape=size, dtype="float64")


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def standard_t(df, size=None):
    numerator = aikit.random_normal(mean=0.0, std=1.0, shape=size, dtype="float64")
    denominator = aikit.gamma(df / 2, 1.0, shape=size, dtype="float64")
    return aikit.sqrt(df / 2) * aikit.divide(numerator, aikit.sqrt(denominator))


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def triangular(left, mode, right, size=None):
    if left > mode or mode > right or left == right:
        raise aikit.utils.exceptions.AikitValueError(
            "left < mode < right is not being followed"
        )
    u = aikit.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    condition = u <= (mode - left) / (right - left)
    values1 = left + (right - left) * (u * (mode - left) / (right - left)) ** 0.5
    values2 = (
        right - (right - mode) * ((1 - u) * (right - mode) / (right - left)) ** 0.5
    )
    return aikit.where(condition, values1, values2)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def uniform(low=0.0, high=1.0, size=None):
    return aikit.random_uniform(low=low, high=high, shape=size, dtype="float64")


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def vonmises(mu, kappa, size=None):
    t_size = 0
    # Output shape. If the given shape is, e.g., (m, n, k),
    # then m * n * k samples are drawn.
    if size is None or len(size) == 0:
        t_size = 1
    else:
        for x in size:
            t_size = t_size * x
    size = t_size
    li = []
    while len(li) < size:
        # Generate samples from the von Mises distribution using numpy
        u = aikit.random_uniform(low=-aikit.pi, high=aikit.pi, shape=size)
        v = aikit.random_uniform(low=0, high=1, shape=size)

        condition = v < (1 + aikit.exp(kappa * aikit.cos(u - mu))) / (
            2 * aikit.pi * aikit.i0(kappa)
        )
        selected_samples = u[condition]
        li.extend(aikit.to_list(selected_samples))

    return aikit.array(li[:size])


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def wald(mean, scale, size=None):
    if size is None:
        size = 1
    mu_2l = mean / (2 * scale)
    Y = aikit.random_normal(mean=0, std=1, shape=size, dtype="float64")
    U = aikit.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")

    Y = mean * aikit.square(Y)
    X = mean + mu_2l * (Y - aikit.sqrt(((4 * scale) * Y) + aikit.square(Y)))

    condition = mean / (mean + X) >= U
    value1 = X
    value2 = mean * mean / X

    return aikit.where(condition, value1, value2)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def weibull(a, size=None):
    if a < 0:
        return 0
    u = aikit.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    return aikit.pow(-aikit.log(1 - u), 1 / a)


@to_aikit_arrays_and_back
@from_zero_dim_arrays_to_scalar
def zipf(a, size=None):
    if a <= 1:
        return 0
    u = aikit.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")
    return aikit.floor(aikit.pow(1 / (1 - u), 1 / a))
