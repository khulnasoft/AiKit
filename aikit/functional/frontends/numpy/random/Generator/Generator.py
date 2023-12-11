# local
import aikit.functional.frontends.numpy as aikit_np


class Generator:
    def __init__(self, bit_generator=None):
        self.seed = bit_generator

    def multinomial(self, n, pvals, size=None):
        aikit_np.random.multinomial(n, pvals, size=size)


def default__rng(seed=None):
    return Generator(bit_generator=seed)
