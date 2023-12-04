import numpy as np


def interleave_complex(a: np.ndarray):
    """
    Interleave real and imaginary parts of a complex-valued array (for saving it to disk).
    """
    return np.stack((a.real, a.imag), axis=-1)


def crandn(size=None, rng: np.random.Generator=None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)
