import numpy as np
import h5py
import pytenet as ptn
from util import crandn, interleave_complex


def lanczos_iteration_d_data():

    # random number generator
    rng = np.random.default_rng(355)

    n = 319
    numiter = 24

    # random symmetric matrix
    a = rng.standard_normal((n, n)) / np.sqrt(n)
    a = 0.5 * (a + a.T)

    # random starting vector
    vstart = rng.standard_normal(n) / np.sqrt(n)

    # perform Lanczos iteration, using 'a' as linear transformation
    alpha, beta, v = ptn.lanczos_iteration(lambda x: a @ x, vstart, numiter)
    # function returns complex matrix by default
    v = v.real

    # check orthogonality of Lanczos vectors
    assert np.allclose(v.T @ v, np.identity(numiter), rtol=1e-12)

    # Lanczos vectors must tridiagonalize 'a'
    t = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    assert np.allclose(v.T @ a @ v, t, rtol=1e-12)

    with h5py.File("data/test_lanczos_iteration_d.hdf5", "w") as file:
        file["a"]      = a
        file["vstart"] = vstart
        file["alpha"]  = alpha
        file["beta"]   = beta
        file["v"]      = v.T  # transposition due to different convention


def lanczos_iteration_z_data():

    # random number generator
    rng = np.random.default_rng(481)

    n = 173
    numiter = 24

    # random Hermitian matrix
    a = crandn((n, n), rng) / np.sqrt(n)
    a = 0.5 * (a + a.conj().T)

    # random complex starting vector
    vstart = crandn(n, rng) / np.sqrt(n)

    # perform Lanczos iteration, using 'a' as linear transformation
    alpha, beta, v = ptn.lanczos_iteration(lambda x: a @ x, vstart, numiter)

    # check orthogonality of Lanczos vectors
    assert np.allclose(v.conj().T @ v, np.identity(numiter), rtol=1e-12)

    # Lanczos vectors must tridiagonalize 'a'
    t = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    assert np.allclose(v.conj().T @ a @ v, t, rtol=1e-12)

    with h5py.File("data/test_lanczos_iteration_z.hdf5", "w") as file:
        file["a"]      = interleave_complex(a)
        file["vstart"] = interleave_complex(vstart)
        file["alpha"]  = alpha
        file["beta"]   = beta
        file["v"]      = interleave_complex(v.T)  # transposition due to different convention


def main():
    lanczos_iteration_d_data()
    lanczos_iteration_z_data()


if __name__ == "__main__":
    main()
