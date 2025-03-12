import numpy as np
import h5py
import pytenet as ptn


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
    a = ptn.crandn((n, n), rng) / np.sqrt(n)
    a = 0.5 * (a + a.conj().T)

    # random complex starting vector
    vstart = ptn.crandn(n, rng) / np.sqrt(n)

    # perform Lanczos iteration, using 'a' as linear transformation
    alpha, beta, v = ptn.lanczos_iteration(lambda x: a @ x, vstart, numiter)

    # check orthogonality of Lanczos vectors
    assert np.allclose(v.conj().T @ v, np.identity(numiter), rtol=1e-12)

    # Lanczos vectors must tridiagonalize 'a'
    t = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    assert np.allclose(v.conj().T @ a @ v, t, rtol=1e-12)

    with h5py.File("data/test_lanczos_iteration_z.hdf5", "w") as file:
        file["a"]      = a
        file["vstart"] = vstart
        file["alpha"]  = alpha
        file["beta"]   = beta
        file["v"]      = v.T  # transposition due to different convention


def eigensystem_krylov_symmetric_data():

    # random number generator
    rng = np.random.default_rng(354)

    n = 197
    numiter = 35
    numeig = 5

    # random symmetric matrix
    a = rng.standard_normal((n, n)) / np.sqrt(n)
    a = 0.5 * (a + a.T)

    # random starting vector
    vstart = rng.standard_normal(n) / np.sqrt(n)

    w, u_ritz = ptn.eigh_krylov(lambda x: a @ x, vstart, numiter, numeig)
    assert np.linalg.norm(u_ritz.imag) == 0
    # function returns complex Ritz vectors by default
    u_ritz = u_ritz.real

    # check orthogonality of Ritz vectors
    assert np.allclose(u_ritz.T @ u_ritz, np.identity(numeig), rtol=1e-12)

    with h5py.File("data/test_eigensystem_krylov_symmetric.hdf5", "w") as file:
        file["a"]      = a
        file["vstart"] = vstart
        file["lambda"] = w
        file["u_ritz"] = u_ritz


def eigensystem_krylov_hermitian_data():

    # random number generator
    rng = np.random.default_rng(207)

    n = 185
    numiter = 37
    numeig = 6

    # random Hermitian matrix
    a = ptn.crandn((n, n), rng) / np.sqrt(n)
    a = 0.5 * (a + a.conj().T)

    # random complex starting vector
    vstart = ptn.crandn(n, rng) / np.sqrt(n)

    w, u_ritz = ptn.eigh_krylov(lambda x: a @ x, vstart, numiter, numeig)

    # check orthogonality of Ritz vectors
    assert np.allclose(u_ritz.conj().T @ u_ritz, np.identity(numeig), rtol=1e-12)

    with h5py.File("data/test_eigensystem_krylov_hermitian.hdf5", "w") as file:
        file["a"]      = a
        file["vstart"] = vstart
        file["lambda"] = w
        file["u_ritz"] = u_ritz


def main():
    lanczos_iteration_d_data()
    lanczos_iteration_z_data()
    eigensystem_krylov_symmetric_data()
    eigensystem_krylov_hermitian_data()


if __name__ == "__main__":
    main()
