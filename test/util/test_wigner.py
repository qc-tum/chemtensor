import numpy as np
import h5py
from wigner import wigner_small_d, wigner_d, real_wigner_d


def wigner_small_d_data():

    # random number generator
    rng = np.random.default_rng(732)

    # random rotation angle
    theta = 2 * np.pi * rng.uniform()

    with h5py.File("data/test_wigner_small_d.hdf5", "w") as file:
        file.attrs["theta"] = theta
        for j in range(6):
            file[f"w{j}"] = wigner_small_d(j, theta)


def wigner_d_data():

    # random number generator
    rng = np.random.default_rng(203)

    # random rotation angles
    psi   = 2 * np.pi * rng.uniform()
    theta = 2 * np.pi * rng.uniform()
    phi   = 2 * np.pi * rng.uniform()

    with h5py.File("data/test_wigner_d.hdf5", "w") as file:
        file.attrs["psi"]   = psi
        file.attrs["theta"] = theta
        file.attrs["phi"]   = phi
        for j in range(6):
            file[f"w{j}"] = wigner_d(j, psi, theta, phi)


def real_wigner_d_data():

    # random number generator
    rng = np.random.default_rng(520)

    # random rotation angles
    psi   = 2 * np.pi * rng.uniform()
    theta = 2 * np.pi * rng.uniform()
    phi   = 2 * np.pi * rng.uniform()

    with h5py.File("data/test_real_wigner_d.hdf5", "w") as file:
        file.attrs["psi"]   = psi
        file.attrs["theta"] = theta
        file.attrs["phi"]   = phi
        for j in range(0, 6, 2):
            file[f"w{j}"] = real_wigner_d(j, psi, theta, phi)


def main():
    wigner_small_d_data()
    wigner_d_data()
    real_wigner_d_data()


if __name__ == "__main__":
    main()
