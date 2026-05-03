import numpy as np
from sympy import Ynm
import h5py


def _spherical_harmonics_to_real(yv):
    """
    Convert spherical harmonics to their real form,
    following the convention used in Wikipedia.
    """
    l = (len(yv) - 1) // 2
    return [(1j/np.sqrt(2) * (yv[l + m] - (-1)**m * yv[l - m])).real if m < 0
        else (yv[l].real if m == 0
        else (1/np.sqrt(2) * ((-1)**m * yv[l + m] + yv[l - m]))).real # m > 0
        for m in range(-l, l + 1)]


def real_spherical_harmonics_data():

    # random number generator
    rng = np.random.default_rng(603)

    # random spherical angles
    theta =     np.pi * rng.uniform()
    phi   = 2 * np.pi * rng.uniform()
    # convert to Cartesian unit vector
    v = np.array([
        np.cos(phi) * np.sin(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(theta)])

    with h5py.File("data/test_real_spherical_harmonics.hdf5", "w") as file:
        file.attrs["v"] = v
        for l in range(4):
            yv = [complex(Ynm(l, m, theta, phi).evalf()) for m in range(-l, l + 1)]
            file[f"yv{l}"] = _spherical_harmonics_to_real(yv)


def main():
    real_spherical_harmonics_data()


if __name__ == "__main__":
    main()
