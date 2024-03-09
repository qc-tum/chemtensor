import numpy as np
import h5py
import pytenet as ptn
from util import interleave_complex


def mps_vdot_data():

    # random number generator
    rng = np.random.default_rng(395)

    # local physical dimension
    d = 3

    # physical quantum numbers
    qd = rng.integers(-1, 2, size=d)

    # virtual bond quantum numbers
    qD_psi = [rng.integers(-1, 2, size=Di) for Di in (1, 13, 17,  8, 1)]
    qD_chi = [rng.integers(-1, 2, size=Di) for Di in (1, 15, 20, 11, 1)]
    # ensure that leading and trailing virtual bond quantum numbers are compatible
    qD_chi[0] = qD_chi[-1] + qD_psi[0] - qD_psi[-1]

    # create random matrix product states
    psi = ptn.MPS(qd, qD_psi, fill="random", rng=rng)
    chi = ptn.MPS(qd, qD_chi, fill="random", rng=rng)
    # rescale tensors such that overall norm is of the order 1
    for i in range(psi.nsites):
        psi.A[i] *= 5
    for i in range(chi.nsites):
        chi.A[i] *= 5

    # calculate dot product <chi | psi>
    s = ptn.vdot(chi, psi)

    with h5py.File("data/test_mps_vdot.hdf5", "w") as file:
        file.attrs["qsite"] = qd
        for i, qbond in enumerate(qD_psi):
            file.attrs[f"qbond_psi_{i}"] = qbond
        for i, qbond in enumerate(qD_chi):
            file.attrs[f"qbond_chi_{i}"] = qbond
        for i, ai in enumerate(psi.A):
            # transposition due to different convention for axis ordering
            file[f"psi_a{i}"] = interleave_complex(ai.transpose((1, 0, 2)))
        for i, ai in enumerate(chi.A):
            # transposition due to different convention for axis ordering
            file[f"chi_a{i}"] = interleave_complex(ai.transpose((1, 0, 2)))
        file["s"] = interleave_complex(s)


def main():
    mps_vdot_data()


if __name__ == "__main__":
    main()
