import numpy as np
import h5py
import pytenet as ptn
import sys
sys.path.append("../")
from util import interleave_complex


def mpo_inner_product_data():

    rng = np.random.default_rng(243)

    # physical dimension
    d = 2

    # physical quantum numbers
    qd = rng.integers(-1, 2, size=d)

    # virtual bond quantum numbers
    qD_psi = [rng.integers(-1, 2, size=Di) for Di in (1, 10, 19, 26,  7, 1)]
    qD_chi = [rng.integers(-1, 2, size=Di) for Di in (1, 17, 23, 12, 13, 1)]
    qD_op  = [rng.integers(-1, 2, size=Di) for Di in (1,  6, 15, 29, 14, 1)]
    # ensure that leading and trailing virtual bond quantum numbers are compatible
    qD_chi[0] = qD_chi[-1] + qD_psi[0] - qD_psi[-1] + qD_op[0] - qD_op[-1]

    # create random matrix product states and a random matrix product operator
    psi = ptn.MPS(qd, qD_psi, fill="random", rng=rng)
    chi = ptn.MPS(qd, qD_chi, fill="random", rng=rng)
    op  = ptn.MPO(qd, qD_op,  fill="random", rng=rng)
    # convert tensor entries to single precision
    psi.A = [a.astype(np.complex64) for a in psi.A]
    chi.A = [a.astype(np.complex64) for a in chi.A]
    op.A  = [a.astype(np.complex64) for a in  op.A]
    # rescale tensors such that overall norm is of the order 1
    for i in range(psi.nsites):
        psi.A[i] *= 5
    for i in range(chi.nsites):
        chi.A[i] *= 5
    for i in range(op.nsites):
        op.A[i] *= 5

    # calculate inner product <chi | op | psi>
    s = ptn.operator_inner_product(chi, op, psi)

    with h5py.File("data/test_mpo_inner_product.hdf5", "w") as file:
        file.attrs["qsite"] = qd
        for i, qbond in enumerate(qD_psi):
            file.attrs[f"qbond_psi_{i}"] = qbond
        for i, qbond in enumerate(qD_chi):
            file.attrs[f"qbond_chi_{i}"] = qbond
        for i, qbond in enumerate(qD_op):
            file.attrs[f"qbond_op_{i}"] = qbond
        for i, a in enumerate(psi.A):
            # transposition due to different convention for axis ordering
            file[f"psi_a{i}"] = interleave_complex(a.transpose((1, 0, 2)))
        for i, a in enumerate(chi.A):
            # transposition due to different convention for axis ordering
            file[f"chi_a{i}"] = interleave_complex(a.transpose((1, 0, 2)))
        for i, a in enumerate(op.A):
            # transposition due to different convention for axis ordering
            file[f"op_a{i}"] = interleave_complex(a.transpose((2, 0, 1, 3)))
        file["s"] = interleave_complex(s)


def apply_mpo_data():

    rng = np.random.default_rng(975)

    # physical dimension
    d = 3

    # physical quantum numbers
    qd = rng.integers(-1, 2, size=d)

    # virtual bond quantum numbers
    qD_psi = [rng.integers(-1, 2, size=Di) for Di in (1, 7, 21, 25, 11,  4, 1)]
    qD_op  = [rng.integers(-1, 2, size=Di) for Di in (1, 6, 15, 33, 29, 14, 1)]

    # create a random matrix product state operator
    psi = ptn.MPS(qd, qD_psi, fill="random", rng=rng)
    op  = ptn.MPO(qd, qD_op,  fill="random", rng=rng)
    # rescale tensors such that overall norm is of the order 1
    for i in range(psi.nsites):
        psi.A[i] *= 5
    for i in range(op.nsites):
        op.A[i] *= 5

    with h5py.File("data/test_apply_mpo.hdf5", "w") as file:
        file.attrs["qsite"] = qd
        for i, qbond in enumerate(qD_psi):
            file.attrs[f"qbond_psi_{i}"] = qbond
        for i, qbond in enumerate(qD_op):
            file.attrs[f"qbond_op_{i}"] = qbond
        for i, a in enumerate(psi.A):
            # transposition due to different convention for axis ordering
            file[f"psi_a{i}"] = interleave_complex(a.transpose((1, 0, 2)))
        for i, a in enumerate(op.A):
            # transposition due to different convention for axis ordering
            file[f"op_a{i}"] = interleave_complex(a.transpose((2, 0, 1, 3)))


def main():
    mpo_inner_product_data()
    apply_mpo_data()


if __name__ == "__main__":
    main()
