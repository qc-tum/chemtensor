import numpy as np
import h5py
import sys
sys.path.append("../operator/")
sys.path.append("../state/")
sys.path.append("../tensor/")
sys.path.append("../util/")
from chain_ops import mpo_inner_product
from mpo import MPO
from mps import MPS


def mpo_inner_product_data():

    rng = np.random.default_rng(243)

    # physical dimension
    d = 2

    # physical quantum numbers
    qsite = rng.integers(-1, 2, size=d)

    # virtual bond quantum numbers
    qbonds_psi = [rng.integers(-1, 2, size=bi) for bi in (1, 10, 19, 26,  7, 1)]
    qbonds_chi = [rng.integers(-1, 2, size=bi) for bi in (1, 17, 23, 12, 13, 1)]
    qbonds_op  = [rng.integers(-1, 2, size=bi) for bi in (1,  6, 15, 29, 14, 1)]
    # ensure that leading and trailing virtual bond quantum numbers are compatible
    qbonds_chi[0] = qbonds_chi[-1] + qbonds_psi[0] - qbonds_psi[-1] + qbonds_op[0] - qbonds_op[-1]

    # create random matrix product states and a random matrix product operator
    psi = MPS(qsite, qbonds_psi, fill="random", rng=rng)
    chi = MPS(qsite, qbonds_chi, fill="random", rng=rng)
    op  = MPO(qsite, qbonds_op,  fill="random", rng=rng)
    # convert tensor entries to single precision
    psi.a = [a.astype(np.complex64) for a in psi.a]
    chi.a = [a.astype(np.complex64) for a in chi.a]
    op.a  = [a.astype(np.complex64) for a in  op.a]
    # rescale tensors such that overall norm is of the order 1
    for i in range(psi.nsites):
        psi.a[i] *= 5
    for i in range(chi.nsites):
        chi.a[i] *= 5
    for i in range(op.nsites):
        op.a[i] *= 5

    # calculate inner product <chi | op | psi>
    s = mpo_inner_product(chi, op, psi)

    with h5py.File("data/test_mpo_inner_product.hdf5", "w") as file:
        file.attrs["qsite"] = qsite
        for i, qbond in enumerate(qbonds_psi):
            file.attrs[f"qbond_psi_{i}"] = qbond
        for i, qbond in enumerate(qbonds_chi):
            file.attrs[f"qbond_chi_{i}"] = qbond
        for i, qbond in enumerate(qbonds_op):
            file.attrs[f"qbond_op_{i}"] = qbond
        for i, ai in enumerate(psi.a):
            file[f"psi_a{i}"] = ai
        for i, ai in enumerate(chi.a):
            file[f"chi_a{i}"] = ai
        for i, ai in enumerate(op.a):
            file[f"op_a{i}"] = ai
        file["s"] = s


def apply_mpo_data():

    rng = np.random.default_rng(975)

    # physical dimension
    d = 3

    # physical quantum numbers
    qsite = rng.integers(-1, 2, size=d)

    # virtual bond quantum numbers
    qbonds_psi = [rng.integers(-1, 2, size=bi) for bi in (1, 7, 21, 25, 11,  4, 1)]
    qbonds_op  = [rng.integers(-1, 2, size=bi) for bi in (1, 6, 15, 33, 29, 14, 1)]

    # create a random matrix product state operator
    psi = MPS(qsite, qbonds_psi, fill="random", rng=rng)
    op  = MPO(qsite, qbonds_op,  fill="random", rng=rng)
    # rescale tensors such that overall norm is of the order 1
    for i in range(psi.nsites):
        psi.a[i] *= 5
    for i in range(op.nsites):
        op.a[i] *= 5

    with h5py.File("data/test_apply_mpo.hdf5", "w") as file:
        file.attrs["qsite"] = qsite
        for i, qbond in enumerate(qbonds_psi):
            file.attrs[f"qbond_psi_{i}"] = qbond
        for i, qbond in enumerate(qbonds_op):
            file.attrs[f"qbond_op_{i}"] = qbond
        for i, ai in enumerate(psi.a):
            file[f"psi_a{i}"] = ai
        for i, ai in enumerate(op.a):
            file[f"op_a{i}"] = ai


def main():
    mpo_inner_product_data()
    apply_mpo_data()


if __name__ == "__main__":
    main()
