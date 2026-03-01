import numpy as np
import h5py
import sys
sys.path.append("../algorithm/")
sys.path.append("../tensor/")
sys.path.append("../util/")
from mps import MPS, mps_merge_tensor_pair, mps_vdot, mps_split_tensor_svd
from block_sparse_util import qnumber_flatten, enforce_qsparsity, is_qsparse


def mps_vdot_data():

    # random number generator
    rng = np.random.default_rng(395)

    # local physical dimension
    d = 3

    # physical quantum numbers
    qsite = rng.integers(-1, 2, size=d)

    # virtual bond quantum numbers
    qbonds_psi = [rng.integers(-1, 2, size=bi) for bi in (1, 13, 17,  8, 1)]
    qbonds_chi = [rng.integers(-1, 2, size=bi) for bi in (1, 15, 20, 11, 1)]
    # ensure that leading and trailing virtual bond quantum numbers are compatible
    qbonds_chi[0] = qbonds_chi[-1] + qbonds_psi[0] - qbonds_psi[-1]

    # create random matrix product states
    psi = MPS(qsite, qbonds_psi, fill="random", rng=rng)
    chi = MPS(qsite, qbonds_chi, fill="random", rng=rng)
    # rescale tensors such that overall norm is of the order 1
    for i in range(psi.nsites):
        psi.a[i] *= 5
    for i in range(chi.nsites):
        chi.a[i] *= 5

    # calculate dot product <chi | psi>
    s = mps_vdot(chi, psi)

    with h5py.File("data/test_mps_vdot.hdf5", "w") as file:
        file.attrs["qsite"] = qsite
        for i, qbond in enumerate(qbonds_psi):
            file.attrs[f"qbond_psi_{i}"] = qbond
        for i, qbond in enumerate(qbonds_chi):
            file.attrs[f"qbond_chi_{i}"] = qbond
        for i, ai in enumerate(psi.a):
            file[f"psi_a{i}"] = ai
        for i, ai in enumerate(chi.a):
            file[f"chi_a{i}"] = ai
        file["s"] = s


def mps_orthonormalize_qr_data():

    # random number generator
    rng = np.random.default_rng(376)

    # local physical dimension
    d = 3

    # physical quantum numbers
    qsite = rng.integers(-1, 2, size=d)

    # virtual bond quantum numbers
    qbonds = [rng.integers(-1, 2, size=bi) for bi in (1, 4, 11, 9, 7, 3, 1)]

    # create a random matrix product state
    mps = MPS(qsite, qbonds, fill="random", rng=rng)
    # convert tensor entries to single precision
    mps.a = [ai.astype(np.complex64) for ai in mps.a]

    with h5py.File("data/test_mps_orthonormalize_qr.hdf5", "w") as file:
        file.attrs["qsite"] = qsite
        for i, qbond in enumerate(qbonds):
            file.attrs[f"qbond{i}"] = qbond
        for i, ai in enumerate(mps.a):
            file[f"a{i}"] = ai


def mps_compress_data():

    # random number generator
    rng = np.random.default_rng(935)

    # physical quantum numbers
    qsite = [-1, 1, 0]

    # virtual bond quantum numbers
    qbonds = [rng.integers(-1, 2, size=bi) for bi in [1, 23, 75, 102, 83, 30, 1]]

    # create random matrix product state with small entanglement
    mps = MPS(qsite, qbonds, fill="random", rng=rng)
    for i in range(mps.nsites):
        # imitate small entanglement by multiplying bonds with small scaling factors
        s = np.exp(-30*(rng.uniform(size=mps.bond_dims[i + 1])))
        s /= np.linalg.norm(s)
        mps.a[i] = mps.a[i] * s
        # rescale to achieve norm of order 1
        mps.a[i] *= 5 / np.linalg.norm(mps.a[i])
        # convert tensor entries to single precision
        mps.a[i] = mps.a[i].astype(np.complex64)

    with h5py.File("data/test_mps_compress.hdf5", "w") as file:
        file.attrs["qsite"] = qsite
        for i, qbond in enumerate(qbonds):
            file.attrs[f"qbond{i}"] = qbond
        for i, ai in enumerate(mps.a):
            file[f"a{i}"] = ai


def mps_split_tensor_svd_data():

    # random number generator
    rng = np.random.default_rng(294)

    # physical dimensions
    d = [4, 5]
    # outer virtual bond dimensions
    b = [13, 17]

    a_pair = rng.standard_normal((b[0], d[0]*d[1], b[1])) / np.sqrt(b[0]*d[0]*d[1]*b[1])

    # fictitious quantum numbers
    qsite = [rng.integers(-2, 3, size=di) for di in d]
    qbonds_outer = [rng.integers(-2, 3, size=bi) for bi in b]

    # enforce block sparsity structure dictated by quantum numbers
    enforce_qsparsity(a_pair, [qbonds_outer[0], qnumber_flatten(qsite), -qbonds_outer[1]])

    with h5py.File("data/test_mps_split_tensor_svd.hdf5", "w") as file:

        file["a_pair"] = a_pair

        for i in range(2):
            file.attrs[f"qsite{i}"] = qsite[i]
        for i in range(2):
            file.attrs[f"qbonds{i}"] = qbonds_outer[i]

        tol = 0.04
        file.attrs["tol"] = tol

        a0, a1, qbond = mps_split_tensor_svd(
            a_pair, qsite[0], qsite[1], qbonds_outer, svd_distr="left", tol=tol)

        assert is_qsparse(a0, [qbonds_outer[0], qsite[0], -qbond])
        assert is_qsparse(a1, [qbond,           qsite[1], -qbonds_outer[1]])

        # merge tensors again, as reference
        a_mrg = mps_merge_tensor_pair(a0, a1)

        file["a_mrg"] = a_mrg


def mps_to_statevector_data():

    # random number generator
    rng = np.random.default_rng(531)

    # local physical dimension
    d = 3

    # physical quantum numbers
    qsite = rng.integers(-1, 2, size=d)

    # virtual bond quantum numbers
    qbonds = [rng.integers(-1, 2, size=Di) for Di in (1, 7, 10, 11, 5, 1)]

    # create a random matrix product state
    mps = MPS(qsite, qbonds, fill="random", rng=rng)

    # convert to a state vector
    vec = mps.to_vector()

    with h5py.File("data/test_mps_to_statevector.hdf5", "w") as file:
        file.attrs["qsite"] = qsite
        for i, qbond in enumerate(qbonds):
            file.attrs[f"qbond{i}"] = qbond
        for i, ai in enumerate(mps.a):
            file[f"a{i}"] = ai
        file["vec"] = vec


def main():
    mps_vdot_data()
    mps_orthonormalize_qr_data()
    mps_compress_data()
    mps_split_tensor_svd_data()
    mps_to_statevector_data()


if __name__ == "__main__":
    main()
