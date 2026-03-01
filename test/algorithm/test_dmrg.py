import copy
import numpy as np
import h5py
import sys
sys.path.append("../operator/")
sys.path.append("../state/")
sys.path.append("../tensor/")
sys.path.append("../util/")
from mpo import MPO
from mps import MPS
from block_sparse_util import is_qsparse
from dmrg import dmrg_singlesite, dmrg_twosite


def dmrg_singlesite_data():

    # random number generator
    rng = np.random.default_rng(105)

    # number of lattice sites
    nsites = 7

    # local physical quantum numbers
    qsite = np.array([0, -1, 0])

    # Hamiltonian
    hamiltonian = _construct_random_hermitian_mpo(qsite, nsites, rng)
    # must be Hermitian
    hmat = hamiltonian.to_matrix()
    assert np.allclose(hmat, hmat.conj().T)

    psi_start = _construct_random_mps(qsite, nsites, rng)
    assert np.linalg.norm(psi_start.to_vector()) > 0.1

    num_sweeps = 6

    psi = copy.deepcopy(psi_start)
    en_sweeps = dmrg_singlesite(hamiltonian, psi, num_sweeps)

    # reference (numerically exact) ground state energy
    en_min = np.linalg.eigvalsh(hmat)[0]
    assert en_sweeps[-1] >= en_min

    with h5py.File("data/test_dmrg_singlesite.hdf5", "w") as file:
        file.attrs["qsite"] = qsite
        for i, qbond in enumerate(hamiltonian.qbonds):
            file.attrs[f"h_qbond{i}"] = qbond
        for i, ai in enumerate(hamiltonian.a):
            file[f"h_a{i}"] = ai
        for i, qbond in enumerate(psi_start.qbonds):
            file.attrs[f"psi_start_qbond{i}"] = qbond
        for i, ai in enumerate(psi_start.a):
            file[f"psi_start_a{i}"] = ai
        file["en_sweeps"] = en_sweeps
        for i, qbond in enumerate(psi.qbonds):
            file.attrs[f"psi_qbond{i}"] = qbond
        for i, ai in enumerate(psi.a):
            file[f"psi_a{i}"] = ai


def dmrg_twosite_data():

    # random number generator
    rng = np.random.default_rng(288)

    # number of lattice sites
    nsites = 11

    # local physical quantum numbers
    qsite = np.array([0, 0])

    # Hamiltonian
    hamiltonian = _construct_random_hermitian_mpo(qsite, nsites, rng)
    # must be Hermitian
    hmat = hamiltonian.to_matrix()
    assert np.allclose(hmat, hmat.conj().T)

    psi_start = _construct_random_mps(qsite, nsites, rng)
    assert np.linalg.norm(psi_start.to_vector()) > 0.1

    num_sweeps = 4
    tol_split = 1e-5

    psi = copy.deepcopy(psi_start)
    en_sweeps = dmrg_twosite(hamiltonian, psi, num_sweeps, numiter_lanczos=25, tol_split=tol_split)

    # reference (numerically exact) ground state energy
    en_min = np.linalg.eigvalsh(hmat)[0]
    assert en_sweeps[-1] >= en_min

    with h5py.File("data/test_dmrg_twosite.hdf5", "w") as file:
        file.attrs["qsite"] = qsite
        file.attrs["tol_split"] = tol_split
        for i, qbond in enumerate(hamiltonian.qbonds):
            file.attrs[f"h_qbond{i}"] = qbond
        for i, ai in enumerate(hamiltonian.a):
            file[f"h_a{i}"] = ai
        for i, qbond in enumerate(psi_start.qbonds):
            file.attrs[f"psi_start_qbond{i}"] = qbond
        for i, ai in enumerate(psi_start.a):
            file[f"psi_start_a{i}"] = ai
        file["en_sweeps"] = en_sweeps
        for i, qbond in enumerate(psi.qbonds):
            file.attrs[f"psi_qbond{i}"] = qbond
        for i, ai in enumerate(psi.a):
            file[f"psi_a{i}"] = ai


def _construct_random_hermitian_mpo(qsite, nsites: int, rng: np.random.Generator):
    """
    Construct a random matrix product operator, such that the logical operator is Hermitian.
    """
    # Hermitian part with zero virtual bond quantum numbers
    qbonds_zero = [np.zeros(bi, dtype=int)
                   for bi in [1] + list(rng.integers(5, 10, size=nsites-1)) + [1]]
    h0 = MPO(qsite, qbonds_zero, fill="random", rng=rng)
    for i in range(nsites):
        h0.a[i] = 1.5 * (h0.a[i] + h0.a[i].conj().transpose((0, 2, 1, 3)))
    assert np.linalg.norm(h0.to_matrix(), ord=2) > 0.1

    # general part (plus adjoint copy)
    qbonds1 = [rng.integers(-1, 2, size=bi)
               for bi in [1] + list(rng.integers(10, 20, size=nsites-1)) + [1]]
    # require zero dummy bonds for addition of adjoint copy
    qbonds1[ 0] = np.array([0])
    qbonds1[-1] = np.array([0])
    h1 = MPO(qsite, qbonds1, fill="random", rng=rng)
    for i in range(nsites):
        h1.a[i] *= 3.5
    # Hermitian conjugated copy
    h1dag = MPO(qsite, [-qbi for qbi in qbonds1], fill="postpone")
    h1dag.a = [a.conj().transpose((0, 2, 1, 3)) for a in h1.a]
    # consistency check
    for i in range(nsites):
        assert is_qsparse(h1dag.a[i],
                          (h1dag.qbonds[i], h1dag.qsite, -h1dag.qsite, -h1dag.qbonds[i+1]))

    mpo = h0 + h1 + h1dag

    # randomly permute virtual bonds
    for i in range(nsites - 1):
        idx = rng.permutation(mpo.bond_dims[i + 1])
        mpo.a[i]    = mpo.a[i]  [ :,  :, :, idx]
        mpo.a[i+1]  = mpo.a[i+1][idx, :, :,  : ]
        mpo.qbonds[i+1] = mpo.qbonds[i+1][idx]
        assert is_qsparse(mpo.a[i],   (mpo.qbonds[i],   mpo.qsite, -mpo.qsite, -mpo.qbonds[i+1]))
        assert is_qsparse(mpo.a[i+1], (mpo.qbonds[i+1], mpo.qsite, -mpo.qsite, -mpo.qbonds[i+2]))

    return mpo


def _construct_random_mps(qsite, nsites: int, rng: np.random.Generator):
    """
    Construct a random matrix product state.
    """
    # virtual bond quantum numbers
    qbonds = [rng.integers(-1, 2, size=bi)
              for bi in [1] + list(rng.integers(12, 20, size=nsites-1)) + [1]]
    qbonds[ 0] = np.array([0])
    qbonds[-1] = np.array([0])
    psi = MPS(qsite, qbonds, fill="random", rng=rng)
    for i in range(nsites):
        psi.a[i] *= 5
    return psi


def main():
    dmrg_singlesite_data()
    dmrg_twosite_data()


if __name__ == "__main__":
    main()
