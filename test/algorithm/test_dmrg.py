import copy
import numpy as np
import h5py
import pytenet as ptn


def _construct_random_hermitian_mpo(qd, nsites: int, rng: np.random.Generator):
    """
    Construct a random matrix product operator, such that the logical operator is Hermitian.
    """

    # Hermitian part with zero virtual bond quantum numbers
    qDzero = [np.zeros(Di, dtype=int) for Di in [1] + list(rng.integers(5, 10, size=nsites-1)) + [1]]
    h0 = ptn.MPO(qd, qDzero, fill="random", rng=rng)
    for i in range(nsites):
        h0.A[i] = 1.5 * (h0.A[i] + h0.A[i].conj().transpose((1, 0, 2, 3)))
    assert np.linalg.norm(h0.as_matrix(), ord=2) > 0.1

    # general part (plus adjoint copy)
    qD1 = [rng.integers(-1, 2, size=Di) for Di in [1] + list(rng.integers(10, 20, size=nsites-1)) + [1]]
    # require zero dummy bonds for addition of adjoint copy
    qD1[ 0] = np.array([0])
    qD1[-1] = np.array([0])
    h1 = ptn.MPO(qd, qD1, fill="random", rng=rng)
    for i in range(nsites):
        h1.A[i] *= 3.5
    # Hermitian conjugated copy
    h1dag = ptn.MPO(qd, [-qDi for qDi in qD1], fill="postpone")
    h1dag.A = [a.conj().transpose((1, 0, 2, 3)) for a in h1.A]
    # consistency check
    for i in range(nsites):
        assert ptn.is_qsparse(h1dag.A[i], [h1dag.qd, -h1dag.qd, h1dag.qD[i], -h1dag.qD[i+1]])

    mpo = h0 + h1 + h1dag

    # randomly permute virtual bonds
    for i in range(nsites - 1):
        idx = rng.permutation(mpo.bond_dims[i + 1])
        mpo.A[i]    = mpo.A[i]  [:, :, :, idx]
        mpo.A[i+1]  = mpo.A[i+1][:, :, idx, :]
        mpo.qD[i+1] = mpo.qD[i+1][idx]
        assert ptn.is_qsparse(mpo.A[i],   [mpo.qd, -mpo.qd, mpo.qD[i],   -mpo.qD[i+1]])
        assert ptn.is_qsparse(mpo.A[i+1], [mpo.qd, -mpo.qd, mpo.qD[i+1], -mpo.qD[i+2]])

    return mpo


def _construct_random_mps(qd, nsites: int, rng: np.random.Generator):
    """
    Construct a random matrix product state.
    """
    # virtual bond quantum numbers
    qD = [rng.integers(-1, 2, size=Di) for Di in [1] + list(rng.integers(12, 20, size=nsites-1)) + [1]]
    qD[ 0] = np.array([0])
    qD[-1] = np.array([0])
    psi = ptn.MPS(qd, qD, fill="random", rng=rng)
    for i in range(nsites):
        psi.A[i] *= 5
    return psi


def dmrg_singlesite_data():

    # random number generator
    rng = np.random.default_rng(105)

    # number of lattice sites
    nsites = 7

    # local physical quantum numbers
    qd = np.array([0, -1, 0])

    # Hamiltonian
    H = _construct_random_hermitian_mpo(qd, nsites, rng)
    # must be Hermitian
    Hmat = H.as_matrix()
    assert np.allclose(Hmat, Hmat.conj().T)

    psi_start = _construct_random_mps(qd, nsites, rng)
    assert np.linalg.norm(psi_start.as_vector()) > 0.1

    numsweeps = 6

    psi = copy.deepcopy(psi_start)
    en_sweeps = ptn.dmrg_singlesite(H, psi, numsweeps)

    # reference (numerically exact) ground state energy
    en_min = np.linalg.eigvalsh(Hmat)[0]
    assert en_sweeps[-1] >= en_min

    with h5py.File("data/test_dmrg_singlesite.hdf5", "w") as file:
        file.attrs["qsite"] = qd
        for i, qbond in enumerate(H.qD):
            file.attrs[f"h_qbond{i}"] = qbond
        for i, a in enumerate(H.A):
            # transposition due to different convention for axis ordering
            file[f"h_a{i}"] = a.transpose((2, 0, 1, 3))
        for i, qbond in enumerate(psi_start.qD):
            file.attrs[f"psi_start_qbond{i}"] = qbond
        for i, a in enumerate(psi_start.A):
            # transposition due to different convention for axis ordering
            file[f"psi_start_a{i}"] = a.transpose((1, 0, 2))
        file["en_sweeps"] = en_sweeps
        for i, qbond in enumerate(psi.qD):
            file.attrs[f"psi_qbond{i}"] = qbond
        for i, a in enumerate(psi.A):
            # transposition due to different convention for axis ordering
            file[f"psi_a{i}"] = a.transpose((1, 0, 2))


def dmrg_twosite_data():

    # random number generator
    rng = np.random.default_rng(287)

    # number of lattice sites
    nsites = 11

    # local physical quantum numbers
    qd = np.array([0, 0])

    # Hamiltonian
    H = _construct_random_hermitian_mpo(qd, nsites, rng)
    # must be Hermitian
    Hmat = H.as_matrix()
    assert np.allclose(Hmat, Hmat.conj().T)

    psi_start = _construct_random_mps(qd, nsites, rng)
    assert np.linalg.norm(psi_start.as_vector()) > 0.1

    numsweeps = 4
    tol_split = 1e-5

    psi = copy.deepcopy(psi_start)
    en_sweeps = ptn.dmrg_twosite(H, psi, numsweeps, numiter_lanczos=25, tol_split=tol_split)

    # reference (numerically exact) ground state energy
    en_min = np.linalg.eigvalsh(Hmat)[0]
    assert en_sweeps[-1] >= en_min

    with h5py.File("data/test_dmrg_twosite.hdf5", "w") as file:
        file.attrs["qsite"] = qd
        file.attrs["tol_split"] = tol_split
        for i, qbond in enumerate(H.qD):
            file.attrs[f"h_qbond{i}"] = qbond
        for i, a in enumerate(H.A):
            # transposition due to different convention for axis ordering
            file[f"h_a{i}"] = a.transpose((2, 0, 1, 3))
        for i, qbond in enumerate(psi_start.qD):
            file.attrs[f"psi_start_qbond{i}"] = qbond
        for i, a in enumerate(psi_start.A):
            # transposition due to different convention for axis ordering
            file[f"psi_start_a{i}"] = a.transpose((1, 0, 2))
        file["en_sweeps"] = en_sweeps
        for i, qbond in enumerate(psi.qD):
            file.attrs[f"psi_qbond{i}"] = qbond
        for i, a in enumerate(psi.A):
            # transposition due to different convention for axis ordering
            file[f"psi_a{i}"] = a.transpose((1, 0, 2))


def main():
    dmrg_singlesite_data()
    dmrg_twosite_data()


if __name__ == "__main__":
    main()
