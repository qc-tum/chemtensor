import numpy as np
import h5py
import sys
sys.path.append("../tensor/")
sys.path.append("../util/")
from hamiltonian import (
    construct_heisenberg_xxx_1d_hamiltonian,
    construct_fermi_hubbard_1d_hamiltonian)


def heisenberg_1d_su2_mpo_data():

    # number of lattice sites
    nsites = 5

    # Hamiltonian parameters
    J = 11./7

    # reference Hamiltonian
    heisenberg_1d_mat = construct_heisenberg_xxx_1d_hamiltonian(nsites, J).todense()

    with h5py.File("data/test_heisenberg_1d_su2_mpo.hdf5", "w") as file:
        file["heisenberg_1d_mat"] = heisenberg_1d_mat


def fermi_hubbard_1d_su2_mpo_data():

    # number of lattice sites
    nsites = 4

    # Hamiltonian parameters
    t  = 17./13
    u  = 7./3
    mu = -5./2

    # reference Hamiltonian
    fermi_hubbard_1d_mat = construct_fermi_hubbard_1d_hamiltonian(nsites, t, u, mu).toarray()
    # permutation of the local site basis states to SU(2) ordering (|00>, |11>, |10>, |01>)
    site_perm = np.array([
        [1., 0., 0., 0.],
        [0., 0., 0., 1.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.]])
    fermi_hubbard_1d_mat = fermi_hubbard_1d_mat.reshape((2*nsites) * (4,))
    for i in range(2*nsites):
        # multiply the i-th axis with 'site_perm'
        mat_idx = list(range(2*nsites))
        mat_idx[i] = 2*nsites
        fermi_hubbard_1d_mat = np.einsum(site_perm, (i, 2*nsites), fermi_hubbard_1d_mat, mat_idx, range(2*nsites))
    fermi_hubbard_1d_mat = fermi_hubbard_1d_mat.reshape(2 * (4**nsites,))

    with h5py.File("data/test_fermi_hubbard_1d_su2_mpo.hdf5", "w") as file:
        file.attrs["nsites"] = nsites
        file.attrs["t"]      = t
        file.attrs["u"]      = u
        file.attrs["mu"]     = mu
        file["fermi_hubbard_1d_mat"] = fermi_hubbard_1d_mat


def main():
    heisenberg_1d_su2_mpo_data()
    fermi_hubbard_1d_su2_mpo_data()


if __name__ == "__main__":
    main()
