import numpy as np
import h5py
import sys
sys.path.append("../tensor/")
sys.path.append("../util/")
from hamiltonian import (
    construct_ising_1d_hamiltonian,
    construct_heisenberg_xxz_1d_hamiltonian,
    construct_bose_hubbard_1d_hamiltonian,
    construct_fermi_operators,
    construct_fermi_hubbard_1d_hamiltonian,
    construct_molecular_hamiltonian,
    construct_spin_molecular_hamiltonian)
from crandn import crandn


def ising_1d_mpo_data():

    # number of lattice sites
    nsites = 7

    # Hamiltonian parameters
    J =  5./11
    h = -2./7
    g = 13./8

    # reference Hamiltonian
    ising_1d_mat = construct_ising_1d_hamiltonian(nsites, J, h, g).todense()

    with h5py.File("data/test_ising_1d_mpo.hdf5", "w") as file:
        file["ising_1d_mat"] = ising_1d_mat


def heisenberg_xxz_1d_mpo_data():

    # number of lattice sites
    nsites = 7

    # Hamiltonian parameters
    J = 14./25
    D = 13./8
    h =  2./7

    # reference Hamiltonian
    heisenberg_xxz_1d_mat = construct_heisenberg_xxz_1d_hamiltonian(nsites, J, D, h).todense()

    with h5py.File("data/test_heisenberg_xxz_1d_mpo.hdf5", "w") as file:
        file["heisenberg_xxz_1d_mat"] = heisenberg_xxz_1d_mat


def bose_hubbard_1d_mpo_data():

    # number of lattice sites
    nsites = 5

    # physical dimension per site (maximal occupancy is d - 1)
    d = 3

    # Hamiltonian parameters
    t  =  7./10
    u  = 17./4
    mu = 13./11

    # reference Hamiltonian
    bose_hubbard_1d_mat = construct_bose_hubbard_1d_hamiltonian(nsites, d, t, u, mu).todense()

    with h5py.File("data/test_bose_hubbard_1d_mpo.hdf5", "w") as file:
        file["bose_hubbard_1d_mat"] = bose_hubbard_1d_mat


def fermi_hubbard_1d_mpo_data():

    # number of lattice sites
    nsites = 4

    # Hamiltonian parameters
    t  = 11./9
    u  = 13./4
    mu =  3./7

    # reference Hamiltonian
    fermi_hubbard_1d_mat = construct_fermi_hubbard_1d_hamiltonian(nsites, t, u, mu).todense()

    with h5py.File("data/test_fermi_hubbard_1d_mpo.hdf5", "w") as file:
        file["fermi_hubbard_1d_mat"] = fermi_hubbard_1d_mat


def molecular_hamiltonian_mpo_data():

    rng = np.random.default_rng(532)

    # number of lattice sites
    nsites = 7

    with h5py.File("data/test_molecular_hamiltonian_mpo.hdf5", "w") as file:
        for j in range(4):
            if j == 0:
                # single precision real
                randgen = lambda size: rng.standard_normal(size).astype(np.float32)
            elif j == 1:
                # double precision real
                randgen = lambda size: rng.standard_normal(size)
            elif j == 2:
                # single precision complex
                randgen = lambda size: crandn(size, rng).astype(np.complex64)
            else:
                # double precision complex
                randgen = lambda size: crandn(size, rng)

            # Hamiltonian coefficients
            tkin = randgen(2 * (nsites,))
            vint = randgen(4 * (nsites,))

            # reference Hamiltonian
            molecular_hamiltonian_mat = construct_molecular_hamiltonian(tkin, vint).todense()
            # cast to single-precision if necessary
            if j == 0:
                molecular_hamiltonian_mat = molecular_hamiltonian_mat.astype(np.float32)
            elif j == 2:
                molecular_hamiltonian_mat = molecular_hamiltonian_mat.astype(np.complex64)

            file[f"tkin_t{j}"] = tkin
            file[f"vint_t{j}"] = vint
            file[f"molecular_hamiltonian_mat_t{j}"] = molecular_hamiltonian_mat


def spin_molecular_hamiltonian_mpo_data():

    rng = np.random.default_rng(205)

    # number of spin-endowed lattice sites
    nsites = 4

    with h5py.File("data/test_spin_molecular_hamiltonian_mpo.hdf5", "w") as file:
        for j in range(4):
            if j == 0:
                # single precision real
                randgen = lambda size: rng.standard_normal(size).astype(np.float32)
            elif j == 1:
                # double precision real
                randgen = lambda size: rng.standard_normal(size)
            elif j == 2:
                # single precision complex
                randgen = lambda size: crandn(size, rng).astype(np.complex64)
            else:
                # double precision complex
                randgen = lambda size: crandn(size, rng)

            # Hamiltonian parameters
            tkin = randgen(2 * (nsites,))
            vint = randgen(4 * (nsites,))

            # reference Hamiltonian
            molecular_hamiltonian_mat = construct_spin_molecular_hamiltonian(tkin, vint).todense()
            # cast to single-precision if necessary
            if j == 0:
                molecular_hamiltonian_mat = molecular_hamiltonian_mat.astype(np.float32)
            elif j == 2:
                molecular_hamiltonian_mat = molecular_hamiltonian_mat.astype(np.complex64)

            file[f"tkin_t{j}"] = tkin
            file[f"vint_t{j}"] = vint
            file[f"molecular_hamiltonian_mat_t{j}"] = molecular_hamiltonian_mat


def quadratic_fermionic_mpo_data():

    rng = np.random.default_rng(932)

    # number of lattice sites
    nsites = 7

    # coefficients
    coeffc = rng.standard_normal(nsites)
    coeffa = rng.standard_normal(nsites)

    # reference operator
    clist, alist, _ = construct_fermi_operators(nsites)
    quadratic_fermionic_mat = (
          sum(coeffc[i] * clist[i] for i in range(nsites))
        @ sum(coeffa[i] * alist[i] for i in range(nsites)))

    with h5py.File("data/test_quadratic_fermionic_mpo.hdf5", "w") as file:
        file["coeffc"] = coeffc
        file["coeffa"] = coeffa
        file["quadratic_fermionic_mat"] = quadratic_fermionic_mat.todense()


def quadratic_spin_fermionic_mpo_data():

    rng = np.random.default_rng(273)

    # number of spin-endowed lattice sites
    nsites = 3

    # coefficients
    coeffc = rng.standard_normal(nsites)
    coeffa = rng.standard_normal(nsites)

    # reference operator
    clist, alist, _ = construct_fermi_operators(2 * nsites)
    quadratic_spin_fermionic_mat = [
          sum(coeffc[i] * clist[2*i + sigma] for i in range(nsites))
        @ sum(coeffa[i] * alist[2*i + sigma] for i in range(nsites)) for sigma in (0, 1)]

    with h5py.File("data/test_quadratic_spin_fermionic_mpo.hdf5", "w") as file:
        file["coeffc"] = coeffc
        file["coeffa"] = coeffa
        for sigma in (0, 1):
            file[f"quadratic_spin_fermionic_mat_{sigma}"] = \
                quadratic_spin_fermionic_mat[sigma].todense()


def main():
    ising_1d_mpo_data()
    heisenberg_xxz_1d_mpo_data()
    bose_hubbard_1d_mpo_data()
    fermi_hubbard_1d_mpo_data()
    molecular_hamiltonian_mpo_data()
    spin_molecular_hamiltonian_mpo_data()
    quadratic_fermionic_mpo_data()
    quadratic_spin_fermionic_mpo_data()


if __name__ == "__main__":
    main()
