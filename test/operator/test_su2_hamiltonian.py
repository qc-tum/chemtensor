import numpy as np
from scipy import sparse
import h5py


def heisenberg_1d_su2_mpo_data():

    # number of lattice sites
    nsites = 5

    # Hamiltonian parameters
    J = 11./7

    # reference Hamiltonian
    heisenberg_1d_mat = construct_heisenberg_1d_hamiltonian(nsites, J).todense()

    with h5py.File("data/test_heisenberg_1d_su2_mpo.hdf5", "w") as file:
        file["heisenberg_1d_mat"] = heisenberg_1d_mat


def construct_heisenberg_1d_hamiltonian(nsites: int, J: float):
    """
    Construct the XXX Heisenberg Hamiltonian `sum J (X X + Y Y + Z Z)`
    on a one-dimensional lattice as sparse matrix.
    """
    # spin operators
    Sup = np.array([[0.,  1.], [0.,  0. ]])
    Sdn = np.array([[0.,  0.], [1.,  0. ]])
    Sz  = np.array([[0.5, 0.], [0., -0.5]])
    # local interaction term
    hint = J * (0.5 * (sparse.kron(Sup, Sdn) + sparse.kron(Sdn, Sup)) + sparse.kron(Sz, Sz))
    return sum(sparse.kron(sparse.identity(2**j),
               sparse.kron(hint,
                           sparse.identity(2**(nsites-j-2)))) for j in range(nsites - 1))


def main():
    heisenberg_1d_su2_mpo_data()


if __name__ == "__main__":
    main()
