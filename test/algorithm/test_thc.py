import numpy as np
import h5py
import pytenet as ptn
from pytenet.hamiltonian import _encode_quantum_number_pair


def apply_thc_spin_molecular_hamiltonian_data():

    rng = np.random.default_rng(108)

    # number of spin-endowed lattice sites
    nsites = 5
    # THC rank
    thc_rank = 7

    h = generate_random_thc_hamiltonian(nsites, thc_rank, rng)

    # create a random matrix product state
    # physical particle number and spin quantum numbers (encoded as single integer)
    qN = [0,  1,  1,  2]
    qS = [0, -1,  1,  0]
    qd = [_encode_quantum_number_pair(q[0], q[1]) for q in zip(qN, qS)]
    D = [1, 19, 39, 41, 23, 1]
    # ensure that the MPS does not represent a zero vector
    while True:
        qD = [[_encode_quantum_number_pair(rng.integers(-1, 2), rng.integers(-1, 2))
               for _ in range(Di)]
               for Di in D]
        psi = ptn.MPS(qd, qD, fill='random', rng=rng)
        for i in range(nsites):
            psi.A[i] = psi.A[i].real
        if ptn.norm(psi) > 0:
            break
    # rescale to achieve norm of order 1
    for i in range(psi.nsites):
        psi.A[i] *= 15

    h_psi = h.as_matrix(sparse_format=True) @ psi.as_vector()

    with h5py.File("data/test_apply_thc_spin_molecular_hamiltonian.hdf5", "w") as file:
        file["tkin"]          = h.tkin
        file["thc_kernel"]    = h.thc_kernel
        file["thc_transform"] = h.thc_transform
        for i, qbond in enumerate(psi.qD):
            file.attrs[f"psi_qbond{i}"] = qbond
        for i, ai in enumerate(psi.A):
            # transposition due to different convention for axis ordering
            file[f"psi_a{i}"] = ai.transpose((1, 0, 2))
        file["h_psi"] = h_psi


def thc_spin_molecular_hamiltonian_to_matrix_data():

    # random number generator
    rng = np.random.default_rng(683)

    # number of spin-endowed lattice sites
    nsites = 4
    # THC rank
    thc_rank = 11

    h = generate_random_thc_hamiltonian(nsites, thc_rank, rng)

    with h5py.File("data/test_thc_spin_molecular_hamiltonian_to_matrix.hdf5", "w") as file:
        file["tkin"]          = h.tkin
        file["thc_kernel"]    = h.thc_kernel
        file["thc_transform"] = h.thc_transform
        file["hmat"]          = h.as_matrix()


def generate_random_thc_hamiltonian(nsites: int, thc_rank: int, rng: np.random.Generator):
    """
    Generate a spin molecular Hamiltonian using the tensor hypercontraction
    representation with random coefficients.
    """
    # kinetic coefficients
    tkin = 0.4 * rng.normal(size=(nsites, nsites))
    tkin = 0.5 * (tkin + tkin.T)
    # THC kernel and transformation
    thc_kernel = rng.normal(size=(thc_rank, thc_rank))
    thc_kernel = 0.5 * (thc_kernel + thc_kernel.T)
    thc_transform = 0.4 * rng.normal(size=(nsites, thc_rank))

    return ptn.THCSpinMolecularHamiltonian(tkin, thc_kernel, thc_transform)


def main():
    apply_thc_spin_molecular_hamiltonian_data()
    thc_spin_molecular_hamiltonian_to_matrix_data()


if __name__ == "__main__":
    main()
