import numpy as np
import h5py
import pytenet as ptn


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
    thc_spin_molecular_hamiltonian_to_matrix_data()


if __name__ == "__main__":
    main()
