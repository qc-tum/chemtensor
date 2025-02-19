import numpy as np
import h5py


def spin_molecular_hamiltonian_coeffs():

    rng = np.random.default_rng(42)

    # number of spin-endowed lattice sites
    nsites = 9
    # Hamiltonian parameters
    tkin = rng.standard_normal(2 * (nsites,))
    vint = rng.standard_normal(4 * (nsites,))
    # symmetrize coefficients to ensure that Hamiltonian is symmetric
    tkin = 0.5 * (tkin + tkin.conj().T)
    vint = 0.5 * (vint + vint.conj().transpose(2, 3, 0, 1))

    with h5py.File("perf_dmrg_coeffs.hdf5", "w") as file:
        file["tkin"] = tkin
        file["vint"] = vint


if __name__ == "__main__":
    spin_molecular_hamiltonian_coeffs()
