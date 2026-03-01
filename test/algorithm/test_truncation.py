import numpy as np
import h5py
from truncation import von_neumann_entropy, retained_bond_indices


def retained_bond_indices_data():

    rng = np.random.default_rng(239)

    # fictitious singular values
    sigma = rng.uniform(0, 1, size=27)

    tol = 3e-2

    ind = retained_bond_indices(sigma, tol)

    # norm and entropy of retained singular values
    norm_sigma = np.linalg.norm(sigma[ind])
    entropy = von_neumann_entropy(sigma[ind])

    with h5py.File("data/test_retained_bond_indices.hdf5", "w") as file:
        file["sigma"]      = sigma
        file["ind"]        = ind
        file["norm_sigma"] = norm_sigma
        file["entropy"]    = entropy
        file.attrs["tol"]  = tol


def main():
    retained_bond_indices_data()


if __name__ == "__main__":
    main()
