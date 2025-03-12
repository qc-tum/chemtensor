import numpy as np
import h5py
import pytenet as ptn


def neumann_entropy(sigma):
    """
    Compute the von Neumann entropy of singular values 'sigma'.
    """
    nrm = np.linalg.norm(sigma)
    if nrm == 0:
        return 0
    sq = (sigma / nrm)**2
    sq = sq[sq > 0]
    return sum(-sq * np.log(sq))


def retained_bond_indices_data():

    rng = np.random.default_rng(239)

    # fictitious singular values
    sigma = rng.uniform(0, 1, size=27)

    tol = 3e-2

    ind = ptn.retained_bond_indices(sigma, tol)

    # norm and entropy of retained singular values
    norm_sigma = np.linalg.norm(sigma[ind])
    entropy = neumann_entropy(sigma[ind])

    with h5py.File("data/test_retained_bond_indices.hdf5", "w") as file:
        file["sigma"]      = sigma
        file["ind"]        = ind
        file["norm_sigma"] = norm_sigma
        file["entropy"]    = entropy
        file.attrs["tol"]  = tol


def split_block_sparse_matrix_svd_data():

    # random number generator
    rng = np.random.default_rng(814)

    # dimensions
    dims = (181, 191)

    tol = 0.1

    with h5py.File("data/test_split_block_sparse_matrix_svd.hdf5", "w") as file:

        # axis directions
        axis_dir = np.array([-1, 1])

        # quantum numbers
        qnums = [rng.integers(-2, 3, size=d).astype(np.int32) for d in dims]

        # dense tensor
        a = 0.02 * ptn.crandn(dims, rng).astype(np.complex64)
        # enforce sparsity pattern based on quantum numbers
        it = np.nditer(a, flags=["multi_index"], op_flags=["readwrite"])
        for x in it:
            qsum = sum(axis_dir[i] * qnums[i][it.multi_index[i]] for i in range(a.ndim))
            if qsum != 0:
                x[...] = 0

        file["a"] = a
        file.attrs["axis_dir"] = axis_dir
        for i, qn in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qn
        file.attrs["tol"] = tol

        sigma = np.linalg.svd(a, compute_uv=False)

        a0, s, a1, q = ptn.split_matrix_svd(a, qnums[0], qnums[1], tol)
        file.attrs["num_retained"] = len(s)

        # reassemble matrix after splitting, as reference
        a_trunc_plain = (a0 * s).astype(np.complex64) @ a1
        file["a_trunc_plain"] = a_trunc_plain

        # renormalize retained singular values
        s_renrm = (np.linalg.norm(sigma) / np.linalg.norm(s)) * s
        a_trunc_renrm = (a0 * s_renrm).astype(np.complex64) @ a1
        file["a_trunc_renrm"] = a_trunc_renrm


def main():
    retained_bond_indices_data()
    split_block_sparse_matrix_svd_data()


if __name__ == "__main__":
    main()
