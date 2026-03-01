import numpy as np
import h5py
import sys
sys.path.append("../tensor/")
sys.path.append("../util/")
from bond_ops import split_block_sparse_matrix_svd
from block_sparse_util import enforce_qsparsity
from crandn import crandn


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
        a = 0.02 * crandn(dims, rng).astype(np.complex64)
        # enforce sparsity pattern based on quantum numbers
        enforce_qsparsity(a, [axis_dir[i] * qnums[i] for i in range(a.ndim)])

        file["a"] = a
        file.attrs["axis_dir"] = axis_dir
        for i, qnum in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qnum
        file.attrs["tol"] = tol

        sigma = np.linalg.svd(a, compute_uv=False)

        a0, s, a1, _ = split_block_sparse_matrix_svd(a, qnums[0], qnums[1], tol)
        file.attrs["num_retained"] = len(s)

        # reassemble matrix after splitting, as reference
        a_trunc_plain = (a0 * s).astype(np.complex64) @ a1
        file["a_trunc_plain"] = a_trunc_plain

        # renormalize retained singular values
        s_renrm = (np.linalg.norm(sigma) / np.linalg.norm(s)) * s
        a_trunc_renrm = (a0 * s_renrm).astype(np.complex64) @ a1
        file["a_trunc_renrm"] = a_trunc_renrm


def main():
    split_block_sparse_matrix_svd_data()


if __name__ == "__main__":
    main()
