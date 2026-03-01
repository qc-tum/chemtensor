from block_sparse_util import block_sparse_svd
from truncation import retained_bond_indices


def split_block_sparse_matrix_svd(a, q0, q1, tol):
    """
    Split a matrix by singular value decomposition,
    taking block sparsity structure dictated by quantum numbers into account,
    and truncate small singular values based on the specified tolerance.
    """
    u, s, v, q = block_sparse_svd(a, q0, q1)
    # truncate small singular values
    idx = retained_bond_indices(s, tol)
    u = u[:, idx]
    v = v[idx, :]
    s = s[idx]
    q = q[idx]
    return (u, s, v, q)
