import numpy as np


def von_neumann_entropy(sigma):
    """
    Compute the von Neumann entropy of the singular values `sigma`.
    """
    nrm = np.linalg.norm(sigma)
    if nrm == 0:
        return 0
    sq = (sigma / nrm)**2
    sq = sq[sq > 0]
    return sum(-sq * np.log(sq))


def retained_bond_indices(s, tol):
    """
    Indices of retained singular values based on the specified tolerance.
    """
    w = np.linalg.norm(s)
    if w == 0:
        return np.array([], dtype=int)

    # normalized squares
    s = (s / w)**2

    # accumulate values from smallest to largest
    sort_idx = np.argsort(s)
    s[sort_idx] = np.cumsum(s[sort_idx])

    return np.where(s > tol)[0]
