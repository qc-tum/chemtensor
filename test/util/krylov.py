import numpy as np
import warnings


def lanczos_iteration(afunc, vstart, numiter):
    """
    Perform a "matrix free" Lanczos iteration.

    Args:
        afunc:      "matrix free" linear transformation of a given vector
        vstart:     starting vector for iteration
        numiter:    number of iterations (should be much smaller than dimension of vstart)

    Returns:
        tuple: tuple containing
          - alpha:      diagonal real entries of Hessenberg matrix
          - beta:       off-diagonal real entries of Hessenberg matrix
          - v:          `len(vstart) x numiter` matrix containing the orthonormal Lanczos vectors
    """
    vstart = np.asarray(vstart)
    # normalize starting vector
    nrmv = np.linalg.norm(vstart)
    assert nrmv > 0
    vstart = vstart / nrmv

    alpha = np.zeros(numiter)
    beta  = np.zeros(numiter-1)

    v = np.zeros((numiter, len(vstart)), dtype=vstart.dtype)
    v[0] = vstart

    for j in range(numiter-1):
        w = afunc(v[j])
        alpha[j] = np.vdot(w, v[j]).real
        w -= alpha[j]*v[j] + (beta[j-1]*v[j-1] if j > 0 else 0)
        beta[j] = np.linalg.norm(w)
        if beta[j] < 100*len(vstart)*np.finfo(float).eps:
            warnings.warn(
                f"beta[{j}] ~= 0 encountered during Lanczos iteration.",
                RuntimeWarning)
            # premature end of iteration
            numiter = j + 1
            return (alpha[:numiter], beta[:numiter-1], v[:numiter, :].T)
        v[j+1] = w / beta[j]

    # complete final iteration
    j = numiter-1
    w = afunc(v[j])
    alpha[j] = np.vdot(w, v[j]).real
    return (alpha, beta, v.T)


def eigh_krylov(afunc, vstart, numiter, numeig):
    """
    Compute Krylov subspace approximation of eigenvalues and vectors.
    """
    alpha, beta, v = lanczos_iteration(afunc, vstart, numiter)
    # diagonalize Hessenberg matrix
    w_hess, u_hess = eigh_tridiag(alpha, beta)
    # compute Ritz eigenvectors
    u_ritz = v @ u_hess[:, 0:numeig]
    return (w_hess[0:numeig], u_ritz)


def eigh_tridiag(d, e):
    """
    Solve the eigenvalue problem for a real symmetric tridiagonal matrix.

    This function is a work-around of `scipy.linalg.eigh_tridiagonal`,
    which runs into convergence problems in some cases.
    """
    a = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
    return np.linalg.eigh(a)
