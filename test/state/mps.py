from typing import Sequence
import numpy as np
from block_sparse_util import qnumber_flatten, enforce_qsparsity, block_sparse_qr
from bond_ops import split_block_sparse_matrix_svd
from crandn import crandn


class MPS:
    """
    Matrix product state (MPS) class.

    The i-th MPS tensor has dimension `(b[i], d, b[i+1])` with `d` the physical
    dimension at each site and `b` the list of virtual bond dimensions.
    """

    def __init__(self, qsite: Sequence[int], qbonds: Sequence[Sequence[int]],
                 fill=0.0, rng: np.random.Generator=None):
        """
        Create a matrix product state.

        Args:
            qsite: physical quantum numbers at each site (same for all sites)
            qbonds: virtual bond quantum numbers (list of quantum number lists)
            fill: explicit scalar number to fill MPS tensors with, or
                  "random"' to initialize tensors with random complex entries
            rng: (optional) random number generator for drawing entries
        """
        # require NumPy arrays
        self.qsite = np.asarray(qsite)
        self.qbonds = [np.asarray(qb) for qb in qbonds]
        # create list of MPS tensors
        d = len(qsite)
        b = [len(qb) for qb in qbonds]
        nsites = len(b) - 1
        # leading and trailing bond dimensions must be 1
        assert b[0] == 1 and b[-1] == 1
        if isinstance(fill, (int, float, complex)):
            self.a = [np.full((b[i], d, b[i+1]), fill) for i in range(nsites)]
        elif fill == "random":
            # random complex entries
            if rng is None:
                rng = np.random.default_rng()
            self.a = [crandn((b[i], d, b[i+1]), rng) / np.sqrt(b[i]*d*b[i+1])
                      for i in range(nsites)]
        else:
            raise ValueError(f'fill = {fill} invalid; must be a number or "random".')
        # enforce block sparsity structure dictated by quantum numbers
        for i in range(nsites):
            enforce_qsparsity(self.a[i], (self.qbonds[i], self.qsite, -self.qbonds[i+1]))

    @property
    def nsites(self) -> int:
        """
        Number of lattice sites.
        """
        return len(self.a)

    @property
    def bond_dims(self) -> list:
        """
        Virtual bond dimensions.
        """
        if len(self.a) == 0:
            return []
        b = [self.a[i].shape[0] for i in range(len(self.a))]
        b.append(self.a[-1].shape[2])
        return b

    def orthonormalize(self, mode="left"):
        """
        Left- or right-orthonormalize the MPS using QR decompositions.
        """
        if len(self.a) == 0:
            return 1

        if mode == "left":
            for i in range(len(self.a) - 1):
                self.a[i], self.a[i+1], self.qbonds[i+1] = mps_local_orthonormalize_left_qr(
                    self.a[i], self.a[i+1], self.qsite, self.qbonds[i:i+2])
            # last tensor
            self.a[-1], t, self.qbonds[-1] = mps_local_orthonormalize_left_qr(
                self.a[-1], np.array([[[1]]]), self.qsite, self.qbonds[-2:])
            # normalization factor (real-valued since diagonal of 'r' matrix is real)
            assert t.shape == (1, 1, 1)
            nrm = t[0, 0, 0].real
            if nrm < 0:
                # flip sign such that normalization factor is always non-negative
                self.a[-1] = -self.a[-1]
                nrm = -nrm
            return nrm
        if mode == "right":
            for i in reversed(range(1, len(self.a))):
                self.a[i], self.a[i-1], self.qbonds[i] = mps_local_orthonormalize_right_qr(
                    self.a[i], self.a[i-1], self.qsite, self.qbonds[i:i+2])
            # first tensor
            self.a[0], t, self.qbonds[0] = mps_local_orthonormalize_right_qr(
                self.a[0], np.array([[[1]]]), self.qsite, self.qbonds[:2])
            # normalization factor (real-valued since diagonal of 'r' matrix is real)
            assert t.shape == (1, 1, 1)
            nrm = t[0, 0, 0].real
            if nrm < 0:
                # flip sign such that normalization factor is always non-negative
                self.a[0] = -self.a[0]
                nrm = -nrm
            return nrm
        raise ValueError(f'mode = {mode} invalid; must be "left" or "right".')

    def to_vector(self) -> np.ndarray:
        """
        Merge all tensors to obtain the vector representation on the full Hilbert space.
        """
        psi = self.a[0]
        for i in range(1, len(self.a)):
            psi = mps_merge_tensor_pair(psi, self.a[i])
        assert psi.ndim == 3
        assert psi.shape[0] == 1 and psi.shape[2] == 1
        return psi.reshape(-1)


def mps_contraction_step_right(a: np.ndarray, b: np.ndarray, r: np.ndarray):
    r"""
    Contraction step from right to left, for example to compute the
    inner product of two matrix product states.

    To-be contracted tensor network::

       ╭───────╮       ╭─────────╮
       │       │       │         │
     ──0   b*  2──   ──1         │
       │       │       │         │
       ╰───1───╯       │         │
           │           │         │
                       │    r    │
           │           │         │
       ╭───1───╮       │         │
       │       │       │         │
     ──0   a   2──   ──0         │
       │       │       │         │
       ╰───────╯       ╰─────────╯
    """

    assert a.ndim == 3
    assert b.ndim == 3
    assert r.ndim == 2
    # multiply with 'a' tensor
    t = np.tensordot(a, r, 1)
    # multiply with conjugated 'b' tensor
    r_next = np.tensordot(t, b.conj(), axes=((1, 2), (1, 2)))
    return r_next


def mps_contraction_step_left(a: np.ndarray, b: np.ndarray, l: np.ndarray):
    r"""
    Contraction step from left to right, for example to compute the
    inner product of two matrix product states.

    To-be contracted tensor network::

     ╭─────────╮       ╭───────╮
     │         │       │       │
     │         1──   ──0   b*  2──
     │         │       │       │
     │         │       ╰───1───╯
     │         │           │
     │    l    │
     │         │           │
     │         │       ╭───1───╮
     │         │       │       │
     │         0──   ──0   a   2──
     │         │       │       │
     ╰─────────╯       ╰───────╯
    """
    assert a.ndim == 3
    assert b.ndim == 3
    assert l.ndim == 2
    # multiply with conjugated 'b' tensor
    t = np.tensordot(l, b.conj(), axes=(1, 0))
    # multiply with 'a' tensor
    l_next = np.tensordot(a, t, axes=((0, 1), (0, 1)))
    return l_next


def mps_vdot(chi: MPS, psi: MPS):
    """
    Compute the dot (scalar) product `<chi | psi>`, complex conjugating `chi`.

    Args:
        chi: wavefunction represented as MPS
        psi: wavefunction represented as MPS

    Returns:
        `<chi | psi>`
    """
    assert psi.nsites == chi.nsites
    if psi.nsites == 0:
        return 0
    # initialize 't' by identity matrix
    t = np.identity(psi.a[-1].shape[2], dtype=psi.a[-1].dtype)
    for i in reversed(range(psi.nsites)):
        t = mps_contraction_step_right(psi.a[i], chi.a[i], t)
    # 't' should now be a 1x1 tensor
    assert t.shape == (1, 1)
    return t[0, 0]


def mps_norm(psi: MPS):
    """
    Compute the standard L2 norm of a matrix product state.
    """
    return np.sqrt(mps_vdot(psi, psi).real)


def mps_local_orthonormalize_left_qr(a: np.ndarray, a_next: np.ndarray,
                                     qsite: Sequence[int], qbonds_outer: Sequence[Sequence[int]]):
    """
    Left-orthonormalize the local site tensor `a` by a QR decomposition,
    and update the tensor at the next site.
    """
    # perform QR decomposition and replace 'a' by reshaped 'q' matrix
    s = a.shape
    assert len(s) == 3
    q0 = qnumber_flatten([qbonds_outer[0], qsite])
    q, r, qbond = block_sparse_qr(a.reshape((s[0]*s[1], s[2])), q0, qbonds_outer[1])
    a = q.reshape((s[0], s[1], q.shape[1]))
    # update a_next tensor: multiply with 'r' from left
    a_next = np.tensordot(r, a_next, (1, 0))
    return (a, a_next, qbond)


def mps_local_orthonormalize_right_qr(a: np.ndarray, a_prev: np.ndarray,
                                      qsite: Sequence[int], qbonds_outer: Sequence[Sequence[int]]):
    """
    Right-orthonormalize the local site tensor `a` by a QR decomposition,
    and update the tensor at the previous site.
    """
    # flip left and right virtual bond dimensions
    a = a.transpose((2, 1, 0))
    # perform QR decomposition and replace 'a' by reshaped 'q' matrix
    s = a.shape
    assert len(s) == 3
    q0 = qnumber_flatten([-qbonds_outer[1], qsite])
    q, r, qbond = block_sparse_qr(a.reshape((s[0]*s[1], s[2])), q0, -qbonds_outer[0])
    a = q.reshape((s[0], s[1], q.shape[1])).transpose((2, 1, 0))
    # update a_prev tensor: multiply with 'r' from right
    a_prev = np.tensordot(a_prev, r, (2, 1))
    return (a, a_prev, -qbond)


def mps_merge_tensor_pair(a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
    """
    Merge two neighboring MPS tensors.
    """
    a = np.einsum(a0, (0, 1, 4), a1, (4, 2, 3), (0, 1, 2, 3), optimize=True)
    # combine original physical dimensions
    a = a.reshape((a.shape[0], a.shape[1]*a.shape[2], a.shape[3]))
    return a


def mps_split_tensor_svd(a: np.ndarray, qd0: Sequence[int], qd1: Sequence[int],
                         qbonds_outer: Sequence[Sequence[int]], svd_distr: str, tol=0):
    """
    Split an MPS tensor with dimension `b0 x d0*d1 x b2` into two MPS tensors
    with dimensions `b0 x d0 x b1` and `b1 x d1 x b2`, respectively.
    """
    assert a.ndim == 3
    d0 = len(qd0)
    d1 = len(qd1)
    assert d0 * d1 == a.shape[1], "physical dimension of MPS tensor must be equal to d0 * d1"
    # reshape as matrix and split by SVD
    s = (a.shape[0], d0, d1, a.shape[2])
    q0 = qnumber_flatten([ qbonds_outer[0], qd0])
    q1 = qnumber_flatten([-qd1, qbonds_outer[1]])
    a0, sigma, a1, qbond = split_block_sparse_matrix_svd(
        a.reshape((s[0]*s[1], s[2]*s[3])), q0, q1, tol)
    a0 = a0.reshape((s[0], s[1], len(sigma)))
    a1 = a1.reshape((len(sigma), s[2], s[3]))
    # use broadcasting to distribute singular values
    if svd_distr == "left":
        a0 = a0 * sigma
    elif svd_distr == "right":
        a1 = a1 * sigma[:, None, None]
    else:
        raise ValueError('svd_distr parameter must be "left" or "right".')
    return (a0, a1, qbond)
