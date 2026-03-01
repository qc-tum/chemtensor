from typing import Sequence, Mapping
import numpy as np
from scipy import sparse
from opgraph import OpGraph
from block_sparse_util import is_qsparse, enforce_qsparsity
from crandn import crandn


class MPO:
    """
    Matrix product operator (MPO) class.

    The i-th MPO tensor has dimension `(b[i], d, d, b[i+1])` with `d` the physical
    dimension at each site and `b` the list of virtual bond dimensions.

    Quantum numbers are assumed to be additive and stored as integers.
    `qsite` stores the list of physical quantum numbers at each site (assumed to agree
    for the first and second physical dimension), and `qbonds` the virtual bond quantum numbers.
    The sum of the first physical and left virtual bond quantum number of each
    non-zero tensor entry must be equal to the sum of the second physical and
    right virtual bond quantum number.
    """

    def __init__(self, qsite: Sequence[int], qbonds: Sequence[Sequence[int]],
                 fill=0.0, rng: np.random.Generator=None):
        """
        Create a matrix product operator.

        Args:
            qsite: physical quantum numbers at each site (same for all sites)
            qbonds: virtual bond quantum numbers (list of quantum number lists)
            fill: explicit scalar number to fill MPO tensors with, or
                  "random" to initialize tensors with random complex entries, or
                  "postpone" to leave the MPO tensors unallocated
            rng: (optional) random number generator for drawing entries
        """
        # require NumPy arrays
        self.qsite = np.asarray(qsite)
        self.qbonds = [np.asarray(qb) for qb in qbonds]
        # create list of MPO tensors
        d = len(qsite)
        b = [len(qb) for qb in qbonds]
        nsites = len(b) - 1
        if isinstance(fill, (int, float, complex)):
            self.a = [np.full((b[i], d, d, b[i+1]), fill) for i in range(nsites)]
        elif fill == "random":
            # random complex entries
            if rng is None:
                rng = np.random.default_rng()
            self.a = [crandn((b[i], d, d, b[i+1]), rng) / np.sqrt(b[i]*d*b[i+1])
                      for i in range(nsites)]
        elif fill == "postpone":
            self.a = nsites * [None]
        else:
            raise ValueError(f'fill = {fill} invalid; must be a number, "random" or "postpone".')
        if fill != "postpone":
            # enforce block sparsity structure dictated by quantum numbers
            for i in range(nsites):
                enforce_qsparsity(self.a[i],
                                  (self.qbonds[i], self.qsite, -self.qsite, -self.qbonds[i+1]))

    @classmethod
    def from_opgraph(cls, qsite: Sequence[int], graph: OpGraph, opmap: Mapping):
        """
        Construct a MPO from an operator graph.

        Args:
            qsite: physical quantum numbers at each site (same for all sites)
            graph: symbolic operator graph
            opmap: local operators as dictionary, using operator IDs as keys
            compute_nid_map: whether to construct the map from node IDs to bond location and index

        Returns:
            MPO: MPO representation of the operator graph
        """
        d = len(qsite)
        if d == 0:
            raise ValueError("require at least one physical quantum number")
        a_list = []
        qbonds = []
        # node IDs at current bond site
        nids0 = [graph.nid_terminal[0]]
        qbonds.append([graph.nodes[graph.nid_terminal[0]].qnum])
        while True:
            # node IDs at next bond site
            nids1 = []
            for nid in nids0:
                node = graph.nodes[nid]
                for eid in node.eids[1]:
                    edge = graph.edges[eid]
                    assert edge.nids[0] == nid
                    if edge.nids[1] not in nids1:
                        nids1.append(edge.nids[1])
            if not nids1:   # reached final site
                break
            # sort by node ID
            nids1 = sorted(nids1)
            qbonds.append([graph.nodes[nid].qnum for nid in nids1])
            a = np.zeros((len(nids0), d, d, len(nids1)))
            for i, nid in enumerate(nids0):
                node = graph.nodes[nid]
                for eid in node.eids[1]:
                    edge = graph.edges[eid]
                    j = nids1.index(edge.nids[1])
                    # update local operator in MPO tensor
                    # (supporting multiple edges between same pair of nodes)
                    daij = sum(c * opmap[k] for k, c in edge.opics)
                    if np.iscomplexobj(daij):
                        a = a.astype(complex)
                    a[i, :, :, j] += daij
            a_list.append(a)
            nids0 = nids1
        assert len(a_list) + 1 == len(qbonds)
        op = cls(qsite, qbonds, fill="postpone")
        op.a = a_list
        # consistency check
        for i in range(op.nsites):
            assert is_qsparse(op.a[i], [op.qbonds[i], op.qsite, -op.qsite, -op.qbonds[i+1]]), \
                "sparsity pattern of MPO tensor does not match quantum numbers"
        return op

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
        b.append(self.a[-1].shape[3])
        return b

    def to_matrix(self, sparse_format:bool=False):
        """
        Merge all tensors to obtain the matrix representation on the full Hilbert space.
        """
        if not sparse_format:
            op = self.a[0]
            for i in range(1, len(self.a)):
                op = mpo_merge_tensor_pair(op, self.a[i])
            assert op.ndim == 4
            assert op.shape[0] == 1 and op.shape[3] == 1
            op = op.reshape((op.shape[1], op.shape[2]))
            return op
        else:
            n = len(self.qsite)
            op = self.a[0]
            assert op.shape[0] == 1
            # keep right virtual bond dimension as column dimension
            op = sparse.csr_array(op.reshape((-1, op.shape[3])))
            for i in range(1, len(self.a)):
                t = self.a[i]
                assert t.shape[1] == len(self.qsite)
                op_next_list = []
                for j in range(len(self.qsite)):
                    # explicitly index physical output axis;
                    # compressed sparse column format for subsequent multiplication
                    tj = sparse.csc_array(t[:, j].reshape(t.shape[0], -1))
                    # contract along virtual bond and isolate physical output axis of 'op'
                    op_next_list.append((op @ tj).reshape((n, -1)))
                op = sparse.csr_array(sparse.hstack(op_next_list))
                n *= len(self.qsite)
                op = op.reshape((n**2, -1))
            assert op.shape[1] == 1
            # restore physical input and output dimensions
            op = sparse.csr_array(op.reshape((n, n)))
            return op

    def __add__(self, other):
        """
        Add an MPO to another.
        """
        return mpo_add(self, other)


def mpo_merge_tensor_pair(a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
    """
    Merge two neighboring MPO tensors.
    """
    a = np.einsum(a0, (0, 1, 3, 6), a1, (6, 2, 4, 5), (0, 1, 2, 3, 4, 5), optimize=True)
    # combine original physical dimensions
    s = a.shape
    a = a.reshape((s[0], s[1]*s[2], s[3]*s[4], s[5]))
    return a


def mpo_add(op0: MPO, op1: MPO) -> MPO:
    """
    Logical addition of two MPOs (summing their virtual bond dimensions).
    """
    # number of lattice sites must agree
    assert op0.nsites == op1.nsites
    nsites = op0.nsites
    assert nsites >= 1
    # physical quantum numbers must agree
    assert np.array_equal(op0.qsite, op1.qsite)
    d = len(op0.qsite)

    # initialize with dummy bond quantum numbers
    op = MPO(op0.qsite, (nsites + 1)*[[0]], fill="postpone")

    # combine virtual bond quantum numbers
    # leading and trailing (dummy) bond quantum numbers must agree
    assert np.array_equal(op0.qbonds[ 0], op1.qbonds[ 0])
    assert np.array_equal(op0.qbonds[-1], op1.qbonds[-1])
    op.qbonds[ 0] = op0.qbonds[ 0].copy()
    op.qbonds[-1] = op0.qbonds[-1].copy()
    # intermediate bond quantum numbers
    for i in range(1, nsites):
        op.qbonds[i] = np.concatenate((op0.qbonds[i], op1.qbonds[i]))

    if nsites == 1:
        # simply add MPO tensors
        op.a[0] = op0.a[0] + op1.a[0]
    else:
        # leftmost tensor
        op.a[0] = np.concatenate((op0.a[0], op1.a[0]), axis=3)
        # intermediate tensors
        for i in range(1, nsites - 1):
            s0 = op0.a[i].shape
            s1 = op1.a[i].shape
            # form block-diagonal tensor
            op.a[i] = np.block([[[[op0.a[i], np.zeros((s0[0], d, d, s1[3]))]]],
                                [[[np.zeros((s1[0], d, d, s0[3])), op1.a[i]]]]])
        # rightmost tensor
        op.a[-1] = np.concatenate((op0.a[-1], op1.a[-1]), axis=0)

    # consistency check
    for i in range(nsites):
        assert is_qsparse(op.a[i], (op.qbonds[i], op.qsite, -op.qsite, -op.qbonds[i+1])), \
            "sparsity pattern of MPO tensor does not match quantum numbers"

    return op
