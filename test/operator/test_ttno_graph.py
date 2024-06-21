import numpy as np
import h5py
import pytenet as ptn
import sys
sys.path.append("../")
from util import interleave_complex


def ttno_graph_from_opchains_data():

    # random number generator
    rng = np.random.default_rng(307)

    # physical quantum numbers
    qd = np.array([0, -1, 1])
    # local physical dimension
    d = len(qd)
    # number of sites
    nsites = 8

    # identity operator ID
    oid_identity = 5
    # number of local operators
    num_local_ops = 18

    chains = [  #     0   1   2   3   4   5   6   7       0   1   2   3   4   5   6   7
        ptn.OpChain([         6,  9,  0, 15        ], [         0, -1, -1,  1,  0        ], rng.standard_normal(), 2),
        ptn.OpChain([ 8, 16,  2, 11,  3            ], [ 0, -1,  1,  2,  1,  0            ], rng.standard_normal(), 0),
        ptn.OpChain([        10,  1,  9            ], [         0,  1,  0,  0            ], rng.standard_normal(), 2),
        ptn.OpChain([             9,  7, 13, 12,  1], [             0,  0, -1, -1,  1,  0], rng.standard_normal(), 3),
        ptn.OpChain([     5, 17,  4,  7            ], [     0,  0,  1,  1,  0            ], rng.standard_normal(), 1),
        ptn.OpChain([15, 14                        ], [ 0, -1,  0                        ], rng.standard_normal(), 0),
        ptn.OpChain([     9,  2,  1                ], [     0,  0,  1,  0                ], rng.standard_normal(), 1),
        ptn.OpChain([ 7,  5,  5,  5, 14,  5,  9    ], [ 0, -1, -1, -1, -1,  0,  0,  0    ], rng.standard_normal(), 0),
        ptn.OpChain([                10,  5,  7    ], [                 0,  1,  1,  0    ], rng.standard_normal(), 4),
    ]

    graph = ptn.OpGraph.from_opchains(chains, nsites, oid_identity)
    assert graph.is_consistent()
    assert graph.length == nsites

    # random local operators
    opmap = [np.identity(len(qd), dtype=complex) if opid == oid_identity else ptn.crandn((d, d), rng)
             for opid in range(num_local_ops)]
    # enforce sparsity pattern according to quantum numbers
    for chain in chains:
        for i, opid in enumerate(chain.oids):
            qDloc = chain.qnums[i:i+2]
            mask = ptn.qnumber_outer_sum([qd, -qd, [qDloc[0]], [-qDloc[1]]])[:, :, 0, 0]
            opmap[opid] = np.where(mask == 0, opmap[opid], 0)
    # sparsity pattern should not lead to zero operators
    for op in opmap:
        assert np.linalg.norm(op) > 0

    # reference matrix representation of operator chains
    mat_ref = 0
    for chain in chains:
        # including leading and trailing identity maps
        mat_ref = mat_ref + np.kron(np.kron(
            np.identity(len(qd)**chain.istart),
            chain.as_matrix(opmap)),
            np.identity(len(qd)**(nsites - (chain.istart + chain.length))))

    # group sites (0, 4, 6) and (1, 2, 3, 5, 7)
    rank_046_12357 = _operator_partition_rank(mat_ref, d, (0, 4, 6), (1, 2, 3, 5, 7))
    # group sites (1, 7) and (0, 2, 3, 4, 5, 6)
    rank_17_023456 = _operator_partition_rank(mat_ref, d, (1, 7), (0, 2, 3, 4, 5, 6))
    # group sites (2) and (0, 1, 3, 4, 5, 6, 7)
    rank_2_0134567 = _operator_partition_rank(mat_ref, d, (2,), (0, 1, 3, 4, 5, 6, 7))
    # group sites (6) and (0, 1, 2, 3, 4, 5, 7)
    rank_6_0123457 = _operator_partition_rank(mat_ref, d, (6,), (0, 1, 2, 3, 4, 5, 7))

    with h5py.File("data/test_ttno_graph_from_opchains.hdf5", "w") as file:
        for i, chain in enumerate(chains):
            file.attrs[f"/chain{i}/length"] = chain.length
            file.attrs[f"/chain{i}/oids"]   = chain.oids
            file.attrs[f"/chain{i}/qnums"]  = chain.qnums
            file.attrs[f"/chain{i}/coeff"]  = chain.coeff
            file.attrs[f"/chain{i}/istart"] = chain.istart
        file["opmap"] = interleave_complex(np.array(opmap))
        file.attrs["rank_046_12357"] = rank_046_12357
        file.attrs["rank_17_023456"] = rank_17_023456
        # note: local operators opmap[2], opmap[10], opmap[17] acting on site 2
        # span a two-dimensional subspace,
        # hence numerically determined rank is one below general abstract case
        file.attrs["rank_2_0134567"] = rank_2_0134567 + 1
        file.attrs["rank_6_0123457"] = rank_6_0123457


def _operator_partition_rank(u, d: int, sites_a, sites_b):
    """
    Compute the matrix rank when partitioning an operator
    between two subsystems corresponding to 'sites_a' and 'sites_b'.
    """
    sites_a = tuple(sites_a)
    sites_b = tuple(sites_b)
    assert set(sites_a).isdisjoint(sites_b)
    na = len(sites_a)
    nb = len(sites_b)
    u = _permute_operation(u, d, sites_a + sites_b)
    u = u.reshape((d**na, d**nb, d**na, d**nb))
    u = np.transpose(u, (0, 2, 1, 3))
    u = u.reshape((d**(2*na), d**(2*nb)))
    return np.linalg.matrix_rank(u)


def _permute_operation(u: np.ndarray, d: int, perm):
    """
    Find the representation of a matrix after permuting lattice sites.
    """
    nsites = len(perm)
    assert u.shape == (d**nsites, d**nsites)
    perm = list(perm)
    u = np.reshape(u, (2*nsites) * (d,))
    u = np.transpose(u, perm + [nsites + p for p in perm])
    u = np.reshape(u, (d**nsites, d**nsites))
    return u


def main():
    ttno_graph_from_opchains_data()


if __name__ == "__main__":
    main()
