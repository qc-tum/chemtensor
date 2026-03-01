import numpy as np
import h5py
import sys
sys.path.append("../operator/")
sys.path.append("../state/")
sys.path.append("../tensor/")
sys.path.append("../util/")
from chain_ops import mpo_inner_product
from opgraph import OpGraphNode, OpGraphEdge, OpGraph
from mpo import MPO
from mps import MPS
from block_sparse_util import qnumber_outer_sum
from crandn import crandn


def operator_average_coefficient_gradient_data():

    rng = np.random.default_rng(538)

    # physical quantum numbers
    qsite = np.array([-1, 1, 0])

    # virtual bond quantum numbers
    qbonds_psi = [rng.integers(-1, 2, size=Di) for Di in (1,  8, 23,  9, 1)]
    qbonds_chi = [rng.integers(-1, 2, size=Di) for Di in (1, 11, 15,  5, 1)]
    # ensure that leading and trailing virtual bond quantum numbers are compatible
    qbonds_chi[0] = qbonds_chi[-1] + qbonds_psi[0] - qbonds_psi[-1]

    # create random matrix product states
    psi = MPS(qsite, qbonds_psi, fill="random", rng=rng)
    chi = MPS(qsite, qbonds_chi, fill="random", rng=rng)
    # convert tensor entries to single precision
    psi.a = [a.astype(np.complex64) for a in psi.a]
    chi.a = [a.astype(np.complex64) for a in chi.a]
    # rescale tensors such that overall norm is of the order 1
    for i in range(psi.nsites):
        psi.a[i] *= 5
    for i in range(chi.nsites):
        chi.a[i] *= 5

    coeffmap = np.concatenate((np.array([0., 1.]), crandn(7, rng))).astype(np.complex64)

    # generate a symbolic operator graph
    graph = OpGraph(
        [OpGraphNode( 0, [          ], [ 0,  1,  2,  3],  0),  # v0
         OpGraphNode( 1, [ 1,  3    ], [ 4            ], -1),  # v1
         OpGraphNode( 2, [ 0        ], [ 5            ],  0),  # v2
         OpGraphNode( 3, [ 2        ], [ 6            ],  1),  # v3
         OpGraphNode( 4, [ 4        ], [ 7            ], -2),  # v4
         OpGraphNode( 5, [ 5,  6    ], [ 8            ],  0),  # v5
         OpGraphNode( 6, [ 7,  8    ], [ 9, 10, 11    ], -1),  # v6
         OpGraphNode( 7, [ 9, 10, 11], [              ],  0)], # v7
        [OpGraphEdge( 0, [ 0,  2], [( 6, coeffmap[4])]),
         OpGraphEdge( 1, [ 0,  1], [( 5, coeffmap[7])]),
         OpGraphEdge( 2, [ 0,  3], [( 2, coeffmap[7])]),
         OpGraphEdge( 3, [ 0,  1], [( 4, coeffmap[3])]),
         OpGraphEdge( 4, [ 1,  4], [( 5, coeffmap[1]), ( 9, coeffmap[3])]),
         OpGraphEdge( 5, [ 2,  5], [( 0, coeffmap[5])]),
         OpGraphEdge( 6, [ 3,  5], [( 1, coeffmap[2])]),
         OpGraphEdge( 7, [ 4,  6], [( 2, coeffmap[6])]),
         OpGraphEdge( 8, [ 5,  6], [( 4, coeffmap[5]), ( 1, coeffmap[3]), ( 9, coeffmap[3])]),
         OpGraphEdge( 9, [ 6,  7], [( 3, coeffmap[2])]),
         OpGraphEdge(10, [ 6,  7], [( 7, coeffmap[8])]),
         OpGraphEdge(11, [ 6,  7], [( 2, coeffmap[7])])],
        [0, 7])
    assert graph.is_consistent()

    # random local operators
    opmap = [crandn(2 * (len(qsite),), rng).astype(np.complex64) for opid in range(10)]
    # enforce sparsity pattern according to quantum numbers
    for edge in graph.edges.values():
        qbonds_loc = [graph.nodes[nid].qnum for nid in edge.nids]
        mask = qnumber_outer_sum(([qbonds_loc[0]], qsite, -qsite, [-qbonds_loc[1]]))[0, :, :, 0]
        for opid, _ in edge.opics:
            opmap[opid] = np.where(mask == 0, opmap[opid], 0)
    # sparsity pattern should not lead to zero operators
    for op in opmap:
        assert np.linalg.norm(op) > 0

    # convert graph to an MPO
    mpo = MPO.from_opgraph(qsite, graph, opmap)
    assert mpo.bond_dims == [1, 3, 2, 1, 1]
    # compare matrix representations
    assert np.allclose(mpo.to_matrix(), graph.to_matrix(opmap))

    # calculate inner product <chi | op | psi>
    avr = mpo_inner_product(chi, mpo, psi).astype(np.complex64)
    assert avr != 0

    with h5py.File("data/test_operator_average_coefficient_gradient.hdf5", "w") as file:
        for i, qbond in enumerate(qbonds_psi):
            file.attrs[f"qbond_psi_{i}"] = qbond
        for i, qbond in enumerate(qbonds_chi):
            file.attrs[f"qbond_chi_{i}"] = qbond
        for i, ai in enumerate(psi.a):
            file[f"psi_a{i}"] = ai
        for i, ai in enumerate(chi.a):
            file[f"chi_a{i}"] = ai
        file["opmap"]    = np.array(opmap)
        file["coeffmap"] = coeffmap
        file["avr"] = avr


def main():
    operator_average_coefficient_gradient_data()


if __name__ == "__main__":
    main()
