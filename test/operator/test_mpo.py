import numpy as np
import h5py
import sys
sys.path.append("../tensor")
sys.path.append("../util/")
from opgraph import OpGraphNode, OpGraphEdge, OpGraph
from mpo import MPO
from block_sparse_util import qnumber_outer_sum
from crandn import crandn


def mpo_from_assembly_data():

    rng = np.random.default_rng(463)

    # physical quantum numbers
    qd = np.array([-1, 0, 2, 0])

    # generate a symbolic operator graph
    graph = OpGraph(
        [OpGraphNode( 0, [          ], [ 2,  5],  0),
         OpGraphNode( 1, [ 2        ], [ 1    ],  1),
         OpGraphNode( 2, [ 5        ], [ 4    ],  0),
         OpGraphNode( 3, [ 1,  4    ], [ 7,  0], -1),
         OpGraphNode( 4, [ 7        ], [ 3    ],  0),
         OpGraphNode( 5, [ 0        ], [ 6,  8],  1),
         OpGraphNode( 6, [ 3,  6    ], [10, 11], -1),
         OpGraphNode( 7, [ 8        ], [ 9    ],  0),
         OpGraphNode( 8, [10,  9, 11], [      ],  1)],
        [OpGraphEdge( 2, [ 0,  1], [(  2, -0.6)]),
         OpGraphEdge( 5, [ 0,  2], [(  5,  1.3), ( 11, -0.4)]),
         OpGraphEdge( 1, [ 1,  3], [(  1,  0.4)]),
         OpGraphEdge( 4, [ 2,  3], [(  4, -1.2)]),
         OpGraphEdge( 7, [ 3,  4], [(  7,  0.7)]),
         OpGraphEdge( 0, [ 3,  5], [(  0,  0.5), ( 10,  0.6)]),
         OpGraphEdge( 3, [ 4,  6], [(  3, -1.6), ( 12, -2.1), (  4,  0.5)]),
         OpGraphEdge( 6, [ 5,  6], [(  6,  0.8)]),
         OpGraphEdge( 8, [ 5,  7], [(  8, -0.3)]),
         OpGraphEdge(10, [ 6,  8], [( 10,  0.9)]),
         OpGraphEdge( 9, [ 7,  8], [(  9, -0.2)]),
         OpGraphEdge(11, [ 6,  8], [( 13,  1.2), ( 14, -0.6)])],
        [0, 8])
    assert graph.is_consistent()

    # random local operators
    opmap = [crandn(2 * (len(qd),), rng) for opid in range(15)]
    # enforce sparsity pattern according to quantum numbers
    for edge in graph.edges.values():
        qbonds_loc = [graph.nodes[nid].qnum for nid in edge.nids]
        mask = qnumber_outer_sum(([qbonds_loc[0]], qd, -qd, [-qbonds_loc[1]]))[0, :, :, 0]
        for opid, _ in edge.opics:
            opmap[opid] = np.where(mask == 0, opmap[opid], 0)
    # sparsity pattern should not lead to zero operators
    for op in opmap:
        assert np.linalg.norm(op) > 0

    # convert graph to an MPO
    mpo = MPO.from_opgraph(qd, graph, opmap)
    assert mpo.bond_dims == [1, 2, 1, 2, 2, 1]
    # compare matrix representations
    assert np.allclose(mpo.to_matrix(), graph.to_matrix(opmap))

    # not storing MPO matrix representation on disk to avoid very large files
    with h5py.File("data/test_mpo_from_assembly.hdf5", "w") as file:
        file["opmap"] = np.array(opmap)


def main():
    mpo_from_assembly_data()


if __name__ == "__main__":
    main()
