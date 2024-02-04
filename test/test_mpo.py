import numpy as np
import h5py
import pytenet as ptn
from util import interleave_complex


def mpo_from_graph_data():

    rng = np.random.default_rng(463)

    # physical quantum numbers
    qd = np.array([-1, 0, 2, 0])

    # generate a symbolic operator graph
    graph = ptn.opgraph.OpGraph(
        [ptn.opgraph.OpGraphNode( 0, [          ], [ 2,  5],  0),
         ptn.opgraph.OpGraphNode( 1, [ 2        ], [ 1    ],  1),
         ptn.opgraph.OpGraphNode( 2, [ 5        ], [ 4    ],  0),
         ptn.opgraph.OpGraphNode( 3, [ 1,  4    ], [ 7,  0], -1),
         ptn.opgraph.OpGraphNode( 4, [ 7        ], [ 3    ],  0),
         ptn.opgraph.OpGraphNode( 5, [ 0        ], [ 6,  8],  1),
         ptn.opgraph.OpGraphNode( 6, [ 3,  6    ], [10, 11], -1),
         ptn.opgraph.OpGraphNode( 7, [ 8        ], [ 9    ],  0),
         ptn.opgraph.OpGraphNode( 8, [10,  9, 11], [      ],  1)],
        [ptn.opgraph.OpGraphEdge( 2, [ 0,  1], [(  2, -0.6)]),
         ptn.opgraph.OpGraphEdge( 5, [ 0,  2], [(  5,  1.3), ( 11, -0.4)]),
         ptn.opgraph.OpGraphEdge( 1, [ 1,  3], [(  1,  0.4)]),
         ptn.opgraph.OpGraphEdge( 4, [ 2,  3], [(  4, -1.2)]),
         ptn.opgraph.OpGraphEdge( 7, [ 3,  4], [(  7,  0.7)]),
         ptn.opgraph.OpGraphEdge( 0, [ 3,  5], [(  0,  0.5), ( 10,  0.6)]),
         ptn.opgraph.OpGraphEdge( 3, [ 4,  6], [(  3, -1.6), ( 12, -2.1), (  4,  0.5)]),
         ptn.opgraph.OpGraphEdge( 6, [ 5,  6], [(  6,  0.8)]),
         ptn.opgraph.OpGraphEdge( 8, [ 5,  7], [(  8, -0.3)]),
         ptn.opgraph.OpGraphEdge(10, [ 6,  8], [( 10,  0.9)]),
         ptn.opgraph.OpGraphEdge( 9, [ 7,  8], [(  9, -0.2)]),
         ptn.opgraph.OpGraphEdge(11, [ 6,  8], [( 13,  1.2), ( 14, -0.6)])],
        [0, 8])
    assert graph.is_consistent()

    # random local operators
    opmap = [ptn.crandn(2 * (len(qd),), rng) for opid in range(15)]
    # enforce sparsity pattern according to quantum numbers
    for edge in graph.edges.values():
        qDloc = [graph.nodes[nid].qnum for nid in edge.nids]
        mask = ptn.qnumber_outer_sum([qd, -qd, [qDloc[0]], [-qDloc[1]]])[:, :, 0, 0]
        for opid, _ in edge.opics:
            opmap[opid] = np.where(mask == 0, opmap[opid], 0)
    # sparsity pattern should not lead to zero operators
    for op in opmap:
        assert np.linalg.norm(op) > 0

    # convert graph to an MPO
    mpo = ptn.MPO.from_opgraph(qd, graph, opmap)
    assert mpo.bond_dims == [1, 2, 1, 2, 2, 1]
    # compare matrix representations
    assert np.allclose(mpo.as_matrix(), graph.as_matrix(opmap))

    # not storing MPO matrix representation on disk to avoid very large files
    with h5py.File("data/test_mpo_from_graph.hdf5", "w") as file:
        file["opmap"] = interleave_complex(np.array(opmap))


def main():
    mpo_from_graph_data()


if __name__ == "__main__":
    main()
