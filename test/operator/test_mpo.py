import numpy as np
import h5py
import pytenet as ptn


def mpo_from_assembly_data():

    rng = np.random.default_rng(463)

    # physical quantum numbers
    qd = np.array([-1, 0, 2, 0])

    # generate a symbolic operator graph
    graph = ptn.OpGraph(
        [ptn.OpGraphNode( 0, [          ], [ 2,  5],  0),
         ptn.OpGraphNode( 1, [ 2        ], [ 1    ],  1),
         ptn.OpGraphNode( 2, [ 5        ], [ 4    ],  0),
         ptn.OpGraphNode( 3, [ 1,  4    ], [ 7,  0], -1),
         ptn.OpGraphNode( 4, [ 7        ], [ 3    ],  0),
         ptn.OpGraphNode( 5, [ 0        ], [ 6,  8],  1),
         ptn.OpGraphNode( 6, [ 3,  6    ], [10, 11], -1),
         ptn.OpGraphNode( 7, [ 8        ], [ 9    ],  0),
         ptn.OpGraphNode( 8, [10,  9, 11], [      ],  1)],
        [ptn.OpGraphEdge( 2, [ 0,  1], [(  2, -0.6)]),
         ptn.OpGraphEdge( 5, [ 0,  2], [(  5,  1.3), ( 11, -0.4)]),
         ptn.OpGraphEdge( 1, [ 1,  3], [(  1,  0.4)]),
         ptn.OpGraphEdge( 4, [ 2,  3], [(  4, -1.2)]),
         ptn.OpGraphEdge( 7, [ 3,  4], [(  7,  0.7)]),
         ptn.OpGraphEdge( 0, [ 3,  5], [(  0,  0.5), ( 10,  0.6)]),
         ptn.OpGraphEdge( 3, [ 4,  6], [(  3, -1.6), ( 12, -2.1), (  4,  0.5)]),
         ptn.OpGraphEdge( 6, [ 5,  6], [(  6,  0.8)]),
         ptn.OpGraphEdge( 8, [ 5,  7], [(  8, -0.3)]),
         ptn.OpGraphEdge(10, [ 6,  8], [( 10,  0.9)]),
         ptn.OpGraphEdge( 9, [ 7,  8], [(  9, -0.2)]),
         ptn.OpGraphEdge(11, [ 6,  8], [( 13,  1.2), ( 14, -0.6)])],
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
    with h5py.File("data/test_mpo_from_assembly.hdf5", "w") as file:
        file["opmap"] = np.array(opmap)


def main():
    mpo_from_assembly_data()


if __name__ == "__main__":
    main()
