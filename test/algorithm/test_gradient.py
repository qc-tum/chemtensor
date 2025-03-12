import numpy as np
import h5py
import pytenet as ptn


def operator_average_coefficient_gradient_data():

    rng = np.random.default_rng(538)

    # physical quantum numbers
    qd = np.array([-1, 1, 0])

    # virtual bond quantum numbers
    qD_psi = [rng.integers(-1, 2, size=Di) for Di in (1,  8, 23,  9, 1)]
    qD_chi = [rng.integers(-1, 2, size=Di) for Di in (1, 11, 15,  5, 1)]
    # ensure that leading and trailing virtual bond quantum numbers are compatible
    qD_chi[0] = qD_chi[-1] + qD_psi[0] - qD_psi[-1]

    # create random matrix product states
    psi = ptn.MPS(qd, qD_psi, fill="random", rng=rng)
    chi = ptn.MPS(qd, qD_chi, fill="random", rng=rng)
    # convert tensor entries to single precision
    psi.A = [a.astype(np.complex64) for a in psi.A]
    chi.A = [a.astype(np.complex64) for a in chi.A]
    # rescale tensors such that overall norm is of the order 1
    for i in range(psi.nsites):
        psi.A[i] *= 5
    for i in range(chi.nsites):
        chi.A[i] *= 5

    coeffmap = np.concatenate((np.array([0., 1.]), ptn.crandn(7, rng))).astype(np.complex64)

    # generate a symbolic operator graph
    graph = ptn.OpGraph(
        [ptn.OpGraphNode( 0, [          ], [ 0,  1,  2,  3],  0),  # v0
         ptn.OpGraphNode( 1, [ 1,  3    ], [ 4            ], -1),  # v1
         ptn.OpGraphNode( 2, [ 0        ], [ 5            ],  0),  # v2
         ptn.OpGraphNode( 3, [ 2        ], [ 6            ],  1),  # v3
         ptn.OpGraphNode( 4, [ 4        ], [ 7            ], -2),  # v4
         ptn.OpGraphNode( 5, [ 5,  6    ], [ 8            ],  0),  # v5
         ptn.OpGraphNode( 6, [ 7,  8    ], [ 9, 10, 11    ], -1),  # v6
         ptn.OpGraphNode( 7, [ 9, 10, 11], [              ],  0)], # v7
        [ptn.OpGraphEdge( 0, [ 0,  2], [( 6, coeffmap[4])]),
         ptn.OpGraphEdge( 1, [ 0,  1], [( 5, coeffmap[7])]),
         ptn.OpGraphEdge( 2, [ 0,  3], [( 2, coeffmap[7])]),
         ptn.OpGraphEdge( 3, [ 0,  1], [( 4, coeffmap[3])]),
         ptn.OpGraphEdge( 4, [ 1,  4], [( 5, coeffmap[1]), ( 9, coeffmap[3])]),
         ptn.OpGraphEdge( 5, [ 2,  5], [( 0, coeffmap[5])]),
         ptn.OpGraphEdge( 6, [ 3,  5], [( 1, coeffmap[2])]),
         ptn.OpGraphEdge( 7, [ 4,  6], [( 2, coeffmap[6])]),
         ptn.OpGraphEdge( 8, [ 5,  6], [( 4, coeffmap[5]), ( 1, coeffmap[3]), ( 9, coeffmap[3])]),
         ptn.OpGraphEdge( 9, [ 6,  7], [( 3, coeffmap[2])]),
         ptn.OpGraphEdge(10, [ 6,  7], [( 7, coeffmap[8])]),
         ptn.OpGraphEdge(11, [ 6,  7], [( 2, coeffmap[7])])],
        [0, 7])
    assert graph.is_consistent()

    # random local operators
    opmap = [ptn.crandn(2 * (len(qd),), rng).astype(np.complex64) for opid in range(10)]
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
    assert mpo.bond_dims == [1, 3, 2, 1, 1]
    # compare matrix representations
    assert np.allclose(mpo.as_matrix(), graph.as_matrix(opmap))

    # calculate inner product <chi | op | psi>
    avr = ptn.operator_inner_product(chi, mpo, psi).astype(np.complex64)
    assert avr != 0

    with h5py.File("data/test_operator_average_coefficient_gradient.hdf5", "w") as file:
        for i, qbond in enumerate(qD_psi):
            file.attrs[f"qbond_psi_{i}"] = qbond
        for i, qbond in enumerate(qD_chi):
            file.attrs[f"qbond_chi_{i}"] = qbond
        for i, a in enumerate(psi.A):
            # transposition due to different convention for axis ordering
            file[f"psi_a{i}"] = a.transpose((1, 0, 2))
        for i, a in enumerate(chi.A):
            # transposition due to different convention for axis ordering
            file[f"chi_a{i}"] = a.transpose((1, 0, 2))
        file["opmap"]    = np.array(opmap)
        file["coeffmap"] = coeffmap
        file["avr"] = avr


def main():
    operator_average_coefficient_gradient_data()


if __name__ == "__main__":
    main()
