import numpy as np
import h5py
import pytenet as ptn


def mpo_graph_from_opchains_basic_data():

    # random number generator
    rng = np.random.default_rng(162)

    # identity operator ID
    oid_identity = 0

	# local physical dimension
    d = 3
	# number of sites
    nsites = 5

    coeffmap = np.array([0, 1, -0.7 + 0.1j, -1.8 + 0.5j, 3.2 - 1.1j, 1.3 + 0.4j, 0.2 - 0.3j])

    cids = [4, 2, 5, 4, 3, 6]
    chains = [  #     0   1   2   3   4       0   1   2   3   4
        ptn.OpChain([10,  8,  6,  3,  1], [ 0,  1, -1,  0, -1,  1], coeffmap[cids[0]], 0),
        ptn.OpChain([     9,  6,  3,  1], [     0, -1,  0, -1,  1], coeffmap[cids[1]], 1),
        ptn.OpChain([10,  8,  7,  4,  1], [ 0,  1, -1,  1, -1,  1], coeffmap[cids[2]], 0),
        ptn.OpChain([     9,  7,  4,  1], [     0, -1,  1, -1,  1], coeffmap[cids[3]], 1),
        ptn.OpChain([10,  8,  7,  5,  2], [ 0,  1, -1,  1,  0,  1], coeffmap[cids[4]], 0),
        ptn.OpChain([     9,  7,  5,  2], [     0, -1,  1,  0,  1], coeffmap[cids[5]], 1),
    ]

    # random local operators (ignoring quantum numbers)
    opmap = [np.identity(d) if opid == oid_identity else ptn.crandn((d, d), rng) for opid in range(12)]

    # reference matrix representation of operator chains
    mat = 0
    for chain in chains:
        # including leading and trailing identity maps
        mat = mat + np.kron(np.kron(
            np.identity(d**chain.istart),
            chain.as_matrix(opmap)),
            np.identity(d**(nsites - (chain.istart + chain.length))))

    graph = ptn.OpGraph.from_opchains(chains, nsites, oid_identity)
    assert graph.is_consistent()
    assert graph.length == nsites
    assert np.allclose(graph.as_matrix(opmap), mat)

    # determine virtual bond dimensions
    bond_dims = [1]
    nids0 = [graph.nid_terminal[0]]
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
        nids0 = nids1
        bond_dims.append(len(nids0))

    with h5py.File("data/test_mpo_graph_from_opchains_basic.hdf5", "w") as file:
        file["opmap"] = np.array(opmap)
        file["mat"]   = mat
        file.attrs["bond_dims"] = bond_dims


def mpo_graph_from_opchains_advanced_data():

    # random number generator
    rng = np.random.default_rng(731)

    # physical quantum numbers
    qd = np.array([-1, 0, 2, 0])
    # number of sites
    nsites = 5

    # identity operator ID
    oid_identity = 0

    coeffmap = np.concatenate((np.array([0., 1.]), ptn.crandn(6, rng))).astype(np.complex64)

    cids = [4, 3, 6, 1, 4, 7, 2]
    chains = [  #     0   1   2   3   4       0   1   2   3   4
        ptn.OpChain([ 1,  4, 10,  8    ], [ 0, -1, -1,  0,  0    ], coeffmap[cids[0]], 0),
        ptn.OpChain([ 2                ], [ 0,  0                ], coeffmap[cids[1]], 0),
        ptn.OpChain([    12,  9,  6,  6], [     0,  1,  0,  0,  0], coeffmap[cids[2]], 1),
        ptn.OpChain([ 2,  5,  7,  0, 11], [ 0,  0, -1,  1,  1,  0], coeffmap[cids[3]], 0),
        ptn.OpChain([    14,  3        ], [     0, -1,  0        ], coeffmap[cids[4]], 1),
        ptn.OpChain([         2        ], [         0,  0        ], coeffmap[cids[5]], 2),
        ptn.OpChain([     1, 10        ], [     0, -1,  0        ], coeffmap[cids[6]], 1),
    ]

    graph = ptn.OpGraph.from_opchains(chains, nsites, oid_identity)
    assert graph.is_consistent()
    assert graph.length == nsites

    # random local operators
    opmap = [np.identity(len(qd), dtype=np.complex64) if opid == oid_identity else ptn.crandn(2 * (len(qd),), rng).astype(np.complex64)
             for opid in range(17)]
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
            np.identity(len(qd)**chain.istart, dtype=np.complex64),
            chain.as_matrix(opmap).astype(np.complex64)),
            np.identity(len(qd)**(nsites - (chain.istart + chain.length)), dtype=np.complex64))

    mpo = ptn.MPO.from_opgraph(qd, graph, opmap)

    # compare matrix representations
    assert np.allclose(graph.as_matrix(opmap), mat_ref)
    assert np.allclose(mpo.as_matrix(), mat_ref)

    with h5py.File("data/test_mpo_graph_from_opchains_advanced.hdf5", "w") as file:
        for i, chain in enumerate(chains):
            file.attrs[f"/chain{i}/length"] = chain.length
            file.attrs[f"/chain{i}/oids"]   = chain.oids
            file.attrs[f"/chain{i}/qnums"]  = chain.qnums
            file.attrs[f"/chain{i}/cid"]    = cids[i]
            file.attrs[f"/chain{i}/istart"] = chain.istart
        file["opmap"]    = np.array(opmap)
        file["coeffmap"] = coeffmap
        file.attrs["bond_dims"] = mpo.bond_dims


def main():
    mpo_graph_from_opchains_basic_data()
    mpo_graph_from_opchains_advanced_data()


if __name__ == "__main__":
    main()
