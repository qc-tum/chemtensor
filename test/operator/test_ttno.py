from collections.abc import Sequence
import numpy as np
import h5py
import pytenet as ptn


class OpHyperedgeInfo:
    """
    Operator hypergraph edge info.
    """
    def __init__(self, vids: Sequence[int], oids: Sequence[int]):
        self.vids = tuple(vids)
        self.oids = tuple(oids)


def ttno_from_graph_data():

    rng = np.random.default_rng(812)

    # physical quantum numbers
    qd = np.array([0, -1, 1])
    # number of lattice sites
    nsites = 7

    # tree topology:
    #
    #  4           6
    #    \       /
    #      \   /
    #        0
    #        |
    #        |
    #  2 --- 3 --- 1
    #        |
    #        |
    #        5

    # vertex quantum numbers
    vert_qnums = [
        -1,  0,      # (0, 3)
         1,  0,      # (0, 4)
         0, -1,  1,  # (0, 6)
         1,  0,      # (1, 3)
        -1,  0,      # (2, 3)
         0,          # (3, 5)
    ]

    edge_info = [
        [   # site 0
            OpHyperedgeInfo((     0,  2 + 1,  4 + 0        ), ( 0,       )),
            OpHyperedgeInfo((     1,  2 + 0,  4 + 2        ), ( 4,  1    )),
            OpHyperedgeInfo((     0,  2 + 0,  4 + 0        ), ( 5,  6    )),
            OpHyperedgeInfo((     0,  2 + 1,  4 + 1        ), ( 7,       ))],
        [   # site 1
            OpHyperedgeInfo(( 7 + 0,                       ), ( 8,       )),
            OpHyperedgeInfo(( 7 + 1,                       ), ( 9,       )),
            OpHyperedgeInfo(( 7 + 0,                       ), (10, 11    ))],
        [   # site 2
            OpHyperedgeInfo(( 9 + 1,                       ), (12,       )),
            OpHyperedgeInfo(( 9 + 0,                       ), (13,       ))],
        [   # site 3
            OpHyperedgeInfo((     1,  7 + 0,  9 + 0, 11 + 0), (14,       )),
            OpHyperedgeInfo((     0,  7 + 1,  9 + 1, 11 + 0), (10,       )),
            OpHyperedgeInfo((     0,  7 + 1,  9 + 1, 11 + 0), (15, 16    ))],
        [   # site 4
            OpHyperedgeInfo(( 2 + 0,                       ), (17,       )),
            OpHyperedgeInfo(( 2 + 1,                       ), (14,       ))],
        [   # site 5
            OpHyperedgeInfo((11 + 0,                       ), (18, 19, 20))],
        [   # site 6
            OpHyperedgeInfo(( 4 + 1,                       ), (21,       )),
            OpHyperedgeInfo(( 4 + 0,                       ), (22,       )),
            OpHyperedgeInfo(( 4 + 2,                       ), (23, 17    )),
            OpHyperedgeInfo(( 4 + 0,                       ), (24,       ))],
    ]

    # sign factors of virtual bond quantum numbers
    qbond_signs = [
        [-1, -1, -1    ],  # attached to site 0
        [-1,           ],  # attached to site 1
        [-1            ],  # attached to site 2
        [ 1,  1,  1, -1],  # attached to site 3
        [ 1            ],  # attached to site 4
        [ 1            ],  # attached to site 5
        [ 1            ],  # attached to site 6
    ]

    # random local operators
    opmap = [rng.standard_normal(2 * (len(qd),)) for oid in range(25)]
    # enforce sparsity pattern according to quantum numbers
    for k in range(nsites):
        for ei in edge_info[k]:
            qsum = np.dot(qbond_signs[k], [vert_qnums[vid] for vid in ei.vids])
            mask = ptn.qnumber_outer_sum([qd, -qd, [qsum]])[:, :, 0]
            for oid in ei.oids:
                opmap[oid] = np.where(mask == 0, opmap[oid], 0)
    # sparsity pattern should not lead to zero operators
    for op in opmap:
        assert np.linalg.norm(op) > 0

    with h5py.File("data/test_ttno_from_graph.hdf5", "w") as file:
        file["opmap"] = np.array(opmap)


def main():
    ttno_from_graph_data()


if __name__ == "__main__":
    main()
