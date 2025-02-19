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


def ttno_from_assembly_data():

    rng = np.random.default_rng(812)

    # physical quantum numbers
    qd = np.array([0, -1, 1])
    # number of physical lattice sites
    nsites_physical = 6

    # tree topology:
    #
    #     4     0
    #      ╲   ╱
    #       ╲ ╱
    #        6
    #        │
    #        │
    #  2 ─── 5 ─── 1 ─── 7
    #        │
    #        │
    #        3

    # vertex quantum numbers
    vert_qnums = [
         0,  1, -1,  # (0, 6)
         1,  0,      # (1, 5)
         0,          # (1, 7)
        -1,  0,      # (2, 5)
         0,          # (3, 5)
        -1,  0,      # (4, 6)
         1,  0,      # (5, 6)
    ]
    # offsets
    vos06 = 0
    vos15 = vos06 + 3
    vos17 = vos15 + 2
    vos25 = vos17 + 1
    vos35 = vos25 + 2
    vos46 = vos35 + 1
    vos56 = vos46 + 2

    edge_info = [
        [   # site 0
            OpHyperedgeInfo((vos06 + 1,                                ), ( 4,       )),
            OpHyperedgeInfo((vos06 + 0,                                ), (22,       )),
            OpHyperedgeInfo((vos06 + 2,                                ), ( 7, 17    )),
            OpHyperedgeInfo((vos06 + 0,                                ), (24,       ))],
        [   # site 1
            OpHyperedgeInfo((vos15 + 0, vos17 + 0,                     ), ( 8,       )),
            OpHyperedgeInfo((vos15 + 1, vos17 + 0,                     ), ( 9,       )),
            OpHyperedgeInfo((vos15 + 0, vos17 + 0,                     ), (10, 11    ))],
        [   # site 2
            OpHyperedgeInfo((vos25 + 1,                                ), (12,       )),
            OpHyperedgeInfo((vos25 + 0,                                ), (13,       ))],
        [   # site 3
            OpHyperedgeInfo((vos35 + 0,                                ), (18, 19, 20))],
        [   # site 4
            OpHyperedgeInfo((vos46 + 0,                                ), (17,       )),
            OpHyperedgeInfo((vos46 + 1,                                ), (14,       ))],
        [   # site 5
            OpHyperedgeInfo((vos15 + 0, vos25 + 0, vos35 + 0, vos56 + 1), (14,       )),
            OpHyperedgeInfo((vos15 + 1, vos25 + 1, vos35 + 0, vos56 + 0), (10,       )),
            OpHyperedgeInfo((vos15 + 1, vos25 + 1, vos35 + 0, vos56 + 0), (15, 16    ))],
        [   # site 6
            OpHyperedgeInfo((vos06 + 2, vos46 + 1, vos56 + 0           ), (-1,       )),
            OpHyperedgeInfo((vos06 + 1, vos46 + 0, vos56 + 1           ), (-1, -1    )),
            OpHyperedgeInfo((vos06 + 0, vos46 + 0, vos56 + 0           ), (-1, -1    )),
            OpHyperedgeInfo((vos06 + 0, vos46 + 1, vos56 + 1           ), (-1,       ))],
        [   # site 7
            OpHyperedgeInfo((vos17 + 0,                                ), (-1,       )),
            OpHyperedgeInfo((vos17 + 0,                                ), (-1,       ))],
    ]

    # tensor axes directions, equal to sign factors of virtual bond quantum numbers
    qbond_signs = [
        [-1            ],  # attached to site 0
        [-1, -1        ],  # attached to site 1
        [-1            ],  # attached to site 2
        [-1            ],  # attached to site 3
        [-1            ],  # attached to site 4
        [ 1,  1,  1, -1],  # attached to site 5
        [ 1,  1,  1    ],  # attached to site 6
        [ 1            ],  # attached to site 7
    ]

    # random local operators
    opmap = [rng.standard_normal(2 * (len(qd),)) for oid in range(25)]
    # enforce sparsity pattern according to quantum numbers
    for k in range(nsites_physical):
        for ei in edge_info[k]:
            qsum = np.dot(qbond_signs[k], [vert_qnums[vid] for vid in ei.vids])
            mask = ptn.qnumber_outer_sum([qd, -qd, [qsum]])[:, :, 0]
            for oid in ei.oids:
                opmap[oid] = np.where(mask == 0, opmap[oid], 0)
    # sparsity pattern should not lead to zero operators
    for op in opmap:
        assert np.linalg.norm(op) > 0

    with h5py.File("data/test_ttno_from_assembly.hdf5", "w") as file:
        file["opmap"] = np.array(opmap)


def main():
    ttno_from_assembly_data()


if __name__ == "__main__":
    main()
