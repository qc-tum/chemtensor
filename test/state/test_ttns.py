import numpy as np
import h5py
import pytenet as ptn


def single_mode_product(a, t, i: int):
    """
    Compute the i-mode product between the matrix `a` and tensor `t`.
    """
    t = np.tensordot(a, t, axes=(1, i))
    # original i-th dimension is now 0-th dimension; move back to i-th place
    t = np.transpose(t, list(range(1, i + 1)) + [0] + list(range(i + 1, t.ndim)))
    return t


def ttns_compress_data():

    # random number generator
    rng = np.random.default_rng(347)

    # tree topology:
    #
    #           6
    #           │
    #           │
    #     2     8 ─── 1
    #      ╲   ╱
    #       ╲ ╱
    #        3
    #        │
    #        │
    #  5 ─── 7 ─── 0
    #        │
    #        │
    #        4
    #
    neighs = (
        (7,),          # neighbors of site 0
        (8,),          # neighbors of site 1
        (3,),          # neighbors of site 2
        (2, 7, 8),     # neighbors of site 3
        (7,),          # neighbors of site 4
        (7,),          # neighbors of site 5
        (8,),          # neighbors of site 6
        (0, 3, 4, 5),  # neighbors of site 7
        (1, 3, 6),     # neighbors of site 8
    )

    nsites_physical = 8

    # local physical dimensions
    d = [3, 2, 3, 2, 1, 2, 5, 3, 1]

    # physical quantum numbers
    qsite = ([rng.integers(-1, 2, size=di) for di in d[:nsites_physical]]
                + [np.zeros(di, dtype=int) for di in d[nsites_physical:]])
    # overall quantum number sector
    qnum_sector = 1

    # virtual bond quantum numbers
    qbonds = {}
    qbonds[(0, 7)] = rng.integers(-1, 2, size=13)
    qbonds[(1, 8)] = rng.integers(-1, 2, size=43)
    qbonds[(2, 3)] = rng.integers(-1, 2, size=18)
    qbonds[(3, 7)] = rng.integers(-2, 3, size=11)
    qbonds[(3, 8)] = rng.integers(-1, 2, size=24)
    qbonds[(4, 7)] = rng.integers(-1, 2, size=12)
    qbonds[(5, 7)] = rng.integers(-1, 2, size= 7)
    qbonds[(6, 8)] = rng.integers(-1, 2, size=32)

    # random local tensors
    alist = []
    for i in range(len(d)):
        dims  = []
        qnums = []
        for j in neighs[i]:
            if j < i:
                dims.append(len(qbonds[(j, i)]))
                qnums.append(qbonds[(j, i)])  # outward direction
        dims.append(d[i])
        qnums.append(qsite[i] - (qnum_sector if i == 0 else 0))
        for j in neighs[i]:
            if j > i:
                dims.append(len(qbonds[(i, j)]))
                qnums.append(-qbonds[(i, j)])  # inward direction
        a = ptn.crandn(dims, rng) / np.sqrt(np.prod(dims))
        # enforce sparsity pattern according to quantum numbers
        ptn.enforce_qsparsity(a, qnums)
        for c, j in enumerate(neighs[i]):
            i_ax = (c if j < i else c + 1)
            # imitate small entanglement by multiplying bonds with small scaling factors
            s = np.exp(-10*(rng.uniform(size=a.shape[i_ax])))
            s /= np.linalg.norm(s)
            a = single_mode_product(np.diag(s), a, i_ax)
        # rescale to achieve norm of order 1
        a *= 5 / np.linalg.norm(a)
        # sparsity pattern should not lead to zero tensors
        assert np.linalg.norm(a) > 0
        alist.append(a)

    with h5py.File("data/test_ttns_compress.hdf5", "w") as file:
        for i, qdi in enumerate(qsite):
            file.attrs[f"qsite{i}"] = qdi
        file.attrs["qnum_sector"] = qnum_sector
        for ij, qbond in qbonds.items():
            file.attrs[f"qbond{ij[0]}{ij[1]}"] = qbond
        for i, a in enumerate(alist):
            file[f"a{i}"] = a


def main():
    ttns_compress_data()


if __name__ == "__main__":
    main()
