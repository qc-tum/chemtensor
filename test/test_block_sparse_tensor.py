import numpy as np
import h5py
from util import crandn, interleave_complex


def block_sparse_tensor_get_block_data():

    # random number generator
    rng = np.random.default_rng(534)

    # dimensions
    dims = (7, 4, 11, 5, 7)

    # tensor degree
    ndim = len(dims)

    # axis directions
    axis_dir = rng.choice((1, -1), size=ndim)

    # quantum numbers
    qnums = [rng.integers(-3, 4, size=d) for d in dims]

    # dense tensors
    t = crandn(dims, rng)
    # enforce sparsity pattern based on quantum numbers
    it = np.nditer(t, flags=["multi_index"], op_flags=["readwrite"])
    for x in it:
        qsum = 0
        for i in range(ndim):
            qsum += axis_dir[i] * qnums[i][it.multi_index[i]]
        if qsum != 0:
            x[...] = 0

    with h5py.File("data/test_block_sparse_tensor_get_block.hdf5", "w") as file:
        file["t"] = interleave_complex(t)
        file.attrs["axis_dir"] = axis_dir
        for i, qn in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qn


def block_sparse_tensor_dot_data():

    # random number generator
    rng = np.random.default_rng(821)

    # tensor degrees
    ndim_s = 5
    ndim_t = 6
    ndim_mult = 3

    # dimensions
    dims = (4, 11, 5, 7, 6, 13, 3, 5)
    assert len(dims) == ndim_s + ndim_t - ndim_mult

    # axis directions
    axis_dir = rng.choice((1, -1), size=len(dims))

    # quantum numbers
    qnums = [rng.integers(-2, 3, size=d) for d in dims]

    # dense tensors
    s = crandn(dims[ :ndim_s], rng)
    t = crandn(dims[-ndim_t:], rng)
    # enforce sparsity pattern based on quantum numbers
    it = np.nditer(s, flags=["multi_index"], op_flags=["readwrite"])
    for x in it:
        qsum = 0
        for i in range(s.ndim):
            qsum += axis_dir[i] * qnums[i][it.multi_index[i]]
        if qsum != 0:
            x[...] = 0
    it = np.nditer(t, flags=["multi_index"], op_flags=["readwrite"])
    for x in it:
        qsum = 0
        for i in range(t.ndim):
            # reversed sign of to-be contracted axes for 't' tensor
            qsum += (-1 if i < ndim_mult else 1)*axis_dir[s.ndim - ndim_mult + i] * qnums[s.ndim - ndim_mult + i][it.multi_index[i]]
        if qsum != 0:
            x[...] = 0

    # contract dense tensors
    r = np.tensordot(s, t, axes=ndim_mult)

    with h5py.File("data/test_block_sparse_tensor_dot.hdf5", "w") as file:
        file["s"] = interleave_complex(s)
        file["t"] = interleave_complex(t)
        file["r"] = interleave_complex(r)
        file.attrs["axis_dir"] = axis_dir
        for i, qn in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qn


def main():
    block_sparse_tensor_get_block_data()
    block_sparse_tensor_dot_data()


if __name__ == "__main__":
    main()
