import numpy as np
import h5py
from util import crandn, interleave_complex


def block_sparse_tensor_copy_data():

    # random number generator
    rng = np.random.default_rng(273)

    # dimensions
    dims = (5, 13, 4, 7)

    # tensor degree
    ndim = len(dims)

    # axis directions
    axis_dir = rng.choice((1, -1), size=ndim)

    # quantum numbers
    qnums = [rng.integers(-3, 4, size=d).astype(np.int32) for d in dims]

    # dense tensor representation
    t = crandn(dims, rng)
    # enforce sparsity pattern based on quantum numbers
    it = np.nditer(t, flags=["multi_index"], op_flags=["readwrite"])
    for x in it:
        qsum = sum(axis_dir[i] * qnums[i][it.multi_index[i]] for i in range(ndim))
        if qsum != 0:
            x[...] = 0

    with h5py.File("data/test_block_sparse_tensor_copy.hdf5", "w") as file:
        file["t"] = interleave_complex(t)
        file.attrs["axis_dir"] = axis_dir
        for i, qn in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qn


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
    qnums = [rng.integers(-3, 4, size=d).astype(np.int32) for d in dims]

    # dense tensors
    t = crandn(dims, rng)
    # enforce sparsity pattern based on quantum numbers
    it = np.nditer(t, flags=["multi_index"], op_flags=["readwrite"])
    for x in it:
        qsum = sum(axis_dir[i] * qnums[i][it.multi_index[i]] for i in range(ndim))
        if qsum != 0:
            x[...] = 0

    with h5py.File("data/test_block_sparse_tensor_get_block.hdf5", "w") as file:
        file["t"] = interleave_complex(t)
        file.attrs["axis_dir"] = axis_dir
        for i, qn in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qn


def block_sparse_tensor_norm2_data():

    # random number generator
    rng = np.random.default_rng(765)

    # dimensions
    dims = (4, 6, 13, 5)

    # tensor degree
    ndim = len(dims)

    # axis directions
    axis_dir = rng.choice((1, -1), size=ndim)

    # quantum numbers
    qnums = [rng.integers(-3, 4, size=d).astype(np.int32) for d in dims]

    # dense tensor
    t = crandn(dims, rng)
    # enforce sparsity pattern based on quantum numbers
    it = np.nditer(t, flags=["multi_index"], op_flags=["readwrite"])
    for x in it:
        qsum = sum(axis_dir[i] * qnums[i][it.multi_index[i]] for i in range(ndim))
        if qsum != 0:
            x[...] = 0

    nrm = np.linalg.norm(t.reshape(-1), ord=2)

    with h5py.File("data/test_block_sparse_tensor_norm2.hdf5", "w") as file:
        file["t"] = interleave_complex(t)
        file["nrm"] = nrm
        file.attrs["axis_dir"] = axis_dir
        for i, qn in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qn


def block_sparse_tensor_transpose_data():

    # random number generator
    rng = np.random.default_rng(432)

    # dimensions
    dims = (5, 4, 11, 7)

    # tensor degree
    ndim = len(dims)

    # axis directions
    axis_dir = rng.choice((1, -1), size=ndim)

    # quantum numbers
    qnums = [rng.integers(-2, 3, size=d).astype(np.int32) for d in dims]

    # tensor with random entries
    t = rng.standard_normal(dims)
    # enforce sparsity pattern based on quantum numbers
    it = np.nditer(t, flags=["multi_index"], op_flags=["readwrite"])
    for x in it:
        qsum = sum(axis_dir[i] * qnums[i][it.multi_index[i]] for i in range(ndim))
        if qsum != 0:
            x[...] = 0

    # transpose tensor
    t_tp = np.transpose(t, (1, 3, 2, 0))

    with h5py.File("data/test_block_sparse_tensor_transpose.hdf5", "w") as file:
        file["t"]    = t
        file["t_tp"] = t_tp
        file.attrs["axis_dir"] = axis_dir
        for i, qn in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qn


def block_sparse_tensor_reshape_data():

    # random number generator
    rng = np.random.default_rng(431)

    # dimensions
    dims = (5, 7, 4, 11, 3)

    # tensor degree
    ndim = len(dims)

    # axis directions
    axis_dir = rng.choice((1, -1), size=ndim)

    # quantum numbers
    qnums = [rng.integers(-2, 3, size=d).astype(np.int32) for d in dims]

    # tensor with random entries
    t = crandn(dims, rng)
    # enforce sparsity pattern based on quantum numbers
    it = np.nditer(t, flags=["multi_index"], op_flags=["readwrite"])
    for x in it:
        qsum = sum(axis_dir[i] * qnums[i][it.multi_index[i]] for i in range(ndim))
        if qsum != 0:
            x[...] = 0

    # save tensor to disk - reshaping of dense tensor is straightforward
    with h5py.File("data/test_block_sparse_tensor_reshape.hdf5", "w") as file:
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
    qnums = [rng.integers(-2, 3, size=d).astype(np.int32) for d in dims]

    # dense tensors
    s = crandn(dims[ :ndim_s], rng)
    t = crandn(dims[-ndim_t:], rng)
    # enforce sparsity pattern based on quantum numbers
    it = np.nditer(s, flags=["multi_index"], op_flags=["readwrite"])
    for x in it:
        qsum = sum(axis_dir[i] * qnums[i][it.multi_index[i]] for i in range(s.ndim))
        if qsum != 0:
            x[...] = 0
    it = np.nditer(t, flags=["multi_index"], op_flags=["readwrite"])
    for x in it:
        # reversed sign of to-be contracted axes for 't' tensor
        qsum = sum((-1 if i < ndim_mult else 1) * axis_dir[s.ndim - ndim_mult + i]
                   * qnums[s.ndim - ndim_mult + i][it.multi_index[i]] for i in range(t.ndim))
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


def block_sparse_tensor_qr_data():

    # random number generator
    rng = np.random.default_rng(817)

    # dimensions
    dims = (173, 105)

    with h5py.File("data/test_block_sparse_tensor_qr.hdf5", "w") as file:
        for c in range(2):

            # axis directions
            axis_dir = np.array([-1, 1]) if c == 0 else np.array([-1, -1])

            # quantum numbers (can never match for c == 1)
            qnums = [rng.integers(-2, 3, size=d).astype(np.int32) if c == 0
                else rng.integers(-3, 0, size=d).astype(np.int32) for d in dims]

            # dense tensor
            a = crandn(dims, rng)
            # enforce sparsity pattern based on quantum numbers
            it = np.nditer(a, flags=["multi_index"], op_flags=["readwrite"])
            for x in it:
                qsum = sum(axis_dir[i] * qnums[i][it.multi_index[i]] for i in range(a.ndim))
                if qsum != 0:
                    x[...] = 0

            file[f"a{c}"] = interleave_complex(a)
            file.attrs[f"axis_dir{c}"] = axis_dir
            for i, qn in enumerate(qnums):
                file.attrs[f"qnums{c}{i}"] = qn


def block_sparse_tensor_rq_data():

    # random number generator
    rng = np.random.default_rng(491)

    # dimensions
    dims = (163, 115)

    with h5py.File("data/test_block_sparse_tensor_rq.hdf5", "w") as file:
        for c in range(2):

            # axis directions
            axis_dir = np.array([-1, 1]) if c == 0 else np.array([-1, -1])

            # quantum numbers (can never match for c == 1)
            qnums = [rng.integers(-2, 3, size=d).astype(np.int32) if c == 0
                else rng.integers(-3, 0, size=d).astype(np.int32) for d in dims]

            # dense tensor
            a = crandn(dims, rng)
            # enforce sparsity pattern based on quantum numbers
            it = np.nditer(a, flags=["multi_index"], op_flags=["readwrite"])
            for x in it:
                qsum = sum(axis_dir[i] * qnums[i][it.multi_index[i]] for i in range(a.ndim))
                if qsum != 0:
                    x[...] = 0

            file[f"a{c}"] = interleave_complex(a)
            file.attrs[f"axis_dir{c}"] = axis_dir
            for i, qn in enumerate(qnums):
                file.attrs[f"qnums{c}{i}"] = qn


def block_sparse_tensor_svd_data():

    # random number generator
    rng = np.random.default_rng(245)

    # dimensions
    dims = (167, 98)

    with h5py.File("data/test_block_sparse_tensor_svd.hdf5", "w") as file:
        for c in range(2):

            # axis directions
            axis_dir = np.array([-1, 1]) if c == 0 else np.array([-1, -1])

            # quantum numbers (can never match for c == 1)
            qnums = [rng.integers(-2, 3, size=d).astype(np.int32) if c == 0
                else rng.integers(-3, 0, size=d).astype(np.int32) for d in dims]

            # dense tensor
            a = crandn(dims, rng)
            # enforce sparsity pattern based on quantum numbers
            it = np.nditer(a, flags=["multi_index"], op_flags=["readwrite"])
            for x in it:
                qsum = sum(axis_dir[i] * qnums[i][it.multi_index[i]] for i in range(a.ndim))
                if qsum != 0:
                    x[...] = 0

            file[f"a{c}"] = interleave_complex(a)
            file.attrs[f"axis_dir{c}"] = axis_dir
            for i, qn in enumerate(qnums):
                file.attrs[f"qnums{c}{i}"] = qn


def main():
    block_sparse_tensor_copy_data()
    block_sparse_tensor_get_block_data()
    block_sparse_tensor_norm2_data()
    block_sparse_tensor_transpose_data()
    block_sparse_tensor_reshape_data()
    block_sparse_tensor_dot_data()
    block_sparse_tensor_qr_data()
    block_sparse_tensor_rq_data()
    block_sparse_tensor_svd_data()


if __name__ == "__main__":
    main()
