import numpy as np
import h5py
import pytenet as ptn


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
    t = ptn.crandn(dims, rng)
    # enforce sparsity pattern based on quantum numbers
    ptn.enforce_qsparsity(t, [axis_dir[i] * qnums[i] for i in range(ndim)])

    with h5py.File("data/test_block_sparse_tensor_copy.hdf5", "w") as file:
        file["t"] = t
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
    t = ptn.crandn(dims, rng)
    # enforce sparsity pattern based on quantum numbers
    ptn.enforce_qsparsity(t, [axis_dir[i] * qnums[i] for i in range(ndim)])

    with h5py.File("data/test_block_sparse_tensor_get_block.hdf5", "w") as file:
        file["t"] = t
        file.attrs["axis_dir"] = axis_dir
        for i, qn in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qn


def block_sparse_tensor_cyclic_partial_trace_data():

    # random number generator
    rng = np.random.default_rng(621)

    # dimensions
    dims = (7, 4, 3, 2, 5, 7, 4)

    # tensor degree
    ndim = len(dims)
    ndim_trace = 2

    # axis directions
    axis_dir = rng.choice((1, -1), size=ndim)
    for j in range(ndim_trace):
        axis_dir[ndim - ndim_trace + j] = -axis_dir[j]

    # quantum numbers
    qnums = [rng.integers(-2, 3, size=d).astype(np.int32) for d in dims[:(ndim-ndim_trace)]]
    for j in range(ndim_trace):
        qnums.append(qnums[j])

    # dense tensor
    t = ptn.crandn(dims, rng).astype(np.complex64)
    # enforce sparsity pattern based on quantum numbers
    ptn.enforce_qsparsity(t, [axis_dir[i] * qnums[i] for i in range(ndim)])

    t_tr = np.trace(np.trace(t, axis1=0, axis2=ndim-ndim_trace), axis1=0, axis2=ndim-ndim_trace-1)

    with h5py.File("data/test_block_sparse_tensor_cyclic_partial_trace.hdf5", "w") as file:
        file["t"] = t
        file["t_tr"] = t_tr
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
    t = ptn.crandn(dims, rng)
    # enforce sparsity pattern based on quantum numbers
    ptn.enforce_qsparsity(t, [axis_dir[i] * qnums[i] for i in range(ndim)])

    nrm = np.linalg.norm(t.reshape(-1), ord=2)

    with h5py.File("data/test_block_sparse_tensor_norm2.hdf5", "w") as file:
        file["t"] = t
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
    ptn.enforce_qsparsity(t, [axis_dir[i] * qnums[i] for i in range(ndim)])

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
    t = ptn.crandn(dims, rng)
    # enforce sparsity pattern based on quantum numbers
    ptn.enforce_qsparsity(t, [axis_dir[i] * qnums[i] for i in range(ndim)])

    # save tensor to disk - reshaping of dense tensor is straightforward
    with h5py.File("data/test_block_sparse_tensor_reshape.hdf5", "w") as file:
        file["t"] = t
        file.attrs["axis_dir"] = axis_dir
        for i, qn in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qn


def matricize(a, i: int):
    """
    Compute the matricization of a tensor along the i-th axis.
    """
    s = (int(np.prod(a.shape[:i])), a.shape[i], int(np.prod(a.shape[i+1:])))
    a = a.reshape(s)
    a = a.transpose((1, 0, 2)).reshape((s[1], s[0]*s[2]))
    return a


def block_sparse_tensor_matricize_axis_data():

    # random number generator
    rng = np.random.default_rng(391)

    # dimensions
    dims = (7, 6, 4, 11)

    # tensor degree
    ndim = len(dims)

    # axis directions
    axis_dir = rng.choice((1, -1), size=ndim)

    # quantum numbers
    qnums = [rng.integers(-2, 3, size=d).astype(np.int32) for d in dims]

    # tensor with random entries
    t = rng.standard_normal(dims).astype(np.float32)
    # enforce sparsity pattern based on quantum numbers
    ptn.enforce_qsparsity(t, [axis_dir[i] * qnums[i] for i in range(ndim)])

    # matricize 't' along each possible axis
    mat = [matricize(t, i) for i in range(ndim)]

    with h5py.File("data/test_block_sparse_tensor_matricize_axis.hdf5", "w") as file:
        file["t"] = t
        file.attrs["axis_dir"] = axis_dir
        for i, qn in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qn
        for i in range(ndim):
            file[f"mat{i}"] = mat[i]


def block_sparse_tensor_slice_data():

    # random number generator
    rng = np.random.default_rng(741)

    # dimensions
    dims = (6, 7, 11, 13)

    # tensor degree
    ndim = len(dims)

    # axis directions
    axis_dir = rng.choice((1, -1), size=ndim)

    # quantum numbers
    qnums = [rng.integers(-2, 3, size=d).astype(np.int32) for d in dims]

    # tensor with random entries
    t = rng.standard_normal(dims).astype(np.float32)
    # enforce sparsity pattern based on quantum numbers
    ptn.enforce_qsparsity(t, [axis_dir[i] * qnums[i] for i in range(ndim)])

    # slice along axis 2
    ind = rng.integers(0, t.shape[2], 17)
    s = t[:, :, ind, :]

    with h5py.File("data/test_block_sparse_tensor_slice.hdf5", "w") as file:
        file["t"] = t
        file["s"] = s
        file.attrs["axis_dir"] = axis_dir
        for i, qn in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qn
        file.attrs["ind"] = ind


def block_sparse_tensor_multiply_pointwise_vector_data():

    # random number generator
    rng = np.random.default_rng(418)

    # dimensions
    dims = (4, 7, 3, 11)

    # tensor degree
    ndim = len(dims)

    # axis directions
    axis_dir = rng.choice((1, -1), size=ndim)

    # quantum numbers
    qnums = [rng.integers(-2, 3, size=d).astype(np.int32) for d in dims]

    # tensor with random entries
    s = rng.standard_normal(dims).astype(np.float32)
    # enforce sparsity pattern based on quantum numbers
    ptn.enforce_qsparsity(s, [axis_dir[i] * qnums[i] for i in range(ndim)])

    t = [rng.standard_normal(dim).astype(np.float32) for dim in [dims[0], dims[-1]]]

    s_mult_t = [s * t[0][:, None, None, None], s * t[1]]

    with h5py.File("data/test_block_sparse_tensor_multiply_pointwise_vector.hdf5", "w") as file:
        file["s"] = s
        file.attrs["axis_dir"] = axis_dir
        for i, qn in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qn
        for i in range(2):
            file[f"t{i}"] = t[i]
            file[f"s_mult_t{i}"] = s_mult_t[i]


def block_sparse_tensor_multiply_axis_data():

    # random number generator
    rng = np.random.default_rng(730)

    i_ax = 2

    # dimensions
    s_dim  = (7, 4, 11, 5)
    t0_dim = (s_dim[i_ax],) + (6, 9)
    t1_dim = (8, 3) + (s_dim[i_ax],)

    # axis directions
    s_axis_dir  = rng.choice((1, -1), size=len(s_dim))
    t0_axis_dir = np.concatenate(((-s_axis_dir[i_ax],), rng.choice((1, -1), size=len(t0_dim)-1)))
    t1_axis_dir = np.concatenate((rng.choice((1, -1), size=len(t1_dim)-1), (-s_axis_dir[i_ax],)))

    # quantum numbers
    s_qnums  = [rng.integers(-2, 3, size=d).astype(np.int32) for d in s_dim]
    t0_qnums = [s_qnums[i_ax]] + [rng.integers(-2, 3, size=d).astype(np.int32) for d in t0_dim[1:]]
    t1_qnums = [rng.integers(-2, 3, size=d).astype(np.int32) for d in t1_dim[:-1]] + [s_qnums[i_ax]]

    # dense tensors
    s  = ptn.crandn(s_dim,  rng)
    t0 = ptn.crandn(t0_dim, rng)
    t1 = ptn.crandn(t1_dim, rng)
    # enforce sparsity pattern based on quantum numbers
    ptn.enforce_qsparsity(s,  [ s_axis_dir[i] *  s_qnums[i] for i in range( s.ndim)])
    ptn.enforce_qsparsity(t0, [t0_axis_dir[i] * t0_qnums[i] for i in range(t0.ndim)])
    ptn.enforce_qsparsity(t1, [t1_axis_dir[i] * t1_qnums[i] for i in range(t1.ndim)])

    # contract with axis 'i_ax' of 's'
    r0 = np.einsum(s, (0, 1, 5, 4), t0, (5, 2, 3), (0, 1, 2, 3, 4))
    r1 = np.einsum(s, (0, 1, 5, 4), t1, (2, 3, 5), (0, 1, 2, 3, 4))

    with h5py.File("data/test_block_sparse_tensor_multiply_axis.hdf5", "w") as file:
        file["s"]  = s
        file["t0"] = t0
        file["r0"] = r0
        file["t1"] = t1
        file["r1"] = r1
        file.attrs["s_axis_dir"]  = s_axis_dir
        file.attrs["t0_axis_dir"] = t0_axis_dir
        file.attrs["t1_axis_dir"] = t1_axis_dir
        for i, qn in enumerate(s_qnums):
            file.attrs[f"s_qnums{i}"] = qn
        for i, qn in enumerate(t0_qnums):
            file.attrs[f"t0_qnums{i}"] = qn
        for i, qn in enumerate(t1_qnums):
            file.attrs[f"t1_qnums{i}"] = qn


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
    s = ptn.crandn(dims[ :ndim_s], rng)
    t = ptn.crandn(dims[-ndim_t:], rng)
    # enforce sparsity pattern based on quantum numbers
    ptn.enforce_qsparsity(s, [axis_dir[i] * qnums[i] for i in range(s.ndim)])
    # reversed sign of to-be contracted axes for 't' tensor
    ptn.enforce_qsparsity(t,
        [(-1 if i < ndim_mult else 1) * axis_dir[s.ndim - ndim_mult + i] *
         qnums[s.ndim - ndim_mult + i] for i in range(t.ndim)])

    # contract dense tensors
    r = np.tensordot(s, t, axes=ndim_mult)

    with h5py.File("data/test_block_sparse_tensor_dot.hdf5", "w") as file:
        file["s"] = s
        file["t"] = t
        file["r"] = r
        file.attrs["axis_dir"] = axis_dir
        for i, qn in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qn


def block_sparse_tensor_concatenate_data():

    # random number generator
    rng = np.random.default_rng(943)

    # number of dimensions (degree) of each tensor
    ndim = 4
    # to-be concatenated axis
    i_ax = 1

    # dimensions
    tdims = [
        (13, 17, 11, 7),
        (13,  4, 11, 7),
        (13, 12, 11, 7),
        (13,  3, 11, 7),
        (13, 19, 11, 7)]

    dims_concat = [tdims[0][i] if i != i_ax else sum(dims[i_ax] for dims in tdims) for i in range(ndim)]
    d_concat_ranges = np.cumsum([0] + [dims[i_ax] for dims in tdims])

    # tensors
    tlist = [rng.standard_normal(dims).astype(np.float32) for dims in tdims]

    # axis directions
    axis_dir = rng.choice((1, -1), size=ndim)

    # quantum numbers
    qnums = [rng.integers(-2, 3, size=d) for d in dims_concat]
    t_qnums = [
        [qnums[i] if i != i_ax else qnums[i_ax][d_concat_ranges[j]:d_concat_ranges[j+1]] for i in range(ndim)]
        for j in range(len(tdims))]

    # enforce sparsity pattern based on quantum numbers
    for j, t in enumerate(tlist):
        ptn.enforce_qsparsity(t, [axis_dir[i] * t_qnums[j][i] for i in range(ndim)])

    r = np.concatenate(tlist, axis=i_ax)

    with h5py.File("data/test_block_sparse_tensor_concatenate.hdf5", "w") as file:
        file.attrs["axis_dir"] = axis_dir
        for j, t in enumerate(tlist):
            file[f"t{j}"] = t
            for i, qn in enumerate(t_qnums[j]):
                file.attrs[f"t{j}_qnums{i}"] = qn
        file["r"] = r


def block_sparse_tensor_block_diag_data():

    # random number generator
    rng = np.random.default_rng(247)

    # number of dimensions (degree) of each tensor
    ndim = 5
    # axes used for blocking
    i_ax = (1, 3, 4)

    # dimensions
    tdims = [
        (4,  2,  5,  7,  1),
        (4, 11,  5,  4,  3),
        (4,  3,  5, 10,  5),
        (4,  1,  5,  3,  8),
        (4,  4,  5,  1,  4),
        (4,  5,  5,  2,  7)]

    dims_block_diag = [tdims[0][i] if i not in i_ax else sum(dims[i] for dims in tdims) for i in range(ndim)]
    d_block_ranges  = { i: np.cumsum([0] + [dims[i] for dims in tdims]) for i in i_ax }

    # tensors
    tlist = [rng.standard_normal(dims).astype(np.float32) for dims in tdims]

    # axis directions
    axis_dir = rng.choice((1, -1), size=ndim)

    # quantum numbers
    qnums = [rng.integers(-2, 3, size=d) for d in dims_block_diag]
    t_qnums = [
        [qnums[i] if i not in i_ax else qnums[i][d_block_ranges[i][j]:d_block_ranges[i][j+1]] for i in range(ndim)]
        for j in range(len(tdims))]

    # enforce sparsity pattern based on quantum numbers
    for j, t in enumerate(tlist):
        ptn.enforce_qsparsity(t, [axis_dir[i] * t_qnums[j][i] for i in range(ndim)])

    r = np.zeros(dims_block_diag, dtype=np.float32)
    for j in range(len(tlist)):
        r[:,
          d_block_ranges[1][j]:d_block_ranges[1][j+1],
          :,
          d_block_ranges[3][j]:d_block_ranges[3][j+1],
          d_block_ranges[4][j]:d_block_ranges[4][j+1]] = tlist[j]

    with h5py.File("data/test_block_sparse_tensor_block_diag.hdf5", "w") as file:
        file.attrs["axis_dir"] = axis_dir
        for j, t in enumerate(tlist):
            file[f"t{j}"] = t
            for i, qn in enumerate(t_qnums[j]):
                file.attrs[f"t{j}_qnums{i}"] = qn
        file["r"] = r


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
            a = ptn.crandn(dims, rng)
            # enforce sparsity pattern based on quantum numbers
            ptn.enforce_qsparsity(a, [axis_dir[i] * qnums[i] for i in range(a.ndim)])

            file[f"a{c}"] = a
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
            a = ptn.crandn(dims, rng)
            # enforce sparsity pattern based on quantum numbers
            ptn.enforce_qsparsity(a, [axis_dir[i] * qnums[i] for i in range(a.ndim)])

            file[f"a{c}"] = a
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
            a = ptn.crandn(dims, rng)
            # enforce sparsity pattern based on quantum numbers
            ptn.enforce_qsparsity(a, [axis_dir[i] * qnums[i] for i in range(a.ndim)])

            file[f"a{c}"] = a
            file.attrs[f"axis_dir{c}"] = axis_dir
            for i, qn in enumerate(qnums):
                file.attrs[f"qnums{c}{i}"] = qn


def block_sparse_tensor_augment_identity_blocks_data():

    # random number generator
    rng = np.random.default_rng(935)

    # dimensions
    dims = (37, 21)

    # axis directions
    axis_dir = (1, -1)

    # quantum numbers
    qnums = [rng.choice([-3, -1, 0, 2, 5] if i == 0 else
                        [-6, -3, -2, 0, 1, 2, 4], size=dims[i], replace=True).astype(np.int32)
             for i in range(2)]

    # dense tensor representation
    t = rng.standard_normal(dims)
    # enforce sparsity pattern based on quantum numbers
    ptn.enforce_qsparsity(t, [axis_dir[i] * qnums[i] for i in range(2)])

    t_ext = [None, None]
    for i in range(2):
        t_ext[i] = t.copy()
        qnums_ext = sorted(list(set(qnums[i]).difference(qnums[1 - i])))
        # ensure that some quantum numbers along current axis do not appear along the other axis
        assert any(qnums_ext)
        for qn in qnums_ext:
            idx = np.where(qnums[i] == qn)[0]
            if i == 0:
                extension = np.zeros((t.shape[0], len(idx)), dtype=t.dtype)
                extension[idx, :] = np.identity(len(idx), dtype=t.dtype)
            else:
                extension = np.zeros((len(idx), t.shape[1]), dtype=t.dtype)
                extension[:, idx] = np.identity(len(idx), dtype=t.dtype)
            t_ext[i] = np.concatenate((t_ext[i], extension), axis = 1 - i)

    with h5py.File("data/test_block_sparse_tensor_augment_identity_blocks.hdf5", "w") as file:
        file["t"] = t
        file.attrs["axis_dir"] = axis_dir
        for i, qn in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qn
        for i in range(2):
            file[f"t_ext{i}"] = t_ext[i]


def block_sparse_tensor_serialize_data():

    # random number generator
    rng = np.random.default_rng(273)

    # dimensions
    dims = (11, 3, 5, 8)

    # tensor degree
    ndim = len(dims)

    # axis directions
    axis_dir = rng.choice((1, -1), size=ndim)

    # quantum numbers
    qnums = [rng.integers(-3, 4, size=d).astype(np.int32) for d in dims]

    # dense tensor representation
    t = ptn.crandn(dims, rng).astype(np.complex64)
    # enforce sparsity pattern based on quantum numbers
    ptn.enforce_qsparsity(t, [axis_dir[i] * qnums[i] for i in range(ndim)])

    with h5py.File("data/test_block_sparse_tensor_serialize.hdf5", "w") as file:
        file["t"] = t
        file.attrs["axis_dir"] = axis_dir
        for i, qn in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qn


def block_sparse_tensor_get_entry_data():

    # random number generator
    rng = np.random.default_rng(613)

    # dimensions
    dims = (7, 6, 13, 4)

    # tensor degree
    ndim = len(dims)

    # axis directions
    axis_dir = rng.choice((1, -1), size=ndim)

    # quantum numbers
    qnums = [rng.integers(-3, 4, size=d).astype(np.int32) for d in dims]

    # dense tensor representation
    t = rng.standard_normal(dims).astype(np.float32)
    # enforce sparsity pattern based on quantum numbers
    ptn.enforce_qsparsity(t, [axis_dir[i] * qnums[i] for i in range(ndim)])

    with h5py.File("data/test_block_sparse_tensor_get_entry.hdf5", "w") as file:
        file["t"] = t
        file.attrs["axis_dir"] = axis_dir
        for i, qn in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qn


def main():
    block_sparse_tensor_copy_data()
    block_sparse_tensor_get_block_data()
    block_sparse_tensor_cyclic_partial_trace_data()
    block_sparse_tensor_norm2_data()
    block_sparse_tensor_transpose_data()
    block_sparse_tensor_reshape_data()
    block_sparse_tensor_matricize_axis_data()
    block_sparse_tensor_slice_data()
    block_sparse_tensor_multiply_pointwise_vector_data()
    block_sparse_tensor_multiply_axis_data()
    block_sparse_tensor_dot_data()
    block_sparse_tensor_concatenate_data()
    block_sparse_tensor_block_diag_data()
    block_sparse_tensor_qr_data()
    block_sparse_tensor_rq_data()
    block_sparse_tensor_svd_data()
    block_sparse_tensor_augment_identity_blocks_data()
    block_sparse_tensor_serialize_data()
    block_sparse_tensor_get_entry_data()


if __name__ == "__main__":
    main()
