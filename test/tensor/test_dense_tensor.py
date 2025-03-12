import numpy as np
import h5py
import pytenet as ptn


def dense_tensor_trace_data():

    # random number generator
    rng = np.random.default_rng(95)

    t = ptn.crandn((5, 5, 5), rng)

    # sum of diagonal entries
    tr = 0
    for i in range(t.shape[0]):
        tr += t[i, i, i]

    with h5py.File("data/test_dense_tensor_trace.hdf5", "w") as file:
        file["t"]  = t
        file["tr"] = tr


def dense_tensor_cyclic_partial_trace_data():

    # random number generator
    rng = np.random.default_rng(316)

    ndim_trace = 2

    t = ptn.crandn((5, 2, 3, 4, 1, 5, 2), rng).astype(np.complex64)

    # compute cyclic trace
    t_tr = np.trace(np.trace(t, axis1=0, axis2=t.ndim-ndim_trace), axis1=0, axis2=t.ndim-ndim_trace-1)

    with h5py.File("data/test_dense_tensor_cyclic_partial_trace.hdf5", "w") as file:
        file["t"] = t
        file["t_tr"] = t_tr


def dense_tensor_transpose_data():

    # random number generator
    rng = np.random.default_rng(782)

    dim  = (1, 4, 5, 1, 1, 2, 1, 3, 1, 7)
    #       0  1  2  3  4  5  6  7  8  9
    perm = (4, 8, 2, 6, 0, 9, 5, 7, 3, 1)
    t = rng.standard_normal(dim).astype(np.float32)
    t_tp = np.transpose(t, perm)

    with h5py.File("data/test_dense_tensor_transpose.hdf5", "w") as file:
        file["t"]    = t
        file["t_tp"] = t_tp


def dense_tensor_slice_data():

    # random number generator
    rng = np.random.default_rng(143)

    t = ptn.crandn((2, 7, 3, 5, 4), rng).astype(np.complex64)

    # slice along axis 1
    ind = rng.integers(0, t.shape[1], 10)
    s = t[:, ind, :, :, :]

    with h5py.File("data/test_dense_tensor_slice.hdf5", "w") as file:
        file["t"] = t
        file["s"] = s
        file.attrs["ind"] = ind


def dense_tensor_pad_zeros_data():

    # random number generator
    rng = np.random.default_rng(592)

    t = rng.standard_normal((2, 5, 1, 4)).astype(np.float32)

    t_pad = np.pad(t, [(0, 1), (3, 0), (2, 7), (0, 0)])

    with h5py.File("data/test_dense_tensor_pad_zeros.hdf5", "w") as file:
        file["t"]     = t
        file["t_pad"] = t_pad


def dense_tensor_multiply_pointwise_data():

    # random number generator
    rng = np.random.default_rng(631)

    t = ptn.crandn((2, 6, 5), rng).astype(np.complex64)
    s = [rng.standard_normal(dim).astype(np.float32) for dim in [(2, 6), (6, 5)]]

    t_mult_s = [t * s[0][:, :, None], t * s[1]]

    with h5py.File("data/test_dense_tensor_multiply_pointwise.hdf5", "w") as file:
        file["t"] = t
        for i in range(2):
            file[f"s{i}"] = s[i]
            file[f"t_mult_s{i}"] = t_mult_s[i]


def dense_tensor_multiply_axis_data():

    # random number generator
    rng = np.random.default_rng(193)

    s = ptn.crandn((3, 8, 5, 7), rng).astype(np.complex64)

    t0 = ptn.crandn((6, 4, 5), rng).astype(np.complex64)
    r0 = np.einsum(s, (0, 1, 5, 4), t0, (2, 3, 5), (0, 1, 2, 3, 4))

    t1 = ptn.crandn((5, 2, 6), rng).astype(np.complex64)
    r1 = np.einsum(s, (0, 1, 5, 4), t1, (5, 2, 3), (0, 1, 2, 3, 4))

    with h5py.File("data/test_dense_tensor_multiply_axis.hdf5", "w") as file:
        file["s"] = s
        file["t0"] = t0
        file["r0"] = r0
        file["t1"] = t1
        file["r1"] = r1


def dense_tensor_dot_data():

    # random number generator
    rng = np.random.default_rng(524)

    t = ptn.crandn((2, 11, 3, 4, 5), rng)
    s = ptn.crandn((4, 5, 7, 6), rng)

    t_dot_s = np.tensordot(t, s, 2)

    with h5py.File("data/test_dense_tensor_dot.hdf5", "w") as file:
        file["t"] = t
        file["s"] = s
        file["t_dot_s"] = t_dot_s


def dense_tensor_dot_update_data():

    # random number generator
    rng = np.random.default_rng(170)

    alpha = np.array( 1.2 - 0.3j).astype(np.complex64)
    beta  = np.array(-0.7 + 0.8j).astype(np.complex64)

    t = ptn.crandn((2, 11, 3, 4, 5), rng).astype(np.complex64)
    s = ptn.crandn((4, 5, 7, 6), rng).astype(np.complex64)

    t_dot_s_0 = ptn.crandn((2, 11, 3, 7, 6), rng).astype(np.complex64)
    t_dot_s_1 = alpha * np.tensordot(t, s, 2) + beta * t_dot_s_0

    with h5py.File("data/test_dense_tensor_dot_update.hdf5", "w") as file:
        file["t"] = t
        file["s"] = s
        file["t_dot_s_0"] = t_dot_s_0
        file["t_dot_s_1"] = t_dot_s_1


def dense_tensor_kronecker_product_data():

    # random number generator
    rng = np.random.default_rng(172)

    s = ptn.crandn((6,  5, 7, 2), rng)
    t = ptn.crandn((3, 11, 2, 5), rng)

    r = np.kron(s, t)

    with h5py.File("data/test_dense_tensor_kronecker_product.hdf5", "w") as file:
        file["s"] = s
        file["t"] = t
        file["r"] = r


def dense_tensor_kronecker_product_degree_zero_data():

    # random number generator
    rng = np.random.default_rng(743)

    s = ptn.crandn((), rng).astype(np.complex64)
    t = ptn.crandn((), rng).astype(np.complex64)

    r = np.kron(s, t)

    with h5py.File("data/test_dense_tensor_kronecker_product_degree_zero.hdf5", "w") as file:
        file["s"] = s
        file["t"] = t
        file["r"] = r


def dense_tensor_concatenate_data():

    # random number generator
    rng = np.random.default_rng(201)

    tlist = [
        rng.standard_normal((5, 8, 7, 3)).astype(np.float32),
        rng.standard_normal((5, 8, 9, 3)).astype(np.float32),
        rng.standard_normal((5, 8, 2, 3)).astype(np.float32)]

    r = np.concatenate(tlist, axis=2)

    with h5py.File("data/test_dense_tensor_concatenate.hdf5", "w") as file:
        file["t0"] = tlist[0]
        file["t1"] = tlist[1]
        file["t2"] = tlist[2]
        file["r"] = r


def dense_tensor_block_diag_data():

    # random number generator
    rng = np.random.default_rng(874)

    tlist = [
        rng.standard_normal((5, 8, 7, 3, 4)),
        rng.standard_normal((8, 2, 7, 4, 4)),
        rng.standard_normal((6, 1, 7, 9, 4))]

    r = np.zeros((19, 11, 7, 16, 4))
    r[  :5,    :8 , :,  :3, :] = tlist[0]
    r[ 5:13,  8:10, :, 3:7, :] = tlist[1]
    r[13:,   10:  , :, 7:,  :] = tlist[2]

    with h5py.File("data/test_dense_tensor_block_diag.hdf5", "w") as file:
        file["t0"] = tlist[0]
        file["t1"] = tlist[1]
        file["t2"] = tlist[2]
        file["r"] = r


def dense_tensor_qr_data():

    # random number generator
    rng = np.random.default_rng(317)

    sizes = [(11, 7), (5, 13)]

    with h5py.File("data/test_dense_tensor_qr.hdf5", "w") as file:
        for i, size in enumerate(sizes):
            for j in range(4):
                if j == 0:
                    # single precision real
                    a = rng.standard_normal(size).astype(np.float32)
                elif j == 1:
                    # double precision real
                    a = rng.standard_normal(size)
                elif j == 2:
                    # single precision complex
                    a = ptn.crandn(size, rng).astype(np.complex64)
                else:
                    # double precision complex
                    a = ptn.crandn(size, rng)
                file[f"a_s{i}_t{j}"] = a


def dense_tensor_rq_data():

    # random number generator
    rng = np.random.default_rng(613)

    sizes = [(11, 7), (5, 13)]

    with h5py.File("data/test_dense_tensor_rq.hdf5", "w") as file:
        for i, size in enumerate(sizes):
            for j in range(4):
                if j == 0:
                    # single precision real
                    a = rng.standard_normal(size).astype(np.float32)
                elif j == 1:
                    # double precision real
                    a = rng.standard_normal(size)
                elif j == 2:
                    # single precision complex
                    a = ptn.crandn(size, rng).astype(np.complex64)
                else:
                    # double precision complex
                    a = ptn.crandn(size, rng)
                file[f"a_s{i}_t{j}"] = a


def dense_tensor_eigh_data():

    # random number generator
    rng = np.random.default_rng(342)

    n = 7

    with h5py.File("data/test_dense_tensor_eigh.hdf5", "w") as file:
        for j in range(4):
            if j == 0:
                # single precision real
                a = rng.standard_normal((n, n)).astype(np.float32)
            elif j == 1:
                # double precision real
                a = rng.standard_normal((n, n))
            elif j == 2:
                # single precision complex
                a = 0.5 * ptn.crandn((n, n), rng).astype(np.complex64)
            else:
                # double precision complex
                a = 0.5 * ptn.crandn((n, n), rng)
            # symmetrize
            a = 0.5 * (a + a.conj().T)
            file[f"a_t{j}"] = a


def dense_tensor_svd_data():

    # random number generator
    rng = np.random.default_rng(345)

    sizes = [(11, 7), (5, 13)]

    with h5py.File("data/test_dense_tensor_svd.hdf5", "w") as file:
        for i, size in enumerate(sizes):
            for j in range(4):
                if j == 0:
                    # single precision real
                    a = rng.standard_normal(size).astype(np.float32)
                elif j == 1:
                    # double precision real
                    a = rng.standard_normal(size)
                elif j == 2:
                    # single precision complex
                    a = 0.5 * ptn.crandn(size, rng).astype(np.complex64)
                else:
                    # double precision complex
                    a = 0.5 * ptn.crandn(size, rng)
                file[f"a_s{i}_t{j}"] = a


def dense_tensor_block_data():

    # random number generator
    rng = np.random.default_rng(945)

    t = ptn.crandn((2, 3, 4, 5), rng)

    # generalized sub-block of 't'
    b = t.copy()
    b = b[1:2, :, :, :]
    b = b[:, [0, 2], :, :]
    b = b[:, :, :, [1, 4, 4]]

    with h5py.File("data/test_dense_tensor_block.hdf5", "w") as file:
        file["t"] = t
        file["b"] = b


def main():
    dense_tensor_trace_data()
    dense_tensor_cyclic_partial_trace_data()
    dense_tensor_transpose_data()
    dense_tensor_slice_data()
    dense_tensor_pad_zeros_data()
    dense_tensor_multiply_pointwise_data()
    dense_tensor_multiply_axis_data()
    dense_tensor_dot_data()
    dense_tensor_dot_update_data()
    dense_tensor_kronecker_product_data()
    dense_tensor_kronecker_product_degree_zero_data()
    dense_tensor_concatenate_data()
    dense_tensor_block_diag_data()
    dense_tensor_qr_data()
    dense_tensor_rq_data()
    dense_tensor_eigh_data()
    dense_tensor_svd_data()
    dense_tensor_block_data()


if __name__ == "__main__":
    main()
