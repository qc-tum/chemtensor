import numpy as np
import h5py
from util import crandn, interleave_complex


def dense_tensor_trace_data():

    # random number generator
    rng = np.random.default_rng(95)

    t = crandn((5, 5, 5), rng)

    # sum of diagonal entries
    tr = 0
    for i in range(t.shape[0]):
        tr += t[i, i, i]

    with h5py.File("data/test_dense_tensor_trace.hdf5", "w") as file:
        file["t"]  = interleave_complex(t)
        file["tr"] = interleave_complex(tr)


def dense_tensor_cyclic_partial_trace_data():

    # random number generator
    rng = np.random.default_rng(316)

    ndim_trace = 2

    t = crandn((5, 2, 3, 4, 1, 5, 2), rng).astype(np.complex64)

    # compute cyclic trace
    t_tr = np.trace(np.trace(t, axis1=0, axis2=t.ndim-ndim_trace), axis1=0, axis2=t.ndim-ndim_trace-1)

    with h5py.File("data/test_dense_tensor_cyclic_partial_trace.hdf5", "w") as file:
        file["t"] = interleave_complex(t)
        file["t_tr"] = interleave_complex(t_tr)


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

    t = crandn((2, 7, 3, 5, 4), rng).astype(np.complex64)

    # slice along axis 1
    ind = rng.integers(0, t.shape[1], 10)
    s = t[:, ind, :, :, :]

    with h5py.File("data/test_dense_tensor_slice.hdf5", "w") as file:
        file["t"] = interleave_complex(t)
        file["s"] = interleave_complex(s)
        file.attrs["ind"] = ind


def dense_tensor_multiply_pointwise_data():

    # random number generator
    rng = np.random.default_rng(631)

    t = crandn((2, 6, 5), rng).astype(np.complex64)
    s = [rng.standard_normal(dim).astype(np.float32) for dim in [(2, 6), (6, 5)]]

    t_mult_s = [t * s[0][:, :, None], t * s[1]]

    with h5py.File("data/test_dense_tensor_multiply_pointwise.hdf5", "w") as file:
        file["t"] = interleave_complex(t)
        for i in range(2):
            file[f"s{i}"] = s[i]
            file[f"t_mult_s{i}"] = interleave_complex(t_mult_s[i])


def dense_tensor_multiply_axis_data():

    # random number generator
    rng = np.random.default_rng(193)

    s = crandn((3, 8, 5, 7), rng).astype(np.complex64)

    t1 = crandn((6, 5), rng).astype(np.complex64)
    r1 = np.einsum(s, (0, 1, 4, 3), t1, (2, 4), (0, 1, 2, 3))

    t2 = crandn((5, 2), rng).astype(np.complex64)
    r2 = np.einsum(s, (0, 1, 4, 3), t2, (4, 2), (0, 1, 2, 3))

    with h5py.File("data/test_dense_tensor_multiply_axis.hdf5", "w") as file:
        file["s"] = interleave_complex(s)
        file["t1"] = interleave_complex(t1)
        file["r1"] = interleave_complex(r1)
        file["t2"] = interleave_complex(t2)
        file["r2"] = interleave_complex(r2)


def dense_tensor_dot_data():

    # random number generator
    rng = np.random.default_rng(524)

    t = crandn((2, 11, 3, 4, 5), rng)
    s = crandn((4, 5, 7, 6), rng)

    t_dot_s = np.tensordot(t, s, 2)

    with h5py.File("data/test_dense_tensor_dot.hdf5", "w") as file:
        file["t"] = interleave_complex(t)
        file["s"] = interleave_complex(s)
        file["t_dot_s"] = interleave_complex(t_dot_s)


def dense_tensor_dot_update_data():

    # random number generator
    rng = np.random.default_rng(170)

    alpha = np.array( 1.2 - 0.3j).astype(np.complex64)
    beta  = np.array(-0.7 + 0.8j).astype(np.complex64)

    t = crandn((2, 11, 3, 4, 5), rng).astype(np.complex64)
    s = crandn((4, 5, 7, 6), rng).astype(np.complex64)

    t_dot_s_0 = crandn((2, 11, 3, 7, 6), rng).astype(np.complex64)
    t_dot_s_1 = alpha * np.tensordot(t, s, 2) + beta * t_dot_s_0

    with h5py.File("data/test_dense_tensor_dot_update.hdf5", "w") as file:
        file["t"] = interleave_complex(t)
        file["s"] = interleave_complex(s)
        file["t_dot_s_0"] = interleave_complex(t_dot_s_0)
        file["t_dot_s_1"] = interleave_complex(t_dot_s_1)


def dense_tensor_kronecker_product_data():

    # random number generator
    rng = np.random.default_rng(172)

    s = crandn((6,  5, 7, 2), rng)
    t = crandn((3, 11, 2, 5), rng)

    r = np.kron(s, t)

    with h5py.File("data/test_dense_tensor_kronecker_product.hdf5", "w") as file:
        file["s"] = interleave_complex(s)
        file["t"] = interleave_complex(t)
        file["r"] = interleave_complex(r)


def dense_tensor_kronecker_product_degree_zero_data():

    # random number generator
    rng = np.random.default_rng(743)

    s = crandn((), rng).astype(np.complex64)
    t = crandn((), rng).astype(np.complex64)

    r = np.kron(s, t)

    with h5py.File("data/test_dense_tensor_kronecker_product_degree_zero.hdf5", "w") as file:
        file["s"] = interleave_complex(s)
        file["t"] = interleave_complex(t)
        file["r"] = interleave_complex(r)


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
                    a = crandn(size, rng).astype(np.complex64)
                else:
                    # double precision complex
                    a = crandn(size, rng)
                file[f"a_s{i}_t{j}"] = a if j < 2 else interleave_complex(a)


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
                    a = crandn(size, rng).astype(np.complex64)
                else:
                    # double precision complex
                    a = crandn(size, rng)
                file[f"a_s{i}_t{j}"] = a if j < 2 else interleave_complex(a)


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
                    a = 0.5 * crandn(size, rng).astype(np.complex64)
                else:
                    # double precision complex
                    a = 0.5 * crandn(size, rng)
                file[f"a_s{i}_t{j}"] = a if j < 2 else interleave_complex(a)


def dense_tensor_block_data():

    # random number generator
    rng = np.random.default_rng(945)

    t = crandn((2, 3, 4, 5), rng)

    # generalized sub-block of 't'
    b = t.copy()
    b = b[1:2, :, :, :]
    b = b[:, [0, 2], :, :]
    b = b[:, :, :, [1, 4, 4]]

    with h5py.File("data/test_dense_tensor_block.hdf5", "w") as file:
        file["t"] = interleave_complex(t)
        file["b"] = interleave_complex(b)


def main():
    dense_tensor_trace_data()
    dense_tensor_cyclic_partial_trace_data()
    dense_tensor_transpose_data()
    dense_tensor_slice_data()
    dense_tensor_multiply_pointwise_data()
    dense_tensor_multiply_axis_data()
    dense_tensor_dot_data()
    dense_tensor_dot_update_data()
    dense_tensor_kronecker_product_data()
    dense_tensor_kronecker_product_degree_zero_data()
    dense_tensor_qr_data()
    dense_tensor_rq_data()
    dense_tensor_svd_data()
    dense_tensor_block_data()


if __name__ == "__main__":
    main()
