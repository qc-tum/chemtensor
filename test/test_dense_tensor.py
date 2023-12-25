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


def dense_tensor_transpose_data():

    # random number generator
    rng = np.random.default_rng(782)

    t = rng.standard_normal((4, 5, 6, 7)).astype(np.float32)
    t_tp = np.transpose(t, (1, 3, 2, 0))

    with h5py.File("data/test_dense_tensor_transpose.hdf5", "w") as file:
        file["t"]    = t
        file["t_tp"] = t_tp


def dense_tensor_dot_data():

    # random number generator
    rng = np.random.default_rng(524)

    t = crandn((2, 3, 4, 5), rng)
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

    t = crandn((2, 3, 4, 5), rng).astype(np.complex64)
    s = crandn((4, 5, 7, 6), rng).astype(np.complex64)

    t_dot_s_0 = crandn((2, 3, 7, 6), rng).astype(np.complex64)
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
    dense_tensor_transpose_data()
    dense_tensor_dot_data()
    dense_tensor_dot_update_data()
    dense_tensor_kronecker_product_data()
    dense_tensor_kronecker_product_degree_zero_data()
    dense_tensor_block_data()


if __name__ == "__main__":
    main()
