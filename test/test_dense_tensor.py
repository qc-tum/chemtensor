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

    t = crandn((4, 5, 6, 7), rng)
    t_tp = np.transpose(t, (1, 3, 2, 0))

    with h5py.File("data/test_dense_tensor_transpose.hdf5", "w") as file:
        file["t"]    = interleave_complex(t)
        file["t_tp"] = interleave_complex(t_tp)


def dense_tensor_dot_data():

    # random number generator
    rng = np.random.default_rng(524)

    t = crandn((2, 3, 4, 5), rng)

    # general dot product
    s = crandn((4, 5, 7, 6), rng)
    t_dot_s = np.tensordot(t, s, 2)

    # matrix-vector multiplication
    p = crandn(5, rng)
    t_dot_p = np.tensordot(t, p, 1)
    q = crandn(2, rng)
    q_dot_t = np.tensordot(q, t, 1)

    # inner product of two vectors
    v = crandn(120, rng)
    t_dot_v = np.dot(t.reshape(-1), v)

    with h5py.File("data/test_dense_tensor_dot.hdf5", "w") as file:
        file["t"] = interleave_complex(t)
        file["s"] = interleave_complex(s)
        file["p"] = interleave_complex(p)
        file["q"] = interleave_complex(q)
        file["v"] = interleave_complex(v)
        file["t_dot_s"] = interleave_complex(t_dot_s)
        file["t_dot_p"] = interleave_complex(t_dot_p)
        file["q_dot_t"] = interleave_complex(q_dot_t)
        file["t_dot_v"] = interleave_complex(t_dot_v)


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
    dense_tensor_kronecker_product_data()
    dense_tensor_block_data()


if __name__ == "__main__":
    main()
