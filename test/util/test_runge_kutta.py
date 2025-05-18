import numpy as np
from scipy.integrate import solve_ivp
import h5py
import pytenet as ptn


def runge_kutta_4_block_sparse_data():

    # random number generator
    rng = np.random.default_rng(492)

    # initial state tensor
    # dimensions
    dims = (14, 13, 17)
    # axis directions
    axis_dir = (1, -1, -1)
    # quantum numbers
    qnums = tuple(rng.integers(-2, 3, size=d).astype(np.int32) for d in dims)
    # dense tensor representation
    y0 = 0.1 * ptn.crandn(dims, rng)
    # enforce sparsity pattern based on quantum numbers
    ptn.enforce_qsparsity(y0, [axis_dir[i] * qnums[i] for i in range(len(dims))])

    # tensor defining linear term of ODE
    lintensor = ptn.crandn(dims[:2] + dims[:2], rng)
    qnums_lin = qnums[:2] + qnums[:2]
    axis_dir_lin = axis_dir[:2] + tuple(-ad for ad in axis_dir[:2])
    ptn.enforce_qsparsity(lintensor, [axis_dir_lin[i] * qnums_lin[i] for i in range(len(axis_dir_lin))])
    # tensors defining nonlinear term of ODE
    nonlintensors = []
    for i in range(y0.ndim):
        i_next = (i + 1) % y0.ndim
        nlt = 0.75 * ptn.crandn((dims[i_next], dims[i]), rng)
        ptn.enforce_qsparsity(nlt, (axis_dir[i_next] * qnums[i_next], -axis_dir[i] * qnums[i]))
        nonlintensors.append(nlt)

    # ODE function
    def f(t: float, y):
        y = y.reshape(y0.shape)
        h = np.einsum(lintensor, (0, 1, 3, 4), y, (3, 4, 2), (0, 1, 2))
        # cyclically permuted axes
        s = np.einsum(nonlintensors[2], (0, 5), nonlintensors[0], (1, 3), nonlintensors[1], (2, 4), y, (3, 4, 5), (0, 1, 2), optimize=True)
        k = np.sin(np.pi * (0.25 + 5*t)) * s**2
        return (h + k).reshape(-1)

    # overall simulation time
    tmax = 0.1

    # reference solution
    sol = solve_ivp(f, [0, tmax], y0.reshape(-1), rtol=1e-12, atol=1e-12)
    y1 = sol.y[:, -1].reshape(y0.shape)

    with h5py.File("data/test_runge_kutta_4_block_sparse.hdf5", "w") as file:
        file["y0"] = y0
        file.attrs["axis_dir"] = axis_dir
        for i, qn in enumerate(qnums):
            file.attrs[f"qnums{i}"] = qn
        file["lintensor"] = lintensor
        for i, t in enumerate(nonlintensors):
            file[f"nonlintensor{i}"] = t
        file["y1"] = y1


def main():
    runge_kutta_4_block_sparse_data()


if __name__ == "__main__":
    main()
