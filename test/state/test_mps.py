import numpy as np
import h5py
import pytenet as ptn


def mps_vdot_data():

    # random number generator
    rng = np.random.default_rng(395)

    # local physical dimension
    d = 3

    # physical quantum numbers
    qd = rng.integers(-1, 2, size=d)

    # virtual bond quantum numbers
    qD_psi = [rng.integers(-1, 2, size=Di) for Di in (1, 13, 17,  8, 1)]
    qD_chi = [rng.integers(-1, 2, size=Di) for Di in (1, 15, 20, 11, 1)]
    # ensure that leading and trailing virtual bond quantum numbers are compatible
    qD_chi[0] = qD_chi[-1] + qD_psi[0] - qD_psi[-1]

    # create random matrix product states
    psi = ptn.MPS(qd, qD_psi, fill="random", rng=rng)
    chi = ptn.MPS(qd, qD_chi, fill="random", rng=rng)
    # rescale tensors such that overall norm is of the order 1
    for i in range(psi.nsites):
        psi.A[i] *= 5
    for i in range(chi.nsites):
        chi.A[i] *= 5

    # calculate dot product <chi | psi>
    s = ptn.vdot(chi, psi)

    with h5py.File("data/test_mps_vdot.hdf5", "w") as file:
        file.attrs["qsite"] = qd
        for i, qbond in enumerate(qD_psi):
            file.attrs[f"qbond_psi_{i}"] = qbond
        for i, qbond in enumerate(qD_chi):
            file.attrs[f"qbond_chi_{i}"] = qbond
        for i, ai in enumerate(psi.A):
            # transposition due to different convention for axis ordering
            file[f"psi_a{i}"] = ai.transpose((1, 0, 2))
        for i, ai in enumerate(chi.A):
            # transposition due to different convention for axis ordering
            file[f"chi_a{i}"] = ai.transpose((1, 0, 2))
        file["s"] = s


def mps_orthonormalize_qr_data():

    # random number generator
    rng = np.random.default_rng(376)

    # local physical dimension
    d = 3

    # physical quantum numbers
    qd = rng.integers(-1, 2, size=d)

    # virtual bond quantum numbers
    qD = [rng.integers(-1, 2, size=Di) for Di in (1, 4, 11, 9, 7, 3, 1)]

    # create a random matrix product state
    mps = ptn.MPS(qd, qD, fill="random", rng=rng)
    # convert tensor entries to single precision
    mps.A = [ai.astype(np.complex64) for ai in mps.A]

    with h5py.File("data/test_mps_orthonormalize_qr.hdf5", "w") as file:
        file.attrs["qsite"] = qd
        for i, qbond in enumerate(qD):
            file.attrs[f"qbond{i}"] = qbond
        for i, ai in enumerate(mps.A):
            # transposition due to different convention for axis ordering
            file[f"a{i}"] = ai.transpose((1, 0, 2))


def mps_compress_data():

    # random number generator
    rng = np.random.default_rng(934)

    # physical quantum numbers
    qd = [-1, 1, 0]

    # virtual bond quantum numbers
    qD = [rng.integers(-1, 2, size=Di) for Di in [1, 23, 75, 102, 83, 30, 1]]

    # create random matrix product state with small entanglement
    mps = ptn.MPS(qd, qD, fill="random", rng=rng)
    for i in range(mps.nsites):
        # imitate small entanglement by multiplying bonds with small scaling factors
        s = np.exp(-30*(rng.uniform(size=mps.bond_dims[i + 1])))
        s /= np.linalg.norm(s)
        mps.A[i] = mps.A[i] * s
        # rescale to achieve norm of order 1
        mps.A[i] *= 5 / np.linalg.norm(mps.A[i])
        # convert tensor entries to single precision
        mps.A[i] = mps.A[i].astype(np.complex64)

    with h5py.File("data/test_mps_compress.hdf5", "w") as file:
        file.attrs["qsite"] = qd
        for i, qbond in enumerate(qD):
            file.attrs[f"qbond{i}"] = qbond
        for i, ai in enumerate(mps.A):
            # transposition due to different convention for axis ordering
            file[f"a{i}"] = ai.transpose((1, 0, 2))


def mps_split_tensor_svd_data():

    # random number generator
    rng = np.random.default_rng(294)

    # physical dimensions
    d = [4, 5]
    # outer virtual bond dimensions
    D = [13, 17]

    a_pair = rng.standard_normal((d[0]*d[1], D[0], D[1])) / np.sqrt(d[0]*d[1]*D[0]*D[1])

    # fictitious quantum numbers
    qd = [rng.integers(-2, 3, size=di) for di in d]
    qD = [rng.integers(-2, 3, size=Di) for Di in D]

    # enforce block sparsity structure dictated by quantum numbers
    mask = ptn.qnumber_outer_sum([ptn.qnumber_flatten(qd), qD[0], -qD[1]])
    a_pair = np.where(mask == 0, a_pair, 0)

    with h5py.File("data/test_mps_split_tensor_svd.hdf5", "w") as file:

        # transposition due to different ordering convention
        file["a_pair"] = a_pair.transpose((1, 0, 2))

        for i in range(2):
            file.attrs[f"qsite{i}"] = qd[i]
        for i in range(2):
            file.attrs[f"qbonds{i}"] = qD[i]

        tol = 0.04
        file.attrs["tol"] = tol

        a0, a1, qbond = ptn.split_mps_tensor(a_pair, qd[0], qd[1], qD, svd_distr="left", tol=tol)

        assert ptn.is_qsparse(a0, [qd[0], qD[0], -qbond])
        assert ptn.is_qsparse(a1, [qd[1], qbond, -qD[1]])

        # merge tensors again, as reference
        a_mrg = ptn.merge_mps_tensor_pair(a0, a1)

        # transposition due to different ordering convention
        file["a_mrg"] = a_mrg.transpose((1, 0, 2))


def mps_to_statevector_data():

    # random number generator
    rng = np.random.default_rng(531)

    # local physical dimension
    d = 3

    # physical quantum numbers
    qd = rng.integers(-1, 2, size=d)

    # virtual bond quantum numbers
    qD = [rng.integers(-1, 2, size=Di) for Di in (1, 7, 10, 11, 5, 1)]

    # create a random matrix product state
    mps = ptn.MPS(qd, qD, fill="random", rng=rng)

    # convert to a state vector
    vec = mps.as_vector()

    with h5py.File("data/test_mps_to_statevector.hdf5", "w") as file:
        file.attrs["qsite"] = qd
        for i, qbond in enumerate(qD):
            file.attrs[f"qbond{i}"] = qbond
        for i, ai in enumerate(mps.A):
            # transposition due to different convention for axis ordering
            file[f"a{i}"] = ai.transpose((1, 0, 2))
        file["vec"] = vec


def main():
    mps_vdot_data()
    mps_orthonormalize_qr_data()
    mps_compress_data()
    mps_split_tensor_svd_data()
    mps_to_statevector_data()


if __name__ == "__main__":
    main()
