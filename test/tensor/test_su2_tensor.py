import numpy as np
from scipy.linalg import expm, block_diag
import h5py


class SU2VectorSpaceDesc:
    """
    Description of a vector space decomposed into irreducible SU(2) subspaces,
    storing a list of 'j' quantum numbers and corresponding degeneracies 'd_j'.
    'j' quantum numbers are represented times 2.
    """
    def __init__(self, jlist, dlist):
        if len(jlist) != len(dlist):
            raise ValueError("quantum number and degeneracy lists must have same length")
        if not all(isinstance(j, int) for j in jlist):
            raise ValueError("'j' quantum numbers (represented times 2) must be integers")
        self.jlist = tuple(jlist)
        self.dlist = tuple(dlist)

    def degeneracy(self, j):
        """
        Get the degeneracy of the irreducible subspace corresponding to quantum number 'j'.
        """
        i = self.jlist.index(j)
        return self.dlist[i]

    @property
    def logical_dim(self):
        """
        Logical dimension of the vector space.
        """
        return sum(d*(j + 1) for j, d in zip(self.jlist, self.dlist))


def rotation_generators(s: int):
    """
    Construct the generators (Jx, Jy, Jz) of the rotation operators for spin 's' (represented times 2).
    """
    # implementation based on https://en.wikipedia.org/wiki/Spin_(physics)
    d = [0.5 * np.sqrt(i*(s + 1 - i)) for i in range(1, s + 1)]
    jx =     np.diag(d, k=-1) + np.diag(d, k=1)
    jy = 1j*(np.diag(d, k=-1) - np.diag(d, k=1))
    jz = np.diag([s/2 - i for i in range(s + 1)])
    return jx, jy, jz


def rotation_operator(s: int, v):
    """
    Rotation operator for spin quantum number 's' and rotation axis 'v'.
    """
    j = rotation_generators(s)
    return expm(-1j*sum(v[i]*j[i] for i in range(3)))


def rotation_generators_vector_space(spacedesc: SU2VectorSpaceDesc):
    """
    Construct the generators (Jx, Jy, Jz) of the rotation operators on the specified vector space.
    """
    jxv = np.zeros(shape=(0, 0))
    jyv = np.zeros(shape=(0, 0))
    jzv = np.zeros(shape=(0, 0))
    for j, d in zip(spacedesc.jlist, spacedesc.dlist):
        jx, jy, jz = rotation_generators(j)
        jxv = block_diag(jxv, np.kron(np.identity(d), jx))
        jyv = block_diag(jyv, np.kron(np.identity(d), jy))
        jzv = block_diag(jzv, np.kron(np.identity(d), jz))
    return jxv, jyv, jzv


def rotation_operator_vector_space(spacedesc: SU2VectorSpaceDesc, v):
    """
    Rotation operator on the specified vector space and rotation axis 'v'.
    """
    J = rotation_generators_vector_space(spacedesc)
    return expm(-1j*sum(v[i]*J[i] for i in range(3)))


def su2_to_dense_tensor_data():

    # random number generator
    rng = np.random.default_rng(592)

    # vector space descriptions
    spacedesc = (
        SU2VectorSpaceDesc((0, 2, 4), (6, 3, 5)),
        SU2VectorSpaceDesc((3, 5),    (7, 4)),
        SU2VectorSpaceDesc((1, 5),    (8, 1)),
        SU2VectorSpaceDesc((2, 4),    (9, 2)))

    # random rotation angles
    r = rng.standard_normal(3)

    # rotation operators
    wlist = tuple(rotation_operator_vector_space(sd, r) for sd in spacedesc)
    # must be unitary
    assert all(np.allclose(w.conj().T @ w, np.identity(sd.logical_dim)) for w, sd in zip(wlist, spacedesc))

    with h5py.File("data/test_su2_to_dense_tensor.hdf5", "w") as file:
        # store rotation operators
        for i, w in enumerate(wlist):
            file[f"w{i}"] = w


def main():
    su2_to_dense_tensor_data()


if __name__ == "__main__":
    main()
