import numpy as np
from scipy import sparse
import h5py


def ising_1d_mpo_data():

    # number of lattice sites
    nsites = 7

    # Hamiltonian parameters
    J =  5./11
    h = -2./7
    g = 13./8

    # reference Hamiltonian
    ising_1d_mat = construct_ising_1d_hamiltonian(nsites, J, h, g).todense()

    with h5py.File("data/test_ising_1d_mpo.hdf5", "w") as file:
        file["ising_1d_mat"] = ising_1d_mat


def heisenberg_xxz_1d_mpo_data():

    # number of lattice sites
    nsites = 7

    # Hamiltonian parameters
    J = 14./25
    D = 13./8
    h =  2./7

    # reference Hamiltonian
    heisenberg_xxz_1d_mat = construct_heisenberg_xxz_1d_hamiltonian(nsites, J, D, h).todense()

    with h5py.File("data/test_heisenberg_xxz_1d_mpo.hdf5", "w") as file:
        file["heisenberg_xxz_1d_mat"] = heisenberg_xxz_1d_mat


def bose_hubbard_1d_mpo_data():

    # number of lattice sites
    nsites = 5

    # physical dimension per site (maximal occupancy is d - 1)
    d = 3

    # Hamiltonian parameters
    t  =  7./10
    u  = 17./4
    mu = 13./11

    # reference Hamiltonian
    bose_hubbard_1d_mat = construct_bose_hubbard_1d_hamiltonian(nsites, d, t, u, mu).todense()

    with h5py.File("data/test_bose_hubbard_1d_mpo.hdf5", "w") as file:
        file["bose_hubbard_1d_mat"] = bose_hubbard_1d_mat


def fermi_hubbard_1d_mpo_data():

    # number of lattice sites
    nsites = 4

    # Hamiltonian parameters
    t  = 11./9
    u  = 13./4
    mu =  3./7

    # reference Hamiltonian
    fermi_hubbard_1d_mat = construct_fermi_hubbard_1d_hamiltonian(nsites, t, u, mu).todense()

    with h5py.File("data/test_fermi_hubbard_1d_mpo.hdf5", "w") as file:
        file["fermi_hubbard_1d_mat"] = fermi_hubbard_1d_mat


def molecular_hamiltonian_mpo_data():

    rng = np.random.default_rng(532)

    # number of lattice sites
    nsites = 7

    # Hamiltonian coefficients
    tkin = rng.standard_normal(2 * (nsites,))
    vint = rng.standard_normal(4 * (nsites,))

    # reference Hamiltonian
    molecular_hamiltonian_mat = construct_molecular_hamiltonian(tkin, vint).todense()

    with h5py.File("data/test_molecular_hamiltonian_mpo.hdf5", "w") as file:
        file["tkin"] = tkin
        file["vint"] = vint
        file["molecular_hamiltonian_mat"] = molecular_hamiltonian_mat


def spin_molecular_hamiltonian_mpo_data():

    rng = np.random.default_rng(205)

    # number of spin-endowed lattice sites
    nsites = 4
    # Hamiltonian parameters
    tkin = rng.standard_normal(2 * (nsites,))
    vint = rng.standard_normal(4 * (nsites,))

    # reference Hamiltonian
    molecular_hamiltonian_mat = construct_spin_molecular_hamiltonian(tkin, vint).todense()

    with h5py.File("data/test_spin_molecular_hamiltonian_mpo.hdf5", "w") as file:
        file["tkin"] = tkin
        file["vint"] = vint
        file["molecular_hamiltonian_mat"] = molecular_hamiltonian_mat


def quadratic_fermionic_mpo_data():

    rng = np.random.default_rng(932)

    # number of lattice sites
    nsites = 7

    # coefficients
    coeffc = rng.standard_normal(nsites)
    coeffa = rng.standard_normal(nsites)

    # reference operator
    clist, alist, _ = construct_fermi_operators(nsites)
    quadratic_fermionic_mat = (
          sum(coeffc[i] * clist[i] for i in range(nsites))
        @ sum(coeffa[i] * alist[i] for i in range(nsites)))

    with h5py.File("data/test_quadratic_fermionic_mpo.hdf5", "w") as file:
        file["coeffc"] = coeffc
        file["coeffa"] = coeffa
        file["quadratic_fermionic_mat"] = quadratic_fermionic_mat.todense()


def quadratic_spin_fermionic_mpo_data():

    rng = np.random.default_rng(273)

    # number of spin-endowed lattice sites
    nsites = 3

    # coefficients
    coeffc = rng.standard_normal(nsites)
    coeffa = rng.standard_normal(nsites)

    # reference operator
    clist, alist, _ = construct_fermi_operators(2 * nsites)
    quadratic_spin_fermionic_mat = [
          sum(coeffc[i] * clist[2*i + sigma] for i in range(nsites))
        @ sum(coeffa[i] * alist[2*i + sigma] for i in range(nsites)) for sigma in (0, 1)]

    with h5py.File("data/test_quadratic_spin_fermionic_mpo.hdf5", "w") as file:
        file["coeffc"] = coeffc
        file["coeffa"] = coeffa
        for sigma in (0, 1):
            file[f"quadratic_spin_fermionic_mat_{sigma}"] = quadratic_spin_fermionic_mat[sigma].todense()


def construct_ising_1d_hamiltonian(nsites: int, J: float, h: float, g: float):
    """
    Construct the Ising Hamiltonian `sum J Z Z + h Z + g X`
    on a one-dimensional lattice as sparse matrix.
    """
    # Pauli-X and Z matrices
    sigma_x = sparse.csr_matrix([[0., 1.], [1.,  0.]])
    sigma_z = sparse.csr_matrix([[1., 0.], [0., -1.]])
    H = sparse.csr_matrix((2**nsites, 2**nsites), dtype=float)
    # interaction terms
    hint = sparse.kron(sigma_z, sigma_z)
    for j in range(nsites - 1):
        H += J * sparse.kron(sparse.identity(2**j),
                 sparse.kron(hint,
                             sparse.identity(2**(nsites-j-2))))
    # external field
    for j in range(nsites):
        H += sparse.kron(sparse.identity(2**j),
             sparse.kron(h*sigma_z + g*sigma_x,
                         sparse.identity(2**(nsites-j-1))))
    return H


def construct_heisenberg_xxz_1d_hamiltonian(nsites: int, J: float, D: float, h: float):
    """
    Construct the XXZ Heisenberg Hamiltonian `sum J (X X + Y Y + D Z Z) - h Z`
    on a one-dimensional lattice as sparse matrix.
    """
    # spin operators
    Sup = np.array([[0.,  1.], [0.,  0. ]])
    Sdn = np.array([[0.,  0.], [1.,  0. ]])
    Sz  = np.array([[0.5, 0.], [0., -0.5]])
    H = sparse.csr_matrix((2**nsites, 2**nsites), dtype=float)
    # interaction terms
    hint = J * (0.5 * (sparse.kron(Sup, Sdn) + sparse.kron(Sdn, Sup)) + D * sparse.kron(Sz, Sz))
    for j in range(nsites - 1):
        H += sparse.kron(sparse.identity(2**j),
             sparse.kron(hint,
                         sparse.identity(2**(nsites-j-2))))
    # external field
    for j in range(nsites):
        H -= sparse.kron(sparse.identity(2**j),
             sparse.kron(h*Sz,
                         sparse.identity(2**(nsites-j-1))))
    return H


def construct_bose_hubbard_1d_hamiltonian(nsites: int, d: int, t: float, u: float, mu: float):
    """
    Construct the Bose-Hubbard Hamiltonian
    with nearest-neighbor hopping on a one-dimensional lattice as sparse matrix.
    """
    # bosonic creation and annihilation operators
    b_dag = np.diag(np.sqrt(np.arange(1, d, dtype=float)), -1)
    b_ann = np.diag(np.sqrt(np.arange(1, d, dtype=float)),  1)
    # number operator
    numop = np.diag(np.arange(d, dtype=float))
    H = sparse.csr_matrix((d**nsites, d**nsites), dtype=float)
    # interaction terms
    hint = -t * (sparse.kron(b_dag, b_ann) + sparse.kron(b_ann, b_dag))
    for j in range(nsites - 1):
        H += sparse.kron(sparse.identity(d**j),
             sparse.kron(hint,
                         sparse.identity(d**(nsites-j-2))))
    # external field
    for j in range(nsites):
        H += sparse.kron(sparse.identity(d**j),
             sparse.kron(0.5*u*(numop @ (numop - np.identity(d))) - mu*numop,
                         sparse.identity(d**(nsites-j-1))))
    return H


def construct_fermi_operators(nmodes: int):
    """
    Generate sparse matrix representations of the fermionic creation and
    annihilation operators for `nmodes` modes (or sites),
    based on Jordan-Wigner transformation.
    """
    I = sparse.identity(2)
    Z = sparse.csr_matrix([[ 1.,  0.], [ 0., -1.]])
    U = sparse.csr_matrix([[ 0.,  0.], [ 1.,  0.]])
    clist = []
    for i in range(nmodes):
        c = sparse.identity(1)
        for j in range(nmodes):
            if j < i:
                c = sparse.kron(c, I)
            elif j == i:
                c = sparse.kron(c, U)
            else:
                c = sparse.kron(c, Z)
        c = sparse.csr_matrix(c)
        c.eliminate_zeros()
        clist.append(c)
    # corresponding annihilation operators
    alist = [sparse.csr_matrix(c.conj().T) for c in clist]
    # corresponding number operators
    nlist = []
    for i in range(nmodes):
        f = 1 << (nmodes - i - 1)
        data = [1. if (n & f == f) else 0. for n in range(2**nmodes)]
        nlist.append(sparse.dia_matrix((data, 0), 2*(2**nmodes,)))
    return clist, alist, nlist


def construct_fermi_hubbard_1d_hamiltonian(nsites: int, t: float, u: float, mu: float):
    """
    Construct the Fermi-Hubbard Hamiltonian
    with nearest-neighbor hopping on a one-dimensional lattice as sparse matrix.
    """
    clist, alist, nlist = construct_fermi_operators(2*nsites)
    H = sparse.csr_matrix((4**nsites, 4**nsites), dtype=float)
    # kinetic hopping terms
    for j in range(2*nsites - 2):
        H -= t * (clist[j] @ alist[j+2] + clist[j+2] @ alist[j])
    # interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn)
    for j in range(0, 2*nsites, 2):
        H += (u * (nlist[j] - 0.5*sparse.identity(4**nsites)) @ (nlist[j+1] - 0.5*sparse.identity(4**nsites))
              - mu * (nlist[j] + nlist[j+1]))
    H.eliminate_zeros()
    return H


def construct_molecular_hamiltonian(tkin, vint):
    """
    Construct the molecular Hamiltonian as sparse matrix.
    """
    nmodes = tkin.shape[0]

    complex_hamiltonian = np.iscomplexobj(tkin) or np.iscomplexobj(vint)
    H = sparse.csr_matrix((2**nmodes, 2**nmodes), dtype=(complex if complex_hamiltonian else float))

    clist, alist, _ = construct_fermi_operators(nmodes)

    # kinetic hopping terms
    for i in range(nmodes):
        for j in range(nmodes):
            H += tkin[i, j] * (clist[i] @ alist[j])
    # interaction terms
    for i in range(nmodes):
        for j in range(nmodes):
            for k in range(nmodes):
                for l in range(nmodes):
                    H += 0.5 * vint[i, j, k, l] * (clist[i] @ clist[j] @ alist[l] @ alist[k])
    H.eliminate_zeros()
    return H


def construct_spin_molecular_hamiltonian(tkin, vint):
    """
    Construct a molecular Hamiltonian for a spin orbital basis as sparse matrix.
    """
    tkin = np.asarray(tkin)
    vint = np.asarray(vint)

    nsites = tkin.shape[0]
    assert tkin.shape == 2 * (nsites,)
    assert vint.shape == 4 * (nsites,)

    # enlarge the single- and two-particle electron overlap integral tensors
    # from an orbital basis without spin to a spin orbital basis

    # single-particle integrals
    tkin_spin = np.kron(tkin, np.eye(2))

    # two-particle integrals
    tmp = np.zeros((2*nsites, nsites, 2*nsites, nsites), dtype=vint.dtype)
    for i in range(nsites):
        for j in range(nsites):
            tmp[:, i, :, j] = np.kron(vint[:, i, :, j], np.eye(2))
    vint_spin = np.zeros((2*nsites, 2*nsites, 2*nsites, 2*nsites), dtype=vint.dtype)
    for i in range(2*nsites):
        for j in range(2*nsites):
            vint_spin[i, :, j, :] = np.kron(tmp[i, :, j, :], np.eye(2))

    return construct_molecular_hamiltonian(tkin_spin, vint_spin)


def main():
    ising_1d_mpo_data()
    heisenberg_xxz_1d_mpo_data()
    bose_hubbard_1d_mpo_data()
    fermi_hubbard_1d_mpo_data()
    molecular_hamiltonian_mpo_data()
    spin_molecular_hamiltonian_mpo_data()
    quadratic_fermionic_mpo_data()
    quadratic_spin_fermionic_mpo_data()


if __name__ == "__main__":
    main()
