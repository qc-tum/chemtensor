from enum import IntEnum
import numpy as np
from scipy import sparse
from opgraph import OpGraphNode, OpGraphEdge, OpGraph
from mpo import MPO


def encode_quantum_number_pair(qa: int, qb: int):
    """
    Encode a pair of quantum numbers into a single quantum number.
    """
    return (qa << 16) + qb


def decode_quantum_number_pair(qnum: int):
    """
    Decode a quantum number into two separate quantum numbers.
    """
    qb = qnum % (1 << 16)
    if qb >= (1 << 15):
        qb -= (1 << 16)
    elif qb < -(1 << 15):
        qb += (1 << 16)
    qa = (qnum - qb) >> 16
    return qa, qb


def quadratic_spin_fermionic_mpo(coeffc, coeffa, sigma: int) -> MPO:
    r"""
    Represent a product of sums of fermionic creation and annihilation operators
    of the following form as MPO, where sigma = 1 indicates spin-up and
    sigma = -1 indicates spin-down:

    .. math::

        op = (\sum_{i=1}^L coeffc_i a^{\dagger}_{i,\sigma}) (\sum_{j=1}^L coeffa_j a_{j,\sigma})
    """
    assert len(coeffc) == len(coeffa)
    nsites = len(coeffc)
    assert nsites >= 1

    if sigma not in (1, -1):
        raise ValueError("'sigma' argument must be 1 (spin-up) or -1 (spin-down)")

    # physical particle number and spin quantum numbers (encoded as single integer)
    qn = [0,  1,  1,  2]
    qs = [0, -1,  1,  0]
    qsite = [encode_quantum_number_pair(q[0], q[1]) for q in zip(qn, qs)]

    id2 = np.identity(2)
    # creation and annihilation operators
    a_ann = np.array([[0., 1.], [0., 0.]])
    a_dag = np.array([[0., 0.], [1., 0.]])
    # number operator
    numop = np.array([[0., 0.], [0., 1.]])
    # Pauli-Z matrix required for Jordan-Wigner transformation
    z = np.array([[1., 0.], [0., -1.]])
    # operator map
    class OID(IntEnum):
        Id =  0
        IC =  1
        IA =  2
        IN =  3
        ZC =  4
        ZA =  5
        CI =  6
        AI =  7
        NI =  8
        CZ =  9
        AZ = 10
        ZZ = 11
    opmap = {
        OID.Id: np.identity(4),
        OID.IC: np.kron(id2, a_dag),
        OID.IA: np.kron(id2, a_ann),
        OID.IN: np.kron(id2, numop),
        OID.ZC: np.kron(z,   a_dag),
        OID.ZA: np.kron(z,   a_ann),
        OID.CI: np.kron(a_dag, id2),
        OID.AI: np.kron(a_ann, id2),
        OID.NI: np.kron(numop, id2),
        OID.CZ: np.kron(a_dag, z  ),
        OID.AZ: np.kron(a_ann, z  ),
        OID.ZZ: np.kron(z,     z  ),
    }

    # construct operator graph
    nid_next = 0
    # identity chains from the left and right
    identity_l = {}
    identity_r = {}
    for i in range(nsites):
        identity_l[i] = OpGraphNode(nid_next, [], [], 0)
        nid_next += 1
    for i in range(1, nsites + 1):
        identity_r[i] = OpGraphNode(nid_next, [], [], 0)
        nid_next += 1
    # nodes connecting creation and annihilation operators
    ca_nodes = {}
    ac_nodes = {}
    for i in range(1, nsites):
        qnum = encode_quantum_number_pair(1, sigma)
        ca_nodes[i] = OpGraphNode(nid_next, [], [], qnum)
        nid_next += 1
    for i in range(1, nsites):
        qnum = encode_quantum_number_pair(-1, -sigma)
        ac_nodes[i] = OpGraphNode(nid_next, [], [], qnum)
        nid_next += 1
    # initialize graph with nodes
    graph = OpGraph(list(identity_l.values()) +
                    list(identity_r.values()) +
                    list(ca_nodes.values()) +
                    list(ac_nodes.values()),
                    [], [identity_l[0].nid, identity_r[nsites].nid])
    # edges
    eid_next = 0
    # identities
    for i in range(nsites - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, identity_l[i + 1].nid], [(OID.Id, 1.)]))
        eid_next += 1
    for i in range(1, nsites):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_r[i].nid, identity_r[i + 1].nid], [(OID.Id, 1.)]))
        eid_next += 1
    # Z strings
    for i in range(1, nsites - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [ca_nodes[i].nid, ca_nodes[i + 1].nid], [(OID.ZZ, 1.)]))
        eid_next += 1
    for i in range(1, nsites - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [ac_nodes[i].nid, ac_nodes[i + 1].nid], [(OID.ZZ, 1.)]))
        eid_next += 1
    # number operators
    for i in range(nsites):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, identity_r[i + 1].nid],
                        [(OID.NI if sigma == 1 else OID.IN, coeffc[i]*coeffa[i])]))
        eid_next += 1
    # creation and annihilation operators
    for i in range(nsites - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, ca_nodes[i + 1].nid],
                        [(OID.CZ if sigma == 1 else OID.IC, coeffc[i])]))
        eid_next += 1
    for i in range(1, nsites):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [ca_nodes[i].nid, identity_r[i + 1].nid],
                        [(OID.AI if sigma == 1 else OID.ZA, coeffa[i])]))
        eid_next += 1
    for i in range(nsites - 1):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [identity_l[i].nid, ac_nodes[i + 1].nid],
                        [(OID.AZ if sigma == 1 else OID.IA, coeffa[i])]))
        eid_next += 1
    for i in range(1, nsites):
        graph.add_connect_edge(
            OpGraphEdge(eid_next, [ac_nodes[i].nid, identity_r[i + 1].nid],
                        [(OID.CI if sigma == 1 else OID.ZC, coeffc[i])]))
        eid_next += 1
    assert graph.is_consistent()

    # convert to MPO
    return MPO.from_opgraph(qsite, graph, opmap)


def construct_ising_1d_hamiltonian(nsites: int, J: float, h: float, g: float):
    """
    Construct the Ising Hamiltonian `sum J Z Z + h Z + g X`
    on a one-dimensional lattice as a sparse matrix.
    """
    # Pauli-X and Z matrices
    sigma_x = sparse.csr_matrix([[0., 1.], [1.,  0.]])
    sigma_z = sparse.csr_matrix([[1., 0.], [0., -1.]])
    # interaction terms and external field
    hint = sparse.kron(sigma_z, sigma_z)
    hamiltonian = \
        sum(J * sparse.kron(sparse.identity(2**j),
                sparse.kron(hint,
                            sparse.identity(2**(nsites-j-2))))
            for j in range(nsites - 1)) \
      + sum(sparse.kron(sparse.identity(2**j),
            sparse.kron(h*sigma_z + g*sigma_x,
                        sparse.identity(2**(nsites-j-1))))
            for j in range(nsites))
    hamiltonian.eliminate_zeros()
    return hamiltonian


def construct_heisenberg_xxz_1d_hamiltonian(nsites: int, J: float, D: float, h: float):
    """
    Construct the XXZ Heisenberg Hamiltonian `sum J (X X + Y Y + D Z Z) - h Z`
    on a one-dimensional lattice as a sparse matrix.
    """
    # spin operators
    sup = np.array([[0.,  1.], [0.,  0. ]])
    sdn = np.array([[0.,  0.], [1.,  0. ]])
    sz  = np.array([[0.5, 0.], [0., -0.5]])
    # interaction terms and external field
    hint = J * (0.5 * (sparse.kron(sup, sdn) + sparse.kron(sdn, sup)) + D * sparse.kron(sz, sz))
    hamiltonian = \
        sum(sparse.kron(sparse.identity(2**j),
            sparse.kron(hint,
                        sparse.identity(2**(nsites-j-2))))
            for j in range(nsites - 1)) \
      + sum(sparse.kron(sparse.identity(2**j),
            sparse.kron(-h * sz,
                        sparse.identity(2**(nsites-j-1))))
            for j in range(nsites))
    hamiltonian.eliminate_zeros()
    return hamiltonian


def construct_heisenberg_xxx_1d_hamiltonian(nsites: int, J: float):
    """
    Construct the XXX Heisenberg Hamiltonian `sum J (X X + Y Y + Z Z)`
    on a one-dimensional lattice as a sparse matrix.
    """
    # spin operators
    sup = np.array([[0.,  1.], [0.,  0. ]])
    sdn = np.array([[0.,  0.], [1.,  0. ]])
    sz  = np.array([[0.5, 0.], [0., -0.5]])
    # local interaction term
    hint = J * (0.5 * (sparse.kron(sup, sdn) + sparse.kron(sdn, sup)) + sparse.kron(sz, sz))
    hamiltonian = \
        sum(sparse.kron(sparse.identity(2**j),
            sparse.kron(hint,
                        sparse.identity(2**(nsites-j-2)))) for j in range(nsites - 1))
    hamiltonian.eliminate_zeros()
    return hamiltonian


def construct_bose_hubbard_1d_hamiltonian(nsites: int, d: int, t: float, u: float, mu: float):
    """
    Construct the Bose-Hubbard Hamiltonian
    with nearest-neighbor hopping on a one-dimensional lattice as a sparse matrix.
    """
    # bosonic creation and annihilation operators
    b_dag = np.diag(np.sqrt(np.arange(1, d, dtype=float)), -1)
    b_ann = np.diag(np.sqrt(np.arange(1, d, dtype=float)),  1)
    # number operator
    numop = np.diag(np.arange(d, dtype=float))
    # kinetic hopping terms, interaction terms and external field
    tkin = -t * (sparse.kron(b_dag, b_ann) + sparse.kron(b_ann, b_dag))
    hint = 0.5 * u * (numop @ (numop - np.identity(d))) - mu * numop
    hamiltonian = \
        sum(sparse.kron(sparse.identity(d**j),
            sparse.kron(tkin,
                        sparse.identity(d**(nsites-j-2))))
            for j in range(nsites - 1)) \
      + sum(sparse.kron(sparse.identity(d**j),
            sparse.kron(hint,
                        sparse.identity(d**(nsites-j-1))))
            for j in range(nsites))
    hamiltonian.eliminate_zeros()
    return hamiltonian


def construct_fermi_operators(nmodes: int):
    """
    Generate sparse matrix representations of the fermionic creation and
    annihilation operators for `nmodes` modes (or sites),
    based on Jordan-Wigner transformation.
    """
    id2 = sparse.identity(2)
    z = sparse.csr_matrix([[ 1.,  0.], [ 0., -1.]])
    u = sparse.csr_matrix([[ 0.,  0.], [ 1.,  0.]])
    clist = []
    for i in range(nmodes):
        c = sparse.identity(1)
        for j in range(nmodes):
            if j < i:
                c = sparse.kron(c, z)
            elif j == i:
                c = sparse.kron(c, u)
            else:
                c = sparse.kron(c, id2)
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
    with nearest-neighbor hopping on a one-dimensional lattice as a sparse matrix.
    """
    clist, alist, nlist = construct_fermi_operators(2*nsites)
    # kinetic hopping terms and
    # interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn)
    hamiltonian = sum(-t * (clist[j] @ alist[j+2] + clist[j+2] @ alist[j])
                      for j in range(2*nsites - 2)) \
                + sum((u * (nlist[j]   - 0.5*sparse.identity(4**nsites)) \
                         @ (nlist[j+1] - 0.5*sparse.identity(4**nsites)) \
                       - mu * (nlist[j] + nlist[j+1]))
                      for j in range(0, 2*nsites, 2))
    hamiltonian.eliminate_zeros()
    return hamiltonian


def construct_molecular_hamiltonian(tkin, vint):
    """
    Construct the molecular Hamiltonian as a sparse matrix.
    """
    nmodes = tkin.shape[0]
    assert tkin.shape == (nmodes, nmodes)
    assert vint.shape == (nmodes, nmodes, nmodes, nmodes)

    clist, alist, _ = construct_fermi_operators(nmodes)

    # kinetic hopping and interaction terms
    hamiltonian = \
        sum(tkin[i, j] * (clist[i] @ alist[j])
            for i in range(nmodes)
            for j in range(nmodes)) + \
        sum(0.5 * vint[i, j, k, l] * (clist[i] @ clist[j] @ alist[l] @ alist[k])
            for i in range(nmodes)
            for j in range(nmodes)
            for k in range(nmodes)
            for l in range(nmodes))
    hamiltonian.eliminate_zeros()
    return hamiltonian


def construct_spin_molecular_hamiltonian(tkin, vint):
    """
    Construct a molecular Hamiltonian for a spin orbital basis as a sparse matrix.
    """
    tkin = np.asarray(tkin)
    vint = np.asarray(vint)

    nsites = tkin.shape[0]
    assert tkin.shape == 2 * (nsites,)
    assert vint.shape == 4 * (nsites,)

    # enlarge the single- and two-particle electron overlap integral tensors
    # from an orbital basis without spin to a spin orbital basis

    # single-particle integrals
    tkin_spin = np.kron(tkin, np.identity(2))

    # two-particle integrals
    tmp = np.zeros((2*nsites, nsites, 2*nsites, nsites), dtype=vint.dtype)
    for i in range(nsites):
        for j in range(nsites):
            tmp[:, i, :, j] = np.kron(vint[:, i, :, j], np.identity(2))
    vint_spin = np.zeros((2*nsites, 2*nsites, 2*nsites, 2*nsites), dtype=vint.dtype)
    for i in range(2*nsites):
        for j in range(2*nsites):
            vint_spin[i, :, j, :] = np.kron(tmp[i, :, j, :], np.identity(2))

    return construct_molecular_hamiltonian(tkin_spin, vint_spin)
