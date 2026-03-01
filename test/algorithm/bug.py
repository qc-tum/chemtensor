"""
Reference implementation of the Basis-Update and Galerkin (BUG) rank-adaptive
integrator for a tree tensor network specialized to a TTNO Hamiltonian

References:
- Gianluca Ceruti, Christian Lubich, Dominik Sulz
  Rank-adaptive time integration of tree tensor networks
  SIAM J. Numer. Anal. 61, 194-222 (2023)
- Christian Lubich, Bart Vandereycken, Hanna Walach
  Time integration of rank-constrained Tucker tensors
  SIAM J. Numer. Anal. 56, 1273-1290 (2018)
"""

from typing import Sequence
import numpy as np
from scipy.linalg import qr
from block_sparse_util import enforce_qsparsity
from crandn import crandn


def single_mode_product(a, t, i: int):
    """
    Compute the i-mode product between the matrix `a` and tensor `t`.
    """
    t = np.tensordot(a, t, axes=(1, i))
    # original i-th dimension is now 0-th dimension; move back to i-th place
    t = np.transpose(t, list(range(1, i + 1)) + [0] + list(range(i + 1, t.ndim)))
    return t


def multi_mode_product(u_list, c):
    """
    Compute the multi-mode product between the matrices `u_list` and the core tensor `c`.
    """
    assert len(u_list) == c.ndim
    t = c
    for i in range(c.ndim):
        t = single_mode_product(u_list[i], t, i)
    return t


def is_isometry(a, tol: float = 1e-10):
    """
    Whether the matrix `a` is an isometry.
    """
    a = np.asarray(a)
    assert a.ndim == 2
    return np.allclose(a.conj().T @ a, np.identity(a.shape[1]), rtol=tol, atol=tol)


def retained_singular_values(s, tol: float):
    """
    Number of retained singular values based on tolerance `tol`.
    """
    sq_sum = 0
    r1 = len(s)
    for i in reversed(range(len(s))):
        sq_sum += s[i]**2
        if np.sqrt(sq_sum) > tol:
            break
        r1 = i
    return r1


def higher_order_svd(t, tol: float, max_ranks=None):
    """
    Compute the higher-order singular value decomposition (Tucker format approximation) of `t`.
    """
    assert (not max_ranks) or (t.ndim == len(max_ranks))
    u_list = []
    s_list = []
    for i in range(t.ndim):
        a = matricize(t, i)
        u, sigma, _ = np.linalg.svd(a, full_matrices=False)
        chi = retained_singular_values(sigma, tol)
        if max_ranks:
            # truncate in case max_ranks[i] < chi
            chi = min(chi, max_ranks[i])
        u_list.append(u[:, :chi])
        s_list.append(sigma)
    # form the core tensor by applying Ui^\dagger to the i-th dimension
    c = multi_mode_product([u.conj().T for u in u_list], t)
    return u_list, c, s_list


class TreeNode:
    """
    Tree node, containing a physical axis also in inner connecting tensors.

    Convention for axis ordering:
      1. axes connecting to children
      2. physical axis or axes
      3. parent axis
    """
    def __init__(self, i_site: int, conn, children: Sequence):
        self.i_site = i_site
        self.conn = np.asarray(conn)
        assert self.conn.ndim >= 2
        self.children = list(children)

    @property
    def is_leaf(self) -> bool:
        """
        Whether the node is a leaf.
        """
        return not self.children

    def orthonormalize(self):
        """
        Orthonormalize the tree in-place.
        """
        for i, c in enumerate(self.children):
            r = c.orthonormalize()
            self.conn = single_mode_product(r, self.conn, i)
        tmat = self.conn.reshape((-1, self.conn.shape[-1]))
        if is_isometry(tmat):
            return np.identity(tmat.shape[1])
        q, r = np.linalg.qr(tmat, mode="reduced")
        self.conn = q.reshape(self.conn.shape[:-1] + (q.shape[1],))
        return r

    def to_full_tensor(self):
        """
        Convert the tree with root 'self' to a full tensor,
        with leading dimensions the physical dimensions and the last dimension the rank.
        """
        if self.is_leaf:
            return self.conn, (self.i_site,)
        t = self.conn
        phys_dims = []
        sites = ()
        for i, c in enumerate(self.children):
            ct, cs = c.to_full_tensor()
            # record physical dimensions and site indices
            phys_dims = phys_dims + list(ct.shape[:-1])
            sites = sites + cs
            t = single_mode_product(ct.reshape(-1, ct.shape[-1]), t, i)
        # local physical dimension and site index
        phys_dims = phys_dims + list(self.conn.shape[len(self.children):-1])
        sites = sites + (self.i_site,)
        return t.reshape(phys_dims + [t.shape[-1]]), sites


def tree_vdot(chi: TreeNode, psi: TreeNode):
    """
    Compute the logical inner product `<chi | psi>` of two trees with the same topology.
    """
    assert chi.i_site == psi.i_site
    assert len(chi.children) == len(psi.children)
    t = psi.conn
    for i in range(len(psi.children)):
        r = tree_vdot(chi.children[i], psi.children[i])
        t = single_mode_product(r, t, i)
    t = chi.conn.reshape((-1, chi.conn.shape[-1])).conj().T @ t.reshape((-1, t.shape[-1]))
    return t


def apply_local_operator(op, psi):
    """
    Apply a local operator represented as a connecting tensor
    by contracting its physical input axis with the physical state axis
    and taking Kronecker products of virtual bond dimensions.
    """
    # operator has a physical input and output axis
    assert op.ndim == psi.ndim + 1
    nc = psi.ndim - 2  # without physical axis and parent bond
    # contract physical axes of 'op' and 'psi' and take the Kronecker product of virtual bonds
    idx_op  = tuple(range(0, 2*nc, 2)) + (2*nc, 2*nc + 1, 2*nc + 2)
    idx_psi = tuple(range(1, 2*nc, 2)) + (2*nc + 1,       2*nc + 3)
    idx_t   = tuple(range(2*nc))       + (2*nc, 2*nc + 2, 2*nc + 3)
    t = np.einsum(op, idx_op, psi, idx_psi, idx_t, optimize=True)
    assert t.ndim == 2*nc + 3
    # flatten respective virtual bonds
    t = t.reshape(tuple(t.shape[2*i] * t.shape[2*i+1] for i in range(nc))
                  + (t.shape[2*nc], t.shape[2*nc+1]*t.shape[2*nc+2]))
    return t


def local_operator_averages(conn_chi, conn_op, conn_psi, avg_children: list):
    """
    Evaluate the local operator average `<chi | op | psi>`
    given the averages of the connected child nodes.
    """
    t = apply_local_operator(conn_op, conn_psi)
    for i in range(len(avg_children)):
        ac = avg_children[i]
        t = single_mode_product(ac.reshape((ac.shape[0], -1)), t, i)
    t = conn_chi.reshape((-1, conn_chi.shape[-1])).conj().T @ t.reshape((-1, t.shape[-1]))
    t = t.reshape((conn_chi.shape[-1], conn_op.shape[-1], conn_psi.shape[-1]))
    return t


def tree_operator_averages(chi: TreeNode, op: TreeNode, psi: TreeNode):
    """
    Compute the operator averages `<chi | op | psi>` on all subtrees,
    representing the result in a tree of the same topology.
    """
    assert chi.i_site == op.i_site == psi.i_site
    if op.is_leaf:
        assert chi.is_leaf
        assert psi.is_leaf
        assert  op.conn.ndim == 3
        assert chi.conn.ndim == 2
        assert psi.conn.ndim == 2
        return TreeNode(op.i_site, np.einsum(chi.conn.conj(), (3, 0), op.conn, (3, 4, 1), psi.conn, (4, 2), (0, 1, 2)), [])
    # contract physical dimensions and interleave remaining dimensions
    assert len(chi.children) == len(op.children) and len(psi.children) == len(op.children)
    nc = len(chi.children)
    avg_children = []
    for i in range(nc):
        avg_children.append(tree_operator_averages(chi.children[i], op.children[i], psi.children[i]))
    avg_conn = local_operator_averages(chi.conn, op.conn, psi.conn, [ac.conn for ac in avg_children])
    return TreeNode(op.i_site, avg_conn, avg_children)


def flatten_local_dimensions(local_dims) -> tuple:
    """
    Flatten a nested tuple of local dimensions.
    """
    if isinstance(local_dims, int):
        return (local_dims,)
    return sum([flatten_local_dimensions(ld) for ld in local_dims], ())


def multiply_local_dimensions(local_dims: tuple) -> int:
    """
    Compute the overall product of local dimensions.
    """
    return np.prod(flatten_local_dimensions(local_dims), dtype=int)


def square_local_dimensions(local_dims):
    """
    Return a new tuple of nested local dimensions with entrywise squared dimensions.
    """
    if isinstance(local_dims, int):
        return local_dims**2
    return tuple(square_local_dimensions(ld) for ld in local_dims)


def tree_from_state(state, local_dims, tol: float, i_site_next: int = 0):
    """
    Construct a tree approximating a given state,
    with `local_dims` a recursively nested tuple of the form
    (dims_subtree_0, ..., dims_subtree_{n-1}, dim_current_node).
    """
    state = np.asarray(state)
    assert state.ndim == 2
    if len(local_dims) == 1:
        assert state.shape[0] == local_dims[0]
        return TreeNode(i_site_next, state, [])
    child_dims = tuple(multiply_local_dimensions(ld) for ld in local_dims[:-1])
    u_list, conn, _ = higher_order_svd(state.reshape(child_dims + (local_dims[-1], state.shape[1])), tol)
    # re-absorb unitaries for physical axis and parent bond
    conn = single_mode_product(u_list[-1], conn, conn.ndim - 1)
    conn = single_mode_product(u_list[-2], conn, conn.ndim - 2)
    # recursive function call on subtrees
    children = []
    for i in range(len(local_dims) - 1):
        child = tree_from_state(u_list[i], local_dims[i], tol, i_site_next)
        children.append(child)
        i_site_next = child.i_site + 1
    return TreeNode(i_site_next, conn, children)


def interleave_local_operator_axes(op, local_dims):
    """
    Interleave the local output and input axes of a linear operator.
    """
    op = np.asarray(op)
    local_dims = flatten_local_dimensions(local_dims)
    assert op.ndim == 2
    assert op.shape == 2 * (np.prod(local_dims, dtype=int),)
    nsites = len(local_dims)
    perm = sum(zip(range(nsites), range(nsites, 2*nsites)), ())
    return op.reshape(local_dims + local_dims).transpose(perm)


def separate_local_operator_axes(op):
    """
    Separate the local output and input axes of a linear operator,
    returning its matrix representation.
    """
    op = np.asarray(op)
    nsites = op.ndim // 2
    assert op.shape[::2] == op.shape[1::2]
    local_dims = op.shape[::2]
    perm = list(range(0, 2*nsites, 2)) + list(range(1, 2*nsites, 2))
    return op.transpose(perm).reshape(2 * (np.prod(local_dims, dtype=int),))


def reshape_nodes_as_operators(node: TreeNode):
    """
    Reshape the tensors of a tree to physical dimensions of the form `(d, d)`.
    """
    d = int(np.sqrt(node.conn.shape[-2]))
    assert node.conn.shape[-2] == d**2
    if node.is_leaf:
        return TreeNode(node.i_site, node.conn.reshape((d, d, node.conn.shape[-1])), [])
    return TreeNode(node.i_site, node.conn.reshape(node.conn.shape[:-2] + (d, d, node.conn.shape[-1])),
                    [reshape_nodes_as_operators(c) for c in node.children])


def tree_from_operator(op, local_dims, tol: float):
    """
    Construct a binary tree approximating a given linear operator,
    with `local_dims` a recursively nested tuple of the form
    (dims_subtree_0, ..., dims_subtree_{n-1}, dim_current_node).
    """
    op_interleaved = interleave_local_operator_axes(op, flatten_local_dimensions(local_dims))
    tree = tree_from_state(op_interleaved.reshape((-1, 1)), square_local_dimensions(local_dims), tol)
    return reshape_nodes_as_operators(tree)


def construct_random_tree(i_sites, i_site_parent: int, local_dims, bond_dims, scaling: float, rng: np.random.Generator):
    """
    Construct a tree with random tensor entries for a given topology and dimensions.
    `i_sites` is a recursively nested tuple of the form
    (i_sites_subtree_0, ..., i_sites_subtree_{n-1}, i_site_current_node)
    """
    i_site = i_sites[-1]
    # recursive function call on subtrees
    children = [construct_random_tree(i_sites_child, i_site, local_dims, bond_dims, scaling, rng)
                for i_sites_child in i_sites[:-1]]
    if i_site_parent < 0:
        # root node
        bond_dim_parent = 1
    else:
        idx_bond = tuple(sorted((i_site, i_site_parent)))
        bond_dim_parent = bond_dims[idx_bond]
    conn = scaling * crandn(tuple(c.conn.shape[-1] for c in children) + (local_dims[i_site], bond_dim_parent), rng)
    return TreeNode(i_site, conn, children)


def tree_enforce_qsparsity(node: TreeNode, i_site_parent, qsites, qbonds):
    """
    Enforce sparsity patterns on the tree nodes based on quantum numbers.
    """
    if i_site_parent < 0:
        # root node
        qbond_parent = [0]
    else:
        if node.i_site < i_site_parent:
            qbond_parent = -qbonds[(node.i_site, i_site_parent)]
        else:
            qbond_parent = qbonds[(i_site_parent, node.i_site)]
    qbonds_children = tuple(
         qbonds[(c.i_site, node.i_site)] if c.i_site < node.i_site else
        -qbonds[(node.i_site, c.i_site)] for c in node.children)
    enforce_qsparsity(node.conn, qbonds_children + (qsites[node.i_site],) + (qbond_parent,))
    # recursive function call on subtrees
    for c in node.children:
        tree_enforce_qsparsity(c, node.i_site, qsites, qbonds)


def rk4(f, y, h: float):
    """
    Runge-Kutta method of order 4.
    """
    k1 = h*f(y)
    k2 = h*f(y + 0.5*k1)
    k3 = h*f(y + 0.5*k2)
    k4 = h*f(y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6


def matricize(a, i: int):
    """
    Compute the matricization of a tensor along the i-th axis.
    """
    s = (int(np.prod(a.shape[:i])), a.shape[i], int(np.prod(a.shape[i+1:])))
    a = a.reshape(s)
    a = a.transpose((1, 0, 2)).reshape((s[1], s[0]*s[2]))
    return a


def tensorize(a, shape, i: int):
    """
    Tensorize a matrix, undoing the matricization along the i-th axis.
    """
    s = (shape[i], int(np.prod(shape[:i])), int(np.prod(shape[i+1:])))
    return a.reshape(s).transpose((1, 0, 2)).reshape(shape)


def orthonormal_basis(a, strict: bool, tol: float = 1e-13):
    """
    Construct an orthonormal basis of the subspace spanned by the column vectors in 'a'.
    """
    q, r, _ = qr(a, mode="economic", pivoting=True)
    if strict:
        idx = np.abs(np.diag(r)) > tol
        return q[:, idx]
    else:
        return q


def bug_flow_update_basis(state: TreeNode, hamiltonian: TreeNode, avg: TreeNode, env_root, s0, prefactor: float, dt: float):
    """
    Update and augment the basis matrix of the i-th subtree
    for a Schrödinger differential equation with Hamiltonian given as TTNO.

    TTNO-adapted Algorithm 5 in "Rank-adaptive time integration of tree tensor networks".
    """
    assert state.i_site == hamiltonian.i_site == avg.i_site
    y = TreeNode(state.i_site, state.conn @ s0, state.children)
    if state.is_leaf:
        # right-hand side of the ordinary differential equation for the basis update
        f = lambda y: prefactor * np.einsum(env_root, (1, 2, 3), hamiltonian.conn, (0, 4, 2), y, (4, 3), (0, 1))
        k1 = rk4(f, y.conn, dt)
        u_hat = orthonormal_basis(np.concatenate((k1, state.conn), axis=1), strict=True)
        m_hat = u_hat.conj().T @ state.conn
        avg_hat = np.einsum(u_hat.conj(), (3, 0), hamiltonian.conn, (3, 4, 1), u_hat, (4, 2), (0, 1, 2))
        return TreeNode(state.i_site, u_hat, []), m_hat, TreeNode(avg.i_site, avg_hat, [])
    else:
        y1, c0_hat, m_hat_children, avg_hat_children = bug_time_step_subtree(y, hamiltonian, avg, env_root, prefactor, dt)
        q_hat = orthonormal_basis(np.concatenate((
            matricize(y1.conn, y1.conn.ndim - 1).T,
            matricize(c0_hat, c0_hat.ndim - 1).T), axis=1), strict=False)
        y1.conn = tensorize(q_hat.T, y1.conn.shape[:-1] + (q_hat.shape[1],), y1.conn.ndim - 1)
        # inner product between new augmented (orthonormal) state and input state
        assert len(y1.children) == len(state.children)
        assert len(y1.children) == len(m_hat_children)
        m_hat = state.conn
        for i in range(len(y1.children)):
            m_hat = single_mode_product(m_hat_children[i], m_hat, i)
        m_hat = y1.conn.reshape((-1, y1.conn.shape[-1])).conj().T @ m_hat.reshape((-1, m_hat.shape[-1]))
        # compute new averages (expectation values)
        assert len(avg_hat_children) == len(hamiltonian.children)
        avg_hat = local_operator_averages(y1.conn, hamiltonian.conn, y1.conn, [ac.conn for ac in avg_hat_children])
        return y1, m_hat, TreeNode(avg.i_site, avg_hat, avg_hat_children)


def bug_flow_update_connecting_tensor(c0_hat, avg_hat_children: Sequence[TreeNode],
                                      hamiltonian: TreeNode, env_root, prefactor: float, dt: float):
    """
    Update the (already augmented) connecting tensor of a tree node
    for a Schrödinger differential equation with Hamiltonian given as TTNO.

    TTNO-adapted Algorithm 6 in "Rank-adaptive time integration of tree tensor networks".
    """
    # cannot be an empty list
    assert avg_hat_children
    assert len(hamiltonian.children) == len(avg_hat_children)
    def f(c):
        t = apply_local_operator(hamiltonian.conn, c)
        for i in range(len(hamiltonian.children)):
            ac = avg_hat_children[i].conn
            t = single_mode_product(ac.reshape((ac.shape[0], -1)), t, i)
        t = single_mode_product(env_root.reshape((env_root.shape[0], -1)), t, t.ndim - 1)
        return prefactor * t
    # perform time evolution
    c1_hat = rk4(f, c0_hat, dt)
    return c1_hat


def bug_compute_child_environment(state: TreeNode, hamiltonian: TreeNode, avg_children: Sequence[TreeNode], env_root, i: int):
    """
    Compute the environment tensor for the i-th child node
    after gauge-transforming the root node into an isometry towards the child.
    """
    assert state.i_site == hamiltonian.i_site
    conn_mat = matricize(state.conn, i).T
    q0 = orthonormal_basis(conn_mat, strict=True)
    s0 = q0.conj().T @ conn_mat
    s0 = s0.T
    q0ten = tensorize(q0.T, state.conn.shape[:i] + (q0.shape[1],) + state.conn.shape[i+1:], i)
    # project onto the orthonormalized tree without the current subtree
    env = apply_local_operator(hamiltonian.conn, q0ten)
    for j in range(len(state.children)):
        if j == i:
            continue
        ac = avg_children[j].conn
        env = single_mode_product(ac.reshape((ac.shape[0], -1)), env, j)
    # upstream axis
    env = single_mode_product(env_root.reshape((env_root.shape[0], -1)), env, env.ndim - 1)
    # isolate the i-th virtual bond and contract all other axes
    env = q0.conj().T @ matricize(env, i).T
    env = env.reshape((q0.shape[1], hamiltonian.conn.shape[i], q0.shape[1]))
    return env, s0


def bug_time_step_subtree(state: TreeNode, hamiltonian: TreeNode, avg_tree: TreeNode, env_root, prefactor: float, dt: float):
    """
    Perform a recursive rank-augmenting TTN integration step on a (sub-)tree
    for a Schrödinger differential equation with Hamiltonian given as TTNO.

    TTNO-adapted Algorithm 4 in "Rank-adaptive time integration of tree tensor networks".
    """
    assert state.i_site == hamiltonian.i_site == avg_tree.i_site
    assert not state.is_leaf
    children_hat_list = []
    m_hat_children = []
    a_hat_children = []
    for i in range(len(state.children)):
        env, s0 = bug_compute_child_environment(state, hamiltonian, avg_tree.children, env_root, i)
        c_hat, m_hat, a_hat = bug_flow_update_basis(state.children[i], hamiltonian.children[i],
                                                    avg_tree.children[i], env, s0, prefactor, dt)
        children_hat_list.append(c_hat)
        m_hat_children.append(m_hat)
        a_hat_children.append(a_hat)
    # augment the initial connecting tensor
    c0_hat = multi_mode_product(m_hat_children + [np.identity(state.conn.shape[-2]), np.identity(state.conn.shape[-1])], state.conn)
    c1_hat = bug_flow_update_connecting_tensor(c0_hat, a_hat_children, hamiltonian, env_root, prefactor, dt)
    return TreeNode(state.i_site, c1_hat, children_hat_list), c0_hat, m_hat_children, a_hat_children


def bug_tree_time_step(state: TreeNode, hamiltonian: TreeNode, prefactor: float, dt: float, rel_tol_trunc: float):
    """
    Perform a rank-augmenting TTN integration step
    for a Schrödinger differential equation with Hamiltonian given as TTNO.
    """
    assert state.i_site == hamiltonian.i_site
    # should be the actual roots of the trees
    assert state.conn.shape[-1] == 1
    assert hamiltonian.conn.shape[-1] == 1
    avg_tree = tree_operator_averages(state, hamiltonian, state)
    state, _, _, _ = bug_time_step_subtree(state, hamiltonian, avg_tree, np.ones((1, 1, 1)), prefactor, dt)
    state = truncate_tree(state, dt * rel_tol_trunc)
    return state


def truncate_tree(node: TreeNode, tol: float, max_rank = None):
    """
    Perform a rank truncation of a tree from root to leaves.
    """
    if node.is_leaf:
        return node
    u_list = []
    children_trunc = []
    for i in range(len(node.children)):
        u, sigma, _ = np.linalg.svd(matricize(node.conn, i), full_matrices=False)
        chi = retained_singular_values(sigma, tol)
        if max_rank:
            # truncate in case max_rank < chi
            chi = min(chi, max_rank)
        u = u[:, :chi]
        u_list.append(u)
        cit = single_mode_product(u.T, node.children[i].conn, node.children[i].conn.ndim - 1)
        # recursion to children
        children_trunc.append(truncate_tree(
            TreeNode(node.children[i].i_site, cit, node.children[i].children), tol, max_rank))
    # form the truncated core tensor
    conn = node.conn
    for i in range(len(node.children)):
        # apply Ui^\dagger to the i-th dimension
        conn = single_mode_product(u_list[i].conj().T, conn, i)
    return TreeNode(node.i_site, conn, children_trunc)


def tree_local_tensors(psi: TreeNode):
    """
    Dictionary of all local tensors in the tree.
    """
    tensors = {psi.i_site: psi.conn}
    for c in psi.children:
        tensors = {**tensors, **tree_local_tensors(c)}
    return tensors
