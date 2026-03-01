import numpy as np
import h5py
import sys
sys.path.append("../operator/")
sys.path.append("../tensor/")
sys.path.append("../util/")
from bug import (
    TreeNode,
    bug_flow_update_basis,
    bug_flow_update_connecting_tensor,
    bug_tree_time_step,
    tree_operator_averages,
    tree_from_operator,
    construct_random_tree,
    tree_enforce_qsparsity,
    tree_local_tensors)
from hamiltonian import construct_bose_hubbard_1d_hamiltonian
from block_sparse_util import enforce_qsparsity
from crandn import crandn


def bug_flow_update_basis_leaf_data():

    # random number generator
    rng = np.random.default_rng(493)

    # integration prefactor
    prefactor = 0.2 - 0.7j
    # time step
    dt = 0.13

    # local physical dimension
    d = 11
    # virtual bond dimension of the local state and operator tensor to parent
    dim_bond_state = 5
    dim_bond_op    = 8

    # physical quantum numbers
    qsite = rng.integers(-2, 3, size=d)
    # virtual bond quantum numbers of the local quantum state and tree operator
    qbond_state = rng.integers(-1, 2, size=dim_bond_state)
    qbond_op    = rng.integers(-1, 2, size=dim_bond_op)
    # quantum number sector
    qnum_sector = 1

    # local TTNS tensor
    a_state = crandn((d, dim_bond_state), rng)
    # enforce sparsity pattern based on quantum numbers
    enforce_qsparsity(a_state, (qsite, -(qnum_sector + qbond_state)))
    assert np.linalg.norm(a_state) > 0

    a_op = crandn((d, d, dim_bond_op), rng)
    # enforce sparsity pattern based on quantum numbers
    enforce_qsparsity(a_op, (qsite, -qsite, -qbond_op))
    assert np.linalg.norm(a_op) > 0

    env_parent = crandn((dim_bond_state, dim_bond_op, dim_bond_state), rng)
    # enforce sparsity pattern based on quantum numbers
    enforce_qsparsity(env_parent, (-qbond_state, qbond_op, qbond_state))
    assert np.linalg.norm(env_parent) > 0

    s0 = crandn((dim_bond_state, dim_bond_state), rng)
    # enforce sparsity pattern based on quantum numbers
    enforce_qsparsity(s0, (qbond_state, -qbond_state))
    assert np.linalg.norm(s0) > 0

    y0 = TreeNode(0, a_state, [])
    op = TreeNode(0, a_op, [])

    y1, _, _ = bug_flow_update_basis(y0, op, tree_operator_averages(y0, op, y0), env_parent, s0, prefactor, dt)

    with h5py.File("data/test_bug_flow_update_basis_leaf.hdf5", "w") as file:
        file.attrs["qsite"]       = qsite
        file.attrs["qbond_state"] = qbond_state
        file.attrs["qbond_op"]    = qbond_op
        file.attrs["qnum_sector"] = qnum_sector
        file.attrs["prefactor"]   = prefactor
        file.attrs["dt"]          = dt
        file["a_state_0"]         = a_state
        file["a_state_1"]         = y1.conn
        file["a_op"]              = a_op
        file["s0"]                = s0
        file["env_parent"]        = env_parent.transpose((2, 1, 0))


def bug_flow_update_connecting_tensor_data():

    # random number generator
    rng = np.random.default_rng(319)

    # integration prefactor
    prefactor = -0.4 - 0.3j
    # time step
    dt = 0.27

    # local physical dimension
    d = 7

    # tree topology:
    #
    #     4
    #     │
    #     │
    #     2
    #    ╱│╲
    #   ╱ │ ╲
    #  1  3  0

    # virtual bond dimensions of the state and operator
    dim_bonds_state = {
        (1, 2): 8,
        (2, 3): 5,
        (0, 2): 7,
        (2, 4): 2,
    }
    dim_bonds_op = {
        (1, 2): 6,
        (2, 3): 9,
        (0, 2): 3,
        (2, 4): 4,
    }

    # physical quantum numbers
    qsite = rng.integers(-2, 3, size=d)
    # virtual bond quantum numbers of the local quantum state and tree operator
    qbonds_state = { idx: rng.integers(-1, 2, size=dim) for idx, dim in dim_bonds_state.items() }
    qbonds_op    = { idx: rng.integers(-1, 2, size=dim) for idx, dim in dim_bonds_op.items() }

    # local TTNS tensor
    c0 = crandn((dim_bonds_state[(1, 2)],
                 dim_bonds_state[(2, 3)],
                 dim_bonds_state[(0, 2)],
                 d,
                 dim_bonds_state[(2, 4)]),
                 rng)
    # enforce sparsity pattern based on quantum numbers
    enforce_qsparsity(c0, (qbonds_state[(1, 2)], -qbonds_state[(2, 3)], qbonds_state[(0, 2)], qsite, -qbonds_state[(2, 4)]))
    assert np.linalg.norm(c0) > 0
    c0 *= rng.uniform(low=0.8, high=1.2) / np.linalg.norm(c0)

    # local TTNO tensor
    a_op = crandn((dim_bonds_op[(1, 2)], dim_bonds_op[(2, 3)], dim_bonds_op[(0, 2)], d, d, dim_bonds_op[(2, 4)]), rng)
    # enforce sparsity pattern based on quantum numbers
    enforce_qsparsity(a_op, (qbonds_op[(1, 2)], -qbonds_op[(2, 3)], qbonds_op[(0, 2)], qsite, -qsite, -qbonds_op[(2, 4)]))
    assert np.linalg.norm(a_op) > 0
    a_op *= rng.uniform(low=0.8, high=1.2) / np.linalg.norm(a_op)
    hamiltonian = TreeNode(2, a_op, [None, None, None])

    # fictitious averages
    avgs = { idx: crandn((dim_bonds_state[idx], dim_bonds_op[idx], dim_bonds_state[idx]), rng)
            for idx in [(1, 2), (2, 3), (0, 2)] }
    for idx, avg in avgs.items():
        # enforce sparsity pattern based on quantum numbers
        enforce_qsparsity(avg, (-qbonds_state[idx], qbonds_op[idx], qbonds_state[idx]))
        assert np.linalg.norm(avg) > 0
        avg *= rng.uniform(low=0.8, high=1.2) / np.linalg.norm(avg)
    avg_children = [
        TreeNode(1, avgs[(1, 2)], []),
        TreeNode(3, avgs[(2, 3)], []),
        TreeNode(0, avgs[(0, 2)], [])]

    env_parent = crandn((dim_bonds_state[(2, 4)], dim_bonds_op[(2, 4)], dim_bonds_state[(2, 4)]), rng)
    # enforce sparsity pattern based on quantum numbers
    enforce_qsparsity(env_parent, (-qbonds_state[(2, 4)], qbonds_op[(2, 4)], qbonds_state[(2, 4)]))
    assert np.linalg.norm(env_parent) > 0
    env_parent *= rng.uniform(low=0.8, high=1.2) / np.linalg.norm(env_parent)

    c1 = bug_flow_update_connecting_tensor(c0, avg_children, hamiltonian, env_parent, prefactor, dt)

    with h5py.File("data/test_bug_flow_update_connecting_tensor.hdf5", "w") as file:
        file.attrs["qsite"]         = qsite
        for idx, qbond in qbonds_state.items():
            file.attrs[f"qbond{idx[0]}{idx[1]}_state"] = qbond
        for idx, qbond in qbonds_op.items():
            file.attrs[f"qbond{idx[0]}{idx[1]}_op"] = qbond
        file.attrs["prefactor"] = prefactor
        file.attrs["dt"] = dt
        file["a_op"] = a_op.transpose((2, 0, 3, 4, 1, 5))
        file["c0"] = c0.transpose((2, 0, 3, 1, 4))
        file["c1"] = c1.transpose((2, 0, 3, 1, 4))
        for idx, avg in avgs.items():
            file[f"avg{idx[0]}{idx[1]}"] = avg.transpose((2, 1, 0))
        file["env_parent"] = env_parent.transpose((2, 1, 0))


def bug_tree_time_step_data():

    # random number generator
    rng = np.random.default_rng(291)

    d = 2
    nsites_physical = 8

    # tree topology (with branching node at site 8):
    #
    #          7
    #         ╱ ╲
    #        ╱   ╲
    #       ╱     ╲
    #      ╱       ╲
    #     3         8
    #    ╱ ╲       ╱│╲
    #   ╱   ╲     ╱ │ ╲
    #  0     2   4  5  6
    #        │
    #        │
    #        1
    #
    local_dims = (((d,), ((d,), d), d), ((d,), (d,), (d,), 1), d)

    # virtual bond dimensions of the initial state
    dim_bonds_init = {
        (0, 3): 2,
        (1, 2): 2,
        (2, 3): 5,
        (3, 7): 9,
        (4, 8): 2,
        (5, 8): 2,
        (6, 8): 2,
        (7, 8): 6,
    }

    i_sites_init = (((0,), ((1,), 2), 3), ((4,), (5,), (6,), 8), 7)
    y_init = construct_random_tree(i_sites_init, -1, [d, d, d, d, d, d, d, d, 1], dim_bonds_init, 0.75, rng)

    # physical quantum numbers
    qsite = np.array([0, 1])
    # quantum number sector
    qnum_sector = 5
    # include quantum number sector at last (branching) tensor by modifying its dummy quantum number
    qsites = 8 * (qsite,) + (np.array([-qnum_sector]),)
    # virtual bond quantum numbers of the initial quantum state,
    # ensuring that allowed bond dimensions are at least 2
    qbonds_init = {
        (0, 3): np.array([0, 1]),
        (1, 2): np.array([1, 0]),
        (2, 3): np.array([1, 0, 2, 0, 1]),
        (3, 7): np.array([2, 4, 3, 2, 5, 3, 3, 2, 4]),
        (4, 8): np.array([1, 0]),
        (5, 8): np.array([0, 1]),
        (6, 8): np.array([0, 1]),
        (7, 8): np.array([4, 3, 5, 5, 2, 4]),
    }
    for idx, dim in dim_bonds_init.items():
        assert len(qbonds_init[idx]) == dim
    tree_enforce_qsparsity(y_init, -1, qsites, qbonds_init)

    local_tensors_init = tree_local_tensors(y_init)

    # construct Hamiltonian as TTNO
    th = 1.1
    u  = 2.4
    mu = 0.7
    hamiltonian_matrix = construct_bose_hubbard_1d_hamiltonian(nsites_physical, 2, th, u, mu).todense()
    assert np.allclose(hamiltonian_matrix.conj().T, hamiltonian_matrix)
    hamiltonian = tree_from_operator(hamiltonian_matrix, local_dims, tol=1e-8)
    # rename site indices 7 <-> 8
    assert hamiltonian.i_site == 8
    assert hamiltonian.children[1].i_site == 7
    hamiltonian.i_site = 7
    hamiltonian.children[1].i_site = 8

    # integration prefactor
    prefactor = -0.1 - 0.8j
    # time step
    dt = 0.1

    # relative truncation tolerance
    rel_tol = 1e-2

    # perform BUG time integration
    nsteps = 5
    y = y_init
    for i in range(nsteps):
        y = bug_tree_time_step(y, hamiltonian, prefactor, dt, rel_tol)

    y_vec = y.to_full_tensor()[0].reshape(-1)

    with h5py.File("data/test_bug_tree_time_step.hdf5", "w") as file:
        file.attrs["qnum_sector"] = qnum_sector
        file.attrs["th"] = th
        file.attrs["u"]  = u
        file.attrs["mu"] = mu
        file.attrs["prefactor"] = prefactor
        file.attrs["dt"] = dt
        file.attrs["rel_tol"] = rel_tol
        file.attrs["nsteps"] = nsteps
        for idx, qbond in qbonds_init.items():
            file.attrs[f"qbond{idx[0]}{idx[1]}"] = qbond
        for i in range(7):
            file[f"a{i}_init"] = local_tensors_init[i]
        # axes are sorted by site indices in C code
        file["a7_init"] = local_tensors_init[7].transpose((0, 2, 1, 3))
        file["a8_init"] = local_tensors_init[8].transpose((0, 1, 2, 4, 3))
        file["y"] = y_vec


def main():
    bug_flow_update_basis_leaf_data()
    bug_flow_update_connecting_tensor_data()
    bug_tree_time_step_data()


if __name__ == "__main__":
    main()
