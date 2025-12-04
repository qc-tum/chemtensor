/// \file bug.c
/// \brief Basis-Update and Galerkin (BUG) integration for tree tensor networks.

#include <memory.h>
#include "bug.h"
#include "tree_ops.h"
#include "runge_kutta.h"
#include "aligned_memory.h"


static void bug_time_step_subtree(const struct ttno* hamiltonian, const int i_site, const int i_parent,
	const struct block_sparse_tensor* restrict avg_bonds, const struct block_sparse_tensor* restrict env_parent,
	const void* prefactor, const double dt, struct ttns* state,
	struct block_sparse_tensor* restrict avg_bonds_augmented, struct block_sparse_tensor* restrict augment_maps,
	struct block_sparse_tensor* restrict c0, struct block_sparse_tensor* restrict c1);


//________________________________________________________________________________________________________________________
///
/// \brief Auxiliary data required for the right side ODE function governing the local basis update.
///
struct bug_ode_func_basis_update_leaf_data
{
	const struct block_sparse_tensor* op;   //!< local TTNO tensor
	const struct block_sparse_tensor* env;  //!< environment tensor
	const void* prefactor;                  //!< prefactor of the differential equation
};


static void bug_ode_func_basis_update_leaf_0(const double time, const struct block_sparse_tensor* restrict y, const void* restrict data, struct block_sparse_tensor* restrict ret)
{
	// case of current site index < neighboring (parent) site index

	// suppress unused parameter warning
	(void)time;

	const struct bug_ode_func_basis_update_leaf_data* fdata = data;

	assert(y->ndim == 2);
	assert(fdata->op->ndim  == 3);
	assert(fdata->env->ndim == 3);

	struct block_sparse_tensor t;
	block_sparse_tensor_dot(y, TENSOR_AXIS_RANGE_TRAILING, fdata->env, TENSOR_AXIS_RANGE_LEADING, 1, &t);
	block_sparse_tensor_dot(fdata->op, TENSOR_AXIS_RANGE_TRAILING, &t, TENSOR_AXIS_RANGE_LEADING, 2, ret);
	delete_block_sparse_tensor(&t);
	scale_block_sparse_tensor(fdata->prefactor, ret);
}

static void bug_ode_func_basis_update_leaf_1(const double time, const struct block_sparse_tensor* restrict y, const void* restrict data, struct block_sparse_tensor* restrict ret)
{
	// case of current site index > neighboring (parent) site index, without auxiliary axis

	// suppress unused parameter warning
	(void)time;

	const struct bug_ode_func_basis_update_leaf_data* fdata = data;

	assert(y->ndim == 2);
	assert(fdata->op->ndim  == 3);
	assert(fdata->env->ndim == 3);

	struct block_sparse_tensor t;
	block_sparse_tensor_dot(y, TENSOR_AXIS_RANGE_TRAILING, fdata->op, TENSOR_AXIS_RANGE_TRAILING, 1, &t);
	block_sparse_tensor_dot(fdata->env, TENSOR_AXIS_RANGE_LEADING, &t, TENSOR_AXIS_RANGE_LEADING, 2, ret);
	delete_block_sparse_tensor(&t);
	scale_block_sparse_tensor(fdata->prefactor, ret);
}

static void bug_ode_func_basis_update_leaf_2(const double time, const struct block_sparse_tensor* restrict y, const void* restrict data, struct block_sparse_tensor* restrict ret)
{
	// case of current site index > neighboring (parent) site index, with auxiliary axis attached to current site tensor

	// suppress unused parameter warning
	(void)time;

	const struct bug_ode_func_basis_update_leaf_data* fdata = data;

	assert(y->ndim == 3);  // virtual bond to neighbor, physical, and auxiliary axis
	assert(fdata->op->ndim  == 3);
	assert(fdata->env->ndim == 3);

	struct block_sparse_tensor t;
	block_sparse_tensor_multiply_axis(y, 1, fdata->op, TENSOR_AXIS_RANGE_TRAILING, &t);
	block_sparse_tensor_dot(fdata->env, TENSOR_AXIS_RANGE_LEADING, &t, TENSOR_AXIS_RANGE_LEADING, 2, ret);
	delete_block_sparse_tensor(&t);
	scale_block_sparse_tensor(fdata->prefactor, ret);
}


//________________________________________________________________________________________________________________________
///
/// \brief Update and augment the basis matrix of a subtree for the Schrödinger differential equation with Hamiltonian given as TTNO.
/// 'state' is updated in-place; 'avg_bonds' contains the operator averages of the initial state and 'avg_bonds_augmented' is filled with the averages after the basis update.
///
/// TTNO-adapted Algorithm 5 in "Rank-adaptive time integration of tree tensor networks".
///
/// Augmentation by identity blocks is necessary to activate all possible quantum number sectors.
///
void bug_flow_update_basis(const struct ttno* hamiltonian, const int i_site, const int i_parent,
	const struct block_sparse_tensor* restrict avg_bonds, const struct block_sparse_tensor* restrict env_parent, const struct block_sparse_tensor* restrict s0,
	const void* prefactor, const double dt, struct ttns* state,
	struct block_sparse_tensor* restrict avg_bonds_augmented, struct block_sparse_tensor* restrict augment_maps)
{
	assert(i_site != i_parent);
	const int nsites = state->topology.num_nodes;

	struct block_sparse_tensor state_ai_prev = state->a[i_site];  // copy internal data pointers

	// parent bond axis
	const int i_ax_p = ttns_tensor_bond_axis_index(&state->topology, i_site, i_parent);
	assert(i_ax_p != -1);

	// multiply parent bond by 's0'
	assert(s0->ndim == 2);
	block_sparse_tensor_multiply_axis(&state_ai_prev, i_ax_p, s0, TENSOR_AXIS_RANGE_LEADING, &state->a[i_site]);

	assert(state->topology.num_neighbors[i_site] > 0);
	if (state->topology.num_neighbors[i_site] == 1)
	{
		// leaf node

		assert(state->topology.neighbor_map[i_site][0] == i_parent);
		assert(hamiltonian->a[i_site].ndim == 3);

		// right-hand side data of the ordinary differential equation for the basis update
		struct bug_ode_func_basis_update_leaf_data fdata = {
			.op        = &hamiltonian->a[i_site],
			.env       = env_parent,
			.prefactor = prefactor
		};
		struct block_sparse_tensor k1;
		runge_kutta_4_block_sparse(0, &state->a[i_site], i_site < i_parent ? bug_ode_func_basis_update_leaf_0 : (i_site < nsites - 1 ? bug_ode_func_basis_update_leaf_1 : bug_ode_func_basis_update_leaf_2), &fdata, dt, &k1);

		// concatenate new and original state
		struct block_sparse_tensor t[2] = { k1, state_ai_prev };  // copy internal data pointers
		struct block_sparse_tensor c;
		block_sparse_tensor_concatenate(t, 2, i_site < i_parent ? state_ai_prev.ndim - 1 : 0, &c);
		delete_block_sparse_tensor(&k1);

		// temporarily combine the physical and auxiliary axis on the last site
		struct block_sparse_tensor_axis_matricization_info mat_info;
		if (i_site == nsites - 1)
		{
			assert(c.ndim == 3);
			assert(c.dim_logical[2] == 1);
			assert(c.axis_dir[1] == TENSOR_AXIS_OUT);
			assert(c.axis_dir[2] == TENSOR_AXIS_IN);
			struct block_sparse_tensor c_mat;
			block_sparse_tensor_matricize_axis(&c, 0, 0, TENSOR_AXIS_OUT, &c_mat, &mat_info);
			delete_block_sparse_tensor(&c);
			c = c_mat;  // copy internal data pointers
		}

		// orthonormalize by QR decomposition and augment by identity blocks to obtain new basis
		struct block_sparse_tensor u_hat;
		if (i_site < i_parent)
		{
			struct block_sparse_tensor q, r;
			block_sparse_tensor_qr(&c, QR_REDUCED, &q, &r);
			delete_block_sparse_tensor(&r);
			block_sparse_tensor_augment_identity_blocks(&q, false, &u_hat);
			delete_block_sparse_tensor(&q);
		}
		else
		{
			struct block_sparse_tensor q, r;
			block_sparse_tensor_rq(&c, QR_REDUCED, &r, &q);
			delete_block_sparse_tensor(&r);
			block_sparse_tensor_invert_axis_quantum_numbers(&q, 0);  // flip axis direction for augmentation with identity blocks
			block_sparse_tensor_augment_identity_blocks(&q, true, &u_hat);
			delete_block_sparse_tensor(&q);
			block_sparse_tensor_invert_axis_quantum_numbers(&u_hat, 0);  // undo axis direction flip
		}
		delete_block_sparse_tensor(&c);

		// split the physical and auxiliary axis on the last site
		if (i_site == nsites - 1)
		{
			struct block_sparse_tensor tmp;
			block_sparse_tensor_dematricize_axis(&u_hat, &mat_info, &tmp);
			delete_block_sparse_tensor(&u_hat);
			u_hat = tmp;  // copy internal data pointers
			assert(u_hat.ndim == 3);
			delete_block_sparse_tensor_axis_matricization_info(&mat_info);
		}

		// compute inner product between new (extended) basis and original state
		local_ttns_inner_product(&u_hat, &state_ai_prev, &state->topology, i_site, i_parent, augment_maps);

		// compute new averages (expectation values)
		local_ttno_inner_product(&u_hat, &hamiltonian->a[i_site], &u_hat, &hamiltonian->topology, i_site, i_parent, avg_bonds_augmented);

		// replace current state tensor
		delete_block_sparse_tensor(&state->a[i_site]);
		state->a[i_site] = u_hat;  // copy internal data pointers
	}
	else  // not a leaf node
	{
		struct block_sparse_tensor c0, c1;
		bug_time_step_subtree(hamiltonian, i_site, i_parent, avg_bonds, env_parent, prefactor, dt, state, avg_bonds_augmented, augment_maps, &c0, &c1);

		// matricize
		struct block_sparse_tensor c01_mat[2];
		struct block_sparse_tensor_axis_matricization_info c01_mat_info[2];
		block_sparse_tensor_matricize_axis(&c0, i_ax_p, 1, -c0.axis_dir[i_ax_p], &c01_mat[0], &c01_mat_info[0]);
		block_sparse_tensor_matricize_axis(&c1, i_ax_p, 1, -c1.axis_dir[i_ax_p], &c01_mat[1], &c01_mat_info[1]);
		delete_block_sparse_tensor(&c0);
		delete_block_sparse_tensor(&c1);

		// concatenate matrices
		struct block_sparse_tensor c;
		block_sparse_tensor_concatenate(c01_mat, 2, 1, &c);
		delete_block_sparse_tensor(&c01_mat[0]);
		delete_block_sparse_tensor(&c01_mat[1]);

		// perform QR decomposition; "complete" QR decomposition can increase accuracy
		struct block_sparse_tensor q, r;
		block_sparse_tensor_qr(&c, QR_COMPLETE, &q, &r);
		delete_block_sparse_tensor(&r);
		delete_block_sparse_tensor(&c);

		// augment 'q'
		struct block_sparse_tensor q_aug;
		block_sparse_tensor_augment_identity_blocks(&q, false, &q_aug);
		delete_block_sparse_tensor(&q);

		// undo matricization and update local site tensor
		delete_block_sparse_tensor(&state->a[i_site]);
		block_sparse_tensor_dematricize_axis(&q_aug, &c01_mat_info[0], &state->a[i_site]);
		delete_block_sparse_tensor(&q_aug);
		delete_block_sparse_tensor_axis_matricization_info(&c01_mat_info[0]);
		delete_block_sparse_tensor_axis_matricization_info(&c01_mat_info[1]);

		// compute inner product between new augmented (orthonormal) state and input state
		local_ttns_inner_product(&state->a[i_site], &state_ai_prev, &state->topology, i_site, i_parent, augment_maps);

		// compute new average (expectation value)
		local_ttno_inner_product(&state->a[i_site], &hamiltonian->a[i_site], &state->a[i_site], &hamiltonian->topology, i_site, i_parent, avg_bonds_augmented);
	}

	delete_block_sparse_tensor(&state_ai_prev);
}


//________________________________________________________________________________________________________________________
///
/// \brief ODE function data for updating the connecting tensor.
///
struct bug_ode_func_connecting_tensor_data
{
	const struct block_sparse_tensor* op;    //!< local TTNO tensor
	const struct abstract_graph* topology;   //!< tree topology
	const struct block_sparse_tensor* envs;  //!< environment tensors for all neighbors
	const void* prefactor;                   //!< prefactor of the differential equation
	const int i_site;                        //!< current site
};


//________________________________________________________________________________________________________________________
///
/// \brief ODE function for updating the connecting tensor.
///
static void bug_ode_func_connecting_tensor(const double t, const struct block_sparse_tensor* restrict psi, const void* restrict data, struct block_sparse_tensor* restrict ret)
{
	// suppress unused parameter warning
	(void)t;

	const struct bug_ode_func_connecting_tensor_data* fdata = data;

	apply_local_ttno_tensor(fdata->op, psi, fdata->topology, fdata->i_site, fdata->envs, ret);
	scale_block_sparse_tensor(fdata->prefactor, ret);
}


//________________________________________________________________________________________________________________________
///
/// \brief Update the (already augmented) connecting tensor of a tree node
/// for a Schrödinger differential equation with Hamiltonian given as TTNO.
///
/// TTNO-adapted Algorithm 6 in "Rank-adaptive time integration of tree tensor networks".
///
void bug_flow_update_connecting_tensor(const struct block_sparse_tensor* op, const struct block_sparse_tensor* restrict c0,
	const struct abstract_graph* topology, const int i_site, const int i_parent,
	const struct block_sparse_tensor* restrict avg_bonds_augmented, const struct block_sparse_tensor* restrict env_parent,
	const void* prefactor, const double dt,
	struct block_sparse_tensor* restrict c1)
{
	// shallow copy of environment tensors
	struct block_sparse_tensor* envs = ct_malloc(topology->num_neighbors[i_site] * sizeof(struct block_sparse_tensor));
	for (int n = 0; n < topology->num_neighbors[i_site]; n++)
	{
		const int k = topology->neighbor_map[i_site][n];
		assert(k != i_site);
		envs[n] = (k == i_parent ? *env_parent : avg_bonds_augmented[k]);  // copy internal data pointers
	}

	struct bug_ode_func_connecting_tensor_data fdata = {
		.op        = op,
		.topology  = topology,
		.envs      = envs,
		.prefactor = prefactor,
		.i_site    = i_site,
	};

	// perform time evolution
	runge_kutta_4_block_sparse(0, c0, bug_ode_func_connecting_tensor, &fdata, dt, c1);

	ct_free(envs);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the environment tensor for the child node 'i_child'
/// after gauge-transforming the node 'i_site' into an isometry towards the child.
///
static void bug_compute_child_environment(const struct block_sparse_tensor* restrict op, const struct block_sparse_tensor* restrict state,
	const struct abstract_graph* topology, const int i_site, const int i_parent, const int i_child,
	const struct block_sparse_tensor* restrict avg_bonds, const struct block_sparse_tensor* restrict env_parent,
	struct block_sparse_tensor* restrict s0, struct block_sparse_tensor* restrict env)
{
	// child bond axis
	const int i_ax_c = ttns_tensor_bond_axis_index(topology, i_site, i_child);
	assert(i_ax_c != -1);

	struct block_sparse_tensor ai_mat;
	struct block_sparse_tensor_axis_matricization_info mat_info;
	block_sparse_tensor_matricize_axis(state, i_ax_c, 1, TENSOR_AXIS_OUT, &ai_mat, &mat_info);

	// perform QR decomposition
	struct block_sparse_tensor q0, r0;
	block_sparse_tensor_qr(&ai_mat, QR_REDUCED, &q0, &r0);
	delete_block_sparse_tensor(&ai_mat);

	const int perm[2] = { 1, 0 };
	block_sparse_tensor_transpose(perm, &r0, s0);
	delete_block_sparse_tensor(&r0);

	struct block_sparse_tensor q0ten;
	block_sparse_tensor_dematricize_axis(&q0, &mat_info, &q0ten);
	delete_block_sparse_tensor_axis_matricization_info(&mat_info);
	delete_block_sparse_tensor(&q0);

	// shallow copy of internal data pointers
	struct block_sparse_tensor* avg_bonds_mod = ct_malloc(topology->num_nodes * sizeof(struct block_sparse_tensor));
	memcpy(avg_bonds_mod, avg_bonds, topology->num_nodes * sizeof(struct block_sparse_tensor));
	if (i_parent != -1) {
		avg_bonds_mod[i_parent] = *env_parent;
	}

	// designating child node as temporary parent
	local_ttno_inner_product(&q0ten, op, &q0ten, topology, i_site, i_child, avg_bonds_mod);

	// copy internal data pointers
	*env = avg_bonds_mod[i_site];

	ct_free(avg_bonds_mod);
	delete_block_sparse_tensor(&q0ten);
}


//________________________________________________________________________________________________________________________
///
/// \brief Perform a recursive rank-augmenting TTN integration step on a (sub-)tree
/// for a Schrödinger differential equation with Hamiltonian given as TTNO.
///
/// TTNO-adapted Algorithm 4 in "Rank-adaptive time integration of tree tensor networks".
///
static void bug_time_step_subtree(const struct ttno* hamiltonian, const int i_site, const int i_parent,
	const struct block_sparse_tensor* restrict avg_bonds, const struct block_sparse_tensor* restrict env_parent,
	const void* prefactor, const double dt, struct ttns* state,
	struct block_sparse_tensor* restrict avg_bonds_augmented, struct block_sparse_tensor* restrict augment_maps,
	struct block_sparse_tensor* restrict c0, struct block_sparse_tensor* restrict c1)
{
	for (int n = 0; n < hamiltonian->topology.num_neighbors[i_site]; n++)
	{
		const int k = hamiltonian->topology.neighbor_map[i_site][n];
		assert(k != i_site);
		if (k == i_parent) {
			continue;
		}

		struct block_sparse_tensor s0, env;
		bug_compute_child_environment(&hamiltonian->a[i_site], &state->a[i_site], &hamiltonian->topology, i_site, i_parent, k, avg_bonds, env_parent, &s0, &env);

		bug_flow_update_basis(hamiltonian, k, i_site, avg_bonds, &env, &s0, prefactor, dt, state, avg_bonds_augmented, augment_maps);

		delete_block_sparse_tensor(&s0);
		delete_block_sparse_tensor(&env);
	}

	// augment the initial connecting tensor
	copy_block_sparse_tensor(&state->a[i_site], c0);
	for (int n = 0; n < hamiltonian->topology.num_neighbors[i_site]; n++)
	{
		const int k = hamiltonian->topology.neighbor_map[i_site][n];
		assert(k != i_site);
		if (k == i_parent) {
			continue;
		}

		struct block_sparse_tensor tmp;
		const int i_ax = (k < i_site ? n : n + 1);
		block_sparse_tensor_multiply_axis(c0, i_ax, &augment_maps[k], TENSOR_AXIS_RANGE_LEADING, &tmp);
		delete_block_sparse_tensor(c0);
		*c0 = tmp;  // copy internal data pointers
	}

	bug_flow_update_connecting_tensor(&hamiltonian->a[i_site], c0, &hamiltonian->topology, i_site, i_parent, avg_bonds_augmented, env_parent, prefactor, dt, c1);
}


//________________________________________________________________________________________________________________________
///
/// \brief Perform a rank-augmenting TTN integration step
/// for a Schrödinger differential equation with Hamiltonian given as TTNO.
/// The 'state' is updated in-place.
///
int bug_tree_time_step(const struct ttno* hamiltonian, const int i_root, const void* prefactor, const double dt, const double rel_tol_compress, const ct_long max_vdim, struct ttns* state)
{
	const int nsites = hamiltonian->topology.num_nodes;

	struct block_sparse_tensor* avg_bonds = ct_malloc(nsites * sizeof(struct block_sparse_tensor));
	ttno_subtrees_inner_products(state, hamiltonian, state, i_root, avg_bonds);
	struct block_sparse_tensor* avg_bonds_augmented = ct_malloc(nsites * sizeof(struct block_sparse_tensor));
	struct block_sparse_tensor* augment_maps = ct_malloc(nsites * sizeof(struct block_sparse_tensor));

	struct block_sparse_tensor c0, c1;
	bug_time_step_subtree(hamiltonian, i_root, -1, avg_bonds, NULL, prefactor, dt, state, avg_bonds_augmented, augment_maps, &c0, &c1);
	delete_block_sparse_tensor(&c0);
	delete_block_sparse_tensor(&state->a[i_root]);
	state->a[i_root] = c1;  // copy internal data pointers

	// compute new average (expectation value)
	local_ttno_inner_product(&state->a[i_root], &hamiltonian->a[i_root], &state->a[i_root], &hamiltonian->topology, i_root, -1, avg_bonds_augmented);

	// compress state
	// squared tolerance corresponds to Euclidean length of truncated singular values
	int ret = ttns_compress(i_root, square(dt * rel_tol_compress), false, max_vdim, state);
	if (ret < 0) {
		return ret;
	}

	for (int l = 0; l < nsites; l++)
	{
		if (l != i_root) {
			delete_block_sparse_tensor(&augment_maps[l]);
		}
		delete_block_sparse_tensor(&avg_bonds_augmented[l]);
		delete_block_sparse_tensor(&avg_bonds[l]);
	}
	ct_free(augment_maps);
	ct_free(avg_bonds_augmented);
	ct_free(avg_bonds);

	return 0;
}
