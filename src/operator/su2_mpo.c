/// \file su2_mpo.c
/// \brief SU(2) symmetric matrix product operator (MPO) data structure and functions.

#include <stdio.h>
#include "su2_mpo.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for an SU(2) symmetric matrix product operator.
///
void allocate_su2_mpo(
	const enum numeric_type dtype, const int nsites,
	const struct su2_irreducible_list* restrict site_irreps, const ct_long* restrict site_dim_degen,
	const struct su2_irreducible_list* restrict bond_irreps, const ct_long** restrict bond_dim_degen,
	struct su2_mpo* mpo)
{
	assert(nsites >= 1);
	mpo->nsites = nsites;

	mpo->a = ct_calloc(nsites, sizeof(struct su2_tensor));

	for (int l = 0; l < nsites; l++)
	{
		// construct the fuse and split tree
		//
		// physical input axis  2    3  right virtual bond
		//                       ╲  ╱
		//                        ╲╱  fuse
		//                        │
		//                        │4
		//                        │
		//                        ╱╲  split
		//                       ╱  ╲
		//   left virtual bond  0    1  physical output axis
		//
		struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
		struct su2_tree_node j4f = { .i_ax = 4, .c = { &j2,  &j3  } };
		struct su2_tree_node j4s = { .i_ax = 4, .c = { &j0,  &j1  } };
		struct su2_fuse_split_tree tree = { .tree_fuse = &j4f, .tree_split = &j4s, .ndim = 5 };
		assert(su2_fuse_split_tree_is_consistent(&tree));

		// outer (logical and auxiliary) 'j' quantum numbers
		const struct su2_irreducible_list outer_irreps[4] = { bond_irreps[l], *site_irreps, *site_irreps, bond_irreps[l + 1] };
		// degeneracy dimensions, indexed by 'j' quantum numbers
		const ct_long* dim_degen[4] = { bond_dim_degen[l], site_dim_degen, site_dim_degen, bond_dim_degen[l + 1] };

		allocate_su2_tensor(dtype, 4, 0, &tree, outer_irreps, dim_degen, &mpo->a[l]);
		assert(su2_tensor_is_consistent(&mpo->a[l]));

		if (mpo->a[l].charge_sectors.nsec == 0) {
			printf("Warning: in 'allocate_su2_mpo': SU(2) tensor at site %i has no charge sectors\n", l);
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an SU(2) symmetric matrix product operator (free memory).
///
void delete_su2_mpo(struct su2_mpo* mpo)
{
	for (int l = 0; l < mpo->nsites; l++)
	{
		delete_su2_tensor(&mpo->a[l]);
	}
	ct_free(mpo->a);
	mpo->a = NULL;
}


//________________________________________________________________________________________________________________________
///
/// \brief Internal consistency check of the SU(2) MPO data structure.
///
bool su2_mpo_is_consistent(const struct su2_mpo* mpo)
{
	if (mpo->nsites <= 0) {
		return false;
	}

	for (int l = 0; l < mpo->nsites; l++)
	{
		if (!su2_tensor_is_consistent(&mpo->a[l])) {
			return false;
		}
		if (mpo->a[l].ndim_logical != 4 || mpo->a[l].ndim_auxiliary != 0) {
			return false;
		}
	}

	// virtual bond quantum numbers and degeneracies must match
	for (int l = 0; l < mpo->nsites - 1; l++)
	{
		if (!su2_irreducible_list_equal(&mpo->a[l].outer_irreps[3], &mpo->a[l + 1].outer_irreps[0])) {
			return false;
		}
		for (int k = 0; k < mpo->a[l].outer_irreps[3].num; k++)
		{
			const qnumber j = mpo->a[l].outer_irreps[3].jlist[k];
			if (mpo->a[l].dim_degen[3][j] != mpo->a[l + 1].dim_degen[0][j]) {
				return false;
			}
		}
	}

	// axis directions
	for (int l = 0; l < mpo->nsites; l++)
	{
		if (su2_tensor_logical_axis_direction(&mpo->a[l], 0) != TENSOR_AXIS_OUT ||
		    su2_tensor_logical_axis_direction(&mpo->a[l], 1) != TENSOR_AXIS_OUT ||
		    su2_tensor_logical_axis_direction(&mpo->a[l], 2) != TENSOR_AXIS_IN  ||
		    su2_tensor_logical_axis_direction(&mpo->a[l], 3) != TENSOR_AXIS_IN) {
			return false;
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Contract two neighboring SU(2) MPO tensors along the virtual bond (without merging the physical axes).
///
void su2_mpo_contract_tensor_pair(const struct su2_tensor* restrict a0, const struct su2_tensor* restrict a1, struct su2_tensor* restrict a)
{
	assert(a0->ndim_auxiliary == 0);
	assert(a1->ndim_auxiliary == 0);

	// combine a0 and a1 by contracting the shared bond;
	// allowing a0 and a1 to have several physical axes
	struct su2_tensor s;
	su2_tensor_contract_yoga(a0, a0->ndim_logical - 1, a1, 0, &s);

	// group physical input and output axes together
	struct su2_tensor t;
	{
		const int ndim_phys_0 = (a0->ndim_logical - 2) / 2;
		const int ndim_phys_1 = (a1->ndim_logical - 2) / 2;
		assert(s.ndim_logical == 2 * (ndim_phys_0 + ndim_phys_1 + 1));
		// interchange ordering of physical input axes of a0 and physical output axes of a1
		int* perm = ct_malloc(s.ndim_logical * sizeof(int));
		for (int i = 0; i < s.ndim_logical; i++) {
			perm[i] = i;
		}
		for (int i = 0; i < ndim_phys_1; i++) {
			perm[1 + ndim_phys_0 + i] = 1 + 2 * ndim_phys_0 + i;
		}
		for (int i = 0; i < ndim_phys_0; i++) {
			perm[1 + ndim_phys_0 + ndim_phys_1 + i] = 1 + ndim_phys_0 + i;
		}
		su2_tensor_transpose_logical(perm, &s, &t);
		delete_su2_tensor(&s);
		ct_free(perm);
	}

	// isolate virtual bond axis in fusion-splitting tree
	int i_ax_p_left;
	{
		const struct su2_tree_node* node = su2_tree_find_parent_node(t.tree.tree_split, 0);
		assert(node != NULL);
		i_ax_p_left = node->i_ax;
	}
	su2_tensor_fmove(&t, i_ax_p_left, &s);
	delete_su2_tensor(&t);
	int i_ax_p_right;
	{
		const struct su2_tree_node* node = su2_tree_find_parent_node(s.tree.tree_fuse, s.ndim_logical - 1);
		assert(node != NULL);
		i_ax_p_right = node->i_ax;
	}
	su2_tensor_fmove(&s, i_ax_p_right, a);
	delete_su2_tensor(&s);
}


//________________________________________________________________________________________________________________________
///
/// \brief Contract all tensors of an SU(2) symmetric MPO to obtain the tensor representation on the full Hilbert space.
/// All physical axes and the (dummy) virtual bonds are retained in the output tensor.
///
void su2_mpo_to_tensor(const struct su2_mpo* mpo, struct su2_tensor* mat)
{
	assert(mpo->nsites > 0);

	if (mpo->nsites == 1)
	{
		copy_su2_tensor(&mpo->a[0], mat);
	}
	else if (mpo->nsites == 2)
	{
		su2_mpo_contract_tensor_pair(&mpo->a[0], &mpo->a[1], mat);
	}
	else
	{
		struct su2_tensor t[2];
		su2_mpo_contract_tensor_pair(&mpo->a[0], &mpo->a[1], &t[0]);
		for (int i = 2; i < mpo->nsites; i++)
		{
			su2_mpo_contract_tensor_pair(&t[i % 2], &mpo->a[i], i < mpo->nsites - 1 ? &t[(i + 1) % 2] : mat);
			delete_su2_tensor(&t[i % 2]);
		}
	}
}
