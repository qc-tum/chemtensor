/// \file su2_mps.c
/// \brief SU(2) symmetric matrix product state (MPS) data structure.

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "su2_mps.h"
#include "su2_util.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Allocate an "empty" SU(2) symmetric matrix product state, without the actual tensors.
///
void allocate_empty_su2_mps(const int nsites, struct su2_mps* mps)
{
	assert(nsites >= 1);
	mps->nsites = nsites;

	mps->a = ct_calloc(nsites, sizeof(struct su2_tensor));
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for an SU(2) symmetric matrix product state.
///
void allocate_su2_mps(
	const enum numeric_type dtype, const int nsites,
	const struct su2_irreducible_list* restrict site_irreps, const ct_long* restrict site_dim_degen,
	const struct su2_irreducible_list* restrict bond_irreps, const ct_long** restrict bond_dim_degen,
	struct su2_mps* mps)
{
	allocate_empty_su2_mps(nsites, mps);

	for (int l = 0; l < nsites; l++)
	{
		// construct the fuse and split tree
		//
		//                       2  right virtual bond
		//                       │
		//                       │   fuse
		//                       ╱╲  split
		//                      ╱  ╲
		//  left virtual bond  0    1  physical axis
		//
		struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2f = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j2s = { .i_ax = 2, .c = { &j0,  &j1  } };
		struct su2_fuse_split_tree tree = { .tree_fuse = &j2f, .tree_split = &j2s, .ndim = 3 };
		assert(su2_fuse_split_tree_is_consistent(&tree));

		// outer (logical and auxiliary) 'j' quantum numbers
		const struct su2_irreducible_list outer_irreps[3] = { bond_irreps[l], *site_irreps, bond_irreps[l + 1] };
		// degeneracy dimensions, indexed by 'j' quantum numbers
		const ct_long* dim_degen[3] = { bond_dim_degen[l], site_dim_degen, bond_dim_degen[l + 1] };

		allocate_su2_tensor(dtype, 3, 0, &tree, outer_irreps, dim_degen, &mps->a[l]);
		assert(su2_tensor_is_consistent(&mps->a[l]));

		if (mps->a[l].charge_sectors.nsec == 0) {
			printf("Warning: in 'allocate_su2_mps': SU(2) tensor at site %i has no charge sectors\n", l);
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an SU(2) symmetric matrix product state (free memory).
///
void delete_su2_mps(struct su2_mps* mps)
{
	for (int l = 0; l < mps->nsites; l++)
	{
		delete_su2_tensor(&mps->a[l]);
	}
	ct_free(mps->a);
	mps->a = NULL;
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct an SU(2) matrix product state with random normal degeneracy tensor entries.
///
void construct_random_su2_mps(
	const enum numeric_type dtype, const int nsites,
	const struct su2_irreducible_list* site_irreps, const ct_long* site_dim_degen, const qnumber irrep_sector,
	const qnumber max_bond_irrep, const ct_long max_bond_dim_degen,
	struct rng_state* rng_state, struct su2_mps* mps)
{
	assert(nsites >= 1);
	assert(site_irreps->num >= 1);
	assert(irrep_sector >= 0);
	assert(max_bond_irrep >= 0);
	assert(max_bond_dim_degen >= 1);

	// virtual bond irreducible configurations
	struct su2_irreducible_list* bond_irreps    = ct_malloc((nsites + 1) * sizeof(struct su2_irreducible_list));
	ct_long**                    bond_dim_degen = ct_malloc((nsites + 1) * sizeof(ct_long*));
	// dummy left virtual bond; set 'j' quantum number to zero
	allocate_su2_irreducible_list(1, &bond_irreps[0]);
	bond_irreps[0].jlist[0] = 0;
	bond_dim_degen[0] = ct_malloc(sizeof(ct_long));
	bond_dim_degen[0][0] = 1;
	// dummy right virtual bond; set 'j' quantum number to overall quantum number sector
	allocate_su2_irreducible_list(1, &bond_irreps[nsites]);
	bond_irreps[nsites].jlist[0] = irrep_sector;
	bond_dim_degen[nsites] = ct_calloc(irrep_sector + 1, sizeof(ct_long));
	bond_dim_degen[nsites][irrep_sector] = 1;
	// virtual bond irreducible configurations on left half
	for (int l = 1; l < (nsites + 1) / 2; l++)
	{
		su2_irreps_tensor_product(&bond_irreps[l - 1], bond_dim_degen[l - 1], site_irreps, site_dim_degen, &bond_irreps[l], &bond_dim_degen[l]);
		// clamp maximum 'j' quantum number
		for (int k = 0; k < bond_irreps[l].num; k++)
		{
			// assuming that 'j' quantum numbers in jlist are sorted
			if (bond_irreps[l].jlist[k] > max_bond_irrep)
			{
				bond_irreps[l].num = k;
				break;
			}
		}
		assert(bond_irreps[l].num > 0);
		// clamp degeneracy dimensions
		for (int k = 0; k < bond_irreps[l].num; k++)
		{
			const qnumber j = bond_irreps[l].jlist[k];
			assert(bond_dim_degen[l][j] > 0);
			if (bond_dim_degen[l][j] > max_bond_dim_degen) {
				bond_dim_degen[l][j] = max_bond_dim_degen;
			}
		}
	}
	// virtual bond irreducible configurations on right half
	for (int l = nsites - 1; l >= (nsites + 1) / 2; l--)
	{
		su2_irreps_tensor_product(site_irreps, site_dim_degen, &bond_irreps[l + 1], bond_dim_degen[l + 1], &bond_irreps[l], &bond_dim_degen[l]);
		// clamp maximum 'j' quantum number
		for (int k = 0; k < bond_irreps[l].num; k++)
		{
			// assuming that 'j' quantum numbers in jlist are sorted
			if (bond_irreps[l].jlist[k] > max_bond_irrep)
			{
				bond_irreps[l].num = k;
				break;
			}
		}
		assert(bond_irreps[l].num > 0);
		// clamp degeneracy dimensions
		for (int k = 0; k < bond_irreps[l].num; k++)
		{
			const qnumber j = bond_irreps[l].jlist[k];
			assert(bond_dim_degen[l][j] > 0);
			if (bond_dim_degen[l][j] > max_bond_dim_degen) {
				bond_dim_degen[l][j] = max_bond_dim_degen;
			}
		}
	}

	allocate_su2_mps(dtype, nsites, site_irreps, site_dim_degen, bond_irreps, (const ct_long**)bond_dim_degen, mps);

	for (int l = 0; l < nsites + 1; l++)
	{
		ct_free(bond_dim_degen[l]);
		delete_su2_irreducible_list(&bond_irreps[l]);
	}
	ct_free(bond_dim_degen);
	ct_free(bond_irreps);

	// fill degeneracy tensor entries with pseudo-random numbers, scaled by 1 / sqrt("number of entries")
	for (int l = 0; l < nsites; l++)
	{
		// logical number of entries in MPS tensor
		const ct_long nelem = su2_tensor_num_elements_logical(&mps->a[l]);
		// ensure that 'alpha' is large enough to store any numeric type
		dcomplex alpha;
		assert(mps->a[l].dtype == dtype);
		numeric_from_double(1.0 / sqrt(nelem), dtype, &alpha);
		su2_tensor_fill_random_normal(&alpha, numeric_zero(dtype), rng_state, &mps->a[l]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Internal consistency check of the SU(2) MPS data structure.
///
bool su2_mps_is_consistent(const struct su2_mps* mps)
{
	if (mps->nsites <= 0) {
		return false;
	}

	for (int l = 0; l < mps->nsites; l++)
	{
		if (!su2_tensor_is_consistent(&mps->a[l])) {
			return false;
		}
		if (mps->a[l].ndim_logical != 3 || mps->a[l].ndim_auxiliary != 0) {
			return false;
		}
	}

	// virtual bond quantum numbers and degeneracies must match
	for (int l = 0; l < mps->nsites - 1; l++)
	{
		if (!su2_irreducible_list_equal(&mps->a[l].outer_irreps[2], &mps->a[l + 1].outer_irreps[0])) {
			return false;
		}
		for (int k = 0; k < mps->a[l].outer_irreps[2].num; k++)
		{
			const qnumber j = mps->a[l].outer_irreps[2].jlist[k];
			if (mps->a[l].dim_degen[2][j] != mps->a[l + 1].dim_degen[0][j]) {
				return false;
			}
		}
	}

	// axis directions and tree topology
	for (int l = 0; l < mps->nsites; l++)
	{
		if (su2_tensor_logical_axis_direction(&mps->a[l], 0) != TENSOR_AXIS_OUT ||
		    su2_tensor_logical_axis_direction(&mps->a[l], 1) != TENSOR_AXIS_OUT ||
		    su2_tensor_logical_axis_direction(&mps->a[l], 2) != TENSOR_AXIS_IN) {
			return false;
		}

		// construct the fuse and split tree
		//
		//                       2  right virtual bond
		//                       │
		//                       │   fuse
		//                       ╱╲  split
		//                      ╱  ╲
		//  left virtual bond  0    1  physical axis
		//
		struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2f = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j2s = { .i_ax = 2, .c = { &j0,  &j1  } };
		struct su2_fuse_split_tree tree_ref = { .tree_fuse = &j2f, .tree_split = &j2s, .ndim = 3 };
		assert(su2_fuse_split_tree_is_consistent(&tree_ref));

		if (!su2_fuse_split_tree_equal(&mps->a[l].tree, &tree_ref)) {
			return false;
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Left-orthonormalize a local SU(2) symmetric MPS site tensor by QR decomposition, and update tensor at next site.
///
void su2_mps_local_orthonormalize_qr(struct su2_tensor* a, struct su2_tensor* a_next)
{
	assert(a->ndim_logical == 3);
	assert(a->ndim_auxiliary == 0);
	assert(a_next->ndim_logical == 3);
	assert(a_next->ndim_auxiliary == 0);

	// combine left virtual bond and physical axis
	struct su2_tensor a_mat;
	su2_tensor_fuse_axes_add_auxiliary(a, 0, 1, &a_mat);

	// perform QR decomposition
	struct su2_tensor q, r;
	su2_tensor_qr(&a_mat, QR_REDUCED, &q, &r);
	delete_su2_tensor(&a_mat);

	// replace 'a' by reshaped 'q' matrix, using original logical dimensions and quantum numbers
	struct su2_tensor a_new;
	su2_tensor_split_axis_remove_auxiliary(&q, 0, 1, a->outer_irreps, (const ct_long**)a->dim_degen, &a_new);
	delete_su2_tensor(&q);
	delete_su2_tensor(a);
	*a = a_new;  // copy internal data pointers

	// update 'a_next' tensor: multiply with 'r' from left
	assert(r.ndim_logical == 2 && r.ndim_auxiliary == 1);
	assert(!su2_tree_node_is_leaf(r.tree.tree_fuse));
	assert(r.tree.tree_fuse->c[0]->i_ax == 1);
	assert(r.tree.tree_fuse->c[1]->i_ax == 2);  // expecting auxiliary axis at right child
	// temporary auxiliary axis for contraction with 'r'
	su2_tensor_add_auxiliary_axis(a_next, 0, false);
	struct su2_tensor a_next_update;
	const int i_ax_cntr_r[2] = { 1, 2 };
	const int i_ax_cntr_a[2] = { 0, 3 };
	su2_tensor_contract_simple(&r, i_ax_cntr_r, a_next, i_ax_cntr_a, 2, &a_next_update);
	delete_su2_tensor(a_next);
	*a_next = a_next_update;  // copy internal data pointers
	delete_su2_tensor(&r);
	assert(a_next->ndim_logical == 3 && a_next->ndim_auxiliary == 0);
}


//________________________________________________________________________________________________________________________
///
/// \brief Right-orthonormalize a local SU(2) symmetric MPS site tensor by RQ decomposition, and update tensor at previous site.
///
void su2_mps_local_orthonormalize_rq(struct su2_tensor* a, struct su2_tensor* a_prev)
{
	assert(a->ndim_logical == 3);
	assert(a->ndim_auxiliary == 0);
	assert(a_prev->ndim_logical == 3);
	assert(a_prev->ndim_auxiliary == 0);

	// combine physical and right virtual bond axis
	su2_tensor_reverse_axis_simple(a, 1);
	struct su2_tensor a_mat;
	su2_tensor_fuse_axes_add_auxiliary(a, 1, 2, &a_mat);

	// perform RQ decomposition
	struct su2_tensor r, q;
	su2_tensor_rq(&a_mat, QR_REDUCED, &r, &q);
	delete_su2_tensor(&a_mat);

	// replace 'a' by reshaped 'q' matrix, using original logical dimensions and quantum numbers
	struct su2_tensor a_new;
	su2_tensor_split_axis_remove_auxiliary(&q, 1, 2, a->outer_irreps + 1, (const ct_long**)(a->dim_degen + 1), &a_new);
	delete_su2_tensor(&q);
	delete_su2_tensor(a);
	su2_tensor_reverse_axis_simple(&a_new, 1);
	*a = a_new;  // copy internal data pointers

	// update 'a_prev' tensor: multiply with 'r' from right
	assert(r.ndim_logical == 2 && r.ndim_auxiliary == 1);
	assert(!su2_tree_node_is_leaf(r.tree.tree_split));
	assert(r.tree.tree_split->c[0]->i_ax == 2);  // expecting auxiliary axis at left child
	assert(r.tree.tree_split->c[1]->i_ax == 0);
	// temporary auxiliary axis for contraction with 'r'
	su2_tensor_add_auxiliary_axis(a_prev, 2, true);
	struct su2_tensor a_prev_update;
	const int i_ax_cntr_r[2] = { 0, 2 };
	const int i_ax_cntr_a[2] = { 2, 3 };
	su2_tensor_contract_simple(a_prev, i_ax_cntr_a, &r, i_ax_cntr_r, 2, &a_prev_update);
	delete_su2_tensor(a_prev);
	*a_prev = a_prev_update;  // copy internal data pointers
	delete_su2_tensor(&r);
	assert(a_prev->ndim_logical == 3 && a_prev->ndim_auxiliary == 0);
}


//________________________________________________________________________________________________________________________
///
/// \brief Create a dummy "head" tensor (at fictitious site -1) adapted to the first tensor of an SU(2) symmetric MPS.
///
void su2_mps_create_dummy_head_tensor(const struct su2_tensor* restrict a0, struct su2_tensor* restrict a_head)
{
	// left virtual bond of first tensor must have a single quantum number and degeneracy dimension 1
	assert(a0->outer_irreps[0].num == 1);
	assert(a0->dim_degen[0][a0->outer_irreps[0].jlist[0]] == 1);

	// construct the fuse and split tree
	//
	//                       2  right virtual bond
	//                       │
	//                       │   fuse
	//                       ╱╲  split
	//                      ╱  ╲
	//  left virtual bond  0    1  physical axis
	//
	struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
	struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
	struct su2_tree_node j2f = { .i_ax = 2, .c = { NULL, NULL } };
	struct su2_tree_node j2s = { .i_ax = 2, .c = { &j0,  &j1  } };
	struct su2_fuse_split_tree tree = { .tree_fuse = &j2f, .tree_split = &j2s, .ndim = 3 };
	assert(su2_fuse_split_tree_is_consistent(&tree));

	// outer (logical and auxiliary) 'j' quantum numbers
	// quantum number zero for the dummy site axis
	qnumber jlist_zero[1] = { 0 };
	struct su2_irreducible_list site_irreps = {
		.jlist = jlist_zero,
		.num = 1,
	};
	const struct su2_irreducible_list outer_irreps[3] = { a0->outer_irreps[0], site_irreps, a0->outer_irreps[0] };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	const ct_long site_dim_degen[1] = { 1 };
	const ct_long* dim_degen[] = { a0->dim_degen[0], site_dim_degen, a0->dim_degen[0] };

	allocate_su2_tensor(a0->dtype, 3, 0, &tree, outer_irreps, dim_degen, a_head);
	assert(su2_tensor_is_consistent(a_head));

	// set single entry of degeneracy tensor to 1
	assert(a_head->charge_sectors.nsec == 1);
	assert(dense_tensor_num_elements(a_head->degensors[0]) == 1);
	memcpy(a_head->degensors[0]->data, numeric_one(a_head->degensors[0]->dtype), sizeof_numeric_type(a_head->degensors[0]->dtype));
}


//________________________________________________________________________________________________________________________
///
/// \brief Create a dummy "tail" tensor adapted to the last tensor of an SU(2) symmetric MPS.
///
void su2_mps_create_dummy_tail_tensor(const struct su2_tensor* restrict al, struct su2_tensor* restrict a_tail)
{
	// right virtual bond of last tensor must have a single quantum number and degeneracy dimension 1
	assert(al->outer_irreps[2].num == 1);
	assert(al->dim_degen[2][al->outer_irreps[2].jlist[0]] == 1);

	// construct the fuse and split tree
	//
	//                       2  right virtual bond
	//                       │
	//                       │   fuse
	//                       ╱╲  split
	//                      ╱  ╲
	//  left virtual bond  0    1  physical axis
	//
	struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
	struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
	struct su2_tree_node j2f = { .i_ax = 2, .c = { NULL, NULL } };
	struct su2_tree_node j2s = { .i_ax = 2, .c = { &j0,  &j1  } };
	struct su2_fuse_split_tree tree = { .tree_fuse = &j2f, .tree_split = &j2s, .ndim = 3 };
	assert(su2_fuse_split_tree_is_consistent(&tree));

	// outer (logical and auxiliary) 'j' quantum numbers
	// quantum number zero for the dummy site axis
	qnumber jlist_zero[1] = { 0 };
	struct su2_irreducible_list site_irreps = {
		.jlist = jlist_zero,
		.num = 1,
	};
	const struct su2_irreducible_list outer_irreps[3] = { al->outer_irreps[2], site_irreps, al->outer_irreps[2] };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	const ct_long site_dim_degen[1] = { 1 };
	const ct_long* dim_degen[] = { al->dim_degen[2], site_dim_degen, al->dim_degen[2] };

	allocate_su2_tensor(al->dtype, 3, 0, &tree, outer_irreps, dim_degen, a_tail);
	assert(su2_tensor_is_consistent(a_tail));

	// set single entry of degeneracy tensor to 1
	assert(a_tail->charge_sectors.nsec == 1);
	assert(dense_tensor_num_elements(a_tail->degensors[0]) == 1);
	memcpy(a_tail->degensors[0]->data, numeric_one(a_tail->degensors[0]->dtype), sizeof_numeric_type(a_tail->degensors[0]->dtype));
}


//________________________________________________________________________________________________________________________
///
/// \brief Left- or right-orthonormalize an SU(2) symmetric MPS using QR decompositions, and return the normalization factor.
///
double su2_mps_orthonormalize_qr(struct su2_mps* mps, const enum su2_mps_orthonormalization_mode mode)
{
	assert(mps->nsites > 0);

	if (mode == SU2_MPS_ORTHONORMAL_LEFT)
	{
		for (int l = 0; l < mps->nsites - 1; l++)
		{
			su2_mps_local_orthonormalize_qr(&mps->a[l], &mps->a[l + 1]);
		}

		const int l = mps->nsites - 1;

		// create a dummy "tail" tensor
		struct su2_tensor a_tail;
		su2_mps_create_dummy_tail_tensor(&mps->a[l], &a_tail);

		// orthonormalize last MPS tensor
		su2_mps_local_orthonormalize_qr(&mps->a[l], &a_tail);

		// retrieve normalization factor (real-valued since diagonal of 'r' matrix is real)
		double norm = 0;
		assert(a_tail.charge_sectors.nsec == 1);
		assert(dense_tensor_num_elements(a_tail.degensors[0]) == 1);
		switch (a_tail.degensors[0]->dtype)
		{
			case CT_SINGLE_REAL:
			{
				norm = *((float*)a_tail.degensors[0]->data);
				break;
			}
			case CT_DOUBLE_REAL:
			{
				norm = *((double*)a_tail.degensors[0]->data);
				break;
			}
			case CT_SINGLE_COMPLEX:
			{
				norm = crealf(*((scomplex*)a_tail.degensors[0]->data));
				break;
			}
			case CT_DOUBLE_COMPLEX:
			{
				norm = creal(*((dcomplex*)a_tail.degensors[0]->data));
				break;
			}
			default:
			{
				// unknown data type
				assert(false);
			}
		}

		delete_su2_tensor(&a_tail);

		if (norm < 0)
		{
			// flip sign such that normalization factor is always non-negative
			rscale_su2_tensor(numeric_neg_one(numeric_real_type(mps->a[l].dtype)), &mps->a[l]);
			norm = -norm;
		}

		return norm;
	}
	else
	{
		assert(mode == SU2_MPS_ORTHONORMAL_RIGHT);

		for (int l = mps->nsites - 1; l > 0; l--)
		{
			su2_mps_local_orthonormalize_rq(&mps->a[l], &mps->a[l - 1]);
		}

		// create a dummy "head" tensor
		struct su2_tensor a_head;
		su2_mps_create_dummy_head_tensor(&mps->a[0], &a_head);

		// orthonormalize first MPS tensor
		su2_mps_local_orthonormalize_rq(&mps->a[0], &a_head);

		// retrieve normalization factor (real-valued since diagonal of 'r' matrix is real)
		double norm = 0;
		assert(a_head.charge_sectors.nsec == 1);
		assert(dense_tensor_num_elements(a_head.degensors[0]) == 1);
		switch (a_head.degensors[0]->dtype)
		{
			case CT_SINGLE_REAL:
			{
				norm = *((float*)a_head.degensors[0]->data);
				break;
			}
			case CT_DOUBLE_REAL:
			{
				norm = *((double*)a_head.degensors[0]->data);
				break;
			}
			case CT_SINGLE_COMPLEX:
			{
				norm = crealf(*((scomplex*)a_head.degensors[0]->data));
				break;
			}
			case CT_DOUBLE_COMPLEX:
			{
				norm = creal(*((dcomplex*)a_head.degensors[0]->data));
				break;
			}
			default:
			{
				// unknown data type
				assert(false);
			}
		}

		delete_su2_tensor(&a_head);

		if (norm < 0)
		{
			// flip sign such that normalization factor is always non-negative
			rscale_su2_tensor(numeric_neg_one(numeric_real_type(mps->a[0].dtype)), &mps->a[0]);
			norm = -norm;
		}

		return norm;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Contract two neighboring SU(2) MPS tensors along the virtual bond (without merging the physical axes).
///
void su2_mps_contract_tensor_pair(const struct su2_tensor* restrict a0, const struct su2_tensor* restrict a1, struct su2_tensor* restrict a)
{
	assert(a0->ndim_auxiliary == 0);
	assert(a1->ndim_auxiliary == 0);

	// combine a0 and a1 by contracting the shared bond;
	// allowing a0 and a1 to have several physical axes
	const int i_ax_cntr_0[1] = { a0->ndim_logical - 1 };
	const int i_ax_cntr_1[1] = { 0 };
	su2_tensor_contract_simple(a0, i_ax_cntr_0, a1, i_ax_cntr_1, 1, a);
}


//________________________________________________________________________________________________________________________
///
/// \brief Merge two neighboring SU(2) MPS tensors.
///
void su2_mps_merge_tensor_pair(const struct su2_tensor* restrict a0, const struct su2_tensor* restrict a1, struct su2_tensor* restrict a)
{
	struct su2_tensor a0_a1_dot;
	su2_mps_contract_tensor_pair(a0, a1, &a0_a1_dot);

	// combine original physical dimensions of a0 and a1 into one dimension
	su2_tensor_fuse_axes(&a0_a1_dot, 1, 2, a);
	delete_su2_tensor(&a0_a1_dot);
}


//________________________________________________________________________________________________________________________
///
/// \brief Contract all tensors of an SU(2) MPS to obtain the vector representation on the full Hilbert space.
/// All physical axes and the (dummy) virtual bonds are retained in the output tensor.
///
void su2_mps_to_statevector(const struct su2_mps* mps, struct su2_tensor* vec)
{
	assert(mps->nsites > 0);

	if (mps->nsites == 1)
	{
		copy_su2_tensor(&mps->a[0], vec);
	}
	else if (mps->nsites == 2)
	{
		su2_mps_contract_tensor_pair(&mps->a[0], &mps->a[1], vec);
	}
	else
	{
		struct su2_tensor t[2];
		su2_mps_contract_tensor_pair(&mps->a[0], &mps->a[1], &t[0]);
		for (int l = 2; l < mps->nsites; l++)
		{
			su2_mps_contract_tensor_pair(&t[l % 2], &mps->a[l], l < mps->nsites - 1 ? &t[(l + 1) % 2] : vec);
			delete_su2_tensor(&t[l % 2]);
		}
	}
}
