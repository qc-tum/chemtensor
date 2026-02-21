/// \file su2_tensor.c
/// \brief Data structures and functions for SU(2) symmetric tensors.

#include <math.h>
#include <assert.h>
#include "su2_tensor.h"
#include "su2_recoupling.h"
#include "su2_graph.h"
#include "su2_util.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Allocate an "empty" SU(2) symmetric tensor, without the charge sectors and dense "degeneracy" tensors.
///
void allocate_empty_su2_tensor(const enum numeric_type dtype, const int ndim_logical, int ndim_auxiliary, const struct su2_fuse_split_tree* tree, const struct su2_irreducible_list* outer_irreps, const ct_long** dim_degen, struct su2_tensor* t)
{
	t->dtype = dtype;

	assert(ndim_logical   >= 1);
	assert(ndim_auxiliary >= 0);
	t->ndim_logical   = ndim_logical;
	t->ndim_auxiliary = ndim_auxiliary;

	copy_su2_fuse_split_tree(tree, &t->tree);

	// irreducible 'j' quantum numbers on outer axes
	const int ndim_outer = ndim_logical + ndim_auxiliary;
	t->outer_irreps = ct_malloc(ndim_outer * sizeof(struct su2_irreducible_list));
	for (int i = 0; i < ndim_outer; i++) {
		copy_su2_irreducible_list(&outer_irreps[i], &t->outer_irreps[i]);
	}

	t->dim_degen = ct_malloc(t->ndim_logical * sizeof(ct_long*));
	for (int i = 0; i < t->ndim_logical; i++)
	{
		assert(t->outer_irreps[i].num > 0);
		qnumber j_max = 0;
		for (int k = 0; k < t->outer_irreps[i].num; k++) {
			j_max = qmax(j_max, t->outer_irreps[i].jlist[k]);
		}
		t->dim_degen[i] = ct_malloc((j_max + 1) * sizeof(ct_long));
		memcpy(t->dim_degen[i], dim_degen[i], (j_max + 1) * sizeof(ct_long));
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for an SU(2) symmetric tensor, including the dense "degeneracy" tensors, and enumerate all possible charge sectors.
///
void allocate_su2_tensor(const enum numeric_type dtype, const int ndim_logical, int ndim_auxiliary, const struct su2_fuse_split_tree* tree, const struct su2_irreducible_list* outer_irreps, const ct_long** dim_degen, struct su2_tensor* t)
{
	allocate_empty_su2_tensor(dtype, ndim_logical, ndim_auxiliary, tree, outer_irreps, dim_degen, t);

	su2_fuse_split_tree_enumerate_charge_sectors(&t->tree, t->outer_irreps, &t->charge_sectors);

	// degeneracy tensors
	t->degensors = ct_malloc(t->charge_sectors.nsec * sizeof(struct dense_tensor*));
	for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
	{
		// current 'j' quantum numbers
		const qnumber* jlist = &t->charge_sectors.jlists[c * t->charge_sectors.ndim];
		// dimension of degeneracy tensor
		ct_long* dim_d = ct_malloc(t->ndim_logical * sizeof(ct_long));
		for (int i = 0; i < t->ndim_logical; i++) {
			const qnumber j = jlist[i];
			assert(t->dim_degen[i][j] > 0);
			dim_d[i] = t->dim_degen[i][j];
		}
		t->degensors[c] = ct_calloc(1, sizeof(struct dense_tensor));
		allocate_zero_dense_tensor(t->dtype, t->ndim_logical, dim_d, t->degensors[c]);
		ct_free(dim_d);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for an SU(2) symmetric tensor of the same type, dimensions, fusion-splitting tree and quantum numbers as the provided tensor.
///
void allocate_su2_tensor_like(const struct su2_tensor* restrict s, struct su2_tensor* restrict t)
{
	allocate_su2_tensor(s->dtype, s->ndim_logical, s->ndim_auxiliary, &s->tree, s->outer_irreps, (const ct_long**)s->dim_degen, t);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an SU(2) symmetric tensor (free memory).
///
void delete_su2_tensor(struct su2_tensor* t)
{
	// degeneracy tensors
	for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
	{
		delete_dense_tensor(t->degensors[c]);
		ct_free(t->degensors[c]);
	}
	ct_free(t->degensors);
	t->degensors = NULL;
	for (int i = 0; i < t->ndim_logical; i++)
	{
		ct_free(t->dim_degen[i]);
	}
	ct_free(t->dim_degen);
	t->dim_degen = NULL;

	delete_charge_sectors(&t->charge_sectors);

	const int ndim_outer = t->ndim_logical + t->ndim_auxiliary;
	for (int i = 0; i < ndim_outer; i++) {
		delete_su2_irreducible_list(&t->outer_irreps[i]);
	}
	ct_free(t->outer_irreps);
	t->outer_irreps = NULL;

	delete_su2_fuse_split_tree(&t->tree);
}


//________________________________________________________________________________________________________________________
///
/// \brief Copy an SU(2) tensor, allocating memory for the copy.
///
void copy_su2_tensor(const struct su2_tensor* src, struct su2_tensor* dst)
{
	allocate_empty_su2_tensor(src->dtype, src->ndim_logical, src->ndim_auxiliary, &src->tree, src->outer_irreps, (const ct_long**)src->dim_degen, dst);

	copy_charge_sectors(&src->charge_sectors, &dst->charge_sectors);

	// degeneracy tensors
	dst->degensors = ct_malloc(dst->charge_sectors.nsec * sizeof(struct dense_tensor*));
	for (ct_long c = 0; c < dst->charge_sectors.nsec; c++)
	{
		dst->degensors[c] = ct_malloc(sizeof(struct dense_tensor));
		copy_dense_tensor(src->degensors[c], dst->degensors[c]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Logical dimension of axis 'i_ax' of SU(2) tensor 't'.
///
ct_long su2_tensor_dim_logical_axis(const struct su2_tensor* t, const int i_ax)
{
	assert(0 <= i_ax && i_ax < t->ndim_logical);

	ct_long d = 0;
	for (int k = 0; k < t->outer_irreps[i_ax].num; k++)
	{
		const qnumber j = t->outer_irreps[i_ax].jlist[k];
		assert(t->dim_degen[i_ax][j] > 0);
		d += t->dim_degen[i_ax][j] * (j + 1);
	}

	return d;
}


//________________________________________________________________________________________________________________________
///
/// \brief Calculate the logical number of elements of an SU(2) tensor.
///
/// Assuming that auxiliary axes have dimension 1.
///
ct_long su2_tensor_num_elements_logical(const struct su2_tensor* t)
{
	ct_long nelem = 1;
	for (int i = 0; i < t->ndim_logical; i++) {
		nelem *= su2_tensor_dim_logical_axis(t, i);
	}
	return nelem;
}


//________________________________________________________________________________________________________________________
///
/// \brief Logical direction of axis 'i_ax' of SU(2) tensor 't'.
///
enum tensor_axis_direction su2_tensor_logical_axis_direction(const struct su2_tensor* t, const int i_ax)
{
	assert(0 <= i_ax && i_ax < t->ndim_logical);

	if (su2_tree_contains_leaf(t->tree.tree_fuse, i_ax)) {
		return TENSOR_AXIS_IN;
	}
	else {
		assert(su2_tree_contains_leaf(t->tree.tree_split, i_ax));
		return TENSOR_AXIS_OUT;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Internal consistency check of an SU(2) symmetric tensor.
///
bool su2_tensor_is_consistent(const struct su2_tensor* t)
{
	if (!su2_fuse_split_tree_is_consistent(&t->tree)) {
		return false;
	}

	if (su2_tensor_ndim_internal(t) < 0) {
		return false;
	}
	if (t->tree.ndim != su2_tensor_ndim(t)) {
		return false;
	}
	if (t->charge_sectors.ndim != su2_tensor_ndim(t)) {
		return false;
	}

	// leaf nodes must correspond to logical and auxiliary axes
	for (int i = 0; i < t->ndim_logical + t->ndim_auxiliary; i++)
	{
		if (!(su2_tree_contains_leaf(t->tree.tree_fuse, i) || su2_tree_contains_leaf(t->tree.tree_split, i))) {
			return false;
		}
	}

	for (int i = 0; i < t->ndim_logical; i++)
	{
		if (t->outer_irreps[i].num <= 0) {
			return false;
		}

		for (int k = 0; k < t->outer_irreps[i].num; k++)
		{
			const qnumber j = t->outer_irreps[i].jlist[k];
			if (j < 0) {
				return false;
			}
			if (t->dim_degen[i][j] <= 0) {
				return false;
			}
		}
	}

	// charge sectors must be sorted
	for (ct_long c = 0; c < t->charge_sectors.nsec - 1; c++)
	{
		const struct su2_irreducible_list list0 = {
			.jlist = &t->charge_sectors.jlists[c * t->charge_sectors.ndim],
			.num   = t->charge_sectors.ndim,
		};
		const struct su2_irreducible_list list1 = {
			.jlist = &t->charge_sectors.jlists[(c + 1) * t->charge_sectors.ndim],
			.num   = t->charge_sectors.ndim,
		};
		if (compare_su2_irreducible_lists(&list0, &list1) >= 0) {
			return false;
		}
	}

	for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
	{
		if (t->degensors[c] == NULL) {
			return false;
		}
		if (t->degensors[c]->dtype != t->dtype) {
			return false;
		}
		if (t->degensors[c]->ndim != t->ndim_logical) {
			return false;
		}

		// current 'j' quantum numbers
		const qnumber* jlist = &t->charge_sectors.jlists[c * t->charge_sectors.ndim];

		if (!su2_fuse_split_tree_is_valid_charge_sector(&t->tree, jlist)) {
			return false;
		}

		// dimension of degeneracy tensor
		for (int i = 0; i < t->ndim_logical; i++)
		{
			const qnumber j = jlist[i];
			if (t->dim_degen[i][j] <= 0) {
				return false;
			}
			if (t->dim_degen[i][j] != t->degensors[c]->dim[i]) {
				return false;
			}
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the logical 2-norm (Frobenius norm) of the tensor.
///
/// Result is returned as double also for single-precision tensor entries.
///
double su2_tensor_norm2(const struct su2_tensor* t)
{
	const int i_ax_root = t->tree.tree_fuse->i_ax;
	assert(i_ax_root == t->tree.tree_split->i_ax);

	double nrm = 0;

	const int ndim = t->charge_sectors.ndim;
	#pragma omp parallel for schedule(dynamic) reduction(+: nrm)
	for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
	{
		// current 'j' quantum numbers
		const qnumber* jlist = &t->charge_sectors.jlists[c * ndim];

		nrm += (jlist[i_ax_root] + 1) * square(dense_tensor_norm2(t->degensors[c]));
	}
	nrm = sqrt(nrm);

	return nrm;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete the charge sector at 'idx' of the SU(2) symmetric tensor.
///
void su2_tensor_delete_charge_sector_by_index(struct su2_tensor* t, const ct_long idx)
{
	assert(0 <= idx && idx < t->charge_sectors.nsec);

	// delete corresponding "degeneracy" tensor
	delete_dense_tensor(t->degensors[idx]);
	ct_free(t->degensors[idx]);

	const int ndim = t->charge_sectors.ndim;
	for (ct_long c = idx; c < t->charge_sectors.nsec - 1; c++)
	{
		memcpy(&t->charge_sectors.jlists[c * ndim], &t->charge_sectors.jlists[(c + 1) * ndim], ndim * sizeof(qnumber));
		// copy pointer
		t->degensors[c] = t->degensors[c + 1];
	}
	t->degensors[t->charge_sectors.nsec - 1] = NULL;

	t->charge_sectors.nsec--;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an individual charge sector of the SU(2) symmetric tensor,
/// returning true if the charge sector was actually found and deleted.
///
bool su2_tensor_delete_charge_sector(struct su2_tensor* t, const qnumber* jlist)
{
	const ct_long idx = charge_sector_index(&t->charge_sectors, jlist);
	if (idx < 0) {
		// not found
		return false;
	}

	su2_tensor_delete_charge_sector_by_index(t, idx);

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Scale tensor 't' by 'alpha'.
///
/// Data types of all "degeneracy" tensors and of 'alpha' must match.
///
void scale_su2_tensor(const void* alpha, struct su2_tensor* t)
{
	#pragma omp parallel for schedule(dynamic)
	for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
	{
		scale_dense_tensor(alpha, t->degensors[c]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Scale tensor 't' by a real number 'alpha'.
///
/// Data type precision of all "degeneracy" tensors and of 'alpha' must match.
///
void rscale_su2_tensor(const void* alpha, struct su2_tensor* t)
{
	#pragma omp parallel for schedule(dynamic)
	for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
	{
		rscale_dense_tensor(alpha, t->degensors[c]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Elementwise complex conjugation of an SU(2) symmetric tensor.
///
/// Function has no effect if entries are real-valued.
///
void conjugate_su2_tensor(struct su2_tensor* t)
{
	if (is_real_numeric_type(t->dtype)) {
		// no effect
		return;
	}

	#pragma omp parallel for schedule(dynamic)
	for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
	{
		conjugate_dense_tensor(t->degensors[c]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Swap outer axes 'i_ax_0' and 'i_ax_1' in the fusion-splitting tree, assuming that they are direct neighbors,
/// such that the logical tensor remains the same.
///
void su2_tensor_swap_tree_axes(struct su2_tensor* t, const int i_ax_0, const int i_ax_1)
{
	assert(i_ax_0 != i_ax_1);
	assert(0 <= i_ax_0 && i_ax_0 < t->ndim_logical + t->ndim_auxiliary);
	assert(0 <= i_ax_1 && i_ax_1 < t->ndim_logical + t->ndim_auxiliary);

	const int i_ax[2] = { i_ax_0, i_ax_1 };
	struct su2_tree_node* node = (struct su2_tree_node*)su2_subtree_with_leaf_axes(t->tree.tree_fuse, i_ax, 2);
	if (node == NULL) {
		node = (struct su2_tree_node*)su2_subtree_with_leaf_axes(t->tree.tree_split, i_ax, 2);
	}
	assert(node != NULL);

	// absorb (-1) factors from R-symbols of SU(2) into degeneracy tensors
	// enumeration of charge sectors is not affected by swap
	const void* neg_one = numeric_neg_one(numeric_real_type(t->dtype));
	const int ndim = t->charge_sectors.ndim;
	#pragma omp parallel for schedule(dynamic)
	for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
	{
		// current 'j' quantum numbers
		const qnumber* jlist = &t->charge_sectors.jlists[c * ndim];
		const qnumber j_eff = jlist[i_ax_0] + jlist[i_ax_1] - jlist[node->i_ax];
		assert(j_eff % 2 == 0);
		// quantum numbers are stored times 2
		if ((j_eff / 2) % 2 == 1) {  // test whether R-symbol is -1
			rscale_dense_tensor(neg_one, t->degensors[c]);
		}
	}

	// actually swap axes indices in tree
	int tmp = node->c[0]->i_ax;
	node->c[0]->i_ax = node->c[1]->i_ax;
	node->c[1]->i_ax = tmp;
}


//________________________________________________________________________________________________________________________
///
/// \brief Insert a "dummy" auxiliary axis with quantum number zero at outer axis 'i_ax_insert'.
/// The index of the new axis is the previously largest auxiliary axis index + 1.
///
void su2_tensor_add_auxiliary_axis(struct su2_tensor* t, const int i_ax_insert, const bool insert_left)
{
	const int ndim = su2_tensor_ndim(t);
	const int ndim_outer = t->ndim_logical + t->ndim_auxiliary;

	// must be an outer axis
	assert(0 <= i_ax_insert && i_ax_insert < ndim_outer);

	// update tree
	{
		assert(t->tree.ndim == ndim);

		const bool in_fuse_tree = su2_tree_contains_leaf(t->tree.tree_fuse, i_ax_insert);
		struct su2_tree_node* node = (struct su2_tree_node*)su2_tree_find_node(in_fuse_tree ? t->tree.tree_fuse : t->tree.tree_split, i_ax_insert);
		assert(node != NULL);
		assert(su2_tree_node_is_leaf(node));

		// add two new nodes
		node->c[0] = ct_calloc(1, sizeof(struct su2_tree_node));
		node->c[1] = ct_calloc(1, sizeof(struct su2_tree_node));
		if (insert_left)
		{
			node->c[0]->i_ax = ndim + 1;  // will be mapped to new auxiliary axis
			node->c[1]->i_ax = ndim;      // will be mapped to hitherto 'i_ax_insert'
		}
		else
		{
			node->c[0]->i_ax = ndim;      // will be mapped to hitherto 'i_ax_insert'
			node->c[1]->i_ax = ndim + 1;  // will be mapped to new auxiliary axis
		}
		t->tree.ndim = ndim + 2;

		int* axis_map = ct_malloc(t->tree.ndim * sizeof(int));
		for (int i = 0; i < t->tree.ndim; i++) {
			axis_map[i] = i;
		}
		axis_map[i_ax_insert] = ndim + 1;
		for (int i = ndim_outer; i < ndim; i++) {
			axis_map[i] = i + 1;
		}
		axis_map[ndim] = i_ax_insert;
		axis_map[ndim + 1] = ndim_outer;  // new auxiliary axis index

		su2_fuse_split_tree_update_axes_indices(&t->tree, axis_map);

		ct_free(axis_map);
	}

	// update irreducible 'j' quantum numbers on outer axes
	{
		struct su2_irreducible_list* outer_irreps_new = ct_malloc((ndim_outer + 1) * sizeof(struct su2_irreducible_list));
		// copy internal data pointers
		memcpy(outer_irreps_new, t->outer_irreps, ndim_outer * sizeof(struct su2_irreducible_list));
		// set irreducible 'j' quantum number to zero for new auxiliary axis
		allocate_su2_irreducible_list(1, &outer_irreps_new[ndim_outer]);
		outer_irreps_new[ndim_outer].jlist[0] = 0;
		ct_free(t->outer_irreps);
		t->outer_irreps = outer_irreps_new;
	}

	// update charge sectors
	{
		assert(t->charge_sectors.ndim == ndim);

		struct charge_sectors charge_sectors_new;
		allocate_charge_sectors(t->charge_sectors.nsec, ndim + 2, &charge_sectors_new);

		for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
		{
			const qnumber* jlist = &t->charge_sectors.jlists[c * ndim];
			qnumber* jlist_new   = &charge_sectors_new.jlists[c * (ndim + 2)];
			// copy 'j' quantum numbers on outer axes
			memcpy(jlist_new, jlist, ndim_outer * sizeof(qnumber));
			jlist_new[ndim_outer] = 0;  // irreducible 'j' quantum number for new auxiliary axis is zero
			// copy internal 'j' quantum numbers
			for (int i = ndim_outer; i < ndim; i++) {
				jlist_new[i + 1] = jlist[i];
			}
			// new internal axis has same quantum number as outer axis 'i_ax_insert'
			jlist_new[ndim + 1] = jlist[i_ax_insert];
		}

		delete_charge_sectors(&t->charge_sectors);
		// copy internal data pointers
		t->charge_sectors = charge_sectors_new;
	}

	// degeneracy dimensions and tensors remain unchanged

	t->ndim_auxiliary++;
}


//________________________________________________________________________________________________________________________
///
/// \brief Fill the dense degeneracy tensors of an SU(2) symmetric tensor with random normal entries.
///
void su2_tensor_fill_random_normal(const void* alpha, const void* shift, struct rng_state* rng_state, struct su2_tensor* t)
{
	// not using OpenMP parallelization here due to random state
	for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
	{
		dense_tensor_fill_random_normal(alpha, shift, rng_state, t->degensors[c]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Charge sector and corresponding "degeneracy" tensor of an SU(2) symmetric tensor (temporary data structure).
///
struct su2_tensor_sector
{
	struct su2_irreducible_list list;  //!< irreducible 'j' quantum number configuration
	struct dense_tensor* degensor;     //!< corresponding dense "degeneracy" tensor
};


//________________________________________________________________________________________________________________________
///
/// \brief Comparison function for sorting.
///
static int compare_su2_tensor_sectors(const void* a, const void* b)
{
	const struct su2_tensor_sector* x = a;
	const struct su2_tensor_sector* y = b;

	return compare_su2_irreducible_lists(&x->list, &y->list);
}


//________________________________________________________________________________________________________________________
///
/// \brief Generalized transpose of a tensor 't' such that
/// the i-th axis in the output tensor 'r' is the perm[i]-th axis of the input tensor 't'.
///
/// 'perm' refers to all dimensions (logical, auxiliary and internal) and must not mix axes types.
///
void su2_tensor_transpose(const int* perm, const struct su2_tensor* restrict t, struct su2_tensor* restrict r)
{
	const int ndim = su2_tensor_ndim(t);
	const int ndim_outer = t->ndim_logical + t->ndim_auxiliary;

	// ensure that 'perm' is a valid permutation
	assert(is_permutation(perm, ndim));
	#ifndef NDEBUG
	// 'perm' must not mix axes types
	for (int i = 0; i < t->ndim_logical; i++)
	{
		assert(0 <= perm[i] && perm[i] < t->ndim_logical);
	}
	for (int i = t->ndim_logical; i < ndim_outer; i++)
	{
		assert(t->ndim_logical <= perm[i] && perm[i] < ndim_outer);
	}
	for (int i = ndim_outer; i < ndim; i++)
	{
		assert(ndim_outer <= perm[i] && perm[i] < ndim);
	}
	#endif

	if (is_identity_permutation(perm, ndim))
	{
		copy_su2_tensor(t, r);
		return;
	}

	r->dtype = t->dtype;
	r->ndim_logical   = t->ndim_logical;
	r->ndim_auxiliary = t->ndim_auxiliary;

	copy_su2_fuse_split_tree(&t->tree, &r->tree);
	int* axis_map = ct_malloc(ndim * sizeof(int));
	// inverse permutation
	for (int i = 0; i < ndim; i++) {
		axis_map[perm[i]] = i;
	}
	su2_fuse_split_tree_update_axes_indices(&r->tree, axis_map);
	ct_free(axis_map);

	// irreducible 'j' quantum numbers on outer axes
	r->outer_irreps = ct_malloc(ndim_outer * sizeof(struct su2_irreducible_list));
	for (int i = 0; i < ndim_outer; i++) {
		copy_su2_irreducible_list(&t->outer_irreps[perm[i]], &r->outer_irreps[i]);
	}

	r->dim_degen = ct_malloc(r->ndim_logical * sizeof(ct_long*));
	for (int i = 0; i < r->ndim_logical; i++)
	{
		assert(r->outer_irreps[i].num > 0);
		qnumber j_max = 0;
		for (int k = 0; k < r->outer_irreps[i].num; k++) {
			j_max = qmax(j_max, r->outer_irreps[i].jlist[k]);
		}
		r->dim_degen[i] = ct_malloc((j_max + 1) * sizeof(ct_long));
		memcpy(r->dim_degen[i], t->dim_degen[perm[i]], (j_max + 1) * sizeof(ct_long));
	}

	const ct_long nsec = t->charge_sectors.nsec;
	struct su2_tensor_sector* sectors_r = ct_malloc(nsec * sizeof(struct su2_tensor_sector));
	for (ct_long c = 0; c < nsec; c++)
	{
		// current 'j' quantum numbers
		const qnumber* jlist_t = &t->charge_sectors.jlists[c * ndim];

		allocate_su2_irreducible_list(ndim, &sectors_r[c].list);
		for (int i = 0; i < ndim; i++) {
			sectors_r[c].list.jlist[i] = jlist_t[perm[i]];
		}
		sectors_r[c].degensor = ct_malloc(sizeof(struct dense_tensor));
		// requiring that 'perm' is an endomorphism of the logical dimensions
		dense_tensor_transpose(perm, t->degensors[c], sectors_r[c].degensor);
	}

	// sort permuted charge sectors and tensors lexicographically
	qsort(sectors_r, nsec, sizeof(struct su2_tensor_sector), compare_su2_tensor_sectors);

	// copy data into output arrays
	allocate_charge_sectors(nsec, ndim, &r->charge_sectors);
	r->degensors = ct_malloc(nsec * sizeof(struct dense_tensor*));
	for (ct_long c = 0; c < nsec; c++)
	{
		memcpy(&r->charge_sectors.jlists[c * ndim], sectors_r[c].list.jlist, ndim * sizeof(qnumber));
		r->degensors[c] = sectors_r[c].degensor;  // copy pointer
	}

	// clean up
	for (ct_long c = 0; c < nsec; c++) {
		delete_su2_irreducible_list(&sectors_r[c].list);
	}
	ct_free(sectors_r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Generalized transpose of the logical axes of the SU(2) tensor 't' such that
/// the i-th axis in the output tensor 'r' is the perm[i]-th axis of the input tensor 't'.
///
/// The auxiliary and internal axes ordering remains unchanged.
///
void su2_tensor_transpose_logical(const int* perm, const struct su2_tensor* t, struct su2_tensor* r)
{
	// ensure that 'perm' is a valid permutation
	assert(is_permutation(perm, t->ndim_logical));

	// extend by an identity permutation on the auxiliary and internal axes
	const int ndim = su2_tensor_ndim(t);
	int* perm_full = ct_malloc(ndim * sizeof(int));
	memcpy(perm_full, perm, t->ndim_logical * sizeof(int));
	for (int i = t->ndim_logical; i < ndim; i++) {
		perm_full[i] = i;
	}

	su2_tensor_transpose(perm_full, t, r);

	ct_free(perm_full);
}


//________________________________________________________________________________________________________________________
///
/// \brief Perform an F-move on the internal axis 'i_ax'. 
///
void su2_tensor_fmove(const struct su2_tensor* restrict t, const int i_ax, struct su2_tensor* restrict r)
{
	// 'i_ax' must be an internal axis
	assert(t->ndim_logical + t->ndim_auxiliary <= i_ax && i_ax < su2_tensor_ndim(t));
	// 'i_ax' must not be the connecting axis of the fuse-split tree
	assert(i_ax != t->tree.tree_fuse->i_ax);

	struct su2_fuse_split_tree tree;
	copy_su2_fuse_split_tree(&t->tree, &tree);

	// find parent node of 'i_ax'
	struct su2_tree_node* parent = (struct su2_tree_node*)su2_tree_find_parent_node(tree.tree_fuse, i_ax);
	if (parent == NULL) {
		parent = (struct su2_tree_node*)su2_tree_find_parent_node(tree.tree_split, i_ax);
	}
	assert(parent != NULL);
	int i_ax_a, i_ax_b, i_ax_c;
	const int i_ax_s = parent->i_ax;
	bool orig_left_child;
	// determine axis indices of child nodes and perform F-move on tree
	if (parent->c[0]->i_ax == i_ax)  // left child
	{
		assert(!su2_tree_node_is_leaf(parent->c[0]));
		i_ax_a = parent->c[0]->c[0]->i_ax;
		i_ax_b = parent->c[0]->c[1]->i_ax;
		i_ax_c = parent->c[1]->i_ax;
		orig_left_child = true;
		su2_tree_fmove_right(parent);
		// right child after move
		assert(parent->c[1]->i_ax == i_ax);
	}
	else  // right child
	{
		assert(parent->c[1]->i_ax == i_ax);
		assert(!su2_tree_node_is_leaf(parent->c[1]));
		i_ax_a = parent->c[0]->i_ax;
		i_ax_b = parent->c[1]->c[0]->i_ax;
		i_ax_c = parent->c[1]->c[1]->i_ax;
		orig_left_child = false;
		su2_tree_fmove_left(parent);
		// left child after move
		assert(parent->c[0]->i_ax == i_ax);
	}

	// create new (empty) tensor
	allocate_empty_su2_tensor(t->dtype, t->ndim_logical, t->ndim_auxiliary, &tree, t->outer_irreps, (const ct_long**)t->dim_degen, r);

	const int ndim_r = su2_tensor_ndim(r);

	// all possible charge sectors
	su2_fuse_split_tree_enumerate_charge_sectors(&r->tree, r->outer_irreps, &r->charge_sectors);
	assert(r->charge_sectors.nsec > 0);
	assert(r->charge_sectors.ndim == ndim_r);
	// unused charge sectors will correspond to NULL pointers
	r->degensors = ct_calloc(r->charge_sectors.nsec, sizeof(struct dense_tensor*));

	delete_su2_fuse_split_tree(&tree);

	#pragma omp parallel for schedule(dynamic)
	for (ct_long cr = 0; cr < r->charge_sectors.nsec; cr++)
	{
		// 'j' quantum numbers of current sector
		const qnumber* jlist_r = &r->charge_sectors.jlists[cr * r->charge_sectors.ndim];

		const qnumber ja = jlist_r[i_ax_a];
		const qnumber jb = jlist_r[i_ax_b];
		const qnumber jc = jlist_r[i_ax_c];
		const qnumber js = jlist_r[i_ax_s];

		// TODO: binary search of matching sectors
		for (ct_long ct = 0; ct < t->charge_sectors.nsec; ct++)
		{
			// 'j' quantum numbers of current sector
			const qnumber* jlist_t = &t->charge_sectors.jlists[ct * t->charge_sectors.ndim];

			bool matching_sector = true;
			for (int i = 0; i < r->charge_sectors.ndim; i++) {
				if (i != i_ax && jlist_r[i] != jlist_t[i]) {
					matching_sector = false;
					break;
				}
			}
			if (!matching_sector) {
				continue;
			}

			const qnumber je = (orig_left_child ? jlist_t[i_ax] : jlist_r[i_ax]);
			const qnumber jf = (orig_left_child ? jlist_r[i_ax] : jlist_t[i_ax]);
			const double coeff = su2_recoupling_coefficient(ja, jb, jc, js, je, jf);
			if (coeff == 0) {
				continue;
			}

			if (r->degensors[cr] == NULL)
			{
				// allocate new degeneracy tensor
				r->degensors[cr] = ct_calloc(1, sizeof(struct dense_tensor));

				copy_dense_tensor(t->degensors[ct], r->degensors[cr]);
				assert(r->degensors[cr]->dtype == r->dtype);

				// ensure that 'alpha' is large enough to store any real numeric type
				double alpha;
				numeric_from_double(coeff, numeric_real_type(r->dtype), &alpha);
				rscale_dense_tensor(&alpha, r->degensors[cr]);
			}
			else
			{
				// ensure that 'alpha' is large enough to store any numeric type
				dcomplex alpha;
				numeric_from_double(coeff, r->dtype, &alpha);

				// accumulate degeneracy tensor from 't' weighted by 'coeff'
				dense_tensor_scalar_multiply_add(&alpha, t->degensors[ct], r->degensors[cr]);
			}
		}
	}

	// condense charge sector quantum numbers and corresponding degeneracy tensors
	// find first unused charge sector
	ct_long c = 0;
	for (; c < r->charge_sectors.nsec; c++) {
		if (r->degensors[c] == NULL) {
			break;
		}
	}
	for (ct_long s = c + 1; s < r->charge_sectors.nsec; s++)
	{
		if (r->degensors[s] != NULL)
		{
			memcpy(&r->charge_sectors.jlists[c * ndim_r], &r->charge_sectors.jlists[s * ndim_r], ndim_r * sizeof(qnumber));
			// copy pointer
			r->degensors[c] = r->degensors[s];
			r->degensors[s] = NULL;
			c++;
		}
	}
	r->charge_sectors.nsec = c;
}


//________________________________________________________________________________________________________________________
///
/// \brief Reverse the logical axis (tensor leg) 'i_ax', assuming that the SU(2) tensor is still described by a
/// fusion-splitting tree after the reversal and that no auxiliary axis needs to be introduced.
///
void su2_tensor_reverse_axis_simple(struct su2_tensor* t, const int i_ax)
{
	// must be an outer axis
	assert(0 <= i_ax && i_ax < t->ndim_logical + t->ndim_auxiliary);

	const bool in_fuse_tree = su2_tree_contains_leaf(t->tree.tree_fuse, i_ax);
	assert(in_fuse_tree != su2_tree_contains_leaf(t->tree.tree_split, i_ax));

	int i_ax_sibling;
	const int i_ax_root = t->tree.tree_fuse->i_ax;

	if (in_fuse_tree)
	{
		// 'i_ax' cannot be at the tree root, otherwise reversal would require an auxiliary axis
		assert(t->tree.tree_fuse->i_ax != i_ax);
		// 'i_ax' must be either a direct left or right child
		assert((t->tree.tree_fuse->c[0]->i_ax == i_ax) != (t->tree.tree_fuse->c[1]->i_ax == i_ax));

		const int m = (t->tree.tree_fuse->c[0]->i_ax == i_ax ? 0 : 1);

		struct su2_tree_node* node = t->tree.tree_fuse;
		assert(su2_tree_node_is_leaf(node->c[m]) && (node->c[m]->i_ax == i_ax));
		i_ax_sibling = node->c[1 - m]->i_ax;
		// rewire the tree
		t->tree.tree_fuse = node->c[1 - m];
		node->c[1 - m] = t->tree.tree_split;
		node->i_ax = t->tree.tree_fuse->i_ax;
		t->tree.tree_split = node;
	}
	else  // i_ax is in splitting tree
	{
		// 'i_ax' cannot be at the tree root, otherwise reversal would require an auxiliary axis
		assert(t->tree.tree_split->i_ax != i_ax);
		// 'i_ax' must be either a direct left or right child
		assert((t->tree.tree_split->c[0]->i_ax == i_ax) != (t->tree.tree_split->c[1]->i_ax == i_ax));

		const int m = (t->tree.tree_split->c[0]->i_ax == i_ax ? 0 : 1);

		struct su2_tree_node* node = t->tree.tree_split;
		assert(su2_tree_node_is_leaf(node->c[m]) && (node->c[m]->i_ax == i_ax));
		i_ax_sibling = node->c[1 - m]->i_ax;
		// rewire the tree
		t->tree.tree_split = node->c[1 - m];
		node->c[1 - m] = t->tree.tree_fuse;
		node->i_ax = t->tree.tree_split->i_ax;
		t->tree.tree_fuse = node;
	}

	assert(su2_fuse_split_tree_is_consistent(&t->tree));

	// reversal does not influence enumeration of charge sectors

	// scale degeneracy tensors
	#pragma omp parallel for schedule(dynamic)
	for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
	{
		// current 'j' quantum numbers
		const qnumber* jlist = &t->charge_sectors.jlists[c * t->charge_sectors.ndim];

		double coeff = sqrt(jlist[i_ax] + 1) *
			su2_recoupling_coefficient(jlist[i_ax], jlist[i_ax_sibling], jlist[i_ax_sibling], jlist[i_ax], jlist[i_ax_root], 0);
		if (in_fuse_tree) {
			coeff *= (1 - 2 * (jlist[i_ax] % 2));
		}

		// ensure that 'alpha' is large enough to store any real numeric type
		double alpha;
		numeric_from_double(coeff, numeric_real_type(t->degensors[c]->dtype), &alpha);

		rscale_dense_tensor(&alpha, t->degensors[c]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Fuse the two logical axes (tensor legs) 'i_ax_0' and 'i_ax_1' (with i_ax_0 < i_ax_1) into a single axis with index 'i_ax_0',
/// corresponding to the contraction with a "fusion" Clebsch-Gordan node.
/// The axes must have the same direction and must be direct siblings in the fusion-splitting tree.
/// The parent node of (i_ax_0, i_ax_1) in the fusion-splitting tree is deleted, and its (internal) upward axis index becomes 'i_ax_0'.
///
void su2_tensor_fuse_axes(const struct su2_tensor* restrict t, const int i_ax_0, const int i_ax_1, struct su2_tensor* restrict r)
{
	const int ndim_t = su2_tensor_ndim(t);
	const int ndim_outer_t = t->ndim_logical + t->ndim_auxiliary;

	assert(i_ax_0 < i_ax_1);
	// must be logical axes
	assert(0 <= i_ax_0 && i_ax_0 < t->ndim_logical);
	assert(0 <= i_ax_1 && i_ax_1 < t->ndim_logical);
	// resulting tensor must have at least 3 outer axes for a valid fusion-splitting tree
	assert(ndim_outer_t > 3);

	const bool in_fuse_tree = su2_tree_contains_leaf(t->tree.tree_fuse, i_ax_0);
	// axes must both be contained either in fusion or in splitting tree
	assert(in_fuse_tree == su2_tree_contains_leaf(t->tree.tree_fuse, i_ax_1));

	// parent node axis index
	int i_ax_p;
	{
		const struct su2_tree_node* node = su2_tree_find_parent_node(in_fuse_tree ? t->tree.tree_fuse : t->tree.tree_split, i_ax_0);
		assert(node != NULL);
		assert((node->c[0]->i_ax == i_ax_0 && node->c[1]->i_ax == i_ax_1) ||
		       (node->c[0]->i_ax == i_ax_1 && node->c[1]->i_ax == i_ax_0));
		i_ax_p = node->i_ax;
		assert(i_ax_p >= ndim_outer_t);  // must be an internal axis
	}

	int* axis_map = ct_malloc(ndim_t * sizeof(int));
	for (int i = 0; i < i_ax_0; i++) {
		axis_map[i] = i;
	}
	axis_map[i_ax_0] = -1;  // mark as invalid
	for (int i = i_ax_0 + 1; i < i_ax_1; i++) {
		axis_map[i] = i;
	}
	axis_map[i_ax_1] = -1;  // mark as invalid
	for (int i = i_ax_1 + 1; i < i_ax_p; i++) {
		axis_map[i] = i - 1;
	}
	// parent axis becomes logical axis 'i_ax_0'
	axis_map[i_ax_p] = i_ax_0;
	for (int i = i_ax_p + 1; i < ndim_t; i++) {
		axis_map[i] = i - 2;
	}

	qnumber j_max_0 = 0;
	for (int k = 0; k < t->outer_irreps[i_ax_0].num; k++) {
		j_max_0 = qmax(j_max_0, t->outer_irreps[i_ax_0].jlist[k]);
	}
	qnumber j_max_1 = 0;
	for (int k = 0; k < t->outer_irreps[i_ax_1].num; k++) {
		j_max_1 = qmax(j_max_1, t->outer_irreps[i_ax_1].jlist[k]);
	}
	// largest possible 'j' quantum number of the new fused axis
	qnumber j_max_fused = j_max_0 + j_max_1;

	// degeneracy tensor offset map for the fused axis
	ct_long* offset_map = ct_calloc((j_max_fused + 1) * (j_max_0 + 1) * (j_max_1 + 1), sizeof(ct_long));

	// allocate (empty) 'r' tensor
	{
		struct su2_fuse_split_tree tree_r;
		copy_su2_fuse_split_tree(&t->tree, &tree_r);
		su2_fuse_split_tree_update_axes_indices(&tree_r, axis_map);
		// 'i_ax_0' corresponds to parent node in original tree
		struct su2_tree_node* node = (struct su2_tree_node*)su2_tree_find_node(in_fuse_tree ? tree_r.tree_fuse : tree_r.tree_split, i_ax_0);
		assert(node != NULL);
		assert(!su2_tree_node_is_leaf(node));
		assert(node->c[0]->i_ax == -1);
		assert(node->c[1]->i_ax == -1);
		assert(su2_tree_node_is_leaf(node->c[0]));
		assert(su2_tree_node_is_leaf(node->c[1]));
		ct_free(node->c[0]);
		ct_free(node->c[1]);
		node->c[0] = NULL;
		node->c[1] = NULL;
		tree_r.ndim -= 2;
		assert(su2_fuse_split_tree_is_consistent(&tree_r));

		ct_long* dim_degen_fused = ct_calloc(j_max_fused + 1, sizeof(ct_long));
		for (int k = 0; k < t->outer_irreps[i_ax_0].num; k++)
		{
			const qnumber j0 = t->outer_irreps[i_ax_0].jlist[k];
			assert(j0 <= j_max_0);
			assert(t->dim_degen[i_ax_0][j0] > 0);

			for (int l = 0; l < t->outer_irreps[i_ax_1].num; l++)
			{
				const qnumber j1 = t->outer_irreps[i_ax_1].jlist[l];
				assert(j1 <= j_max_1);
				assert(t->dim_degen[i_ax_1][j1] > 0);

				for (qnumber jf = abs(j0 - j1); jf <= j0 + j1; jf += 2)
				{
					// set 'offset_map' entry
					offset_map[(jf*(j_max_0 + 1) + j0)*(j_max_1 + 1) + j1] = dim_degen_fused[jf];
					// product of degeneracy dimensions
					dim_degen_fused[jf] += t->dim_degen[i_ax_0][j0] * t->dim_degen[i_ax_1][j1];
				}
			}
		}

		struct su2_irreducible_list outer_irreps_r_fused = {
			.jlist = ct_malloc((j_max_fused + 1) * sizeof(qnumber)),
			.num = 0,
		};
		for (qnumber j = 0; j <= j_max_fused; j++) {
			if (dim_degen_fused[j] > 0) {
				outer_irreps_r_fused.jlist[outer_irreps_r_fused.num] = j;
				outer_irreps_r_fused.num++;
			}
		}
		assert(outer_irreps_r_fused.num > 0);
		struct su2_irreducible_list* outer_irreps_r = ct_malloc((ndim_outer_t - 1) * sizeof(struct su2_irreducible_list));
		for (int i = 0; i < ndim_outer_t; i++) {
			if (axis_map[i] != -1) {
				outer_irreps_r[axis_map[i]] = t->outer_irreps[i];
			}
		}
		outer_irreps_r[i_ax_0] = outer_irreps_r_fused;

		const ct_long** dim_degen_r = ct_malloc((t->ndim_logical - 1) * sizeof(ct_long*));
		for (int i = 0; i < t->ndim_logical; i++)
		{
			if (axis_map[i] == -1) {
				continue;
			}
			assert(0 <= axis_map[i] && axis_map[i] < t->ndim_logical - 1);
			assert(outer_irreps_r[axis_map[i]].num > 0);
			// simply copy the pointer
			dim_degen_r[axis_map[i]] = t->dim_degen[i];
		}
		dim_degen_r[i_ax_0] = dim_degen_fused;

		allocate_empty_su2_tensor(t->dtype, t->ndim_logical - 1, t->ndim_auxiliary, &tree_r, outer_irreps_r, dim_degen_r, r);

		ct_free(dim_degen_r);

		delete_su2_irreducible_list(&outer_irreps_r_fused);
		ct_free(outer_irreps_r);
		ct_free(dim_degen_fused);

		delete_su2_fuse_split_tree(&tree_r);
	}

	const int ndim_r = su2_tensor_ndim(r);

	// all possible charge sectors
	su2_fuse_split_tree_enumerate_charge_sectors(&r->tree, r->outer_irreps, &r->charge_sectors);
	assert(r->charge_sectors.nsec > 0);
	assert(r->charge_sectors.ndim == ndim_r);
	// unused charge sectors will correspond to NULL pointers
	r->degensors = ct_calloc(r->charge_sectors.nsec, sizeof(struct dense_tensor*));

	// permutations of dense degeneracy tensors before axes flattening
	int* perm = ct_malloc(t->ndim_logical * sizeof(int));
	for (int i = 0; i <= i_ax_0; i++) {
		perm[i] = i;
	}
	perm[i_ax_0 + 1] = i_ax_1;
	for (int i = i_ax_0 + 2; i <= i_ax_1; i++) {
		perm[i] = i - 1;
	}
	for (int i = i_ax_1 + 1; i < t->ndim_logical; i++) {
		perm[i] = i;
	}
	const bool perm_is_identity = (i_ax_1 == i_ax_0 + 1);
	assert(perm_is_identity == is_identity_permutation(perm, t->ndim_logical));

	const size_t dtype_size = sizeof_numeric_type(t->dtype);

	qnumber* jlist_r = ct_malloc(ndim_r * sizeof(qnumber));
	for (ct_long ct = 0; ct < t->charge_sectors.nsec; ct++)
	{
		// current 'j' quantum numbers
		const qnumber* jlist_t = &t->charge_sectors.jlists[ct * t->charge_sectors.ndim];

		for (int i = 0; i < ndim_t; i++) {
			if (axis_map[i] != -1) {
				jlist_r[axis_map[i]] = jlist_t[i];
			}
		}

		const ct_long cr = charge_sector_index(&r->charge_sectors, jlist_r);
		assert(cr != -1);

		const qnumber j0 = jlist_t[i_ax_0];
		const qnumber j1 = jlist_t[i_ax_1];
		const qnumber jp = jlist_t[i_ax_p];
		assert(j0 <= j_max_0);
		assert(j1 <= j_max_1);
		assert((abs(j0 - j1) <= jp) && (jp <= j0 + j1));
		assert((j0 + j1 + jp) % 2 == 0);

		const ct_long offset = offset_map[(jp*(j_max_0 + 1) + j0)*(j_max_1 + 1) + j1];

		// corresponding "degeneracy" tensor of 't'
		const struct dense_tensor* dt = t->degensors[ct];
		assert(dt != NULL);
		assert(dt->dtype == t->dtype);
		assert(dt->ndim  == t->ndim_logical);
		struct dense_tensor dt_perm;
		if (perm_is_identity) {
			dt_perm = *dt;  // only copy pointers
		}
		else {
			dense_tensor_transpose(perm, dt, &dt_perm);
		}

		// corresponding "degeneracy" tensor of 'r'
		if (r->degensors[cr] == NULL)
		{
			// allocate new degeneracy tensor with zero entries
			// dimension of degeneracy tensor
			ct_long* dim_d = ct_malloc(r->ndim_logical * sizeof(ct_long));
			for (int i = 0; i < r->ndim_logical; i++) {
				const qnumber j = jlist_r[i];
				assert(r->dim_degen[i][j] > 0);
				dim_d[i] = r->dim_degen[i][j];
			}
			r->degensors[cr] = ct_calloc(1, sizeof(struct dense_tensor));
			allocate_zero_dense_tensor(r->dtype, r->ndim_logical, dim_d, r->degensors[cr]);
			ct_free(dim_d);
		}
		struct dense_tensor* dr = r->degensors[cr];
		assert(dr->dtype == r->dtype);
		assert(dr->ndim  == r->ndim_logical);

		// copy tensor entries
		const ct_long n = integer_product(dr->dim, i_ax_0);
		assert(n == integer_product(dt_perm.dim, i_ax_0));
		// trailing dimensions times data type size
		const ct_long tdd_r = integer_product(dr->dim + (i_ax_0 + 1), dr->ndim - (i_ax_0 + 1)) * dtype_size;
		const ct_long offset_r = offset*tdd_r;
		const ct_long stride_t = integer_product(dt_perm.dim + i_ax_0, dt_perm.ndim - i_ax_0) * dtype_size;
		const ct_long stride_r = dr->dim[i_ax_0] * tdd_r;
		for (ct_long i = 0; i < n; i++)
		{
			// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
			memcpy((int8_t*)dr->data + (i*stride_r + offset_r),
			       (int8_t*)dt_perm.data + i*stride_t,
			       stride_t);
		}

		if (!perm_is_identity) {
			delete_dense_tensor(&dt_perm);
		}
	}

	ct_free(jlist_r);
	ct_free(perm);
	ct_free(offset_map);
	ct_free(axis_map);

	// condense charge sector quantum numbers and corresponding degeneracy tensors
	// find first unused charge sector
	ct_long c = 0;
	for (; c < r->charge_sectors.nsec; c++) {
		if (r->degensors[c] == NULL) {
			break;
		}
	}
	for (ct_long s = c + 1; s < r->charge_sectors.nsec; s++)
	{
		if (r->degensors[s] != NULL)
		{
			memcpy(&r->charge_sectors.jlists[c * ndim_r], &r->charge_sectors.jlists[s * ndim_r], ndim_r * sizeof(qnumber));
			// copy pointer
			r->degensors[c] = r->degensors[s];
			r->degensors[s] = NULL;
			c++;
		}
	}
	r->charge_sectors.nsec = c;
}


//________________________________________________________________________________________________________________________
///
/// \brief Fuse the two logical axes (tensor legs) 'i_ax_0' and 'i_ax_1' (with i_ax_0 < i_ax_1) into a single axis with index 'i_ax_0',
/// corresponding to the contraction with a "fusion" Clebsch-Gordan node, and add a dummy auxiliary axis with quantum number zero.
/// The axes must have the same direction and must be direct siblings in the fusion-splitting tree.
/// The fusion-splitting tree topology and overall number of axes remain unchanged, since the hitherto 'i_ax_1' becomes the new first auxiliary axis.
///
void su2_tensor_fuse_axes_add_auxiliary(const struct su2_tensor* restrict t, const int i_ax_0, const int i_ax_1, struct su2_tensor* restrict r)
{
	// same for 't' and 'r'
	const int ndim = su2_tensor_ndim(t);
	const int ndim_outer = t->ndim_logical + t->ndim_auxiliary;

	assert(i_ax_0 < i_ax_1);
	// must be logical axes
	assert(0 <= i_ax_0 && i_ax_0 < t->ndim_logical);
	assert(0 <= i_ax_1 && i_ax_1 < t->ndim_logical);
	// resulting tensor must have at least 3 outer axes for a valid fusion-splitting tree
	assert(ndim_outer >= 3);

	const bool in_fuse_tree = su2_tree_contains_leaf(t->tree.tree_fuse, i_ax_0);
	// axes must both be contained either in fusion or in splitting tree
	assert(in_fuse_tree == su2_tree_contains_leaf(t->tree.tree_fuse, i_ax_1));

	// parent node axis index
	int i_ax_p;
	{
		const struct su2_tree_node* node = su2_tree_find_parent_node(in_fuse_tree ? t->tree.tree_fuse : t->tree.tree_split, i_ax_0);
		assert(node != NULL);
		assert((node->c[0]->i_ax == i_ax_0 && node->c[1]->i_ax == i_ax_1) ||
		       (node->c[0]->i_ax == i_ax_1 && node->c[1]->i_ax == i_ax_0));
		i_ax_p = node->i_ax;
	}

	int* axis_map = ct_malloc(ndim * sizeof(int));
	for (int i = 0; i < i_ax_1; i++) {
		axis_map[i] = i;
	}
	// 'i_ax_1' becomes first auxiliary axis in 'r'
	axis_map[i_ax_1] = t->ndim_logical - 1;
	for (int i = i_ax_1 + 1; i < t->ndim_logical; i++) {
		axis_map[i] = i - 1;
	}
	for (int i = t->ndim_logical; i < ndim; i++) {
		axis_map[i] = i;
	}
	assert(is_permutation(axis_map, ndim));

	qnumber j_max_0 = 0;
	for (int k = 0; k < t->outer_irreps[i_ax_0].num; k++) {
		j_max_0 = qmax(j_max_0, t->outer_irreps[i_ax_0].jlist[k]);
	}
	qnumber j_max_1 = 0;
	for (int k = 0; k < t->outer_irreps[i_ax_1].num; k++) {
		j_max_1 = qmax(j_max_1, t->outer_irreps[i_ax_1].jlist[k]);
	}
	// largest possible 'j' quantum number of the new fused axis
	qnumber j_max_fused = j_max_0 + j_max_1;

	// degeneracy tensor offset map for the fused axis
	ct_long* offset_map = ct_calloc((j_max_fused + 1) * (j_max_0 + 1) * (j_max_1 + 1), sizeof(ct_long));

	// allocate (empty) 'r' tensor
	{
		struct su2_fuse_split_tree tree_r;
		copy_su2_fuse_split_tree(&t->tree, &tree_r);
		su2_fuse_split_tree_update_axes_indices(&tree_r, axis_map);
		#ifndef NDEBUG
		struct su2_tree_node* node = (struct su2_tree_node*)su2_tree_find_node(in_fuse_tree ? tree_r.tree_fuse : tree_r.tree_split, axis_map[i_ax_p]);
		assert(node != NULL);
		assert((node->c[0]->i_ax == axis_map[i_ax_0] && node->c[1]->i_ax == axis_map[i_ax_1]) ||
		       (node->c[0]->i_ax == axis_map[i_ax_1] && node->c[1]->i_ax == axis_map[i_ax_0]));
		assert(su2_fuse_split_tree_is_consistent(&tree_r));
		#endif

		ct_long* dim_degen_fused = ct_calloc(j_max_fused + 1, sizeof(ct_long));
		for (int k = 0; k < t->outer_irreps[i_ax_0].num; k++)
		{
			const qnumber j0 = t->outer_irreps[i_ax_0].jlist[k];
			assert(j0 <= j_max_0);
			assert(t->dim_degen[i_ax_0][j0] > 0);

			for (int l = 0; l < t->outer_irreps[i_ax_1].num; l++)
			{
				const qnumber j1 = t->outer_irreps[i_ax_1].jlist[l];
				assert(j1 <= j_max_1);
				assert(t->dim_degen[i_ax_1][j1] > 0);

				for (qnumber jf = abs(j0 - j1); jf <= j0 + j1; jf += 2)
				{
					// set 'offset_map' entry
					offset_map[(jf*(j_max_0 + 1) + j0)*(j_max_1 + 1) + j1] = dim_degen_fused[jf];
					// product of degeneracy dimensions
					dim_degen_fused[jf] += t->dim_degen[i_ax_0][j0] * t->dim_degen[i_ax_1][j1];
				}
			}
		}

		struct su2_irreducible_list outer_irreps_r_fused = {
			.jlist = ct_malloc((j_max_fused + 1) * sizeof(qnumber)),
			.num = 0,
		};
		for (qnumber j = 0; j <= j_max_fused; j++) {
			if (dim_degen_fused[j] > 0) {
				outer_irreps_r_fused.jlist[outer_irreps_r_fused.num] = j;
				outer_irreps_r_fused.num++;
			}
		}
		assert(outer_irreps_r_fused.num > 0);
		qnumber jlist_zero[1] = { 0 };
		struct su2_irreducible_list outer_irreps_r_auxiliary = {
			.jlist = jlist_zero,
			.num = 1,
		};
		struct su2_irreducible_list* outer_irreps_r = ct_malloc(ndim_outer * sizeof(struct su2_irreducible_list));
		for (int i = 0; i < ndim_outer; i++) {
			outer_irreps_r[axis_map[i]] = t->outer_irreps[i];
		}
		outer_irreps_r[i_ax_0] = outer_irreps_r_fused;
		outer_irreps_r[t->ndim_logical - 1] = outer_irreps_r_auxiliary;

		const ct_long** dim_degen_r = ct_malloc((t->ndim_logical - 1) * sizeof(ct_long*));
		for (int i = 0; i < t->ndim_logical; i++)
		{
			if (i == i_ax_1) {
				continue;
			}
			assert(0 <= axis_map[i] && axis_map[i] < t->ndim_logical - 1);
			assert(outer_irreps_r[axis_map[i]].num > 0);
			// simply copy the pointer
			dim_degen_r[axis_map[i]] = t->dim_degen[i];
		}
		dim_degen_r[i_ax_0] = dim_degen_fused;

		allocate_empty_su2_tensor(t->dtype, t->ndim_logical - 1, t->ndim_auxiliary + 1, &tree_r, outer_irreps_r, dim_degen_r, r);

		ct_free(dim_degen_r);

		delete_su2_irreducible_list(&outer_irreps_r_fused);
		ct_free(outer_irreps_r);
		ct_free(dim_degen_fused);

		delete_su2_fuse_split_tree(&tree_r);
	}

	// all possible charge sectors
	su2_fuse_split_tree_enumerate_charge_sectors(&r->tree, r->outer_irreps, &r->charge_sectors);
	assert(r->charge_sectors.nsec > 0);
	assert(r->charge_sectors.ndim == ndim);
	// unused charge sectors will correspond to NULL pointers
	r->degensors = ct_calloc(r->charge_sectors.nsec, sizeof(struct dense_tensor*));

	// permutations of dense degeneracy tensors before axes flattening
	int* perm = ct_malloc(t->ndim_logical * sizeof(int));
	for (int i = 0; i <= i_ax_0; i++) {
		perm[i] = i;
	}
	perm[i_ax_0 + 1] = i_ax_1;
	for (int i = i_ax_0 + 2; i <= i_ax_1; i++) {
		perm[i] = i - 1;
	}
	for (int i = i_ax_1 + 1; i < t->ndim_logical; i++) {
		perm[i] = i;
	}
	const bool perm_is_identity = (i_ax_1 == i_ax_0 + 1);
	assert(perm_is_identity == is_identity_permutation(perm, t->ndim_logical));

	const size_t dtype_size = sizeof_numeric_type(t->dtype);

	qnumber* jlist_r = ct_malloc(ndim * sizeof(qnumber));
	for (ct_long ct = 0; ct < t->charge_sectors.nsec; ct++)
	{
		// current 'j' quantum numbers
		const qnumber* jlist_t = &t->charge_sectors.jlists[ct * t->charge_sectors.ndim];

		for (int i = 0; i < ndim; i++) {
			jlist_r[axis_map[i]] = jlist_t[i];
		}
		// fused axis inherits quantum number from parent since auxiliary quantum number is zero
		jlist_r[i_ax_0] = jlist_t[i_ax_p];
		// quantum number of new auxiliary axis is zero
		jlist_r[t->ndim_logical - 1] = 0;

		const ct_long cr = charge_sector_index(&r->charge_sectors, jlist_r);
		assert(cr != -1);

		const qnumber j0 = jlist_t[i_ax_0];
		const qnumber j1 = jlist_t[i_ax_1];
		const qnumber jp = jlist_t[i_ax_p];
		assert(j0 <= j_max_0);
		assert(j1 <= j_max_1);
		assert((abs(j0 - j1) <= jp) && (jp <= j0 + j1));
		assert((j0 + j1 + jp) % 2 == 0);

		const ct_long offset = offset_map[(jp*(j_max_0 + 1) + j0)*(j_max_1 + 1) + j1];

		// corresponding "degeneracy" tensor of 't'
		const struct dense_tensor* dt = t->degensors[ct];
		assert(dt != NULL);
		assert(dt->dtype == t->dtype);
		assert(dt->ndim  == t->ndim_logical);
		struct dense_tensor dt_perm;
		if (perm_is_identity) {
			dt_perm = *dt;  // only copy pointers
		}
		else {
			dense_tensor_transpose(perm, dt, &dt_perm);
		}

		// corresponding "degeneracy" tensor of 'r'
		if (r->degensors[cr] == NULL)
		{
			// allocate new degeneracy tensor with zero entries
			// dimension of degeneracy tensor
			ct_long* dim_d = ct_malloc(r->ndim_logical * sizeof(ct_long));
			for (int i = 0; i < r->ndim_logical; i++) {
				const qnumber j = jlist_r[i];
				assert(r->dim_degen[i][j] > 0);
				dim_d[i] = r->dim_degen[i][j];
			}
			r->degensors[cr] = ct_calloc(1, sizeof(struct dense_tensor));
			allocate_zero_dense_tensor(r->dtype, r->ndim_logical, dim_d, r->degensors[cr]);
			ct_free(dim_d);
		}
		struct dense_tensor* dr = r->degensors[cr];
		assert(dr->dtype == r->dtype);
		assert(dr->ndim  == r->ndim_logical);

		// copy tensor entries
		const ct_long n = integer_product(dr->dim, i_ax_0);
		assert(n == integer_product(dt_perm.dim, i_ax_0));
		// trailing dimensions times data type size
		const ct_long tdd_r = integer_product(dr->dim + (i_ax_0 + 1), dr->ndim - (i_ax_0 + 1)) * dtype_size;
		const ct_long offset_r = offset*tdd_r;
		const ct_long stride_t = integer_product(dt_perm.dim + i_ax_0, dt_perm.ndim - i_ax_0) * dtype_size;
		const ct_long stride_r = dr->dim[i_ax_0] * tdd_r;
		for (ct_long i = 0; i < n; i++)
		{
			// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
			memcpy((int8_t*)dr->data + (i*stride_r + offset_r),
			       (int8_t*)dt_perm.data + i*stride_t,
			       stride_t);
		}

		if (!perm_is_identity) {
			delete_dense_tensor(&dt_perm);
		}
	}

	ct_free(jlist_r);
	ct_free(perm);
	ct_free(offset_map);
	ct_free(axis_map);

	// condense charge sector quantum numbers and corresponding degeneracy tensors
	// find first unused charge sector
	ct_long c = 0;
	for (; c < r->charge_sectors.nsec; c++) {
		if (r->degensors[c] == NULL) {
			break;
		}
	}
	for (ct_long s = c + 1; s < r->charge_sectors.nsec; s++)
	{
		if (r->degensors[s] != NULL)
		{
			memcpy(&r->charge_sectors.jlists[c * ndim], &r->charge_sectors.jlists[s * ndim], ndim * sizeof(qnumber));
			// copy pointer
			r->degensors[c] = r->degensors[s];
			r->degensors[s] = NULL;
			c++;
		}
	}
	r->charge_sectors.nsec = c;
}


//________________________________________________________________________________________________________________________
///
/// \brief Split the logical axis (tensor leg) 'i_ax_split' into two logical axes, 'i_ax_split' and 'i_ax_add' (with i_ax_split < i_ax_add),
/// corresponding to attaching a "splitting" Clebsch-Gordan node.
/// 'tree_left_child' indicates whether the new axis 'i_ax_split' is the left child in the fusion-splitting tree.
/// The arguments 'outer_irreps' and 'dim_degen' specify the 'j' quantum numbers and degeneracies of the new axes 'i_ax_split' and 'i_ax_add'.
///
/// The axis 'i_ax_split' in the original fusion-splitting tree becomes the last inner axis of the new tensor.
///
void su2_tensor_split_axis(const struct su2_tensor* restrict t, const int i_ax_split, const int i_ax_add, const bool tree_left_child, const struct su2_irreducible_list outer_irreps[2], const ct_long* dim_degen[2], struct su2_tensor* restrict r)
{
	assert(0 <= i_ax_split && i_ax_split < t->ndim_logical);
	assert(i_ax_split < i_ax_add);
	// must be a logical axis of the new tensor
	assert(i_ax_add < t->ndim_logical + 1);

	const bool in_fuse_tree = su2_tree_contains_leaf(t->tree.tree_fuse, i_ax_split);

	const int ndim_t = su2_tensor_ndim(t);
	const int ndim_outer_t = t->ndim_logical + t->ndim_auxiliary;

	int* axis_map = ct_malloc((ndim_t + 2) * sizeof(int));
	for (int i = 0; i < i_ax_split; i++) {
		axis_map[i] = i;
	}
	// last axis index of new tensor
	axis_map[i_ax_split] = ndim_t + 1;
	for (int i = i_ax_split + 1; i < i_ax_add; i++) {
		axis_map[i] = i;
	}
	for (int i = i_ax_add; i < ndim_t; i++) {
		axis_map[i] = i + 1;
	}
	axis_map[ndim_t]     = i_ax_split;
	axis_map[ndim_t + 1] = i_ax_add;

	qnumber j_max_0 = 0;
	for (int k = 0; k < outer_irreps[0].num; k++) {
		j_max_0 = qmax(j_max_0, outer_irreps[0].jlist[k]);
	}
	qnumber j_max_1 = 0;
	for (int k = 0; k < outer_irreps[1].num; k++) {
		j_max_1 = qmax(j_max_1, outer_irreps[1].jlist[k]);
	}
	qnumber j_max_split = 0;
	for (int k = 0; k < t->outer_irreps[i_ax_split].num; k++) {
		j_max_split = qmax(j_max_split, t->outer_irreps[i_ax_split].jlist[k]);
	}
	// must be compatible
	assert(j_max_split == j_max_0 + j_max_1);

	// degeneracy tensor offset map for the split axis
	ct_long* offset_map = ct_calloc((j_max_split + 1) * (j_max_0 + 1) * (j_max_1 + 1), sizeof(ct_long));
	ct_long* dim_degen_split = ct_calloc(j_max_split + 1, sizeof(ct_long));
	for (int k = 0; k < outer_irreps[0].num; k++)
	{
		const qnumber j0 = outer_irreps[0].jlist[k];
		assert(j0 <= j_max_0);
		assert(dim_degen[0][j0] > 0);

		for (int l = 0; l < outer_irreps[1].num; l++)
		{
			const qnumber j1 = outer_irreps[1].jlist[l];
			assert(j1 <= j_max_1);
			assert(dim_degen[1][j1] > 0);

			for (qnumber js = abs(j0 - j1); js <= j0 + j1; js += 2)
			{
				// set 'offset_map' entry
				offset_map[(js*(j_max_0 + 1) + j0)*(j_max_1 + 1) + j1] = dim_degen_split[js];
				// product of degeneracy dimensions
				dim_degen_split[js] += dim_degen[0][j0] * dim_degen[1][j1];
			}
		}
	}
	// check consistency of quantum numbers and degeneracies
	#ifndef NDEBUG
	for (qnumber js = 0; js <= j_max_split; js++) {
		assert(dim_degen_split[js] == t->dim_degen[i_ax_split][js]);
	}
	#endif
	ct_free(dim_degen_split);

	// allocate (empty) 'r' tensor
	{
		struct su2_fuse_split_tree tree_r;
		copy_su2_fuse_split_tree(&t->tree, &tree_r);
		struct su2_tree_node* node = (struct su2_tree_node*)su2_tree_find_node(in_fuse_tree ? tree_r.tree_fuse : tree_r.tree_split, i_ax_split);
		assert(node != NULL);
		assert(su2_tree_node_is_leaf(node));
		node->c[0] = ct_calloc(1, sizeof(struct su2_tree_node));
		node->c[1] = ct_calloc(1, sizeof(struct su2_tree_node));
		if (tree_left_child)
		{
			node->c[0]->i_ax = ndim_t;
			node->c[1]->i_ax = ndim_t + 1;
		}
		else
		{
			node->c[0]->i_ax = ndim_t + 1;
			node->c[1]->i_ax = ndim_t;
		}
		tree_r.ndim += 2;
		su2_fuse_split_tree_update_axes_indices(&tree_r, axis_map);
		assert(su2_fuse_split_tree_is_consistent(&tree_r));

		struct su2_irreducible_list* outer_irreps_r = ct_malloc((ndim_outer_t + 1) * sizeof(struct su2_irreducible_list));
		for (int i = 0; i < ndim_outer_t; i++) {
			if (i == i_ax_split) {
				continue;
			}
			assert(axis_map[i] != i_ax_add);
			outer_irreps_r[axis_map[i]] = t->outer_irreps[i];
		}
		outer_irreps_r[i_ax_split] = outer_irreps[0];
		outer_irreps_r[i_ax_add]   = outer_irreps[1];

		const ct_long** dim_degen_r = ct_malloc((t->ndim_logical + 1) * sizeof(ct_long*));
		for (int i = 0; i < t->ndim_logical; i++)
		{
			if (i == i_ax_split) {
				continue;
			}
			assert(axis_map[i] != i_ax_add);
			assert(outer_irreps_r[axis_map[i]].num > 0);
			// simply copy the pointer
			dim_degen_r[axis_map[i]] = t->dim_degen[i];
		}
		dim_degen_r[i_ax_split] = dim_degen[0];
		dim_degen_r[i_ax_add]   = dim_degen[1];

		allocate_empty_su2_tensor(t->dtype, t->ndim_logical + 1, t->ndim_auxiliary, &tree_r, outer_irreps_r, dim_degen_r, r);

		ct_free(dim_degen_r);
		ct_free(outer_irreps_r);

		delete_su2_fuse_split_tree(&tree_r);
	}

	const int ndim_r = su2_tensor_ndim(r);
	assert(ndim_r == ndim_t + 2);

	// all possible charge sectors
	su2_fuse_split_tree_enumerate_charge_sectors(&r->tree, r->outer_irreps, &r->charge_sectors);
	assert(r->charge_sectors.nsec > 0);
	assert(r->charge_sectors.ndim == ndim_r);
	// unused charge sectors will correspond to NULL pointers
	r->degensors = ct_calloc(r->charge_sectors.nsec, sizeof(struct dense_tensor*));

	// permutations of dense degeneracy tensors after axes splitting
	int* perm = ct_malloc(r->ndim_logical * sizeof(int));
	for (int i = 0; i <= i_ax_split; i++) {
		perm[i] = i;
	}
	for (int i = i_ax_split + 1; i < i_ax_add; i++) {
		perm[i] = i + 1;
	}
	perm[i_ax_add] = i_ax_split + 1;
	for (int i = i_ax_add + 1; i < r->ndim_logical; i++) {
		perm[i] = i;
	}
	const bool perm_is_identity = (i_ax_add == i_ax_split + 1);
	assert(is_permutation(perm, r->ndim_logical));
	assert(perm_is_identity == is_identity_permutation(perm, r->ndim_logical));

	const size_t dtype_size = sizeof_numeric_type(t->dtype);

	qnumber* jlist_t = ct_malloc(ndim_t * sizeof(qnumber));
	for (ct_long cr = 0; cr < r->charge_sectors.nsec; cr++)
	{
		// current 'j' quantum numbers
		const qnumber* jlist_r = &r->charge_sectors.jlists[cr * r->charge_sectors.ndim];

		for (int i = 0; i < ndim_t; i++)
		{
			assert(axis_map[i] != i_ax_split);
			assert(axis_map[i] != i_ax_add);
			jlist_t[i] = jlist_r[axis_map[i]];
		}

		const ct_long ct = charge_sector_index(&t->charge_sectors, jlist_t);
		if (ct == -1) {
			continue;
		}

		const qnumber j0 = jlist_r[i_ax_split];
		const qnumber j1 = jlist_r[i_ax_add];
		const qnumber js = jlist_r[ndim_t + 1];  // last axis index of new tensor 'r' corresponds to 'i_ax_split' in 't'
		assert(j0 <= j_max_0);
		assert(j1 <= j_max_1);
		assert((abs(j0 - j1) <= js) && (js <= j0 + j1));
		assert((j0 + j1 + js) % 2 == 0);

		const ct_long offset = offset_map[(js*(j_max_0 + 1) + j0)*(j_max_1 + 1) + j1];

		// corresponding "degeneracy" tensor of 't'
		const struct dense_tensor* dt = t->degensors[ct];
		assert(dt != NULL);
		assert(dt->dtype == t->dtype);
		assert(dt->ndim  == t->ndim_logical);

		// corresponding (permuted) "degeneracy" tensor of 'r'
		struct dense_tensor dr_perm;
		{
			// dimension of permuted degeneracy tensor
			ct_long* dim_d = ct_malloc(r->ndim_logical * sizeof(ct_long));
			for (int i = 0; i < r->ndim_logical; i++) {
				const qnumber j = jlist_r[i];
				assert(r->dim_degen[i][j] > 0);
				dim_d[perm[i]] = r->dim_degen[i][j];
			}
			allocate_zero_dense_tensor(r->dtype, r->ndim_logical, dim_d, &dr_perm);
			ct_free(dim_d);
		}

		// copy tensor entries
		const ct_long n = integer_product(dt->dim, i_ax_split);
		assert(n == integer_product(dr_perm.dim, i_ax_split));
		// trailing dimensions times data type size
		const ct_long tdd_t = integer_product(dt->dim + (i_ax_split + 1), dt->ndim - (i_ax_split + 1)) * dtype_size;
		const ct_long offset_t = offset*tdd_t;
		const ct_long stride_t = dt->dim[i_ax_split] * tdd_t;
		const ct_long stride_r = integer_product(dr_perm.dim + i_ax_split, dr_perm.ndim - i_ax_split) * dtype_size;
		for (ct_long i = 0; i < n; i++)
		{
			// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
			memcpy((int8_t*)dr_perm.data + i*stride_r,
			       (int8_t*)dt->data + (i*stride_t + offset_t),
			       stride_r);
		}

		r->degensors[cr] = ct_malloc(sizeof(struct dense_tensor));
		if (perm_is_identity)
		{
			// copy internal data pointers
			*r->degensors[cr] = dr_perm;
		}
		else
		{
			dense_tensor_transpose(perm, &dr_perm, r->degensors[cr]);
			delete_dense_tensor(&dr_perm);
		}
	}

	ct_free(jlist_t);
	ct_free(perm);
	ct_free(offset_map);
	ct_free(axis_map);

	// condense charge sector quantum numbers and corresponding degeneracy tensors
	// find first unused charge sector
	ct_long c = 0;
	for (; c < r->charge_sectors.nsec; c++) {
		if (r->degensors[c] == NULL) {
			break;
		}
	}
	for (ct_long s = c + 1; s < r->charge_sectors.nsec; s++)
	{
		if (r->degensors[s] != NULL)
		{
			memcpy(&r->charge_sectors.jlists[c * ndim_r], &r->charge_sectors.jlists[s * ndim_r], ndim_r * sizeof(qnumber));
			// copy pointer
			r->degensors[c] = r->degensors[s];
			r->degensors[s] = NULL;
			c++;
		}
	}
	r->charge_sectors.nsec = c;
}


//________________________________________________________________________________________________________________________
///
/// \brief Split the logical axis (tensor leg) 'i_ax_split' into two logical axes, 'i_ax_split' and 'i_ax_add' (with i_ax_split < i_ax_add),
/// corresponding to attaching a "splitting" Clebsch-Gordan node.
/// The sibling axis of 'i_ax_split', which must be a dummy auxiliary axis with quantum number zero, is replaced by 'i_ax_add' in the fusion-splitting tree.
/// The arguments 'outer_irreps' and 'dim_degen' specify the 'j' quantum numbers and degeneracies of the new axes 'i_ax_split' and 'i_ax_add'.
/// The fusion-splitting tree topology and overall number of axes remain unchanged.
///
void su2_tensor_split_axis_remove_auxiliary(const struct su2_tensor* restrict t, const int i_ax_split, const int i_ax_add, const struct su2_irreducible_list outer_irreps[2], const ct_long* dim_degen[2], struct su2_tensor* restrict r)
{
	assert(0 <= i_ax_split && i_ax_split < t->ndim_logical);
	assert(i_ax_split < i_ax_add);
	// must be a logical axis of the new tensor
	assert(i_ax_add < t->ndim_logical + 1);

	const bool in_fuse_tree = su2_tree_contains_leaf(t->tree.tree_fuse, i_ax_split);

	// same for 't' and 'r'
	const int ndim = su2_tensor_ndim(t);
	const int ndim_outer = t->ndim_logical + t->ndim_auxiliary;

	// parent node and auxiliary axis index
	int i_ax_p, i_ax_aux;
	{
		const struct su2_tree_node* node = su2_tree_find_parent_node(in_fuse_tree ? t->tree.tree_fuse : t->tree.tree_split, i_ax_split);
		assert(node != NULL);
		i_ax_p = node->i_ax;
		i_ax_aux = (node->c[0]->i_ax == i_ax_split ? node->c[1]->i_ax : node->c[0]->i_ax);
	}
	// must be a "trivial" (solely quantum number zero) auxiliary axis
	assert(t->ndim_logical <= i_ax_aux && i_ax_aux < ndim_outer);
	assert(t->outer_irreps[i_ax_aux].num == 1);
	assert(t->outer_irreps[i_ax_aux].jlist[0] == 0);

	int* axis_map = ct_malloc(ndim * sizeof(int));
	for (int i = 0; i < i_ax_add; i++) {
		axis_map[i] = i;
	}
	for (int i = i_ax_add; i < i_ax_aux; i++) {
		axis_map[i] = i + 1;
	}
	// 'i_ax_aux' becomes added axis in 'r'
	axis_map[i_ax_aux] = i_ax_add;
	for (int i = i_ax_aux + 1; i < ndim; i++) {
		axis_map[i] = i;
	}
	assert(is_permutation(axis_map, ndim));

	qnumber j_max_0 = 0;
	for (int k = 0; k < outer_irreps[0].num; k++) {
		j_max_0 = qmax(j_max_0, outer_irreps[0].jlist[k]);
	}
	qnumber j_max_1 = 0;
	for (int k = 0; k < outer_irreps[1].num; k++) {
		j_max_1 = qmax(j_max_1, outer_irreps[1].jlist[k]);
	}
	qnumber j_max_split = 0;
	for (int k = 0; k < t->outer_irreps[i_ax_split].num; k++) {
		j_max_split = qmax(j_max_split, t->outer_irreps[i_ax_split].jlist[k]);
	}
	// must be compatible
	assert(j_max_split == j_max_0 + j_max_1);

	// degeneracy tensor offset map for the split axis
	ct_long* offset_map = ct_calloc((j_max_split + 1) * (j_max_0 + 1) * (j_max_1 + 1), sizeof(ct_long));
	ct_long* dim_degen_split = ct_calloc(j_max_split + 1, sizeof(ct_long));
	for (int k = 0; k < outer_irreps[0].num; k++)
	{
		const qnumber j0 = outer_irreps[0].jlist[k];
		assert(j0 <= j_max_0);
		assert(dim_degen[0][j0] > 0);

		for (int l = 0; l < outer_irreps[1].num; l++)
		{
			const qnumber j1 = outer_irreps[1].jlist[l];
			assert(j1 <= j_max_1);
			assert(dim_degen[1][j1] > 0);

			for (qnumber js = abs(j0 - j1); js <= j0 + j1; js += 2)
			{
				// set 'offset_map' entry
				offset_map[(js*(j_max_0 + 1) + j0)*(j_max_1 + 1) + j1] = dim_degen_split[js];
				// product of degeneracy dimensions
				dim_degen_split[js] += dim_degen[0][j0] * dim_degen[1][j1];
			}
		}
	}
	// check consistency of quantum numbers and degeneracies
	#ifndef NDEBUG
	for (qnumber js = 0; js <= j_max_split; js++) {
		assert(dim_degen_split[js] == t->dim_degen[i_ax_split][js]);
	}
	#endif
	ct_free(dim_degen_split);

	// allocate (empty) 'r' tensor
	{
		struct su2_fuse_split_tree tree_r;
		copy_su2_fuse_split_tree(&t->tree, &tree_r);
		su2_fuse_split_tree_update_axes_indices(&tree_r, axis_map);
		assert(su2_fuse_split_tree_is_consistent(&tree_r));

		struct su2_irreducible_list* outer_irreps_r = ct_malloc(ndim_outer * sizeof(struct su2_irreducible_list));
		for (int i = 0; i < ndim_outer; i++) {
			outer_irreps_r[axis_map[i]] = t->outer_irreps[i];
		}
		outer_irreps_r[i_ax_split] = outer_irreps[0];
		outer_irreps_r[i_ax_add]   = outer_irreps[1];

		const ct_long** dim_degen_r = ct_malloc((t->ndim_logical + 1) * sizeof(ct_long*));
		for (int i = 0; i < t->ndim_logical; i++)
		{
			assert(axis_map[i] < t->ndim_logical + 1);
			assert(axis_map[i] != i_ax_add);
			// simply copy the pointer
			dim_degen_r[axis_map[i]] = t->dim_degen[i];
		}
		dim_degen_r[i_ax_split] = dim_degen[0];
		dim_degen_r[i_ax_add]   = dim_degen[1];

		allocate_empty_su2_tensor(t->dtype, t->ndim_logical + 1, t->ndim_auxiliary - 1, &tree_r, outer_irreps_r, dim_degen_r, r);

		ct_free(dim_degen_r);
		ct_free(outer_irreps_r);

		delete_su2_fuse_split_tree(&tree_r);
	}

	// all possible charge sectors
	su2_fuse_split_tree_enumerate_charge_sectors(&r->tree, r->outer_irreps, &r->charge_sectors);
	assert(r->charge_sectors.nsec > 0);
	assert(r->charge_sectors.ndim == ndim);
	// unused charge sectors will correspond to NULL pointers
	r->degensors = ct_calloc(r->charge_sectors.nsec, sizeof(struct dense_tensor*));

	// permutations of dense degeneracy tensors after axes splitting
	int* perm = ct_malloc(r->ndim_logical * sizeof(int));
	for (int i = 0; i <= i_ax_split; i++) {
		perm[i] = i;
	}
	for (int i = i_ax_split + 1; i < i_ax_add; i++) {
		perm[i] = i + 1;
	}
	perm[i_ax_add] = i_ax_split + 1;
	for (int i = i_ax_add + 1; i < r->ndim_logical; i++) {
		perm[i] = i;
	}
	const bool perm_is_identity = (i_ax_add == i_ax_split + 1);
	assert(is_permutation(perm, r->ndim_logical));
	assert(perm_is_identity == is_identity_permutation(perm, r->ndim_logical));

	const size_t dtype_size = sizeof_numeric_type(t->dtype);

	qnumber* jlist_t = ct_malloc(ndim * sizeof(qnumber));
	for (ct_long cr = 0; cr < r->charge_sectors.nsec; cr++)
	{
		// current 'j' quantum numbers
		const qnumber* jlist_r = &r->charge_sectors.jlists[cr * r->charge_sectors.ndim];

		for (int i = 0; i < ndim; i++) {
			jlist_t[i] = jlist_r[axis_map[i]];
		}
		// 'i_ax_split' in 't' inherits quantum number from parent since auxiliary quantum number is zero
		jlist_t[i_ax_split] = jlist_r[axis_map[i_ax_p]];
		// quantum number of auxiliary axis is zero
		jlist_t[i_ax_aux] = 0;

		const ct_long ct = charge_sector_index(&t->charge_sectors, jlist_t);
		if (ct == -1) {
			continue;
		}

		const qnumber j0 = jlist_r[i_ax_split];
		const qnumber j1 = jlist_r[i_ax_add];
		const qnumber jp = jlist_r[axis_map[i_ax_p]];
		assert(j0 <= j_max_0);
		assert(j1 <= j_max_1);
		assert((abs(j0 - j1) <= jp) && (jp <= j0 + j1));
		assert((j0 + j1 + jp) % 2 == 0);

		const ct_long offset = offset_map[(jp*(j_max_0 + 1) + j0)*(j_max_1 + 1) + j1];

		// corresponding "degeneracy" tensor of 't'
		const struct dense_tensor* dt = t->degensors[ct];
		assert(dt != NULL);
		assert(dt->dtype == t->dtype);
		assert(dt->ndim  == t->ndim_logical);

		// corresponding (permuted) "degeneracy" tensor of 'r'
		struct dense_tensor dr_perm;
		{
			// dimension of permuted degeneracy tensor
			ct_long* dim_d = ct_malloc(r->ndim_logical * sizeof(ct_long));
			for (int i = 0; i < r->ndim_logical; i++) {
				const qnumber j = jlist_r[i];
				assert(r->dim_degen[i][j] > 0);
				dim_d[perm[i]] = r->dim_degen[i][j];
			}
			allocate_zero_dense_tensor(r->dtype, r->ndim_logical, dim_d, &dr_perm);
			ct_free(dim_d);
		}

		// copy tensor entries
		const ct_long n = integer_product(dt->dim, i_ax_split);
		assert(n == integer_product(dr_perm.dim, i_ax_split));
		// trailing dimensions times data type size
		const ct_long tdd_t = integer_product(dt->dim + (i_ax_split + 1), dt->ndim - (i_ax_split + 1)) * dtype_size;
		const ct_long offset_t = offset*tdd_t;
		const ct_long stride_t = dt->dim[i_ax_split] * tdd_t;
		const ct_long stride_r = integer_product(dr_perm.dim + i_ax_split, dr_perm.ndim - i_ax_split) * dtype_size;
		for (ct_long i = 0; i < n; i++)
		{
			// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
			memcpy((int8_t*)dr_perm.data + i*stride_r,
			       (int8_t*)dt->data + (i*stride_t + offset_t),
			       stride_r);
		}

		r->degensors[cr] = ct_malloc(sizeof(struct dense_tensor));
		if (perm_is_identity)
		{
			// copy internal data pointers
			*r->degensors[cr] = dr_perm;
		}
		else
		{
			dense_tensor_transpose(perm, &dr_perm, r->degensors[cr]);
			delete_dense_tensor(&dr_perm);
		}
	}

	ct_free(jlist_t);
	ct_free(perm);
	ct_free(offset_map);
	ct_free(axis_map);

	// condense charge sector quantum numbers and corresponding degeneracy tensors
	// find first unused charge sector
	ct_long c = 0;
	for (; c < r->charge_sectors.nsec; c++) {
		if (r->degensors[c] == NULL) {
			break;
		}
	}
	for (ct_long s = c + 1; s < r->charge_sectors.nsec; s++)
	{
		if (r->degensors[s] != NULL)
		{
			memcpy(&r->charge_sectors.jlists[c * ndim], &r->charge_sectors.jlists[s * ndim], ndim * sizeof(qnumber));
			// copy pointer
			r->degensors[c] = r->degensors[s];
			r->degensors[s] = NULL;
			c++;
		}
	}
	r->charge_sectors.nsec = c;
}


//________________________________________________________________________________________________________________________
///
/// \brief Slice a logical axis of the tensor by selecting "degeneracy" indices 'ind' along this axis.
///
/// Indices 'ind' cannot be duplicate and must be sorted.
/// 'j' quantum numbers along 'i_ax' for which the new degeneracy dimension would be zero are removed.
/// Memory will be allocated for 'r'.
///
void su2_tensor_slice(const struct su2_tensor* restrict t, const int i_ax, const ct_long* ind, const ct_long nind, struct su2_tensor* restrict r)
{
	assert(0 <= i_ax && i_ax < t->ndim_logical);
	assert(nind > 0);

	// maximum 'j' quantum number along axis 'i_ax'
	qnumber j_max = 0;
	for (int k = 0; k < t->outer_irreps[i_ax].num; k++)
	{
		const qnumber j = t->outer_irreps[i_ax].jlist[k];
		j_max = qmax(j_max, j);
	}

	// selected indices for each 'j' quantum number along 'i_ax'
	ct_long** ind_sectors     = ct_calloc(j_max + 1, sizeof(ct_long*));
	ct_long* dim_degen_r_i_ax = ct_calloc(j_max + 1, sizeof(ct_long));
	ct_long d = 0;
	ct_long i = 0;
	for (int k = 0; k < t->outer_irreps[i_ax].num; k++)
	{
		const qnumber j = t->outer_irreps[i_ax].jlist[k];
		const ct_long dim_degen = t->dim_degen[i_ax][j];
		assert(dim_degen > 0);
		ind_sectors[j] = ct_malloc(dim_degen * sizeof(ct_long));  // upper bound on required memory
		for (; i < nind; i++)
		{
			assert(i == 0 || ind[i - 1] < ind[i]);
			if (ind[i] - d >= dim_degen) {
				break;
			}
			ind_sectors[j][dim_degen_r_i_ax[j]] = ind[i] - d;
			dim_degen_r_i_ax[j]++;
		}
		assert(dim_degen_r_i_ax[j] <= dim_degen);

		d += dim_degen;
	}
	assert(i == nind);

	// allocate (empty) 'r' tensor
	{
		const int ndim_outer = t->ndim_logical + t->ndim_auxiliary;

		struct su2_irreducible_list outer_irreps_r_i_ax = {
			.jlist = ct_malloc(t->outer_irreps[i_ax].num * sizeof(qnumber)),  // upper bound on required memory
			.num = 0,
		};
		for (int k = 0; k < t->outer_irreps[i_ax].num; k++)
		{
			const qnumber j = t->outer_irreps[i_ax].jlist[k];
			assert(j <= j_max);

			if (dim_degen_r_i_ax[j] > 0)
			{
				outer_irreps_r_i_ax.jlist[outer_irreps_r_i_ax.num] = j;
				outer_irreps_r_i_ax.num++;
			}
		}
		assert(outer_irreps_r_i_ax.num > 0);

		struct su2_irreducible_list* outer_irreps_r = ct_malloc(ndim_outer * sizeof(struct su2_irreducible_list));
		for (int i = 0; i < ndim_outer; i++) {
			outer_irreps_r[i] = t->outer_irreps[i];
		}
		outer_irreps_r[i_ax] = outer_irreps_r_i_ax;

		const ct_long** dim_degen_r = ct_malloc(t->ndim_logical * sizeof(ct_long*));
		for (int i = 0; i < t->ndim_logical; i++) {
			// simply copy the pointer
			dim_degen_r[i] = t->dim_degen[i];
		}
		dim_degen_r[i_ax] = dim_degen_r_i_ax;

		allocate_empty_su2_tensor(t->dtype, t->ndim_logical, t->ndim_auxiliary, &t->tree, outer_irreps_r, (const ct_long**)dim_degen_r, r);

		ct_free(dim_degen_r);
		ct_free(outer_irreps_r);
		delete_su2_irreducible_list(&outer_irreps_r_i_ax);

		// all possible charge sectors
		su2_fuse_split_tree_enumerate_charge_sectors(&r->tree, r->outer_irreps, &r->charge_sectors);
		assert(r->charge_sectors.nsec > 0);
		// unused charge sectors will correspond to NULL pointers
		r->degensors = ct_calloc(r->charge_sectors.nsec, sizeof(struct dense_tensor*));
	}

	for (ct_long cr = 0; cr < r->charge_sectors.nsec; cr++)
	{
		// current 'j' quantum numbers
		const qnumber* jlist = &r->charge_sectors.jlists[cr * r->charge_sectors.ndim];

		const ct_long ct = charge_sector_index(&t->charge_sectors, jlist);
		if (ct == -1) {
			continue;
		}

		// corresponding "degeneracy" tensor of 't'
		const struct dense_tensor* dt = t->degensors[ct];
		assert(dt != NULL);

		// corresponding "degeneracy" tensor of 'r'
		assert(r->degensors[cr] == NULL);
		r->degensors[cr] = ct_calloc(1, sizeof(struct dense_tensor));

		const qnumber j = jlist[i_ax];
		assert(r->dim_degen[i_ax][j] == dim_degen_r_i_ax[j]);
		dense_tensor_slice(dt, i_ax, ind_sectors[j], r->dim_degen[i_ax][j], r->degensors[cr]);
	}

	for (int k = 0; k < t->outer_irreps[i_ax].num; k++)
	{
		const qnumber j = t->outer_irreps[i_ax].jlist[k];
		ct_free(ind_sectors[j]);
	}
	ct_free(ind_sectors);
	ct_free(dim_degen_r_i_ax);

	// condense charge sector quantum numbers and corresponding degeneracy tensors
	// find first unused charge sector
	ct_long c = 0;
	for (; c < r->charge_sectors.nsec; c++) {
		if (r->degensors[c] == NULL) {
			break;
		}
	}
	const int ndim = r->charge_sectors.ndim;
	assert(ndim == su2_tensor_ndim(r));
	for (ct_long s = c + 1; s < r->charge_sectors.nsec; s++)
	{
		if (r->degensors[s] != NULL)
		{
			memcpy(&r->charge_sectors.jlists[c * ndim], &r->charge_sectors.jlists[s * ndim], ndim * sizeof(qnumber));
			// copy pointer
			r->degensors[c] = r->degensors[s];
			r->degensors[s] = NULL;
			c++;
		}
	}
	r->charge_sectors.nsec = c;
}


//________________________________________________________________________________________________________________________
///
/// \brief Contraction mode of two SU(2) symmetric tensors.
///
enum su2_tensor_contraction_mode
{
	SU2_TENSOR_CONTRACT_FUSE_SPLIT = 0,  //!< contract fusion tree of first tensor with splitting tree of second tensor
	SU2_TENSOR_CONTRACT_SPLIT_FUSE = 1,  //!< contract splitting tree of first tensor with fusion tree of second tensor
};


//________________________________________________________________________________________________________________________
///
/// \brief Contract two SU(2) tensors for the scenario that the to-be contracted axes form subtrees with matching topology and
/// that the resulting fusion-splitting tree is again simple.
///
void su2_tensor_contract_simple(const struct su2_tensor* restrict s, const int* restrict i_ax_s, const struct su2_tensor* restrict t, const int* restrict i_ax_t, const int ndim_mult, struct su2_tensor* restrict r)
{
	assert(s->dtype == t->dtype);

	// dimension and quantum number compatibility checks
	assert(ndim_mult >= 1);
	#ifndef NDEBUG
	for (int i = 0; i < ndim_mult; i++)
	{
		assert(0 <= i_ax_s[i] && i_ax_s[i] < s->ndim_logical + s->ndim_auxiliary);
		assert(0 <= i_ax_t[i] && i_ax_t[i] < t->ndim_logical + t->ndim_auxiliary);

		// must both be logical or both be auxiliary axes
		assert((i_ax_s[i] < s->ndim_logical && i_ax_t[i] < t->ndim_logical) || (i_ax_s[i] >= s->ndim_logical && i_ax_t[i] >= t->ndim_logical));

		const struct su2_irreducible_list* irred_s = &s->outer_irreps[i_ax_s[i]];
		const struct su2_irreducible_list* irred_t = &t->outer_irreps[i_ax_t[i]];
		assert(su2_irreducible_list_equal(irred_s, irred_t));
		if (i_ax_s[i] < s->ndim_logical) {
			for (int k = 0; k < irred_s->num; k++) {
				// degeneracy dimensions for to-be contracted axes must match
				assert(s->dim_degen[i_ax_s[i]][irred_s->jlist[k]] ==
				       t->dim_degen[i_ax_t[i]][irred_t->jlist[k]]);
			}
		}
	}
	#endif

	const int ndim_outer_s = s->ndim_logical + s->ndim_auxiliary;
	const int ndim_outer_t = t->ndim_logical + t->ndim_auxiliary;

	const struct su2_tree_node* subtree_s;
	const struct su2_tree_node* subtree_t;
	enum su2_tensor_contraction_mode mode;
	subtree_s = su2_subtree_with_leaf_axes(s->tree.tree_fuse, i_ax_s, ndim_mult);
	if (subtree_s != NULL)
	{
		subtree_t = su2_subtree_with_leaf_axes(t->tree.tree_split, i_ax_t, ndim_mult);
		mode = SU2_TENSOR_CONTRACT_FUSE_SPLIT;
	}
	else
	{
		subtree_s = su2_subtree_with_leaf_axes(s->tree.tree_split, i_ax_s, ndim_mult);
		subtree_t = su2_subtree_with_leaf_axes(t->tree.tree_fuse,  i_ax_t, ndim_mult);
		mode = SU2_TENSOR_CONTRACT_SPLIT_FUSE;
	}
	assert(subtree_s != NULL);
	assert(subtree_t != NULL);
	assert(su2_tree_equal_topology(subtree_s, subtree_t));
	// at least one of the subtrees must actually be the full tree, otherwise the resulting fusion-splitting tree is not simple
	assert((subtree_s == (mode == SU2_TENSOR_CONTRACT_FUSE_SPLIT ? s->tree.tree_fuse  : s->tree.tree_split))
	    || (subtree_t == (mode == SU2_TENSOR_CONTRACT_FUSE_SPLIT ? t->tree.tree_split : t->tree.tree_fuse)));

	const int ndim_s = su2_tensor_ndim(s);
	const int ndim_t = su2_tensor_ndim(t);

	int* i_ax_subtree_s = ct_malloc(ndim_s * sizeof(int));  // upper bound on required memory
	int* i_ax_subtree_t = ct_malloc(ndim_t * sizeof(int));
	const int ndim_subtree_s = su2_tree_axes_list(subtree_s, i_ax_subtree_s);
	const int ndim_subtree_t = su2_tree_axes_list(subtree_t, i_ax_subtree_t);
	assert(ndim_subtree_s == ndim_subtree_t);  // also contains internal axes
	// ensure that axes paired according to tree topology match the provided to-be contracted axes indices
	#ifndef NDEBUG
	for (int i = 0; i < ndim_mult; i++) {
		bool found = false;
		for (int j = 0; j < ndim_subtree_s; j++) {
			if (i_ax_s[i] == i_ax_subtree_s[j]) {
				assert(i_ax_t[i] == i_ax_subtree_t[j]);
				found = true;
				break;
			}
		}
		assert(found);
	}
	#endif

	// map axes of 's' and 't' to axes of contracted tensor
	int* axis_map_s = ct_calloc(ndim_s, sizeof(int));
	int* axis_map_t = ct_calloc(ndim_t, sizeof(int));
	// mark to-be contracted axes as invalid
	for (int i = 0; i < ndim_subtree_s; i++) {
		assert(0 <= i_ax_subtree_s[i] && i_ax_subtree_s[i] < ndim_s);
		axis_map_s[i_ax_subtree_s[i]] = -1;
	}
	for (int i = 0; i < ndim_subtree_t; i++) {
		assert(0 <= i_ax_subtree_t[i] && i_ax_subtree_t[i] < ndim_t);
		axis_map_t[i_ax_subtree_t[i]] = -1;
	}
	assert(axis_map_s[subtree_s->i_ax] == -1);
	assert(axis_map_t[subtree_t->i_ax] == -1);
	if (ndim_mult > 1)
	{
		// retain one of the subtree roots as axis;
		// special case ndim_mult == 1 (both subtrees singles leaves with outer axes indices) will be handled below

		if (subtree_t->i_ax >= ndim_outer_t) {  // subtree root in 't' is an internal axis, does not need to be retained
			// retain subtree root in 's'
			axis_map_s[subtree_s->i_ax] = 0;
		}
		else {
			assert(subtree_s->i_ax >= ndim_outer_s);  // subtree root in 's' is an internal axis, does not need to be retained
			// retain subtree root in 't'
			axis_map_t[subtree_t->i_ax] = 0;
		}
	}
	// new logical axes
	int ndim_logical_r = 0;
	for (int i = 0; i < s->ndim_logical; i++) {
		if (axis_map_s[i] != -1) {
			axis_map_s[i] = ndim_logical_r++;
		}
	}
	for (int i = 0; i < t->ndim_logical; i++) {
		if (axis_map_t[i] != -1) {
			axis_map_t[i] = ndim_logical_r++;
		}
	}
	// new auxiliary axes
	int ndim_auxiliary_r = 0;
	for (int i = s->ndim_logical; i < s->ndim_logical + s->ndim_auxiliary; i++) {
		if (axis_map_s[i] != -1) {
			axis_map_s[i] = ndim_logical_r + ndim_auxiliary_r++;
		}
	}
	for (int i = t->ndim_logical; i < t->ndim_logical + t->ndim_auxiliary; i++) {
		if (axis_map_t[i] != -1) {
			axis_map_t[i] = ndim_logical_r + ndim_auxiliary_r++;
		}
	}
	// new internal axes
	const int ndim_outer_r = ndim_logical_r + ndim_auxiliary_r;
	int ndim_internal_r = 0;
	for (int i = s->ndim_logical + s->ndim_auxiliary; i < ndim_s; i++) {
		if (axis_map_s[i] != -1) {
			axis_map_s[i] = ndim_outer_r + ndim_internal_r++;
		}
	}
	for (int i = t->ndim_logical + t->ndim_auxiliary; i < ndim_t; i++) {
		if (axis_map_t[i] != -1) {
			axis_map_t[i] = ndim_outer_r + ndim_internal_r++;
		}
	}
	if (ndim_mult == 1)
	{
		// subtrees are single leaf nodes, and their axes need to be merged and mapped to an internal axis
		assert(su2_tree_node_is_leaf(subtree_s));
		assert(su2_tree_node_is_leaf(subtree_t));
		assert(subtree_s->i_ax == i_ax_s[0]);
		assert(subtree_t->i_ax == i_ax_t[0]);
		assert(axis_map_s[subtree_s->i_ax] == -1);
		assert(axis_map_t[subtree_t->i_ax] == -1);
		axis_map_s[subtree_s->i_ax] = ndim_outer_r + ndim_internal_r++;
	}
	assert(ndim_internal_r == ndim_outer_r - 3);
	// identify roots of subtrees as same axis (after contracting subtrees to identity)
	if (axis_map_t[subtree_t->i_ax] == -1) {
		assert(axis_map_s[subtree_s->i_ax] >= 0);
		axis_map_t[subtree_t->i_ax] = axis_map_s[subtree_s->i_ax];
	}
	else {
		assert(axis_map_s[subtree_s->i_ax] == -1);
		axis_map_s[subtree_s->i_ax] = axis_map_t[subtree_t->i_ax];
	}

	struct su2_fuse_split_tree mapped_tree_s, mapped_tree_t;
	copy_su2_fuse_split_tree(&s->tree, &mapped_tree_s);
	copy_su2_fuse_split_tree(&t->tree, &mapped_tree_t);
	su2_fuse_split_tree_update_axes_indices(&mapped_tree_s, axis_map_s);
	su2_fuse_split_tree_update_axes_indices(&mapped_tree_t, axis_map_t);

	// construct the new fusion-splitting tree
	struct su2_fuse_split_tree tree_r;
	tree_r.ndim = ndim_logical_r + ndim_auxiliary_r + ndim_internal_r;

	if (mode == SU2_TENSOR_CONTRACT_FUSE_SPLIT)
	{
		if (subtree_s == s->tree.tree_fuse)
		{
			tree_r.tree_fuse = mapped_tree_t.tree_fuse;

			if (subtree_t == t->tree.tree_split)
			{
				tree_r.tree_split = mapped_tree_s.tree_split;
			}
			else
			{
				tree_r.tree_split = mapped_tree_t.tree_split;
				// make a copy to avoid freeing same memory twice
				struct su2_tree_node* tree_split_s = ct_malloc(sizeof(struct su2_tree_node));
				copy_su2_tree(mapped_tree_s.tree_split, tree_split_s);
				struct su2_tree_node* old_subtree = su2_tree_replace_subtree(tree_r.tree_split, axis_map_t[subtree_t->i_ax], tree_split_s);
				assert(old_subtree != NULL);
				delete_su2_tree(old_subtree);
				ct_free(old_subtree);  // free actual node
			}
		}
		else
		{
			assert(subtree_t == t->tree.tree_split);
			tree_r.tree_split = mapped_tree_s.tree_split;

			tree_r.tree_fuse = mapped_tree_s.tree_fuse;
			// make a copy to avoid freeing same memory twice
			struct su2_tree_node* tree_fuse_t = ct_malloc(sizeof(struct su2_tree_node));
			copy_su2_tree(mapped_tree_t.tree_fuse, tree_fuse_t);
			struct su2_tree_node* old_subtree = su2_tree_replace_subtree(tree_r.tree_fuse, axis_map_s[subtree_s->i_ax], tree_fuse_t);
			assert(old_subtree != NULL);
			delete_su2_tree(old_subtree);
			ct_free(old_subtree);  // free actual node
		}
	}
	else  // mode == SU2_TENSOR_CONTRACT_SPLIT_FUSE
	{
		assert(mode == SU2_TENSOR_CONTRACT_SPLIT_FUSE);

		if (subtree_s == s->tree.tree_split)
		{
			tree_r.tree_split = mapped_tree_t.tree_split;

			if (subtree_t == t->tree.tree_fuse)
			{
				tree_r.tree_fuse = mapped_tree_s.tree_fuse;
			}
			else
			{
				tree_r.tree_fuse = mapped_tree_t.tree_fuse;
				// make a copy to avoid freeing same memory twice
				struct su2_tree_node* tree_fuse_s = ct_malloc(sizeof(struct su2_tree_node));
				copy_su2_tree(mapped_tree_s.tree_fuse, tree_fuse_s);
				struct su2_tree_node* old_subtree = su2_tree_replace_subtree(tree_r.tree_fuse, axis_map_t[subtree_t->i_ax], tree_fuse_s);
				assert(old_subtree != NULL);
				delete_su2_tree(old_subtree);
				ct_free(old_subtree);  // free actual node
			}
		}
		else
		{
			assert(subtree_t == t->tree.tree_fuse);
			tree_r.tree_fuse = mapped_tree_s.tree_fuse;

			tree_r.tree_split = mapped_tree_s.tree_split;
			// make a copy to avoid freeing same memory twice
			struct su2_tree_node* tree_split_t = ct_malloc(sizeof(struct su2_tree_node));
			copy_su2_tree(mapped_tree_t.tree_split, tree_split_t);
			struct su2_tree_node* old_subtree = su2_tree_replace_subtree(tree_r.tree_split, axis_map_s[subtree_s->i_ax], tree_split_t);
			assert(old_subtree != NULL);
			delete_su2_tree(old_subtree);
			ct_free(old_subtree);  // free actual node
		}
	}

	struct su2_irreducible_list* outer_irreps_r = ct_calloc(ndim_outer_r, sizeof(struct su2_irreducible_list));
	for (int i = 0; i < ndim_outer_s; i++) {
		if (axis_map_s[i] == -1 || axis_map_s[i] >= ndim_outer_r) {
			continue;
		}
		copy_su2_irreducible_list(&s->outer_irreps[i], &outer_irreps_r[axis_map_s[i]]);
	}
	for (int i = 0; i < ndim_outer_t; i++) {
		if (axis_map_t[i] == -1 || axis_map_t[i] >= ndim_outer_r) {
			continue;
		}
		copy_su2_irreducible_list(&t->outer_irreps[i], &outer_irreps_r[axis_map_t[i]]);
	}

	ct_long** dim_degen_r = ct_calloc(ndim_logical_r, sizeof(ct_long*));
	for (int is = 0; is < s->ndim_logical; is++)
	{
		if (axis_map_s[is] == -1 || axis_map_s[is] >= ndim_outer_r) {
			continue;
		}
		const int ir = axis_map_s[is];
		assert(0 <= ir && ir < ndim_logical_r);
		assert(outer_irreps_r[ir].num > 0);
		// simply copy the pointer
		dim_degen_r[ir] = s->dim_degen[is];
	}
	for (int it = 0; it < t->ndim_logical; it++)
	{
		if (axis_map_t[it] == -1 || axis_map_t[it] >= ndim_outer_r) {
			continue;
		}
		const int ir = axis_map_t[it];
		assert(0 <= ir && ir < ndim_logical_r);
		assert(outer_irreps_r[ir].num > 0);
		// simply copy the pointer
		dim_degen_r[ir] = t->dim_degen[it];
	}

	allocate_empty_su2_tensor(s->dtype, ndim_logical_r, ndim_auxiliary_r, &tree_r, outer_irreps_r, (const ct_long**)dim_degen_r, r);

	ct_free(dim_degen_r);
	for (int i = 0; i < ndim_outer_r; i++) {
		delete_su2_irreducible_list(&outer_irreps_r[i]);
	}
	ct_free(outer_irreps_r);

	delete_su2_fuse_split_tree(&mapped_tree_t);
	delete_su2_fuse_split_tree(&mapped_tree_s);

	// permutations of dense degeneracy tensors before contraction
	int* perm_s = ct_malloc(s->ndim_logical * sizeof(int));
	int c = 0;
	for (int i = 0; i < s->ndim_logical; i++) {
		if (axis_map_s[i] != -1 && axis_map_s[i] < ndim_outer_r) {
			perm_s[c++] = i;
		}
	}
	for (int i = 0; i < ndim_mult; i++) {
		// ignore auxiliary axes for permutation
		if (i_ax_s[i] < s->ndim_logical) {
			perm_s[c++] = i_ax_s[i];
		}
	}
	assert(c == s->ndim_logical);
	const bool perm_s_is_identity = is_identity_permutation(perm_s, s->ndim_logical);
	int* perm_t = ct_malloc(t->ndim_logical * sizeof(int));
	c = 0;
	for (int i = 0; i < ndim_mult; i++) {
		// ignore auxiliary axes for permutation
		if (i_ax_t[i] < t->ndim_logical) {
			perm_t[c++] = i_ax_t[i];
		}
	}
	for (int i = 0; i < t->ndim_logical; i++) {
		if (axis_map_t[i] != -1 && axis_map_t[i] < ndim_outer_r) {
			perm_t[c++] = i;
		}
	}
	assert(c == t->ndim_logical);
	const bool perm_t_is_identity = is_identity_permutation(perm_t, t->ndim_logical);
	// number of to-be multiplied logical dimensions
	int ndim_mult_logical = 0;
	for (int i = 0; i < ndim_mult; i++) {
		// ignore auxiliary axes
		if (i_ax_s[i] < s->ndim_logical) {
			ndim_mult_logical++;
		}
	}
	assert(ndim_mult_logical >= 1);

	const int ndim_r = su2_tensor_ndim(r);

	// contract degeneracy tensors
	qnumber* jlist_r = ct_malloc(ndim_r * sizeof(qnumber));
	struct su2_irrep_trie_node irrep_trie = { 0 };
	for (ct_long cs = 0; cs < s->charge_sectors.nsec; cs++)
	{
		// current 'j' quantum numbers
		const qnumber* jlist_s = &s->charge_sectors.jlists[cs * s->charge_sectors.ndim];

		// corresponding "degeneracy" tensor
		const struct dense_tensor* ds = s->degensors[cs];
		assert(ds->dtype == s->dtype);
		assert(ds->ndim  == s->ndim_logical);
		struct dense_tensor ds_perm;
		if (perm_s_is_identity) {
			ds_perm = *ds;  // copy internal data pointers
		}
		else {
			dense_tensor_transpose(perm_s, ds, &ds_perm);
		}

		// fill 'j' quantum numbers for charge sector in to-be contracted tensor 'r'
		for (int i = 0; i < ndim_s; i++) {
			if (axis_map_s[i] == -1) {
				continue;
			}
			jlist_r[axis_map_s[i]] = jlist_s[i];
		}

		for (ct_long ct = 0; ct < t->charge_sectors.nsec; ct++)
		{
			// 'j' quantum numbers of current sector
			const qnumber* jlist_t = &t->charge_sectors.jlists[ct * t->charge_sectors.ndim];

			bool compatible_sector = true;
			for (int i = 0; i < ndim_subtree_s; i++) {
				if (jlist_s[i_ax_subtree_s[i]] != jlist_t[i_ax_subtree_t[i]]) {
					compatible_sector = false;
					break;
				}
			}
			if (!compatible_sector) {
				continue;
			}

			// corresponding "degeneracy" tensor
			const struct dense_tensor* dt = t->degensors[ct];
			assert(dt->dtype == t->dtype);
			assert(dt->ndim  == t->ndim_logical);
			struct dense_tensor dt_perm;
			if (perm_t_is_identity) {
				dt_perm = *dt;  // copy internal data pointers
			}
			else {
				dense_tensor_transpose(perm_t, dt, &dt_perm);
			}

			// fill 'j' quantum numbers for charge sector in to-be contracted tensor 'r'
			for (int i = 0; i < ndim_t; i++) {
				if (axis_map_t[i] == -1) {
					continue;
				}
				jlist_r[axis_map_t[i]] = jlist_t[i];
			}

			// corresponding "degeneracy" tensor
			struct dense_tensor** dr = (struct dense_tensor**)su2_irrep_trie_search_insert(jlist_r, ndim_r, &irrep_trie);
			if ((*dr) == NULL)
			{
				(*dr) = ct_calloc(1, sizeof(struct dense_tensor));

				// actually multiply dense tensors and store result in 'dr'
				dense_tensor_dot(&ds_perm, TENSOR_AXIS_RANGE_TRAILING, &dt_perm, TENSOR_AXIS_RANGE_LEADING, ndim_mult_logical, (*dr));
			}
			else
			{
				assert((*dr)->dtype == r->dtype);
				assert((*dr)->ndim  == r->ndim_logical);

				// actually multiply dense tensors and add result to 'dr'
				dense_tensor_dot_update(numeric_one(s->dtype), &ds_perm, TENSOR_AXIS_RANGE_TRAILING, &dt_perm, TENSOR_AXIS_RANGE_LEADING, ndim_mult_logical, numeric_one(s->dtype), (*dr));
			}

			if (!perm_t_is_identity) {
				delete_dense_tensor(&dt_perm);
			}
		}

		if (!perm_s_is_identity) {
			delete_dense_tensor(&ds_perm);
		}
	}

	r->degensors = (struct dense_tensor**)su2_irrep_trie_enumerate_configurations(ndim_r, &irrep_trie, &r->charge_sectors);

	delete_su2_irrep_trie(ndim_r, &irrep_trie);

	ct_free(perm_t);
	ct_free(perm_s);

	ct_free(jlist_r);

	ct_free(axis_map_t);
	ct_free(axis_map_s);

	ct_free(i_ax_subtree_t);
	ct_free(i_ax_subtree_s);
}


//________________________________________________________________________________________________________________________
///
/// \brief Comparison function used by 'qsort'.
///
static int compare_su2_irreducible_lists_wrapper(const void* a, const void* b)
{
	return compare_su2_irreducible_lists(
		(const struct su2_irreducible_list*)a,
		(const struct su2_irreducible_list*)b);
}


//________________________________________________________________________________________________________________________
///
/// \brief Contract two SU(2) tensors along a single logical axis,
/// for the scenario that the resulting structural tensor contains a yoga subtree
/// which can be converted to a fusion-splitting tree in a single step.
///
void su2_tensor_contract_yoga(const struct su2_tensor* restrict s, const int i_ax_s, const struct su2_tensor* restrict t, const int i_ax_t, struct su2_tensor* restrict r)
{
	assert(s->dtype == t->dtype);

	// dimension and quantum number compatibility checks
	assert(0 <= i_ax_s && i_ax_s < s->ndim_logical);
	assert(0 <= i_ax_t && i_ax_t < t->ndim_logical);
	assert(su2_irreducible_list_equal(&s->outer_irreps[i_ax_s], &t->outer_irreps[i_ax_t]));
	#ifndef NDEBUG
	for (int k = 0; k < s->outer_irreps[i_ax_s].num; k++) {
		// degeneracy dimensions for to-be contracted axes must match
		assert(s->dim_degen[i_ax_s][s->outer_irreps[i_ax_s].jlist[k]] ==
		       t->dim_degen[i_ax_t][t->outer_irreps[i_ax_t].jlist[k]]);
	}
	#endif

	const int ndim_outer_s = s->ndim_logical + s->ndim_auxiliary;
	const int ndim_outer_t = t->ndim_logical + t->ndim_auxiliary;

	const int ndim_s = su2_tensor_ndim(s);
	const int ndim_t = su2_tensor_ndim(t);

	// map axes of 's' and 't' to axes of contracted tensor
	int* axis_map_s = ct_malloc(ndim_s * sizeof(int));
	int* axis_map_t = ct_malloc(ndim_t * sizeof(int));

	// new logical axes
	int ndim_logical_r = 0;
	for (int i = 0; i < s->ndim_logical; i++) {
		if (i != i_ax_s) {  // skip to-be contracted axis
			axis_map_s[i] = ndim_logical_r++;
		}
	}
	for (int i = 0; i < t->ndim_logical; i++) {
		if (i != i_ax_t) {  // skip to-be contracted axis
			axis_map_t[i] = ndim_logical_r++;
		}
	}
	// new auxiliary axes
	int ndim_auxiliary_r = 0;
	for (int i = s->ndim_logical; i < s->ndim_logical + s->ndim_auxiliary; i++) {
		// 'i_ax_s' cannot be an auxiliary axis
		axis_map_s[i] = ndim_logical_r + ndim_auxiliary_r++;
	}
	for (int i = t->ndim_logical; i < t->ndim_logical + t->ndim_auxiliary; i++) {
		// 'i_ax_t' cannot be an auxiliary axis
		axis_map_t[i] = ndim_logical_r + ndim_auxiliary_r++;
	}
	// new internal axes
	const int ndim_outer_r = ndim_logical_r + ndim_auxiliary_r;
	int ndim_internal_r = 0;
	// to-be contracted axis becomes an internal axis
	axis_map_s[i_ax_s] = ndim_outer_r;
	axis_map_t[i_ax_t] = ndim_outer_r;
	ndim_internal_r++;
	for (int i = s->ndim_logical + s->ndim_auxiliary; i < ndim_s; i++) {
		axis_map_s[i] = ndim_outer_r + ndim_internal_r++;
	}
	for (int i = t->ndim_logical + t->ndim_auxiliary; i < ndim_t; i++) {
		axis_map_t[i] = ndim_outer_r + ndim_internal_r++;
	}
	assert(ndim_internal_r == ndim_outer_r - 3);
	assert(ndim_outer_r + ndim_internal_r == ndim_s + ndim_t - 1);
	const int ndim_r = ndim_outer_r + ndim_internal_r;

	struct su2_graph graph_r;
	{
		struct su2_graph graph_s, graph_t;
		su2_graph_from_fuse_split_tree(&s->tree, &graph_s);
		su2_graph_from_fuse_split_tree(&t->tree, &graph_t);

		assert(su2_graph_is_consistent(&graph_s));
		assert(su2_graph_is_consistent(&graph_t));

		assert(su2_graph_has_fuse_split_tree_topology(&graph_s));
		assert(su2_graph_has_fuse_split_tree_topology(&graph_t));

		su2_graph_connect(&graph_s, axis_map_s, &graph_t, axis_map_t, &graph_r);
		assert(su2_graph_is_yoga_edge(&graph_r, axis_map_s[i_ax_s]));

		delete_su2_graph(&graph_t);
		delete_su2_graph(&graph_s);
	}

	struct su2_irreducible_list* outer_irreps_r = ct_calloc(ndim_outer_r, sizeof(struct su2_irreducible_list));
	for (int i = 0; i < ndim_outer_s; i++) {
		if (i == i_ax_s) {  // skip to-be contracted axis
			continue;
		}
		assert(axis_map_s[i] < ndim_outer_r);
		copy_su2_irreducible_list(&s->outer_irreps[i], &outer_irreps_r[axis_map_s[i]]);
	}
	for (int i = 0; i < ndim_outer_t; i++) {
		if (i == i_ax_t) {  // skip to-be contracted axis
			continue;
		}
		assert(axis_map_t[i] < ndim_outer_r);
		copy_su2_irreducible_list(&t->outer_irreps[i], &outer_irreps_r[axis_map_t[i]]);
	}

	ct_long** dim_degen_r = ct_calloc(ndim_logical_r, sizeof(ct_long*));
	for (int is = 0; is < s->ndim_logical; is++)
	{
		if (is == i_ax_s) {  // skip to-be contracted axis
			continue;
		}
		const int ir = axis_map_s[is];
		assert(0 <= ir && ir < ndim_logical_r);
		assert(outer_irreps_r[ir].num > 0);
		// simply copy the pointer
		dim_degen_r[ir] = s->dim_degen[is];
	}
	for (int it = 0; it < t->ndim_logical; it++)
	{
		if (it == i_ax_t) {  // skip to-be contracted axis
			continue;
		}
		const int ir = axis_map_t[it];
		assert(0 <= ir && ir < ndim_logical_r);
		assert(outer_irreps_r[ir].num > 0);
		// simply copy the pointer
		dim_degen_r[ir] = t->dim_degen[it];
	}

	// merge the charge sectors of 's' and 't'
	assert(s->charge_sectors.ndim == ndim_s);
	assert(t->charge_sectors.ndim == ndim_t);
	struct su2_irreducible_list* merged_sectors = ct_malloc(s->charge_sectors.nsec * t->charge_sectors.nsec * sizeof(struct su2_irreducible_list));
	ct_long nsec_r = 0;
	for (ct_long j = 0; j < s->charge_sectors.nsec; j++)
	{
		for (ct_long k = 0; k < t->charge_sectors.nsec; k++)
		{
			// quantum number at to-be contracted edge must match
			if (s->charge_sectors.jlists[j * ndim_s + i_ax_s] == t->charge_sectors.jlists[k * ndim_t + i_ax_t])
			{
				merged_sectors[nsec_r].num = ndim_r;
				merged_sectors[nsec_r].jlist = ct_malloc(ndim_r * sizeof(qnumber));
				// merge quantum numbers
				for (int i = 0; i < ndim_s; i++) {
					merged_sectors[nsec_r].jlist[axis_map_s[i]] = s->charge_sectors.jlists[j * ndim_s + i];
				}
				for (int i = 0; i < ndim_t; i++) {
					merged_sectors[nsec_r].jlist[axis_map_t[i]] = t->charge_sectors.jlists[k * ndim_t + i];
				}

				nsec_r++;
			}
		}
	}

	// sort lexicographically
	qsort(merged_sectors, nsec_r, sizeof(struct su2_irreducible_list), compare_su2_irreducible_lists_wrapper);

	struct su2_tensor_data data_yoga;

	// copy charge sectors into 'data_yoga'
	allocate_charge_sectors(nsec_r, ndim_r, &data_yoga.charge_sectors);
	for (ct_long i = 0; i < nsec_r; i++)
	{
		memcpy(&data_yoga.charge_sectors.jlists[i * ndim_r], merged_sectors[i].jlist, ndim_r * sizeof(qnumber));
		ct_free(merged_sectors[i].jlist);
	}

	ct_free(merged_sectors);

	// permutations of dense degeneracy tensors before contraction
	// transpose degeneracy tensors in 's' such that to-be contracted axis is the trailing axis
	int* perm_s = ct_malloc(s->ndim_logical * sizeof(int));
	for (int i = 0; i < s->ndim_logical - 1; i++) {
		perm_s[i] = (i < i_ax_s ? i : i + 1);
	}
	perm_s[s->ndim_logical - 1] = i_ax_s;
	const bool perm_s_is_identity = is_identity_permutation(perm_s, s->ndim_logical);
	// transpose degeneracy tensors in 't' such that to-be contracted axis is the leading axis
	int* perm_t = ct_malloc(t->ndim_logical * sizeof(int));
	perm_t[0] = i_ax_t;
	for (int i = 1; i < t->ndim_logical; i++) {
		perm_t[i] = (i <= i_ax_t ? i - 1 : i);
	}
	const bool perm_t_is_identity = is_identity_permutation(perm_t, t->ndim_logical);

	// contract degeneracy tensors
	data_yoga.degensors = ct_calloc(nsec_r, sizeof(struct dense_tensor*));
	qnumber* jlist_r = ct_malloc(ndim_r * sizeof(qnumber));
	for (ct_long cs = 0; cs < s->charge_sectors.nsec; cs++)
	{
		// current 'j' quantum numbers
		const qnumber* jlist_s = &s->charge_sectors.jlists[cs * s->charge_sectors.ndim];

		// corresponding "degeneracy" tensor
		const struct dense_tensor* ds = s->degensors[cs];
		assert(ds->dtype == s->dtype);
		assert(ds->ndim  == s->ndim_logical);
		struct dense_tensor ds_perm;
		if (perm_s_is_identity) {
			ds_perm = *ds;  // only copy pointers
		}
		else {
			dense_tensor_transpose(perm_s, ds, &ds_perm);
		}

		// fill 'j' quantum numbers for merged charge sector
		for (int i = 0; i < ndim_s; i++) {
			jlist_r[axis_map_s[i]] = jlist_s[i];
		}

		for (ct_long ct = 0; ct < t->charge_sectors.nsec; ct++)
		{
			// 'j' quantum numbers of current sector
			const qnumber* jlist_t = &t->charge_sectors.jlists[ct * t->charge_sectors.ndim];

			// quantum number at to-be contracted edge must match
			if (jlist_s[i_ax_s] != jlist_t[i_ax_t]) {
				continue;
			}

			// corresponding "degeneracy" tensor
			const struct dense_tensor* dt = t->degensors[ct];
			assert(dt->dtype == t->dtype);
			assert(dt->ndim  == t->ndim_logical);
			struct dense_tensor dt_perm;
			if (perm_t_is_identity) {
				dt_perm = *dt;  // only copy pointers
			}
			else {
				dense_tensor_transpose(perm_t, dt, &dt_perm);
			}

			// fill 'j' quantum numbers for merged charge sector
			for (int i = 0; i < ndim_t; i++) {
				jlist_r[axis_map_t[i]] = jlist_t[i];
			}

			const ct_long cr = charge_sector_index(&data_yoga.charge_sectors, jlist_r);
			assert(cr != -1);

			if (data_yoga.degensors[cr] == NULL)
			{
				data_yoga.degensors[cr] = ct_malloc(sizeof(struct dense_tensor));
				// actually multiply dense tensors
				dense_tensor_dot(&ds_perm, TENSOR_AXIS_RANGE_TRAILING, &dt_perm, TENSOR_AXIS_RANGE_LEADING, 1, data_yoga.degensors[cr]);
			}
			else
			{
				assert(data_yoga.degensors[cr]->dtype == s->dtype);
				assert(data_yoga.degensors[cr]->ndim  == ndim_logical_r);
				// actually multiply dense tensors and add result to existing tensor
				dense_tensor_dot_update(numeric_one(s->dtype), &ds_perm, TENSOR_AXIS_RANGE_TRAILING, &dt_perm, TENSOR_AXIS_RANGE_LEADING, 1, numeric_one(s->dtype), data_yoga.degensors[cr]);
			}

			if (!perm_t_is_identity) {
				delete_dense_tensor(&dt_perm);
			}
		}

		if (!perm_s_is_identity) {
			delete_dense_tensor(&ds_perm);
		}
	}
	ct_free(jlist_r);

	ct_free(perm_t);
	ct_free(perm_s);

	// convert yoga to simple subtree, and update degeneracy tensors accordingly
	struct su2_tensor_data data_simple;
	su2_convert_yoga_to_simple_subtree(&data_yoga, &graph_r, axis_map_s[i_ax_s], &data_simple);
	assert(su2_graph_has_fuse_split_tree_topology(&graph_r));

	ct_free(axis_map_t);
	ct_free(axis_map_s);

	for (ct_long i = 0; i < nsec_r; i++)
	{
		assert(data_yoga.degensors[i] != NULL);
		delete_dense_tensor(data_yoga.degensors[i]);
		ct_free(data_yoga.degensors[i]);
	}
	ct_free(data_yoga.degensors);
	delete_charge_sectors(&data_yoga.charge_sectors);

	struct su2_fuse_split_tree tree_r;
	su2_graph_to_fuse_split_tree(&graph_r, &tree_r);
	delete_su2_graph(&graph_r);

	allocate_empty_su2_tensor(s->dtype, ndim_logical_r, ndim_auxiliary_r, &tree_r, outer_irreps_r, (const ct_long**)dim_degen_r, r);
	// copy pointers
	r->charge_sectors = data_simple.charge_sectors;
	r->degensors      = data_simple.degensors;

	delete_su2_fuse_split_tree(&tree_r);

	ct_free(dim_degen_r);
	for (int i = 0; i < ndim_outer_r; i++) {
		delete_su2_irreducible_list(&outer_irreps_r[i]);
	}
	ct_free(outer_irreps_r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the lexicographically next quantum index.
///
static inline void next_quantum_index(const int ndim, const int* restrict dim, int* restrict index)
{
	for (int i = ndim - 1; i >= 0; i--)
	{
		index[i]++;
		if (index[i] < dim[i])
		{
			return;
		}
		else
		{
			index[i] = 0;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert an SU(2) symmetric to an equivalent dense tensor.
///
void su2_to_dense_tensor(const struct su2_tensor* restrict s, struct dense_tensor* restrict t)
{
	// start of 'j' charge sector entries, along each logical axis
	ct_long** sector_offsets = ct_malloc(s->ndim_logical * sizeof(ct_long*));

	// logical dimensions of dense tensor
	ct_long* dim_t = ct_calloc(s->ndim_logical, sizeof(ct_long));

	for (int i = 0; i < s->ndim_logical; i++)
	{
		assert(s->outer_irreps[i].num > 0);
		qnumber j_max = 0;
		for (int k = 0; k < s->outer_irreps[i].num; k++) {
			j_max = qmax(j_max, s->outer_irreps[i].jlist[k]);
		}
		sector_offsets[i] = ct_calloc(j_max + 1, sizeof(ct_long));
		for (int k = 0; k < s->outer_irreps[i].num; k++)
		{
			const qnumber j = s->outer_irreps[i].jlist[k];
			assert(s->dim_degen[i][j] > 0);
			sector_offsets[i][j] = dim_t[i];
			dim_t[i] += s->dim_degen[i][j] * (j + 1);
		}
	}

	// allocate dense tensor
	allocate_zero_dense_tensor(s->dtype, s->ndim_logical, dim_t, t);
	ct_free(dim_t);

	// number of outer dimensions, i.e., number of leaves
	const int ndim_outer = s->ndim_logical + s->ndim_auxiliary;

	// accumulate contributions from all charge sectors
	for (ct_long c = 0; c < s->charge_sectors.nsec; c++)
	{
		// corresponding "degeneracy" tensor
		const struct dense_tensor* d = s->degensors[c];
		assert(d->dtype == s->dtype);
		assert(d->ndim  == s->ndim_logical);

		// current 'j' quantum numbers
		const qnumber* jlist = &s->charge_sectors.jlists[c * s->charge_sectors.ndim];

		// dimensions of outer 'm' quantum numbers (defined on leaves)
		int* dim_m_outer = ct_malloc(ndim_outer * sizeof(int));
		int nconfigs = 1;
		for (int k = 0; k < ndim_outer; k++) {
			dim_m_outer[k] = jlist[k] + 1;
			nconfigs *= dim_m_outer[k];
		}

		// iterate over outer 'm' quantum numbers; auxiliary 'm' quantum numbers are traced out
		int* im_outer = ct_calloc(ndim_outer, sizeof(int));
		for (int k = 0; k < nconfigs; k++, next_quantum_index(ndim_outer, dim_m_outer, im_outer))
		{
			// evaluate Clebsch-Gordan coefficients of tree nodes
			double cg = su2_fuse_split_tree_eval_clebsch_gordan(&s->tree, jlist, im_outer);
			if (cg == 0) {
				continue;
			}

			ct_long* index_t = ct_calloc(t->ndim, sizeof(ct_long));

			// distribute degeneracy tensor entries multiplied by Clebsch-Gordan factor
			const ct_long nelem_d = dense_tensor_num_elements(d);
			ct_long* index_d = ct_calloc(d->ndim, sizeof(ct_long));
			for (ct_long l = 0; l < nelem_d; l++, next_tensor_index(d->ndim, d->dim, index_d))
			{
				// index in 't' tensor
				for (int i = 0; i < t->ndim; i++)
				{
					const qnumber j = jlist[i];
					assert(d->dim[i] == s->dim_degen[i][j]);
					index_t[i] = sector_offsets[i][j] + index_d[i] * (j + 1) + im_outer[i];
				}
				const ct_long it = tensor_index_to_offset(t->ndim, t->dim, index_t);

				switch (t->dtype)
				{
					case CT_SINGLE_REAL:
					{
						const float* ddata = d->data;
						float*       tdata = t->data;
						tdata[it] += ((float)cg) * ddata[l];
						break;
					}
					case CT_DOUBLE_REAL:
					{
						const double* ddata = d->data;
						double*       tdata = t->data;
						tdata[it] += cg * ddata[l];
						break;
					}
					case CT_SINGLE_COMPLEX:
					{
						const scomplex* ddata = d->data;
						scomplex*       tdata = t->data;
						tdata[it] += ((float)cg) * ddata[l];
						break;
					}
					case CT_DOUBLE_COMPLEX:
					{
						const dcomplex* ddata = d->data;
						dcomplex*       tdata = t->data;
						tdata[it] += cg * ddata[l];
						break;
					}
					default:
					{
						// unknown data type
						assert(false);
					}
				}
			}

			ct_free(index_d);
			ct_free(index_t);
		}

		ct_free(im_outer);
		ct_free(dim_m_outer);
	}

	for (int i = 0; i < s->ndim_logical; i++) {
		ct_free(sector_offsets[i]);
	}
	ct_free(sector_offsets);
}


//________________________________________________________________________________________________________________________
///
/// \brief Determine the set intersection of two sorted quantum number lists.
///
static inline int qnumber_intersection(const qnumber* restrict list_a, const int num_a, const qnumber* restrict list_b, const int num_b, qnumber* restrict intersection)
{
	int i = 0;
	int j = 0;
	int c = 0;
	while (i < num_a && j < num_b)
	{
		// require that quantum number lists are sorted
		assert(i == num_a - 1 || (list_a[i] < list_a[i + 1]));
		assert(j == num_b - 1 || (list_b[j] < list_b[j + 1]));

		if (list_a[i] == list_b[j])
		{
			intersection[c] = list_a[i];
			i++;
			j++;
			c++;
		}
		else if (list_a[i] < list_b[j])
		{
			i++;
		}
		else
		{
			assert(list_b[j] < list_a[i]);
			j++;
		}
	}

	return c;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the logical QR decomposition of an SU(2) symmetric tensor.
///
/// The input tensor must have an input and an output logical axis and one auxiliary axis with quantum number zero.
///
int su2_tensor_qr(const struct su2_tensor* restrict a, const enum qr_mode mode, struct su2_tensor* restrict q, struct su2_tensor* restrict r)
{
	// require a logical matrix
	assert(a->ndim_logical == 2);
	assert(su2_tensor_logical_axis_direction(a, 0) != su2_tensor_logical_axis_direction(a, 1));
	// expecting a single, "trivial" (solely quantum number zero) auxiliary axis in 'a'
	assert(a->ndim_auxiliary == 1);
	assert(a->outer_irreps[2].num == 1);
	assert(a->outer_irreps[2].jlist[0] == 0);

	// allocate (empty) 'q' tensor (with same tree topology and number of axes as 'a')
	{
		struct su2_irreducible_list outer_irreps_q[3];
		// copy irreducible quantum numbers from first axis of 'a'
		outer_irreps_q[0] = a->outer_irreps[0];  // only copy pointers
		// irreducible quantum numbers for second axis of 'q'
		if (mode == QR_REDUCED)
		{
			// find joint irreducible quantum numbers of first and second axis of 'a'
			// upper bound on required memory
			allocate_su2_irreducible_list(a->outer_irreps[0].num, &outer_irreps_q[1]);
			outer_irreps_q[1].num = qnumber_intersection(
				a->outer_irreps[0].jlist, a->outer_irreps[0].num,
				a->outer_irreps[1].jlist, a->outer_irreps[1].num,
				outer_irreps_q[1].jlist);
			// require at least one common quantum number
			assert(outer_irreps_q[1].num > 0);
			assert(outer_irreps_q[1].num <= a->outer_irreps[0].num);
		}
		else
		{
			assert(mode == QR_COMPLETE);
			// use the same quantum numbers for second axis of 'q' as for first
			copy_su2_irreducible_list(&outer_irreps_q[0], &outer_irreps_q[1]);
		}
		// auxiliary axis has quantum number zero
		qnumber jlist_zero[1] = { 0 };
		outer_irreps_q[2].num = 1;
		outer_irreps_q[2].jlist = jlist_zero;

		ct_long* dim_degen_q[2];
		dim_degen_q[0] = a->dim_degen[0];  // copy pointer
		qnumber j_max = 0;
		for (int k = 0; k < outer_irreps_q[1].num; k++) {
			j_max = qmax(j_max, outer_irreps_q[1].jlist[k]);
		}
		dim_degen_q[1] = ct_calloc(j_max + 1, sizeof(ct_long));
		if (mode == QR_REDUCED)
		{
			for (int k = 0; k < outer_irreps_q[1].num; k++)
			{
				const qnumber j = outer_irreps_q[1].jlist[k];
				dim_degen_q[1][j] = lmin(a->dim_degen[0][j], a->dim_degen[1][j]);
				assert(dim_degen_q[1][j] > 0);
			}
		}
		else
		{
			assert(mode == QR_COMPLETE);
			// same degeneracy dimensions as for first axis of 'a'
			memcpy(dim_degen_q[1], a->dim_degen[0], (j_max + 1) * sizeof(ct_long));
		}

		allocate_empty_su2_tensor(a->dtype, 2, 1, &a->tree, outer_irreps_q, (const ct_long**)dim_degen_q, q);

		ct_free(dim_degen_q[1]);
		delete_su2_irreducible_list(&outer_irreps_q[1]);

		// all possible charge sectors
		su2_fuse_split_tree_enumerate_charge_sectors(&q->tree, q->outer_irreps, &q->charge_sectors);
		assert(q->charge_sectors.nsec > 0);
		assert(q->charge_sectors.ndim == 3);
		// unused charge sectors will correspond to NULL pointers
		q->degensors = ct_calloc(q->charge_sectors.nsec, sizeof(struct dense_tensor*));
	}

	// allocate (empty) 'r' tensor
	{
		// use a mirrored version of the fusion-splitting tree of 'a' for 'r'
		struct su2_fuse_split_tree tree_r;
		copy_su2_fuse_split_tree(&a->tree, &tree_r);
		su2_fuse_split_tree_flip(&tree_r);
		// flip axes 0 <-> 1
		const int axis_map[3] = { 1, 0, 2 };
		su2_fuse_split_tree_update_axes_indices(&tree_r, axis_map);

		struct su2_irreducible_list outer_irreps_r[3] = {
			q->outer_irreps[1],  // logical axis 0
			a->outer_irreps[1],  // logical axis 1
			q->outer_irreps[2],  // auxiliary axis
		};

		const ct_long* dim_degen_r[2] = {
			q->dim_degen[1],
			a->dim_degen[1],
		};

		allocate_empty_su2_tensor(a->dtype, 2, 1, &tree_r, outer_irreps_r, dim_degen_r, r);

		delete_su2_fuse_split_tree(&tree_r);

		// all possible charge sectors
		su2_fuse_split_tree_enumerate_charge_sectors(&r->tree, r->outer_irreps, &r->charge_sectors);
		assert(r->charge_sectors.nsec > 0);
		assert(r->charge_sectors.ndim == 3);
		// unused charge sectors will correspond to NULL pointers
		r->degensors = ct_calloc(r->charge_sectors.nsec, sizeof(struct dense_tensor*));
	}

	// perform QR decompositions of the individual blocks
	bool failed = false;
	#pragma omp parallel for schedule(dynamic)
	for (ct_long ca = 0; ca < a->charge_sectors.nsec; ca++)
	{
		// 'j' quantum numbers of current sector
		const qnumber* jlist = &a->charge_sectors.jlists[ca * a->charge_sectors.ndim];
		// quantum numbers of of first and second logical axis must agree
		assert(jlist[0] == jlist[1]);
		// quantum number of auxiliary axis must be zero
		assert(jlist[2] == 0);
		// corresponding "degeneracy" tensor of 'a'
		const struct dense_tensor* da = a->degensors[ca];

		// charge sector must also exist in 'q'
		const ct_long cq = charge_sector_index(&q->charge_sectors, jlist);
		assert(cq != -1);
		// allocate new degeneracy tensor
		{
			ct_long dim_d[2] = {
				q->dim_degen[0][jlist[0]],
				q->dim_degen[1][jlist[1]],
			};
			assert(q->degensors[cq] == NULL);
			q->degensors[cq] = ct_calloc(1, sizeof(struct dense_tensor));
			allocate_dense_tensor(q->dtype, 2, dim_d, q->degensors[cq]);
		}

		// charge sector must also exist in 'r'
		const ct_long cr = charge_sector_index(&r->charge_sectors, jlist);
		assert(cr != -1);
		// allocate new degeneracy tensor
		{
			ct_long dim_d[2] = {
				r->dim_degen[0][jlist[0]],
				r->dim_degen[1][jlist[1]],
			};
			assert(r->degensors[cr] == NULL);
			r->degensors[cr] = ct_calloc(1, sizeof(struct dense_tensor));
			allocate_dense_tensor(r->dtype, 2, dim_d, r->degensors[cr]);
		}

		// perform QR decomposition of block
		int ret = dense_tensor_qr_fill(da, mode, q->degensors[cq], r->degensors[cr]);
		if (ret != 0) {
			failed = true;
		}
	}

	if (failed) {
		return -1;
	}

	// set unused blocks in 'q' to standard basis vectors (i.e., identities for square blocks) to ensure that 'q' is a valid isometry
	for (ct_long c = 0; c < q->charge_sectors.nsec; c++)
	{
		if (q->degensors[c] != NULL) {
			continue;
		}

		// 'j' quantum numbers of current sector
		const qnumber* jlist = &q->charge_sectors.jlists[c * q->charge_sectors.ndim];
		// quantum numbers of of first and second logical axis must agree
		assert(jlist[0] == jlist[1]);
		// quantum number of auxiliary axis must be zero
		assert(jlist[2] == 0);

		// allocate new degeneracy tensor with zero entries
		ct_long dim_d[2] = {
			q->dim_degen[0][jlist[0]],
			q->dim_degen[1][jlist[1]],
		};
		assert(dim_d[0] >= dim_d[1]);
		q->degensors[c] = ct_calloc(1, sizeof(struct dense_tensor));
		allocate_zero_dense_tensor(q->dtype, 2, dim_d, q->degensors[c]);
		// set diagonal entries to ones
		switch (q->degensors[c]->dtype)
		{
			case CT_SINGLE_REAL:
			{
				float* data = q->degensors[c]->data;
				for (ct_long j = 0; j < dim_d[1]; j++)
				{
					data[j*dim_d[1] + j] = 1;
				}
				break;
			}
			case CT_DOUBLE_REAL:
			{
				double* data = q->degensors[c]->data;
				for (ct_long j = 0; j < dim_d[1]; j++)
				{
					data[j*dim_d[1] + j] = 1;
				}
				break;
			}
			case CT_SINGLE_COMPLEX:
			{
				scomplex* data = q->degensors[c]->data;
				for (ct_long j = 0; j < dim_d[1]; j++)
				{
					data[j*dim_d[1] + j] = 1;
				}
				break;
			}
			case CT_DOUBLE_COMPLEX:
			{
				dcomplex* data = q->degensors[c]->data;
				for (ct_long j = 0; j < dim_d[1]; j++)
				{
					data[j*dim_d[1] + j] = 1;
				}
				break;
			}
			default:
			{
				// unknown data type
				assert(false);
			}
		}
	}

	// condense charge sector quantum numbers and corresponding degeneracy tensors in 'r'
	{
		// find first unused charge sector
		ct_long c = 0;
		for (; c < r->charge_sectors.nsec; c++) {
			if (r->degensors[c] == NULL) {
				break;
			}
		}
		for (ct_long s = c + 1; s < r->charge_sectors.nsec; s++)
		{
			if (r->degensors[s] != NULL)
			{
				memcpy(&r->charge_sectors.jlists[c * 3], &r->charge_sectors.jlists[s * 3], 3 * sizeof(qnumber));
				// copy pointer
				r->degensors[c] = r->degensors[s];
				r->degensors[s] = NULL;
				c++;
			}
		}
		r->charge_sectors.nsec = c;
		assert(r->charge_sectors.nsec > 0);
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the logical RQ decomposition of an SU(2) symmetric tensor.
///
/// The input tensor must have an input and an output logical axis and one auxiliary axis with quantum number zero.
///
int su2_tensor_rq(const struct su2_tensor* restrict a, const enum qr_mode mode, struct su2_tensor* restrict r, struct su2_tensor* restrict q)
{
	// require a logical matrix
	assert(a->ndim_logical == 2);
	assert(su2_tensor_logical_axis_direction(a, 0) != su2_tensor_logical_axis_direction(a, 1));
	// expecting a single, "trivial" (solely quantum number zero) auxiliary axis in 'a'
	assert(a->ndim_auxiliary == 1);
	assert(a->outer_irreps[2].num == 1);
	assert(a->outer_irreps[2].jlist[0] == 0);

	// allocate (empty) 'q' tensor (with same tree topology and number of axes as 'a')
	{
		struct su2_irreducible_list outer_irreps_q[3];
		// copy irreducible quantum numbers from second axis of 'a'
		outer_irreps_q[1] = a->outer_irreps[1];  // only copy pointers
		// irreducible quantum numbers for first axis of 'q'
		if (mode == QR_REDUCED)
		{
			// find joint irreducible quantum numbers of first and second axis of 'a'
			// upper bound on required memory
			allocate_su2_irreducible_list(a->outer_irreps[1].num, &outer_irreps_q[0]);
			outer_irreps_q[0].num = qnumber_intersection(
				a->outer_irreps[0].jlist, a->outer_irreps[0].num,
				a->outer_irreps[1].jlist, a->outer_irreps[1].num,
				outer_irreps_q[0].jlist);
			// require at least one common quantum number
			assert(outer_irreps_q[0].num > 0);
			assert(outer_irreps_q[0].num <= a->outer_irreps[1].num);
		}
		else
		{
			assert(mode == QR_COMPLETE);
			// use the same quantum numbers for first axis of 'q' as for second
			copy_su2_irreducible_list(&outer_irreps_q[1], &outer_irreps_q[0]);
		}
		// auxiliary axis has quantum number zero
		qnumber jlist_zero[1] = { 0 };
		outer_irreps_q[2].num = 1;
		outer_irreps_q[2].jlist = jlist_zero;

		ct_long* dim_degen_q[2];
		dim_degen_q[1] = a->dim_degen[1];  // copy pointer
		qnumber j_max = 0;
		for (int k = 0; k < outer_irreps_q[0].num; k++) {
			j_max = qmax(j_max, outer_irreps_q[0].jlist[k]);
		}
		dim_degen_q[0] = ct_calloc(j_max + 1, sizeof(ct_long));
		if (mode == QR_REDUCED)
		{
			for (int k = 0; k < outer_irreps_q[0].num; k++)
			{
				const qnumber j = outer_irreps_q[0].jlist[k];
				dim_degen_q[0][j] = lmin(a->dim_degen[0][j], a->dim_degen[1][j]);
				assert(dim_degen_q[0][j] > 0);
			}
		}
		else
		{
			assert(mode == QR_COMPLETE);
			// same degeneracy dimensions as for second axis of 'a'
			memcpy(dim_degen_q[0], a->dim_degen[1], (j_max + 1) * sizeof(ct_long));
		}

		allocate_empty_su2_tensor(a->dtype, 2, 1, &a->tree, outer_irreps_q, (const ct_long**)dim_degen_q, q);

		ct_free(dim_degen_q[0]);
		delete_su2_irreducible_list(&outer_irreps_q[0]);

		// all possible charge sectors
		su2_fuse_split_tree_enumerate_charge_sectors(&q->tree, q->outer_irreps, &q->charge_sectors);
		assert(q->charge_sectors.nsec > 0);
		assert(q->charge_sectors.ndim == 3);
		// unused charge sectors will correspond to NULL pointers
		q->degensors = ct_calloc(q->charge_sectors.nsec, sizeof(struct dense_tensor*));
	}

	// allocate (empty) 'r' tensor
	{
		// use a mirrored version of the fusion-splitting tree of 'a' for 'r'
		struct su2_fuse_split_tree tree_r;
		copy_su2_fuse_split_tree(&a->tree, &tree_r);
		su2_fuse_split_tree_flip(&tree_r);
		// flip axes 0 <-> 1
		const int axis_map[3] = { 1, 0, 2 };
		su2_fuse_split_tree_update_axes_indices(&tree_r, axis_map);

		struct su2_irreducible_list outer_irreps_r[3] = {
			a->outer_irreps[0],  // logical axis 0
			q->outer_irreps[0],  // logical axis 1
			q->outer_irreps[2],  // auxiliary axis
		};

		const ct_long* dim_degen_r[2] = {
			a->dim_degen[0],
			q->dim_degen[0],
		};

		allocate_empty_su2_tensor(a->dtype, 2, 1, &tree_r, outer_irreps_r, dim_degen_r, r);

		delete_su2_fuse_split_tree(&tree_r);

		// all possible charge sectors
		su2_fuse_split_tree_enumerate_charge_sectors(&r->tree, r->outer_irreps, &r->charge_sectors);
		assert(r->charge_sectors.nsec > 0);
		assert(r->charge_sectors.ndim == 3);
		// unused charge sectors will correspond to NULL pointers
		r->degensors = ct_calloc(r->charge_sectors.nsec, sizeof(struct dense_tensor*));
	}

	// perform RQ decompositions of the individual blocks
	bool failed = false;
	#pragma omp parallel for schedule(dynamic)
	for (ct_long ca = 0; ca < a->charge_sectors.nsec; ca++)
	{
		// 'j' quantum numbers of current sector
		const qnumber* jlist = &a->charge_sectors.jlists[ca * a->charge_sectors.ndim];
		// quantum numbers of of first and second logical axis must agree
		assert(jlist[0] == jlist[1]);
		// quantum number of auxiliary axis must be zero
		assert(jlist[2] == 0);
		// corresponding "degeneracy" tensor of 'a'
		const struct dense_tensor* da = a->degensors[ca];

		// charge sector must also exist in 'q'
		const ct_long cq = charge_sector_index(&q->charge_sectors, jlist);
		assert(cq != -1);
		// allocate new degeneracy tensor
		{
			ct_long dim_d[2] = {
				q->dim_degen[0][jlist[0]],
				q->dim_degen[1][jlist[1]],
			};
			assert(q->degensors[cq] == NULL);
			q->degensors[cq] = ct_calloc(1, sizeof(struct dense_tensor));
			allocate_dense_tensor(q->dtype, 2, dim_d, q->degensors[cq]);
		}

		// charge sector must also exist in 'r'
		const ct_long cr = charge_sector_index(&r->charge_sectors, jlist);
		assert(cr != -1);
		// allocate new degeneracy tensor
		{
			ct_long dim_d[2] = {
				r->dim_degen[0][jlist[0]],
				r->dim_degen[1][jlist[1]],
			};
			assert(r->degensors[cr] == NULL);
			r->degensors[cr] = ct_calloc(1, sizeof(struct dense_tensor));
			allocate_dense_tensor(r->dtype, 2, dim_d, r->degensors[cr]);
		}

		// perform RQ decomposition of block
		int ret = dense_tensor_rq_fill(da, mode, r->degensors[cr], q->degensors[cq]);
		if (ret != 0) {
			failed = true;
		}
	}

	if (failed) {
		return -1;
	}

	// set unused blocks in 'q' to standard basis vectors (i.e., identities for square blocks) to ensure that 'q' is a valid isometry
	for (ct_long c = 0; c < q->charge_sectors.nsec; c++)
	{
		if (q->degensors[c] != NULL) {
			continue;
		}

		// 'j' quantum numbers of current sector
		const qnumber* jlist = &q->charge_sectors.jlists[c * q->charge_sectors.ndim];
		// quantum numbers of of first and second logical axis must agree
		assert(jlist[0] == jlist[1]);
		// quantum number of auxiliary axis must be zero
		assert(jlist[2] == 0);

		// allocate new degeneracy tensor with zero entries
		ct_long dim_d[2] = {
			q->dim_degen[0][jlist[0]],
			q->dim_degen[1][jlist[1]],
		};
		assert(dim_d[0] <= dim_d[1]);
		q->degensors[c] = ct_calloc(1, sizeof(struct dense_tensor));
		allocate_zero_dense_tensor(q->dtype, 2, dim_d, q->degensors[c]);
		// set diagonal entries to ones
		switch (q->degensors[c]->dtype)
		{
			case CT_SINGLE_REAL:
			{
				float* data = q->degensors[c]->data;
				for (ct_long j = 0; j < dim_d[0]; j++)
				{
					data[j*dim_d[1] + j] = 1;
				}
				break;
			}
			case CT_DOUBLE_REAL:
			{
				double* data = q->degensors[c]->data;
				for (ct_long j = 0; j < dim_d[0]; j++)
				{
					data[j*dim_d[1] + j] = 1;
				}
				break;
			}
			case CT_SINGLE_COMPLEX:
			{
				scomplex* data = q->degensors[c]->data;
				for (ct_long j = 0; j < dim_d[0]; j++)
				{
					data[j*dim_d[1] + j] = 1;
				}
				break;
			}
			case CT_DOUBLE_COMPLEX:
			{
				dcomplex* data = q->degensors[c]->data;
				for (ct_long j = 0; j < dim_d[0]; j++)
				{
					data[j*dim_d[1] + j] = 1;
				}
				break;
			}
			default:
			{
				// unknown data type
				assert(false);
			}
		}
	}

	// condense charge sector quantum numbers and corresponding degeneracy tensors in 'r'
	{
		// find first unused charge sector
		ct_long c = 0;
		for (; c < r->charge_sectors.nsec; c++) {
			if (r->degensors[c] == NULL) {
				break;
			}
		}
		for (ct_long s = c + 1; s < r->charge_sectors.nsec; s++)
		{
			if (r->degensors[s] != NULL)
			{
				memcpy(&r->charge_sectors.jlists[c * 3], &r->charge_sectors.jlists[s * 3], 3 * sizeof(qnumber));
				// copy pointer
				r->degensors[c] = r->degensors[s];
				r->degensors[s] = NULL;
				c++;
			}
		}
		r->charge_sectors.nsec = c;
		assert(r->charge_sectors.nsec > 0);
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the logical "economical" singular value decomposition of an SU(2) symmetric matrix.
///
/// The parameter 'copy_tree_left' indicates whether to transfer the SU(2) tree of 'a' to 'u' or to 'vh'.
/// The respective other matrix uses a mirrored version of the tree.
///
/// The singular values are returned in a dense vector.
///
/// The 'multiplicities' vector is allocated and filled by the function, with the same length as the singular values.
/// The i-th entry is the logical multiplicity '2 j + 1' of the 'j' quantum sector containing the i-th singular value.
///
int su2_tensor_svd(const struct su2_tensor* restrict a, const bool copy_tree_left, struct su2_tensor* restrict u, struct dense_tensor* restrict s, int** multiplicities, struct su2_tensor* restrict vh)
{
	// require a logical matrix
	assert(a->ndim_logical == 2);
	assert(su2_tensor_logical_axis_direction(a, 0) != su2_tensor_logical_axis_direction(a, 1));
	// expecting a single, "trivial" (solely quantum number zero) auxiliary axis in 'a'
	assert(a->ndim_auxiliary == 1);
	assert(a->outer_irreps[2].num == 1);
	assert(a->outer_irreps[2].jlist[0] == 0);

	// take only the charge sectors which are present in 'a' into account
	assert(a->charge_sectors.nsec > 0);
	struct su2_irreducible_list irreps_shared;
	allocate_su2_irreducible_list(a->charge_sectors.nsec, &irreps_shared);
	qnumber j_max_shared = 0;
	for (ct_long c = 0; c < a->charge_sectors.nsec; c++)
	{
		// 'j' quantum numbers of current sector
		const qnumber* jlist = &a->charge_sectors.jlists[c * a->charge_sectors.ndim];
		// quantum numbers of of first and second logical axis must agree
		assert(jlist[0] == jlist[1]);
		// quantum number of auxiliary axis must be zero
		assert(jlist[2] == 0);
		const qnumber j = jlist[0];

		irreps_shared.jlist[c] = j;

		j_max_shared = qmax(j_max_shared, j);
	}
	ct_long* offset_map = ct_malloc((irreps_shared.num + 1) * sizeof(ct_long));
	offset_map[0] = 0;
	ct_long* dim_degen_shared = ct_calloc(j_max_shared + 1, sizeof(ct_long));
	for (int k = 0; k < irreps_shared.num; k++)
	{
		const qnumber j = irreps_shared.jlist[k];
		dim_degen_shared[j] = lmin(a->dim_degen[0][j], a->dim_degen[1][j]);
		assert(dim_degen_shared[j] > 0);
		offset_map[k + 1] = offset_map[k] + dim_degen_shared[j];
	}

	(*multiplicities) = ct_malloc(offset_map[irreps_shared.num] * sizeof(int));

	struct su2_fuse_split_tree flipped_tree;
	{
		copy_su2_fuse_split_tree(&a->tree, &flipped_tree);
		su2_fuse_split_tree_flip(&flipped_tree);
		// flip axes 0 <-> 1
		const int axis_map[3] = { 1, 0, 2 };
		su2_fuse_split_tree_update_axes_indices(&flipped_tree, axis_map);
	}

	// allocate 'u' tensor
	{
		struct su2_irreducible_list outer_irreps_u[3];
		// copy irreducible quantum numbers from first axis of 'a'
		outer_irreps_u[0] = a->outer_irreps[0];
		outer_irreps_u[1] = irreps_shared;
		// auxiliary axis has quantum number zero
		qnumber jlist_zero[1] = { 0 };
		outer_irreps_u[2].num = 1;
		outer_irreps_u[2].jlist = jlist_zero;
		// degeneracy dimensions
		const ct_long* dim_degen_u[2] = {
			a->dim_degen[0],
			dim_degen_shared,
		};

		allocate_su2_tensor(a->dtype, 2, 1, copy_tree_left ? &a->tree : &flipped_tree, outer_irreps_u, dim_degen_u, u);
		assert(charge_sectors_equal(&u->charge_sectors, &a->charge_sectors));
	}

	// allocate 'vh' tensor
	{
		struct su2_irreducible_list outer_irreps_vh[3];
		// copy irreducible quantum numbers from first axis of 'a'
		outer_irreps_vh[0] = irreps_shared;
		outer_irreps_vh[1] = a->outer_irreps[1];
		// auxiliary axis has quantum number zero
		qnumber jlist_zero[1] = { 0 };
		outer_irreps_vh[2].num = 1;
		outer_irreps_vh[2].jlist = jlist_zero;
		// degeneracy dimensions
		const ct_long* dim_degen_vh[2] = {
			dim_degen_shared,
			a->dim_degen[1],
		};

		allocate_su2_tensor(a->dtype, 2, 1, copy_tree_left ? &flipped_tree : &a->tree, outer_irreps_vh, dim_degen_vh, vh);
		assert(charge_sectors_equal(&vh->charge_sectors, &a->charge_sectors));
	}

	delete_su2_fuse_split_tree(&flipped_tree);
	ct_free(dim_degen_shared);
	delete_su2_irreducible_list(&irreps_shared);

	// allocate vector for singular values
	const ct_long dim_s[1] = { offset_map[a->charge_sectors.nsec] };
	allocate_dense_tensor(numeric_real_type(a->dtype), 1, dim_s, s);
	const size_t sval_dtype_size = sizeof_numeric_type(s->dtype);

	// perform SVD decompositions of the individual blocks
	bool failed = false;
	#pragma omp parallel for schedule(dynamic)
	for (ct_long c = 0; c < a->charge_sectors.nsec; c++)
	{
		const struct dense_tensor* da = a->degensors[c];

		// allocate vector for singular values of current block
		struct dense_tensor s_block;
		const ct_long dim_s_block[1] = { u->degensors[c]->dim[1] };
		allocate_dense_tensor(s->dtype, 1, dim_s_block, &s_block);

		// perform singular value decomposition of block
		int ret = dense_tensor_svd_fill(da, u->degensors[c], &s_block, vh->degensors[c]);
		if (ret != 0) {
			failed = true;
		}

		// copy singular values
		// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
		memcpy((int8_t*)s->data + offset_map[c] * sval_dtype_size, s_block.data, s_block.dim[0] * sval_dtype_size);

		// 'j' quantum numbers of current sector
		const qnumber* jlist = &a->charge_sectors.jlists[c * a->charge_sectors.ndim];
		const qnumber j = jlist[0];

		for (ct_long i = 0; i < s_block.dim[0]; i++)
		{
			(*multiplicities)[offset_map[c] + i] = j + 1;
		}

		delete_dense_tensor(&s_block);
	}

	ct_free(offset_map);

	if (failed) {
		return -1;
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Test whether two SU(2) tensors agree in terms of quantum numbers, internal tree structure and degeneracy tensor entries within tolerance 'tol'.
///
bool su2_tensor_allclose(const struct su2_tensor* restrict s, const struct su2_tensor* restrict t, const double tol)
{
	// compare data types
	if (s->dtype != t->dtype) {
		return false;
	}

	// compare degrees
	if (s->ndim_logical != t->ndim_logical) {
		return false;
	}
	if (s->ndim_auxiliary != t->ndim_auxiliary) {
		return false;
	}

	// compare trees
	if (!su2_fuse_split_tree_equal(&s->tree, &t->tree)) {
		return false;
	}

	// compare irreducible 'j' quantum numbers on outer axes
	const int ndim_outer = s->ndim_logical + s->ndim_auxiliary;
	for (int i = 0; i < ndim_outer; i++) {
		if (!su2_irreducible_list_equal(&s->outer_irreps[i], &t->outer_irreps[i])) {
			return false;
		}
	}

	// compare charge sectors
	if (!charge_sectors_equal(&s->charge_sectors, &t->charge_sectors)) {
		return false;
	}

	// compare degeneracy dimensions
	for (int i = 0; i < s->ndim_logical; i++)
	{
		assert(s->outer_irreps[i].num > 0);
		qnumber j_max = 0;
		for (int k = 0; k < s->outer_irreps[i].num; k++) {
			j_max = qmax(j_max, s->outer_irreps[i].jlist[k]);
		}
		for (qnumber j = 0; j <= j_max; j++) {
			if (s->dim_degen[i][j] != t->dim_degen[i][j]) {
				return false;
			}
		}
	}

	// compare degeneracy tensors
	for (ct_long c = 0; c < s->charge_sectors.nsec; c++) {
		if (!dense_tensor_allclose(s->degensors[c], t->degensors[c], tol)) {
			return false;
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Test whether a dense and an SU(2) symmetric tensor agree elementwise within tolerance 'tol'.
///
bool dense_su2_tensor_allclose(const struct dense_tensor* s, const struct su2_tensor* t, const double tol)
{
	struct dense_tensor t_dns;
	su2_to_dense_tensor(t, &t_dns);

	bool is_close = dense_tensor_allclose(s, &t_dns, tol);

	delete_dense_tensor(&t_dns);

	return is_close;
}


//________________________________________________________________________________________________________________________
///
/// \brief Test whether an SU(2) symmetric tensors is close to the identity map within tolerance 'tol'.
///
bool su2_tensor_is_identity(const struct su2_tensor* t, const double tol)
{
	// must be a logical matrix
	if (t->ndim_logical != 2) {
		return false;
	}

	// one logical axis must be an input axis and the other an output axis
	if (su2_tensor_logical_axis_direction(t, 0) == su2_tensor_logical_axis_direction(t, 1)) {
		return false;
	}

	// all quantum numbers and degeneracy dimensions of the two logical axes must agree
	if (!su2_irreducible_list_equal(&t->outer_irreps[0], &t->outer_irreps[1])) {
		return false;
	}
	for (int k = 0; k < t->outer_irreps[0].num; k++)
	{
		assert(t->outer_irreps[0].jlist[k] == t->outer_irreps[1].jlist[k]);
		const qnumber j = t->outer_irreps[0].jlist[k];
		if (t->dim_degen[0][j] != t->dim_degen[1][j]) {
			return false;
		}
	}

	// all auxiliary axes must be "trivial" (soleley quantum number zero)
	const int ndim_outer = t->ndim_logical + t->ndim_auxiliary;
	for (int i = t->ndim_logical; i < ndim_outer; i++)
	{
		if (t->outer_irreps[i].num != 1) {
			return false;
		}
		if (t->outer_irreps[i].jlist[0] != 0) {
			return false;
		}
	}

	// all possible diagonal blocks must be identities
	if (t->charge_sectors.nsec != t->outer_irreps[0].num) {
		return false;
	}
	for (int k = 0; k < t->outer_irreps[0].num; k++)
	{
		const qnumber j = t->outer_irreps[0].jlist[k];

		ct_long c;
		for (c = 0; c < t->charge_sectors.nsec; c++)
		{
			// current 'j' quantum numbers
			const qnumber* jlist = &t->charge_sectors.jlists[c * t->charge_sectors.ndim];
			if (jlist[0] == j && jlist[1] == j) {
				// found it
				break;
			}
		}
		if (c == t->charge_sectors.nsec) {
			return false;
		}

		assert(t->degensors[c]->ndim == 2);
		if (!dense_tensor_is_identity(t->degensors[c], tol)) {
			return false;
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Test whether an SU(2) tensors is an isometry within tolerance 'tol'.
///
bool su2_tensor_is_isometry(const struct su2_tensor* t, const double tol, const bool transpose)
{
	// must be a logical matrix
	if (t->ndim_logical != 2) {
		return false;
	}

	const int i_ax_open = (transpose ? 0 : 1);
	const int i_ax_cntr = (transpose ? 1 : 0);

	const bool i_ax_open_in_fuse_tree = su2_tree_contains_leaf(t->tree.tree_fuse, i_ax_open);
	const bool i_ax_cntr_in_fuse_tree = su2_tree_contains_leaf(t->tree.tree_fuse, i_ax_cntr);
	// one logical axis must be an input axis and the other an output axis
	if (i_ax_open_in_fuse_tree == i_ax_cntr_in_fuse_tree) {
		return false;
	}

	// all auxiliary axes must be "trivial" (soleley quantum number zero)
	const int ndim_outer = t->ndim_logical + t->ndim_auxiliary;
	for (int i = t->ndim_logical; i < ndim_outer; i++)
	{
		if (t->outer_irreps[i].num != 1) {
			return false;
		}
		if (t->outer_irreps[i].jlist[0] != 0) {
			return false;
		}
	}

	// adjoint (conjugate transpose) tensor
	struct su2_tensor tdag;
	// TODO: avoid full copy
	copy_su2_tensor(t, &tdag);
	conjugate_su2_tensor(&tdag);
	// revert tensor axes directions for multiplication
	su2_tensor_flip_trees(&tdag);

	struct su2_tensor t2;
	const int ndim = su2_tensor_ndim(t);
	// include potential auxiliary axes for contraction
	int* i_ax_cntr_aux = ct_malloc(ndim * sizeof(int));  // upper bound on required memory
	const int ndim_mult = su2_tree_leaf_axes_list(i_ax_cntr_in_fuse_tree ? t->tree.tree_fuse : t->tree.tree_split, i_ax_cntr_aux);
	assert(ndim_mult >= 1);
	// add an auxiliary dummy axis to "open" axis part of tree if necessary
	if (su2_tree_node_is_leaf(i_ax_open_in_fuse_tree ? t->tree.tree_fuse : t->tree.tree_split))
	{
		su2_tensor_add_auxiliary_axis(&tdag, i_ax_open, false);
	}
	su2_tensor_contract_simple(&tdag, i_ax_cntr_aux, t, i_ax_cntr_aux, ndim_mult, &t2);
	ct_free(i_ax_cntr_aux);
	delete_su2_tensor(&tdag);

	const bool is_isometry = su2_tensor_is_identity(&t2, tol);

	delete_su2_tensor(&t2);

	return is_isometry;
}


//________________________________________________________________________________________________________________________
///
/// \brief Overall number of entries in the dense degeneracy tensors.
///
ct_long su2_tensor_num_elements_degensors(const struct su2_tensor* t)
{
	ct_long nelem = 0;
	for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
	{
		nelem += dense_tensor_num_elements(t->degensors[c]);
	}

	return nelem;
}


//________________________________________________________________________________________________________________________
///
/// \brief Store the degeneracy tensor entries in a linear array, which must have been allocated beforehand.
///
void su2_tensor_serialize_entries(const struct su2_tensor* t, void* entries)
{
	const size_t dtype_size = sizeof_numeric_type(t->dtype);

	// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
	int8_t* pentries = (int8_t*)entries;

	for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
	{
		const struct dense_tensor* d = t->degensors[c];
		assert(d != NULL);
		assert(d->dtype == t->dtype);
		const size_t nbytes = dense_tensor_num_elements(d) * dtype_size;
		memcpy(pentries, d->data, nbytes);
		pentries += nbytes;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Fill the degeneracy tensor entries from a linear array.
///
void su2_tensor_deserialize_entries(struct su2_tensor* t, const void* entries)
{
	const size_t dtype_size = sizeof_numeric_type(t->dtype);

	// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
	const int8_t* pentries = (const int8_t*)entries;

	for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
	{
		struct dense_tensor* d = t->degensors[c];
		assert(d != NULL);
		assert(d->dtype == t->dtype);
		const size_t nbytes = dense_tensor_num_elements(d) * dtype_size;
		memcpy(d->data, pentries, nbytes);
		pentries += nbytes;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Store the degeneracy tensor entries in a linear array, renormalized by the structural part.
///
void su2_tensor_serialize_renormalized_entries(const struct su2_tensor* t, void* entries)
{
	const int i_ax_root = t->tree.tree_fuse->i_ax;
	assert(i_ax_root == t->tree.tree_split->i_ax);

	switch (t->dtype)
	{
		case CT_SINGLE_REAL:
		{
			float* pentries = entries;

			const int ndim = t->charge_sectors.ndim;
			for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
			{
				// current 'j' quantum numbers
				const qnumber* jlist = &t->charge_sectors.jlists[c * ndim];

				const float factor = sqrtf(jlist[i_ax_root] + 1);

				const struct dense_tensor* d = t->degensors[c];
				const ct_long nelem = dense_tensor_num_elements(d);
				const float* ddata = d->data;
				for (ct_long i = 0; i < nelem; i++) {
					pentries[i] = factor * ddata[i];
				}
				pentries += nelem;
			}

			break;
		}
		case CT_DOUBLE_REAL:
		{
			double* pentries = entries;

			const int ndim = t->charge_sectors.ndim;
			for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
			{
				// current 'j' quantum numbers
				const qnumber* jlist = &t->charge_sectors.jlists[c * ndim];

				const double factor = sqrt(jlist[i_ax_root] + 1);

				const struct dense_tensor* d = t->degensors[c];
				const ct_long nelem = dense_tensor_num_elements(d);
				const double* ddata = d->data;
				for (ct_long i = 0; i < nelem; i++) {
					pentries[i] = factor * ddata[i];
				}
				pentries += nelem;
			}

			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			scomplex* pentries = entries;

			const int ndim = t->charge_sectors.ndim;
			for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
			{
				// current 'j' quantum numbers
				const qnumber* jlist = &t->charge_sectors.jlists[c * ndim];

				const float factor = sqrtf(jlist[i_ax_root] + 1);

				const struct dense_tensor* d = t->degensors[c];
				const ct_long nelem = dense_tensor_num_elements(d);
				const scomplex* ddata = d->data;
				for (ct_long i = 0; i < nelem; i++) {
					pentries[i] = factor * ddata[i];
				}
				pentries += nelem;
			}

			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			dcomplex* pentries = entries;

			const int ndim = t->charge_sectors.ndim;
			for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
			{
				// current 'j' quantum numbers
				const qnumber* jlist = &t->charge_sectors.jlists[c * ndim];

				const double factor = sqrt(jlist[i_ax_root] + 1);

				const struct dense_tensor* d = t->degensors[c];
				const ct_long nelem = dense_tensor_num_elements(d);
				const dcomplex* ddata = d->data;
				for (ct_long i = 0; i < nelem; i++) {
					pentries[i] = factor * ddata[i];
				}
				pentries += nelem;
			}

			break;
		}
		default:
		{
			// unknown data type
			assert(false);
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Fill the degeneracy tensor entries from a linear array, taking renormalization by the structural part into account.
///
void su2_tensor_deserialize_renormalized_entries(struct su2_tensor* t, const void* entries)
{
	const int i_ax_root = t->tree.tree_fuse->i_ax;
	assert(i_ax_root == t->tree.tree_split->i_ax);

	switch (t->dtype)
	{
		case CT_SINGLE_REAL:
		{
			const float* pentries = entries;

			const int ndim = t->charge_sectors.ndim;
			for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
			{
				// current 'j' quantum numbers
				const qnumber* jlist = &t->charge_sectors.jlists[c * ndim];

				const float factor = 1 / sqrtf(jlist[i_ax_root] + 1);

				struct dense_tensor* d = t->degensors[c];
				const ct_long nelem = dense_tensor_num_elements(d);
				float* ddata = d->data;
				for (ct_long i = 0; i < nelem; i++) {
					ddata[i] = factor * pentries[i];
				}
				pentries += nelem;
			}

			break;
		}
		case CT_DOUBLE_REAL:
		{
			const double* pentries = entries;

			const int ndim = t->charge_sectors.ndim;
			for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
			{
				// current 'j' quantum numbers
				const qnumber* jlist = &t->charge_sectors.jlists[c * ndim];

				const double factor = 1 / sqrt(jlist[i_ax_root] + 1);

				struct dense_tensor* d = t->degensors[c];
				const ct_long nelem = dense_tensor_num_elements(d);
				double* ddata = d->data;
				for (ct_long i = 0; i < nelem; i++) {
					ddata[i] = factor * pentries[i];
				}
				pentries += nelem;
			}

			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			const scomplex* pentries = entries;

			const int ndim = t->charge_sectors.ndim;
			for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
			{
				// current 'j' quantum numbers
				const qnumber* jlist = &t->charge_sectors.jlists[c * ndim];

				const float factor = 1 / sqrtf(jlist[i_ax_root] + 1);

				struct dense_tensor* d = t->degensors[c];
				const ct_long nelem = dense_tensor_num_elements(d);
				scomplex* ddata = d->data;
				for (ct_long i = 0; i < nelem; i++) {
					ddata[i] = factor *  pentries[i];
				}
				pentries += nelem;
			}

			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			const dcomplex* pentries = entries;

			const int ndim = t->charge_sectors.ndim;
			for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
			{
				// current 'j' quantum numbers
				const qnumber* jlist = &t->charge_sectors.jlists[c * ndim];

				const double factor = 1 / sqrt(jlist[i_ax_root] + 1);

				struct dense_tensor* d = t->degensors[c];
				const ct_long nelem = dense_tensor_num_elements(d);
				dcomplex* ddata = d->data;
				for (ct_long i = 0; i < nelem; i++) {
					ddata[i] = factor * pentries[i];
				}
				pentries += nelem;
			}

			break;
		}
		default:
		{
			// unknown data type
			assert(false);
		}
	}
}
