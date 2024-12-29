/// \file su2_tensor.c
/// \brief Data structures and functions for SU(2) symmetric tensors.

#include <assert.h>
#include "su2_tensor.h"
#include "su2_recoupling.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for an SU(2) symmetric tensor, including the dense "degeneracy" tensors.
///
void allocate_su2_tensor(const enum numeric_type dtype, const int ndim_logical, int ndim_auxiliary, const struct su2_fuse_split_tree* tree, const struct su2_irreducible_list* outer_jlists, const long** dim_degen, struct su2_tensor* t)
{
	t->dtype = dtype;

	assert(ndim_logical   >= 1);
	assert(ndim_auxiliary >= 0);
	t->ndim_logical   = ndim_logical;
	t->ndim_auxiliary = ndim_auxiliary;

	copy_su2_fuse_split_tree(tree, &t->tree);

	// irreducible 'j' quantum numbers on outer axes
	const int ndim_outer = ndim_logical + ndim_auxiliary;
	t->outer_jlists = ct_malloc(ndim_outer * sizeof(struct su2_irreducible_list));
	for (int i = 0; i < ndim_outer; i++) {
		copy_su2_irreducible_list(&outer_jlists[i], &t->outer_jlists[i]);
	}

	su2_fuse_split_tree_enumerate_charge_sectors(&t->tree, t->outer_jlists, &t->charge_sectors);

	// degeneracy tensors
	t->dim_degen = ct_malloc(t->ndim_logical * sizeof(long*));
	for (int i = 0; i < t->ndim_logical; i++)
	{
		assert(t->outer_jlists[i].num > 0);
		qnumber j_max = 0;
		for (int k = 0; k < t->outer_jlists[i].num; k++) {
			j_max = imax(j_max, t->outer_jlists[i].jlist[k]);
		}
		t->dim_degen[i] = ct_malloc((j_max + 1) * sizeof(long));
		memcpy(t->dim_degen[i], dim_degen[i], (j_max + 1) * sizeof(long));
	}
	t->degensors = ct_malloc(t->charge_sectors.nsec * sizeof(struct dense_tensor*));
	for (long c = 0; c < t->charge_sectors.nsec; c++)
	{
		// current 'j' quantum numbers
		const qnumber* jlist = &t->charge_sectors.jlists[c * t->charge_sectors.ndim];
		// dimension of degeneracy tensor
		long* dim_d = ct_malloc(t->ndim_logical * sizeof(long));
		for (int i = 0; i < t->ndim_logical; i++) {
			const qnumber j = jlist[i];
			assert(t->dim_degen[i][j] > 0);
			dim_d[i] = t->dim_degen[i][j];
		}
		t->degensors[c] = ct_calloc(1, sizeof(struct dense_tensor));
		allocate_dense_tensor(t->dtype, t->ndim_logical, dim_d, t->degensors[c]);
		ct_free(dim_d);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an SU(2) symmetric tensor (free memory).
///
void delete_su2_tensor(struct su2_tensor* t)
{
	// degeneracy tensors
	for (long c = 0; c < t->charge_sectors.nsec; c++)
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
		delete_su2_irreducible_list(&t->outer_jlists[i]);
	}
	ct_free(t->outer_jlists);
	t->outer_jlists = NULL;

	delete_su2_fuse_split_tree(&t->tree);
}


//________________________________________________________________________________________________________________________
///
/// \brief Logical dimension of axis 'i_ax' of SU(2) tensor 't'.
///
long su2_tensor_dim_logical_axis(const struct su2_tensor* t, const int i_ax)
{
	assert(0 <= i_ax && i_ax < t->ndim_logical);

	long d = 0;
	for (int k = 0; k < t->outer_jlists[i_ax].num; k++)
	{
		const qnumber j = t->outer_jlists[i_ax].jlist[k];
		assert(t->dim_degen[i_ax][j] > 0);
		d += t->dim_degen[i_ax][j] * (j + 1);
	}

	return d;
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
		if (t->outer_jlists[i].num <= 0) {
			return false;
		}

		for (int k = 0; k < t->outer_jlists[i].num; k++)
		{
			const qnumber j = t->outer_jlists[i].jlist[k];
			if (j < 0) {
				return false;
			}
			if (t->dim_degen[i][j] <= 0) {
				return false;
			}
		}
	}

	for (long c = 0; c < t->charge_sectors.nsec; c++)
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

	// create new tensor
	allocate_su2_tensor(t->dtype, t->ndim_logical, t->ndim_auxiliary, &tree, t->outer_jlists, (const long**)t->dim_degen, r);
	delete_su2_fuse_split_tree(&tree);

	for (long cr = 0; cr < r->charge_sectors.nsec; cr++)
	{
		// 'j' quantum numbers of current sector
		const qnumber* jlist_r = &r->charge_sectors.jlists[cr * r->charge_sectors.ndim];

		const qnumber ja = jlist_r[i_ax_a];
		const qnumber jb = jlist_r[i_ax_b];
		const qnumber jc = jlist_r[i_ax_c];
		const qnumber js = jlist_r[i_ax_s];

		// TODO: binary search of matching sectors
		for (long ct = 0; ct < t->charge_sectors.nsec; ct++)
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
			const double coeff = su_recoupling_coefficient(ja, jb, jc, js, je, jf);
			if (coeff == 0) {
				continue;
			}

			// ensure that 'alpha' is large enough to store any numeric type
			dcomplex alpha;
			numeric_from_double(coeff, r->dtype, &alpha);

			// accumulate degeneracy tensor from 't' weighted by 'coeff'
			dense_tensor_scalar_multiply_add(&alpha, t->degensors[ct], r->degensors[cr]);
		}
	}
}


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
	for (int i = 0; i < ndim_mult; i++)
	{
		assert(0 <= i_ax_s[i] && i_ax_s[i] < s->ndim_logical + s->ndim_auxiliary);
		assert(0 <= i_ax_t[i] && i_ax_t[i] < t->ndim_logical + t->ndim_auxiliary);

		// must both be logical or both be auxiliary axes
		assert((i_ax_s[i] < s->ndim_logical && i_ax_t[i] < t->ndim_logical) || (i_ax_s[i] >= s->ndim_logical && i_ax_t[i] >= t->ndim_logical));

		const struct su2_irreducible_list* irred_s = &s->outer_jlists[i_ax_s[i]];
		#ifndef NDEBUG
		const struct su2_irreducible_list* irred_t = &t->outer_jlists[i_ax_t[i]];
		#endif
		assert(su2_irreducible_list_equal(irred_s, irred_t));
		if (i_ax_s[i] < s->ndim_logical) {
			for (int k = 0; k < irred_s->num; k++) {
				// degeneracy dimensions for to-be contracted axes must match
				assert(s->dim_degen[i_ax_s[i]][irred_s->jlist[k]] ==
				       t->dim_degen[i_ax_t[i]][irred_t->jlist[k]]);
			}
		}
	}

	const int ndim_outer_s = s->ndim_logical + s->ndim_auxiliary;
	const int ndim_outer_t = t->ndim_logical + t->ndim_auxiliary;

	const struct su2_tree_node* subtree_s = su2_subtree_with_leaf_axes(s->tree.tree_split, i_ax_s, ndim_mult);
	const struct su2_tree_node* subtree_t = su2_subtree_with_leaf_axes(t->tree.tree_fuse,  i_ax_t, ndim_mult);
	assert(subtree_s != NULL);
	assert(subtree_t != NULL);
	assert(su2_tree_equal_topology(subtree_s, subtree_t));
	// at least one of the subtrees must actually be the full tree, otherwise the resulting fusion-splitting tree is not simple
	assert((subtree_s == s->tree.tree_split) || (subtree_t == t->tree.tree_fuse));
	// at least one of the subtree roots must be an internal axis (to ensure consistency of axis enumeration)
	assert((subtree_s->i_ax >= ndim_outer_s) || (subtree_t->i_ax >= ndim_outer_t));

	const int ndim_s = su2_tensor_ndim(s);
	const int ndim_t = su2_tensor_ndim(t);

	int* i_ax_subtree_s = ct_malloc(ndim_s * sizeof(int));
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
	// retain one of the subtree roots as axis
	assert(axis_map_s[subtree_s->i_ax] == -1);
	assert(axis_map_t[subtree_t->i_ax] == -1);
	if (subtree_t->i_ax >= ndim_outer_t) {  // subtree root in 't' is an internal axis, does not need to be retained
		// retain subtree root in 's'
		axis_map_s[subtree_s->i_ax] = 0;
	}
	else {
		assert(subtree_s->i_ax >= ndim_outer_s);  // subtree root in 's' is an internal axis, does not need to be retained
		// retain subtree root in 't'
		axis_map_t[subtree_t->i_ax] = 0;
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

	struct su2_irreducible_list* outer_jlists_r = ct_calloc(ndim_outer_r, sizeof(struct su2_irreducible_list));
	for (int i = 0; i < ndim_outer_s; i++) {
		if (axis_map_s[i] == -1) {
			continue;
		}
		assert(axis_map_s[i] < ndim_outer_r);
		copy_su2_irreducible_list(&s->outer_jlists[i], &outer_jlists_r[axis_map_s[i]]);
	}
	for (int i = 0; i < ndim_outer_t; i++) {
		if (axis_map_t[i] == -1) {
			continue;
		}
		assert(axis_map_t[i] < ndim_outer_r);
		copy_su2_irreducible_list(&t->outer_jlists[i], &outer_jlists_r[axis_map_t[i]]);
	}

	long** dim_degen_r = ct_calloc(ndim_logical_r, sizeof(long*));
	for (int is = 0; is < s->ndim_logical; is++)
	{
		if (axis_map_s[is] == -1) {
			continue;
		}
		const int ir = axis_map_s[is];
		assert(0 <= ir && ir < ndim_logical_r);
		assert(outer_jlists_r[ir].num > 0);
		qnumber j_max = 0;
		for (int k = 0; k < outer_jlists_r[ir].num; k++) {
			j_max = imax(j_max, outer_jlists_r[ir].jlist[k]);
		}
		dim_degen_r[ir] = ct_malloc((j_max + 1) * sizeof(long));
		memcpy(dim_degen_r[ir], s->dim_degen[is], (j_max + 1) * sizeof(long));
	}
	for (int it = 0; it < t->ndim_logical; it++)
	{
		if (axis_map_t[it] == -1) {
			continue;
		}
		const int ir = axis_map_t[it];
		assert(0 <= ir && ir < ndim_logical_r);
		assert(outer_jlists_r[ir].num > 0);
		qnumber j_max = 0;
		for (int k = 0; k < outer_jlists_r[ir].num; k++) {
			j_max = imax(j_max, outer_jlists_r[ir].jlist[k]);
		}
		dim_degen_r[ir] = ct_malloc((j_max + 1) * sizeof(long));
		memcpy(dim_degen_r[ir], t->dim_degen[it], (j_max + 1) * sizeof(long));
	}

	allocate_su2_tensor(s->dtype, ndim_logical_r, ndim_auxiliary_r, &tree_r, outer_jlists_r, (const long**)dim_degen_r, r);

	// permutations of dense degeneracy tensors before contraction
	int* perm_s = ct_malloc(s->ndim_logical * sizeof(int));
	int c = 0;
	for (int i = 0; i < s->ndim_logical; i++) {
		if (axis_map_s[i] != -1) {
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
	int* perm_t = ct_malloc(t->ndim_logical * sizeof(int));
	c = 0;
	for (int i = 0; i < ndim_mult; i++) {
		// ignore auxiliary axes for permutation
		if (i_ax_t[i] < t->ndim_logical) {
			perm_t[c++] = i_ax_t[i];
		}
	}
	for (int i = 0; i < t->ndim_logical; i++) {
		if (axis_map_t[i] != -1) {
			perm_t[c++] = i;
		}
	}
	assert(c == t->ndim_logical);
	// number of to-be multiplied logical dimensions
	int ndim_mult_logical = 0;
	for (int i = 0; i < ndim_mult; i++) {
		// ignore auxiliary axes
		if (i_ax_s[i] < s->ndim_logical) {
			ndim_mult_logical++;
		}
	}

	// contract degeneracy tensors
	qnumber* jlist_r = ct_malloc(su2_tensor_ndim(r) * sizeof(qnumber));
	for (long cs = 0; cs < s->charge_sectors.nsec; cs++)
	{
		// current 'j' quantum numbers
		const qnumber* jlist_s = &s->charge_sectors.jlists[cs * s->charge_sectors.ndim];

		// corresponding "degeneracy" tensor
		const struct dense_tensor* ds = s->degensors[cs];
		assert(ds->dtype == s->dtype);
		assert(ds->ndim  == s->ndim_logical);
		struct dense_tensor ds_perm;
		transpose_dense_tensor(perm_s, ds, &ds_perm);

		// fill 'j' quantum numbers for charge sector in to-be contracted tensor 'r'
		for (int i = 0; i < ndim_s; i++) {
			if (axis_map_s[i] == -1) {
				continue;
			}
			jlist_r[axis_map_s[i]] = jlist_s[i];
		}

		for (long ct = 0; ct < t->charge_sectors.nsec; ct++)
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
			transpose_dense_tensor(perm_t, dt, &dt_perm);

			// fill 'j' quantum numbers for charge sector in to-be contracted tensor 'r'
			for (int i = 0; i < ndim_t; i++) {
				if (axis_map_t[i] == -1) {
					continue;
				}
				jlist_r[axis_map_t[i]] = jlist_t[i];
			}

			const long cr = charge_sector_index(&r->charge_sectors, jlist_r);
			assert(cr != -1);

			// corresponding "degeneracy" tensor
			struct dense_tensor* dr = r->degensors[cr];
			assert(dr->dtype == r->dtype);
			assert(dr->ndim  == r->ndim_logical);

			// actually multiply dense tensors and add result to 'dr'
			dense_tensor_dot_update(numeric_one(s->dtype), &ds_perm, TENSOR_AXIS_RANGE_TRAILING, &dt_perm, TENSOR_AXIS_RANGE_LEADING, ndim_mult_logical, numeric_one(s->dtype), dr);

			delete_dense_tensor(&dt_perm);
		}

		delete_dense_tensor(&ds_perm);
	}

	ct_free(perm_t);
	ct_free(perm_s);

	ct_free(jlist_r);

	for (int i = 0; i < ndim_logical_r; i++) {
		ct_free(dim_degen_r[i]);
	}
	ct_free(dim_degen_r);

	for (int i = 0; i < ndim_outer_r; i++) {
		delete_su2_irreducible_list(&outer_jlists_r[i]);
	}
	ct_free(outer_jlists_r);

	delete_su2_fuse_split_tree(&mapped_tree_t);
	delete_su2_fuse_split_tree(&mapped_tree_s);

	ct_free(axis_map_t);
	ct_free(axis_map_s);

	ct_free(i_ax_subtree_t);
	ct_free(i_ax_subtree_s);
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
	long** sector_offsets = ct_malloc(s->ndim_logical * sizeof(long*));

	// logical dimensions of dense tensor
	long* dim_t = ct_calloc(s->ndim_logical, sizeof(long));

	for (int i = 0; i < s->ndim_logical; i++)
	{
		assert(s->outer_jlists[i].num > 0);
		qnumber j_max = 0;
		for (int k = 0; k < s->outer_jlists[i].num; k++) {
			j_max = imax(j_max, s->outer_jlists[i].jlist[k]);
		}
		sector_offsets[i] = ct_calloc(j_max + 1, sizeof(long));
		for (int k = 0; k < s->outer_jlists[i].num; k++)
		{
			const qnumber j = s->outer_jlists[i].jlist[k];
			assert(s->dim_degen[i][j] > 0);
			sector_offsets[i][j] = dim_t[i];
			dim_t[i] += s->dim_degen[i][j] * (j + 1);
		}
	}

	// allocate dense tensor
	allocate_dense_tensor(s->dtype, s->ndim_logical, dim_t, t);
	ct_free(dim_t);

	// number of outer dimensions, i.e., number of leaves
	const int ndim_outer = s->ndim_logical + s->ndim_auxiliary;

	// accumulate contributions from all charge sectors
	for (long c = 0; c < s->charge_sectors.nsec; c++)
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

			long* index_t = ct_calloc(t->ndim, sizeof(long));

			// distribute degeneracy tensor entries multiplied by Clebsch-Gordan factor
			const long nelem_d = dense_tensor_num_elements(d);
			long* index_d = ct_calloc(d->ndim, sizeof(long));
			for (long l = 0; l < nelem_d; l++, next_tensor_index(d->ndim, d->dim, index_d))
			{
				// index in 't' tensor
				for (int i = 0; i < t->ndim; i++)
				{
					const qnumber j = jlist[i];
					assert(d->dim[i] == s->dim_degen[i][j]);
					index_t[i] = sector_offsets[i][j] + index_d[i] * (j + 1) + im_outer[i];
				}
				const long it = tensor_index_to_offset(t->ndim, t->dim, index_t);

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
