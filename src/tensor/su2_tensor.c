/// \file su2_tensor.c
/// \brief Data structures and functions for SU(2) symmetric tensors.

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
void allocate_empty_su2_tensor(const enum numeric_type dtype, const int ndim_logical, int ndim_auxiliary, const struct su2_fuse_split_tree* tree, const struct su2_irreducible_list* outer_irreps, const long** dim_degen, struct su2_tensor* t)
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

	t->dim_degen = ct_malloc(t->ndim_logical * sizeof(long*));
	for (int i = 0; i < t->ndim_logical; i++)
	{
		assert(t->outer_irreps[i].num > 0);
		qnumber j_max = 0;
		for (int k = 0; k < t->outer_irreps[i].num; k++) {
			j_max = qmax(j_max, t->outer_irreps[i].jlist[k]);
		}
		t->dim_degen[i] = ct_malloc((j_max + 1) * sizeof(long));
		memcpy(t->dim_degen[i], dim_degen[i], (j_max + 1) * sizeof(long));
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for an SU(2) symmetric tensor, including the dense "degeneracy" tensors, and enumerate all possible charge sectors.
///
void allocate_su2_tensor(const enum numeric_type dtype, const int ndim_logical, int ndim_auxiliary, const struct su2_fuse_split_tree* tree, const struct su2_irreducible_list* outer_irreps, const long** dim_degen, struct su2_tensor* t)
{
	allocate_empty_su2_tensor(dtype, ndim_logical, ndim_auxiliary, tree, outer_irreps, dim_degen, t);

	su2_fuse_split_tree_enumerate_charge_sectors(&t->tree, t->outer_irreps, &t->charge_sectors);

	// degeneracy tensors
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
		delete_su2_irreducible_list(&t->outer_irreps[i]);
	}
	ct_free(t->outer_irreps);
	t->outer_irreps = NULL;

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
	for (long c = 0; c < t->charge_sectors.nsec - 1; c++)
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
/// \brief Delete an individual charge sector of the SU(2) symmetric tensor,
/// returning true if the charge sector was actually found and deleted.
///
bool su2_tensor_delete_charge_sector(struct su2_tensor* t, const qnumber* jlist)
{
	const long idx = charge_sector_index(&t->charge_sectors, jlist);
	if (idx < 0) {
		// not found
		return false;
	}

	// delete corresponding "degeneracy" tensor
	delete_dense_tensor(t->degensors[idx]);
	ct_free(t->degensors[idx]);

	const int ndim = t->charge_sectors.ndim;
	for (long c = idx; c < t->charge_sectors.nsec - 1; c++)
	{
		memcpy(&t->charge_sectors.jlists[c * ndim], &t->charge_sectors.jlists[(c + 1) * ndim], ndim * sizeof(qnumber));
		// copy pointer
		t->degensors[c] = t->degensors[c + 1];
	}
	t->degensors[t->charge_sectors.nsec - 1] = NULL;

	t->charge_sectors.nsec--;

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
	allocate_su2_tensor(t->dtype, t->ndim_logical, t->ndim_auxiliary, &tree, t->outer_irreps, (const long**)t->dim_degen, r);
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
			const double coeff = su2_recoupling_coefficient(ja, jb, jc, js, je, jf);
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
/// \brief Fuse the two logical axes (tensor legs) 'i_ax_0' and 'i_ax_1' (with i_ax_0 < i_ax_1) into a single axis with index 'i_ax_0',
/// corresponding to the contraction with a "fusion" Clebsch-Gordan node.
/// The axes must have the same direction and must be direct siblings in the fusion-splitting tree.
/// The parent node of (i_ax_0, i_ax_1) in the fusion-splitting tree is deleted, and its (internal) upward axis index becomes 'i_ax_0'.
///
void su2_tensor_fuse_axes(const struct su2_tensor* restrict t, const int i_ax_0, const int i_ax_1, struct su2_tensor* restrict r)
{
	assert(i_ax_0 < i_ax_1);
	// must be logical axes
	assert(0 <= i_ax_0 && i_ax_0 < t->ndim_logical);
	assert(0 <= i_ax_1 && i_ax_1 < t->ndim_logical);
	// resulting tensor must have at least 3 outer axes for a valid fusion-splitting tree
	assert(t->ndim_logical + t->ndim_auxiliary > 3);

	const bool in_fuse_tree = su2_tree_contains_leaf(t->tree.tree_fuse, i_ax_0);
	// axes must both be contained either in fusion or in splitting tree
	assert(in_fuse_tree == su2_tree_contains_leaf(t->tree.tree_fuse, i_ax_1));

	const int ndim_t = su2_tensor_ndim(t);
	const int ndim_outer_t = t->ndim_logical + t->ndim_auxiliary;

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
	long* offset_map = ct_calloc((j_max_fused + 1) * (j_max_0 + 1) * (j_max_1 + 1), sizeof(long));

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

		long* dim_degen_fused = ct_calloc(j_max_fused + 1, sizeof(long));
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

		const long** dim_degen_r = ct_malloc((t->ndim_logical - 1) * sizeof(long*));
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
	for (long ct = 0; ct < t->charge_sectors.nsec; ct++)
	{
		// current 'j' quantum numbers
		const qnumber* jlist_t = &t->charge_sectors.jlists[ct * t->charge_sectors.ndim];

		for (int i = 0; i < ndim_t; i++) {
			if (axis_map[i] != -1) {
				jlist_r[axis_map[i]] = jlist_t[i];
			}
		}

		const long cr = charge_sector_index(&r->charge_sectors, jlist_r);
		assert(cr != -1);

		const qnumber j0 = jlist_t[i_ax_0];
		const qnumber j1 = jlist_t[i_ax_1];
		const qnumber jp = jlist_t[i_ax_p];
		assert(j0 <= j_max_0);
		assert(j1 <= j_max_1);
		assert((abs(j0 - j1) <= jp) && (jp <= j0 + j1));
		assert((j0 + j1 + jp) % 2 == 0);

		const long offset = offset_map[(jp*(j_max_0 + 1) + j0)*(j_max_1 + 1) + j1];

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
			transpose_dense_tensor(perm, dt, &dt_perm);
		}

		// corresponding "degeneracy" tensor of 'r'
		if (r->degensors[cr] == NULL)
		{
			// allocate new degeneracy tensor with zero entries
			// dimension of degeneracy tensor
			long* dim_d = ct_malloc(r->ndim_logical * sizeof(long));
			for (int i = 0; i < r->ndim_logical; i++) {
				const qnumber j = jlist_r[i];
				assert(r->dim_degen[i][j] > 0);
				dim_d[i] = r->dim_degen[i][j];
			}
			r->degensors[cr] = ct_calloc(1, sizeof(struct dense_tensor));
			allocate_dense_tensor(r->dtype, r->ndim_logical, dim_d, r->degensors[cr]);
			ct_free(dim_d);
		}
		struct dense_tensor* dr = r->degensors[cr];
		assert(dr->dtype == r->dtype);
		assert(dr->ndim  == r->ndim_logical);

		// copy tensor entries
		const long n = integer_product(dr->dim, i_ax_0);
		assert(n == integer_product(dt_perm.dim, i_ax_0));
		// trailing dimensions times data type size
		const long tdd_r = integer_product(dr->dim + (i_ax_0 + 1), dr->ndim - (i_ax_0 + 1)) * dtype_size;
		const long offset_r = offset*tdd_r;
		const long stride_t = integer_product(dt_perm.dim + i_ax_0, dt_perm.ndim - i_ax_0) * dtype_size;
		const long stride_r = dr->dim[i_ax_0] * tdd_r;
		for (long i = 0; i < n; i++)
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
	long c = 0;
	for (; c < r->charge_sectors.nsec; c++) {
		if (r->degensors[c] == NULL) {
			break;
		}
	}
	for (long s = c + 1; s < r->charge_sectors.nsec; s++)
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
/// 'tree_left_child' indicates whether the new axis 'i_ax_split' is the left child in the fusion-splitting tree.
/// The arguments 'outer_irreps' and 'dim_degen' specify the 'j' quantum numbers and degeneracies of the new axes 'i_ax_split' and 'i_ax_add'.
///
/// The axis 'i_ax_split' in the original fusion-splitting tree becomes the last inner axis of the new tensor.
///
void su2_tensor_split_axis(const struct su2_tensor* restrict t, const int i_ax_split, const int i_ax_add, const bool tree_left_child, const struct su2_irreducible_list outer_irreps[2], const long* dim_degen[2], struct su2_tensor* restrict r)
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
	long* offset_map = ct_calloc((j_max_split + 1) * (j_max_0 + 1) * (j_max_1 + 1), sizeof(long));
	long* dim_degen_split = ct_calloc(j_max_split + 1, sizeof(long));
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
	// check consistency of quantum numbers end degeneracies
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

		const long** dim_degen_r = ct_malloc((t->ndim_logical + 1) * sizeof(long*));
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
	assert(perm_is_identity == is_identity_permutation(perm, r->ndim_logical));

	const size_t dtype_size = sizeof_numeric_type(t->dtype);

	qnumber* jlist_t = ct_malloc(ndim_t * sizeof(qnumber));
	for (long cr = 0; cr < r->charge_sectors.nsec; cr++)
	{
		// current 'j' quantum numbers
		const qnumber* jlist_r = &r->charge_sectors.jlists[cr * r->charge_sectors.ndim];

		for (int i = 0; i < ndim_t; i++)
		{
			assert(axis_map[i] != i_ax_split);
			assert(axis_map[i] != i_ax_add);
			jlist_t[i] = jlist_r[axis_map[i]];
		}

		const long ct = charge_sector_index(&t->charge_sectors, jlist_t);
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

		const long offset = offset_map[(js*(j_max_0 + 1) + j0)*(j_max_1 + 1) + j1];

		// corresponding "degeneracy" tensor of 't'
		const struct dense_tensor* dt = t->degensors[ct];
		assert(dt != NULL);
		assert(dt->dtype == t->dtype);
		assert(dt->ndim  == t->ndim_logical);

		// corresponding (permuted) "degeneracy" tensor of 'r'
		struct dense_tensor dr_perm;
		{
			// dimension of permuted degeneracy tensor
			long* dim_d = ct_malloc(r->ndim_logical * sizeof(long));
			for (int i = 0; i < r->ndim_logical; i++) {
				const qnumber j = jlist_r[i];
				assert(r->dim_degen[i][j] > 0);
				dim_d[perm[i]] = r->dim_degen[i][j];
			}
			allocate_dense_tensor(r->dtype, r->ndim_logical, dim_d, &dr_perm);
			ct_free(dim_d);
		}

		// copy tensor entries
		const long n = integer_product(dt->dim, i_ax_split);
		assert(n == integer_product(dr_perm.dim, i_ax_split));
		// trailing dimensions times data type size
		const long tdd_t = integer_product(dt->dim + (i_ax_split + 1), dt->ndim - (i_ax_split + 1)) * dtype_size;
		const long offset_t = offset*tdd_t;
		const long stride_t = dt->dim[i_ax_split] * tdd_t;
		const long stride_r = integer_product(dr_perm.dim + i_ax_split, dr_perm.ndim - i_ax_split) * dtype_size;
		for (long i = 0; i < n; i++)
		{
			// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
			memcpy((int8_t*)dr_perm.data + i*stride_r,
			       (int8_t*)dt->data + (i*stride_t + offset_t),
			       stride_r);
		}

		r->degensors[cr] = ct_malloc(sizeof(struct dense_tensor));
		if (perm_is_identity)
		{
			move_dense_tensor_data(&dr_perm, r->degensors[cr]);
		}
		else
		{
			transpose_dense_tensor(perm, &dr_perm, r->degensors[cr]);
			delete_dense_tensor(&dr_perm);
		}
	}

	ct_free(jlist_t);
	ct_free(perm);
	ct_free(offset_map);
	ct_free(axis_map);

	// condense charge sector quantum numbers and corresponding degeneracy tensors
	// find first unused charge sector
	long c = 0;
	for (; c < r->charge_sectors.nsec; c++) {
		if (r->degensors[c] == NULL) {
			break;
		}
	}
	for (long s = c + 1; s < r->charge_sectors.nsec; s++)
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

	struct su2_irreducible_list* outer_irreps_r = ct_calloc(ndim_outer_r, sizeof(struct su2_irreducible_list));
	for (int i = 0; i < ndim_outer_s; i++) {
		if (axis_map_s[i] == -1) {
			continue;
		}
		assert(axis_map_s[i] < ndim_outer_r);
		copy_su2_irreducible_list(&s->outer_irreps[i], &outer_irreps_r[axis_map_s[i]]);
	}
	for (int i = 0; i < ndim_outer_t; i++) {
		if (axis_map_t[i] == -1) {
			continue;
		}
		assert(axis_map_t[i] < ndim_outer_r);
		copy_su2_irreducible_list(&t->outer_irreps[i], &outer_irreps_r[axis_map_t[i]]);
	}

	long** dim_degen_r = ct_calloc(ndim_logical_r, sizeof(long*));
	for (int is = 0; is < s->ndim_logical; is++)
	{
		if (axis_map_s[is] == -1) {
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
		if (axis_map_t[it] == -1) {
			continue;
		}
		const int ir = axis_map_t[it];
		assert(0 <= ir && ir < ndim_logical_r);
		assert(outer_irreps_r[ir].num > 0);
		// simply copy the pointer
		dim_degen_r[ir] = t->dim_degen[it];
	}

	allocate_su2_tensor(s->dtype, ndim_logical_r, ndim_auxiliary_r, &tree_r, outer_irreps_r, (const long**)dim_degen_r, r);

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
	assert(ndim_mult_logical >= 1);

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

	ct_free(dim_degen_r);

	for (int i = 0; i < ndim_outer_r; i++) {
		delete_su2_irreducible_list(&outer_irreps_r[i]);
	}
	ct_free(outer_irreps_r);

	delete_su2_fuse_split_tree(&mapped_tree_t);
	delete_su2_fuse_split_tree(&mapped_tree_s);

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

	long** dim_degen_r = ct_calloc(ndim_logical_r, sizeof(long*));
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
	long nsec_r = 0;
	for (long j = 0; j < s->charge_sectors.nsec; j++)
	{
		for (long k = 0; k < t->charge_sectors.nsec; k++)
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
	for (long i = 0; i < nsec_r; i++)
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
	for (long cs = 0; cs < s->charge_sectors.nsec; cs++)
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
			transpose_dense_tensor(perm_s, ds, &ds_perm);
		}

		// fill 'j' quantum numbers for merged charge sector
		for (int i = 0; i < ndim_s; i++) {
			jlist_r[axis_map_s[i]] = jlist_s[i];
		}

		for (long ct = 0; ct < t->charge_sectors.nsec; ct++)
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
				transpose_dense_tensor(perm_t, dt, &dt_perm);
			}

			// fill 'j' quantum numbers for merged charge sector
			for (int i = 0; i < ndim_t; i++) {
				jlist_r[axis_map_t[i]] = jlist_t[i];
			}

			const long cr = charge_sector_index(&data_yoga.charge_sectors, jlist_r);
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

	for (long i = 0; i < nsec_r; i++)
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

	allocate_empty_su2_tensor(s->dtype, ndim_logical_r, ndim_auxiliary_r, &tree_r, outer_irreps_r, (const long**)dim_degen_r, r);
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
	long** sector_offsets = ct_malloc(s->ndim_logical * sizeof(long*));

	// logical dimensions of dense tensor
	long* dim_t = ct_calloc(s->ndim_logical, sizeof(long));

	for (int i = 0; i < s->ndim_logical; i++)
	{
		assert(s->outer_irreps[i].num > 0);
		qnumber j_max = 0;
		for (int k = 0; k < s->outer_irreps[i].num; k++) {
			j_max = qmax(j_max, s->outer_irreps[i].jlist[k]);
		}
		sector_offsets[i] = ct_calloc(j_max + 1, sizeof(long));
		for (int k = 0; k < s->outer_irreps[i].num; k++)
		{
			const qnumber j = s->outer_irreps[i].jlist[k];
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
