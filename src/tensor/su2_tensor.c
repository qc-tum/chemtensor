/// \file su2_tensor.c
/// \brief Data structures and functions for SU(2) symmetric tensors.

#include <assert.h>
#include "su2_tensor.h"
#include "su2_recoupling.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Maximum of two integers.
///
static inline int maxi(const int a, const int b)
{
	if (a >= b) {
		return a;
	}
	else {
		return b;
	}
}


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
	t->outer_jlists = aligned_alloc(MEM_DATA_ALIGN, ndim_outer * sizeof(struct su2_irreducible_list));
	for (int i = 0; i < ndim_outer; i++) {
		copy_su2_irreducible_list(&outer_jlists[i], &t->outer_jlists[i]);
	}

	su2_fuse_split_tree_enumerate_charge_sectors(&t->tree, t->outer_jlists, &t->charge_sectors);

	// degeneracy tensors
	t->dim_degen = aligned_alloc(MEM_DATA_ALIGN, t->ndim_logical * sizeof(long*));
	for (int i = 0; i < t->ndim_logical; i++)
	{
		assert(t->outer_jlists[i].num > 0);
		qnumber j_max = 0;
		for (int k = 0; k < t->outer_jlists[i].num; k++) {
			j_max = maxi(j_max, t->outer_jlists[i].jlist[k]);
		}
		t->dim_degen[i] = aligned_alloc(MEM_DATA_ALIGN, (j_max + 1) * sizeof(long));
		memcpy(t->dim_degen[i], dim_degen[i], (j_max + 1) * sizeof(long));
	}
	t->degensors = aligned_alloc(MEM_DATA_ALIGN, t->charge_sectors.nsec * sizeof(struct dense_tensor*));
	for (long c = 0; c < t->charge_sectors.nsec; c++)
	{
		// current 'j' quantum numbers
		const qnumber* jlist = &t->charge_sectors.jlists[c * t->charge_sectors.ndim];
		// dimension of degeneracy tensor
		long* dim_d = aligned_alloc(MEM_DATA_ALIGN, t->ndim_logical * sizeof(long));
		for (int i = 0; i < t->ndim_logical; i++) {
			const qnumber j = jlist[i];
			assert(t->dim_degen[i][j] > 0);
			dim_d[i] = t->dim_degen[i][j];
		}
		t->degensors[c] = aligned_calloc(MEM_DATA_ALIGN, 1, sizeof(struct dense_tensor));
		allocate_dense_tensor(t->dtype, t->ndim_logical, dim_d, t->degensors[c]);
		aligned_free(dim_d);
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
		aligned_free(t->degensors[c]);
	}
	aligned_free(t->degensors);
	t->degensors = NULL;
	for (int i = 0; i < t->ndim_logical; i++)
	{
		aligned_free(t->dim_degen[i]);
	}
	aligned_free(t->dim_degen);
	t->dim_degen = NULL;

	delete_charge_sectors(&t->charge_sectors);

	const int ndim_outer = t->ndim_logical + t->ndim_auxiliary;
	for (int i = 0; i < ndim_outer; i++) {
		delete_su2_irreducible_list(&t->outer_jlists[i]);
	}
	aligned_free(t->outer_jlists);
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
	long** sector_offsets = aligned_alloc(MEM_DATA_ALIGN, s->ndim_logical * sizeof(long*));

	// logical dimensions of dense tensor
	long* dim_t = aligned_calloc(MEM_DATA_ALIGN, s->ndim_logical, sizeof(long));

	for (int i = 0; i < s->ndim_logical; i++)
	{
		assert(s->outer_jlists[i].num > 0);
		qnumber j_max = 0;
		for (int k = 0; k < s->outer_jlists[i].num; k++) {
			j_max = maxi(j_max, s->outer_jlists[i].jlist[k]);
		}
		sector_offsets[i] = aligned_calloc(MEM_DATA_ALIGN, j_max + 1, sizeof(long));
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
	aligned_free(dim_t);

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
		int* dim_m_outer = aligned_alloc(MEM_DATA_ALIGN, ndim_outer * sizeof(int));
		int nconfigs = 1;
		for (int k = 0; k < ndim_outer; k++) {
			dim_m_outer[k] = jlist[k] + 1;
			nconfigs *= dim_m_outer[k];
		}

		// iterate over outer 'm' quantum numbers; auxiliary 'm' quantum numbers are traced out
		int* im_outer = aligned_calloc(MEM_DATA_ALIGN, ndim_outer, sizeof(int));
		for (int k = 0; k < nconfigs; k++, next_quantum_index(ndim_outer, dim_m_outer, im_outer))
		{
			// evaluate Clebsch-Gordan coefficients of tree nodes
			double cg = su2_fuse_split_tree_eval_clebsch_gordan(&s->tree, jlist, im_outer);
			if (cg == 0) {
				continue;
			}

			long* index_t = aligned_calloc(MEM_DATA_ALIGN, t->ndim, sizeof(long));

			// distribute degeneracy tensor entries multiplied by Clebsch-Gordan factor
			const long nelem_d = dense_tensor_num_elements(d);
			long* index_d = aligned_calloc(MEM_DATA_ALIGN, d->ndim, sizeof(long));
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
					case SINGLE_REAL:
					{
						const float* ddata = d->data;
						float*       tdata = t->data;
						tdata[it] += ((float)cg) * ddata[l];
						break;
					}
					case DOUBLE_REAL:
					{
						const double* ddata = d->data;
						double*       tdata = t->data;
						tdata[it] += cg * ddata[l];
						break;
					}
					case SINGLE_COMPLEX:
					{
						const scomplex* ddata = d->data;
						scomplex*       tdata = t->data;
						tdata[it] += ((float)cg) * ddata[l];
						break;
					}
					case DOUBLE_COMPLEX:
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

			aligned_free(index_d);
			aligned_free(index_t);
		}

		aligned_free(im_outer);
		aligned_free(dim_m_outer);
	}

	for (int i = 0; i < s->ndim_logical; i++) {
		aligned_free(sector_offsets[i]);
	}
	aligned_free(sector_offsets);
}
