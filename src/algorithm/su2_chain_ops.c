/// \file su2_chain_ops.c
/// \brief Higher-level tensor network operations on a chain topology for SU(2) symmetric tensors.

#include <memory.h>
#include <assert.h>
#include "su2_chain_ops.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Create a dummy right operator block for computing the operator average of SU(2) symmetric tensors.
///
static void su2_create_dummy_operator_block_right(const struct su2_tensor* restrict a, const struct su2_tensor* restrict b,
	const struct su2_tensor* restrict w, struct su2_tensor* restrict r)
{
	assert(a->dtype == b->dtype);
	assert(a->dtype == w->dtype);

	assert(a->ndim_logical == 3);
	assert(b->ndim_logical == 3);
	assert(w->ndim_logical == 4);

	// construct the fuse and split tree
	//
	//  5   4    3
	//   ╲   ╲  ╱
	//    ╲   ╲╱       fuse
	//     ╲  ╱7
	//      ╲╱
	//      │
	//      │8
	//      │
	//      ╱╲
	//     ╱  ╲6       split
	//    ╱   ╱╲
	//   ╱   ╱  ╲
	//  2   1    0
	//
	struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
	struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
	struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
	struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
	struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
	struct su2_tree_node j5  = { .i_ax = 5, .c = { NULL, NULL } };
	struct su2_tree_node j6  = { .i_ax = 6, .c = { &j1,  &j0  } };
	struct su2_tree_node j7  = { .i_ax = 7, .c = { &j4,  &j3  } };
	struct su2_tree_node j8f = { .i_ax = 8, .c = { &j5,  &j7  } };
	struct su2_tree_node j8s = { .i_ax = 8, .c = { &j2,  &j6  } };
	struct su2_fuse_split_tree tree = { .tree_fuse = &j8f, .tree_split = &j8s, .ndim = 9 };
	assert(su2_fuse_split_tree_is_consistent(&tree));

	// outer (logical and auxiliary) 'j' quantum numbers
	const struct su2_irreducible_list outer_irreps[6] = {
		a->outer_irreps[2],
		w->outer_irreps[3],
		b->outer_irreps[2],
		a->outer_irreps[2],
		w->outer_irreps[3],
		b->outer_irreps[2],
	};
	// degeneracy dimensions
	const ct_long* dim_degen[6] = {
		a->dim_degen[2],
		w->dim_degen[3],
		b->dim_degen[2],
		a->dim_degen[2],
		w->dim_degen[3],
		b->dim_degen[2],
	};

	struct su2_tensor t;
	allocate_su2_tensor(a->dtype, 6, 0, &tree, outer_irreps, dim_degen, &t);

	ct_long c = 0;
	while (c < t.charge_sectors.nsec)
	{
		// current 'j' quantum numbers
		const qnumber* jlist = &t.charge_sectors.jlists[c * t.charge_sectors.ndim];
		if (jlist[0] == jlist[3] && jlist[1] == jlist[4] && jlist[2] == jlist[5])
		{
			assert(t.degensors[c]->ndim == 6);
			assert(t.degensors[c]->dim[0] == t.degensors[c]->dim[3]);
			assert(t.degensors[c]->dim[1] == t.degensors[c]->dim[4]);
			assert(t.degensors[c]->dim[2] == t.degensors[c]->dim[5]);

			// set degeneracy tensor to identity
			// save original dimensions
			const ct_long dim[6] = {
				t.degensors[c]->dim[0], t.degensors[c]->dim[1], t.degensors[c]->dim[2],
				t.degensors[c]->dim[3], t.degensors[c]->dim[4], t.degensors[c]->dim[5],
			};
			const ct_long dim_mat[2] = {
				dim[0] * dim[1] * dim[2],
				dim[3] * dim[4] * dim[5],
			};
			reshape_dense_tensor(2, dim_mat, t.degensors[c]);
			dense_tensor_set_identity(t.degensors[c]);
			reshape_dense_tensor(6, dim, t.degensors[c]);

			c++;
		}
		else
		{
			su2_tensor_delete_charge_sector_by_index(&t, c);
		}
	}

	// fuse axes 3, 4, 5 into one axis
	struct su2_tensor s;
	su2_tensor_fuse_axes(&t, 3, 4, &s);
	delete_su2_tensor(&t);
	su2_tensor_fuse_axes(&s, 3, 4, r);
	delete_su2_tensor(&s);

	// reverse direction of axis 2
	su2_tensor_reverse_axis_simple(r, 2);
}


//________________________________________________________________________________________________________________________
///
/// \brief Contraction step from right to left for SU(2) symmetric tensors, with a matrix product operator sandwiched in between.
///
/// To-be contracted tensor network:
///
///        ╭───────╮         ╭─────────╮
///        │   b*  │         │         │
///     ─>─0───┬───2─>─   ─>─2─╮     ╭─3─<─
///        │   │   │         │  ╲   ╱  │
///        ╰───1───╯         │   ╲ ╱   │
///            │             │    │    │
///            ^             │    │    │
///            │             │    │    │
///        ╭───1───╮         │    │    │
///        │   │   │         │   ╱ ╲   │
///     ─<─0───┤   │         │  ╱ r ╲  │
///        │ w ├───3─<─   ─<─1─╯     │ │
///        │   │   │         │       │ │
///        ╰───2───╯         │       │ │
///            │             │       │ │
///            ^             │       │ │
///            │             │       │ │
///        ╭───1───╮         │       │ │
///        │   │   │         │       │ │
///     ─<─0───┴───2─<─   ─<─0───────╯ │
///        │   a   │         │         │
///        ╰───────╯         ╰─────────╯
///
static void su2_contraction_operator_step_right(const struct su2_tensor* restrict a, const struct su2_tensor* restrict b,
	const struct su2_tensor* restrict w, const struct su2_tensor* restrict r, struct su2_tensor* restrict r_next)
{
	assert(a->ndim_logical == 3);
	assert(b->ndim_logical == 3);
	assert(w->ndim_logical == 4);
	assert(r->ndim_logical == 4);
	assert(r->tree.tree_split->c[1]->i_ax == 0);
	assert(r->tree.tree_split->c[0]->i_ax == 1);
	assert( r->tree.tree_fuse->c[0]->i_ax == 2);
	assert( r->tree.tree_fuse->c[1]->i_ax == 3);

	// multiply with 'a' tensor
	struct su2_tensor s;
	{
		const int i_ax_a[1] = { 2 };
		const int i_ax_r[1] = { 0 };
		su2_tensor_contract_simple(a, i_ax_a, r, i_ax_r, 1, &s);
	}

	// tree of current tensor 's'
	// (with 0 the left virtual bond and 1 the physical axis of the original 'a'):
	//
	//    3    4
	//     ╲  ╱        fuse
	//      ╲╱
	//      │
	//      │5
	//      │
	//      ╱╲
	//     ╱  ╲6
	//    ╱   ╱╲       split
	//   ╱   ╱  ╲
	//  2   0    1

	// multiply with 'w' tensor
	{
		// swap axes 0 <-> 1 in the tree and perform an F-move to group axes 1 and 2 together
		su2_tensor_swap_tree_axes(&s, 0, 1);
		struct su2_tensor t;
		// common axis from contraction with 'a' becomes last internal axis in fusion-splitting tree
		su2_tensor_fmove(&s, su2_tensor_ndim(&s) - 1, &t);
		delete_su2_tensor(&s);
		su2_tensor_swap_tree_axes(&t, 1, 2);

		const int i_ax_w[2] = { 2, 3 };
		const int i_ax_t[2] = { 1, 2 };
		su2_tensor_contract_simple(w, i_ax_w, &t, i_ax_t, 2, &s);
		delete_su2_tensor(&t);
	}

	// tree of current tensor 's'
	// (with 0 the left virtual bond and 1 the physical output axis of the original 'w'):
	//
	//    3    4
	//     ╲  ╱        fuse
	//      ╲╱
	//      │
	//      │6
	//      │
	//      ╱╲
	//    5╱  ╲
	//    ╱╲   ╲       split
	//   ╱  ╲   ╲
	//  0    1   2

	// multiply with conjugated 'b' tensor
	{
		// swap axes 0 <-> 1 in the tree and perform an F-move to isolate axis 1
		su2_tensor_swap_tree_axes(&s, 0, 1);
		struct su2_tensor t;
		su2_tensor_fmove(&s, s.tree.tree_split->c[0]->i_ax, &t);
		delete_su2_tensor(&s);
		// reverse axis 1 (physical output axis of 'w')
		su2_tensor_reverse_axis_simple(&t, 1);
		// perform an F-move to group axes 1 and 3
		su2_tensor_fmove(&t, t.tree.tree_fuse->c[1]->i_ax, &s);
		delete_su2_tensor(&t);

		// flip fusion-splitting tree, complex-conjugate entries, reverse axis 1 and swap 1 <-> 2 in 'b'
		struct su2_tensor bdag;
		copy_su2_tensor(b, &bdag);
		su2_tensor_flip_trees(&bdag);
		conjugate_su2_tensor(&bdag);
		su2_tensor_reverse_axis_simple(&bdag, 1);
		su2_tensor_swap_tree_axes(&bdag, 1, 2);

		// actually contract 's' with adjoint of 'b'
		const int i_ax_s[2] = { 1, 3 };
		const int i_ax_b[2] = { 1, 2 };
		su2_tensor_contract_simple(&s, i_ax_s, &bdag, i_ax_b, 2, &t);
		delete_su2_tensor(&s);
		delete_su2_tensor(&bdag);

		assert(t.ndim_logical == 4);
		const int perm[4] = { 1, 0, 3, 2 };
		su2_tensor_transpose_logical(perm, &t, r_next);
		delete_su2_tensor(&t);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the inner product between an SU(2) symmetric MPO and two MPS, `<chi | op | psi>`.
///
/// The result is returned as an SU(2) symmetric tensor with the tree layout (and 'aux' pointing outwards and having quantum number 0):
///
///        ╭───────╮
///        │       │
///     ─<─0───┬───1─<─
///        │   │   │
///        ╰───2───╯
///            │
///           aux
///
void su2_mpo_inner_product(const struct su2_mps* chi, const struct su2_mpo* op, const struct su2_mps* psi, struct su2_tensor* ret)
{
	const int nsites = op->nsites;
	assert(chi->nsites == nsites);
	assert(psi->nsites == nsites);
	assert(nsites >= 1);

	// initialize 'r'
	struct su2_tensor r;
	su2_create_dummy_operator_block_right(&psi->a[nsites - 1], &chi->a[nsites - 1], &op->a[nsites - 1], &r);

	for (int i = nsites - 1; i >= 0; i--)
	{
		struct su2_tensor r_next;
		su2_contraction_operator_step_right(&psi->a[i], &chi->a[i], &op->a[i], &r, &r_next);
		delete_su2_tensor(&r);
		r = r_next;  // copy internal data pointers
	}

	// fuse left virtual bonds
	{
		// reverse direction of axis 2
		su2_tensor_reverse_axis_simple(&r, 2);

		struct su2_tensor t;
		su2_tensor_fuse_axes(&r, 0, 1, &t);
		delete_su2_tensor(&r);
		// previous axis 2 is now axis 1

		// fuse axes 0 and 1 and add an auxiliary axis (to retain a minimum of 3 outer axes)
		su2_tensor_fuse_axes_add_auxiliary(&t, 0, 1, ret);
		delete_su2_tensor(&t);
	}
	assert(ret->ndim_logical == 2);

	// retain SU(2) information in 'ret'
}
