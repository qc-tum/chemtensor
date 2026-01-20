/// \file su2_chain_ops.c
/// \brief Higher-level tensor network operations on a chain topology for SU(2) symmetric tensors.

#include <memory.h>
#include <assert.h>
#include "su2_chain_ops.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Create a dummy right operator block for computing the operator average of SU(2) symmetric tensors,
/// assuming that the degeneracy dimensions of all trailing virtual bonds are 1, and
/// that the 'j' quantum numbers of the trailing virtual bonds are 'irrep_sector_state' for the states and 0 for the operator.
///
void su2_create_dummy_operator_block_right(const enum numeric_type dtype, const qnumber irrep_sector_state, struct su2_tensor* r)
{
	assert(irrep_sector_state >= 0);

	// construct the fuse and split tree
	//
	//  2    3
	//   ╲  ╱
	//    ╲╱   fuse
	//    │
	//    │4
	//    │
	//    ╱╲   split
	//   ╱  ╲
	//  0    1
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
	qnumber jlist_zero[1]  = { 0 };
	qnumber jlist_state[1] = { irrep_sector_state };
	const struct su2_irreducible_list outer_irreps[4] = {
		{ .jlist = jlist_state, .num = 1 },  // virtual bond connected to 'a'
		{ .jlist = jlist_zero,  .num = 1 },  // virtual bond connected to 'w'
		{ .jlist = jlist_state, .num = 1 },  // virtual bond connected to conjugated 'b'
		{ .jlist = jlist_zero,  .num = 1 },  // outer virtual bond
	};
	// degeneracy dimensions
	const ct_long dim_degen_0_one[1] = { 1 };
	ct_long* dim_degen_s_one = ct_calloc(irrep_sector_state + 1, sizeof(ct_long));
	dim_degen_s_one[irrep_sector_state] = 1;
	const ct_long* dim_degen[4] = {
		dim_degen_s_one,
		dim_degen_0_one,
		dim_degen_s_one,
		dim_degen_0_one
	};

	allocate_su2_tensor(dtype, 4, 0, &tree, outer_irreps, dim_degen, r);

	ct_free(dim_degen_s_one);

	// set single entry of degeneracy tensor to 1
	assert(r->charge_sectors.nsec == 1);
	assert(r->degensors[0] != NULL);
	assert(dense_tensor_num_elements(r->degensors[0]) == 1);
	memcpy(r->degensors[0]->data, numeric_one(dtype), sizeof_numeric_type(dtype));
}


//________________________________________________________________________________________________________________________
///
/// \brief Create a dummy left operator block for computing the operator average of SU(2) symmetric tensors,
/// assuming that all leading virtual bonds are "trivial" (quantum number 0 and degeneracy dimension 1).
///
void su2_create_dummy_operator_block_left(const enum numeric_type dtype, struct su2_tensor* l)
{
	// construct the fuse and split tree
	//
	//  1    2
	//   ╲  ╱
	//    ╲╱   fuse
	//    │
	//    │4
	//    │
	//    ╱╲   split
	//   ╱  ╲
	//  0    3
	//
	struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
	struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
	struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
	struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
	struct su2_tree_node j4f = { .i_ax = 4, .c = { &j1,  &j2  } };
	struct su2_tree_node j4s = { .i_ax = 4, .c = { &j0,  &j3  } };
	struct su2_fuse_split_tree tree = { .tree_fuse = &j4f, .tree_split = &j4s, .ndim = 5 };
	assert(su2_fuse_split_tree_is_consistent(&tree));

	// outer (logical and auxiliary) 'j' quantum numbers
	qnumber jlist_zero[1]  = { 0 };
	const struct su2_irreducible_list outer_irreps[4] = {
		{ .jlist = jlist_zero, .num = 1 },  // outer virtual bond
		{ .jlist = jlist_zero, .num = 1 },  // virtual bond connected to 'a'
		{ .jlist = jlist_zero, .num = 1 },  // virtual bond connected to 'w'
		{ .jlist = jlist_zero, .num = 1 },  // virtual bond connected to conjugated 'b'
	};
	// degeneracy dimensions
	const ct_long dim_degen_one[1] = { 1 };
	const ct_long* dim_degen[4] = {
		dim_degen_one,
		dim_degen_one,
		dim_degen_one,
		dim_degen_one,
	};

	allocate_su2_tensor(dtype, 4, 0, &tree, outer_irreps, dim_degen, l);

	// set single entry of degeneracy tensor to 1
	assert(l->charge_sectors.nsec == 1);
	assert(l->degensors[0] != NULL);
	assert(dense_tensor_num_elements(l->degensors[0]) == 1);
	memcpy(l->degensors[0]->data, numeric_one(dtype), sizeof_numeric_type(dtype));
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
///        │ w ├───3─<─   ─<─1─│─────╯ │
///        │   │   │         │ │       │
///        ╰───2───╯         │ │       │
///            │             │ │       │
///            ^             │ │       │
///            │             │ │       │
///        ╭───1───╮         │ │       │
///        │   │   │         │ │       │
///     ─<─0───┴───2─<─   ─<─0─╯       │
///        │   a   │         │         │
///        ╰───────╯         ╰─────────╯
///
void su2_contraction_operator_step_right(const struct su2_tensor* restrict a, const struct su2_tensor* restrict b,
	const struct su2_tensor* restrict w, const struct su2_tensor* restrict r, struct su2_tensor* restrict r_next)
{
	assert(a->ndim_logical == 3);
	assert(b->ndim_logical == 3);
	assert(w->ndim_logical == 4);
	assert(r->ndim_logical == 4);
	assert(r->tree.tree_split->c[0]->i_ax == 0);
	assert(r->tree.tree_split->c[1]->i_ax == 1);
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
	//    6╱  ╲
	//    ╱╲   ╲       split
	//   ╱  ╲   ╲
	//  0    1   2

	// multiply with 'w' tensor
	{
		// perform an F-move to group axes 1 and 2 together
		struct su2_tensor t;
		assert(s.tree.tree_split->c[0]->i_ax == 6);
		su2_tensor_fmove(&s, 6, &t);
		delete_su2_tensor(&s);

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
	//     ╱  ╲5
	//    ╱   ╱╲       split
	//   ╱   ╱  ╲
	//  2   0    1

	// multiply with conjugated 'b' tensor
	{
		// perform an F-move to isolate axis 1
		struct su2_tensor t;
		assert(s.tree.tree_split->c[1]->i_ax == 5);
		su2_tensor_fmove(&s, 5, &t);
		delete_su2_tensor(&s);
		// reverse axis 1 (physical output axis of 'w')
		su2_tensor_reverse_axis_simple(&t, 1);
		// swap 3 <-> 4 and perform an F-move to group axes 1 and 3
		su2_tensor_swap_tree_axes(&t, 3, 4);
		assert(t.tree.tree_fuse->c[0]->i_ax == 6);
		su2_tensor_fmove(&t, 6, &s);
		delete_su2_tensor(&t);

		// flip fusion-splitting tree, complex-conjugate entries and reverse axis 1 in 'b'
		struct su2_tensor bdag;
		// TODO: avoid full copy
		copy_su2_tensor(b, &bdag);
		// reverse axis before tree flip to avoid spurious (-1) factor
		su2_tensor_reverse_axis_simple(&bdag, 1);
		su2_tensor_flip_trees(&bdag);
		conjugate_su2_tensor(&bdag);

		// actually contract 's' with adjoint of 'b'
		const int i_ax_s[2] = { 1, 3 };
		const int i_ax_b[2] = { 1, 2 };
		su2_tensor_contract_simple(&s, i_ax_s, &bdag, i_ax_b, 2, &t);
		delete_su2_tensor(&s);
		delete_su2_tensor(&bdag);

		// reorder axes
		assert(t.ndim_logical == 4);
		su2_tensor_swap_tree_axes(&t, 2, 3);
		const int perm[4] = { 1, 0, 3, 2 };
		su2_tensor_transpose_logical(perm, &t, r_next);
		delete_su2_tensor(&t);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Contraction step from left to right for SU(2) symmetric tensors, with a matrix product operator sandwiched in between.
///
/// To-be contracted tensor network:
///
///        ╭─────────╮         ╭───────╮
///        │         │         │   b*  │
///     ─<─0─╮     ╭─3─>─   ─>─0───┬───2─>─
///        │  ╲   ╱  │         │   │   │
///        │   ╲ ╱   │         ╰───1───╯
///        │    │    │             │
///        │    │    │             ^
///        │    │    │             │
///        │   ╱ ╲   │         ╭───1───╮
///        │  ╱   ╲  │         │   │   │
///        │ │  l  ╰─2─<─   ─<─0───┤   │
///        │ │       │         │ w ├───3─<─
///        │ │       │         │   │   │
///        │ │       │         ╰───2───╯
///        │ │       │             │
///        │ │       │             ^
///        │ │       │             │
///        │ │       │         ╭───1───╮
///        │ │       │         │   │   │
///        │ ╰───────1─<─   ─<─0───┴───2─<─
///        │         │         │   a   │
///        ╰─────────╯         ╰───────╯
///
void su2_contraction_operator_step_left(const struct su2_tensor* restrict a, const struct su2_tensor* restrict b,
	const struct su2_tensor* restrict w, const struct su2_tensor* restrict l, struct su2_tensor* restrict l_next)
{
	assert(a->ndim_logical == 3);
	assert(b->ndim_logical == 3);
	assert(w->ndim_logical == 4);
	assert(l->ndim_logical == 4);
	assert(l->tree.tree_split->c[0]->i_ax == 0);
	assert( l->tree.tree_fuse->c[0]->i_ax == 1);
	assert( l->tree.tree_fuse->c[1]->i_ax == 2);
	assert(l->tree.tree_split->c[1]->i_ax == 3);

	// multiply with conjugated 'b' tensor
	struct su2_tensor s;
	{
		// flip fusion-splitting tree, complex-conjugate entries, and reverse axis 1 in 'b'
		struct su2_tensor bdag;
		// TODO: avoid full copy
		copy_su2_tensor(b, &bdag);
		su2_tensor_flip_trees(&bdag);
		conjugate_su2_tensor(&bdag);
		su2_tensor_reverse_axis_simple(&bdag, 1);

		const int i_ax_l[1] = { 3 };
		const int i_ax_b[1] = { 0 };
		su2_tensor_contract_simple(l, i_ax_l, &bdag, i_ax_b, 1, &s);
		delete_su2_tensor(&bdag);
	}

	// tree of current tensor 's'
	// (with 3 the physical axis and 4 the right virtual bond of the original 'b'):
	//
	//    1    2
	//     ╲  ╱        fuse
	//      ╲╱
	//      │
	//      │5
	//      │
	//      ╱╲
	//     ╱  ╲6
	//    ╱   ╱╲       split
	//   ╱   ╱  ╲
	//  0   4    3

	// multiply with 'w' tensor
	struct su2_tensor t;
	{
		// perform an F-move to isolate axis 3
		su2_tensor_fmove(&s, 6, &t);
		delete_su2_tensor(&s);
		// move axis 3 into the fusion tree
		su2_tensor_reverse_axis_simple(&t, 3);
		// perform an F-move to group axes 2 and 3 together
		su2_tensor_fmove(&t, 5, &s);
		delete_su2_tensor(&t);

		const int i_ax_w[2] = { 0, 1 };
		const int i_ax_s[2] = { 2, 3 };
		su2_tensor_contract_simple(&s, i_ax_s, w, i_ax_w, 2, &t);
		delete_su2_tensor(&s);
	}

	// tree of current tensor 't'
	// (with 3 the physical input axis and 4 the right virtual bond of the original 'w'):
	//
	//  1   3    4
	//   ╲   ╲  ╱
	//    ╲   ╲╱
	//     ╲  ╱5       fuse
	//      ╲╱
	//      │
	//      │6
	//      │
	//      ╱╲
	//     ╱  ╲        split
	//    0    2

	// multiply with 'a' tensor
	{
		// perform an F-move to group axes 1 and 3 together
		su2_tensor_fmove(&t, 5, &s);
		delete_su2_tensor(&t);

		const int i_ax_a[2] = { 0, 1 };
		const int i_ax_s[2] = { 1, 3 };
		su2_tensor_contract_simple(a, i_ax_a, &s, i_ax_s, 2, &t);
		delete_su2_tensor(&s);

		// reorder axes
		assert(t.ndim_logical == 4);
		const int perm[4] = { 1, 0, 3, 2 };
		su2_tensor_transpose_logical(perm, &t, l_next);
		delete_su2_tensor(&t);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute all partial contractions from the right for SU(2) symmetric tensors.
/// 'r_list' must point to an array of (uninitialized) tensors of length 'nsites' at input.
///
void su2_compute_right_operator_blocks(const struct su2_mps* restrict psi, const struct su2_mps* restrict chi, const struct su2_mpo* op, struct su2_tensor* r_list)
{
	const int nsites = op->nsites;
	assert(psi->nsites == nsites);
	assert(chi->nsites == nsites);
	assert(nsites >= 1);

	// initialize rightmost tensor
	assert(su2_irreducible_list_equal(&psi->a[nsites - 1].outer_irreps[2],
	                                  &chi->a[nsites - 1].outer_irreps[2]));
	assert(psi->a[nsites - 1].outer_irreps[2].num == 1);
	assert( op->a[nsites - 1].outer_irreps[3].num == 1);
	assert( op->a[nsites - 1].outer_irreps[3].jlist[0] == 0);
	const qnumber irrep_sector = psi->a[nsites - 1].outer_irreps[2].jlist[0];
	su2_create_dummy_operator_block_right(psi->a[nsites - 1].dtype, irrep_sector, &r_list[nsites - 1]);

	for (int i = nsites - 1; i > 0; i--)
	{
		su2_contraction_operator_step_right(&psi->a[i], &chi->a[i], &op->a[i], &r_list[i], &r_list[i - 1]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the inner product between an SU(2) symmetric MPO and two MPS, `<chi | op | psi>`.
///
void su2_mpo_inner_product(const struct su2_mps* chi, const struct su2_mpo* op, const struct su2_mps* psi, void* ret)
{
	const int nsites = op->nsites;
	assert(chi->nsites == nsites);
	assert(psi->nsites == nsites);
	assert(nsites >= 1);

	// initialize 'r'
	struct su2_tensor r;
	assert(su2_irreducible_list_equal(&psi->a[nsites - 1].outer_irreps[2],
	                                  &chi->a[nsites - 1].outer_irreps[2]));
	assert(psi->a[nsites - 1].outer_irreps[2].num == 1);
	assert( op->a[nsites - 1].outer_irreps[3].num == 1);
	assert( op->a[nsites - 1].outer_irreps[3].jlist[0] == 0);
	const qnumber irrep_sector = psi->a[nsites - 1].outer_irreps[2].jlist[0];
	su2_create_dummy_operator_block_right(psi->a[nsites - 1].dtype, irrep_sector, &r);

	for (int i = nsites - 1; i >= 0; i--)
	{
		struct su2_tensor r_next;
		su2_contraction_operator_step_right(&psi->a[i], &chi->a[i], &op->a[i], &r, &r_next);
		delete_su2_tensor(&r);
		r = r_next;  // copy internal data pointers
	}

	// expecting solely "trivial" virtual bonds and a single degeneracy tensor
	assert(r.ndim_logical == 4 && r.ndim_auxiliary == 0);
	assert(r.outer_irreps[0].num == 1 && r.outer_irreps[0].jlist[0] == 0);
	assert(r.outer_irreps[1].num == 1 && r.outer_irreps[1].jlist[0] == 0);
	assert(r.outer_irreps[2].num == 1 && r.outer_irreps[2].jlist[0] == 0);
	assert(r.outer_irreps[3].num == 1 && r.outer_irreps[3].jlist[0] == 0);
	assert(r.charge_sectors.nsec == 1);
	assert(r.degensors[0] != NULL);
	assert(dense_tensor_num_elements(r.degensors[0]) == 1);

	// store single entry in return argument
	memcpy(ret, r.degensors[0]->data, sizeof_numeric_type(r.degensors[0]->dtype));

	delete_su2_tensor(&r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply a local Hamiltonian operator for SU(2) symmetric tensors.
///
/// To-be contracted tensor network:
///
///                .................................
///               '                                 '
///        ╭──────:──╮                           ╭──:──────╮
///        │      :  │                           │  :      │
///     ─<─0─╮    :╭─3─>─  0               2  ─>─2─╮:    ╭─3─<─
///        │  ╲   ╱  │                           │  ╲   ╱  │
///        │   ╲ ╱:  │                           │  :╲ ╱   │
///        │    │ :  │             1             │  : │    │
///        │    │ :  │             ^             │  : │    │
///        │    │ :  │             │             │  : │    │
///        │   ╱ ╲'..│.........╭───1───╮.........│..' │    │
///        │  ╱   ╲  │         │   │   │         │   ╱ ╲   │
///        │ │  l  ╰─2─<─   ─<─0───┤   │         │  ╱ r ╲  │
///        │ │       │         │ w ├───3─<─   ─<─1─│─────╯ │
///        │ │       │         │   │   │         │ │       │
///        │ │       │         ╰───2───╯         │ │       │
///        │ │       │             │             │ │       │
///        │ │       │             ^             │ │       │
///        │ │       │             │             │ │       │
///        │ │       │         ╭───1───╮         │ │       │
///        │ │       │         │   │   │         │ │       │
///        │ ╰───────1─<─   ─<─0───┴───2─<─   ─<─0─╯       │
///        │         │         │   a   │         │         │
///        ╰─────────╯         ╰───────╯         ╰─────────╯
///
/// The dotted outline marks the output tensor.
/// The outer virtual bonds are contracted as well.
///
void su2_apply_local_hamiltonian(const struct su2_tensor* restrict a, const struct su2_tensor* restrict w,
	const struct su2_tensor* restrict l, const struct su2_tensor* restrict r, struct su2_tensor* restrict b)
{
	assert(a->ndim_logical == 3);
	assert(w->ndim_logical == 4);
	assert(l->ndim_logical == 4);
	assert(r->ndim_logical == 4);
	assert(l->tree.tree_split->c[0]->i_ax == 0);
	assert( l->tree.tree_fuse->c[0]->i_ax == 1);
	assert( l->tree.tree_fuse->c[1]->i_ax == 2);
	assert(l->tree.tree_split->c[1]->i_ax == 3);
	assert(r->tree.tree_split->c[0]->i_ax == 0);
	assert(r->tree.tree_split->c[1]->i_ax == 1);
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
	//    6╱  ╲
	//    ╱╲   ╲       split
	//   ╱  ╲   ╲
	//  0    1   2

	// multiply with 'w' tensor
	{
		// perform an F-move to group axes 1 and 2 together
		struct su2_tensor t;
		assert(s.tree.tree_split->c[0]->i_ax == 6);
		su2_tensor_fmove(&s, 6, &t);
		delete_su2_tensor(&s);

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
	//     ╱  ╲5
	//    ╱   ╱╲       split
	//   ╱   ╱  ╲
	//  2   0    1

	// multiply with 'l' tensor (including outer virtual bonds)
	{
		// group axes 0 and 2 in 's' together
		struct su2_tensor t;
		assert(s.tree.tree_split->c[1]->i_ax == 5);
		su2_tensor_fmove(&s, 5, &t);
		delete_su2_tensor(&s);
		// group axes 0, 2 with axis 4 (original right virtual bond of 'r')
		su2_tensor_swap_tree_axes(&t, 3, 4);
		su2_tensor_reverse_axis_simple(&t, 4);
		assert(t.tree.tree_split->c[1]->i_ax == 6);
		su2_tensor_fmove(&t, 6, &s);
		delete_su2_tensor(&t);

		struct su2_tensor k;
		copy_su2_tensor(l, &k);
		su2_tensor_reverse_axis_simple(&k, 0);

		const int i_ax_k[3] = { 0, 1, 2 };
		const int i_ax_s[3] = { 4, 2, 0 };
		su2_tensor_contract_simple(&k, i_ax_k, &s, i_ax_s, 3, b);
		delete_su2_tensor(&s);
		delete_su2_tensor(&k);
	}
}
