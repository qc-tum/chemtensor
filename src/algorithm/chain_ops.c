/// \file chain_ops.c
/// \brief Higher-level tensor network operations on a chain topology.

#include <memory.h>
#include <assert.h>
#include "chain_ops.h"


//________________________________________________________________________________________________________________________
///
/// \brief Create a dummy right operator block.
///
void create_dummy_operator_block_right(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict b,
	const struct block_sparse_tensor* restrict w, struct block_sparse_tensor* restrict r)
{
	assert(a->ndim == 3);
	assert(b->ndim == 3);
	assert(w->ndim == 4);

	// for now requiring dummy trailing virtual bond dimensions
	assert(a->dim_logical[2] == 1);
	assert(b->dim_logical[2] == 1);
	assert(w->dim_logical[3] == 1);

	const enum numeric_type dtype = a->dtype;

	const long dim[6] = { 1, 1, 1, 1, 1, 1 };
	const enum tensor_axis_direction axis_dir[6] = {
		TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN,
		TENSOR_AXIS_IN,  TENSOR_AXIS_IN,  TENSOR_AXIS_OUT,
	};
	const qnumber* qnums[6] = {
		a->qnums_logical[2], w->qnums_logical[3], b->qnums_logical[2],
		a->qnums_logical[2], w->qnums_logical[3], b->qnums_logical[2],
	};
	struct block_sparse_tensor s;
	allocate_block_sparse_tensor(dtype, 6, dim, axis_dir, qnums, &s);
	assert(s.blocks[0] != NULL);
	memcpy(s.blocks[0]->data, numeric_one(dtype), sizeof_numeric_type(dtype));
	struct block_sparse_tensor t;
	flatten_block_sparse_tensor_axes(&s, 4, TENSOR_AXIS_IN, &t);
	delete_block_sparse_tensor(&s);
	flatten_block_sparse_tensor_axes(&t, 3, TENSOR_AXIS_IN, r);
	delete_block_sparse_tensor(&t);
}


//________________________________________________________________________________________________________________________
///
/// \brief Create a dummy left operator block.
///
void create_dummy_operator_block_left(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict b,
	const struct block_sparse_tensor* restrict w, struct block_sparse_tensor* restrict l)
{
	assert(a->ndim == 3);
	assert(b->ndim == 3);
	assert(w->ndim == 4);

	// for now requiring dummy leading virtual bond dimensions
	assert(a->dim_logical[0] == 1);
	assert(b->dim_logical[0] == 1);
	assert(w->dim_logical[0] == 1);

	const enum numeric_type dtype = a->dtype;

	const long dim[6] = { 1, 1, 1, 1, 1, 1 };
	const enum tensor_axis_direction axis_dir[6] = {
		TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN,
		TENSOR_AXIS_IN,  TENSOR_AXIS_IN,  TENSOR_AXIS_OUT,
	};
	const qnumber* qnums[6] = {
		a->qnums_logical[0], w->qnums_logical[0], b->qnums_logical[0],
		a->qnums_logical[0], w->qnums_logical[0], b->qnums_logical[0],
	};
	struct block_sparse_tensor s;
	allocate_block_sparse_tensor(dtype, 6, dim, axis_dir, qnums, &s);
	assert(s.blocks[0] != NULL);
	memcpy(s.blocks[0]->data, numeric_one(dtype), sizeof_numeric_type(dtype));
	struct block_sparse_tensor t;
	flatten_block_sparse_tensor_axes(&s, 0, TENSOR_AXIS_OUT, &t);
	delete_block_sparse_tensor(&s);
	flatten_block_sparse_tensor_axes(&t, 0, TENSOR_AXIS_OUT, l);
	delete_block_sparse_tensor(&t);
}


//________________________________________________________________________________________________________________________
///
/// \brief Contraction step from right to left, with a matrix product operator sandwiched in between.
///
/// To-be contracted tensor network:
///
///       _____           _______
///      /     \         /       \
///   ->-|0 b*2|->-   ->-|2      |
///      \__1__/         |       |
///         |            |       |
///         ^            |       |
///       __|__          |       |
///      /  1  \         |       |
///   -<-|0 w 3|-<-   -<-|1  r  3|-<-
///      \__2__/         |       |
///         |            |       |
///         ^            |       |
///       __|__          |       |
///      /  1  \         |       |
///   -<-|0 a 2|-<-   -<-|0      |
///      \_____/         \_______/
///
void contraction_operator_step_right(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict b,
	const struct block_sparse_tensor* restrict w, const struct block_sparse_tensor* restrict r, struct block_sparse_tensor* restrict r_next)
{
	assert(a->ndim == 3);
	assert(b->ndim == 3);
	assert(w->ndim == 4);
	assert(r->ndim == 4);

	// multiply with 'a' tensor
	struct block_sparse_tensor s;
	block_sparse_tensor_dot(a, TENSOR_AXIS_RANGE_TRAILING, r, TENSOR_AXIS_RANGE_LEADING, 1, &s);

	// multiply with 'w' tensor
	// re-order first three dimensions
	const int perm0[5] = { 1, 2, 0, 3, 4 };
	struct block_sparse_tensor t;
	transpose_block_sparse_tensor(perm0, &s, &t);
	delete_block_sparse_tensor(&s);
	block_sparse_tensor_dot(w, TENSOR_AXIS_RANGE_TRAILING, &t, TENSOR_AXIS_RANGE_LEADING, 2, &s);
	delete_block_sparse_tensor(&t);
	// undo re-ordering
	const int perm1[5] = { 2, 0, 1, 3, 4 };
	transpose_block_sparse_tensor(perm1, &s, &t);
	delete_block_sparse_tensor(&s);

	// TODO: fuse transpositions

	// multiply with conjugated 'b' tensor
	struct block_sparse_tensor bc;
	copy_block_sparse_tensor(b, &bc);  // TODO: fuse conjugation with dot product
	conjugate_block_sparse_tensor(&bc);
	block_sparse_tensor_reverse_axis_directions(&bc);
	// temporarily make trailing dimension in 't' the leading dimension
	const int perm2[5] = { 4, 0, 1, 2, 3 };
	transpose_block_sparse_tensor(perm2, &t, &s);
	delete_block_sparse_tensor(&t);
	block_sparse_tensor_dot(&s, TENSOR_AXIS_RANGE_TRAILING, &bc, TENSOR_AXIS_RANGE_TRAILING, 2, &t);
	delete_block_sparse_tensor(&s);
	delete_block_sparse_tensor(&bc);
	// restore original trailing dimension
	const int perm3[4] = { 1, 2, 3, 0 };
	transpose_block_sparse_tensor(perm3, &t, r_next);
	delete_block_sparse_tensor(&t);
}


//________________________________________________________________________________________________________________________
///
/// \brief Contraction step from left to right, with a matrix product operator sandwiched in between.
///
/// To-be contracted tensor network:
///
///       _______           _____
///      /       \         /     \
///      |      3|->-   ->-|0 b*2|->-
///      |       |         \__1__/
///      |       |            |
///      |       |            ^
///      |       |          __|__
///      |       |         /  1  \
///   -<-|0  l  2|-<-   -<-|0 w 3|-<-
///      |       |         \__2__/
///      |       |            |
///      |       |            ^
///      |       |          __|__
///      |       |         /  1  \
///      |      1|-<-   -<-|0 a 2|-<-
///      \_______/         \_____/
///
void contraction_operator_step_left(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict b,
	const struct block_sparse_tensor* restrict w, const struct block_sparse_tensor* restrict l, struct block_sparse_tensor* restrict l_next)
{
	assert(a->ndim == 3);
	assert(b->ndim == 3);
	assert(w->ndim == 4);
	assert(l->ndim == 4);

	// multiply with conjugated 'b' tensor
	struct block_sparse_tensor bc;
	copy_block_sparse_tensor(b, &bc);  // TODO: fuse conjugation with dot product
	conjugate_block_sparse_tensor(&bc);
	block_sparse_tensor_reverse_axis_directions(&bc);
	struct block_sparse_tensor s;
	block_sparse_tensor_dot(l, TENSOR_AXIS_RANGE_TRAILING, &bc, TENSOR_AXIS_RANGE_LEADING, 1, &s);
	delete_block_sparse_tensor(&bc);

	// multiply with 'w' tensor
	// re-order last three dimensions
	const int perm0[5] = { 0, 1, 4, 2, 3 };
	struct block_sparse_tensor t;
	transpose_block_sparse_tensor(perm0, &s, &t);
	delete_block_sparse_tensor(&s);
	block_sparse_tensor_dot(&t, TENSOR_AXIS_RANGE_TRAILING, w, TENSOR_AXIS_RANGE_LEADING, 2, &s);
	delete_block_sparse_tensor(&t);
	// undo re-ordering
	const int perm1[5] = { 0, 1, 3, 4, 2 };
	transpose_block_sparse_tensor(perm1, &s, &t);
	delete_block_sparse_tensor(&s);

	// TODO: fuse transpositions

	// multiply with 'a' tensor
	// temporarily make leading dimension in 't' the trailing dimension
	const int perm2[5] = { 1, 2, 3, 4, 0 };
	transpose_block_sparse_tensor(perm2, &t, &s);
	delete_block_sparse_tensor(&t);
	block_sparse_tensor_dot(a, TENSOR_AXIS_RANGE_LEADING, &s, TENSOR_AXIS_RANGE_LEADING, 2, &t);
	delete_block_sparse_tensor(&s);
	// restore original leading dimension
	const int perm3[4] = { 3, 0, 1, 2 };
	transpose_block_sparse_tensor(perm3, &t, l_next);
	delete_block_sparse_tensor(&t);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute all partial contractions from the right.
/// 'r_list' must point to an array of (uninitialized) tensors of length 'nsites' at input.
///
void compute_right_operator_blocks(const struct mps* restrict psi, const struct mps* restrict chi, const struct mpo* op, struct block_sparse_tensor* r_list)
{
	const int nsites = op->nsites;
	assert(psi->nsites == nsites);
	assert(chi->nsites == nsites);
	assert(nsites >= 1);

	// initialize rightmost tensor
	create_dummy_operator_block_right(&psi->a[nsites - 1], &chi->a[nsites - 1], &op->a[nsites - 1], &r_list[nsites - 1]);

	for (int i = nsites - 1; i > 0; i--)
	{
		contraction_operator_step_right(&psi->a[i], &chi->a[i], &op->a[i], &r_list[i], &r_list[i - 1]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the inner product between an MPO and two MPS, `<chi | op | psi>`.
///
void mpo_inner_product(const struct mps* chi, const struct mpo* op, const struct mps* psi, void* ret)
{
	const int nsites = op->nsites;
	assert(chi->nsites == nsites);
	assert(psi->nsites == nsites);
	assert(nsites >= 1);
	// for now requiring dummy leading and trailing virtual bond dimensions
	assert(chi->a[         0].dim_logical[0] == 1);
	assert( op->a[         0].dim_logical[0] == 1);
	assert(psi->a[         0].dim_logical[0] == 1);
	assert(chi->a[nsites - 1].dim_logical[2] == 1);
	assert( op->a[nsites - 1].dim_logical[3] == 1);
	assert(psi->a[nsites - 1].dim_logical[2] == 1);

	const enum numeric_type dtype = psi->a[0].dtype;

	// initialize 'r'
	struct block_sparse_tensor r;
	create_dummy_operator_block_right(&psi->a[nsites - 1], &chi->a[nsites - 1], &op->a[nsites - 1], &r);

	for (int i = nsites - 1; i >= 0; i--)
	{
		struct block_sparse_tensor r_next;
		contraction_operator_step_right(&psi->a[i], &chi->a[i], &op->a[i], &r, &r_next);
		delete_block_sparse_tensor(&r);
		move_block_sparse_tensor_data(&r_next, &r);
	}

	// flatten left virtual bonds
	{
		struct block_sparse_tensor t;
		flatten_block_sparse_tensor_axes(&r, 0, TENSOR_AXIS_OUT, &t);
		delete_block_sparse_tensor(&r);
		flatten_block_sparse_tensor_axes(&t, 0, TENSOR_AXIS_OUT, &r);
		delete_block_sparse_tensor(&t);
	}

	// 'r' should now be a 1 x 1 tensor
	assert(r.ndim == 2);
	assert(r.dim_logical[0] == 1 && r.dim_logical[1] == 1);
	assert(r.blocks[0] != NULL);
	memcpy(ret, r.blocks[0]->data, sizeof_numeric_type(dtype));
	delete_block_sparse_tensor(&r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply a local Hamiltonian operator.
///
/// To-be contracted tensor network:
///
///           .................................
///          '                                 '
///       ___:___                           ___:___
///      /   :   \                         /   :   \
///      |   :  3|->-   0           2   ->-|2  :   |
///      |   :   |                         |   :   |
///      |   :   |            1            |   :   |
///      |   :   |            ^            |   :   |
///      |   '...|..........__|__..........|...'   |
///      |       |         /  1  \         |       |
///   -<-|0  l  2|-<-   -<-|0 w 3|-<-   -<-|1  r  3|-<-
///      |       |         \__2__/         |       |
///      |       |            |            |       |
///      |       |            ^            |       |
///      |       |          __|__          |       |
///      |       |         /  1  \         |       |
///      |      1|-<-   -<-|0 a 2|-<-   -<-|0      |
///      \_______/         \_____/         \_______/
///
/// The dotted outline marks the output tensor.
/// The outer virtual bonds are contracted as well.
///
void apply_local_hamiltonian(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict w,
	const struct block_sparse_tensor* restrict l, const struct block_sparse_tensor* restrict r, struct block_sparse_tensor* restrict b)
{
	assert(a->ndim == 3);
	assert(w->ndim == 4);
	assert(l->ndim == 4);
	assert(r->ndim == 4);

	// multiply with 'a' tensor
	struct block_sparse_tensor s;
	block_sparse_tensor_dot(a, TENSOR_AXIS_RANGE_TRAILING, r, TENSOR_AXIS_RANGE_LEADING, 1, &s);

	// multiply with 'w' tensor
	// re-order first three dimensions
	const int perm0[5] = { 1, 2, 0, 3, 4 };
	struct block_sparse_tensor t;
	transpose_block_sparse_tensor(perm0, &s, &t);
	delete_block_sparse_tensor(&s);
	block_sparse_tensor_dot(w, TENSOR_AXIS_RANGE_TRAILING, &t, TENSOR_AXIS_RANGE_LEADING, 2, &s);
	delete_block_sparse_tensor(&t);
	// undo re-ordering
	const int perm1[5] = { 2, 0, 1, 3, 4 };
	transpose_block_sparse_tensor(perm1, &s, &t);
	delete_block_sparse_tensor(&s);

	// multiply with 'l' tensor
	// re-order last three dimensions
	const int perm2[4] = { 0, 3, 1, 2 };
	struct block_sparse_tensor k;
	transpose_block_sparse_tensor(perm2, l, &k);
	block_sparse_tensor_dot(&k, TENSOR_AXIS_RANGE_TRAILING, &t, TENSOR_AXIS_RANGE_LEADING, 2, &s);
	delete_block_sparse_tensor(&k);
	delete_block_sparse_tensor(&t);

	// trace out outer virtual bonds (assumed to be low-dimensional)
	block_sparse_tensor_cyclic_partial_trace(&s, 1, b);
	delete_block_sparse_tensor(&s);
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the network environment of a local Hamiltonian operator.
///
/// To-be contracted tensor network:
///
///       _______           _____           _______
///      /       \         /     \         /       \
///      |      3|->-   ->-|0 b*2|->-   ->-|2      |
///      |    ...|.........\__1__/.........|...    |
///      |   '   |            |            |   '   |
///      |   :   |            ^            |   :   |
///      |   :   |            1            |   :   |
///      |   :   |                         |   :   |
///   -<-|0  l  2|-<-   0           3   -<-|1  r  3|-<-
///      |   :   |                         |   :   |
///      |   :   |            2            |   :   |
///      |   :   |            ^            |   :   |
///      |   '...|..........__|__..........|...'   |
///      |       |         /  1  \         |       |
///      |      1|-<-   -<-|0 a 2|-<-   -<-|0      |
///      \_______/         \_____/         \_______/
///
/// The dotted outline marks the output tensor.
/// The outer virtual bonds are contracted as well.
///
void compute_local_hamiltonian_environment(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict b,
	const struct block_sparse_tensor* restrict l, const struct block_sparse_tensor* restrict r, struct block_sparse_tensor* restrict dw)
{
	assert(a->ndim == 3);
	assert(b->ndim == 3);
	assert(l->ndim == 4);
	assert(r->ndim == 4);

	// multiply with 'a' tensor
	struct block_sparse_tensor s;
	block_sparse_tensor_dot(a, TENSOR_AXIS_RANGE_TRAILING, r, TENSOR_AXIS_RANGE_LEADING, 1, &s);

	// multiply with conjugated 'b' tensor
	struct block_sparse_tensor bc;
	copy_block_sparse_tensor(b, &bc);  // TODO: fuse conjugation with dot product
	conjugate_block_sparse_tensor(&bc);
	block_sparse_tensor_reverse_axis_directions(&bc);
	// re-order last two dimensions
	const int perm0[5] = { 0, 1, 2, 4, 3 };
	struct block_sparse_tensor t;
	transpose_block_sparse_tensor(perm0, &s, &t);
	delete_block_sparse_tensor(&s);
	block_sparse_tensor_dot(&bc, TENSOR_AXIS_RANGE_TRAILING, &t, TENSOR_AXIS_RANGE_TRAILING, 1, &s);
	delete_block_sparse_tensor(&t);
	delete_block_sparse_tensor(&bc);

	// multiply with 'l' tensor
	// re-order second and third dimension of 'l'
	const int perm1[4] = { 0, 2, 1, 3 };
	struct block_sparse_tensor k;
	transpose_block_sparse_tensor(perm1, l, &k);
	// re-order leading three dimensions of 's' (to make MPS virtual bond dimensions the leading dimensions)
	const int perm2[6] = { 2, 0, 1, 3, 4, 5 };
	transpose_block_sparse_tensor(perm2, &s, &t);
	delete_block_sparse_tensor(&s);
	block_sparse_tensor_dot(&k, TENSOR_AXIS_RANGE_TRAILING, &t, TENSOR_AXIS_RANGE_LEADING, 2, &s);
	delete_block_sparse_tensor(&k);
	delete_block_sparse_tensor(&t);

	// trace out outer virtual bonds (assumed to be low-dimensional)
	block_sparse_tensor_cyclic_partial_trace(&s, 1, dw);
	delete_block_sparse_tensor(&s);
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply an operator represented as MPO to a state in MPS form.
///
void apply_mpo(const struct mpo* op, const struct mps* psi, struct mps* op_psi)
{
	// quantum numbers on physical sites must match
	assert(psi->d == op->d);
	assert(qnumber_all_equal(psi->d, psi->qsite, op->qsite));
	assert(psi->nsites == op->nsites);
	assert(psi->nsites >= 1);

	allocate_empty_mps(psi->nsites, psi->d, psi->qsite, op_psi);

	for (int i = 0; i < psi->nsites; i++)
	{
		// move physical input axis of op->a[i] to the end
		const int perm_op[4] = { 0, 1, 3, 2 };
		struct block_sparse_tensor r;
		transpose_block_sparse_tensor(perm_op, &op->a[i], &r);

		// move physical axis of psi->a[i] to the beginning
		const int perm_psi[3] = { 1, 0, 2 };
		struct block_sparse_tensor s;
		transpose_block_sparse_tensor(perm_psi, &psi->a[i], &s);

		// contract local tensors
		struct block_sparse_tensor t;
		block_sparse_tensor_dot(&r, TENSOR_AXIS_RANGE_TRAILING, &s, TENSOR_AXIS_RANGE_LEADING, 1, &t);
		delete_block_sparse_tensor(&r);
		delete_block_sparse_tensor(&s);

		// reorder axes
		const int perm_ax[5] = { 0, 3, 1, 2, 4 };
		transpose_block_sparse_tensor(perm_ax, &t, &r);
		delete_block_sparse_tensor(&t);

		// flatten left and right virtual bonds
		flatten_block_sparse_tensor_axes(&r, 0, TENSOR_AXIS_OUT, &s);
		delete_block_sparse_tensor(&r);
		flatten_block_sparse_tensor_axes(&s, 2, TENSOR_AXIS_IN, &op_psi->a[i]);
		delete_block_sparse_tensor(&s);
	}
}
