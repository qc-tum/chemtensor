/// \file operation.c
/// \brief Higher-level tensor network operations.

#include <memory.h>
#include <assert.h>
#include "operation.h"


//________________________________________________________________________________________________________________________
///
/// \brief Contraction step from right to left of two MPS tensors,
/// for example to compute the inner product of two matrix product states.
///
/// To-be contracted tensor network:
///
///       _____           _______
///      /     \         /       \
///   ->-|0 b*2|->-   ->-|1     3|->-
///      \__1__/         |       |
///         |            |       |
///         ^            |   r   |
///       __|__          |       |
///      /  1  \         |       |
///   -<-|0 a 2|-<-   -<-|0     2|-<-
///      \_____/         \_______/
///
static void mps_contraction_step_right(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict b,
	const struct block_sparse_tensor* restrict r, struct block_sparse_tensor* restrict r_next)
{
	assert(a->ndim == 3);
	assert(b->ndim == 3);
	assert(r->ndim == 4);

	// multiply with 'a' tensor
	struct block_sparse_tensor t;
	block_sparse_tensor_dot(a, r, 1, &t);

	// multiply with conjugated 'b' tensor
	struct block_sparse_tensor b_conj;
	copy_block_sparse_tensor(b, &b_conj);  // TODO: fuse conjugation with dot product
	conjugate_block_sparse_tensor(&b_conj);
	block_sparse_tensor_reverse_axis_directions(&b_conj);
	const int perm0[5] = { 1, 2, 0, 3, 4 };
	struct block_sparse_tensor tp;
	transpose_block_sparse_tensor(perm0, &t, &tp);
	delete_block_sparse_tensor(&t);
	block_sparse_tensor_dot(&b_conj, &tp, 2, &t);
	delete_block_sparse_tensor(&b_conj);
	delete_block_sparse_tensor(&tp);

	// re-order first two dimensions
	const int perm1[5] = { 1, 0, 2, 3, 4 };
	transpose_block_sparse_tensor(perm1, &t, r_next);
	delete_block_sparse_tensor(&t);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the dot (scalar) product `<chi | psi>` of two MPS, complex conjugating `chi`.
///
void mps_vdot(const struct mps* chi, const struct mps* psi, void* ret)
{
	assert(psi->nsites == chi->nsites);
	assert(psi->nsites >= 1);
	// for now requiring dummy leading and trailing virtual bond dimensions
	assert(chi->a[              0].dim_logical[0] == 1);
	assert(psi->a[              0].dim_logical[0] == 1);
	assert(chi->a[chi->nsites - 1].dim_logical[2] == 1);
	assert(psi->a[psi->nsites - 1].dim_logical[2] == 1);

	// initialize 'r'
	const enum numeric_type dtype = psi->a[0].dtype;
	const long dim[4] = { 1, 1, 1, 1 };
	const enum tensor_axis_direction axis_dir[4] = { TENSOR_AXIS_OUT, TENSOR_AXIS_IN, TENSOR_AXIS_IN, TENSOR_AXIS_OUT };
	const qnumber* qnums[4] = {
		psi->a[psi->nsites - 1].qnums_logical[2],
		chi->a[chi->nsites - 1].qnums_logical[2],
		psi->a[psi->nsites - 1].qnums_logical[2],
		chi->a[chi->nsites - 1].qnums_logical[2], };
	struct block_sparse_tensor r;
	allocate_block_sparse_tensor(dtype, 4, dim, axis_dir, qnums, &r);
	assert(r.blocks[0] != NULL);
	memcpy(r.blocks[0]->data, numeric_one(dtype), sizeof_numeric_type(dtype));

	for (int i = psi->nsites - 1; i >= 0; i--)
	{
		struct block_sparse_tensor r_next;
		mps_contraction_step_right(&psi->a[i], &chi->a[i], &r, &r_next);
		delete_block_sparse_tensor(&r);
		move_block_sparse_tensor_data(&r_next, &r);
	}

	// 'r' should now be a 1 x 1 x 1 x 1 tensor
	assert(r.ndim == 4);
	assert(r.dim_logical[0] == 1 && r.dim_logical[1] == 1 && r.dim_logical[2] == 1 && r.dim_logical[3] == 1);
	assert(r.blocks[0] != NULL);
	memcpy(ret, r.blocks[0]->data, sizeof_numeric_type(dtype));
	delete_block_sparse_tensor(&r);
}
