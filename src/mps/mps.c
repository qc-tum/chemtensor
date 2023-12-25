/// \file mps.c
/// \brief Matrix product state (MPS) data structure.

#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "mps.h"
#include "config.h"


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for a matrix product state. 'dim_bonds' and 'qbonds' must be arrays of length 'nsites + 1'.
///
void allocate_mps(const enum numeric_type dtype, const int nsites, const long d, const qnumber* qsite, const long* dim_bonds, const qnumber** qbonds, struct mps* mps)
{
	assert(nsites >= 1);
	assert(d >= 1);
	mps->nsites = nsites;
	mps->d = d;

	mps->qsite = aligned_alloc(MEM_DATA_ALIGN, d * sizeof(qnumber));
	memcpy(mps->qsite, qsite, d * sizeof(qnumber));

	mps->a = aligned_calloc(MEM_DATA_ALIGN, nsites, sizeof(struct block_sparse_tensor));

	for (int i = 0; i < nsites; i++)
	{
		const long dim[3] = { d, dim_bonds[i], dim_bonds[i + 1] };
		const enum tensor_axis_direction axis_dir[3] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN };
		const qnumber* qnums[3] = { qsite, qbonds[i], qbonds[i + 1] };
		allocate_block_sparse_tensor(dtype, 3, dim, axis_dir, qnums, &mps->a[i]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a matrix product state (free memory).
///
void delete_mps(struct mps* mps)
{
	for (int i = 0; i < mps->nsites; i++)
	{
		delete_block_sparse_tensor(&mps->a[i]);
	}
	aligned_free(mps->a);
	mps->a = NULL;
	mps->nsites = 0;

	aligned_free(mps->qsite);
	mps->qsite = NULL;
	mps->d = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Internal consistency check of the MPS data structure.
///
bool mps_is_consistent(const struct mps* mps)
{
	if (mps->nsites <= 0) {
		return false;
	}
	if (mps->d <= 0) {
		return false;
	}

	// quantum numbers for physical legs of individual tensors must agree with 'qsite'
	for (int i = 0; i < mps->nsites; i++)
	{
		if (mps->a[i].ndim != 3) {
			return false;
		}
		if (mps->a[i].dim_logical[0] != mps->d) {
			return false;
		}
		if (!qnumber_all_equal(mps->d, mps->a[i].qnums_logical[0], mps->qsite)) {
			return false;
		}
	}

	// virtual bond quantum numbers must match
	for (int i = 0; i < mps->nsites - 1; i++)
	{
		if (mps->a[i].dim_logical[2] != mps->a[i + 1].dim_logical[1]) {
			return false;
		}
		if (!qnumber_all_equal(mps->a[i].dim_logical[2], mps->a[i].qnums_logical[2], mps->a[i + 1].qnums_logical[1])) {
			return false;
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Merge two neighboring MPS tensors.
///
void mps_merge_tensor_pair(const struct block_sparse_tensor* restrict a0, const struct block_sparse_tensor* restrict a1, struct block_sparse_tensor* restrict a)
{
	assert(a0->ndim == 3);
	assert(a1->ndim == 3);

	// combine a0 and a1 by contracting the shared bond
	struct block_sparse_tensor a0_a1_dot;
	const int perm01[3] = { 1, 0, 2 };
	struct block_sparse_tensor a1_tp;
	transpose_block_sparse_tensor(perm01, a1, &a1_tp);
	block_sparse_tensor_dot(a0, &a1_tp, 1, &a0_a1_dot);
	delete_block_sparse_tensor(&a1_tp);

	// pair original physical dimensions of a0 and a1
	struct block_sparse_tensor a0_a1_dot_reorder;
	assert(a0_a1_dot.ndim == 4);
	const int perm12[4] = { 0, 2, 1, 3 };
	transpose_block_sparse_tensor(perm12, &a0_a1_dot, &a0_a1_dot_reorder);
	delete_block_sparse_tensor(&a0_a1_dot);

	// combine original physical dimensions of a0 and a1 into one dimension
	flatten_block_sparse_tensor_axes(&a0_a1_dot_reorder, 0, TENSOR_AXIS_OUT, a);
	delete_block_sparse_tensor(&a0_a1_dot_reorder);
}


//________________________________________________________________________________________________________________________
///
/// \brief Merge all tensors of a MPS to obtain the vector representation on the full Hilbert space.
/// The (dummy) virtual bonds are retained in the output tensor.
///
void mps_to_statevector(const struct mps* mps, struct block_sparse_tensor* vec)
{
	assert(mps->nsites > 0);

	if (mps->nsites == 1)
	{
		copy_block_sparse_tensor(&mps->a[0], vec);
	}
	else if (mps->nsites == 2)
	{
		mps_merge_tensor_pair(&mps->a[0], &mps->a[1], vec);
	}
	else
	{
		struct block_sparse_tensor t[2];
		mps_merge_tensor_pair(&mps->a[0], &mps->a[1], &t[0]);
		for (int i = 2; i < mps->nsites; i++)
		{
			mps_merge_tensor_pair(&t[i % 2], &mps->a[i], i < mps->nsites - 1 ? &t[(i + 1) % 2] : vec);
			delete_block_sparse_tensor(&t[i % 2]);
		}
	}

	assert(vec->ndim == 3);
}
