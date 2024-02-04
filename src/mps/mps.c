/// \file mps.c
/// \brief Matrix product state (MPS) data structure.

#include <stdlib.h>
#include <memory.h>
#include <complex.h>
#include "mps.h"
#include "aligned_memory.h"


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
		const long dim[3] = { dim_bonds[i], d, dim_bonds[i + 1] };
		const enum tensor_axis_direction axis_dir[3] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN };
		const qnumber* qnums[3] = { qbonds[i], qsite, qbonds[i + 1] };
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
		if (mps->a[i].dim_logical[1] != mps->d) {
			return false;
		}
		if (!qnumber_all_equal(mps->d, mps->a[i].qnums_logical[1], mps->qsite)) {
			return false;
		}
	}

	// virtual bond quantum numbers must match
	for (int i = 0; i < mps->nsites - 1; i++)
	{
		if (mps->a[i].dim_logical[2] != mps->a[i + 1].dim_logical[0]) {
			return false;
		}
		if (!qnumber_all_equal(mps->a[i].dim_logical[2], mps->a[i].qnums_logical[2], mps->a[i + 1].qnums_logical[0])) {
			return false;
		}
	}

	// axis directions
	for (int i = 0; i < mps->nsites; i++)
	{
		if (mps->a[i].axis_dir[0] != TENSOR_AXIS_OUT ||
		    mps->a[i].axis_dir[1] != TENSOR_AXIS_OUT ||
		    mps->a[i].axis_dir[2] != TENSOR_AXIS_IN) {
			return false;
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Left-orthonormalize a local MPS site tensor by QR decomposition, and update tensor at next site.
///
void mps_local_orthonormalize_qr(struct block_sparse_tensor* restrict a, struct block_sparse_tensor* restrict a_next)
{
	assert(a->ndim == 3);
	assert(a_next->ndim == 3);

	// save original logical dimensions and quantum numbers for later splitting
	const long dim_logical_left[2] = { a->dim_logical[0], a->dim_logical[1] };
	qnumber* qnums_logical_left[2];
	for (int i = 0; i < 2; i++)
	{
		qnums_logical_left[i] = aligned_alloc(MEM_DATA_ALIGN, dim_logical_left[i] * sizeof(qnumber));
		memcpy(qnums_logical_left[i], a->qnums_logical[i], dim_logical_left[i] * sizeof(qnumber));
	}
	assert(a->axis_dir[0] == TENSOR_AXIS_OUT && a->axis_dir[1] == TENSOR_AXIS_OUT);
	const enum tensor_axis_direction axis_dir_left[2] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT };

	// combine left virtual bond and physical axis
	struct block_sparse_tensor a_mat;
	flatten_block_sparse_tensor_axes(a, 0, TENSOR_AXIS_OUT, &a_mat);
	delete_block_sparse_tensor(a);

	// perform QR decomposition
	struct block_sparse_tensor q, r;
	block_sparse_tensor_qr(&a_mat, &q, &r);
	delete_block_sparse_tensor(&a_mat);

	// replace 'a' by reshaped 'q' matrix
	split_block_sparse_tensor_axis(&q, 0, dim_logical_left, axis_dir_left, (const qnumber**)qnums_logical_left, a);
	delete_block_sparse_tensor(&q);
	for (int i = 0; i < 2; i++)
	{
		aligned_free(qnums_logical_left[i]);
	}

	// update 'a_next' tensor: multiply with 'r' from left
	struct block_sparse_tensor a_next_update;
	block_sparse_tensor_dot(&r, a_next, 1, &a_next_update);
	delete_block_sparse_tensor(a_next);
	move_block_sparse_tensor_data(&a_next_update, a_next);
	delete_block_sparse_tensor(&r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Right-orthonormalize a local MPS site tensor by RQ decomposition, and update tensor at next site.
///
void mps_local_orthonormalize_rq(struct block_sparse_tensor* restrict a, struct block_sparse_tensor* restrict a_prev)
{
	assert(a->ndim == 3);
	assert(a_prev->ndim == 3);

	// save original logical dimensions and quantum numbers for later splitting
	const long dim_logical_right[2] = { a->dim_logical[1], a->dim_logical[2] };
	qnumber* qnums_logical_right[2];
	for (int i = 0; i < 2; i++)
	{
		qnums_logical_right[i] = aligned_alloc(MEM_DATA_ALIGN, dim_logical_right[i] * sizeof(qnumber));
		memcpy(qnums_logical_right[i], a->qnums_logical[1 + i], dim_logical_right[i] * sizeof(qnumber));
	}
	assert(a->axis_dir[1] == TENSOR_AXIS_OUT && a->axis_dir[2] == TENSOR_AXIS_IN);
	const enum tensor_axis_direction axis_dir_right[2] = { TENSOR_AXIS_OUT, TENSOR_AXIS_IN };

	// combine physical and right virtual bond axis
	struct block_sparse_tensor a_mat;
	flatten_block_sparse_tensor_axes(a, 1, TENSOR_AXIS_IN, &a_mat);
	delete_block_sparse_tensor(a);

	// perform RQ decomposition
	struct block_sparse_tensor r, q;
	block_sparse_tensor_rq(&a_mat, &r, &q);
	delete_block_sparse_tensor(&a_mat);

	// replace 'a' by reshaped 'q' matrix
	split_block_sparse_tensor_axis(&q, 1, dim_logical_right, axis_dir_right, (const qnumber**)qnums_logical_right, a);
	delete_block_sparse_tensor(&q);
	for (int i = 0; i < 2; i++)
	{
		aligned_free(qnums_logical_right[i]);
	}

	// update 'a_prev' tensor: multiply with 'r' from right
	struct block_sparse_tensor a_prev_update;
	block_sparse_tensor_dot(a_prev, &r, 1, &a_prev_update);
	delete_block_sparse_tensor(a_prev);
	move_block_sparse_tensor_data(&a_prev_update, a_prev);
	delete_block_sparse_tensor(&r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Left- or right-orthonormalize the MPS using QR decompositions, and return the normalization factor.
///
/// Assuming that rightmost (dummy) virtual bond dimension is 1.
///
double mps_orthonormalize_qr(struct mps* mps, const enum mps_orthonormalization_mode mode)
{
	assert(mps->nsites > 0);

	if (mode == MPS_ORTHONORMAL_LEFT)
	{
		for (int i = 0; i < mps->nsites - 1; i++)
		{
			mps_local_orthonormalize_qr(&mps->a[i], &mps->a[i + 1]);
		}

		// last tensor
		const int i = mps->nsites - 1;
		assert(mps->a[i].dim_logical[2] == 1);

		// create a dummy "tail" tensor
		const long dim_tail[3] = { mps->a[i].dim_logical[2], 1, mps->a[i].dim_logical[2] };
		const enum tensor_axis_direction axis_dir_tail[3] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN };
		qnumber qsite_tail[1] = { 0 };
		const qnumber* qnums_tail[3] = { mps->a[i].qnums_logical[2], qsite_tail, mps->a[i].qnums_logical[2] };
		struct block_sparse_tensor a_tail;
		allocate_block_sparse_tensor(mps->a[i].dtype, 3, dim_tail, axis_dir_tail, qnums_tail, &a_tail);
		assert(a_tail.dim_blocks[0] == 1 && a_tail.dim_blocks[1] == 1 && a_tail.dim_blocks[2] == 1);
		// set single entry to 1
		assert(a_tail.blocks[0] != NULL);
		memcpy(a_tail.blocks[0]->data, numeric_one(a_tail.blocks[0]->dtype), sizeof_numeric_type(a_tail.blocks[0]->dtype));

		// orthonormalize last MPS tensor
		mps_local_orthonormalize_qr(&mps->a[i], &a_tail);

		// retrieve normalization factor (real-valued since diagonal of 'r' matrix is real)
		double nrm = 1;
		switch (a_tail.blocks[0]->dtype)
		{
			case SINGLE_REAL:
			{
				nrm = *((float*)a_tail.blocks[0]->data);
				break;
			}
			case DOUBLE_REAL:
			{
				nrm = *((double*)a_tail.blocks[0]->data);
				break;
			}
			case SINGLE_COMPLEX:
			{
				nrm = crealf(*((scomplex*)a_tail.blocks[0]->data));
				break;
			}
			case DOUBLE_COMPLEX:
			{
				nrm = creal(*((dcomplex*)a_tail.blocks[0]->data));
				break;
			}
			default:
			{
				// unknown data type
				assert(false);
			}
		}
		if (nrm < 0)
		{
			// flip sign such that normalization factor is always non-negative
			rscale_block_sparse_tensor(numeric_neg_one(numeric_real_type(mps->a[i].dtype)), &mps->a[i]);
			nrm = -nrm;
		}

		delete_block_sparse_tensor(&a_tail);

		return nrm;
	}
	else
	{
		assert(mode == MPS_ORTHONORMAL_RIGHT);

		for (int i = mps->nsites - 1; i > 0; i--)
		{
			mps_local_orthonormalize_rq(&mps->a[i], &mps->a[i - 1]);
		}

		// first tensor
		assert(mps->a[0].dim_logical[0] == 1);

		// create a dummy "head" tensor
		const long dim_head[3] = { mps->a[0].dim_logical[0], 1, mps->a[0].dim_logical[0] };
		const enum tensor_axis_direction axis_dir_head[3] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN };
		qnumber qsite_head[1] = { 0 };
		const qnumber* qnums_head[3] = { mps->a[0].qnums_logical[0], qsite_head, mps->a[0].qnums_logical[0] };
		struct block_sparse_tensor a_head;
		allocate_block_sparse_tensor(mps->a[0].dtype, 3, dim_head, axis_dir_head, qnums_head, &a_head);
		assert(a_head.dim_blocks[0] == 1 && a_head.dim_blocks[1] == 1 && a_head.dim_blocks[2] == 1);
		// set single entry to 1
		assert(a_head.blocks[0] != NULL);
		memcpy(a_head.blocks[0]->data, numeric_one(a_head.blocks[0]->dtype), sizeof_numeric_type(a_head.blocks[0]->dtype));

		// orthonormalize first MPS tensor
		mps_local_orthonormalize_rq(&mps->a[0], &a_head);

		// retrieve normalization factor (real-valued since diagonal of 'r' matrix is real)
		double nrm = 1;
		switch (a_head.blocks[0]->dtype)
		{
			case SINGLE_REAL:
			{
				nrm = *((float*)a_head.blocks[0]->data);
				break;
			}
			case DOUBLE_REAL:
			{
				nrm = *((double*)a_head.blocks[0]->data);
				break;
			}
			case SINGLE_COMPLEX:
			{
				nrm = crealf(*((scomplex*)a_head.blocks[0]->data));
				break;
			}
			case DOUBLE_COMPLEX:
			{
				nrm = creal(*((dcomplex*)a_head.blocks[0]->data));
				break;
			}
			default:
			{
				// unknown data type
				assert(false);
			}
		}
		if (nrm < 0)
		{
			// flip sign such that normalization factor is always non-negative
			rscale_block_sparse_tensor(numeric_neg_one(numeric_real_type(mps->a[0].dtype)), &mps->a[0]);
			nrm = -nrm;
		}

		delete_block_sparse_tensor(&a_head);

		return nrm;
	}
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
	block_sparse_tensor_dot(a0, a1, 1, &a0_a1_dot);

	// combine original physical dimensions of a0 and a1 into one dimension
	flatten_block_sparse_tensor_axes(&a0_a1_dot, 1, TENSOR_AXIS_OUT, a);
	delete_block_sparse_tensor(&a0_a1_dot);
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
