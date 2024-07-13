/// \file mps.c
/// \brief Matrix product state (MPS) data structure.

#include <stdlib.h>
#include <memory.h>
#include <math.h>
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
/// \brief Construct a matrix product state with random normal tensor entries, given an overall quantum number sector and maximum virtual bond dimension.
///
void construct_random_mps(const enum numeric_type dtype, const int nsites, const long d, const qnumber* qsite, const qnumber qnum_sector, const long max_vdim, struct rng_state* rng_state, struct mps* mps)
{
	assert(nsites >= 1);
	assert(d >= 1);

	// virtual bond quantum numbers
	long* dim_bonds  = aligned_alloc(MEM_DATA_ALIGN, (nsites + 1) * sizeof(long));
	qnumber** qbonds = aligned_alloc(MEM_DATA_ALIGN, (nsites + 1) * sizeof(qnumber*));
	// dummy left virtual bond; set quantum number to zero
	dim_bonds[0] = 1;
	qbonds[0] = aligned_alloc(MEM_DATA_ALIGN, sizeof(qnumber));
	qbonds[0][0] = 0;
	// dummy right virtual bond; set quantum number to overall quantum number sector
	dim_bonds[nsites] = 1;
	qbonds[nsites] = aligned_alloc(MEM_DATA_ALIGN, sizeof(qnumber));
	qbonds[nsites][0] = qnum_sector;
	// virtual bond quantum numbers on left half
	for (int l = 1; l < (nsites + 1) / 2; l++)
	{
		// enumerate all combinations of left bond quantum numbers and local physical quantum numbers
		const long dim_full = dim_bonds[l - 1] * d;
		qnumber* qnums_full = aligned_alloc(MEM_DATA_ALIGN, dim_full * sizeof(qnumber));
		for (long i = 0; i < dim_bonds[l - 1]; i++) {
			for (long j = 0; j < d; j++) {
				qnums_full[i*d + j] = qbonds[l - 1][i] + qsite[j];
			}
		}
		dim_bonds[l] = lmin(dim_full, max_vdim);
		qbonds[l] = aligned_alloc(MEM_DATA_ALIGN, dim_bonds[l] * sizeof(qnumber));
		if (dim_full <= max_vdim) {
			memcpy(qbonds[l], qnums_full, dim_bonds[l] * sizeof(qnumber));
		}
		else {
			// randomly select quantum numbers
			for (long i = 0; i < dim_bonds[l]; i++) {
				qbonds[l][i] = qnums_full[rand_interval(dim_full, rng_state)];
			}
		}
		aligned_free(qnums_full);
	}
	// virtual bond quantum numbers on right half
	for (int l = nsites - 1; l >= (nsites + 1) / 2; l--)
	{
		// enumerate all combinations of right bond quantum numbers and local physical quantum numbers
		const long dim_full = dim_bonds[l + 1] * d;
		qnumber* qnums_full = aligned_alloc(MEM_DATA_ALIGN, dim_full * sizeof(qnumber));
		for (long i = 0; i < dim_bonds[l + 1]; i++) {
			for (long j = 0; j < d; j++) {
				qnums_full[i*d + j] = qbonds[l + 1][i] - qsite[j];
			}
		}
		dim_bonds[l] = lmin(dim_full, max_vdim);
		qbonds[l] = aligned_alloc(MEM_DATA_ALIGN, dim_bonds[l] * sizeof(qnumber));
		if (dim_full <= max_vdim) {
			memcpy(qbonds[l], qnums_full, dim_bonds[l] * sizeof(qnumber));
		}
		else {
			// randomly select quantum numbers
			for (long i = 0; i < dim_bonds[l]; i++) {
				qbonds[l][i] = qnums_full[rand_interval(dim_full, rng_state)];
			}
		}
		aligned_free(qnums_full);
	}

	allocate_mps(dtype, nsites, d, qsite, dim_bonds, (const qnumber**)qbonds, mps);

	for (int l = 0; l < nsites + 1; l++) {
		aligned_free(qbonds[l]);
	}
	aligned_free(qbonds);
	aligned_free(dim_bonds);

	// fill MPS tensor entries with pseudo-random numbers, scaled by 1 / sqrt("number of entries")
	for (int l = 0; l < nsites; l++)
	{
		// logical number of entries in MPS tensor
		const long nelem = integer_product(mps->a[l].dim_logical, mps->a[l].ndim);
		// ensure that 'alpha' is large enough to store any numeric type
		dcomplex alpha;
		assert(mps->a[l].dtype == dtype);
		numeric_from_double(1.0 / sqrt(nelem), dtype, &alpha);
		block_sparse_tensor_fill_random_normal(&alpha, numeric_zero(dtype), rng_state, &mps->a[l]);
	}
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
/// \brief Contraction step from right to left of two MPS tensors,
/// for example to compute the inner product of two matrix product states.
///
/// To-be contracted tensor network:
///
///       _____           _______
///      /     \         /       \
///   ->-|0 b*2|->-   ->-|1      |
///      \__1__/         |       |
///         |            |       |
///         ^            |   r  2|-<-
///       __|__          |       |
///      /  1  \         |       |
///   -<-|0 a 2|-<-   -<-|0      |
///      \_____/         \_______/
///
static void mps_contraction_step_right(const struct block_sparse_tensor* restrict a, const struct block_sparse_tensor* restrict b,
	const struct block_sparse_tensor* restrict r, struct block_sparse_tensor* restrict r_next)
{
	assert(a->ndim == 3);
	assert(b->ndim == 3);
	assert(r->ndim == 3);

	// multiply with 'a' tensor
	struct block_sparse_tensor s;
	block_sparse_tensor_dot(a, TENSOR_AXIS_RANGE_TRAILING, r, TENSOR_AXIS_RANGE_LEADING, 1, &s);

	// multiply with conjugated 'b' tensor
	struct block_sparse_tensor bc;
	copy_block_sparse_tensor(b, &bc);  // TODO: fuse conjugation with dot product
	conjugate_block_sparse_tensor(&bc);
	block_sparse_tensor_reverse_axis_directions(&bc);
	// temporarily make trailing dimension in 's' the leading dimension
	const int perm0[4] = { 3, 0, 1, 2 };
	struct block_sparse_tensor t;
	transpose_block_sparse_tensor(perm0, &s, &t);
	delete_block_sparse_tensor(&s);
	block_sparse_tensor_dot(&t, TENSOR_AXIS_RANGE_TRAILING, &bc, TENSOR_AXIS_RANGE_TRAILING, 2, &s);
	delete_block_sparse_tensor(&t);
	delete_block_sparse_tensor(&bc);
	// restore original trailing dimension
	const int perm1[3] = { 1, 2, 0 };
	transpose_block_sparse_tensor(perm1, &s, r_next);
	delete_block_sparse_tensor(&s);
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

	const enum numeric_type dtype = psi->a[0].dtype;

	// initialize 'r'
	struct block_sparse_tensor r;
	{
		const long dim[4] = { 1, 1, 1, 1 };
		const enum tensor_axis_direction axis_dir[4] = { TENSOR_AXIS_OUT, TENSOR_AXIS_IN, TENSOR_AXIS_IN, TENSOR_AXIS_OUT };
		const qnumber* qnums[4] = {
			psi->a[psi->nsites - 1].qnums_logical[2],
			chi->a[chi->nsites - 1].qnums_logical[2],
			psi->a[psi->nsites - 1].qnums_logical[2],
			chi->a[chi->nsites - 1].qnums_logical[2], };
		struct block_sparse_tensor t;
		allocate_block_sparse_tensor(dtype, 4, dim, axis_dir, qnums, &t);
		assert(t.blocks[0] != NULL);
		memcpy(t.blocks[0]->data, numeric_one(dtype), sizeof_numeric_type(dtype));
		flatten_block_sparse_tensor_axes(&t, 2, TENSOR_AXIS_IN, &r);
		delete_block_sparse_tensor(&t);
	}

	for (int i = psi->nsites - 1; i >= 0; i--)
	{
		struct block_sparse_tensor r_next;
		mps_contraction_step_right(&psi->a[i], &chi->a[i], &r, &r_next);
		delete_block_sparse_tensor(&r);
		move_block_sparse_tensor_data(&r_next, &r);
	}

	// flatten left virtual bonds
	{
		struct block_sparse_tensor r_flat;
		flatten_block_sparse_tensor_axes(&r, 0, TENSOR_AXIS_OUT, &r_flat);
		delete_block_sparse_tensor(&r);
		move_block_sparse_tensor_data(&r_flat, &r);
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
/// \brief Compute the Euclidean norm of the MPS.
///
/// Result is returned as double also for single-precision tensor entries.
///
double mps_norm(const struct mps* psi)
{
	if (psi->nsites == 0) {
		return 0;
	}

	switch (psi->a[0].dtype)
	{
		case SINGLE_REAL:
		{
			float nrm2;
			mps_vdot(psi, psi, &nrm2);
			assert(nrm2 >= 0);
			return sqrt(nrm2);
		}
		case DOUBLE_REAL:
		{
			double nrm2;
			mps_vdot(psi, psi, &nrm2);
			assert(nrm2 >= 0);
			return sqrt(nrm2);
		}
		case SINGLE_COMPLEX:
		{
			scomplex vdot;
			mps_vdot(psi, psi, &vdot);
			float nrm2 = crealf(vdot);
			assert(nrm2 >= 0);
			return sqrt(nrm2);
		}
		case DOUBLE_COMPLEX:
		{
			dcomplex vdot;
			mps_vdot(psi, psi, &vdot);
			double nrm2 = creal(vdot);
			assert(nrm2 >= 0);
			return sqrt(nrm2);
		}
		default:
		{
			// unknown data type
			assert(false);
			return 0;
		}
	}
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
	block_sparse_tensor_dot(&r, TENSOR_AXIS_RANGE_TRAILING, a_next, TENSOR_AXIS_RANGE_LEADING, 1, &a_next_update);
	delete_block_sparse_tensor(a_next);
	move_block_sparse_tensor_data(&a_next_update, a_next);
	delete_block_sparse_tensor(&r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Right-orthonormalize a local MPS site tensor by RQ decomposition, and update tensor at previous site.
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
	block_sparse_tensor_dot(a_prev, TENSOR_AXIS_RANGE_TRAILING, &r, TENSOR_AXIS_RANGE_LEADING, 1, &a_prev_update);
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
/// \brief Split a MPS tensor with dimension `D0 x d0*d1 x D2` into two MPS tensors
/// with dimensions `D0 x d0 x D1` and `D1 x d1 x D2`, respectively, using SVD.
///
int mps_split_tensor_svd(const struct block_sparse_tensor* restrict a, const long d[2], const qnumber* new_qsite[2],
	const double tol, const long max_vdim, const bool renormalize, const enum singular_value_distr svd_distr,
	struct block_sparse_tensor* restrict a0, struct block_sparse_tensor* restrict a1, struct trunc_info* info)
{
	assert(a->ndim == 3);
	// physical dimension of MPS tensor must be equal to product of new dimensions
	assert(d[0] * d[1] == a->dim_logical[1]);

	// reshape to tensor with two physical legs
	struct block_sparse_tensor a_twosite;
	assert(a->axis_dir[1] == TENSOR_AXIS_OUT);
	const enum tensor_axis_direction axis_dir[2] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT };
	split_block_sparse_tensor_axis(a, 1, d, axis_dir, new_qsite, &a_twosite);
	// reshape to a matrix
	struct block_sparse_tensor tmp;
	flatten_block_sparse_tensor_axes(&a_twosite, 0, TENSOR_AXIS_OUT, &tmp);
	assert(tmp.ndim == 3);
	struct block_sparse_tensor a_mat;
	flatten_block_sparse_tensor_axes(&tmp, 1, TENSOR_AXIS_IN, &a_mat);
	delete_block_sparse_tensor(&tmp);

	// split by truncated SVD
	struct block_sparse_tensor m0, m1;
	int ret = split_block_sparse_matrix_svd(&a_mat, tol, max_vdim, renormalize, svd_distr, &m0, &m1, info);
	delete_block_sparse_tensor(&a_mat);
	if (ret < 0) {
		return ret;
	}

	// restore original virtual bonds and physical axes
	assert(a_twosite.ndim == 4);
	split_block_sparse_tensor_axis(&m0, 0, a_twosite.dim_logical,     a_twosite.axis_dir,     (const qnumber**) a_twosite.qnums_logical,      a0);
	split_block_sparse_tensor_axis(&m1, 1, a_twosite.dim_logical + 2, a_twosite.axis_dir + 2, (const qnumber**)(a_twosite.qnums_logical + 2), a1);

	delete_block_sparse_tensor(&m1);
	delete_block_sparse_tensor(&m0);

	delete_block_sparse_tensor(&a_twosite);

	return 0;
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
	block_sparse_tensor_dot(a0, TENSOR_AXIS_RANGE_TRAILING, a1, TENSOR_AXIS_RANGE_LEADING, 1, &a0_a1_dot);

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
