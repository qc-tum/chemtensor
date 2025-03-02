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
/// \brief Allocate an "empty" matrix product state, without the actual tensors.
///
void allocate_empty_mps(const int nsites, const long d, const qnumber* qsite, struct mps* mps)
{
	assert(nsites >= 1);
	assert(d >= 1);
	mps->nsites = nsites;
	mps->d = d;

	mps->qsite = ct_malloc(d * sizeof(qnumber));
	memcpy(mps->qsite, qsite, d * sizeof(qnumber));

	mps->a = ct_calloc(nsites, sizeof(struct block_sparse_tensor));
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for a matrix product state. 'dim_bonds' and 'qbonds' must be arrays of length 'nsites + 1'.
///
void allocate_mps(const enum numeric_type dtype, const int nsites, const long d, const qnumber* qsite, const long* dim_bonds, const qnumber** qbonds, struct mps* mps)
{
	allocate_empty_mps(nsites, d, qsite, mps);

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
	ct_free(mps->a);
	mps->a = NULL;
	mps->nsites = 0;

	ct_free(mps->qsite);
	mps->qsite = NULL;
	mps->d = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Copy a matrix product state and its block sparse tensors.
///
void copy_mps(const struct mps* restrict src, struct mps* restrict dst)
{
	dst->nsites = src->nsites;
	
	dst->d = src->d;
	dst->qsite = ct_malloc(src->d * sizeof(qnumber));
	memcpy(dst->qsite, src->qsite, src->d * sizeof(qnumber));

	dst->a = ct_malloc(src->nsites * sizeof(struct block_sparse_tensor));
	for (int i = 0; i < src->nsites; i++) {
		copy_block_sparse_tensor(&src->a[i], &dst->a[i]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Move MPS data (without allocating new memory).
///
void move_mps_data(struct mps* restrict src, struct mps* restrict dst)
{
	dst->nsites = src->nsites;
	dst->d      = src->d;
	dst->qsite  = src->qsite;
	dst->a      = src->a;

	src->nsites = 0;
	src->d      = 0;
	src->qsite  = NULL;
	src->a      = NULL;
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
	long* dim_bonds  = ct_malloc((nsites + 1) * sizeof(long));
	qnumber** qbonds = ct_malloc((nsites + 1) * sizeof(qnumber*));
	// dummy left virtual bond; set quantum number to zero
	dim_bonds[0] = 1;
	qbonds[0] = ct_malloc(sizeof(qnumber));
	qbonds[0][0] = 0;
	// dummy right virtual bond; set quantum number to overall quantum number sector
	dim_bonds[nsites] = 1;
	qbonds[nsites] = ct_malloc(sizeof(qnumber));
	qbonds[nsites][0] = qnum_sector;
	// virtual bond quantum numbers on left half
	for (int l = 1; l < (nsites + 1) / 2; l++)
	{
		// enumerate all combinations of left bond quantum numbers and local physical quantum numbers
		const long dim_full = dim_bonds[l - 1] * d;
		qnumber* qnums_full = ct_malloc(dim_full * sizeof(qnumber));
		qnumber_outer_sum(1, qbonds[l - 1], dim_bonds[l - 1], 1, qsite, d, qnums_full);
		dim_bonds[l] = lmin(dim_full, max_vdim);
		qbonds[l] = ct_malloc(dim_bonds[l] * sizeof(qnumber));
		if (dim_full <= max_vdim) {
			memcpy(qbonds[l], qnums_full, dim_bonds[l] * sizeof(qnumber));
		}
		else {
			// randomly select quantum numbers
			uint64_t* idx = ct_malloc(max_vdim * sizeof(uint64_t));
			rand_choice(dim_full, max_vdim, rng_state, idx);
			for (long i = 0; i < max_vdim; i++) {
				qbonds[l][i] = qnums_full[idx[i]];
			}
			ct_free(idx);
		}
		ct_free(qnums_full);
	}
	// virtual bond quantum numbers on right half
	for (int l = nsites - 1; l >= (nsites + 1) / 2; l--)
	{
		// enumerate all combinations of right bond quantum numbers and local physical quantum numbers
		const long dim_full = dim_bonds[l + 1] * d;
		qnumber* qnums_full = ct_malloc(dim_full * sizeof(qnumber));
		qnumber_outer_sum(1, qbonds[l + 1], dim_bonds[l + 1], -1, qsite, d, qnums_full);
		dim_bonds[l] = lmin(dim_full, max_vdim);
		qbonds[l] = ct_malloc(dim_bonds[l] * sizeof(qnumber));
		if (dim_full <= max_vdim) {
			memcpy(qbonds[l], qnums_full, dim_bonds[l] * sizeof(qnumber));
		}
		else {
			// randomly select quantum numbers
			uint64_t* idx = ct_malloc(max_vdim * sizeof(uint64_t));
			rand_choice(dim_full, max_vdim, rng_state, idx);
			for (long i = 0; i < max_vdim; i++) {
				qbonds[l][i] = qnums_full[idx[i]];
			}
			ct_free(idx);
		}
		ct_free(qnums_full);
	}

	allocate_mps(dtype, nsites, d, qsite, dim_bonds, (const qnumber**)qbonds, mps);

	for (int l = 0; l < nsites + 1; l++) {
		ct_free(qbonds[l]);
	}
	ct_free(qbonds);
	ct_free(dim_bonds);

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
/// \brief Compare two MPSs for equality.
///
bool mps_equals(const struct mps* m1, const struct mps* m2) {
	if (m1->nsites != m2->nsites || m1->d != m2->d || !qnumber_all_equal(m1->d, m1->qsite, m2->qsite)) {
		return false;
	}

	for (int i = 0; i < m1->nsites; i++) {
		if (!block_sparse_tensor_allclose(&m1->a[i], &m2->a[i], 0.)) {
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
///        ╭───────╮         ╭─────────╮
///        │       │         │         │
///     ─>─0   b*  2─>─   ─>─1         │
///        │       │         │         │
///        ╰───1───╯         │         │
///            │             │         │
///            ^             │    r    2─<─
///            │             │         │
///        ╭───1───╮         │         │
///        │       │         │         │
///     ─<─0   a   2─<─   ─<─0         │
///        │       │         │         │
///        ╰───────╯         ╰─────────╯
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
		block_sparse_tensor_flatten_axes(&t, 2, TENSOR_AXIS_IN, &r);
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
		block_sparse_tensor_flatten_axes(&r, 0, TENSOR_AXIS_OUT, &r_flat);
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
		case CT_SINGLE_REAL:
		{
			float nrm2;
			mps_vdot(psi, psi, &nrm2);
			assert(nrm2 >= 0);
			return sqrt(nrm2);
		}
		case CT_DOUBLE_REAL:
		{
			double nrm2;
			mps_vdot(psi, psi, &nrm2);
			assert(nrm2 >= 0);
			return sqrt(nrm2);
		}
		case CT_SINGLE_COMPLEX:
		{
			scomplex vdot;
			mps_vdot(psi, psi, &vdot);
			float nrm2 = crealf(vdot);
			assert(nrm2 >= 0);
			return sqrt(nrm2);
		}
		case CT_DOUBLE_COMPLEX:
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
/// \brief Compute the logical addition of two MPS `chi` and `psi` (summing their virtual bond dimensions).
///
void mps_add(const struct mps* chi, const struct mps* psi, struct mps* ret)
{
	// number of lattice sites must agree
	assert(chi->nsites == psi->nsites);
	// number of lattice sites must be larger than 0
	assert(chi->nsites > 0);

	const int nsites = chi->nsites;

	// physical quantum numbers must agree
	assert(chi->d == psi->d);
	assert(qnumber_all_equal(chi->d, chi->qsite, psi->qsite));

	// leading and trailing (dummy) bond quantum numbers must agree
	assert(chi->a[0].dim_logical[0] ==
	       psi->a[0].dim_logical[0]);
	assert(qnumber_all_equal(
			chi->a[0].dim_logical[0],
			chi->a[0].qnums_blocks[0],
			psi->a[0].qnums_blocks[0]));
	assert(chi->a[nsites - 1].dim_logical[2] ==
	       psi->a[nsites - 1].dim_logical[2]);
	assert(qnumber_all_equal(
			chi->a[nsites - 1].dim_logical[2],
			chi->a[nsites - 1].qnums_blocks[2],
			psi->a[nsites - 1].qnums_blocks[2]));

	// initialize return MPS
	allocate_empty_mps(nsites, chi->d, chi->qsite, ret);

	if (nsites == 1)
	{
		// copy sparse tensor into resulting tensor
		copy_block_sparse_tensor(&chi->a[0], &ret->a[0]);

		// add individual dense tensors
		const long nblocks = integer_product(ret->a[0].dim_blocks, ret->a[0].ndim);
		for (long k = 0; k < nblocks; k++)
		{
			struct dense_tensor* a = psi->a[0].blocks[k];
			struct dense_tensor* b = ret->a[0].blocks[k];
			if (a != NULL) {
				assert(b != NULL);
				dense_tensor_scalar_multiply_add(numeric_one(a->dtype), a, b);
			}
		}
	}
	else  // nsites > 1
	{
		// left-most tensor
		{
			const int i_ax[1] = { 2 };
			struct block_sparse_tensor tlist[2] = {
				chi->a[0],
				psi->a[0],
			};
			block_sparse_tensor_block_diag(tlist, 2, i_ax, 1, &ret->a[0]);
		}

		// intermediate tensors
		for (int i = 1; i < nsites - 1; i++) {
			const int i_ax[2] = { 0, 2 };
			struct block_sparse_tensor tlist[2] = {
				chi->a[i],
				psi->a[i],
			};
			block_sparse_tensor_block_diag(tlist, 2, i_ax, 2, &ret->a[i]);
		}

		// right-most tensor
		{
			const int i_ax[1] = { 0 };
			struct block_sparse_tensor tlist[2] = {
				chi->a[nsites - 1],
				psi->a[nsites - 1],
			};
			block_sparse_tensor_block_diag(tlist, 2, i_ax, 1, &ret->a[nsites - 1]);
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
		qnums_logical_left[i] = ct_malloc(dim_logical_left[i] * sizeof(qnumber));
		memcpy(qnums_logical_left[i], a->qnums_logical[i], dim_logical_left[i] * sizeof(qnumber));
	}
	assert(a->axis_dir[0] == TENSOR_AXIS_OUT && a->axis_dir[1] == TENSOR_AXIS_OUT);
	const enum tensor_axis_direction axis_dir_left[2] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT };

	// combine left virtual bond and physical axis
	struct block_sparse_tensor a_mat;
	block_sparse_tensor_flatten_axes(a, 0, TENSOR_AXIS_OUT, &a_mat);
	delete_block_sparse_tensor(a);

	// perform QR decomposition
	struct block_sparse_tensor q, r;
	block_sparse_tensor_qr(&a_mat, &q, &r);
	delete_block_sparse_tensor(&a_mat);

	// replace 'a' by reshaped 'q' matrix
	block_sparse_tensor_split_axis(&q, 0, dim_logical_left, axis_dir_left, (const qnumber**)qnums_logical_left, a);
	delete_block_sparse_tensor(&q);
	for (int i = 0; i < 2; i++)
	{
		ct_free(qnums_logical_left[i]);
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
		qnums_logical_right[i] = ct_malloc(dim_logical_right[i] * sizeof(qnumber));
		memcpy(qnums_logical_right[i], a->qnums_logical[1 + i], dim_logical_right[i] * sizeof(qnumber));
	}
	assert(a->axis_dir[1] == TENSOR_AXIS_OUT && a->axis_dir[2] == TENSOR_AXIS_IN);
	const enum tensor_axis_direction axis_dir_right[2] = { TENSOR_AXIS_OUT, TENSOR_AXIS_IN };

	// combine physical and right virtual bond axis
	struct block_sparse_tensor a_mat;
	block_sparse_tensor_flatten_axes(a, 1, TENSOR_AXIS_IN, &a_mat);
	delete_block_sparse_tensor(a);

	// perform RQ decomposition
	struct block_sparse_tensor r, q;
	block_sparse_tensor_rq(&a_mat, &r, &q);
	delete_block_sparse_tensor(&a_mat);

	// replace 'a' by reshaped 'q' matrix
	block_sparse_tensor_split_axis(&q, 1, dim_logical_right, axis_dir_right, (const qnumber**)qnums_logical_right, a);
	delete_block_sparse_tensor(&q);
	for (int i = 0; i < 2; i++)
	{
		ct_free(qnums_logical_right[i]);
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
		double norm = 0;
		// 'a_tail' can be logically zero in case quantum numbers do not match
		if (a_tail.blocks[0] != NULL)
		{
			switch (a_tail.blocks[0]->dtype)
			{
				case CT_SINGLE_REAL:
				{
					norm = *((float*)a_tail.blocks[0]->data);
					break;
				}
				case CT_DOUBLE_REAL:
				{
					norm = *((double*)a_tail.blocks[0]->data);
					break;
				}
				case CT_SINGLE_COMPLEX:
				{
					norm = crealf(*((scomplex*)a_tail.blocks[0]->data));
					break;
				}
				case CT_DOUBLE_COMPLEX:
				{
					norm = creal(*((dcomplex*)a_tail.blocks[0]->data));
					break;
				}
				default:
				{
					// unknown data type
					assert(false);
				}
			}

			if (norm < 0)
			{
				// flip sign such that normalization factor is always non-negative
				rscale_block_sparse_tensor(numeric_neg_one(numeric_real_type(mps->a[i].dtype)), &mps->a[i]);
				norm = -norm;
			}
		}

		delete_block_sparse_tensor(&a_tail);

		return norm;
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
		double norm = 0;
		// 'a_head' can be logically zero in case quantum numbers do not match
		if (a_head.blocks[0] != NULL)
		{
			switch (a_head.blocks[0]->dtype)
			{
				case CT_SINGLE_REAL:
				{
					norm = *((float*)a_head.blocks[0]->data);
					break;
				}
				case CT_DOUBLE_REAL:
				{
					norm = *((double*)a_head.blocks[0]->data);
					break;
				}
				case CT_SINGLE_COMPLEX:
				{
					norm = crealf(*((scomplex*)a_head.blocks[0]->data));
					break;
				}
				case CT_DOUBLE_COMPLEX:
				{
					norm = creal(*((dcomplex*)a_head.blocks[0]->data));
					break;
				}
				default:
				{
					// unknown data type
					assert(false);
				}
			}
			if (norm < 0)
			{
				// flip sign such that normalization factor is always non-negative
				rscale_block_sparse_tensor(numeric_neg_one(numeric_real_type(mps->a[0].dtype)), &mps->a[0]);
				norm = -norm;
			}
		}

		delete_block_sparse_tensor(&a_head);

		return norm;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Left-orthonormalize a local MPS site tensor by a SVD with truncation, and update tensor at next site.
///
int mps_local_orthonormalize_left_svd(const double tol, const long max_vdim, const bool renormalize, struct block_sparse_tensor* restrict a, struct block_sparse_tensor* restrict a_next, struct trunc_info* info)
{
	assert(a->ndim == 3);
	assert(a_next->ndim == 3);

	// save original logical dimensions and quantum numbers for later splitting
	const long dim_logical_left[2] = { a->dim_logical[0], a->dim_logical[1] };
	qnumber* qnums_logical_left[2];
	for (int i = 0; i < 2; i++)
	{
		qnums_logical_left[i] = ct_malloc(dim_logical_left[i] * sizeof(qnumber));
		memcpy(qnums_logical_left[i], a->qnums_logical[i], dim_logical_left[i] * sizeof(qnumber));
	}
	assert(a->axis_dir[0] == TENSOR_AXIS_OUT && a->axis_dir[1] == TENSOR_AXIS_OUT);
	const enum tensor_axis_direction axis_dir_left[2] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT };

	// combine left virtual bond and physical axis
	struct block_sparse_tensor a_mat;
	block_sparse_tensor_flatten_axes(a, 0, TENSOR_AXIS_OUT, &a_mat);
	delete_block_sparse_tensor(a);

	// perform truncated SVD
	struct block_sparse_tensor m0, m1;
	int ret = split_block_sparse_matrix_svd(&a_mat, tol, max_vdim, renormalize, SVD_DISTR_RIGHT, &m0, &m1, info);
	delete_block_sparse_tensor(&a_mat);
	if (ret < 0) {
		return ret;
	}

	// replace 'a' by reshaped 'm0' matrix
	block_sparse_tensor_split_axis(&m0, 0, dim_logical_left, axis_dir_left, (const qnumber**)qnums_logical_left, a);
	delete_block_sparse_tensor(&m0);
	for (int i = 0; i < 2; i++) {
		ct_free(qnums_logical_left[i]);
	}

	// update 'a_next' tensor: multiply with 'm1' from left
	struct block_sparse_tensor a_next_update;
	block_sparse_tensor_dot(&m1, TENSOR_AXIS_RANGE_TRAILING, a_next, TENSOR_AXIS_RANGE_LEADING, 1, &a_next_update);
	delete_block_sparse_tensor(a_next);
	move_block_sparse_tensor_data(&a_next_update, a_next);
	delete_block_sparse_tensor(&m1);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Right-orthonormalize a local MPS site tensor by a SVD with truncation, and update tensor at previous site.
///
int mps_local_orthonormalize_right_svd(const double tol, const long max_vdim, const bool renormalize, struct block_sparse_tensor* restrict a, struct block_sparse_tensor* restrict a_prev, struct trunc_info* info)
{
	assert(a->ndim == 3);
	assert(a_prev->ndim == 3);

	// save original logical dimensions and quantum numbers for later splitting
	const long dim_logical_right[2] = { a->dim_logical[1], a->dim_logical[2] };
	qnumber* qnums_logical_right[2];
	for (int i = 0; i < 2; i++)
	{
		qnums_logical_right[i] = ct_malloc(dim_logical_right[i] * sizeof(qnumber));
		memcpy(qnums_logical_right[i], a->qnums_logical[1 + i], dim_logical_right[i] * sizeof(qnumber));
	}
	assert(a->axis_dir[1] == TENSOR_AXIS_OUT && a->axis_dir[2] == TENSOR_AXIS_IN);
	const enum tensor_axis_direction axis_dir_right[2] = { TENSOR_AXIS_OUT, TENSOR_AXIS_IN };

	// combine physical and right virtual bond axis
	struct block_sparse_tensor a_mat;
	block_sparse_tensor_flatten_axes(a, 1, TENSOR_AXIS_IN, &a_mat);
	delete_block_sparse_tensor(a);

	// perform truncated SVD
	struct block_sparse_tensor m0, m1;
	int ret = split_block_sparse_matrix_svd(&a_mat, tol, max_vdim, renormalize, SVD_DISTR_LEFT, &m0, &m1, info);
	delete_block_sparse_tensor(&a_mat);
	if (ret < 0) {
		return ret;
	}

	// replace 'a' by reshaped 'm1' matrix
	block_sparse_tensor_split_axis(&m1, 1, dim_logical_right, axis_dir_right, (const qnumber**)qnums_logical_right, a);
	delete_block_sparse_tensor(&m1);
	for (int i = 0; i < 2; i++) {
		ct_free(qnums_logical_right[i]);
	}

	// update 'a_prev' tensor: multiply with 'm0' from right
	struct block_sparse_tensor a_prev_update;
	block_sparse_tensor_dot(a_prev, TENSOR_AXIS_RANGE_TRAILING, &m0, TENSOR_AXIS_RANGE_LEADING, 1, &a_prev_update);
	delete_block_sparse_tensor(a_prev);
	move_block_sparse_tensor_data(&a_prev_update, a_prev);
	delete_block_sparse_tensor(&m0);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compress and orthonormalize an MPS by site-local SVDs and singular value truncations.
///
/// Returns original norm and scaling factor due to compression.
///
int mps_compress(const double tol, const long max_vdim, const enum mps_orthonormalization_mode mode,
	struct mps* mps, double* restrict norm, double* restrict trunc_scale, struct trunc_info* info)
{
	const bool renormalize = false;

	if (mode == MPS_ORTHONORMAL_LEFT)
	{
		// transform to right-canonical form first
		(*norm) = mps_orthonormalize_qr(mps, MPS_ORTHONORMAL_RIGHT);

		for (int i = 0; i < mps->nsites - 1; i++)
		{
			int ret = mps_local_orthonormalize_left_svd(tol, max_vdim, renormalize, &mps->a[i], &mps->a[i + 1], &info[i]);
			if (ret < 0) {
				return ret;
			}
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
		int ret = mps_local_orthonormalize_left_svd(tol, max_vdim, renormalize, &mps->a[i], &a_tail, &info[i]);
		if (ret < 0) {
			return ret;
		}
		assert(a_tail.dtype == mps->a[i].dtype);
		assert(a_tail.dim_logical[0] == 1 && a_tail.dim_logical[1] == 1 && a_tail.dim_logical[2] == 1);
		// quantum numbers for 'a_tail' should match due to preceeding QR orthonormalization
		assert(a_tail.blocks[0] != NULL);
		switch (a_tail.blocks[0]->dtype)
		{
			case CT_SINGLE_REAL:
			{
				float d = *((float*)a_tail.blocks[0]->data);
				// absorb potential phase factor into MPS tensor
				if (d < 0) {
					scale_block_sparse_tensor(numeric_neg_one(CT_SINGLE_REAL), &mps->a[i]);
				}
				(*trunc_scale) = fabsf(d);
				break;
			}
			case CT_DOUBLE_REAL:
			{
				double d = *((double*)a_tail.blocks[0]->data);
				// absorb potential phase factor into MPS tensor
				if (d < 0) {
					scale_block_sparse_tensor(numeric_neg_one(CT_DOUBLE_REAL), &mps->a[i]);
				}
				(*trunc_scale) = fabs(d);
				break;
			}
			case CT_SINGLE_COMPLEX:
			{
				scomplex d = *((scomplex*)a_tail.blocks[0]->data);
				// absorb potential phase factor into MPS tensor
				float abs_d = cabsf(d);
				if (abs_d != 0) {
					scomplex phase = d / abs_d;
					scale_block_sparse_tensor(&phase, &mps->a[i]);
				}
				(*trunc_scale) = abs_d;
				break;
			}
			case CT_DOUBLE_COMPLEX:
			{
				dcomplex d = *((dcomplex*)a_tail.blocks[0]->data);
				// absorb potential phase factor into MPS tensor
				double abs_d = cabs(d);
				if (abs_d != 0) {
					dcomplex phase = d / abs_d;
					scale_block_sparse_tensor(&phase, &mps->a[i]);
				}
				(*trunc_scale) = abs_d;
				break;
			}
			default:
			{
				// unknown data type
				assert(false);
			}
		}

		delete_block_sparse_tensor(&a_tail);
	}
	else
	{
		assert(mode == MPS_ORTHONORMAL_RIGHT);

		// transform to left-canonical form first
		(*norm) = mps_orthonormalize_qr(mps, MPS_ORTHONORMAL_LEFT);

		for (int i = mps->nsites - 1; i > 0; i--)
		{
			int ret = mps_local_orthonormalize_right_svd(tol, max_vdim, renormalize, &mps->a[i], &mps->a[i - 1], &info[i]);
			if (ret < 0) {
				return ret;
			}
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
		int ret = mps_local_orthonormalize_right_svd(tol, max_vdim, renormalize, &mps->a[0], &a_head, &info[0]);
		if (ret < 0) {
			return ret;
		}
		assert(a_head.dtype == mps->a[0].dtype);
		assert(a_head.dim_logical[0] == 1 && a_head.dim_logical[1] == 1 && a_head.dim_logical[2] == 1);
		// quantum numbers for 'a_head' should match due to preceeding QR orthonormalization
		assert(a_head.blocks[0] != NULL);
		switch (a_head.blocks[0]->dtype)
		{
			case CT_SINGLE_REAL:
			{
				float d = *((float*)a_head.blocks[0]->data);
				// absorb potential phase factor into MPS tensor
				if (d < 0) {
					scale_block_sparse_tensor(numeric_neg_one(CT_SINGLE_REAL), &mps->a[0]);
				}
				(*trunc_scale) = fabsf(d);
				break;
			}
			case CT_DOUBLE_REAL:
			{
				double d = *((double*)a_head.blocks[0]->data);
				// absorb potential phase factor into MPS tensor
				if (d < 0) {
					scale_block_sparse_tensor(numeric_neg_one(CT_DOUBLE_REAL), &mps->a[0]);
				}
				(*trunc_scale) = fabs(d);
				break;
			}
			case CT_SINGLE_COMPLEX:
			{
				scomplex d = *((scomplex*)a_head.blocks[0]->data);
				// absorb potential phase factor into MPS tensor
				float abs_d = cabsf(d);
				if (abs_d != 0) {
					scomplex phase = d / abs_d;
					scale_block_sparse_tensor(&phase, &mps->a[0]);
				}
				(*trunc_scale) = abs_d;
				break;
			}
			case CT_DOUBLE_COMPLEX:
			{
				dcomplex d = *((dcomplex*)a_head.blocks[0]->data);
				// absorb potential phase factor into MPS tensor
				double abs_d = cabs(d);
				if (abs_d != 0) {
					dcomplex phase = d / abs_d;
					scale_block_sparse_tensor(&phase, &mps->a[0]);
				}
				(*trunc_scale) = abs_d;
				break;
			}
			default:
			{
				// unknown data type
				assert(false);
			}
		}

		delete_block_sparse_tensor(&a_head);
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compress an MPS and rescale it by its original norm.
///
/// The rescaling is applied to the first tensor for right orthonormalization and to the last tensor for left orthonormalization.
///
int mps_compress_rescale(const double tol, const long max_vdim, const enum mps_orthonormalization_mode mode,
	struct mps* mps, double* trunc_scale, struct trunc_info* info)
{
	double norm;
	int ret = mps_compress(tol, max_vdim, mode, mps, &norm, trunc_scale, info);
	if (ret < 0) {
		return ret;
	}

	// rescale by original normalization factor
	if (mode == MPS_ORTHONORMAL_LEFT)
	{
		const int i = mps->nsites - 1;

		if (numeric_real_type(mps->a[i].dtype) == CT_DOUBLE_REAL)
		{
			rscale_block_sparse_tensor(&norm, &mps->a[i]);
		}
		else
		{
			assert(numeric_real_type(mps->a[i].dtype) == CT_SINGLE_REAL);
			const float normf = (float)norm;
			rscale_block_sparse_tensor(&normf, &mps->a[i]);
		}
	}
	else
	{
		if (numeric_real_type(mps->a[0].dtype) == CT_DOUBLE_REAL)
		{
			rscale_block_sparse_tensor(&norm, &mps->a[0]);
		}
		else
		{
			assert(numeric_real_type(mps->a[0].dtype) == CT_SINGLE_REAL);
			const float normf = (float)norm;
			rscale_block_sparse_tensor(&normf, &mps->a[0]);
		}
	}

	return 0;
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
	block_sparse_tensor_split_axis(a, 1, d, axis_dir, new_qsite, &a_twosite);
	// reshape to a matrix
	struct block_sparse_tensor tmp;
	block_sparse_tensor_flatten_axes(&a_twosite, 0, TENSOR_AXIS_OUT, &tmp);
	assert(tmp.ndim == 3);
	struct block_sparse_tensor a_mat;
	block_sparse_tensor_flatten_axes(&tmp, 1, TENSOR_AXIS_IN, &a_mat);
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
	block_sparse_tensor_split_axis(&m0, 0, a_twosite.dim_logical,     a_twosite.axis_dir,     (const qnumber**) a_twosite.qnums_logical,      a0);
	block_sparse_tensor_split_axis(&m1, 1, a_twosite.dim_logical + 2, a_twosite.axis_dir + 2, (const qnumber**)(a_twosite.qnums_logical + 2), a1);

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
	block_sparse_tensor_flatten_axes(&a0_a1_dot, 1, TENSOR_AXIS_OUT, a);
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