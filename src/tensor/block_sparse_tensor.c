/// \file block_sparse_tensor.c
/// \brief Block-sparse tensor structure.

#include <memory.h>
#include <math.h>
#include <inttypes.h>
#include <cblas.h>
#include "block_sparse_tensor.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Temporary data structure for enumerating quantum numbers and their multiplicities.
///
struct qnumber_count
{
	qnumber qnum;
	int count;
};

//________________________________________________________________________________________________________________________
///
/// \brief Comparison function for sorting.
///
static int compare_qnumber_count(const void* a, const void* b)
{
	const struct qnumber_count* x = (const struct qnumber_count*)a;
	const struct qnumber_count* y = (const struct qnumber_count*)b;
	// assuming that quantum numbers are distinct
	return x->qnum - y->qnum;
}

//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for a block-sparse tensor, including the dense blocks for conserved quantum numbers.
///
void allocate_block_sparse_tensor(const enum numeric_type dtype, const int ndim, const long* restrict dim, const enum tensor_axis_direction* axis_dir, const qnumber** restrict qnums, struct block_sparse_tensor* restrict t)
{
	t->dtype = dtype;

	assert(ndim >= 0);
	t->ndim = ndim;

	if (ndim == 0)  // special case
	{
		t->dim_logical = NULL;
		t->dim_blocks  = NULL;

		t->axis_dir = NULL;

		t->qnums_logical = NULL;
		t->qnums_blocks  = NULL;

		// allocate memory for a single block
		t->blocks = ct_calloc(1, sizeof(struct dense_tensor*));
		t->blocks[0] = ct_calloc(1, sizeof(struct dense_tensor));
		allocate_dense_tensor(dtype, ndim, dim, t->blocks[0]);

		return;
	}

	t->dim_logical = ct_malloc(ndim * sizeof(long));
	memcpy(t->dim_logical, dim, ndim * sizeof(long));

	t->dim_blocks = ct_calloc(ndim, sizeof(long));

	t->axis_dir = ct_malloc(ndim * sizeof(enum tensor_axis_direction));
	memcpy(t->axis_dir, axis_dir, ndim * sizeof(enum tensor_axis_direction));

	t->qnums_logical = ct_calloc(ndim, sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		t->qnums_logical[i] = ct_malloc(dim[i] * sizeof(qnumber));
		memcpy(t->qnums_logical[i], qnums[i], dim[i] * sizeof(qnumber));
	}

	t->qnums_blocks = ct_calloc(ndim, sizeof(qnumber*));

	// count quantum numbers along each dimension axis
	struct qnumber_count** qcounts = ct_malloc(ndim * sizeof(struct qnumber_count*));
	for (int i = 0; i < ndim; i++)
	{
		assert(dim[i] > 0);

		// likely not requiring all the allocated memory
		qcounts[i] = ct_calloc(dim[i], sizeof(struct qnumber_count));
		long nqc = 0;
		for (long j = 0; j < dim[i]; j++)
		{
			int k;
			for (k = 0; k < nqc; k++)
			{
				if (qcounts[i][k].qnum == qnums[i][j]) {
					qcounts[i][k].count++;
					break;
				}
			}
			if (k == nqc)
			{
				// add new entry
				qcounts[i][k].qnum = qnums[i][j];
				qcounts[i][k].count = 1;
				nqc++;
			}
		}
		assert(nqc <= dim[i]);

		qsort(qcounts[i], nqc, sizeof(struct qnumber_count), compare_qnumber_count);

		t->dim_blocks[i] = nqc;

		// store quantum numbers
		t->qnums_blocks[i] = ct_calloc(nqc, sizeof(qnumber));
		for (long j = 0; j < nqc; j++) {
			t->qnums_blocks[i][j] = qcounts[i][j].qnum;
		}
	}

	// allocate dense tensor blocks
	const long nblocks = integer_product(t->dim_blocks, ndim);
	t->blocks = ct_calloc(nblocks, sizeof(struct dense_tensor*));
	long* index_block = ct_calloc(ndim, sizeof(long));
	for (long k = 0; k < nblocks; k++, next_tensor_index(ndim, t->dim_blocks, index_block))
	{
		// probe whether quantum numbers sum to zero
		qnumber qsum = 0;
		for (int i = 0; i < ndim; i++)
		{
			qsum += axis_dir[i] * t->qnums_blocks[i][index_block[i]];
		}
		if (qsum != 0) {
			continue;
		}

		// allocate dense tensor block
		t->blocks[k] = ct_calloc(1, sizeof(struct dense_tensor));
		long* bdim = ct_malloc(ndim * sizeof(long));
		for (int i = 0; i < ndim; i++) {
			bdim[i] = qcounts[i][index_block[i]].count;
		}
		allocate_dense_tensor(dtype, ndim, bdim, t->blocks[k]);
		ct_free(bdim);
	}
	ct_free(index_block);

	for (int i = 0; i < ndim; i++)
	{
		ct_free(qcounts[i]);
	}
	ct_free(qcounts);
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for a block-sparse tensor of the same type, dimensions, and quantum numbers as the provided tensor.
///
void allocate_block_sparse_tensor_like(const struct block_sparse_tensor* restrict s, struct block_sparse_tensor* restrict t)
{
	allocate_block_sparse_tensor(s->dtype, s->ndim, s->dim_logical, s->axis_dir, (const qnumber**)s->qnums_logical, t);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a block-sparse tensor (free memory).
///
void delete_block_sparse_tensor(struct block_sparse_tensor* t)
{
	if (t->ndim == 0)  // special case
	{
		delete_dense_tensor(t->blocks[0]);
		ct_free(t->blocks[0]);
		ct_free(t->blocks);

		return;
	}

	// free dense tensor blocks
	const long nblocks = integer_product(t->dim_blocks, t->ndim);
	for (long k = 0; k < nblocks; k++)
	{
		if (t->blocks[k] != NULL)
		{
			delete_dense_tensor(t->blocks[k]);
			ct_free(t->blocks[k]);
			t->blocks[k] = NULL;
		}
	}
	ct_free(t->blocks);
	t->blocks = NULL;

	for (int i = 0; i < t->ndim; i++)
	{
		ct_free(t->qnums_blocks[i]);
		ct_free(t->qnums_logical[i]);
	}
	ct_free(t->qnums_blocks);
	ct_free(t->qnums_logical);

	ct_free(t->axis_dir);

	ct_free(t->dim_blocks);
	ct_free(t->dim_logical);
}


//________________________________________________________________________________________________________________________
///
/// \brief Copy a block-sparse tensor, allocating memory for the copy.
///
void copy_block_sparse_tensor(const struct block_sparse_tensor* restrict src, struct block_sparse_tensor* restrict dst)
{
	dst->dtype = src->dtype;

	const int ndim = src->ndim;
	dst->ndim = ndim;

	if (ndim == 0)  // special case
	{
		dst->dim_logical = NULL;
		dst->dim_blocks  = NULL;

		dst->axis_dir = NULL;

		dst->qnums_logical = NULL;
		dst->qnums_blocks  = NULL;

		// allocate memory for a single block
		dst->blocks = ct_calloc(1, sizeof(struct dense_tensor*));
		dst->blocks[0] = ct_calloc(1, sizeof(struct dense_tensor));
		copy_dense_tensor(src->blocks[0], dst->blocks[0]);

		return;
	}

	dst->dim_blocks = ct_calloc(ndim, sizeof(long));
	memcpy(dst->dim_blocks, src->dim_blocks, ndim * sizeof(long));

	dst->dim_logical = ct_malloc(ndim * sizeof(long));
	memcpy(dst->dim_logical, src->dim_logical, ndim * sizeof(long));

	dst->axis_dir = ct_malloc(ndim * sizeof(enum tensor_axis_direction));
	memcpy(dst->axis_dir, src->axis_dir, ndim * sizeof(enum tensor_axis_direction));

	dst->qnums_blocks = ct_calloc(ndim, sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		dst->qnums_blocks[i] = ct_malloc(src->dim_blocks[i] * sizeof(qnumber));
		memcpy(dst->qnums_blocks[i], src->qnums_blocks[i], src->dim_blocks[i] * sizeof(qnumber));
	}

	dst->qnums_logical = ct_calloc(ndim, sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		dst->qnums_logical[i] = ct_malloc(src->dim_logical[i] * sizeof(qnumber));
		memcpy(dst->qnums_logical[i], src->qnums_logical[i], src->dim_logical[i] * sizeof(qnumber));
	}

	// copy dense tensor blocks
	const long nblocks = integer_product(src->dim_blocks, ndim);
	dst->blocks = ct_calloc(nblocks, sizeof(struct dense_tensor*));
	#pragma omp parallel for schedule(dynamic)
	for (long k = 0; k < nblocks; k++)
	{
		if (src->blocks[k] == NULL) {
			continue;
		}

		// allocate and copy dense tensor block
		dst->blocks[k] = ct_calloc(1, sizeof(struct dense_tensor));
		copy_dense_tensor(src->blocks[k], dst->blocks[k]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Move tensor data (without allocating new memory).
///
void move_block_sparse_tensor_data(struct block_sparse_tensor* restrict src, struct block_sparse_tensor* restrict dst)
{
	dst->blocks        = src->blocks;
	dst->dim_blocks    = src->dim_blocks;
	dst->dim_logical   = src->dim_logical;
	dst->axis_dir      = src->axis_dir;
	dst->qnums_blocks  = src->qnums_blocks;
	dst->qnums_logical = src->qnums_logical;
	dst->dtype         = src->dtype;
	dst->ndim          = src->ndim;

	src->blocks        = NULL;
	src->dim_blocks    = NULL;
	src->dim_logical   = NULL;
	src->axis_dir      = NULL;
	src->qnums_blocks  = NULL;
	src->qnums_logical = NULL;
}


//________________________________________________________________________________________________________________________
///
/// \brief Retrieve a dense block based on its quantum numbers.
///
struct dense_tensor* block_sparse_tensor_get_block(const struct block_sparse_tensor* t, const qnumber* qnums)
{
	assert(t->ndim > 0);

	// find indices corresponding to quantum numbers
	long* index = ct_malloc(t->ndim * sizeof(long));
	for (int i = 0; i < t->ndim; i++)
	{
		index[i] = -1;
		for (long j = 0; j < t->dim_blocks[i]; j++)
		{
			if (t->qnums_blocks[i][j] == qnums[i]) {
				index[i] = j;
				break;
			}
		}
		if (index[i] == -1) {
			// quantum number not found
			ct_free(index);
			return NULL;
		}
	}

	long o = tensor_index_to_offset(t->ndim, t->dim_blocks, index);

	ct_free(index);

	return t->blocks[o];
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the 2-norm (Frobenius norm) of the tensor.
///
/// Result is returned as double also for single-precision tensor entries.
///
double block_sparse_tensor_norm2(const struct block_sparse_tensor* t)
{
	double nrm = 0;

	const long nblocks = integer_product(t->dim_blocks, t->ndim);
	#pragma omp parallel for schedule(dynamic) reduction(+: nrm)
	for (long k = 0; k < nblocks; k++)
	{
		struct dense_tensor* b = t->blocks[k];
		if (b != NULL) {
			nrm += square(dense_tensor_norm2(b));
		}
	}
	nrm = sqrt(nrm);

	return nrm;
}


//________________________________________________________________________________________________________________________
///
/// \brief Reverse the tensor axis directions.
///
/// Reversing the axis directions does not affect the block sparsity structure.
///
void block_sparse_tensor_reverse_axis_directions(struct block_sparse_tensor* t)
{
	for (int i = 0; i < t->ndim; i++)
	{
		t->axis_dir[i] *= (-1);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Scale tensor 't' by 'alpha'.
///
/// Data types of all blocks and of 'alpha' must match.
///
void scale_block_sparse_tensor(const void* alpha, struct block_sparse_tensor* t)
{
	const long nblocks = integer_product(t->dim_blocks, t->ndim);
	#pragma omp parallel for schedule(dynamic)
	for (long k = 0; k < nblocks; k++)
	{
		struct dense_tensor* b = t->blocks[k];
		if (b != NULL) {
			scale_dense_tensor(alpha, b);
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Scale tensor 't' by a real number 'alpha'.
///
/// Data type precision of all blocks and of 'alpha' must match.
///
void rscale_block_sparse_tensor(const void* alpha, struct block_sparse_tensor* t)
{
	const long nblocks = integer_product(t->dim_blocks, t->ndim);
	#pragma omp parallel for schedule(dynamic)
	for (long k = 0; k < nblocks; k++)
	{
		struct dense_tensor* b = t->blocks[k];
		if (b != NULL) {
			rscale_dense_tensor(alpha, b);
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Elementwise conjugation of a block-sparse tensor.
///
void conjugate_block_sparse_tensor(struct block_sparse_tensor* t)
{
	if ((t->dtype == CT_SINGLE_REAL) || (t->dtype == CT_DOUBLE_REAL)) {
		// no effect
		return;
	}

	const long nblocks = integer_product(t->dim_blocks, t->ndim);
	#pragma omp parallel for schedule(dynamic)
	for (long k = 0; k < nblocks; k++)
	{
		struct dense_tensor* b = t->blocks[k];
		if (b != NULL) {
			conjugate_dense_tensor(b);
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Fill the non-zero blocks of a block-sparse tensor with random normal entries.
///
void block_sparse_tensor_fill_random_normal(const void* alpha, const void* shift, struct rng_state* rng_state, struct block_sparse_tensor* t)
{
	const long nblocks = integer_product(t->dim_blocks, t->ndim);
	// not using OpenMP parallelization here due to random state
	for (long k = 0; k < nblocks; k++)
	{
		struct dense_tensor* b = t->blocks[k];
		if (b != NULL) {
			dense_tensor_fill_random_normal(alpha, shift, rng_state, b);
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert a block-sparse to an equivalent dense tensor.
///
void block_sparse_to_dense_tensor(const struct block_sparse_tensor* restrict s, struct dense_tensor* restrict t)
{
	allocate_dense_tensor(s->dtype, s->ndim, s->dim_logical, t);

	// for each block with matching quantum numbers...
	const long nblocks = integer_product(s->dim_blocks, s->ndim);
	#pragma omp parallel for schedule(dynamic)
	for (long k = 0; k < nblocks; k++)
	{
		const struct dense_tensor* b = s->blocks[k];
		if (b == NULL) {
			continue;
		}
		assert(b->ndim == s->ndim);
		assert(b->dtype == s->dtype);

		long* index_block = ct_malloc(s->ndim * sizeof(long));
		offset_to_tensor_index(s->ndim, s->dim_blocks, k, index_block);

		// fan-out dense to logical indices
		long** index_map = ct_calloc(s->ndim, sizeof(long*));
		for (int i = 0; i < s->ndim; i++)
		{
			index_map[i] = ct_calloc(b->dim[i], sizeof(long));
			long c = 0;
			for (long j = 0; j < s->dim_logical[i]; j++)
			{
				if (s->qnums_logical[i][j] == s->qnums_blocks[i][index_block[i]])
				{
					index_map[i][c] = j;
					c++;
				}
			}
			assert(c == b->dim[i]);
		}

		// distribute dense tensor entries
		long* index_b = ct_calloc(b->ndim, sizeof(long));
		long* index_t = ct_calloc(t->ndim, sizeof(long));
		const long nelem = dense_tensor_num_elements(b);
		switch (s->dtype)
		{
			case CT_SINGLE_REAL:
			{
				const float* bdata = b->data;
				float*       tdata = t->data;
				for (long j = 0; j < nelem; j++, next_tensor_index(b->ndim, b->dim, index_b))
				{
					for (int i = 0; i < b->ndim; i++)
					{
						index_t[i] = index_map[i][index_b[i]];
					}
					tdata[tensor_index_to_offset(t->ndim, t->dim, index_t)] = bdata[j];
				}
				break;
			}
			case CT_DOUBLE_REAL:
			{
				const double* bdata = b->data;
				double*       tdata = t->data;
				for (long j = 0; j < nelem; j++, next_tensor_index(b->ndim, b->dim, index_b))
				{
					for (int i = 0; i < b->ndim; i++)
					{
						index_t[i] = index_map[i][index_b[i]];
					}
					tdata[tensor_index_to_offset(t->ndim, t->dim, index_t)] = bdata[j];
				}
				break;
			}
			case CT_SINGLE_COMPLEX:
			{
				const scomplex* bdata = b->data;
				scomplex*       tdata = t->data;
				for (long j = 0; j < nelem; j++, next_tensor_index(b->ndim, b->dim, index_b))
				{
					for (int i = 0; i < b->ndim; i++)
					{
						index_t[i] = index_map[i][index_b[i]];
					}
					tdata[tensor_index_to_offset(t->ndim, t->dim, index_t)] = bdata[j];
				}
				break;
			}
			case CT_DOUBLE_COMPLEX:
			{
				const dcomplex* bdata = b->data;
				dcomplex*       tdata = t->data;
				for (long j = 0; j < nelem; j++, next_tensor_index(b->ndim, b->dim, index_b))
				{
					for (int i = 0; i < b->ndim; i++)
					{
						index_t[i] = index_map[i][index_b[i]];
					}
					tdata[tensor_index_to_offset(t->ndim, t->dim, index_t)] = bdata[j];
				}
				break;
			}
			default:
			{
				// unknown data type
				assert(false);
			}
		}
		ct_free(index_t);
		ct_free(index_b);

		for (int i = 0; i < s->ndim; i++) {
			ct_free(index_map[i]);
		}
		ct_free(index_map);

		ct_free(index_block);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert a dense to an equivalent block-sparse tensor, using the sparsity pattern imposed by the provided quantum numbers.
///
/// Entries in the dense tensor not adhering to the quantum number sparsity pattern are ignored.
///
void dense_to_block_sparse_tensor(const struct dense_tensor* restrict t, const enum tensor_axis_direction* axis_dir, const qnumber** restrict qnums, struct block_sparse_tensor* restrict s)
{
	allocate_block_sparse_tensor(t->dtype, t->ndim, t->dim, axis_dir, qnums, s);

	dense_to_block_sparse_tensor_entries(t, s);
}


//________________________________________________________________________________________________________________________
///
/// \brief Copy entries of a dense to an already allocated block-sparse tensor.
///
/// Entries in the dense tensor not adhering to the quantum number sparsity pattern are ignored.
///
void dense_to_block_sparse_tensor_entries(const struct dense_tensor* restrict t, struct block_sparse_tensor* restrict s)
{
	// data types must match
	assert(t->dtype == s->dtype);

	// dimensions must match
	assert(t->ndim == s->ndim);
	for (int i = 0; i < t->ndim; i++) {
		assert(t->dim[i] == s->dim_logical[i]);
	}

	// for each block with matching quantum numbers...
	const long nblocks = integer_product(s->dim_blocks, s->ndim);
	#pragma omp parallel for schedule(dynamic)
	for (long k = 0; k < nblocks; k++)
	{
		struct dense_tensor* b = s->blocks[k];
		if (b == NULL) {
			continue;
		}
		assert(b->ndim == s->ndim);
		assert(b->dtype == t->dtype);

		long* index_block = ct_malloc(s->ndim * sizeof(long));
		offset_to_tensor_index(s->ndim, s->dim_blocks, k, index_block);

		// fan-out dense to logical indices
		long** index_map = ct_calloc(s->ndim, sizeof(long*));
		for (int i = 0; i < s->ndim; i++)
		{
			index_map[i] = ct_calloc(b->dim[i], sizeof(long));
			long c = 0;
			for (long j = 0; j < s->dim_logical[i]; j++)
			{
				if (s->qnums_logical[i][j] == s->qnums_blocks[i][index_block[i]])
				{
					index_map[i][c] = j;
					c++;
				}
			}
			assert(c == b->dim[i]);
		}

		// collect dense tensor entries
		long* index_b = ct_calloc(b->ndim, sizeof(long));
		long* index_t = ct_calloc(t->ndim, sizeof(long));
		const long nelem = dense_tensor_num_elements(b);
		switch (s->dtype)
		{
			case CT_SINGLE_REAL:
			{
				const float* tdata = t->data;
				float*       bdata = b->data;
				for (long j = 0; j < nelem; j++, next_tensor_index(b->ndim, b->dim, index_b))
				{
					for (int i = 0; i < b->ndim; i++)
					{
						index_t[i] = index_map[i][index_b[i]];
					}
					bdata[j] = tdata[tensor_index_to_offset(t->ndim, t->dim, index_t)];
				}
				break;
			}
			case CT_DOUBLE_REAL:
			{
				const double* tdata = t->data;
				double*       bdata = b->data;
				for (long j = 0; j < nelem; j++, next_tensor_index(b->ndim, b->dim, index_b))
				{
					for (int i = 0; i < b->ndim; i++)
					{
						index_t[i] = index_map[i][index_b[i]];
					}
					bdata[j] = tdata[tensor_index_to_offset(t->ndim, t->dim, index_t)];
				}
				break;
			}
			case CT_SINGLE_COMPLEX:
			{
				const scomplex* tdata = t->data;
				scomplex*       bdata = b->data;
				for (long j = 0; j < nelem; j++, next_tensor_index(b->ndim, b->dim, index_b))
				{
					for (int i = 0; i < b->ndim; i++)
					{
						index_t[i] = index_map[i][index_b[i]];
					}
					bdata[j] = tdata[tensor_index_to_offset(t->ndim, t->dim, index_t)];
				}
				break;
			}
			case CT_DOUBLE_COMPLEX:
			{
				const dcomplex* tdata = t->data;
				dcomplex*       bdata = b->data;
				for (long j = 0; j < nelem; j++, next_tensor_index(b->ndim, b->dim, index_b))
				{
					for (int i = 0; i < b->ndim; i++)
					{
						index_t[i] = index_map[i][index_b[i]];
					}
					bdata[j] = tdata[tensor_index_to_offset(t->ndim, t->dim, index_t)];
				}
				break;
			}
			default:
			{
				// unknown data type
				assert(false);
			}
		}
		ct_free(index_t);
		ct_free(index_b);

		for (int i = 0; i < s->ndim; i++)
		{
			ct_free(index_map[i]);
		}
		ct_free(index_map);

		ct_free(index_block);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Generalized transpose of a tensor 't' such that
/// the i-th axis in the output tensor 'r' is the perm[i]-th axis of the input tensor 't'.
///
/// Memory will be allocated for 'r'.
///
void transpose_block_sparse_tensor(const int* restrict perm, const struct block_sparse_tensor* restrict t, struct block_sparse_tensor* restrict r)
{
	r->dtype = t->dtype;
	r->ndim = t->ndim;

	if (t->ndim == 0)  // special case
	{
		r->dim_logical = NULL;
		r->dim_blocks  = NULL;

		r->axis_dir = NULL;

		r->qnums_logical = NULL;
		r->qnums_blocks  = NULL;

		// allocate memory for a single block
		r->blocks = ct_calloc(1, sizeof(struct dense_tensor*));
		r->blocks[0] = ct_calloc(1, sizeof(struct dense_tensor));
		allocate_dense_tensor(r->dtype, 0, NULL, r->blocks[0]);
		// copy single number
		memcpy(r->blocks[0]->data, t->blocks[0]->data, sizeof_numeric_type(t->blocks[0]->dtype));

		return;
	}

	// ensure that 'perm' is a valid permutation
	#ifndef NDEBUG
	int* ax_list = ct_calloc(t->ndim, sizeof(int));
	for (int i = 0; i < t->ndim; i++)
	{
		assert(0 <= perm[i] && perm[i] < t->ndim);
		ax_list[perm[i]] = 1;
	}
	for (int i = 0; i < t->ndim; i++)
	{
		assert(ax_list[i] == 1);
	}
	ct_free(ax_list);
	#endif

	// dimensions
	r->dim_logical = ct_malloc(t->ndim * sizeof(long));
	r->dim_blocks  = ct_malloc(t->ndim * sizeof(long));
	r->axis_dir    = ct_malloc(t->ndim * sizeof(enum tensor_axis_direction));
	for (int i = 0; i < t->ndim; i++)
	{
		r->dim_logical[i] = t->dim_logical[perm[i]];
		r->dim_blocks [i] = t->dim_blocks [perm[i]];
		r->axis_dir   [i] = t->axis_dir   [perm[i]];
	}

	// logical quantum numbers
	r->qnums_logical = ct_calloc(t->ndim, sizeof(qnumber*));
	for (int i = 0; i < t->ndim; i++)
	{
		r->qnums_logical[i] = ct_malloc(r->dim_logical[i] * sizeof(qnumber));
		memcpy(r->qnums_logical[i], t->qnums_logical[perm[i]], r->dim_logical[i] * sizeof(qnumber));
	}

	// block quantum numbers
	r->qnums_blocks = ct_calloc(t->ndim, sizeof(qnumber*));
	for (int i = 0; i < t->ndim; i++)
	{
		r->qnums_blocks[i] = ct_malloc(r->dim_blocks[i] * sizeof(qnumber));
		memcpy(r->qnums_blocks[i], t->qnums_blocks[perm[i]], r->dim_blocks[i] * sizeof(qnumber));
	}

	// dense tensor blocks
	const long nblocks = integer_product(t->dim_blocks, t->ndim);
	r->blocks = ct_calloc(nblocks, sizeof(struct dense_tensor*));
	#pragma omp parallel for schedule(dynamic)
	for (long k = 0; k < nblocks; k++)
	{
		const struct dense_tensor* bt = t->blocks[k];
		if (bt == NULL) {
			continue;
		}
		assert(bt->ndim == t->ndim);
		assert(bt->dtype == t->dtype);

		long* index_block_t = ct_malloc(t->ndim * sizeof(long));
		offset_to_tensor_index(t->ndim, t->dim_blocks, k, index_block_t);

		// corresponding block index in 'r'
		long* index_block_r = ct_malloc(r->ndim * sizeof(long));
		for (int i = 0; i < t->ndim; i++) {
			index_block_r[i] = index_block_t[perm[i]];
		}
		long j = tensor_index_to_offset(r->ndim, r->dim_blocks, index_block_r);

		// transpose dense tensor block
		r->blocks[j] = ct_calloc(1, sizeof(struct dense_tensor));
		transpose_dense_tensor(perm, bt, r->blocks[j]);

		ct_free(index_block_r);
		ct_free(index_block_t);	
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Generalized conjugate transpose of a tensor 't' such that
/// the i-th axis in the output tensor 'r' is the perm[i]-th axis of the input tensor 't'.
///
/// Memory will be allocated for 'r'.
///
void conjugate_transpose_block_sparse_tensor(const int* restrict perm, const struct block_sparse_tensor* restrict t, struct block_sparse_tensor* restrict r)
{
	transpose_block_sparse_tensor(perm, t, r);
	conjugate_block_sparse_tensor(r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Flatten the two neighboring axes (tensor legs) 'i_ax' and 'i_ax + 1' into a single axis.
///
/// Memory will be allocated for 'r'.
///
/// Note: this operation changes the internal dense block structure.
///
void block_sparse_tensor_flatten_axes(const struct block_sparse_tensor* restrict t, const int i_ax, const enum tensor_axis_direction new_axis_dir, struct block_sparse_tensor* restrict r)
{
	assert(0 <= i_ax && i_ax + 1 < t->ndim);

	// construct new block-sparse tensor 'r'
	{
		long* r_dim_logical = ct_malloc((t->ndim - 1) * sizeof(long));
		enum tensor_axis_direction* r_axis_dir = ct_malloc((t->ndim - 1) * sizeof(enum tensor_axis_direction));
		for (int i = 0; i < i_ax; i++)
		{
			r_dim_logical[i] = t->dim_logical[i];
			r_axis_dir   [i] = t->axis_dir   [i];
		}
		r_dim_logical[i_ax] = t->dim_logical[i_ax] * t->dim_logical[i_ax + 1];
		r_axis_dir[i_ax]    = new_axis_dir;
		for (int i = i_ax + 2; i < t->ndim; i++)
		{
			r_dim_logical[i - 1] = t->dim_logical[i];
			r_axis_dir   [i - 1] = t->axis_dir   [i];
		}
		// logical quantum numbers
		qnumber* r_qnums_ax_flat = ct_malloc(r_dim_logical[i_ax] * sizeof(qnumber));
		for (long j = 0; j < t->dim_logical[i_ax]; j++)
		{
			for (long k = 0; k < t->dim_logical[i_ax + 1]; k++)
			{
				r_qnums_ax_flat[j*t->dim_logical[i_ax + 1] + k] =
					new_axis_dir * (t->axis_dir[i_ax    ] * t->qnums_logical[i_ax    ][j] +
					                t->axis_dir[i_ax + 1] * t->qnums_logical[i_ax + 1][k]);
			}
		}
		qnumber** r_qnums_logical = ct_malloc((t->ndim - 1) * sizeof(qnumber*));
		for (int i = 0; i < i_ax; i++)
		{
			// simply copy the pointer
			r_qnums_logical[i] = t->qnums_logical[i];
		}
		r_qnums_logical[i_ax] = r_qnums_ax_flat;
		for (int i = i_ax + 2; i < t->ndim; i++)
		{
			// simply copy the pointer
			r_qnums_logical[i - 1] = t->qnums_logical[i];
		}

		// allocate new block-sparse tensor 'r'
		allocate_block_sparse_tensor(t->dtype, t->ndim - 1, r_dim_logical, r_axis_dir, (const qnumber**)r_qnums_logical, r);

		ct_free(r_qnums_logical);
		ct_free(r_qnums_ax_flat);
		ct_free(r_axis_dir);
		ct_free(r_dim_logical);
	}

	const size_t dtype_size = sizeof_numeric_type(t->dtype);

	// for each block with matching quantum numbers...
	const long nblocks = integer_product(t->dim_blocks, t->ndim);
	#pragma omp parallel for schedule(dynamic)
	for (long k = 0; k < nblocks; k++)
	{
		const struct dense_tensor* bt = t->blocks[k];
		if (bt == NULL) {
			continue;
		}
		assert(bt->ndim == t->ndim);
		assert(bt->dtype == t->dtype);

		long* index_block_t = ct_malloc(t->ndim * sizeof(long));
		offset_to_tensor_index(t->ndim, t->dim_blocks, k, index_block_t);

		const qnumber qnums_i_ax[2] = {
			t->qnums_blocks[i_ax    ][index_block_t[i_ax    ]],
			t->qnums_blocks[i_ax + 1][index_block_t[i_ax + 1]] };

		const qnumber qnum_flat =
			new_axis_dir * (t->axis_dir[i_ax    ] * qnums_i_ax[0] +
			                t->axis_dir[i_ax + 1] * qnums_i_ax[1]);

		// corresponding block index in 'r'
		long* index_block_r = ct_malloc(r->ndim * sizeof(long));
		for (int i = 0; i < i_ax; i++) {
			index_block_r[i] = index_block_t[i];
		}
		index_block_r[i_ax] = -1;
		for (long j = 0; j < r->dim_blocks[i_ax]; j++)
		{
			if (r->qnums_blocks[i_ax][j] == qnum_flat) {
				index_block_r[i_ax] = j;
				break;
			}
		}
		assert(index_block_r[i_ax] != -1);
		for (int i = i_ax + 2; i < t->ndim; i++)
		{
			index_block_r[i - 1] = index_block_t[i];
		}
		struct dense_tensor* br = r->blocks[tensor_index_to_offset(r->ndim, r->dim_blocks, index_block_r)];
		assert(br != NULL);
		assert(br->ndim == r->ndim);
		assert(br->dtype == r->dtype);

		// construct index map for dense block entries along to-be flattened dimensions
		long* index_map_block = ct_malloc(bt->dim[i_ax] * bt->dim[i_ax + 1] * sizeof(long));
		{
			// fan-out dense to logical indices for original axes 'i_ax' and 'i_ax + 1'
			long* index_map_fanout[2];
			for (int i = 0; i < 2; i++)
			{
				index_map_fanout[i] = ct_calloc(bt->dim[i_ax + i], sizeof(long));
				long c = 0;
				for (long j = 0; j < t->dim_logical[i_ax + i]; j++)
				{
					if (t->qnums_logical[i_ax + i][j] == qnums_i_ax[i])
					{
						index_map_fanout[i][c] = j;
						c++;
					}
				}
				assert(c == bt->dim[i_ax + i]);
			}
			// fan-in logical to block indices for flattened axis
			long* index_map_fanin = ct_calloc(r->dim_logical[i_ax], sizeof(long));
			long c = 0;
			for (long j = 0; j < r->dim_logical[i_ax]; j++)
			{
				if (r->qnums_logical[i_ax][j] == qnum_flat) {
					index_map_fanin[j] = c;
					c++;
				}
			}
			for (long j0 = 0; j0 < bt->dim[i_ax]; j0++)
			{
				for (long j1 = 0; j1 < bt->dim[i_ax + 1]; j1++)
				{
					index_map_block[j0*bt->dim[i_ax + 1] + j1] = index_map_fanin[index_map_fanout[0][j0] * t->dim_logical[i_ax + 1] + index_map_fanout[1][j1]];
				}
			}
			ct_free(index_map_fanin);
			for (int i = 0; i < 2; i++) {
				ct_free(index_map_fanout[i]);
			}
		}

		// copy block tensor entries
		const long nslices = integer_product(bt->dim, i_ax + 2);
		long* index_slice_bt = ct_calloc(i_ax + 2, sizeof(long));
		long* index_slice_br = ct_calloc(i_ax + 1, sizeof(long));
		const long stride = integer_product(bt->dim + (i_ax + 2), bt->ndim - (i_ax + 2));
		assert(stride == integer_product(br->dim + (i_ax + 1), br->ndim - (i_ax + 1)));
		for (long j = 0; j < nslices; j++, next_tensor_index(i_ax + 2, bt->dim, index_slice_bt))
		{
			for (int i = 0; i < i_ax; i++) {
				index_slice_br[i] = index_slice_bt[i];
			}
			index_slice_br[i_ax] = index_map_block[index_slice_bt[i_ax]*bt->dim[i_ax + 1] + index_slice_bt[i_ax + 1]];
			const long l = tensor_index_to_offset(i_ax + 1, br->dim, index_slice_br);
			// copy slice of entries
			// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
			memcpy((int8_t*)br->data + (l*stride) * dtype_size, (int8_t*)bt->data + (j*stride) * dtype_size, stride * dtype_size);
		}
		ct_free(index_slice_br);
		ct_free(index_slice_bt);

		ct_free(index_map_block);
		ct_free(index_block_t);
		ct_free(index_block_r);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Split the axis (tensor leg) 'i_ax' into two neighboring axes, using the provided quantum numbers.
///
/// Memory will be allocated for 'r'.
///
/// Note: this operation changes the internal dense block structure.
///
void block_sparse_tensor_split_axis(const struct block_sparse_tensor* restrict t, const int i_ax, const long new_dim_logical[2], const enum tensor_axis_direction new_axis_dir[2], const qnumber* new_qnums_logical[2], struct block_sparse_tensor* restrict r)
{
	assert(0 <= i_ax && i_ax < t->ndim);
	assert(new_dim_logical[0] * new_dim_logical[1] == t->dim_logical[i_ax]);
	// consistency check of provided quantum numbers
	for (long j = 0; j < new_dim_logical[0]; j++) {
		for (long k = 0; k < new_dim_logical[1]; k++) {
			assert(t->axis_dir[i_ax] * t->qnums_logical[i_ax][j*new_dim_logical[1] + k]
				== new_axis_dir[0] * new_qnums_logical[0][j] + new_axis_dir[1] * new_qnums_logical[1][k]);
		}
	}

	// construct new block-sparse tensor 'r'
	{
		long* r_dim_logical = ct_malloc((t->ndim + 1) * sizeof(long));
		enum tensor_axis_direction* r_axis_dir = ct_malloc((t->ndim + 1) * sizeof(enum tensor_axis_direction));
		for (int i = 0; i < i_ax; i++)
		{
			r_dim_logical[i] = t->dim_logical[i];
			r_axis_dir   [i] = t->axis_dir   [i];
		}
		r_dim_logical[i_ax    ] = new_dim_logical[0];
		r_dim_logical[i_ax + 1] = new_dim_logical[1];
		r_axis_dir[i_ax    ] = new_axis_dir[0];
		r_axis_dir[i_ax + 1] = new_axis_dir[1];
		for (int i = i_ax + 1; i < t->ndim; i++)
		{
			r_dim_logical[i + 1] = t->dim_logical[i];
			r_axis_dir   [i + 1] = t->axis_dir   [i];
		}
		// logical quantum numbers
		const qnumber** r_qnums_logical = ct_malloc((t->ndim + 1) * sizeof(qnumber*));
		for (int i = 0; i < i_ax; i++)
		{
			// simply copy the pointer
			r_qnums_logical[i] = t->qnums_logical[i];
		}
		r_qnums_logical[i_ax    ] = new_qnums_logical[0];
		r_qnums_logical[i_ax + 1] = new_qnums_logical[1];
		for (int i = i_ax + 1; i < t->ndim; i++)
		{
			// simply copy the pointer
			r_qnums_logical[i + 1] = t->qnums_logical[i];
		}

		// allocate new block-sparse tensor 'r'
		allocate_block_sparse_tensor(t->dtype, t->ndim + 1, r_dim_logical, r_axis_dir, r_qnums_logical, r);

		ct_free(r_qnums_logical);
		ct_free(r_axis_dir);
		ct_free(r_dim_logical);
	}

	const size_t dtype_size = sizeof_numeric_type(t->dtype);

	// for each block with matching quantum numbers...
	const long nblocks = integer_product(r->dim_blocks, r->ndim);
	#pragma omp parallel for schedule(dynamic)
	for (long k = 0; k < nblocks; k++)
	{
		const struct dense_tensor* br = r->blocks[k];
		if (br == NULL) {
			continue;
		}
		assert(br->ndim == r->ndim);
		assert(br->dtype == r->dtype);

		long* index_block_r = ct_malloc(r->ndim * sizeof(long));
		offset_to_tensor_index(r->ndim, r->dim_blocks, k, index_block_r);

		const qnumber qnums_i_ax[2] = {
			r->qnums_blocks[i_ax    ][index_block_r[i_ax    ]],
			r->qnums_blocks[i_ax + 1][index_block_r[i_ax + 1]] };

		const qnumber qnum_flat =
			t->axis_dir[i_ax] * (r->axis_dir[i_ax    ] * qnums_i_ax[0] +
			                     r->axis_dir[i_ax + 1] * qnums_i_ax[1]);

		// corresponding block index in 't'
		long* index_block_t = ct_malloc(t->ndim * sizeof(long));
		for (int i = 0; i < i_ax; i++) {
			index_block_t[i] = index_block_r[i];
		}
		index_block_t[i_ax] = -1;
		for (long j = 0; j < t->dim_blocks[i_ax]; j++)
		{
			if (t->qnums_blocks[i_ax][j] == qnum_flat) {
				index_block_t[i_ax] = j;
				break;
			}
		}
		assert(index_block_t[i_ax] != -1);
		for (int i = i_ax + 1; i < t->ndim; i++)
		{
			index_block_t[i] = index_block_r[i + 1];
		}
		const struct dense_tensor* bt = t->blocks[tensor_index_to_offset(t->ndim, t->dim_blocks, index_block_t)];
		assert(bt != NULL);
		assert(bt->ndim == t->ndim);
		assert(bt->dtype == t->dtype);

		// construct index map for dense block entries along to-be split dimensions
		long* index_map_block = ct_malloc(br->dim[i_ax] * br->dim[i_ax + 1] * sizeof(long));
		{
			// fan-out dense to logical indices for new axes 'i_ax' and 'i_ax + 1'
			long* index_map_fanout[2];
			for (int i = 0; i < 2; i++)
			{
				index_map_fanout[i] = ct_calloc(br->dim[i_ax + i], sizeof(long));
				long c = 0;
				for (long j = 0; j < r->dim_logical[i_ax + i]; j++)
				{
					if (r->qnums_logical[i_ax + i][j] == qnums_i_ax[i])
					{
						index_map_fanout[i][c] = j;
						c++;
					}
				}
				assert(c == br->dim[i_ax + i]);
			}
			// fan-in logical to block indices for original axis
			long* index_map_fanin = ct_calloc(t->dim_logical[i_ax], sizeof(long));
			long c = 0;
			for (long j = 0; j < t->dim_logical[i_ax]; j++)
			{
				if (t->qnums_logical[i_ax][j] == qnum_flat) {
					index_map_fanin[j] = c;
					c++;
				}
			}
			for (long j0 = 0; j0 < br->dim[i_ax]; j0++)
			{
				for (long j1 = 0; j1 < br->dim[i_ax + 1]; j1++)
				{
					index_map_block[j0*br->dim[i_ax + 1] + j1] = index_map_fanin[index_map_fanout[0][j0] * r->dim_logical[i_ax + 1] + index_map_fanout[1][j1]];
				}
			}
			ct_free(index_map_fanin);
			for (int i = 0; i < 2; i++) {
				ct_free(index_map_fanout[i]);
			}
		}

		// copy block tensor entries
		const long nslices = integer_product(br->dim, i_ax + 2);
		long* index_slice_bt = ct_calloc(i_ax + 1, sizeof(long));
		long* index_slice_br = ct_calloc(i_ax + 2, sizeof(long));
		const long stride = integer_product(br->dim + (i_ax + 2), br->ndim - (i_ax + 2));
		assert(stride == integer_product(bt->dim + (i_ax + 1), bt->ndim - (i_ax + 1)));
		for (long j = 0; j < nslices; j++, next_tensor_index(i_ax + 2, br->dim, index_slice_br))
		{
			for (int i = 0; i < i_ax; i++) {
				index_slice_bt[i] = index_slice_br[i];
			}
			index_slice_bt[i_ax] = index_map_block[index_slice_br[i_ax]*br->dim[i_ax + 1] + index_slice_br[i_ax + 1]];
			const long l = tensor_index_to_offset(i_ax + 1, bt->dim, index_slice_bt);
			// copy slice of entries
			// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
			memcpy((int8_t*)br->data + (j*stride) * dtype_size, (int8_t*)bt->data + (l*stride) * dtype_size, stride * dtype_size);
		}
		ct_free(index_slice_bt);
		ct_free(index_slice_br);

		ct_free(index_map_block);
		ct_free(index_block_t);
		ct_free(index_block_r);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Slice an axis of the tensor by selecting logical indices 'ind' along this axis.
///
/// Indices 'ind' can be duplicate and need not be sorted.
/// Memory will be allocated for 'r'.
///
void block_sparse_tensor_slice(const struct block_sparse_tensor* restrict t, const int i_ax, const long* ind, const long nind, struct block_sparse_tensor* restrict r)
{
	assert(0 <= i_ax && i_ax < t->ndim);
	assert(nind > 0);

	// construct new block-sparse tensor 'r'
	{
		long* r_dim_logical = ct_malloc(t->ndim * sizeof(long));
		memcpy(r_dim_logical, t->dim_logical, t->ndim * sizeof(long));
		r_dim_logical[i_ax] = nind;
		// logical quantum numbers
		qnumber* r_qnums_logical_i_ax = ct_malloc(nind * sizeof(qnumber));
		for (long j = 0; j < nind; j++) {
			assert(0 <= ind[j] && ind[j] < t->dim_logical[i_ax]);
			r_qnums_logical_i_ax[j] = t->qnums_logical[i_ax][ind[j]];
		}
		const qnumber** r_qnums_logical = ct_malloc(t->ndim * sizeof(qnumber*));
		for (int i = 0; i < t->ndim; i++)
		{
			// simply copy the pointer for i != i_ax
			r_qnums_logical[i] = (i == i_ax ? r_qnums_logical_i_ax : t->qnums_logical[i]);
		}

		// allocate new block-sparse tensor 'r'
		allocate_block_sparse_tensor(t->dtype, t->ndim, r_dim_logical, t->axis_dir, r_qnums_logical, r);

		ct_free(r_qnums_logical);
		ct_free(r_qnums_logical_i_ax);
		ct_free(r_dim_logical);
	}

	// for each block with matching quantum numbers...
	const long nblocks = integer_product(r->dim_blocks, r->ndim);
	#pragma omp parallel for schedule(dynamic)
	for (long k = 0; k < nblocks; k++)
	{
		struct dense_tensor* br = r->blocks[k];
		if (br == NULL) {
			continue;
		}
		assert(br->ndim == r->ndim);
		assert(br->dtype == r->dtype);

		long* index_block_r = ct_malloc(r->ndim * sizeof(long));
		offset_to_tensor_index(r->ndim, r->dim_blocks, k, index_block_r);

		const qnumber qnum = r->qnums_blocks[i_ax][index_block_r[i_ax]];

		// find corresponding block index in 't' tensor
		long* index_block_t = ct_malloc(t->ndim * sizeof(long));
		memcpy(index_block_t, index_block_r, t->ndim * sizeof(long));
		index_block_t[i_ax] = -1;
		for (long j = 0; j < t->dim_blocks[i_ax]; j++)
		{
			if (t->qnums_blocks[i_ax][j] == qnum) {
				index_block_t[i_ax] = j;
				break;
			}
		}
		assert(index_block_t[i_ax] != -1);

		const struct dense_tensor* bt = t->blocks[tensor_index_to_offset(t->ndim, t->dim_blocks, index_block_t)];
		assert(bt != NULL);
		assert(bt->ndim == t->ndim);
		assert(bt->dtype == t->dtype);

		// fan-out dense to logical indices for axis 'i_ax' in 'r'
		long* index_map_fanout_r = ct_calloc(br->dim[i_ax], sizeof(long));
		long c = 0;
		for (long j = 0; j < r->dim_logical[i_ax]; j++)
		{
			if (r->qnums_logical[i_ax][j] == qnum)
			{
				index_map_fanout_r[c] = j;
				c++;
			}
		}
		assert(c == br->dim[i_ax]);

		// fan-in logical to block indices for axis i_ax in 't'
		long* index_map_fanin_t = ct_calloc(t->dim_logical[i_ax], sizeof(long));
		c = 0;
		for (long j = 0; j < t->dim_logical[i_ax]; j++)
		{
			if (t->qnums_logical[i_ax][j] == qnum) {
				index_map_fanin_t[j] = c;
				c++;
			}
		}

		// indices for slicing of current block
		long* ind_block = ct_malloc(br->dim[i_ax] * sizeof(long));
		for (long j = 0; j < br->dim[i_ax]; j++)
		{
			assert(t->qnums_logical[i_ax][ind[index_map_fanout_r[j]]] == qnum);
			ind_block[j] = index_map_fanin_t[ind[index_map_fanout_r[j]]];
		}

		// slice block
		dense_tensor_slice_fill(bt, i_ax, ind_block, br->dim[i_ax], br);

		ct_free(ind_block);
		ct_free(index_map_fanin_t);
		ct_free(index_map_fanout_r);
		ct_free(index_block_r);
		ct_free(index_block_t);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the "cyclic" partial trace, by tracing out the 'ndim_trace' leading with the 'ndim_trace' trailing axes.
///
void block_sparse_tensor_cyclic_partial_trace(const struct block_sparse_tensor* restrict t, const int ndim_trace, struct block_sparse_tensor* restrict r)
{
	assert(ndim_trace >= 0);
	assert(t->ndim >= 2 * ndim_trace);
	for (int i = 0; i < ndim_trace; i++)
	{
		assert(t->dim_logical[i] == t->dim_logical[t->ndim - ndim_trace + i]);
		assert(t->axis_dir[i] == -t->axis_dir[t->ndim - ndim_trace + i]);
		assert(qnumber_all_equal(t->dim_logical[i], t->qnums_logical[i], t->qnums_logical[t->ndim - ndim_trace + i]));
	}

	if (ndim_trace == 0) {
		copy_block_sparse_tensor(t, r);
		return;
	}

	// construct new block-sparse tensor 'r'
	allocate_block_sparse_tensor(t->dtype, t->ndim - 2 * ndim_trace, t->dim_logical + ndim_trace, t->axis_dir + ndim_trace, (const qnumber**)(t->qnums_logical + ndim_trace), r);

	// for each block with matching quantum numbers...
	const long nblocks_r = integer_product(r->dim_blocks, r->ndim);
	const long nblocks_p = integer_product(t->dim_blocks, ndim_trace);
	#pragma omp parallel for schedule(dynamic)
	for (long k = 0; k < nblocks_r; k++)
	{
		struct dense_tensor* br = r->blocks[k];
		if (br == NULL) {
			continue;
		}

		long* index_block_r  = ct_malloc(r->ndim * sizeof(long));
		offset_to_tensor_index(r->ndim, r->dim_blocks, k, index_block_r);

		// require zero initialization
		long* index_block_t  = ct_calloc(t->ndim, sizeof(long));
		memcpy(index_block_t + ndim_trace, index_block_r, r->ndim * sizeof(long));

		for (long j = 0; j < nblocks_p; j++, next_tensor_index(ndim_trace, t->dim_blocks, index_block_t))
		{
			// duplicate leading 'ndim_trace' indices at the end
			memcpy(index_block_t + ndim_trace + r->ndim, index_block_t, ndim_trace * sizeof(long));

			const struct dense_tensor* bt = t->blocks[tensor_index_to_offset(t->ndim, t->dim_blocks, index_block_t)];
			assert(bt != NULL);

			dense_tensor_cyclic_partial_trace_update(bt, ndim_trace, br);
		}

		ct_free(index_block_r);
		ct_free(index_block_t);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Scalar multiply and add two tensors: t = alpha*s + t; dimensions, data types and quantum numbers of s and t must agree,
/// and alpha must be of the same data type as tensor entries.
///
void block_sparse_tensor_scalar_multiply_add(const void* alpha, const struct block_sparse_tensor* restrict s, struct block_sparse_tensor* restrict t)
{
	assert(s->dtype == t->dtype);
	assert(s->ndim  == t->ndim);
	for (int i = 0; i < t->ndim; i++)
	{
		assert(s->dim_logical[i] == t->dim_logical[i]);
		assert(s->axis_dir[i]    == t->axis_dir[i]);
		assert(qnumber_all_equal(t->dim_logical[i], s->qnums_logical[i], t->qnums_logical[i]));
	}

	// for each block with matching quantum numbers...
	const long nblocks = integer_product(t->dim_blocks, t->ndim);
	#pragma omp parallel for schedule(dynamic)
	for (long k = 0; k < nblocks; k++)
	{
		const struct dense_tensor* bs = s->blocks[k];
		struct dense_tensor* bt       = t->blocks[k];
		if (bt == NULL) {
			continue;
		}
		assert(bs != NULL);

		dense_tensor_scalar_multiply_add(alpha, bs, bt);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Pointwise multiply the entries of 's' along its leading or trailing axis with the vector 't'.
/// The output tensor 'r' has the same data type and dimension as 's'.
///
/// Memory will be allocated for 'r'.
///
void block_sparse_tensor_multiply_pointwise_vector(const struct block_sparse_tensor* restrict s, const struct dense_tensor* restrict t, const enum tensor_axis_range axrange, struct block_sparse_tensor* restrict r)
{
	assert(t->ndim == 1);

	const int i_ax = (axrange == TENSOR_AXIS_RANGE_LEADING ? 0 : s->ndim - 1);
	assert(t->dim[0] == s->dim_logical[i_ax]);

	// allocate new block-sparse tensor 'r' with same layout as 's'
	allocate_block_sparse_tensor(s->dtype, s->ndim, s->dim_logical, s->axis_dir, (const qnumber**)s->qnums_logical, r);

	// for each block with matching quantum numbers...
	const long nblocks = integer_product(r->dim_blocks, r->ndim);
	#pragma omp parallel for schedule(dynamic)
	for (long k = 0; k < nblocks; k++)
	{
		struct dense_tensor* br = r->blocks[k];
		if (br == NULL) {
			continue;
		}

		// corresponding block in 's'
		const struct dense_tensor* bs = s->blocks[k];
		assert(bs != NULL);

		long* index_block_r = ct_malloc(r->ndim * sizeof(long));
		offset_to_tensor_index(r->ndim, r->dim_blocks, k, index_block_r);

		// fan-out dense to logical indices
		long* index_map = ct_calloc(br->dim[i_ax], sizeof(long));
		long c = 0;
		for (long j = 0; j < r->dim_logical[i_ax]; j++)
		{
			if (r->qnums_logical[i_ax][j] == r->qnums_blocks[i_ax][index_block_r[i_ax]])
			{
				index_map[c] = j;
				c++;
			}
		}
		assert(c == br->dim[i_ax]);

		struct dense_tensor t_slice;
		dense_tensor_slice(t, 0, index_map, br->dim[i_ax], &t_slice);

		dense_tensor_multiply_pointwise_fill(bs, &t_slice, axrange, br);

		delete_dense_tensor(&t_slice);
		ct_free(index_map);
		ct_free(index_block_r);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Multiply the 'i_ax' axis of 's' with the leading or trailing axis of 't', preserving the overall dimension ordering of 's'.
///
void block_sparse_tensor_multiply_axis(const struct block_sparse_tensor* restrict s, const int i_ax, const struct block_sparse_tensor* restrict t, const enum tensor_axis_range axrange_t, struct block_sparse_tensor* restrict r)
{
	const int i_mt = (axrange_t == TENSOR_AXIS_RANGE_LEADING ? 0 : t->ndim - 1);
	const int offset_t = (axrange_t == TENSOR_AXIS_RANGE_LEADING ? 1 : 0);

	// data types must agree
	assert(s->dtype == t->dtype);
	// 't' must have a degree of at least 1
	assert(t->ndim >= 1);
	// 'i_ax' must be a valid axis index for 's'
	assert(0 <= i_ax && i_ax < s->ndim);
	// to-be contracted dimensions must match, and axis directions must be reversed
	assert(s->dim_logical[i_ax] ==  t->dim_logical[i_mt]);
	assert(s->dim_blocks [i_ax] ==  t->dim_blocks [i_mt]);
	assert(s->axis_dir   [i_ax] == -t->axis_dir   [i_mt]);
	// quantum numbers must match entrywise
	assert(qnumber_all_equal(s->dim_logical[i_ax], s->qnums_logical[i_ax], t->qnums_logical[i_mt]));

	// allocate new block-sparse tensor 'r'
	{
		const int ndimr = s->ndim + t->ndim - 2;

		// logical dimensions of new tensor 'r'
		long* r_dim_logical = ct_malloc(ndimr * sizeof(long));
		memcpy( r_dim_logical, s->dim_logical, i_ax * sizeof(long));
		memcpy(&r_dim_logical[i_ax], &t->dim_logical[offset_t], (t->ndim - 1) * sizeof(long));
		memcpy(&r_dim_logical[i_ax + t->ndim - 1], &s->dim_logical[i_ax + 1], (s->ndim - i_ax - 1) * sizeof(long));
		// axis directions of new tensor 'r'
		enum tensor_axis_direction* r_axis_dir = ct_malloc(ndimr * sizeof(enum tensor_axis_direction));
		memcpy( r_axis_dir, s->axis_dir, i_ax * sizeof(enum tensor_axis_direction));
		memcpy(&r_axis_dir[i_ax], &t->axis_dir[offset_t], (t->ndim - 1) * sizeof(enum tensor_axis_direction));
		memcpy(&r_axis_dir[i_ax + t->ndim - 1], &s->axis_dir[i_ax + 1], (s->ndim - i_ax - 1) * sizeof(enum tensor_axis_direction));
		// logical quantum numbers along each dimension
		const qnumber** r_qnums_logical = ct_malloc(ndimr * sizeof(qnumber*));
		// simply copy the pointers
		memcpy( r_qnums_logical, s->qnums_logical, i_ax * sizeof(qnumber*));
		memcpy(&r_qnums_logical[i_ax], &t->qnums_logical[offset_t], (t->ndim - 1) * sizeof(qnumber*));
		memcpy(&r_qnums_logical[i_ax + t->ndim - 1], &s->qnums_logical[i_ax + 1], (s->ndim - i_ax - 1) * sizeof(qnumber*));
		// create new tensor 'r'
		allocate_block_sparse_tensor(s->dtype, ndimr, r_dim_logical, r_axis_dir, r_qnums_logical, r);
		ct_free(r_qnums_logical);
		ct_free(r_axis_dir);
		ct_free(r_dim_logical);
	}

	// for each dense block of 'r'...
	const long nblocks = integer_product(r->dim_blocks, r->ndim);
	#pragma omp parallel for schedule(dynamic)
	for (long k = 0; k < nblocks; k++)
	{
		struct dense_tensor* br = r->blocks[k];
		if (br == NULL) {
			continue;
		}
		assert(br->ndim == r->ndim);

		long* index_block_r = ct_malloc(r->ndim * sizeof(long));
		offset_to_tensor_index(r->ndim, r->dim_blocks, k, index_block_r);

		long* index_block_s = ct_malloc(s->ndim * sizeof(long));
		long* index_block_t = ct_malloc(t->ndim * sizeof(long));
		memcpy( index_block_s, index_block_r, i_ax * sizeof(long));
		memcpy(&index_block_t[offset_t], &index_block_r[i_ax], (t->ndim - 1) * sizeof(long));
		memcpy(&index_block_s[i_ax + 1], &index_block_r[i_ax + t->ndim - 1], (s->ndim - i_ax - 1) * sizeof(long));

		// for each quantum number of the to-be contracted axis...
		for (long m = 0; m < s->dim_blocks[i_ax]; m++)
		{
			index_block_s[i_ax] = m;
			index_block_t[i_mt] = m;

			// probe whether quantum numbers in 's' sum to zero
			qnumber qsum = 0;
			for (int i = 0; i < s->ndim; i++)
			{
				qsum += s->axis_dir[i] * s->qnums_blocks[i][index_block_s[i]];
			}
			if (qsum != 0) {
				continue;
			}

			// quantum numbers in 't' must now also sum to zero
			#ifndef NDEBUG
			qsum = 0;
			for (int i = 0; i < t->ndim; i++)
			{
				qsum += t->axis_dir[i] * t->qnums_blocks[i][index_block_t[i]];
			}
			assert(qsum == 0);
			#endif

			const struct dense_tensor* bs = s->blocks[tensor_index_to_offset(s->ndim, s->dim_blocks, index_block_s)];
			const struct dense_tensor* bt = t->blocks[tensor_index_to_offset(t->ndim, t->dim_blocks, index_block_t)];
			assert(bs != NULL);
			assert(bt != NULL);

			// actually multiply dense tensor blocks and store result in 'br';
			// there is only a single possible block index combination which contributes to this 'br'
			dense_tensor_multiply_axis_update(numeric_one(s->dtype), bs, i_ax, bt, axrange_t, numeric_zero(s->dtype), br);
		}

		ct_free(index_block_r);
		ct_free(index_block_t);
		ct_free(index_block_s);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Multiply (leading or trailing) 'ndim_mult' axes in 's' by 'ndim_mult' axes in 't', and store result in 'r'.
/// Whether to use leading or trailing axes is specified by axis range.
///
/// Memory will be allocated for 'r'. Operation requires that the quantum numbers of the to-be contracted axes match,
/// and that the axis directions are reversed between the tensors.
///
void block_sparse_tensor_dot(const struct block_sparse_tensor* restrict s, const enum tensor_axis_range axrange_s, const struct block_sparse_tensor* restrict t, const enum tensor_axis_range axrange_t, const int ndim_mult, struct block_sparse_tensor* restrict r)
{
	assert(s->dtype == t->dtype);

	const int shift_s = (axrange_s == TENSOR_AXIS_RANGE_LEADING ? 0 : s->ndim - ndim_mult);
	const int shift_t = (axrange_t == TENSOR_AXIS_RANGE_LEADING ? 0 : t->ndim - ndim_mult);

	// dimension and quantum number compatibility checks
	assert(ndim_mult >= 1);
	assert(s->ndim >= ndim_mult && t->ndim >= ndim_mult);
	for (int i = 0; i < ndim_mult; i++)
	{
		assert(s->dim_logical[shift_s + i] ==  t->dim_logical[shift_t + i]);
		assert(s->dim_blocks [shift_s + i] ==  t->dim_blocks [shift_t + i]);
		assert(s->axis_dir   [shift_s + i] == -t->axis_dir   [shift_t + i]);
		// quantum numbers must match entrywise
		assert(qnumber_all_equal(t->dim_logical[shift_t + i], s->qnums_logical[shift_s + i], t->qnums_logical[shift_t + i]));
		assert(qnumber_all_equal(t->dim_blocks [shift_t + i], s->qnums_blocks [shift_s + i], t->qnums_blocks [shift_t + i]));
	}

	const void* one = numeric_one(s->dtype);

	if (s->ndim + t->ndim == 2*ndim_mult)
	{
		// special case: resulting tensor has degree 0

		assert(s->ndim == ndim_mult && t->ndim == ndim_mult);

		allocate_block_sparse_tensor(s->dtype, 0, NULL, NULL, NULL, r);
		assert(r->blocks[0] != NULL);

		// for each quantum number combination of the to-be contracted axes...
		const long ncontract = integer_product(s->dim_blocks, ndim_mult);
		long* index_contract = ct_calloc(ndim_mult, sizeof(long));
		for (long m = 0; m < ncontract; m++, next_tensor_index(ndim_mult, s->dim_blocks, index_contract))
		{
			// probe whether quantum numbers in 's' sum to zero
			qnumber qsum = 0;
			for (int i = 0; i < s->ndim; i++)
			{
				qsum += s->axis_dir[i] * s->qnums_blocks[i][index_contract[i]];
			}
			if (qsum != 0) {
				continue;
			}

			const long ib = tensor_index_to_offset(s->ndim, s->dim_blocks, index_contract);
			const struct dense_tensor* bs = s->blocks[ib];
			const struct dense_tensor* bt = t->blocks[ib];
			assert(bs != NULL);
			assert(bt != NULL);

			// actually multiply dense tensor blocks and add result to block in 'r'
			dense_tensor_dot_update(one, bs, axrange_s, bt, axrange_t, ndim_mult, one, r->blocks[0]);
		}

		ct_free(index_contract);

		return;
	}

	const int offset_s = (axrange_s == TENSOR_AXIS_RANGE_LEADING ? ndim_mult : 0);
	const int offset_t = (axrange_t == TENSOR_AXIS_RANGE_LEADING ? ndim_mult : 0);

	// allocate new block-sparse tensor 'r'
	{
		const int ndimr = s->ndim + t->ndim - 2*ndim_mult;

		// logical dimensions of new tensor 'r'
		long* r_dim_logical = ct_malloc(ndimr * sizeof(long));
		for (int i = 0; i < s->ndim - ndim_mult; i++)
		{
			r_dim_logical[i] = s->dim_logical[offset_s + i];
		}
		for (int i = 0; i < t->ndim - ndim_mult; i++)
		{
			r_dim_logical[(s->ndim - ndim_mult) + i] = t->dim_logical[offset_t + i];
		}
		// axis directions of new tensor 'r'
		enum tensor_axis_direction* r_axis_dir = ct_malloc(ndimr * sizeof(enum tensor_axis_direction));
		for (int i = 0; i < s->ndim - ndim_mult; i++)
		{
			r_axis_dir[i] = s->axis_dir[offset_s + i];
		}
		for (int i = 0; i < t->ndim - ndim_mult; i++)
		{
			r_axis_dir[(s->ndim - ndim_mult) + i] = t->axis_dir[offset_t + i];
		}
		// logical quantum numbers along each dimension
		const qnumber** r_qnums_logical = ct_malloc(ndimr * sizeof(qnumber*));
		for (int i = 0; i < s->ndim - ndim_mult; i++)
		{
			// simply copy the pointer
			r_qnums_logical[i] = s->qnums_logical[offset_s + i];
		}
		for (int i = 0; i < t->ndim - ndim_mult; i++)
		{
			// simply copy the pointer
			r_qnums_logical[(s->ndim - ndim_mult) + i] = t->qnums_logical[offset_t + i];
		}
		// create new tensor 'r'
		allocate_block_sparse_tensor(s->dtype, ndimr, r_dim_logical, r_axis_dir, r_qnums_logical, r);
		ct_free(r_qnums_logical);
		ct_free(r_axis_dir);
		ct_free(r_dim_logical);
	}

	// for each dense block of 'r'...
	const long nblocks = integer_product(r->dim_blocks, r->ndim);
	#pragma omp parallel for schedule(dynamic)
	for (long k = 0; k < nblocks; k++)
	{
		struct dense_tensor* br = r->blocks[k];
		if (br == NULL) {
			continue;
		}
		assert(br->ndim == r->ndim);

		long* index_block_r = ct_malloc(r->ndim * sizeof(long));
		long* index_block_s = ct_malloc(s->ndim * sizeof(long));
		long* index_block_t = ct_malloc(t->ndim * sizeof(long));

		offset_to_tensor_index(r->ndim, r->dim_blocks, k, index_block_r);

		// for each quantum number combination of the to-be contracted axes...
		const long ncontract = integer_product(t->dim_blocks + shift_t, ndim_mult);
		long* index_contract = ct_calloc(ndim_mult, sizeof(long));
		for (long m = 0; m < ncontract; m++, next_tensor_index(ndim_mult, t->dim_blocks + shift_t, index_contract))
		{
			for (int i = 0; i < s->ndim - ndim_mult; i++) {
				index_block_s[offset_s + i] = index_block_r[i];
			}
			for (int i = 0; i < ndim_mult; i++) {
				index_block_s[shift_s + i] = index_contract[i];
			}
			// probe whether quantum numbers in 's' sum to zero
			qnumber qsum = 0;
			for (int i = 0; i < s->ndim; i++)
			{
				qsum += s->axis_dir[i] * s->qnums_blocks[i][index_block_s[i]];
			}
			if (qsum != 0) {
				continue;
			}

			for (int i = 0; i < t->ndim - ndim_mult; i++) {
				index_block_t[offset_t + i] = index_block_r[(s->ndim - ndim_mult) + i];
			}
			for (int i = 0; i < ndim_mult; i++) {
				index_block_t[shift_t + i] = index_contract[i];
			}
			// quantum numbers in 't' must now also sum to zero
			#ifndef NDEBUG
			qsum = 0;
			for (int i = 0; i < t->ndim; i++)
			{
				qsum += t->axis_dir[i] * t->qnums_blocks[i][index_block_t[i]];
			}
			assert(qsum == 0);
			#endif

			const struct dense_tensor* bs = s->blocks[tensor_index_to_offset(s->ndim, s->dim_blocks, index_block_s)];
			const struct dense_tensor* bt = t->blocks[tensor_index_to_offset(t->ndim, t->dim_blocks, index_block_t)];
			assert(bs != NULL);
			assert(bt != NULL);

			// actually multiply dense tensor blocks and add result to 'br'
			dense_tensor_dot_update(one, bs, axrange_s, bt, axrange_t, ndim_mult, one, br);
		}

		ct_free(index_contract);
		ct_free(index_block_t);
		ct_free(index_block_s);
		ct_free(index_block_r);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Concatenate tensors along the specified axis. All other dimensions and their quantum numbers must respectively agree.
///
void block_sparse_tensor_concatenate(const struct block_sparse_tensor* restrict tlist, const int num_tensors, const int i_ax, struct block_sparse_tensor* restrict r)
{
	assert(num_tensors >= 1);

	const int ndim = tlist[0].ndim;
	assert(0 <= i_ax && i_ax < ndim);

	#ifndef NDEBUG
	for (int j = 0; j < num_tensors - 1; j++)
	{
		// data types must match
		assert(tlist[j].dtype == tlist[j + 1].dtype);
		// degrees must match
		assert(tlist[j].ndim == tlist[j + 1].ndim);
		for (int i = 0; i < tlist[j].ndim; i++)
		{
			// axis directions must match
			assert(tlist[j].axis_dir[i] == tlist[j + 1].axis_dir[i]);
			if (i != i_ax) {
				// other dimensions must match
				assert(tlist[j].dim_logical[i] == tlist[j + 1].dim_logical[i]);
				// quantum numbers must match entrywise
				assert(qnumber_all_equal(tlist[j].dim_logical[i], tlist[j].qnums_logical[i], tlist[j + 1].qnums_logical[i]));
			}
		}
	}
	#endif

	// allocate new block-sparse tensor 'r'
	{
		// dimensions of new tensor
		long dim_concat = 0;
		for (int j = 0; j < num_tensors; j++) {
			dim_concat += tlist[j].dim_logical[i_ax];
		}
		long* r_dim_logical = ct_malloc(ndim * sizeof(long));
		for (int i = 0; i < ndim; i++)
		{
			if (i == i_ax) {
				r_dim_logical[i] = dim_concat;
			}
			else {
				r_dim_logical[i] = tlist[0].dim_logical[i];
			}
		}

		// logical quantum numbers along each dimension
		qnumber* r_qnums_logical_i_ax = ct_malloc(dim_concat * sizeof(qnumber));
		long c = 0;
		for (int j = 0; j < num_tensors; j++)
		{
			memcpy(&r_qnums_logical_i_ax[c], tlist[j].qnums_logical[i_ax], tlist[j].dim_logical[i_ax] * sizeof(qnumber));
			c += tlist[j].dim_logical[i_ax];
		}
		const qnumber** r_qnums_logical = ct_malloc(ndim * sizeof(qnumber*));
		for (int i = 0; i < ndim; i++)
		{
			if (i == i_ax) {
				r_qnums_logical[i] = r_qnums_logical_i_ax;
			}
			else {
				// simply copy the pointer
				r_qnums_logical[i] = tlist[0].qnums_logical[i];
			}
		}

		allocate_block_sparse_tensor(tlist[0].dtype, ndim, r_dim_logical, tlist[0].axis_dir, r_qnums_logical, r);

		ct_free(r_qnums_logical);
		ct_free(r_qnums_logical_i_ax);
		ct_free(r_dim_logical);
	}

	// for each dense block of 'r'...
	const long nblocks = integer_product(r->dim_blocks, r->ndim);
	#pragma omp parallel for schedule(dynamic)
	for (long kr = 0; kr < nblocks; kr++)
	{
		struct dense_tensor* br = r->blocks[kr];
		if (br == NULL) {
			continue;
		}

		long* index_block_r = ct_malloc(r->ndim * sizeof(long));
		offset_to_tensor_index(r->ndim, r->dim_blocks, kr, index_block_r);

		struct dense_tensor* tlist_blocks = ct_malloc(num_tensors * sizeof(struct dense_tensor));

		// collect the input tensors containing the current quantum number of axis 'i_ax'
		const qnumber qnum_i_ax = r->qnums_blocks[i_ax][index_block_r[i_ax]];
		// indices for other axes must be identical (since quantum numbers agree)
		long* index_block_t = ct_malloc(r->ndim * sizeof(long));
		memcpy(index_block_t, index_block_r, r->ndim * sizeof(long));
		int num_tlist_blocks = 0;
		for (int j = 0; j < num_tensors; j++)
		{
			bool found = false;
			for (long l = 0; l < tlist[j].dim_blocks[i_ax]; l++) {
				if (tlist[j].qnums_blocks[i_ax][l] == qnum_i_ax) {
					index_block_t[i_ax] = l;
					found = true;
					break;
				}
			}
			if (!found) {
				continue;
			}

			const long kt = tensor_index_to_offset(tlist[j].ndim, tlist[j].dim_blocks, index_block_t);
			const struct dense_tensor* bt = tlist[j].blocks[kt];
			assert(bt != NULL);

			// copy pointers, not actual data
			memcpy(&tlist_blocks[num_tlist_blocks], bt, sizeof(struct dense_tensor));
			num_tlist_blocks++;
		}
		assert(1 <= num_tlist_blocks && num_tlist_blocks <= num_tensors);

		dense_tensor_concatenate_fill(tlist_blocks, num_tlist_blocks, i_ax, br);

		ct_free(tlist_blocks);
		ct_free(index_block_r);
		ct_free(index_block_t);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Count the number of occurrences of each unique quantum number.
///
static void count_quantum_numbers(const qnumber* qnums_logical, const long dim, const qnumber* unique_qnums, const long num_qnums, long* qnum_counts)
{
	memset(qnum_counts, 0, num_qnums * sizeof(long));

	for (long i = 0; i < dim; i++) {
		for (long j = 0; j < num_qnums; j++) {
			if (qnums_logical[i] == unique_qnums[j]) {
				qnum_counts[j]++;
				break;
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Quantum number counter utility data structure for a block-sparse tensor.
///
struct block_sparse_tensor_qnumber_counter
{
	long** qnum_counts;  //!< number of occurrences of the quantum numbers along each axis
	int ndim;            //!< number of dimensions (degree)
};


//________________________________________________________________________________________________________________________
///
/// \brief Create and fill a block-sparse tensor quantum number counter.
///
static void create_block_sparse_tensor_qnumber_counter(const struct block_sparse_tensor* t, struct block_sparse_tensor_qnumber_counter* qcounter)
{
	qcounter->ndim = t->ndim;

	qcounter->qnum_counts = ct_malloc(t->ndim * sizeof(long*));
	for (int i = 0; i < t->ndim; i++)
	{
		qcounter->qnum_counts[i] = ct_malloc(t->dim_blocks[i] * sizeof(long));
		count_quantum_numbers(
			t->qnums_logical[i], t->dim_logical[i], t->qnums_blocks[i], t->dim_blocks[i],
			qcounter->qnum_counts[i]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a block-sparse tensor quantum number counter (free memory).
///
static void delete_block_sparse_tensor_qnumber_counter(struct block_sparse_tensor_qnumber_counter* qcounter)
{
	for (int i = 0; i < qcounter->ndim; i++)
	{
		ct_free(qcounter->qnum_counts[i]);
	}
	ct_free(qcounter->qnum_counts);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compose tensors along the specified axes, creating a block-diagonal pattern. All other dimensions and their quantum numbers must respectively agree.
///
void block_sparse_tensor_block_diag(const struct block_sparse_tensor* restrict tlist, const int num_tensors, const int* i_ax, const int ndim_block, struct block_sparse_tensor* restrict r)
{
	assert(num_tensors >= 1);
	const int ndim = tlist[0].ndim;
	assert(1 <= ndim_block && ndim_block <= ndim);

	bool* i_ax_indicator = ct_calloc(ndim, sizeof(bool));
	for (int i = 0; i < ndim_block; i++)
	{
		assert(0 <= i_ax[i] && i_ax[i] < ndim);
		assert(!i_ax_indicator[i_ax[i]]);  // axis index can only appear once
		i_ax_indicator[i_ax[i]] = true;
	}

	#ifndef NDEBUG
	for (int j = 0; j < num_tensors - 1; j++)
	{
		// data types must match
		assert(tlist[j].dtype == tlist[j + 1].dtype);
		// degrees must match
		assert(tlist[j].ndim == tlist[j + 1].ndim);
		for (int i = 0; i < tlist[j].ndim; i++)
		{
			// axis directions must match
			assert(tlist[j].axis_dir[i] == tlist[j + 1].axis_dir[i]);
			if (!i_ax_indicator[i]) {
				// other dimensions must match
				assert(tlist[j].dim_logical[i] == tlist[j + 1].dim_logical[i]);
				// quantum numbers must match entrywise
				assert(qnumber_all_equal(tlist[j].dim_logical[i], tlist[j].qnums_logical[i], tlist[j + 1].qnums_logical[i]));
			}
		}
	}
	#endif

	// allocate new block-sparse tensor 'r'
	{
		// dimensions of new tensor
		long* r_dim_logical = ct_malloc(ndim * sizeof(long));
		for (int i = 0; i < ndim; i++)
		{
			if (i_ax_indicator[i])
			{
				long dsum = 0;
				for (int j = 0; j < num_tensors; j++) {
					dsum += tlist[j].dim_logical[i];
				}
				r_dim_logical[i] = dsum;
			}
			else {
				r_dim_logical[i] = tlist[0].dim_logical[i];
			}
		}

		// logical quantum numbers along each dimension
		qnumber** r_qnums_logical = ct_malloc(ndim * sizeof(qnumber*));
		for (int i = 0; i < ndim; i++)
		{
			if (i_ax_indicator[i])
			{
				r_qnums_logical[i] = ct_malloc(r_dim_logical[i] * sizeof(qnumber));
				long c = 0;
				for (int j = 0; j < num_tensors; j++)
				{
					memcpy(&r_qnums_logical[i][c], tlist[j].qnums_logical[i], tlist[j].dim_logical[i] * sizeof(qnumber));
					c += tlist[j].dim_logical[i];
				}
				assert(c == r_dim_logical[i]);
			}
			else {
				// simply copy the pointer
				r_qnums_logical[i] = tlist[0].qnums_logical[i];
			}
		}

		allocate_block_sparse_tensor(tlist[0].dtype, ndim, r_dim_logical, tlist[0].axis_dir, (const qnumber**)r_qnums_logical, r);

		for (int i = 0; i < ndim; i++) {
			if (i_ax_indicator[i]) {
				ct_free(r_qnums_logical[i]);
			}
		}
		ct_free(r_qnums_logical);
		ct_free(r_dim_logical);
	}

	struct block_sparse_tensor_qnumber_counter* qcounter = ct_malloc(num_tensors * sizeof(struct block_sparse_tensor_qnumber_counter));
	for (int j = 0; j < num_tensors; j++) {
		create_block_sparse_tensor_qnumber_counter(&tlist[j], &qcounter[j]);
	}

	long* zero_list = ct_calloc(r->ndim, sizeof(long));

	// for each dense block of 'r'...
	const long nblocks = integer_product(r->dim_blocks, r->ndim);
	#pragma omp parallel for schedule(dynamic)
	for (long kr = 0; kr < nblocks; kr++)
	{
		struct dense_tensor* br = r->blocks[kr];
		if (br == NULL) {
			continue;
		}

		long* index_block_r = ct_malloc(r->ndim * sizeof(long));
		offset_to_tensor_index(r->ndim, r->dim_blocks, kr, index_block_r);

		// block quantum numbers
		qnumber* qnums_block = ct_malloc(r->ndim * sizeof(qnumber));
		for (int i = 0; i < r->ndim; i++) {
			qnums_block[i] = r->qnums_blocks[i][index_block_r[i]];
		}

		// require zero initialization
		long* offset = ct_calloc(r->ndim, sizeof(long));

		// collect the input tensors containing a dense block with the current quantum numbers
		struct dense_tensor* tlist_blocks = ct_malloc(num_tensors * sizeof(struct dense_tensor));
		int num_tlist_blocks = 0;
		for (int j = 0; j < num_tensors; j++)
		{
			const struct dense_tensor* bt = block_sparse_tensor_get_block(&tlist[j], qnums_block);
			if (bt != NULL)
			{
				// use padding to take index offsets from preceding tensors into account
				dense_tensor_pad_zeros(bt, offset, zero_list, &tlist_blocks[num_tlist_blocks]);
				// reset offsets
				memset(offset, 0, r->ndim * sizeof(long));
				num_tlist_blocks++;
			}
			else
			{
				// 'tlist[j]' can contain current block quantum numbers along some of its axes
				for (int i = 0; i < r->ndim; i++) {
					if (i_ax_indicator[i]) {
						for (long k = 0; k < tlist[j].dim_blocks[i]; k++) {
							if (tlist[j].qnums_blocks[i][k] == qnums_block[i]) {
								offset[i] += qcounter[j].qnum_counts[i][k];
								break;
							}
						}
					}
				}
			}
		}
		assert(num_tlist_blocks <= num_tensors);

		if (num_tlist_blocks == 0)
		{
			ct_free(tlist_blocks);
			ct_free(offset);
			ct_free(qnums_block);
			ct_free(index_block_r);
			continue;
		}

		// use padding to take remaining dimensions from trailing tensors into account
		{
			bool must_pad = false;
			for (int i = 0; i < r->ndim; i++) {
				if (offset[i] > 0) {
					must_pad = true;
					break;
				}
			}
			if (must_pad) {
				struct dense_tensor tmp;
				dense_tensor_pad_zeros(&tlist_blocks[num_tlist_blocks - 1], zero_list, offset, &tmp);
				delete_dense_tensor(&tlist_blocks[num_tlist_blocks - 1]);
				move_dense_tensor_data(&tmp, &tlist_blocks[num_tlist_blocks - 1]);
			}
		}

		dense_tensor_block_diag_fill(tlist_blocks, num_tlist_blocks, i_ax, ndim_block, br);

		for (int j = 0; j < num_tlist_blocks; j++) {
			delete_dense_tensor(&tlist_blocks[j]);
		}

		ct_free(tlist_blocks);
		ct_free(offset);
		ct_free(qnums_block);
		ct_free(index_block_r);
	}

	ct_free(zero_list);
	for (int j = 0; j < num_tensors; j++) {
		delete_block_sparse_tensor_qnumber_counter(&qcounter[j]);
	}
	ct_free(qcounter);
	ct_free(i_ax_indicator);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the logical QR decomposition of a block-sparse matrix.
///
/// The logical quantum numbers of the axis connecting Q and R will be sorted.
/// The logical R matrix is upper-triangular after sorting its second dimension by quantum numbers.
///
int block_sparse_tensor_qr(const struct block_sparse_tensor* restrict a, struct block_sparse_tensor* restrict q, struct block_sparse_tensor* restrict r)
{
	// require a matrix
	assert(a->ndim == 2);

	// determine dimensions and quantum numbers of intermediate axis connecting Q and R,
	// and allocate Q and R matrices

	long dim_interm = 0;

	// allocating array of maximum possible length
	qnumber* qnums_interm = ct_calloc(a->dim_logical[1], sizeof(qnumber));

	// loop over second dimension is outer loop to ensure that logical quantum numbers are sorted
	for (long j = 0; j < a->dim_blocks[1]; j++)
	{
		for (long i = 0; i < a->dim_blocks[0]; i++)
		{
			// probe whether quantum numbers sum to zero
			qnumber qsum = a->axis_dir[0] * a->qnums_blocks[0][i] + a->axis_dir[1] * a->qnums_blocks[1][j];
			if (qsum != 0) {
				continue;
			}

			const struct dense_tensor* b = a->blocks[i*a->dim_blocks[1] + j];
			assert(b != NULL);
			assert(b->ndim == 2);
			assert(b->dtype == a->dtype);

			const long k = lmin(b->dim[0], b->dim[1]);

			// append a sequence of logical quantum numbers of length k
			for (long l = 0; l < k; l++) {
				qnums_interm[dim_interm + l] = a->qnums_blocks[1][j];
			}
			dim_interm += k;
		}
	}
	assert(dim_interm <= a->dim_logical[1]);

	bool create_dummy_qr = false;

	if (dim_interm == 0)
	{
		// special case: 'a' does not contain any non-zero blocks
		// -> create dummy 'q' and 'r' matrices
		create_dummy_qr = true;
		dim_interm = 1;

		// use first block quantum number in 'a' to create a non-zero entry in 'q'
		const qnumber qnum_ax0 = a->qnums_blocks[0][0];
		const qnumber qnum_ax1 = -a->axis_dir[0]*a->axis_dir[1] * qnum_ax0;
		qnums_interm[0] = qnum_ax1;
	}

	// allocate the 'q' matrix
	const long dim_q[2] = { a->dim_logical[0], dim_interm };
	const qnumber* qnums_q[2] = { a->qnums_logical[0], qnums_interm };
	allocate_block_sparse_tensor(a->dtype, 2, dim_q, a->axis_dir, qnums_q, q);

	// allocate the 'r' matrix
	const long dim_r[2] = { dim_interm, a->dim_logical[1] };
	const enum tensor_axis_direction axis_dir_r[2] = { -a->axis_dir[1], a->axis_dir[1] };
	const qnumber* qnums_r[2] = { qnums_interm, a->qnums_logical[1] };
	allocate_block_sparse_tensor(a->dtype, 2, dim_r, axis_dir_r, qnums_r, r);

	ct_free(qnums_interm);

	if (create_dummy_qr)
	{
		// make 'q' an isometry with a single non-zero entry 1

		// use first block quantum number in 'a' to create a non-zero entry in 'q'
		const qnumber qnum_ax0 = a->qnums_blocks[0][0];
		const qnumber qnum_ax1 = -a->axis_dir[0]*a->axis_dir[1] * qnum_ax0;
		const qnumber qnums_block[2] = { qnum_ax0, qnum_ax1 };
		struct dense_tensor* bq = block_sparse_tensor_get_block(q, qnums_block);
		assert(bq != NULL);
		// set first entry in block to 1
		memcpy(bq->data, numeric_one(q->dtype), sizeof_numeric_type(q->dtype));

		// 'r' matrix is logically zero

		return 0;
	}

	// perform QR decompositions of the individual blocks
	bool failed = false;
	#pragma omp parallel for schedule(dynamic) collapse(2)
	for (long i = 0; i < a->dim_blocks[0]; i++)
	{
		for (long j = 0; j < a->dim_blocks[1]; j++)
		{
			const struct dense_tensor* ba = a->blocks[i*a->dim_blocks[1] + j];
			if (ba == NULL) {
				continue;
			}
			assert(ba->ndim == 2);
			assert(ba->dtype == a->dtype);
			assert(a->axis_dir[0] * a->qnums_blocks[0][i] + a->axis_dir[1] * a->qnums_blocks[1][j] == 0);

			// find corresponding blocks in 'q' and 'r'
			assert(q->qnums_blocks[0][i] == a->qnums_blocks[0][i]);
			assert(r->qnums_blocks[1][j] == a->qnums_blocks[1][j]);
			// connecting axis contains only the quantum numbers of non-empty blocks in 'a', so cannot use the block index from 'a'
			long k;
			for (k = 0; k < q->dim_blocks[1]; k++) {
				if (q->qnums_blocks[1][k] == a->qnums_blocks[1][j]) {
					break;
				}
			}
			assert(k < q->dim_blocks[1]);
			assert(q->qnums_blocks[1][k] == a->qnums_blocks[1][j]);
			assert(r->qnums_blocks[0][k] == a->qnums_blocks[1][j]);
			struct dense_tensor* bq = q->blocks[i*q->dim_blocks[1] + k];
			struct dense_tensor* br = r->blocks[k*r->dim_blocks[1] + j];
			assert(bq != NULL);
			assert(br != NULL);

			// perform QR decomposition of block
			int ret = dense_tensor_qr_fill(ba, bq, br);
			if (ret != 0) {
				failed = true;
			}
		}
	}

	if (failed) {
		return -1;
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the logical RQ decomposition of a block-sparse matrix.
///
/// The logical quantum numbers of the axis connecting R and Q will be sorted.
/// The logical R matrix is upper-triangular after sorting its first dimension by quantum numbers.
///
int block_sparse_tensor_rq(const struct block_sparse_tensor* restrict a, struct block_sparse_tensor* restrict r, struct block_sparse_tensor* restrict q)
{
	// require a matrix
	assert(a->ndim == 2);

	// determine dimensions and quantum numbers of intermediate axis connecting R and Q,
	// and allocate R and Q matrices

	long dim_interm = 0;

	// allocating array of maximum possible length
	qnumber* qnums_interm = ct_calloc(a->dim_logical[0], sizeof(qnumber));

	// loop over first dimension is outer loop to ensure that logical quantum numbers are sorted
	for (long i = 0; i < a->dim_blocks[0]; i++)
	{
		for (long j = 0; j < a->dim_blocks[1]; j++)
		{
			// probe whether quantum numbers sum to zero
			qnumber qsum = a->axis_dir[0] * a->qnums_blocks[0][i] + a->axis_dir[1] * a->qnums_blocks[1][j];
			if (qsum != 0) {
				continue;
			}

			const struct dense_tensor* b = a->blocks[i*a->dim_blocks[1] + j];
			assert(b != NULL);
			assert(b->ndim == 2);
			assert(b->dtype == a->dtype);

			const long k = lmin(b->dim[0], b->dim[1]);

			// append a sequence of logical quantum numbers of length k
			for (long l = 0; l < k; l++) {
				qnums_interm[dim_interm + l] = a->qnums_blocks[0][i];
			}
			dim_interm += k;
		}
	}
	assert(dim_interm <= a->dim_logical[0]);

	bool create_dummy_qr = false;

	if (dim_interm == 0)
	{
		// special case: 'a' does not contain any non-zero blocks
		// -> create dummy 'q' and 'r' matrices
		create_dummy_qr = true;
		dim_interm = 1;

		// use first block quantum number in 'a' to create a non-zero entry in 'q'
		const qnumber qnum_ax1 = a->qnums_blocks[1][0];
		const qnumber qnum_ax0 = -a->axis_dir[0]*a->axis_dir[1] * qnum_ax1;
		qnums_interm[0] = qnum_ax0;
	}

	// allocate the 'r' matrix
	const long dim_r[2] = { a->dim_logical[0], dim_interm };
	const enum tensor_axis_direction axis_dir_r[2] = { a->axis_dir[0], -a->axis_dir[0] };
	const qnumber* qnums_r[2] = { a->qnums_logical[0], qnums_interm };
	allocate_block_sparse_tensor(a->dtype, 2, dim_r, axis_dir_r, qnums_r, r);

	// allocate the 'q' matrix
	const long dim_q[2] = { dim_interm, a->dim_logical[1] };
	const qnumber* qnums_q[2] = { qnums_interm, a->qnums_logical[1] };
	allocate_block_sparse_tensor(a->dtype, 2, dim_q, a->axis_dir, qnums_q, q);

	ct_free(qnums_interm);

	if (create_dummy_qr)
	{
		// make 'q' an isometry with a single non-zero entry 1

		// use first block quantum number in 'a' to create a non-zero entry in 'q'
		const qnumber qnum_ax1 = a->qnums_blocks[1][0];
		const qnumber qnum_ax0 = -a->axis_dir[0]*a->axis_dir[1] * qnum_ax1;
		const qnumber qnums_block[2] = { qnum_ax0, qnum_ax1 };
		struct dense_tensor* bq = block_sparse_tensor_get_block(q, qnums_block);
		assert(bq != NULL);
		// set first entry in block to 1
		memcpy(bq->data, numeric_one(q->dtype), sizeof_numeric_type(q->dtype));

		// 'r' matrix is logically zero

		return 0;
	}

	// perform RQ decompositions of the individual blocks
	bool failed = false;
	#pragma omp parallel for schedule(dynamic) collapse(2)
	for (long i = 0; i < a->dim_blocks[0]; i++)
	{
		for (long j = 0; j < a->dim_blocks[1]; j++)
		{
			const struct dense_tensor* ba = a->blocks[i*a->dim_blocks[1] + j];
			if (ba == NULL) {
				continue;
			}
			assert(ba->ndim == 2);
			assert(ba->dtype == a->dtype);
			assert(a->axis_dir[0] * a->qnums_blocks[0][i] + a->axis_dir[1] * a->qnums_blocks[1][j] == 0);

			// find corresponding blocks in 'q' and 'r'
			assert(r->qnums_blocks[0][i] == a->qnums_blocks[0][i]);
			assert(q->qnums_blocks[1][j] == a->qnums_blocks[1][j]);
			// connecting axis contains only the quantum numbers of non-empty blocks in 'a', so cannot use the block index from 'a'
			long k;
			for (k = 0; k < q->dim_blocks[0]; k++) {
				if (q->qnums_blocks[0][k] == a->qnums_blocks[0][i]) {
					break;
				}
			}
			assert(k < q->dim_blocks[0]);
			assert(q->qnums_blocks[0][k] == a->qnums_blocks[0][i]);
			assert(r->qnums_blocks[1][k] == a->qnums_blocks[0][i]);
			struct dense_tensor* br = r->blocks[i*r->dim_blocks[1] + k];
			struct dense_tensor* bq = q->blocks[k*q->dim_blocks[1] + j];
			assert(br != NULL);
			assert(bq != NULL);

			// perform RQ decomposition of block
			int ret = dense_tensor_rq_fill(ba, br, bq);
			if (ret != 0) {
				failed = true;
			}
		}
	}

	if (failed) {
		return -1;
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the logical SVD decomposition of a block-sparse matrix.
///
/// The logical quantum numbers of the axis connecting U and Vh will be sorted.
/// The singular values are returned in a dense vector.
///
int block_sparse_tensor_svd(const struct block_sparse_tensor* restrict a, struct block_sparse_tensor* restrict u, struct dense_tensor* restrict s, struct block_sparse_tensor* restrict vh)
{
	// require a matrix
	assert(a->ndim == 2);

	// determine dimensions and quantum numbers of intermediate axis connecting 'u' and 'vh',
	// and allocate 'u' and 'vh' matrices

	long dim_interm = 0;

	// allocating array of maximum possible length
	qnumber* qnums_interm = ct_calloc(a->dim_logical[1], sizeof(qnumber));

	// loop over second dimension is outer loop to ensure that logical quantum numbers are sorted
	for (long j = 0; j < a->dim_blocks[1]; j++)
	{
		for (long i = 0; i < a->dim_blocks[0]; i++)
		{
			// probe whether quantum numbers sum to zero
			qnumber qsum = a->axis_dir[0] * a->qnums_blocks[0][i] + a->axis_dir[1] * a->qnums_blocks[1][j];
			if (qsum != 0) {
				continue;
			}

			const struct dense_tensor* b = a->blocks[i*a->dim_blocks[1] + j];
			assert(b != NULL);
			assert(b->ndim == 2);
			assert(b->dtype == a->dtype);

			const long k = lmin(b->dim[0], b->dim[1]);

			// append a sequence of logical quantum numbers of length k
			for (long l = 0; l < k; l++) {
				qnums_interm[dim_interm + l] = a->qnums_blocks[1][j];
			}
			dim_interm += k;
		}
	}
	assert(dim_interm <= a->dim_logical[1]);

	bool create_dummy_svd = false;

	if (dim_interm == 0)
	{
		// special case: 'a' does not contain any non-zero blocks
		// -> create dummy 'u', 's' and 'vh' matrices
		create_dummy_svd = true;
		dim_interm = 1;

		// use first block quantum number in 'a' to create a non-zero entry in 'u'
		const qnumber qnum_ax0 = a->qnums_blocks[0][0];
		const qnumber qnum_ax1 = -a->axis_dir[0]*a->axis_dir[1] * qnum_ax0;
		qnums_interm[0] = qnum_ax1;
	}

	// allocate the 'u' matrix
	const long dim_u[2] = { a->dim_logical[0], dim_interm };
	const qnumber* qnums_u[2] = { a->qnums_logical[0], qnums_interm };
	allocate_block_sparse_tensor(a->dtype, 2, dim_u, a->axis_dir, qnums_u, u);

	// allocate the 's' vector
	const long dim_s[1] = { dim_interm };
	allocate_dense_tensor(numeric_real_type(a->dtype), 1, dim_s, s);

	// allocate the 'vh' matrix
	const long dim_vh[2] = { dim_interm, a->dim_logical[1] };
	const enum tensor_axis_direction axis_dir_vh[2] = { -a->axis_dir[1], a->axis_dir[1] };
	const qnumber* qnums_vh[2] = { qnums_interm, a->qnums_logical[1] };
	allocate_block_sparse_tensor(a->dtype, 2, dim_vh, axis_dir_vh, qnums_vh, vh);

	ct_free(qnums_interm);

	if (create_dummy_svd)
	{
		// make 'u' an isometry with a single non-zero entry 1

		// use first block quantum number in 'a' to create a non-zero entry in 'u'
		const qnumber qnum_ax0 = a->qnums_blocks[0][0];
		const qnumber qnum_ax1 = -a->axis_dir[0]*a->axis_dir[1] * qnum_ax0;
		const qnumber qnums_block[2] = { qnum_ax0, qnum_ax1 };
		struct dense_tensor* bu = block_sparse_tensor_get_block(u, qnums_block);
		assert(bu != NULL);
		// set first entry in block to 1
		memcpy(bu->data, numeric_one(u->dtype), sizeof_numeric_type(u->dtype));

		// single entry in 's' is zero

		// 'vh' can only be zero due to quantum number incompatibility

		return 0;
	}

	// perform SVD decompositions of the individual blocks
	bool failed = false;
	#pragma omp parallel for schedule(dynamic) collapse(2)
	for (long i = 0; i < a->dim_blocks[0]; i++)
	{
		for (long j = 0; j < a->dim_blocks[1]; j++)
		{
			const struct dense_tensor* ba = a->blocks[i*a->dim_blocks[1] + j];
			if (ba == NULL) {
				continue;
			}
			assert(ba->ndim == 2);
			assert(ba->dtype == a->dtype);
			assert(a->axis_dir[0] * a->qnums_blocks[0][i] + a->axis_dir[1] * a->qnums_blocks[1][j] == 0);

			// find corresponding blocks in 'u' and 'vh'
			assert( u->qnums_blocks[0][i] == a->qnums_blocks[0][i]);
			assert(vh->qnums_blocks[1][j] == a->qnums_blocks[1][j]);
			// connecting axis contains only the quantum numbers of non-empty blocks in 'a', so cannot use the block index from 'a'
			long k;
			for (k = 0; k < u->dim_blocks[1]; k++) {
				if (u->qnums_blocks[1][k] == a->qnums_blocks[1][j]) {
					break;
				}
			}
			assert(k < u->dim_blocks[1]);
			assert( u->qnums_blocks[1][k] == a->qnums_blocks[1][j]);
			assert(vh->qnums_blocks[0][k] == a->qnums_blocks[1][j]);
			struct dense_tensor* bu  =  u->blocks[i* u->dim_blocks[1] + k];
			struct dense_tensor* bvh = vh->blocks[k*vh->dim_blocks[1] + j];
			assert(bu  != NULL);
			assert(bvh != NULL);
			assert(bu->dim[1] == bvh->dim[0]);

			struct dense_tensor bs;
			const long dim_bs[1] = { bu->dim[1] };
			allocate_dense_tensor(numeric_real_type(a->dtype), 1, dim_bs, &bs);

			// perform SVD decomposition of block
			int ret = dense_tensor_svd_fill(ba, bu, &bs, bvh);
			if (ret != 0) {
				failed = true;
			}

			// copy entries from 'bs' into output vector 's'
			// find range of logical indices corresponding to current block
			for (k = 0; k < u->dim_logical[1]; k++) {
				if (u->qnums_logical[1][k] == a->qnums_blocks[1][j]) {
					break;
				}
			}
			// expecting continuous range of quantum number 'a->qnums_blocks[1][j]' along connecting axis
			assert(k + bs.dim[0] <= u->dim_logical[1]);
			for (long l = 0; l < bs.dim[0]; l++) {
				assert(u->qnums_logical[1][k + l] == a->qnums_blocks[1][j]);
			}
			assert(bs.dtype == s->dtype);
			// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
			const size_t dtype_size = sizeof_numeric_type(s->dtype);
			memcpy((int8_t*)s->data + k * dtype_size, bs.data, bs.dim[0] * dtype_size);

			delete_dense_tensor(&bs);
		}
	}

	if (failed) {
		return -1;
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Test whether two block-sparse tensors agree in terms of quantum numbers and entries within tolerance 'tol'.
///
bool block_sparse_tensor_allclose(const struct block_sparse_tensor* restrict s, const struct block_sparse_tensor* restrict t, const double tol)
{
	// compare data types
	if (s->dtype != t->dtype) {
		return false;
	}

	// compare degrees
	if (s->ndim != t->ndim) {
		return false;
	}

	// compare dimensions
	for (int i = 0; i < s->ndim; i++)
	{
		if (s->dim_logical[i] != t->dim_logical[i]) {
			return false;
		}
	}

	// compare logical quantum numbers
	for (int i = 0; i < s->ndim; i++)
	{
		if (!qnumber_all_equal(s->dim_logical[i], s->qnums_logical[i], t->qnums_logical[i])) {
			return false;
		}
	}

	// compare axis directions
	for (int i = 0; i < s->ndim; i++)
	{
		if (s->axis_dir[i] != t->axis_dir[i]) {
			return false;
		}
	}

	// block dimensions and quantum numbers must now also agree
	for (int i = 0; i < s->ndim; i++)
	{
		assert(s->dim_blocks[i] == t->dim_blocks[i]);
		assert(qnumber_all_equal(s->dim_blocks[i], s->qnums_blocks[i], t->qnums_blocks[i]));
	}

	// for each block with matching quantum numbers...
	const long nblocks = integer_product(s->dim_blocks, s->ndim);
	long* index_block = ct_calloc(s->ndim, sizeof(long));
	for (long k = 0; k < nblocks; k++, next_tensor_index(s->ndim, s->dim_blocks, index_block))
	{
		// probe whether quantum numbers sum to zero
		qnumber qsum = 0;
		for (int i = 0; i < s->ndim; i++)
		{
			qsum += s->axis_dir[i] * s->qnums_blocks[i][index_block[i]];
		}
		if (qsum != 0) {
			continue;
		}

		// retrieve and compare blocks
		const struct dense_tensor* bs = s->blocks[k];
		const struct dense_tensor* bt = t->blocks[k];
		assert(bs != NULL);
		assert(bt != NULL);
		if (!dense_tensor_allclose(bs, bt, tol))
		{
			ct_free(index_block);
			return false;
		}
	}
	ct_free(index_block);

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Test whether a block-sparse tensors is close to the identity map within tolerance 'tol'.
///
bool block_sparse_tensor_is_identity(const struct block_sparse_tensor* t, const double tol)
{
	// must be a matrix
	if (t->ndim != 2) {
		return false;
	}

	// must be a square matrix
	if (t->dim_logical[0] != t->dim_logical[1]) {
		return false;
	}

	// require one "in" and one "out" axis direction
	if (t->axis_dir[0] + t->axis_dir[1] != 0) {
		return false;
	}

	// logical quantum numbers must match
	if (!qnumber_all_equal(t->dim_logical[0], t->qnums_logical[0], t->qnums_logical[1])) {
		return false;
	}

	// individual blocks must be identity matrices
	const long nblocks = integer_product(t->dim_blocks, t->ndim);
	for (long k = 0; k < nblocks; k++)
	{
		const struct dense_tensor* b = t->blocks[k];
		if (b != NULL) {
			if (!dense_tensor_is_identity(b, tol)) {
				return false;
			}
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Test whether a block-sparse tensors is an isometry within tolerance 'tol'.
///
bool block_sparse_tensor_is_isometry(const struct block_sparse_tensor* t, const double tol, const bool transpose)
{
	// must be a matrix
	if (t->ndim != 2) {
		return false;
	}

	// entry-wise conjugation
	struct block_sparse_tensor tc;
	// TODO: avoid full copy
	copy_block_sparse_tensor(t, &tc);
	conjugate_block_sparse_tensor(&tc);
	// revert tensor axes directions for multiplication
	tc.axis_dir[0] = -tc.axis_dir[0];
	tc.axis_dir[1] = -tc.axis_dir[1];

	struct block_sparse_tensor t2;
	if (!transpose) {
		block_sparse_tensor_dot(&tc, TENSOR_AXIS_RANGE_LEADING, t, TENSOR_AXIS_RANGE_LEADING, 1, &t2);
	}
	else {
		block_sparse_tensor_dot(t, TENSOR_AXIS_RANGE_TRAILING, &tc, TENSOR_AXIS_RANGE_TRAILING, 1, &t2);
	}

	const bool is_isometry = block_sparse_tensor_is_identity(&t2, tol);

	delete_block_sparse_tensor(&t2);
	delete_block_sparse_tensor(&tc);

	return is_isometry;
}


//________________________________________________________________________________________________________________________
///
/// \brief Overall number of entries in the blocks of the tensor.
///
long block_sparse_tensor_num_elements_blocks(const struct block_sparse_tensor* t)
{
	const long nblocks = integer_product(t->dim_blocks, t->ndim);

	long nelem = 0;
	for (long k = 0; k < nblocks; k++)
	{
		if (t->blocks[k] != NULL) {
			nelem += dense_tensor_num_elements(t->blocks[k]);
		}
	}

	return nelem;
}


//________________________________________________________________________________________________________________________
///
/// \brief Store the block entries in a linear array.
///
void block_sparse_tensor_serialize_entries(const struct block_sparse_tensor* t, void* entries)
{
	const size_t dtype_size = sizeof_numeric_type(t->dtype);

	// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
	int8_t* pentries = (int8_t*)entries;

	const long nblocks = integer_product(t->dim_blocks, t->ndim);
	long offset = 0;
	for (long k = 0; k < nblocks; k++)
	{
		const struct dense_tensor* b = t->blocks[k];
		if (b != NULL)
		{
			assert(t->dtype == b->dtype);
			const long nelem = dense_tensor_num_elements(b);
			memcpy(pentries + offset * dtype_size, b->data, nelem * dtype_size);
			offset += nelem;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Fill the block entries from a linear array.
///
void block_sparse_tensor_deserialize_entries(struct block_sparse_tensor* t, const void* entries)
{
	const size_t dtype_size = sizeof_numeric_type(t->dtype);

	// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
	const int8_t* pentries = (const int8_t*)entries;

	const long nblocks = integer_product(t->dim_blocks, t->ndim);
	long offset = 0;
	for (long k = 0; k < nblocks; k++)
	{
		struct dense_tensor* b = t->blocks[k];
		if (b != NULL)
		{
			assert(t->dtype == b->dtype);
			const long nelem = dense_tensor_num_elements(b);
			memcpy(b->data, pentries + offset * dtype_size, nelem * dtype_size);
			offset += nelem;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct the maps from a logical index to the corresponding dense block and entry index along an axis.
///
static void construct_logical_to_block_index_maps(const qnumber* qnums_logical, const long dim, const qnumber* unique_qnums, const long num_qnums, long* index_map_block, long* index_map_block_entry)
{
	long* counter = ct_calloc(num_qnums, sizeof(long));

	for (long i = 0; i < dim; i++)
	{
		#ifndef NDEBUG
		bool found = false;
		#endif
		for (long j = 0; j < num_qnums; j++)
		{
			if (qnums_logical[i] == unique_qnums[j]) {
				index_map_block[i] = j;
				index_map_block_entry[i] = counter[j]++;
				#ifndef NDEBUG
				found = true;
				#endif
				break;
			}
		}
		assert(found);
	}

	ct_free(counter);
}


//________________________________________________________________________________________________________________________
///
/// \brief Create a block-sparse tensor element accessor.
///
void create_block_sparse_tensor_entry_accessor(const struct block_sparse_tensor* t, struct block_sparse_tensor_entry_accessor* acc)
{
	acc->tensor = t;

	acc->index_map_blocks        = ct_malloc(t->ndim * sizeof(long*));
	acc->index_map_block_entries = ct_malloc(t->ndim * sizeof(long*));

	for (int i = 0; i < t->ndim; i++)
	{
		acc->index_map_blocks[i]        = ct_malloc(t->dim_logical[i] * sizeof(long));
		acc->index_map_block_entries[i] = ct_malloc(t->dim_logical[i] * sizeof(long));

		construct_logical_to_block_index_maps(
			t->qnums_logical[i], t->dim_logical[i], t->qnums_blocks[i], t->dim_blocks[i],
			acc->index_map_blocks[i], acc->index_map_block_entries[i]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a block-sparse tensor element accessor (free memory).
///
void delete_block_sparse_tensor_entry_accessor(struct block_sparse_tensor_entry_accessor* acc)
{
	for (int i = 0; i < acc->tensor->ndim; i++)
	{
		ct_free(acc->index_map_blocks[i]);
		ct_free(acc->index_map_block_entries[i]);
	}

	ct_free(acc->index_map_blocks);
	ct_free(acc->index_map_block_entries);
}


//________________________________________________________________________________________________________________________
///
/// \brief Retrieve a pointer to an individual entry of a block-sparse tensor, or NULL if the provided index does not correspond to a valid block.
///
void* block_sparse_tensor_get_entry(const struct block_sparse_tensor_entry_accessor* acc, const long* index)
{
	// retrieve block
	long* index_block = ct_malloc(acc->tensor->ndim * sizeof(long));
	for (int i = 0; i < acc->tensor->ndim; i++)
	{
		assert(0 <= index[i] && index[i] < acc->tensor->dim_logical[i]);
		index_block[i] = acc->index_map_blocks[i][index[i]];
	}
	const long ob = tensor_index_to_offset(acc->tensor->ndim, acc->tensor->dim_blocks, index_block);
	ct_free(index_block);

	const struct dense_tensor* block = acc->tensor->blocks[ob];
	if (block == NULL) {
		// logical entry is zero
		return NULL;
	}

	long* index_entry = ct_malloc(acc->tensor->ndim * sizeof(long));
	for (int i = 0; i < acc->tensor->ndim; i++)
	{
		index_entry[i] = acc->index_map_block_entries[i][index[i]];
	}
	const long oe = tensor_index_to_offset(block->ndim, block->dim, index_entry);
	ct_free(index_entry);

	// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
	return (int8_t*)block->data + oe * sizeof_numeric_type(acc->tensor->dtype);
}
