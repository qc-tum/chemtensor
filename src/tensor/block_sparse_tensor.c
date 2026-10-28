/// \file block_sparse_tensor.c
/// \brief Block-sparse tensor structure.

#include <math.h>
#include <memory.h>
#include <complex.h>
#include "block_sparse_tensor.h"
#include "cblas_ct.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Utility function for setting the linear list of block pointers and corresponding table indices.
///
static inline void block_sparse_tensor_fill_block_list_indices(struct block_sparse_tensor* t)
{
	ct_long* index_block = ct_calloc(t->ndim, sizeof(ct_long));
	const ct_long max_nblocks = integer_product(t->dim_sorted, t->ndim);
	ct_long c = 0;
	for (ct_long k = 0; k < max_nblocks; k++, next_tensor_index(t->ndim, t->dim_sorted, index_block))
	{
		if (t->block_table[k] == NULL) {
			continue;
		}
		t->block_list[c] = t->block_table[k];
		memcpy(&t->table_index_list[c * t->ndim], index_block, t->ndim * sizeof(ct_long));
		c++;
	}
	assert(c == t->nblocks);
	ct_free(index_block);
}


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
void allocate_block_sparse_tensor(const enum numeric_type dtype, const int ndim, const ct_long* restrict dim, const enum tensor_axis_direction* axis_dir, const qnumber** restrict qnums, struct block_sparse_tensor* restrict t)
{
	t->dtype = dtype;

	assert(ndim >= 0);
	t->ndim = ndim;

	if (ndim == 0)  // special case
	{
		t->table_index_list = NULL;

		t->dim_logical = NULL;
		t->dim_sorted  = NULL;

		t->axis_dir = NULL;

		t->axis_qnums_logical = NULL;
		t->axis_qnums_sorted  = NULL;

		// allocate memory for a single block
		struct dense_tensor* b = ct_calloc(1, sizeof(struct dense_tensor));
		allocate_zero_dense_tensor(dtype, ndim, dim, b);
		assert(dense_tensor_num_elements(b) == 1);
		t->block_table = ct_calloc(1, sizeof(struct dense_tensor*));
		t->block_list  = ct_calloc(1, sizeof(struct dense_tensor*));
		t->block_table[0] = b;
		t->block_list[0]  = b;
		t->data = b->data;
		t->nelem = 1;
		t->nblocks = 1;

		return;
	}

	t->dim_logical = ct_malloc(ndim * sizeof(ct_long));
	memcpy(t->dim_logical, dim, ndim * sizeof(ct_long));

	t->dim_sorted = ct_calloc(ndim, sizeof(ct_long));

	t->axis_dir = ct_malloc(ndim * sizeof(enum tensor_axis_direction));
	memcpy(t->axis_dir, axis_dir, ndim * sizeof(enum tensor_axis_direction));

	t->axis_qnums_logical = ct_calloc(ndim, sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		t->axis_qnums_logical[i] = ct_malloc(dim[i] * sizeof(qnumber));
		memcpy(t->axis_qnums_logical[i], qnums[i], dim[i] * sizeof(qnumber));
	}

	t->axis_qnums_sorted = ct_calloc(ndim, sizeof(qnumber*));

	// count quantum numbers along each dimension axis
	struct qnumber_count** qcounts = ct_malloc(ndim * sizeof(struct qnumber_count*));
	for (int i = 0; i < ndim; i++)
	{
		assert(dim[i] > 0);

		// likely not requiring all the allocated memory
		qcounts[i] = ct_calloc(dim[i], sizeof(struct qnumber_count));
		ct_long nqc = 0;
		for (ct_long j = 0; j < dim[i]; j++)
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

		t->dim_sorted[i] = nqc;

		// store quantum numbers
		t->axis_qnums_sorted[i] = ct_calloc(nqc, sizeof(qnumber));
		for (ct_long j = 0; j < nqc; j++) {
			t->axis_qnums_sorted[i][j] = qcounts[i][j].qnum;
		}
	}

	// allocate (still empty) dense tensor blocks and count total number of entries
	const ct_long max_nblocks = integer_product(t->dim_sorted, ndim);
	t->block_table = ct_calloc(max_nblocks, sizeof(struct dense_tensor*));
	t->nelem = 0;
	t->nblocks = 0;
	ct_long* index_block = ct_calloc(ndim, sizeof(ct_long));
	for (ct_long k = 0; k < max_nblocks; k++, next_tensor_index(ndim, t->dim_sorted, index_block))
	{
		// probe whether quantum numbers sum to zero
		qnumber qsum = 0;
		for (int i = 0; i < ndim; i++)
		{
			qsum += axis_dir[i] * t->axis_qnums_sorted[i][index_block[i]];
		}
		if (qsum != 0) {
			continue;
		}

		// allocate dense tensor block
		struct dense_tensor* b = ct_calloc(1, sizeof(struct dense_tensor));
		ct_long* bdim = ct_malloc(ndim * sizeof(ct_long));
		for (int i = 0; i < ndim; i++) {
			bdim[i] = qcounts[i][index_block[i]].count;
		}
		allocate_empty_dense_tensor(dtype, ndim, bdim, b);
		ct_free(bdim);
		t->block_table[k] = b;

		t->nelem += dense_tensor_num_elements(b);
		t->nblocks++;
	}
	ct_free(index_block);
	assert(t->nblocks <= max_nblocks);

	t->block_list = ct_malloc(t->nblocks * sizeof(struct dense_tensor*)); 
	t->table_index_list = ct_malloc(t->nblocks * ndim * sizeof(ct_long));
	block_sparse_tensor_fill_block_list_indices(t);

	for (int i = 0; i < ndim; i++)
	{
		ct_free(qcounts[i]);
	}
	ct_free(qcounts);

	const size_t dtype_size = sizeof_numeric_type(t->dtype);

	// contiguous array for block entries
	t->data = ct_calloc(t->nelem, dtype_size);

	// set dense block data pointers
	// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
	int8_t* pentries = (int8_t*)t->data;
	for (ct_long k = 0; k < t->nblocks; k++)
	{
		struct dense_tensor* b = t->block_list[k];
		assert(b != NULL);

		// set data pointer
		b->data = pentries;
		const size_t nbytes = dense_tensor_num_elements(b) * dtype_size;
		pentries += nbytes;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for a block-sparse tensor of the same type, dimensions, and quantum numbers as the provided tensor.
///
void allocate_block_sparse_tensor_like(const struct block_sparse_tensor* restrict s, struct block_sparse_tensor* restrict t)
{
	allocate_block_sparse_tensor(s->dtype, s->ndim, s->dim_logical, s->axis_dir, (const qnumber**)s->axis_qnums_logical, t);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a block-sparse tensor (free memory).
///
void delete_block_sparse_tensor(struct block_sparse_tensor* t)
{
	// special case t->ndim == 0 is handled correctly

	for (int i = 0; i < t->ndim; i++)
	{
		ct_free(t->axis_qnums_sorted[i]);
		ct_free(t->axis_qnums_logical[i]);
	}
	ct_free(t->axis_qnums_sorted);
	ct_free(t->axis_qnums_logical);

	ct_free(t->axis_dir);

	ct_free(t->dim_sorted);
	ct_free(t->dim_logical);

	ct_free(t->table_index_list);

	// free dense tensor blocks
	for (ct_long k = 0; k < t->nblocks; k++)
	{
		delete_empty_dense_tensor(t->block_list[k]);
		ct_free(t->block_list[k]);
	}
	ct_free(t->block_list);
	ct_free(t->block_table);
	t->block_list  = NULL;
	t->block_table = NULL;

	// free block entries
	ct_free(t->data);
	t->data = NULL;
	t->nelem = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Copy a block-sparse tensor, allocating memory for the copy.
///
void copy_block_sparse_tensor(const struct block_sparse_tensor* restrict src, struct block_sparse_tensor* restrict dst)
{
	dst->nelem   = src->nelem;
	dst->nblocks = src->nblocks;
	dst->dtype   = src->dtype;
	dst->ndim    = src->ndim;

	const int ndim = src->ndim;

	if (ndim == 0)  // special case
	{
		dst->table_index_list = NULL;

		dst->dim_logical = NULL;
		dst->dim_sorted  = NULL;

		dst->axis_dir = NULL;

		dst->axis_qnums_logical = NULL;
		dst->axis_qnums_sorted  = NULL;

		// copy the single block
		struct dense_tensor* b = ct_calloc(1, sizeof(struct dense_tensor));
		copy_dense_tensor(src->block_list[0], b);
		assert(dense_tensor_num_elements(b) == 1);
		dst->block_table = ct_calloc(1, sizeof(struct dense_tensor*));
		dst->block_list  = ct_calloc(1, sizeof(struct dense_tensor*));
		dst->block_table[0] = b;
		dst->block_list[0]  = b;
		dst->data = b->data;
		assert(dst->nelem == 1);
		assert(dst->nblocks == 1);

		return;
	}

	dst->dim_sorted = ct_malloc(ndim * sizeof(ct_long));
	memcpy(dst->dim_sorted, src->dim_sorted, ndim * sizeof(ct_long));

	dst->dim_logical = ct_malloc(ndim * sizeof(ct_long));
	memcpy(dst->dim_logical, src->dim_logical, ndim * sizeof(ct_long));

	dst->axis_dir = ct_malloc(ndim * sizeof(enum tensor_axis_direction));
	memcpy(dst->axis_dir, src->axis_dir, ndim * sizeof(enum tensor_axis_direction));

	dst->axis_qnums_sorted = ct_calloc(ndim, sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		dst->axis_qnums_sorted[i] = ct_malloc(src->dim_sorted[i] * sizeof(qnumber));
		memcpy(dst->axis_qnums_sorted[i], src->axis_qnums_sorted[i], src->dim_sorted[i] * sizeof(qnumber));
	}

	dst->axis_qnums_logical = ct_calloc(ndim, sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		dst->axis_qnums_logical[i] = ct_malloc(src->dim_logical[i] * sizeof(qnumber));
		memcpy(dst->axis_qnums_logical[i], src->axis_qnums_logical[i], src->dim_logical[i] * sizeof(qnumber));
	}

	// copy contiguous array for block entries
	const size_t dtype_size = sizeof_numeric_type(dst->dtype);
	dst->data = ct_malloc(dst->nelem * dtype_size);
	memcpy(dst->data, src->data, dst->nelem * dtype_size);

	// copy dense tensor blocks
	const ct_long max_nblocks = integer_product(src->dim_sorted, ndim);
	dst->block_table = ct_calloc(max_nblocks,  sizeof(struct dense_tensor*));
	ct_long* index_block = ct_calloc(ndim, sizeof(ct_long));
	for (ct_long k = 0; k < max_nblocks; k++, next_tensor_index(ndim, dst->dim_sorted, index_block))
	{
		if (src->block_table[k] == NULL) {
			continue;
		}

		// copy empty dense tensor block
		struct dense_tensor* b = ct_calloc(1, sizeof(struct dense_tensor));
		allocate_empty_dense_tensor(src->block_table[k]->dtype, src->block_table[k]->ndim, src->block_table[k]->dim, b);
		// preserve relative pointer offsets (relevant in case blocks are reshuffled)
		b->data = dst->data + (src->block_table[k]->data - src->data);
		dst->block_table[k] = b;
	}
	ct_free(index_block);

	// block list and table indices
	dst->block_list = ct_malloc(dst->nblocks * sizeof(struct dense_tensor*));
	dst->table_index_list = ct_malloc(dst->nblocks * dst->ndim * sizeof(ct_long));
	block_sparse_tensor_fill_block_list_indices(dst);
}


//________________________________________________________________________________________________________________________
///
/// \brief Retrieve a dense block based on its quantum numbers.
///
struct dense_tensor* block_sparse_tensor_get_block(const struct block_sparse_tensor* t, const qnumber* qnums)
{
	assert(t->ndim > 0);

	// find indices corresponding to quantum numbers
	ct_long* index = ct_malloc(t->ndim * sizeof(ct_long));
	for (int i = 0; i < t->ndim; i++)
	{
		index[i] = -1;
		for (ct_long j = 0; j < t->dim_sorted[i]; j++)
		{
			if (t->axis_qnums_sorted[i][j] == qnums[i]) {
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

	ct_long k = tensor_index_to_offset(t->ndim, t->dim_sorted, index);

	ct_free(index);

	return t->block_table[k];
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the 2-norm (Frobenius norm) of the tensor.
///
/// Result is returned as double also for single-precision tensor entries.
///
double block_sparse_tensor_norm2(const struct block_sparse_tensor* t)
{
	switch (t->dtype)
	{
		case CT_SINGLE_REAL:
		{
			return cblas_snrm2(t->nelem, t->data, 1);
		}
		case CT_DOUBLE_REAL:
		{
			return cblas_dnrm2(t->nelem, t->data, 1);
		}
		case CT_SINGLE_COMPLEX:
		{
			return cblas_scnrm2(t->nelem, t->data, 1);
		}
		case CT_DOUBLE_COMPLEX:
		{
			return cblas_dznrm2(t->nelem, t->data, 1);
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
/// \brief Flip the signs of the quantum numbers associated with a single tensor axis and reverse the direction of this axis.
///
void block_sparse_tensor_invert_axis_quantum_numbers(struct block_sparse_tensor* t, const int i_ax)
{
	assert(0 <= i_ax && i_ax < t->ndim);

	// allocate new dense tensor block table
	const ct_long max_nblocks = integer_product(t->dim_sorted, t->ndim);
	struct dense_tensor** block_table_new = ct_calloc(max_nblocks, sizeof(struct dense_tensor*));
	ct_long* index_block = ct_calloc(t->ndim, sizeof(ct_long));
	for (ct_long k = 0; k < max_nblocks; k++, next_tensor_index(t->ndim, t->dim_sorted, index_block))
	{
		if (t->block_table[k] == NULL) {
			continue;
		}

		const ct_long ib_i_ax_ref = index_block[i_ax];
		// new index after sign flip
		index_block[i_ax] = t->dim_sorted[i_ax] - ib_i_ax_ref - 1;
		const ct_long k_new = tensor_index_to_offset(t->ndim, t->dim_sorted, index_block);
		index_block[i_ax] = ib_i_ax_ref;  // restore

		// copy dense tensor pointer
		block_table_new[k_new] = t->block_table[k];
	}
	ct_free(index_block);
	ct_free(t->block_table);
	t->block_table = block_table_new;

	// re-create linear block list and table indices
	block_sparse_tensor_fill_block_list_indices(t);

	// flip block quantum number signs along axis 'i_ax'
	qnumber* qnums_blocks_i_ax_new = ct_malloc(t->dim_sorted[i_ax] * sizeof(qnumber));
	for (ct_long j = 0; j < t->dim_sorted[i_ax]; j++) {
		// ensure that block quantum numbers remain sorted
		qnums_blocks_i_ax_new[j] = -t->axis_qnums_sorted[i_ax][t->dim_sorted[i_ax] - j - 1];
	}
	ct_free(t->axis_qnums_sorted[i_ax]);
	t->axis_qnums_sorted[i_ax] = qnums_blocks_i_ax_new;

	// flip logical quantum number signs along axis 'i_ax'
	for (ct_long j = 0; j < t->dim_logical[i_ax]; j++) {
		t->axis_qnums_logical[i_ax][j] = -t->axis_qnums_logical[i_ax][j];
	}

	// flip tensor axis direction
	t->axis_dir[i_ax] = -t->axis_dir[i_ax];
}


//________________________________________________________________________________________________________________________
///
/// \brief Scale tensor 't' by 'alpha'.
///
/// Data types of all blocks and of 'alpha' must match.
///
void scale_block_sparse_tensor(const void* alpha, struct block_sparse_tensor* t)
{
	switch (t->dtype)
	{
		case CT_SINGLE_REAL:
		{
			cblas_sscal(t->nelem, *((float*)alpha), t->data, 1);
			break;
		}
		case CT_DOUBLE_REAL:
		{
			cblas_dscal(t->nelem, *((double*)alpha), t->data, 1);
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			cblas_cscal(t->nelem, alpha, t->data, 1);
			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			cblas_zscal(t->nelem, alpha, t->data, 1);
			break;
		}
		default:
		{
			// unknown data type
			assert(false);
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
	switch (t->dtype)
	{
		case CT_SINGLE_REAL:
		{
			cblas_sscal(t->nelem, *((float*)alpha), t->data, 1);
			break;
		}
		case CT_DOUBLE_REAL:
		{
			cblas_dscal(t->nelem, *((double*)alpha), t->data, 1);
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			cblas_csscal(t->nelem, *((float*)alpha), t->data, 1);
			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			cblas_zdscal(t->nelem, *((double*)alpha), t->data, 1);
			break;
		}
		default:
		{
			// unknown data type
			assert(false);
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Elementwise conjugation of a block-sparse tensor.
///
void conjugate_block_sparse_tensor(struct block_sparse_tensor* t)
{
	switch (t->dtype)
	{
		case CT_SINGLE_REAL:
		case CT_DOUBLE_REAL:
		{
			// no effect
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			scomplex* data = t->data;
			for (ct_long i = 0; i < t->nelem; i++)
			{
				data[i] = conjf(data[i]);
			}
			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			dcomplex* data = t->data;
			for (ct_long i = 0; i < t->nelem; i++)
			{
				data[i] = conj(data[i]);
			}
			break;
		}
		default:
		{
			// unknown data type
			assert(false);
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Set the dense blocks of the tensor to identity matrices, requiring same dimensions for quantum numbers appearing on both axes.
///
void block_sparse_tensor_set_identity_blocks(struct block_sparse_tensor* t)
{
	assert(t->ndim == 2);
	assert(t->axis_dir[0] != t->axis_dir[1]);

	#pragma omp parallel for schedule(dynamic)
	for (ct_long k = 0; k < t->nblocks; k++)
	{
		dense_tensor_set_identity(t->block_list[k]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Fill the non-zero blocks of a block-sparse tensor with random normal entries.
///
void block_sparse_tensor_fill_random_normal(const void* alpha, const void* shift, struct rng_state* rng_state, struct block_sparse_tensor* t)
{
	switch (t->dtype)
	{
		case CT_SINGLE_REAL:
		{
			float* data = t->data;
			const float a = *((float*)alpha);
			const float s = *((float*)shift);
			for (ct_long j = 0; j < t->nelem; j++) {
				data[j] = a * randnf(rng_state) + s;
			}
			break;
		}
		case CT_DOUBLE_REAL:
		{
			double* data = t->data;
			const double a = *((double*)alpha);
			const double s = *((double*)shift);
			for (ct_long j = 0; j < t->nelem; j++) {
				data[j] = a * randn(rng_state) + s;
			}
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			scomplex* data = t->data;
			const scomplex a = *((scomplex*)alpha);
			const scomplex s = *((scomplex*)shift);
			for (ct_long j = 0; j < t->nelem; j++) {
				data[j] = a * crandnf(rng_state) + s;
			}
			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			dcomplex* data = t->data;
			const dcomplex a = *((dcomplex*)alpha);
			const dcomplex s = *((dcomplex*)shift);
			for (ct_long j = 0; j < t->nelem; j++) {
				data[j] = a * crandn(rng_state) + s;
			}
			break;
		}
		default:
		{
			// unknown data type
			assert(false);
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Redirect the data pointer and corresponding dense block pointers to 'new_root'
///
void block_sparse_tensor_redirect_data_pointer(struct block_sparse_tensor* t, void* new_root)
{
	const ct_long shift = new_root - t->data;

	for (ct_long k = 0; k < t->nblocks; k++)
	{
		t->block_list[k]->data += shift;
	}

	t->data = new_root;
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert a block-sparse to an equivalent dense tensor.
///
void block_sparse_to_dense_tensor(const struct block_sparse_tensor* restrict s, struct dense_tensor* restrict t)
{
	allocate_zero_dense_tensor(s->dtype, s->ndim, s->dim_logical, t);

	// for each block...
	#pragma omp parallel for schedule(dynamic)
	for (ct_long k = 0; k < s->nblocks; k++)
	{
		const struct dense_tensor* b = s->block_list[k];
		assert(b->ndim == s->ndim);
		assert(b->dtype == s->dtype);

		const ct_long* index_block = &s->table_index_list[k * s->ndim];

		// fan-out dense to logical indices
		ct_long** index_map = ct_calloc(s->ndim, sizeof(ct_long*));
		for (int i = 0; i < s->ndim; i++)
		{
			index_map[i] = ct_calloc(b->dim[i], sizeof(ct_long));
			ct_long c = 0;
			for (ct_long j = 0; j < s->dim_logical[i]; j++)
			{
				if (s->axis_qnums_logical[i][j] == s->axis_qnums_sorted[i][index_block[i]])
				{
					index_map[i][c] = j;
					c++;
				}
			}
			assert(c == b->dim[i]);
		}

		// distribute dense tensor entries
		ct_long* index_b = ct_calloc(b->ndim, sizeof(ct_long));
		ct_long* index_t = ct_calloc(t->ndim, sizeof(ct_long));
		const ct_long nelem = dense_tensor_num_elements(b);
		switch (s->dtype)
		{
			case CT_SINGLE_REAL:
			{
				const float* bdata = b->data;
				float*       tdata = t->data;
				for (ct_long j = 0; j < nelem; j++, next_tensor_index(b->ndim, b->dim, index_b))
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
				for (ct_long j = 0; j < nelem; j++, next_tensor_index(b->ndim, b->dim, index_b))
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
				for (ct_long j = 0; j < nelem; j++, next_tensor_index(b->ndim, b->dim, index_b))
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
				for (ct_long j = 0; j < nelem; j++, next_tensor_index(b->ndim, b->dim, index_b))
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

	// for each block...
	#pragma omp parallel for schedule(dynamic)
	for (ct_long k = 0; k < s->nblocks; k++)
	{
		struct dense_tensor* b = s->block_list[k];
		assert(b->ndim == s->ndim);
		assert(b->dtype == s->dtype);

		const ct_long* index_block = &s->table_index_list[k * s->ndim];

		// fan-out dense to logical indices
		ct_long** index_map = ct_calloc(s->ndim, sizeof(ct_long*));
		for (int i = 0; i < s->ndim; i++)
		{
			index_map[i] = ct_calloc(b->dim[i], sizeof(ct_long));
			ct_long c = 0;
			for (ct_long j = 0; j < s->dim_logical[i]; j++)
			{
				if (s->axis_qnums_logical[i][j] == s->axis_qnums_sorted[i][index_block[i]])
				{
					index_map[i][c] = j;
					c++;
				}
			}
			assert(c == b->dim[i]);
		}

		// collect dense tensor entries
		ct_long* index_b = ct_calloc(b->ndim, sizeof(ct_long));
		ct_long* index_t = ct_calloc(t->ndim, sizeof(ct_long));
		const ct_long nelem = dense_tensor_num_elements(b);
		switch (s->dtype)
		{
			case CT_SINGLE_REAL:
			{
				const float* tdata = t->data;
				float*       bdata = b->data;
				for (ct_long j = 0; j < nelem; j++, next_tensor_index(b->ndim, b->dim, index_b))
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
				for (ct_long j = 0; j < nelem; j++, next_tensor_index(b->ndim, b->dim, index_b))
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
				for (ct_long j = 0; j < nelem; j++, next_tensor_index(b->ndim, b->dim, index_b))
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
				for (ct_long j = 0; j < nelem; j++, next_tensor_index(b->ndim, b->dim, index_b))
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
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Generalized transpose of a tensor 't' such that
/// the i-th axis in the output tensor 'r' is the perm[i]-th axis of the input tensor 't'.
///
/// Memory will be allocated for 'r'.
///
void block_sparse_tensor_transpose(const int* restrict perm, const struct block_sparse_tensor* restrict t, struct block_sparse_tensor* restrict r)
{
	r->nelem   = t->nelem;
	r->nblocks = t->nblocks;
	r->dtype   = t->dtype;
	r->ndim    = t->ndim;

	if (t->ndim == 0)  // special case
	{
		r->table_index_list = NULL;

		r->dim_logical = NULL;
		r->dim_sorted  = NULL;

		r->axis_dir = NULL;

		r->axis_qnums_logical = NULL;
		r->axis_qnums_sorted  = NULL;

		// allocate memory for a single block
		struct dense_tensor* b = ct_calloc(1, sizeof(struct dense_tensor));
		allocate_dense_tensor(r->dtype, 0, NULL, b);
		assert(dense_tensor_num_elements(b) == 1);
		// copy single number
		memcpy(b->data, t->block_list[0]->data, sizeof_numeric_type(t->block_list[0]->dtype));
		r->block_table = ct_calloc(1, sizeof(struct dense_tensor*));
		r->block_list  = ct_calloc(1, sizeof(struct dense_tensor*));
		r->block_table[0] = b;
		r->block_list[0]  = b;
		// set data pointer
		r->data = b->data;
		assert(r->nelem == 1);
		assert(r->nblocks == 1);

		return;
	}

	// ensure that 'perm' is a valid permutation
	assert(is_permutation(perm, t->ndim));

	// dimensions
	r->dim_logical = ct_malloc(t->ndim * sizeof(ct_long));
	r->dim_sorted  = ct_malloc(t->ndim * sizeof(ct_long));
	r->axis_dir    = ct_malloc(t->ndim * sizeof(enum tensor_axis_direction));
	for (int i = 0; i < t->ndim; i++)
	{
		r->dim_logical[i] = t->dim_logical[perm[i]];
		r->dim_sorted [i] = t->dim_sorted [perm[i]];
		r->axis_dir   [i] = t->axis_dir   [perm[i]];
	}

	// logical quantum numbers
	r->axis_qnums_logical = ct_calloc(t->ndim, sizeof(qnumber*));
	for (int i = 0; i < t->ndim; i++)
	{
		r->axis_qnums_logical[i] = ct_malloc(r->dim_logical[i] * sizeof(qnumber));
		memcpy(r->axis_qnums_logical[i], t->axis_qnums_logical[perm[i]], r->dim_logical[i] * sizeof(qnumber));
	}

	// block quantum numbers
	r->axis_qnums_sorted = ct_calloc(t->ndim, sizeof(qnumber*));
	for (int i = 0; i < t->ndim; i++)
	{
		r->axis_qnums_sorted[i] = ct_malloc(r->dim_sorted[i] * sizeof(qnumber));
		memcpy(r->axis_qnums_sorted[i], t->axis_qnums_sorted[perm[i]], r->dim_sorted[i] * sizeof(qnumber));
	}

	// allocate (still empty) dense tensor blocks
	const ct_long max_nblocks = integer_product(t->dim_sorted, t->ndim);
	r->block_table = ct_calloc(max_nblocks, sizeof(struct dense_tensor*));
	ct_long* block_map = ct_calloc(max_nblocks, sizeof(ct_long));
	ct_long* index_block_r = ct_malloc(r->ndim * sizeof(ct_long));
	#ifndef NDEBUG
	ct_long nelem_r = 0;
	#endif
	for (ct_long k = 0; k < t->nblocks; k++)
	{
		const struct dense_tensor* bt = t->block_list[k];
		assert(bt->ndim == t->ndim);
		assert(bt->dtype == t->dtype);

		const ct_long* index_block_t = &t->table_index_list[k * t->ndim];

		// corresponding block index in 'r'
		for (int i = 0; i < t->ndim; i++) {
			index_block_r[i] = index_block_t[perm[i]];
		}
		const ct_long j = tensor_index_to_offset(r->ndim, r->dim_sorted, index_block_r);
		block_map[j] = k;

		// create new empty block
		struct dense_tensor* br = ct_calloc(1, sizeof(struct dense_tensor));
		// dimensions of new block
		ct_long* br_dim = ct_malloc(bt->ndim * sizeof(ct_long));
		for (int i = 0; i < bt->ndim; i++) {
			br_dim[i] = bt->dim[perm[i]];
		}
		allocate_empty_dense_tensor(bt->dtype, bt->ndim, br_dim, br);
		ct_free(br_dim);

		r->block_table[j] = br;

		#ifndef NDEBUG
		nelem_r += dense_tensor_num_elements(r->block_table[j]);
		#endif
	}
	ct_free(index_block_r);
	#ifndef NDEBUG
	assert(nelem_r == r->nelem);
	#endif

	// block list and table indices
	r->block_list = ct_malloc(r->nblocks * sizeof(struct dense_tensor*));
	r->table_index_list = ct_malloc(r->nblocks * r->ndim * sizeof(ct_long));
	block_sparse_tensor_fill_block_list_indices(r);

	const size_t dtype_size = sizeof_numeric_type(r->dtype);

	// contiguous array for block entries
	r->data = ct_malloc(r->nelem * dtype_size);

	// set dense block data pointers
	// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
	int8_t* pentries = (int8_t*)r->data;
	for (ct_long j = 0; j < r->nblocks; j++)
	{
		// set data pointer
		r->block_list[j]->data = pentries;
		const size_t nbytes = dense_tensor_num_elements(r->block_list[j]) * dtype_size;
		pentries += nbytes;
	}

	// transpose the dense blocks
	#pragma omp parallel for schedule(dynamic)
	for (ct_long j = 0; j < max_nblocks; j++)  // have to iterate over table for correct 'block_map' index
	{
		struct dense_tensor* br = r->block_table[j];
		if (br == NULL) {
			continue;
		}
		dense_tensor_transpose_fill(perm, t->block_list[block_map[j]], br);
	}

	ct_free(block_map);
}


//________________________________________________________________________________________________________________________
///
/// \brief Generalized conjugate transpose of a tensor 't' such that
/// the i-th axis in the output tensor 'r' is the perm[i]-th axis of the input tensor 't'.
///
/// Memory will be allocated for 'r'.
///
void block_sparse_tensor_conjugate_transpose(const int* restrict perm, const struct block_sparse_tensor* restrict t, struct block_sparse_tensor* restrict r)
{
	block_sparse_tensor_transpose(perm, t, r);
	conjugate_block_sparse_tensor(r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct the maps from a logical index to the corresponding dense block and entry index along an axis.
///
static void construct_logical_to_block_index_maps(const qnumber* qnums_logical, const ct_long dim, const qnumber* unique_qnums, const ct_long num_qnums, ct_long* index_map_block, ct_long* index_map_block_entry)
{
	ct_long* counter = ct_calloc(num_qnums, sizeof(ct_long));

	for (ct_long i = 0; i < dim; i++)
	{
		#ifndef NDEBUG
		bool found = false;
		#endif
		for (ct_long j = 0; j < num_qnums; j++)
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
/// \brief Construct the map from a dense block entry index to the corresponding logical index along an axis.
///
static inline void construct_block_to_logical_index_map(const qnumber* qnums_logical, const ct_long dim, const qnumber qnum, const ct_long qnum_multiplicity, ct_long* index_map_logical_entry)
{
	// suppress unused parameter warning
	#ifdef NDEBUG
	(void)qnum_multiplicity;
	#endif

	ct_long c = 0;
	for (ct_long j = 0; j < dim; j++)
	{
		if (qnums_logical[j] == qnum)
		{
			index_map_logical_entry[c] = j;
			c++;
		}
	}
	assert(c == qnum_multiplicity);
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
		ct_long* r_dim_logical = ct_malloc((t->ndim - 1) * sizeof(ct_long));
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
		for (ct_long j = 0; j < t->dim_logical[i_ax]; j++)
		{
			for (ct_long k = 0; k < t->dim_logical[i_ax + 1]; k++)
			{
				r_qnums_ax_flat[j*t->dim_logical[i_ax + 1] + k] =
					new_axis_dir * (t->axis_dir[i_ax    ] * t->axis_qnums_logical[i_ax    ][j] +
					                t->axis_dir[i_ax + 1] * t->axis_qnums_logical[i_ax + 1][k]);
			}
		}
		qnumber** r_qnums_logical = ct_malloc((t->ndim - 1) * sizeof(qnumber*));
		for (int i = 0; i < i_ax; i++)
		{
			// simply copy the pointer
			r_qnums_logical[i] = t->axis_qnums_logical[i];
		}
		r_qnums_logical[i_ax] = r_qnums_ax_flat;
		for (int i = i_ax + 2; i < t->ndim; i++)
		{
			// simply copy the pointer
			r_qnums_logical[i - 1] = t->axis_qnums_logical[i];
		}

		// allocate new block-sparse tensor 'r'
		allocate_block_sparse_tensor(t->dtype, t->ndim - 1, r_dim_logical, r_axis_dir, (const qnumber**)r_qnums_logical, r);

		ct_free(r_qnums_logical);
		ct_free(r_qnums_ax_flat);
		ct_free(r_axis_dir);
		ct_free(r_dim_logical);
	}

	const size_t dtype_size = sizeof_numeric_type(t->dtype);

	// fan-in logical to block indices for flattened axis
	ct_long* index_map_fanin = ct_malloc(r->dim_logical[i_ax] * sizeof(ct_long));
	{
		ct_long* index_map_block = ct_malloc(r->dim_logical[i_ax] * sizeof(ct_long));
		construct_logical_to_block_index_maps(
			r->axis_qnums_logical[i_ax], r->dim_logical[i_ax], r->axis_qnums_sorted[i_ax], r->dim_sorted[i_ax],
			index_map_block, index_map_fanin);
		ct_free(index_map_block);
	}

	// for each block...
	#pragma omp parallel for schedule(dynamic)
	for (ct_long k = 0; k < t->nblocks; k++)
	{
		const struct dense_tensor* bt = t->block_list[k];
		assert(bt->ndim == t->ndim);
		assert(bt->dtype == t->dtype);

		const ct_long* index_block_t = &t->table_index_list[k * t->ndim];

		const qnumber qnums_i_ax[2] = {
			t->axis_qnums_sorted[i_ax    ][index_block_t[i_ax    ]],
			t->axis_qnums_sorted[i_ax + 1][index_block_t[i_ax + 1]] };

		const qnumber qnum_flat =
			new_axis_dir * (t->axis_dir[i_ax    ] * qnums_i_ax[0] +
			                t->axis_dir[i_ax + 1] * qnums_i_ax[1]);

		// corresponding block index in 'r'
		ct_long* index_block_r = ct_malloc(r->ndim * sizeof(ct_long));
		for (int i = 0; i < i_ax; i++) {
			index_block_r[i] = index_block_t[i];
		}
		index_block_r[i_ax] = -1;
		for (ct_long j = 0; j < r->dim_sorted[i_ax]; j++)
		{
			if (r->axis_qnums_sorted[i_ax][j] == qnum_flat) {
				index_block_r[i_ax] = j;
				break;
			}
		}
		assert(index_block_r[i_ax] != -1);
		for (int i = i_ax + 2; i < t->ndim; i++)
		{
			index_block_r[i - 1] = index_block_t[i];
		}
		struct dense_tensor* br = r->block_table[tensor_index_to_offset(r->ndim, r->dim_sorted, index_block_r)];
		assert(br != NULL);
		assert(br->ndim == r->ndim);
		assert(br->dtype == r->dtype);

		// construct index map for dense block entries along to-be flattened dimensions
		ct_long* index_map_block = ct_malloc(bt->dim[i_ax] * bt->dim[i_ax + 1] * sizeof(ct_long));
		{
			// fan-out dense to logical indices for original axes 'i_ax' and 'i_ax + 1'
			ct_long* index_map_fanout[2];
			for (int i = 0; i < 2; i++)
			{
				index_map_fanout[i] = ct_calloc(bt->dim[i_ax + i], sizeof(ct_long));
				construct_block_to_logical_index_map(
					t->axis_qnums_logical[i_ax + i], t->dim_logical[i_ax + i],
					qnums_i_ax[i], bt->dim[i_ax + i],
					index_map_fanout[i]);
			}
			for (ct_long j0 = 0; j0 < bt->dim[i_ax]; j0++)
			{
				for (ct_long j1 = 0; j1 < bt->dim[i_ax + 1]; j1++)
				{
					index_map_block[j0*bt->dim[i_ax + 1] + j1] = index_map_fanin[index_map_fanout[0][j0] * t->dim_logical[i_ax + 1] + index_map_fanout[1][j1]];
				}
			}
			for (int i = 0; i < 2; i++) {
				ct_free(index_map_fanout[i]);
			}
		}

		// copy block tensor entries
		const ct_long nslices = integer_product(bt->dim, i_ax + 2);
		ct_long* index_slice_bt = ct_calloc(i_ax + 2, sizeof(ct_long));
		ct_long* index_slice_br = ct_calloc(i_ax + 1, sizeof(ct_long));
		const ct_long stride = integer_product(bt->dim + (i_ax + 2), bt->ndim - (i_ax + 2));
		assert(stride == integer_product(br->dim + (i_ax + 1), br->ndim - (i_ax + 1)));
		for (ct_long j = 0; j < nslices; j++, next_tensor_index(i_ax + 2, bt->dim, index_slice_bt))
		{
			for (int i = 0; i < i_ax; i++) {
				index_slice_br[i] = index_slice_bt[i];
			}
			index_slice_br[i_ax] = index_map_block[index_slice_bt[i_ax]*bt->dim[i_ax + 1] + index_slice_bt[i_ax + 1]];
			const ct_long l = tensor_index_to_offset(i_ax + 1, br->dim, index_slice_br);
			// copy slice of entries
			// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
			memcpy((int8_t*)br->data + (l*stride) * dtype_size, (int8_t*)bt->data + (j*stride) * dtype_size, stride * dtype_size);
		}
		ct_free(index_slice_br);
		ct_free(index_slice_bt);

		ct_free(index_map_block);
		ct_free(index_block_r);
	}

	ct_free(index_map_fanin);
}


//________________________________________________________________________________________________________________________
///
/// \brief Split the axis (tensor leg) 'i_ax' into two neighboring axes, using the provided quantum numbers.
///
/// Memory will be allocated for 'r'.
///
/// Note: this operation changes the internal dense block structure.
///
void block_sparse_tensor_split_axis(const struct block_sparse_tensor* restrict t, const int i_ax, const ct_long new_dim_logical[2], const enum tensor_axis_direction new_axis_dir[2], const qnumber* new_qnums_logical[2], struct block_sparse_tensor* restrict r)
{
	assert(0 <= i_ax && i_ax < t->ndim);
	assert(new_dim_logical[0] * new_dim_logical[1] == t->dim_logical[i_ax]);
	// consistency check of provided quantum numbers
	for (ct_long j = 0; j < new_dim_logical[0]; j++) {
		for (ct_long k = 0; k < new_dim_logical[1]; k++) {
			assert(t->axis_dir[i_ax] * t->axis_qnums_logical[i_ax][j*new_dim_logical[1] + k]
				== new_axis_dir[0] * new_qnums_logical[0][j] + new_axis_dir[1] * new_qnums_logical[1][k]);
		}
	}

	// construct new block-sparse tensor 'r'
	{
		ct_long* r_dim_logical = ct_malloc((t->ndim + 1) * sizeof(ct_long));
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
			r_qnums_logical[i] = t->axis_qnums_logical[i];
		}
		r_qnums_logical[i_ax    ] = new_qnums_logical[0];
		r_qnums_logical[i_ax + 1] = new_qnums_logical[1];
		for (int i = i_ax + 1; i < t->ndim; i++)
		{
			// simply copy the pointer
			r_qnums_logical[i + 1] = t->axis_qnums_logical[i];
		}

		// allocate new block-sparse tensor 'r'
		allocate_block_sparse_tensor(t->dtype, t->ndim + 1, r_dim_logical, r_axis_dir, r_qnums_logical, r);

		ct_free(r_qnums_logical);
		ct_free(r_axis_dir);
		ct_free(r_dim_logical);
	}

	const size_t dtype_size = sizeof_numeric_type(t->dtype);

	// fan-in logical to block indices for original axis
	ct_long* index_map_fanin = ct_malloc(t->dim_logical[i_ax] * sizeof(ct_long));
	{
		ct_long* index_map_block = ct_malloc(t->dim_logical[i_ax] * sizeof(ct_long));
		construct_logical_to_block_index_maps(
			t->axis_qnums_logical[i_ax], t->dim_logical[i_ax], t->axis_qnums_sorted[i_ax], t->dim_sorted[i_ax],
			index_map_block, index_map_fanin);
		ct_free(index_map_block);
	}

	// for each block...
	#pragma omp parallel for schedule(dynamic)
	for (ct_long k = 0; k < r->nblocks; k++)
	{
		const struct dense_tensor* br = r->block_list[k];
		assert(br->ndim == r->ndim);
		assert(br->dtype == r->dtype);

		const ct_long* index_block_r = &r->table_index_list[k * r->ndim];

		const qnumber qnums_i_ax[2] = {
			r->axis_qnums_sorted[i_ax    ][index_block_r[i_ax    ]],
			r->axis_qnums_sorted[i_ax + 1][index_block_r[i_ax + 1]] };

		const qnumber qnum_flat =
			t->axis_dir[i_ax] * (r->axis_dir[i_ax    ] * qnums_i_ax[0] +
			                     r->axis_dir[i_ax + 1] * qnums_i_ax[1]);

		// corresponding block index in 't'
		ct_long* index_block_t = ct_malloc(t->ndim * sizeof(ct_long));
		for (int i = 0; i < i_ax; i++) {
			index_block_t[i] = index_block_r[i];
		}
		index_block_t[i_ax] = -1;
		for (ct_long j = 0; j < t->dim_sorted[i_ax]; j++)
		{
			if (t->axis_qnums_sorted[i_ax][j] == qnum_flat) {
				index_block_t[i_ax] = j;
				break;
			}
		}
		assert(index_block_t[i_ax] != -1);
		for (int i = i_ax + 1; i < t->ndim; i++)
		{
			index_block_t[i] = index_block_r[i + 1];
		}
		const struct dense_tensor* bt = t->block_table[tensor_index_to_offset(t->ndim, t->dim_sorted, index_block_t)];
		assert(bt != NULL);
		assert(bt->ndim == t->ndim);
		assert(bt->dtype == t->dtype);

		// construct index map for dense block entries along to-be split dimensions
		ct_long* index_map_block = ct_malloc(br->dim[i_ax] * br->dim[i_ax + 1] * sizeof(ct_long));
		{
			// fan-out dense to logical indices for new axes 'i_ax' and 'i_ax + 1'
			ct_long* index_map_fanout[2];
			for (int i = 0; i < 2; i++)
			{
				index_map_fanout[i] = ct_calloc(br->dim[i_ax + i], sizeof(ct_long));
				construct_block_to_logical_index_map(
					r->axis_qnums_logical[i_ax + i], r->dim_logical[i_ax + i],
					qnums_i_ax[i], br->dim[i_ax + i],
					index_map_fanout[i]);
			}
			for (ct_long j0 = 0; j0 < br->dim[i_ax]; j0++)
			{
				for (ct_long j1 = 0; j1 < br->dim[i_ax + 1]; j1++)
				{
					index_map_block[j0*br->dim[i_ax + 1] + j1] = index_map_fanin[index_map_fanout[0][j0] * r->dim_logical[i_ax + 1] + index_map_fanout[1][j1]];
				}
			}
			for (int i = 0; i < 2; i++) {
				ct_free(index_map_fanout[i]);
			}
		}

		// copy block tensor entries
		const ct_long nslices = integer_product(br->dim, i_ax + 2);
		ct_long* index_slice_bt = ct_calloc(i_ax + 1, sizeof(ct_long));
		ct_long* index_slice_br = ct_calloc(i_ax + 2, sizeof(ct_long));
		const ct_long stride = integer_product(br->dim + (i_ax + 2), br->ndim - (i_ax + 2));
		assert(stride == integer_product(bt->dim + (i_ax + 1), bt->ndim - (i_ax + 1)));
		for (ct_long j = 0; j < nslices; j++, next_tensor_index(i_ax + 2, br->dim, index_slice_br))
		{
			for (int i = 0; i < i_ax; i++) {
				index_slice_bt[i] = index_slice_br[i];
			}
			index_slice_bt[i_ax] = index_map_block[index_slice_br[i_ax]*br->dim[i_ax + 1] + index_slice_br[i_ax + 1]];
			const ct_long l = tensor_index_to_offset(i_ax + 1, bt->dim, index_slice_bt);
			// copy slice of entries
			// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
			memcpy((int8_t*)br->data + (j*stride) * dtype_size, (int8_t*)bt->data + (l*stride) * dtype_size, stride * dtype_size);
		}
		ct_free(index_slice_bt);
		ct_free(index_slice_br);

		ct_free(index_map_block);
		ct_free(index_block_t);
	}

	ct_free(index_map_fanin);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a block-sparse tensor matricization information object (free memory).
///
void delete_block_sparse_tensor_axis_matricization_info(struct block_sparse_tensor_axis_matricization_info* info)
{
	for (int i = 0; i < info->ndim - 2; i++)
	{
		ct_free(info->records[i].qnums_logical[0]);
		ct_free(info->records[i].qnums_logical[1]);
	}
	ct_free(info->records);
}


//________________________________________________________________________________________________________________________
///
/// \brief Matricize a tensor by isolating the axis 'i_ax_tns' and combining (flattening) the remaining axes.
/// 'i_ax_mat' specifies whether 'i_ax_tns' is the leading or trailing dimension of the new matrix.
/// 'flattened_axes_dir' is the axis direction of the combined (flattened) axes.
/// 'info' will be filled and can be used to undo the matricization.
///
void block_sparse_tensor_matricize_axis(const struct block_sparse_tensor* restrict t, const int i_ax_tns,
	const int i_ax_mat, const enum tensor_axis_direction flattened_axes_dir,
	struct block_sparse_tensor* restrict mat, struct block_sparse_tensor_axis_matricization_info* info)
{
	assert(0 <= i_ax_tns && i_ax_tns < t->ndim);
	assert(0 <= i_ax_mat && i_ax_mat < 2);
	assert(t->ndim >= 2);

	// make 'i_ax_tns' the leading or trailing axis
	int* perm = ct_malloc(t->ndim * sizeof(int));
	if (i_ax_mat == 0)
	{
		perm[0] = i_ax_tns;
		for (int i = 1; i <= i_ax_tns; i++) {
			perm[i] = i - 1;
		}
		for (int i = i_ax_tns + 1; i < t->ndim; i++) {
			perm[i] = i;
		}
	}
	else
	{
		for (int i = 0; i < i_ax_tns; i++) {
			perm[i] = i;
		}
		for (int i = i_ax_tns; i < t->ndim - 1; i++) {
			perm[i] = i + 1;
		}
		perm[t->ndim - 1] = i_ax_tns;
	}
	if (is_identity_permutation(perm, t->ndim))
	{
		// for consistency, output tensor must be a copy even if 't' is already in the correct matrix form
		copy_block_sparse_tensor(t, mat);
	}
	else
	{
		block_sparse_tensor_transpose(perm, t, mat);
	}
	ct_free(perm);

	info->i_ax_tns = i_ax_tns;
	info->i_ax_mat = i_ax_mat;
	info->ndim = t->ndim;
	if (t->ndim > 2) {
		info->records = ct_malloc((t->ndim - 2) * sizeof(struct block_sparse_tensor_flatten_axes_record));
	}
	else {
		info->records = NULL;
	}

	const int i_ax_flatten = 1 - i_ax_mat;

	// sequentially flatten all remaining axes
	for (int i = 0; i < t->ndim - 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			const ct_long dim_logical = mat->dim_logical[i_ax_flatten + j];
			info->records[i].dim_logical[j] = dim_logical;
			info->records[i].axis_dir[j] = mat->axis_dir[i_ax_flatten + j];
			info->records[i].qnums_logical[j] = ct_malloc(dim_logical * sizeof(qnumber));
			memcpy(info->records[i].qnums_logical[j], mat->axis_qnums_logical[i_ax_flatten + j], dim_logical * sizeof(qnumber));
		}

		struct block_sparse_tensor tmp = *mat;  // copy internal data pointers
		block_sparse_tensor_flatten_axes(&tmp, i_ax_flatten, flattened_axes_dir, mat);
		delete_block_sparse_tensor(&tmp);
	}

	assert(mat->ndim == 2);
}


//________________________________________________________________________________________________________________________
///
/// \brief Undo the axis matricization of a tensor.
///
void block_sparse_tensor_dematricize_axis(const struct block_sparse_tensor* restrict mat, const struct block_sparse_tensor_axis_matricization_info* info, struct block_sparse_tensor* restrict t)
{
	assert(mat->ndim == 2);
	assert(info->ndim >= 2);

	const int i_ax_flatten = 1 - info->i_ax_mat;

	// for consistency, output tensor must be a copy even if 'mat' is already in the correct tensor form
	copy_block_sparse_tensor(mat, t);

	// sequentially split the other axes
	for (int i = info->ndim - 3; i >= 0; i--)
	{
		struct block_sparse_tensor tmp = *t;  // copy internal data pointers
		block_sparse_tensor_split_axis(&tmp, i_ax_flatten, info->records[i].dim_logical, info->records[i].axis_dir, (const qnumber**)info->records[i].qnums_logical, t);
		delete_block_sparse_tensor(&tmp);
	}
	assert(t->ndim == info->ndim);

	// transpose isolated axis to 'i_ax_tns' in the output tensor
	int* perm = ct_malloc(info->ndim * sizeof(int));
	if (info->i_ax_mat == 0)
	{
		for (int i = 0; i < info->i_ax_tns; i++) {
			perm[i] = i + 1;
		}
		perm[info->i_ax_tns] = 0;
		for (int i = info->i_ax_tns + 1; i < info->ndim; i++) {
			perm[i] = i;
		}
	}
	else
	{
		for (int i = 0; i < info->i_ax_tns; i++) {
			perm[i] = i;
		}
		perm[info->i_ax_tns] = info->ndim - 1;
		for (int i = info->i_ax_tns + 1; i < info->ndim; i++) {
			perm[i] = i - 1;
		}
	}
	if (!is_identity_permutation(perm, info->ndim))
	{
		struct block_sparse_tensor tmp = *t;  // copy internal data pointers
		block_sparse_tensor_transpose(perm, &tmp, t);
		delete_block_sparse_tensor(&tmp);
	}
	ct_free(perm);
}


//________________________________________________________________________________________________________________________
///
/// \brief Slice an axis of the tensor by selecting logical indices 'ind' along this axis.
///
/// Indices 'ind' can be duplicate and need not be sorted.
/// Memory will be allocated for 'r'.
///
void block_sparse_tensor_slice(const struct block_sparse_tensor* restrict t, const int i_ax, const ct_long* ind, const ct_long nind, struct block_sparse_tensor* restrict r)
{
	assert(0 <= i_ax && i_ax < t->ndim);
	assert(nind > 0);

	// construct new block-sparse tensor 'r'
	{
		ct_long* r_dim_logical = ct_malloc(t->ndim * sizeof(ct_long));
		memcpy(r_dim_logical, t->dim_logical, t->ndim * sizeof(ct_long));
		r_dim_logical[i_ax] = nind;
		// logical quantum numbers
		qnumber* r_qnums_logical_i_ax = ct_malloc(nind * sizeof(qnumber));
		for (ct_long j = 0; j < nind; j++) {
			assert(0 <= ind[j] && ind[j] < t->dim_logical[i_ax]);
			r_qnums_logical_i_ax[j] = t->axis_qnums_logical[i_ax][ind[j]];
		}
		const qnumber** r_qnums_logical = ct_malloc(t->ndim * sizeof(qnumber*));
		for (int i = 0; i < t->ndim; i++)
		{
			// simply copy the pointer for i != i_ax
			r_qnums_logical[i] = (i == i_ax ? r_qnums_logical_i_ax : t->axis_qnums_logical[i]);
		}

		// allocate new block-sparse tensor 'r'
		allocate_block_sparse_tensor(t->dtype, t->ndim, r_dim_logical, t->axis_dir, r_qnums_logical, r);

		ct_free(r_qnums_logical);
		ct_free(r_qnums_logical_i_ax);
		ct_free(r_dim_logical);
	}

	// for each block...
	#pragma omp parallel for schedule(dynamic)
	for (ct_long k = 0; k < r->nblocks; k++)
	{
		struct dense_tensor* br = r->block_list[k];
		assert(br->ndim == r->ndim);
		assert(br->dtype == r->dtype);

		const ct_long* index_block_r = &r->table_index_list[k * r->ndim];

		const qnumber qnum = r->axis_qnums_sorted[i_ax][index_block_r[i_ax]];

		// find corresponding block index in 't' tensor
		ct_long* index_block_t = ct_malloc(t->ndim * sizeof(ct_long));
		memcpy(index_block_t, index_block_r, t->ndim * sizeof(ct_long));
		index_block_t[i_ax] = -1;
		for (ct_long j = 0; j < t->dim_sorted[i_ax]; j++)
		{
			if (t->axis_qnums_sorted[i_ax][j] == qnum) {
				index_block_t[i_ax] = j;
				break;
			}
		}
		assert(index_block_t[i_ax] != -1);

		const struct dense_tensor* bt = t->block_table[tensor_index_to_offset(t->ndim, t->dim_sorted, index_block_t)];
		assert(bt != NULL);
		assert(bt->ndim == t->ndim);
		assert(bt->dtype == t->dtype);

		// fan-out dense to logical indices for axis 'i_ax' in 'r'
		ct_long* index_map_fanout_r = ct_calloc(br->dim[i_ax], sizeof(ct_long));
		ct_long c = 0;
		for (ct_long j = 0; j < r->dim_logical[i_ax]; j++)
		{
			if (r->axis_qnums_logical[i_ax][j] == qnum)
			{
				index_map_fanout_r[c] = j;
				c++;
			}
		}
		assert(c == br->dim[i_ax]);

		// fan-in logical to block indices for axis i_ax in 't'
		ct_long* index_map_fanin_t = ct_calloc(t->dim_logical[i_ax], sizeof(ct_long));
		c = 0;
		for (ct_long j = 0; j < t->dim_logical[i_ax]; j++)
		{
			if (t->axis_qnums_logical[i_ax][j] == qnum) {
				index_map_fanin_t[j] = c;
				c++;
			}
		}

		// indices for slicing of current block
		ct_long* ind_block = ct_malloc(br->dim[i_ax] * sizeof(ct_long));
		for (ct_long j = 0; j < br->dim[i_ax]; j++)
		{
			assert(t->axis_qnums_logical[i_ax][ind[index_map_fanout_r[j]]] == qnum);
			ind_block[j] = index_map_fanin_t[ind[index_map_fanout_r[j]]];
		}

		// slice block
		dense_tensor_slice_fill(bt, i_ax, ind_block, br->dim[i_ax], br);

		ct_free(ind_block);
		ct_free(index_map_fanin_t);
		ct_free(index_map_fanout_r);
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
		assert(qnumber_all_equal(t->dim_logical[i], t->axis_qnums_logical[i], t->axis_qnums_logical[t->ndim - ndim_trace + i]));
	}

	if (ndim_trace == 0) {
		copy_block_sparse_tensor(t, r);
		return;
	}

	// construct new block-sparse tensor 'r'
	allocate_block_sparse_tensor(t->dtype, t->ndim - 2 * ndim_trace, t->dim_logical + ndim_trace, t->axis_dir + ndim_trace, (const qnumber**)(t->axis_qnums_logical + ndim_trace), r);

	// for each block in 'r'...
	const ct_long max_nblocks_p = integer_product(t->dim_sorted, ndim_trace);
	#pragma omp parallel for schedule(dynamic)
	for (ct_long k = 0; k < r->nblocks; k++)
	{
		struct dense_tensor* br = r->block_list[k];

		const ct_long* index_block_r = &r->table_index_list[k * r->ndim];

		// require zero initialization
		ct_long* index_block_t = ct_calloc(t->ndim, sizeof(ct_long));
		memcpy(index_block_t + ndim_trace, index_block_r, r->ndim * sizeof(ct_long));

		for (ct_long j = 0; j < max_nblocks_p; j++, next_tensor_index(ndim_trace, t->dim_sorted, index_block_t))
		{
			// duplicate leading 'ndim_trace' indices at the end
			memcpy(index_block_t + ndim_trace + r->ndim, index_block_t, ndim_trace * sizeof(ct_long));

			const struct dense_tensor* bt = t->block_table[tensor_index_to_offset(t->ndim, t->dim_sorted, index_block_t)];
			assert(bt != NULL);

			dense_tensor_cyclic_partial_trace_update(bt, ndim_trace, br);
		}

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
		assert(qnumber_all_equal(t->dim_logical[i], s->axis_qnums_logical[i], t->axis_qnums_logical[i]));
	}

	// for each block with matching quantum numbers...
	const ct_long max_nblocks = integer_product(t->dim_sorted, t->ndim);
	#pragma omp parallel for schedule(dynamic)
	for (ct_long k = 0; k < max_nblocks; k++)  // iterate over block tables because block lists might be ordered differently
	{
		const struct dense_tensor* bs = s->block_table[k];
		struct dense_tensor* bt       = t->block_table[k];
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
	allocate_block_sparse_tensor(s->dtype, s->ndim, s->dim_logical, s->axis_dir, (const qnumber**)s->axis_qnums_logical, r);

	// for each block with matching quantum numbers...
	const ct_long max_nblocks = integer_product(r->dim_sorted, r->ndim);
	#pragma omp parallel for schedule(dynamic)
	for (ct_long k = 0; k < max_nblocks; k++)  // iterate over block tables because block lists might be ordered differently
	{
		struct dense_tensor* br = r->block_table[k];
		if (br == NULL) {
			continue;
		}

		// corresponding block in 's'
		const struct dense_tensor* bs = s->block_table[k];
		assert(bs != NULL);

		ct_long* index_block_r = ct_malloc(r->ndim * sizeof(ct_long));
		offset_to_tensor_index(r->ndim, r->dim_sorted, k, index_block_r);

		// fan-out dense to logical indices
		ct_long* index_map = ct_calloc(br->dim[i_ax], sizeof(ct_long));
		ct_long c = 0;
		for (ct_long j = 0; j < r->dim_logical[i_ax]; j++)
		{
			if (r->axis_qnums_logical[i_ax][j] == r->axis_qnums_sorted[i_ax][index_block_r[i_ax]])
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
/// \brief Multiply the 'i_ax_start, ..., i_ax_start + ndim_mult - 1' axes of 's' with the leading or trailing axis of 't', preserving the overall dimension ordering of 's'.
///
void block_sparse_tensor_multiply_axes(const struct block_sparse_tensor* restrict s, const int i_ax_start, const struct block_sparse_tensor* restrict t, const enum tensor_axis_range axrange_t, const int ndim_mult, struct block_sparse_tensor* restrict r)
{
	assert(ndim_mult >= 1);

	const int i_mt = (axrange_t == TENSOR_AXIS_RANGE_LEADING ? 0 : t->ndim - ndim_mult);
	const int offset_t = (axrange_t == TENSOR_AXIS_RANGE_LEADING ? ndim_mult : 0);

	// data types must agree
	assert(s->dtype == t->dtype);
	// 't' must have a degree of at least 'ndim_mult'
	assert(t->ndim >= ndim_mult);
	// to-be contracted axes indices must be valid for 's'
	assert(0 <= i_ax_start && i_ax_start + ndim_mult <= s->ndim);
	// to-be contracted dimensions must match, and axes directions must be reversed
	for (int i = 0; i < ndim_mult; i++)
	{
		assert(s->dim_logical[i_ax_start + i] ==  t->dim_logical[i_mt + i]);
		assert(s->dim_sorted [i_ax_start + i] ==  t->dim_sorted [i_mt + i]);
		assert(s->axis_dir   [i_ax_start + i] == -t->axis_dir   [i_mt + i]);
		// quantum numbers must match entrywise
		assert(qnumber_all_equal(s->dim_logical[i_ax_start + i], s->axis_qnums_logical[i_ax_start + i], t->axis_qnums_logical[i_mt + i]));
	}

	// allocate new block-sparse tensor 'r'
	{
		const int ndimr = s->ndim + t->ndim - 2*ndim_mult;

		// logical dimensions of new tensor 'r'
		ct_long* r_dim_logical = ct_malloc(ndimr * sizeof(ct_long));
		memcpy( r_dim_logical, s->dim_logical, i_ax_start * sizeof(ct_long));
		memcpy(&r_dim_logical[i_ax_start], &t->dim_logical[offset_t], (t->ndim - ndim_mult) * sizeof(ct_long));
		memcpy(&r_dim_logical[i_ax_start + t->ndim - ndim_mult], &s->dim_logical[i_ax_start + ndim_mult], (s->ndim - i_ax_start - ndim_mult) * sizeof(ct_long));
		// axis directions of new tensor 'r'
		enum tensor_axis_direction* r_axis_dir = ct_malloc(ndimr * sizeof(enum tensor_axis_direction));
		memcpy( r_axis_dir, s->axis_dir, i_ax_start * sizeof(enum tensor_axis_direction));
		memcpy(&r_axis_dir[i_ax_start], &t->axis_dir[offset_t], (t->ndim - ndim_mult) * sizeof(enum tensor_axis_direction));
		memcpy(&r_axis_dir[i_ax_start + t->ndim - ndim_mult], &s->axis_dir[i_ax_start + ndim_mult], (s->ndim - i_ax_start - ndim_mult) * sizeof(enum tensor_axis_direction));
		// logical quantum numbers along each dimension
		const qnumber** r_qnums_logical = ct_malloc(ndimr * sizeof(qnumber*));
		// simply copy the pointers
		memcpy( r_qnums_logical, s->axis_qnums_logical, i_ax_start * sizeof(qnumber*));
		memcpy(&r_qnums_logical[i_ax_start], &t->axis_qnums_logical[offset_t], (t->ndim - ndim_mult) * sizeof(qnumber*));
		memcpy(&r_qnums_logical[i_ax_start + t->ndim - ndim_mult], &s->axis_qnums_logical[i_ax_start + ndim_mult], (s->ndim - i_ax_start - ndim_mult) * sizeof(qnumber*));
		// create new tensor 'r'
		allocate_block_sparse_tensor(s->dtype, ndimr, r_dim_logical, r_axis_dir, r_qnums_logical, r);
		ct_free(r_qnums_logical);
		ct_free(r_axis_dir);
		ct_free(r_dim_logical);
	}

	const void* one = numeric_one(s->dtype);

	// for each dense block of 'r'...
	#pragma omp parallel for schedule(dynamic)
	for (ct_long k = 0; k < r->nblocks; k++)
	{
		struct dense_tensor* br = r->block_list[k];
		assert(br->ndim == r->ndim);

		const ct_long* index_block_r = &r->table_index_list[k * r->ndim];

		ct_long* index_block_s = ct_malloc(s->ndim * sizeof(ct_long));
		ct_long* index_block_t = ct_malloc(t->ndim * sizeof(ct_long));
		memcpy( index_block_s, index_block_r, i_ax_start * sizeof(ct_long));
		memcpy(&index_block_t[offset_t], &index_block_r[i_ax_start], (t->ndim - ndim_mult) * sizeof(ct_long));
		memcpy(&index_block_s[i_ax_start + ndim_mult], &index_block_r[i_ax_start + t->ndim - ndim_mult], (s->ndim - i_ax_start - ndim_mult) * sizeof(ct_long));

		// for each quantum number combination of the to-be contracted axes...
		const ct_long ncontract = integer_product(t->dim_sorted + i_mt, ndim_mult);
		ct_long* index_contract = ct_calloc(ndim_mult, sizeof(ct_long));
		for (ct_long m = 0; m < ncontract; m++, next_tensor_index(ndim_mult, t->dim_sorted + i_mt, index_contract))
		{
			memcpy(&index_block_s[i_ax_start], index_contract, ndim_mult * sizeof(ct_long));
			memcpy(&index_block_t[i_mt], index_contract, ndim_mult * sizeof(ct_long));

			// probe whether quantum numbers in 's' sum to zero
			qnumber qsum = 0;
			for (int i = 0; i < s->ndim; i++)
			{
				qsum += s->axis_dir[i] * s->axis_qnums_sorted[i][index_block_s[i]];
			}
			if (qsum != 0) {
				continue;
			}

			// quantum numbers in 't' must now also sum to zero
			#ifndef NDEBUG
			qsum = 0;
			for (int i = 0; i < t->ndim; i++)
			{
				qsum += t->axis_dir[i] * t->axis_qnums_sorted[i][index_block_t[i]];
			}
			assert(qsum == 0);
			#endif

			const struct dense_tensor* bs = s->block_table[tensor_index_to_offset(s->ndim, s->dim_sorted, index_block_s)];
			const struct dense_tensor* bt = t->block_table[tensor_index_to_offset(t->ndim, t->dim_sorted, index_block_t)];
			assert(bs != NULL);
			assert(bt != NULL);

			// actually multiply dense tensor blocks and add result to 'br'
			dense_tensor_multiply_axes_update(one, bs, i_ax_start, bt, axrange_t, ndim_mult, one, br);
		}

		ct_free(index_contract);
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
		assert(s->dim_sorted [shift_s + i] ==  t->dim_sorted [shift_t + i]);
		assert(s->axis_dir   [shift_s + i] == -t->axis_dir   [shift_t + i]);
		// quantum numbers must match entrywise
		assert(qnumber_all_equal(t->dim_logical[shift_t + i], s->axis_qnums_logical[shift_s + i], t->axis_qnums_logical[shift_t + i]));
		assert(qnumber_all_equal(t->dim_sorted [shift_t + i], s->axis_qnums_sorted [shift_s + i], t->axis_qnums_sorted [shift_t + i]));
	}

	const void* one = numeric_one(s->dtype);

	if (s->ndim + t->ndim == 2*ndim_mult)
	{
		// special case: resulting tensor has degree 0

		assert(s->ndim == ndim_mult && t->ndim == ndim_mult);

		allocate_block_sparse_tensor(s->dtype, 0, NULL, NULL, NULL, r);
		assert(r->block_table[0] != NULL);

		// for each quantum number combination of the to-be contracted axes...
		const ct_long ncontract = integer_product(s->dim_sorted, ndim_mult);
		ct_long* index_contract = ct_calloc(ndim_mult, sizeof(ct_long));
		for (ct_long m = 0; m < ncontract; m++, next_tensor_index(ndim_mult, s->dim_sorted, index_contract))
		{
			// probe whether quantum numbers in 's' sum to zero
			qnumber qsum = 0;
			for (int i = 0; i < s->ndim; i++)
			{
				qsum += s->axis_dir[i] * s->axis_qnums_sorted[i][index_contract[i]];
			}
			if (qsum != 0) {
				continue;
			}

			const ct_long ib = tensor_index_to_offset(s->ndim, s->dim_sorted, index_contract);
			const struct dense_tensor* bs = s->block_table[ib];
			const struct dense_tensor* bt = t->block_table[ib];
			assert(bs != NULL);
			assert(bt != NULL);

			// actually multiply dense tensor blocks and add result to block in 'r'
			dense_tensor_dot_update(one, bs, axrange_s, bt, axrange_t, ndim_mult, one, r->block_table[0]);
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
		ct_long* r_dim_logical = ct_malloc(ndimr * sizeof(ct_long));
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
			r_qnums_logical[i] = s->axis_qnums_logical[offset_s + i];
		}
		for (int i = 0; i < t->ndim - ndim_mult; i++)
		{
			// simply copy the pointer
			r_qnums_logical[(s->ndim - ndim_mult) + i] = t->axis_qnums_logical[offset_t + i];
		}
		// create new tensor 'r'
		allocate_block_sparse_tensor(s->dtype, ndimr, r_dim_logical, r_axis_dir, r_qnums_logical, r);
		ct_free(r_qnums_logical);
		ct_free(r_axis_dir);
		ct_free(r_dim_logical);
	}

	// for each dense block of 'r'...
	#pragma omp parallel for schedule(dynamic)
	for (ct_long k = 0; k < r->nblocks; k++)
	{
		struct dense_tensor* br = r->block_list[k];
		assert(br->ndim == r->ndim);

		const ct_long* index_block_r = &r->table_index_list[k * r->ndim];
		ct_long* index_block_s = ct_malloc(s->ndim * sizeof(ct_long));
		ct_long* index_block_t = ct_malloc(t->ndim * sizeof(ct_long));

		// for each quantum number combination of the to-be contracted axes...
		const ct_long ncontract = integer_product(t->dim_sorted + shift_t, ndim_mult);
		ct_long* index_contract = ct_calloc(ndim_mult, sizeof(ct_long));
		for (ct_long m = 0; m < ncontract; m++, next_tensor_index(ndim_mult, t->dim_sorted + shift_t, index_contract))
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
				qsum += s->axis_dir[i] * s->axis_qnums_sorted[i][index_block_s[i]];
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
				qsum += t->axis_dir[i] * t->axis_qnums_sorted[i][index_block_t[i]];
			}
			assert(qsum == 0);
			#endif

			const struct dense_tensor* bs = s->block_table[tensor_index_to_offset(s->ndim, s->dim_sorted, index_block_s)];
			const struct dense_tensor* bt = t->block_table[tensor_index_to_offset(t->ndim, t->dim_sorted, index_block_t)];
			assert(bs != NULL);
			assert(bt != NULL);

			// actually multiply dense tensor blocks and add result to 'br'
			dense_tensor_dot_update(one, bs, axrange_s, bt, axrange_t, ndim_mult, one, br);
		}

		ct_free(index_contract);
		ct_free(index_block_t);
		ct_free(index_block_s);
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
				assert(qnumber_all_equal(tlist[j].dim_logical[i], tlist[j].axis_qnums_logical[i], tlist[j + 1].axis_qnums_logical[i]));
			}
		}
	}
	#endif

	// allocate new block-sparse tensor 'r'
	{
		// dimensions of new tensor
		ct_long dim_concat = 0;
		for (int j = 0; j < num_tensors; j++) {
			dim_concat += tlist[j].dim_logical[i_ax];
		}
		ct_long* r_dim_logical = ct_malloc(ndim * sizeof(ct_long));
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
		ct_long c = 0;
		for (int j = 0; j < num_tensors; j++)
		{
			memcpy(&r_qnums_logical_i_ax[c], tlist[j].axis_qnums_logical[i_ax], tlist[j].dim_logical[i_ax] * sizeof(qnumber));
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
				r_qnums_logical[i] = tlist[0].axis_qnums_logical[i];
			}
		}

		allocate_block_sparse_tensor(tlist[0].dtype, ndim, r_dim_logical, tlist[0].axis_dir, r_qnums_logical, r);

		ct_free(r_qnums_logical);
		ct_free(r_qnums_logical_i_ax);
		ct_free(r_dim_logical);
	}

	// for each dense block of 'r'...
	#pragma omp parallel for schedule(dynamic)
	for (ct_long kr = 0; kr < r->nblocks; kr++)
	{
		struct dense_tensor* br = r->block_list[kr];
		const ct_long* index_block_r = &r->table_index_list[kr * r->ndim];

		struct dense_tensor* tlist_blocks = ct_malloc(num_tensors * sizeof(struct dense_tensor));

		// collect the input tensors containing the current quantum number of axis 'i_ax'
		const qnumber qnum_i_ax = r->axis_qnums_sorted[i_ax][index_block_r[i_ax]];
		// indices for other axes must be identical (since quantum numbers agree)
		ct_long* index_block_t = ct_malloc(r->ndim * sizeof(ct_long));
		memcpy(index_block_t, index_block_r, r->ndim * sizeof(ct_long));
		int num_tlist_blocks = 0;
		for (int j = 0; j < num_tensors; j++)
		{
			bool found = false;
			for (ct_long l = 0; l < tlist[j].dim_sorted[i_ax]; l++) {
				if (tlist[j].axis_qnums_sorted[i_ax][l] == qnum_i_ax) {
					index_block_t[i_ax] = l;
					found = true;
					break;
				}
			}
			if (!found) {
				continue;
			}

			const ct_long kt = tensor_index_to_offset(tlist[j].ndim, tlist[j].dim_sorted, index_block_t);
			const struct dense_tensor* bt = tlist[j].block_table[kt];
			assert(bt != NULL);

			// copy pointers, not actual data
			memcpy(&tlist_blocks[num_tlist_blocks], bt, sizeof(struct dense_tensor));
			num_tlist_blocks++;
		}
		assert(1 <= num_tlist_blocks && num_tlist_blocks <= num_tensors);

		dense_tensor_concatenate_fill(tlist_blocks, num_tlist_blocks, i_ax, br);

		ct_free(tlist_blocks);
		ct_free(index_block_t);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Count the number of occurrences of each unique quantum number.
///
static void count_quantum_numbers(const qnumber* qnums_logical, const ct_long dim, const qnumber* unique_qnums, const ct_long num_qnums, ct_long* qnum_counts)
{
	memset(qnum_counts, 0, num_qnums * sizeof(ct_long));

	for (ct_long i = 0; i < dim; i++) {
		for (ct_long j = 0; j < num_qnums; j++) {
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
	ct_long** qnum_counts;  //!< number of occurrences of the quantum numbers along each axis
	int ndim;               //!< number of dimensions (degree)
};


//________________________________________________________________________________________________________________________
///
/// \brief Create and fill a block-sparse tensor quantum number counter.
///
static void create_block_sparse_tensor_qnumber_counter(const struct block_sparse_tensor* t, struct block_sparse_tensor_qnumber_counter* qcounter)
{
	qcounter->ndim = t->ndim;

	qcounter->qnum_counts = ct_malloc(t->ndim * sizeof(ct_long*));
	for (int i = 0; i < t->ndim; i++)
	{
		qcounter->qnum_counts[i] = ct_malloc(t->dim_sorted[i] * sizeof(ct_long));
		count_quantum_numbers(
			t->axis_qnums_logical[i], t->dim_logical[i], t->axis_qnums_sorted[i], t->dim_sorted[i],
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
				assert(qnumber_all_equal(tlist[j].dim_logical[i], tlist[j].axis_qnums_logical[i], tlist[j + 1].axis_qnums_logical[i]));
			}
		}
	}
	#endif

	// allocate new block-sparse tensor 'r'
	{
		// dimensions of new tensor
		ct_long* r_dim_logical = ct_malloc(ndim * sizeof(ct_long));
		for (int i = 0; i < ndim; i++)
		{
			if (i_ax_indicator[i])
			{
				ct_long dsum = 0;
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
				ct_long c = 0;
				for (int j = 0; j < num_tensors; j++)
				{
					memcpy(&r_qnums_logical[i][c], tlist[j].axis_qnums_logical[i], tlist[j].dim_logical[i] * sizeof(qnumber));
					c += tlist[j].dim_logical[i];
				}
				assert(c == r_dim_logical[i]);
			}
			else {
				// simply copy the pointer
				r_qnums_logical[i] = tlist[0].axis_qnums_logical[i];
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

	ct_long* zero_list = ct_calloc(r->ndim, sizeof(ct_long));

	// for each dense block of 'r'...
	#pragma omp parallel for schedule(dynamic)
	for (ct_long kr = 0; kr < r->nblocks; kr++)
	{
		struct dense_tensor* br = r->block_list[kr];
		const ct_long* index_block_r = &r->table_index_list[kr * r->ndim];

		// block quantum numbers
		qnumber* qnums_block = ct_malloc(r->ndim * sizeof(qnumber));
		for (int i = 0; i < r->ndim; i++) {
			qnums_block[i] = r->axis_qnums_sorted[i][index_block_r[i]];
		}

		// require zero initialization
		ct_long* offset = ct_calloc(r->ndim, sizeof(ct_long));

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
				memset(offset, 0, r->ndim * sizeof(ct_long));
				num_tlist_blocks++;
			}
			else
			{
				// 'tlist[j]' can contain current block quantum numbers along some of its axes
				for (int i = 0; i < r->ndim; i++) {
					if (i_ax_indicator[i]) {
						for (ct_long k = 0; k < tlist[j].dim_sorted[i]; k++) {
							if (tlist[j].axis_qnums_sorted[i][k] == qnums_block[i]) {
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
			if (must_pad)
			{
				struct dense_tensor tmp;
				dense_tensor_pad_zeros(&tlist_blocks[num_tlist_blocks - 1], zero_list, offset, &tmp);
				delete_dense_tensor(&tlist_blocks[num_tlist_blocks - 1]);
				tlist_blocks[num_tlist_blocks - 1] = tmp;  // copy internal data pointers
			}
		}

		dense_tensor_block_diag_fill(tlist_blocks, num_tlist_blocks, i_ax, ndim_block, br);

		for (int j = 0; j < num_tlist_blocks; j++) {
			delete_dense_tensor(&tlist_blocks[j]);
		}

		ct_free(tlist_blocks);
		ct_free(offset);
		ct_free(qnums_block);
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
int block_sparse_tensor_qr(const struct block_sparse_tensor* restrict a, const enum qr_mode mode, struct block_sparse_tensor* restrict q, struct block_sparse_tensor* restrict r)
{
	// require a matrix
	assert(a->ndim == 2);

	// determine dimensions and quantum numbers of intermediate axis connecting Q and R,
	// and allocate Q and R matrices

	ct_long dim_interm = 0;

	// allocating array of maximum possible length
	qnumber* qnums_interm = ct_calloc(a->dim_logical[0], sizeof(qnumber));

	// loop over second dimension is outer loop to ensure that logical quantum numbers are sorted
	for (ct_long j = 0; j < a->dim_sorted[1]; j++)
	{
		for (ct_long i = 0; i < a->dim_sorted[0]; i++)
		{
			// probe whether quantum numbers sum to zero
			qnumber qsum = a->axis_dir[0] * a->axis_qnums_sorted[0][i] + a->axis_dir[1] * a->axis_qnums_sorted[1][j];
			if (qsum != 0) {
				continue;
			}

			const struct dense_tensor* b = a->block_table[i*a->dim_sorted[1] + j];
			assert(b != NULL);
			assert(b->ndim == 2);
			assert(b->dtype == a->dtype);

			const ct_long k = (mode == QR_REDUCED ? lmin(b->dim[0], b->dim[1]) : b->dim[0]);

			// append a sequence of logical quantum numbers of length k
			for (ct_long l = 0; l < k; l++) {
				qnums_interm[dim_interm + l] = a->axis_qnums_sorted[1][j];
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
		const qnumber qnum_ax0 = a->axis_qnums_sorted[0][0];
		const qnumber qnum_ax1 = -a->axis_dir[0]*a->axis_dir[1] * qnum_ax0;
		qnums_interm[0] = qnum_ax1;
	}

	// allocate the 'q' matrix
	const ct_long dim_q[2] = { a->dim_logical[0], dim_interm };
	const qnumber* qnums_q[2] = { a->axis_qnums_logical[0], qnums_interm };
	allocate_block_sparse_tensor(a->dtype, 2, dim_q, a->axis_dir, qnums_q, q);

	// allocate the 'r' matrix
	const ct_long dim_r[2] = { dim_interm, a->dim_logical[1] };
	const enum tensor_axis_direction axis_dir_r[2] = { -a->axis_dir[1], a->axis_dir[1] };
	const qnumber* qnums_r[2] = { qnums_interm, a->axis_qnums_logical[1] };
	allocate_block_sparse_tensor(a->dtype, 2, dim_r, axis_dir_r, qnums_r, r);

	ct_free(qnums_interm);

	if (create_dummy_qr)
	{
		// make 'q' an isometry with a single non-zero entry 1

		// use first block quantum number in 'a' to create a non-zero entry in 'q'
		const qnumber qnum_ax0 = a->axis_qnums_sorted[0][0];
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
	for (ct_long i = 0; i < a->dim_sorted[0]; i++)
	{
		for (ct_long j = 0; j < a->dim_sorted[1]; j++)
		{
			const struct dense_tensor* ba = a->block_table[i*a->dim_sorted[1] + j];
			if (ba == NULL) {
				continue;
			}
			assert(ba->ndim == 2);
			assert(ba->dtype == a->dtype);
			assert(a->axis_dir[0] * a->axis_qnums_sorted[0][i] + a->axis_dir[1] * a->axis_qnums_sorted[1][j] == 0);

			// find corresponding blocks in 'q' and 'r'
			assert(q->axis_qnums_sorted[0][i] == a->axis_qnums_sorted[0][i]);
			assert(r->axis_qnums_sorted[1][j] == a->axis_qnums_sorted[1][j]);
			// connecting axis contains only the quantum numbers of non-empty blocks in 'a', so cannot use the block index from 'a'
			ct_long k;
			for (k = 0; k < q->dim_sorted[1]; k++) {
				if (q->axis_qnums_sorted[1][k] == a->axis_qnums_sorted[1][j]) {
					break;
				}
			}
			assert(k < q->dim_sorted[1]);
			assert(q->axis_qnums_sorted[1][k] == a->axis_qnums_sorted[1][j]);
			assert(r->axis_qnums_sorted[0][k] == a->axis_qnums_sorted[1][j]);
			struct dense_tensor* bq = q->block_table[i*q->dim_sorted[1] + k];
			struct dense_tensor* br = r->block_table[k*r->dim_sorted[1] + j];
			assert(bq != NULL);
			assert(br != NULL);

			// perform QR decomposition of block
			int ret = dense_tensor_qr_fill(ba, mode, bq, br);
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
int block_sparse_tensor_rq(const struct block_sparse_tensor* restrict a, const enum qr_mode mode, struct block_sparse_tensor* restrict r, struct block_sparse_tensor* restrict q)
{
	// require a matrix
	assert(a->ndim == 2);

	// determine dimensions and quantum numbers of intermediate axis connecting R and Q,
	// and allocate R and Q matrices

	ct_long dim_interm = 0;

	// allocating array of maximum possible length
	qnumber* qnums_interm = ct_calloc(a->dim_logical[1], sizeof(qnumber));

	// loop over first dimension is outer loop to ensure that logical quantum numbers are sorted
	for (ct_long i = 0; i < a->dim_sorted[0]; i++)
	{
		for (ct_long j = 0; j < a->dim_sorted[1]; j++)
		{
			// probe whether quantum numbers sum to zero
			qnumber qsum = a->axis_dir[0] * a->axis_qnums_sorted[0][i] + a->axis_dir[1] * a->axis_qnums_sorted[1][j];
			if (qsum != 0) {
				continue;
			}

			const struct dense_tensor* b = a->block_table[i*a->dim_sorted[1] + j];
			assert(b != NULL);
			assert(b->ndim == 2);
			assert(b->dtype == a->dtype);

			const ct_long k = (mode == QR_REDUCED ? lmin(b->dim[0], b->dim[1]) : b->dim[1]);

			// append a sequence of logical quantum numbers of length k
			for (ct_long l = 0; l < k; l++) {
				qnums_interm[dim_interm + l] = a->axis_qnums_sorted[0][i];
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
		const qnumber qnum_ax1 = a->axis_qnums_sorted[1][0];
		const qnumber qnum_ax0 = -a->axis_dir[0]*a->axis_dir[1] * qnum_ax1;
		qnums_interm[0] = qnum_ax0;
	}

	// allocate the 'r' matrix
	const ct_long dim_r[2] = { a->dim_logical[0], dim_interm };
	const enum tensor_axis_direction axis_dir_r[2] = { a->axis_dir[0], -a->axis_dir[0] };
	const qnumber* qnums_r[2] = { a->axis_qnums_logical[0], qnums_interm };
	allocate_block_sparse_tensor(a->dtype, 2, dim_r, axis_dir_r, qnums_r, r);

	// allocate the 'q' matrix
	const ct_long dim_q[2] = { dim_interm, a->dim_logical[1] };
	const qnumber* qnums_q[2] = { qnums_interm, a->axis_qnums_logical[1] };
	allocate_block_sparse_tensor(a->dtype, 2, dim_q, a->axis_dir, qnums_q, q);

	ct_free(qnums_interm);

	if (create_dummy_qr)
	{
		// make 'q' an isometry with a single non-zero entry 1

		// use first block quantum number in 'a' to create a non-zero entry in 'q'
		const qnumber qnum_ax1 = a->axis_qnums_sorted[1][0];
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
	for (ct_long i = 0; i < a->dim_sorted[0]; i++)
	{
		for (ct_long j = 0; j < a->dim_sorted[1]; j++)
		{
			const struct dense_tensor* ba = a->block_table[i*a->dim_sorted[1] + j];
			if (ba == NULL) {
				continue;
			}
			assert(ba->ndim == 2);
			assert(ba->dtype == a->dtype);
			assert(a->axis_dir[0] * a->axis_qnums_sorted[0][i] + a->axis_dir[1] * a->axis_qnums_sorted[1][j] == 0);

			// find corresponding blocks in 'q' and 'r'
			assert(r->axis_qnums_sorted[0][i] == a->axis_qnums_sorted[0][i]);
			assert(q->axis_qnums_sorted[1][j] == a->axis_qnums_sorted[1][j]);
			// connecting axis contains only the quantum numbers of non-empty blocks in 'a', so cannot use the block index from 'a'
			ct_long k;
			for (k = 0; k < q->dim_sorted[0]; k++) {
				if (q->axis_qnums_sorted[0][k] == a->axis_qnums_sorted[0][i]) {
					break;
				}
			}
			assert(k < q->dim_sorted[0]);
			assert(q->axis_qnums_sorted[0][k] == a->axis_qnums_sorted[0][i]);
			assert(r->axis_qnums_sorted[1][k] == a->axis_qnums_sorted[0][i]);
			struct dense_tensor* br = r->block_table[i*r->dim_sorted[1] + k];
			struct dense_tensor* bq = q->block_table[k*q->dim_sorted[1] + j];
			assert(br != NULL);
			assert(bq != NULL);

			// perform RQ decomposition of block
			int ret = dense_tensor_rq_fill(ba, mode, br, bq);
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
/// \brief Compute the logical singular value decomposition of a block-sparse matrix.
///
/// The logical quantum numbers of the axis connecting 'u' and 'vh' will be sorted.
/// The singular values are returned in a dense vector.
///
int block_sparse_tensor_svd(const struct block_sparse_tensor* restrict a, struct block_sparse_tensor* restrict u, struct dense_tensor* restrict s, struct block_sparse_tensor* restrict vh)
{
	// require a matrix
	assert(a->ndim == 2);

	// determine dimensions and quantum numbers of intermediate axis connecting 'u' and 'vh',
	// and allocate 'u' and 'vh' matrices

	ct_long dim_interm = 0;

	// allocating array of maximum possible length
	qnumber* qnums_interm = ct_calloc(a->dim_logical[1], sizeof(qnumber));

	// loop over second dimension is outer loop to ensure that logical quantum numbers are sorted
	for (ct_long j = 0; j < a->dim_sorted[1]; j++)
	{
		for (ct_long i = 0; i < a->dim_sorted[0]; i++)
		{
			// probe whether quantum numbers sum to zero
			qnumber qsum = a->axis_dir[0] * a->axis_qnums_sorted[0][i] + a->axis_dir[1] * a->axis_qnums_sorted[1][j];
			if (qsum != 0) {
				continue;
			}

			const struct dense_tensor* b = a->block_table[i*a->dim_sorted[1] + j];
			assert(b != NULL);
			assert(b->ndim == 2);
			assert(b->dtype == a->dtype);

			const ct_long k = lmin(b->dim[0], b->dim[1]);

			// append a sequence of logical quantum numbers of length k
			for (ct_long l = 0; l < k; l++) {
				qnums_interm[dim_interm + l] = a->axis_qnums_sorted[1][j];
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
		const qnumber qnum_ax0 = a->axis_qnums_sorted[0][0];
		const qnumber qnum_ax1 = -a->axis_dir[0]*a->axis_dir[1] * qnum_ax0;
		qnums_interm[0] = qnum_ax1;
	}

	// allocate the 'u' matrix
	const ct_long dim_u[2] = { a->dim_logical[0], dim_interm };
	const qnumber* qnums_u[2] = { a->axis_qnums_logical[0], qnums_interm };
	allocate_block_sparse_tensor(a->dtype, 2, dim_u, a->axis_dir, qnums_u, u);

	// allocate the 's' vector
	const ct_long dim_s[1] = { dim_interm };
	allocate_zero_dense_tensor(numeric_real_type(a->dtype), 1, dim_s, s);

	// allocate the 'vh' matrix
	const ct_long dim_vh[2] = { dim_interm, a->dim_logical[1] };
	const enum tensor_axis_direction axis_dir_vh[2] = { -a->axis_dir[1], a->axis_dir[1] };
	const qnumber* qnums_vh[2] = { qnums_interm, a->axis_qnums_logical[1] };
	allocate_block_sparse_tensor(a->dtype, 2, dim_vh, axis_dir_vh, qnums_vh, vh);

	ct_free(qnums_interm);

	if (create_dummy_svd)
	{
		// make 'u' an isometry with a single non-zero entry 1

		// use first block quantum number in 'a' to create a non-zero entry in 'u'
		const qnumber qnum_ax0 = a->axis_qnums_sorted[0][0];
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
	for (ct_long i = 0; i < a->dim_sorted[0]; i++)
	{
		for (ct_long j = 0; j < a->dim_sorted[1]; j++)
		{
			const struct dense_tensor* ba = a->block_table[i*a->dim_sorted[1] + j];
			if (ba == NULL) {
				continue;
			}
			assert(ba->ndim == 2);
			assert(ba->dtype == a->dtype);
			assert(a->axis_dir[0] * a->axis_qnums_sorted[0][i] + a->axis_dir[1] * a->axis_qnums_sorted[1][j] == 0);

			// find corresponding blocks in 'u' and 'vh'
			assert( u->axis_qnums_sorted[0][i] == a->axis_qnums_sorted[0][i]);
			assert(vh->axis_qnums_sorted[1][j] == a->axis_qnums_sorted[1][j]);
			// connecting axis contains only the quantum numbers of non-empty blocks in 'a', so cannot use the block index from 'a'
			ct_long k;
			for (k = 0; k < u->dim_sorted[1]; k++) {
				if (u->axis_qnums_sorted[1][k] == a->axis_qnums_sorted[1][j]) {
					break;
				}
			}
			assert(k < u->dim_sorted[1]);
			assert( u->axis_qnums_sorted[1][k] == a->axis_qnums_sorted[1][j]);
			assert(vh->axis_qnums_sorted[0][k] == a->axis_qnums_sorted[1][j]);
			struct dense_tensor* bu  =  u->block_table[i* u->dim_sorted[1] + k];
			struct dense_tensor* bvh = vh->block_table[k*vh->dim_sorted[1] + j];
			assert(bu  != NULL);
			assert(bvh != NULL);
			assert(bu->dim[1] == bvh->dim[0]);

			struct dense_tensor bs;
			const ct_long dim_bs[1] = { bu->dim[1] };
			allocate_dense_tensor(numeric_real_type(a->dtype), 1, dim_bs, &bs);

			// perform singular value decomposition of block
			int ret = dense_tensor_svd_fill(ba, bu, &bs, bvh);
			if (ret != 0) {
				failed = true;
			}

			// copy entries from 'bs' into output vector 's'
			// find range of logical indices corresponding to current block
			for (k = 0; k < u->dim_logical[1]; k++) {
				if (u->axis_qnums_logical[1][k] == a->axis_qnums_sorted[1][j]) {
					break;
				}
			}
			// expecting continuous range of quantum number 'a->axis_qnums_sorted[1][j]' along connecting axis
			assert(k + bs.dim[0] <= u->dim_logical[1]);
			for (ct_long l = 0; l < bs.dim[0]; l++) {
				assert(u->axis_qnums_logical[1][k + l] == a->axis_qnums_sorted[1][j]);
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
		if (!qnumber_all_equal(s->dim_logical[i], s->axis_qnums_logical[i], t->axis_qnums_logical[i])) {
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
		assert(s->dim_sorted[i] == t->dim_sorted[i]);
		assert(qnumber_all_equal(s->dim_sorted[i], s->axis_qnums_sorted[i], t->axis_qnums_sorted[i]));
	}

	// iterate over block table...
	const ct_long max_nblocks = integer_product(s->dim_sorted, s->ndim);
	for (ct_long k = 0; k < max_nblocks; k++)
	{
		// retrieve and compare blocks;
		const struct dense_tensor* bs = s->block_table[k];
		const struct dense_tensor* bt = t->block_table[k];
		// a dense block can be absent even for matching quantum numbers
		if (bs == NULL)
		{
			if (bt == NULL) {
				continue;
			}
			if (!dense_tensor_is_zero(bt, tol)) {
				return false;
			}
		}
		else if (bt == NULL)
		{
			if (!dense_tensor_is_zero(bs, tol)) {
				return false;
			}
		}
		else
		{
			if (!dense_tensor_allclose(bs, bt, tol)) {
				return false;
			}
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Test whether a dense and a block-sparse tensor agree elementwise within tolerance 'tol'.
///
bool dense_block_sparse_tensor_allclose(const struct dense_tensor* s, const struct block_sparse_tensor* t, const double tol)
{
	struct dense_tensor t_dns;
	block_sparse_to_dense_tensor(t, &t_dns);

	bool is_close = dense_tensor_allclose(s, &t_dns, tol);

	delete_dense_tensor(&t_dns);

	return is_close;
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
	if (!qnumber_all_equal(t->dim_logical[0], t->axis_qnums_logical[0], t->axis_qnums_logical[1])) {
		return false;
	}

	// individual diagonal blocks must be identity matrices
	const ct_long max_nblocks = integer_product(t->dim_sorted, t->ndim);
	ct_long index_block[2];
	for (ct_long k = 0; k < max_nblocks; k++, next_tensor_index(t->ndim, t->dim_sorted, index_block))
	{
		// probe whether quantum numbers sum to zero
		const qnumber qsum =
			  t->axis_dir[0] * t->axis_qnums_sorted[0][index_block[0]]
			+ t->axis_dir[1] * t->axis_qnums_sorted[1][index_block[1]];
		if (qsum != 0)
		{
			assert(t->block_table[k] == NULL);
			continue;
		}

		const struct dense_tensor* b = t->block_table[k];
		if (b == NULL) {
			// missing diagonal block
			return false;
		}
		if (!dense_tensor_is_identity(b, tol)) {
			return false;
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
/// \brief Augment a block-sparse matrix by padding identities for quantum number sectors along the first axis which are absent along the second axis.
///
void block_sparse_tensor_augment_identity_blocks(const struct block_sparse_tensor* restrict t, const bool transpose, struct block_sparse_tensor* restrict ret)
{
	// must be a matrix
	assert(t->ndim == 2);
	assert(t->axis_dir[0] != t->axis_dir[1]);

	const int i_ax = (transpose ? 1 : 0);

	// upper bound on required memory
	qnumber* qnums_aug = ct_malloc(t->dim_logical[i_ax] * sizeof(qnumber));
	ct_long dim_aug = 0;

	for (ct_long i = 0; i < t->dim_sorted[i_ax]; i++)
	{
		const qnumber qnum = t->axis_qnums_sorted[i_ax][i];

		// test whether current quantum number is absent along other axis
		bool absent = true;
		for (ct_long j = 0; j < t->dim_sorted[1 - i_ax]; j++)
		{
			if (t->axis_qnums_sorted[1 - i_ax][j] == qnum) {
				absent = false;
				break;
			}
		}
		if (!absent) {
			continue;
		}

		// multiplicity of quantum number 'qnum'
		ct_long multiplicity = 0;
		for (ct_long j = 0; j < t->dim_logical[i_ax]; j++)
		{
			if (t->axis_qnums_logical[i_ax][j] == qnum) {
				multiplicity++;
			}
		}
		assert(multiplicity > 0);

		// add absent quantum number to augmentation
		for (ct_long j = 0; j < multiplicity; j++)
		{
			qnums_aug[dim_aug] = qnum;
			dim_aug++;
		}
	}
	assert(dim_aug <= t->dim_logical[i_ax]);

	if (dim_aug == 0)
	{
		// augmentation not required
		ct_free(qnums_aug);
		copy_block_sparse_tensor(t, ret);
		return;
	}

	ct_long dim[2];
	dim[i_ax] = t->dim_logical[i_ax];
	dim[1 - i_ax] = dim_aug;

	qnumber* axis_qnums_logical[2];
	axis_qnums_logical[i_ax] = t->axis_qnums_logical[i_ax];  // copy the pointer
	axis_qnums_logical[1 - i_ax] = qnums_aug;

	struct block_sparse_tensor tlist[2];
	tlist[0] = *t;  // copy internal data pointers

	allocate_block_sparse_tensor(t->dtype, 2, dim, t->axis_dir, (const qnumber**)axis_qnums_logical, &tlist[1]);
	block_sparse_tensor_set_identity_blocks(&tlist[1]);

	block_sparse_tensor_concatenate(tlist, 2, 1 - i_ax, ret);

	delete_block_sparse_tensor(&tlist[1]);
	ct_free(qnums_aug);
}


//________________________________________________________________________________________________________________________
///
/// \brief Create a block-sparse tensor element accessor.
///
void create_block_sparse_tensor_entry_accessor(const struct block_sparse_tensor* t, struct block_sparse_tensor_entry_accessor* acc)
{
	acc->tensor = t;

	acc->index_map_blocks        = ct_malloc(t->ndim * sizeof(ct_long*));
	acc->index_map_block_entries = ct_malloc(t->ndim * sizeof(ct_long*));

	for (int i = 0; i < t->ndim; i++)
	{
		acc->index_map_blocks[i]        = ct_malloc(t->dim_logical[i] * sizeof(ct_long));
		acc->index_map_block_entries[i] = ct_malloc(t->dim_logical[i] * sizeof(ct_long));

		construct_logical_to_block_index_maps(
			t->axis_qnums_logical[i], t->dim_logical[i], t->axis_qnums_sorted[i], t->dim_sorted[i],
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
void* block_sparse_tensor_get_entry(const struct block_sparse_tensor_entry_accessor* acc, const ct_long* index)
{
	// retrieve block
	ct_long* index_block = ct_malloc(acc->tensor->ndim * sizeof(ct_long));
	for (int i = 0; i < acc->tensor->ndim; i++)
	{
		assert(0 <= index[i] && index[i] < acc->tensor->dim_logical[i]);
		index_block[i] = acc->index_map_blocks[i][index[i]];
	}
	const ct_long ob = tensor_index_to_offset(acc->tensor->ndim, acc->tensor->dim_sorted, index_block);
	ct_free(index_block);

	const struct dense_tensor* block = acc->tensor->block_table[ob];
	if (block == NULL) {
		// logical entry is zero
		return NULL;
	}

	ct_long* index_entry = ct_malloc(acc->tensor->ndim * sizeof(ct_long));
	for (int i = 0; i < acc->tensor->ndim; i++)
	{
		index_entry[i] = acc->index_map_block_entries[i][index[i]];
	}
	const ct_long oe = tensor_index_to_offset(block->ndim, block->dim, index_entry);
	ct_free(index_entry);

	// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
	return (int8_t*)block->data + oe * sizeof_numeric_type(acc->tensor->dtype);
}
