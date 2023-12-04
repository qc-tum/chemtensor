/// \file dense_tensor.c
/// \brief Dense tensor structure, using row-major storage convention.

#include <memory.h>
#include <cblas.h>
#include "dense_tensor.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Convert tensor index to data offset.
///
static inline long index_to_offset(const int ndim, const long* restrict dim, const long* restrict index)
{
	long offset = 0;
	long dimfac = 1;
	for (int i = ndim - 1; i >= 0; i--)
	{
		offset += dimfac * index[i];
		dimfac *= dim[i];
	}

	return offset;
}

//________________________________________________________________________________________________________________________
///
/// \brief Compute the lexicographically next tensor index.
///
static inline void next_index(const int ndim, const long* restrict dim, long* restrict index)
{
	for (int i = ndim - 1; i >= 0; i--)
	{
		index[i]++;
		if (index[i] < dim[i])
		{
			return;
		}
		else
		{
			index[i] = 0;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for a dense tensor.
///
void allocate_dense_tensor(const int ndim, const long* restrict dim, struct dense_tensor* restrict t)
{
	assert(ndim >= 0);
	t->ndim = ndim;

	if (ndim > 0)
	{
		t->dim = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(long));
		memcpy(t->dim, dim, ndim * sizeof(long));

		const long nelem = dense_tensor_num_elements(t);
		// dimensions must be strictly positive
		assert(nelem > 0);
		t->data = aligned_calloc(MEM_DATA_ALIGN, nelem, sizeof(numeric));
		assert(t->data != NULL);
	}
	else    // ndim == 0
	{
		// aligned_alloc(MEM_DATA_ALIGN, 0) not guaranteed to return NULL
		t->dim = NULL;

		// allocate memory for a single number
		t->data = aligned_calloc(MEM_DATA_ALIGN, 1, sizeof(numeric));
		assert(t->data != NULL);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a dense tensor (free memory).
///
void delete_dense_tensor(struct dense_tensor* t)
{
	if (t->ndim > 0)
	{
		aligned_free(t->dim);
	}
	t->ndim = 0;

	aligned_free(t->data);
	t->data = NULL;
}


//________________________________________________________________________________________________________________________
///
/// \brief Copy a tensor, allocating memory for the copy.
///
void copy_dense_tensor(const struct dense_tensor* restrict src, struct dense_tensor* restrict dst)
{
	allocate_dense_tensor(src->ndim, src->dim, dst);

	const long nelem = dense_tensor_num_elements(src);

	////assume_aligned(src->data);
	////assume_aligned(dst->data);
	memcpy(dst->data, src->data, nelem * sizeof(numeric));
}


//________________________________________________________________________________________________________________________
///
/// \brief Move tensor data (without allocating new memory).
///
void move_dense_tensor_data(struct dense_tensor* restrict src, struct dense_tensor* restrict dst)
{
	dst->ndim = src->ndim;

	// move dimension pointers
	dst->dim = src->dim;
	src->dim = NULL;

	// move data pointers
	dst->data = src->data;
	src->data = NULL;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the 'trace' of a tensor (generalization of the matrix trace); all dimensions of the tensor must agree.
///
numeric dense_tensor_trace(const struct dense_tensor* t)
{
	assert(t->ndim >= 1);
	const long n = t->dim[0];

	// geometric sum: stride = 1 + n + n*n + ...
	long stride = 1;
	long dp = 1;
	for (int i = 1; i < t->ndim; i++)
	{
		assert(t->dim[i] == n);
		dp *= n;
		stride += dp;
	}

	numeric tr = 0;
	for (long j = 0; j < n; j++)
	{
		tr += t->data[j * stride];
	}

	return tr;
}


//________________________________________________________________________________________________________________________
///
/// \brief Scale tensor t by alpha.
///
void scale_dense_tensor(const double alpha, struct dense_tensor* t)
{
	cblas_zdscal(dense_tensor_num_elements(t), alpha, t->data, 1);
}


//________________________________________________________________________________________________________________________
///
/// \brief Reshape dimensions, i.e., interpret as tensor of different dimension with the same number of elements.
///
void reshape_dense_tensor(const int ndim, const long* restrict dim, struct dense_tensor* t)
{
	// consistency check: number of elements must not change
	assert(integer_product(dim, ndim) == dense_tensor_num_elements(t));

	// new dimensions
	t->ndim = ndim;
	aligned_free(t->dim);
	t->dim = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(long));
	memcpy(t->dim, dim, ndim * sizeof(long));
}


//________________________________________________________________________________________________________________________
///
/// \brief Elementwise conjugation of a dense tensor.
///
void conjugate_dense_tensor(struct dense_tensor* t)
{
	const long nelem = dense_tensor_num_elements(t);
	for (long i = 0; i < nelem; i++)
	{
		t->data[i] = conj(t->data[i]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Set the tensor to the identity operator; all dimensions of the tensor must agree.
///
void dense_tensor_set_identity(struct dense_tensor* t)
{
	assert(t->ndim >= 1);
	const long n = t->dim[0];

	// geometric sum: stride = 1 + n + n*n + ...
	long stride = 1;
	long dp = 1;
	for (int i = 1; i < t->ndim; i++)
	{
		assert(t->dim[i] == n);
		dp *= n;
		stride += dp;
	}
	assert(dense_tensor_num_elements(t) == n*dp);

	memset(t->data, 0, n*dp * sizeof(numeric));
	for (long j = 0; j < n; j++)
	{
		t->data[j*stride] = 1;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Generalized transpose of a tensor 't' such that
/// the i-th axis in the output tensor 'r' is the perm[i]-th axis of the input tensor 't'.
///
/// Memory will be allocated for 'r'.
///
void transpose_dense_tensor(const int* restrict perm, const struct dense_tensor* restrict t, struct dense_tensor* restrict r)
{
	// TODO: consider using an optimized library like https://github.com/springer13/hptt

	// dimensions of new tensor 'r'
	long* rdim = aligned_alloc(MEM_DATA_ALIGN, t->ndim * sizeof(long));
	for (int i = 0; i < t->ndim; i++)
	{
		assert(0 <= perm[i] && perm[i] < t->ndim);
		rdim[i] = t->dim[perm[i]];
	}
	// create new tensor 'r'
	allocate_dense_tensor(t->ndim, rdim, r);
	aligned_free(rdim);

	// stride (offset between successive elements) in new tensor 'r' corresponding to original last axis
	int ax_r_last = -1;
	for (int i = 0; i < t->ndim; i++) {
		if (perm[i] == t->ndim - 1) {
			ax_r_last = i;
			break;
		}
	}
	assert(ax_r_last != -1);
	const long stride = integer_product(r->dim + ax_r_last + 1, r->ndim - ax_r_last - 1);

	const long nelem = dense_tensor_num_elements(t);

	long* index_t = aligned_calloc(MEM_DATA_ALIGN, t->ndim,  sizeof(long));
	long* index_r =  aligned_alloc(MEM_DATA_ALIGN, r->ndim * sizeof(long));

	for (long ot = 0; ot < nelem; ot += t->dim[t->ndim - 1])
	{
		// map index of tensor 't' to index of tensor 'r'
		for (int i = 0; i < t->ndim; i++) {
			index_r[i] = index_t[perm[i]];
		}
		// convert back to offset of tensor 'r'
		const long or = index_to_offset(r->ndim, r->dim, index_r);

		// main copy loop
		const long n = t->dim[t->ndim - 1];
		////assume_aligned(t->data);
		////assume_aligned(r->data);
		#pragma ivdep
		for (long j = 0; j < n; j++)
		{
			r->data[or + j*stride] = t->data[ot + j];
		}

		// advance index of tensor 't' by t->dim[t->ndim - 1] elements
		next_index(t->ndim - 1, t->dim, index_t);
	}

	// clean up
	aligned_free(index_r);
	aligned_free(index_t);
}


//________________________________________________________________________________________________________________________
///
/// \brief Generalized conjugate transpose of a tensor 't' such that
/// the i-th axis in the output tensor 'r' is the perm[i]-th axis of the input tensor 't'.
///
/// Memory will be allocated for 'r'.
///
void conjugate_transpose_dense_tensor(const int* restrict perm, const struct dense_tensor* restrict t, struct dense_tensor* restrict r)
{
	transpose_dense_tensor(perm, t, r);
	conjugate_dense_tensor(r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Scalar multiply and add two tensors: t = alpha*s + t; dimensions of s and t must agree.
///
void dense_tensor_scalar_multiply_add(const numeric alpha, const struct dense_tensor* restrict s, struct dense_tensor* restrict t)
{
	const long nelem = dense_tensor_num_elements(s);

	assert(s->ndim == t->ndim);
	assert(nelem == dense_tensor_num_elements(t));

	cblas_zaxpy((int)nelem, &alpha, s->data, 1, t->data, 1);
}


//________________________________________________________________________________________________________________________
///
/// \brief Multiply trailing 'ndim_mult' axes in 's' by leading 'ndim_mult' axes in 't', and store result in 'r'.
///
/// Memory will be allocated for 'r'.
///
void dense_tensor_dot(const struct dense_tensor* restrict s, const struct dense_tensor* restrict t, const int ndim_mult, struct dense_tensor* restrict r)
{
	assert(ndim_mult >= 1);
	assert(s->ndim >= ndim_mult && t->ndim >= ndim_mult);
	for (int i = 0; i < ndim_mult; i++)
	{
		assert(s->dim[s->ndim - ndim_mult + i] == t->dim[i]);
	}

	// dimensions of new tensor 'r'
	long* rdim = aligned_alloc(MEM_DATA_ALIGN, (s->ndim + t->ndim - 2 * ndim_mult) * sizeof(long));
	for (int i = 0; i < s->ndim - ndim_mult; i++)
	{
		rdim[i] = s->dim[i];
	}
	for (int i = ndim_mult; i < t->ndim; i++)
	{
		rdim[s->ndim + i - 2*ndim_mult] = t->dim[i];
	}
	// create new tensor 'r'
	allocate_dense_tensor(s->ndim + t->ndim - 2 * ndim_mult, rdim, r);
	aligned_free(rdim);

	// leading dimension of 's' as a matrix
	const long lds = integer_product(s->dim, s->ndim - ndim_mult);
	// trailing dimension of 's' as a matrix
	const long tds = integer_product(&s->dim[s->ndim - ndim_mult], ndim_mult);
	// overall number of entries in 's'
	const long nelem_s = lds * tds;

	// leading dimension of 't' as a matrix
	const long ldt = integer_product(t->dim, ndim_mult);
	// trailing dimension of 't' as a matrix
	const long tdt = integer_product(&t->dim[ndim_mult], t->ndim - ndim_mult);
	// overall number of entries in 't'
	const long nelem_t = ldt * tdt;

	assert(tds == ldt);

	if (lds == 1)
	{
		if (tdt == 1)
		{
			// inner product of two vectors
			assert(nelem_s == nelem_t);
			assert(dense_tensor_num_elements(r) == 1);
			cblas_zdotu_sub((int)nelem_s, s->data, 1, t->data, 1, &r->data[0]);
		}
		else    // tdt > 1
		{
			// multiply vector 's' from left, i.e., (t^T * s)^T
			const numeric one  = 1;
			const numeric zero = 0;
			cblas_zgemv(CblasRowMajor, CblasTrans, (int)ldt, (int)tdt, &one, t->data, (int)tdt, s->data, 1, &zero, r->data, 1);
		}
	}
	else    // lds > 1
	{
		if (tdt == 1)
		{
			// matrix-vector multiplication
			const numeric one  = 1;
			const numeric zero = 0;
			cblas_zgemv(CblasRowMajor, CblasNoTrans, (int)lds, (int)ldt, &one, s->data, (int)tds, t->data, 1, &zero, r->data, 1);
		}
		else    // tdt > 1
		{
			// matrix-matrix multiplication
			const numeric one  = 1;
			const numeric zero = 0;
			cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)lds, (int)tdt, (int)ldt, &one, s->data, (int)tds, t->data, (int)tdt, &zero, r->data, (int)tdt);
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the Kronecker product of two tensors; the tensors must have the same degree.
///
void dense_tensor_kronecker_product(const struct dense_tensor* restrict s, const struct dense_tensor* restrict t, struct dense_tensor* restrict r)
{
	// tensors must have the same degree
	assert(s->ndim == t->ndim);

	// interleave dimensions of 's' and 't'
	long* rdim_il = aligned_alloc(MEM_DATA_ALIGN, (s->ndim + t->ndim) * sizeof(long));
	for (int i = 0; i < s->ndim; i++)
	{
		rdim_il[2*i  ] = s->dim[i];
		rdim_il[2*i+1] = t->dim[i];
	}
	allocate_dense_tensor(s->ndim + t->ndim, rdim_il, r);
	aligned_free(rdim_il);

	long* index_s = aligned_calloc(MEM_DATA_ALIGN, s->ndim, sizeof(long));
	long* index_t = aligned_calloc(MEM_DATA_ALIGN, t->ndim, sizeof(long));
	long* index_r = aligned_calloc(MEM_DATA_ALIGN, r->ndim, sizeof(long));

	const long last_dim_s = s->dim[s->ndim - 1];
	const long last_dim_t = t->dim[t->ndim - 1];

	const long nelem = dense_tensor_num_elements(r);

	for (long or = 0; or < nelem; or += last_dim_s*last_dim_t)
	{
		// decode indices for 's' and 't'
		for (int i = 0; i < s->ndim; i++)
		{
			index_s[i] = index_r[2*i  ];
			index_t[i] = index_r[2*i+1];
		}
		const long os = index_to_offset(s->ndim, s->dim, index_s);
		const long ot = index_to_offset(t->ndim, t->dim, index_t);

		// outer product; r->data must have been initialized with zeros
		const numeric one = 1;
		cblas_zgeru(CblasRowMajor, (int)last_dim_s, (int)last_dim_t, &one, s->data + os, 1, t->data + ot, 1, r->data + or, (int)last_dim_t);

		// advance index of tensor 'r' by last_dim_s*last_dim_t elements
		next_index(r->ndim - 2, r->dim, index_r);
	}

	aligned_free(index_r);
	aligned_free(index_t);
	aligned_free(index_s);

	// actual dimensions of 'r'
	long* rdim = aligned_alloc(MEM_DATA_ALIGN, s->ndim * sizeof(long));
	for (int i = 0; i < s->ndim; i++)
	{
		rdim[i] = s->dim[i] * t->dim[i];
	}
	reshape_dense_tensor(s->ndim, rdim, r);
	aligned_free(rdim);
}


//________________________________________________________________________________________________________________________
///
/// \brief Extract the sub-block with dimensions 'bdim' by taking elements indexed by 'idx' along each dimension
/// from the original tensor 't'; the degree remains the same as in 't'.
///
void dense_tensor_block(const struct dense_tensor* restrict t, const long* restrict bdim, const long* restrict* idx, struct dense_tensor* restrict b)
{
	// create new tensor 'b'; degree agrees with 't'
	allocate_dense_tensor(t->ndim, bdim, b);
	if (t->ndim == 0)
	{
		b->data[0] = t->data[0];
		return;
	}

	const long nelem = dense_tensor_num_elements(b);

	long *index_t = aligned_calloc(MEM_DATA_ALIGN, t->ndim, sizeof(long));
	long *index_b = aligned_calloc(MEM_DATA_ALIGN, b->ndim, sizeof(long));

	// map first index of tensor 'b' to index of tensor 't', except for last dimension (will be handled in copy loop)
	for (int i = 0; i < t->ndim - 1; i++)
	{
		assert(t->dim[i] > 0 && b->dim[i] > 0);
		index_t[i] = idx[i][0];
	}
	// convert back to offset of tensor 't'
	long ot = index_to_offset(t->ndim, t->dim, index_t);

	const long* last_idx = idx[b->ndim - 1];

	for (long os = 0; os < nelem; os += b->dim[b->ndim - 1])
	{
		// main copy loop along last dimension
		// (treating last dimension separately to avoid calling index conversion function for each single element)
		const long n = b->dim[b->ndim - 1];
		////assume_aligned(t->data);
		////assume_aligned(b->data);
		#pragma ivdep
		for (long j = 0; j < n; j++)
		{
			b->data[os + j] = t->data[ot + last_idx[j]];
		}

		// advance index of tensor 'b' by b->dim[b->ndim - 1] elements
		next_index(b->ndim - 1, b->dim, index_b);
		// map index of tensor 'b' to index of tensor 't', except for last dimension (will be handled in copy loop)
		for (int i = 0; i < t->ndim - 1; i++) {
			index_t[i] = idx[i][index_b[i]];
		}
		// convert back to offset of tensor 't'
		ot = index_to_offset(t->ndim, t->dim, index_t);
	}

	// clean up
	aligned_free(index_b);
	aligned_free(index_t);
}
