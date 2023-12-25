/// \file dense_tensor.c
/// \brief Dense tensor structure, using row-major storage convention.

#include <stdbool.h>
#include <memory.h>
#include <complex.h>
#include <cblas.h>
#include "dense_tensor.h"
#include "config.h"


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for a dense tensor, and initialize the data entries with zeros.
///
void allocate_dense_tensor(const enum numeric_type dtype, const int ndim, const long* restrict dim, struct dense_tensor* restrict t)
{
	t->dtype = dtype;

	assert(ndim >= 0);
	t->ndim = ndim;

	if (ndim > 0)
	{
		t->dim = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(long));
		memcpy(t->dim, dim, ndim * sizeof(long));
	}
	else    // ndim == 0
	{
		// aligned_alloc(MEM_DATA_ALIGN, 0) not guaranteed to return NULL
		t->dim = NULL;
	}

	const long nelem = dense_tensor_num_elements(t);
	// dimensions must be strictly positive
	assert(nelem > 0);
	t->data = aligned_calloc(MEM_DATA_ALIGN, nelem, sizeof_numeric_type(dtype));
	assert(t->data != NULL);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a dense tensor (free memory).
///
void delete_dense_tensor(struct dense_tensor* t)
{
	aligned_free(t->data);
	t->data = NULL;

	if (t->ndim > 0)
	{
		aligned_free(t->dim);
	}
	t->ndim = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Copy a tensor, allocating memory for the copy.
///
void copy_dense_tensor(const struct dense_tensor* restrict src, struct dense_tensor* restrict dst)
{
	allocate_dense_tensor(src->dtype, src->ndim, src->dim, dst);

	const long nelem = dense_tensor_num_elements(src);

	////assume_aligned(src->data);
	////assume_aligned(dst->data);
	memcpy(dst->data, src->data, nelem * sizeof_numeric_type(src->dtype));
}


//________________________________________________________________________________________________________________________
///
/// \brief Move tensor data (without allocating new memory).
///
void move_dense_tensor_data(struct dense_tensor* restrict src, struct dense_tensor* restrict dst)
{
	dst->dtype = src->dtype;

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
void dense_tensor_trace(const struct dense_tensor* t, void* ret)
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

	switch (t->dtype)
	{
		case SINGLE_REAL:
		{
			float tr = 0;
			const float* data = t->data;
			for (long j = 0; j < n; j++)
			{
				tr += data[j * stride];
			}
			*((float*)ret) = tr;
			break;
		}
		case DOUBLE_REAL:
		{
			double tr = 0;
			const double* data = t->data;
			for (long j = 0; j < n; j++)
			{
				tr += data[j * stride];
			}
			*((double*)ret) = tr;
			break;
		}
		case SINGLE_COMPLEX:
		{
			scomplex tr = 0;
			const scomplex* data = t->data;
			for (long j = 0; j < n; j++)
			{
				tr += data[j * stride];
			}
			*((scomplex*)ret) = tr;
			break;
		}
		case DOUBLE_COMPLEX:
		{
			dcomplex tr = 0;
			const dcomplex* data = t->data;
			for (long j = 0; j < n; j++)
			{
				tr += data[j * stride];
			}
			*((dcomplex*)ret) = tr;
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
/// \brief Scale tensor 't' by 'alpha', which must be of the same data type as the tensor entries.
///
void scale_dense_tensor(const void* alpha, struct dense_tensor* t)
{
	switch (t->dtype)
	{
		case SINGLE_REAL:
		{
			cblas_sscal(dense_tensor_num_elements(t), *((float*)alpha), t->data, 1);
			break;
		}
		case DOUBLE_REAL:
		{
			cblas_dscal(dense_tensor_num_elements(t), *((double*)alpha), t->data, 1);
			break;
		}
		case SINGLE_COMPLEX:
		{
			cblas_cscal(dense_tensor_num_elements(t), alpha, t->data, 1);
			break;
		}
		case DOUBLE_COMPLEX:
		{
			cblas_zscal(dense_tensor_num_elements(t), alpha, t->data, 1);
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
/// \brief Scale tensor 't' by a real number 'alpha', which must be of the same precision as the tensor entries.
///
void rscale_dense_tensor(const void* alpha, struct dense_tensor* t)
{
	switch (t->dtype)
	{
		case SINGLE_REAL:
		{
			cblas_sscal(dense_tensor_num_elements(t), *((float*)alpha), t->data, 1);
			break;
		}
		case DOUBLE_REAL:
		{
			cblas_dscal(dense_tensor_num_elements(t), *((double*)alpha), t->data, 1);
			break;
		}
		case SINGLE_COMPLEX:
		{
			cblas_csscal(dense_tensor_num_elements(t), *((float*)alpha), t->data, 1);
			break;
		}
		case DOUBLE_COMPLEX:
		{
			cblas_zdscal(dense_tensor_num_elements(t), *((double*)alpha), t->data, 1);
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
/// \brief Reshape dimensions, i.e., interpret as tensor of different dimension with the same number of elements.
///
void reshape_dense_tensor(const int ndim, const long* dim, struct dense_tensor* t)
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
/// \brief Elementwise complex conjugation of a dense tensor.
///
/// Function has no effect if entries are real-valued.
///
void conjugate_dense_tensor(struct dense_tensor* t)
{
	switch (t->dtype)
	{
		case SINGLE_REAL:
		case DOUBLE_REAL:
		{
			// no effect
			break;
		}
		case SINGLE_COMPLEX:
		{
			const long nelem = dense_tensor_num_elements(t);
			scomplex* data = t->data;
			for (long i = 0; i < nelem; i++)
			{
				data[i] = conjf(data[i]);
			}
			break;
		}
		case DOUBLE_COMPLEX:
		{
			const long nelem = dense_tensor_num_elements(t);
			dcomplex* data = t->data;
			for (long i = 0; i < nelem; i++)
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

	switch (t->dtype)
	{
		case SINGLE_REAL:
		{
			float* data = t->data;
			memset(data, 0, n*dp * sizeof(float));
			for (long j = 0; j < n; j++)
			{
				data[j*stride] = 1;
			}
			break;
		}
		case DOUBLE_REAL:
		{
			double* data = t->data;
			memset(data, 0, n*dp * sizeof(double));
			for (long j = 0; j < n; j++)
			{
				data[j*stride] = 1;
			}
			break;
		}
		case SINGLE_COMPLEX:
		{
			scomplex* data = t->data;
			memset(data, 0, n*dp * sizeof(scomplex));
			for (long j = 0; j < n; j++)
			{
				data[j*stride] = 1;
			}
			break;
		}
		case DOUBLE_COMPLEX:
		{
			dcomplex* data = t->data;
			memset(data, 0, n*dp * sizeof(dcomplex));
			for (long j = 0; j < n; j++)
			{
				data[j*stride] = 1;
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
/// \brief Generalized transpose of a tensor 't' such that
/// the i-th axis in the output tensor 'r' is the perm[i]-th axis of the input tensor 't'.
///
/// Memory will be allocated for 'r'.
///
void transpose_dense_tensor(const int* restrict perm, const struct dense_tensor* restrict t, struct dense_tensor* restrict r)
{
	// TODO: consider using an optimized library like https://github.com/springer13/hptt

	if (t->ndim == 0)
	{
		allocate_dense_tensor(t->dtype, 0, NULL, r);
		// copy single scalar entry
		memcpy(r->data, t->data, sizeof_numeric_type(t->dtype));
		return;
	}

	// ensure that 'perm' is a valid permutation
	int* ax_list = aligned_calloc(MEM_DATA_ALIGN, t->ndim, sizeof(int));
	for (int i = 0; i < t->ndim; i++)
	{
		assert(0 <= perm[i] && perm[i] < t->ndim);
		ax_list[perm[i]] = 1;
	}
	for (int i = 0; i < t->ndim; i++)
	{
		assert(ax_list[i] == 1);
	}
	aligned_free(ax_list);

	// dimensions of new tensor 'r'
	long* rdim = aligned_alloc(MEM_DATA_ALIGN, t->ndim * sizeof(long));
	for (int i = 0; i < t->ndim; i++) {
		rdim[i] = t->dim[perm[i]];
	}
	// create new tensor 'r'
	allocate_dense_tensor(t->dtype, t->ndim, rdim, r);
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
		const long or = tensor_index_to_offset(r->ndim, r->dim, index_r);

		// main copy loop
		const long n = t->dim[t->ndim - 1];
		switch (t->dtype)
		{
			case SINGLE_REAL:
			{
				const float* tdata = (const float*)t->data + ot;
				float*       rdata =       (float*)r->data + or;
				////assume_aligned(tdata);
				////assume_aligned(rdata);
				#pragma ivdep
				for (long j = 0; j < n; j++)
				{
					rdata[j*stride] = tdata[j];
				}
				break;
			}
			case DOUBLE_REAL:
			{
				const double* tdata = (const double*)t->data + ot;
				double*       rdata =       (double*)r->data + or;
				////assume_aligned(tdata);
				////assume_aligned(rdata);
				#pragma ivdep
				for (long j = 0; j < n; j++)
				{
					rdata[j*stride] = tdata[j];
				}
				break;
			}
			case SINGLE_COMPLEX:
			{
				const scomplex* tdata = (const scomplex*)t->data + ot;
				scomplex*       rdata =       (scomplex*)r->data + or;
				////assume_aligned(tdata);
				////assume_aligned(rdata);
				#pragma ivdep
				for (long j = 0; j < n; j++)
				{
					rdata[j*stride] = tdata[j];
				}
				break;
			}
			case DOUBLE_COMPLEX:
			{
				const dcomplex* tdata = (const dcomplex*)t->data + ot;
				dcomplex*       rdata =       (dcomplex*)r->data + or;
				////assume_aligned(tdata);
				////assume_aligned(rdata);
				#pragma ivdep
				for (long j = 0; j < n; j++)
				{
					rdata[j*stride] = tdata[j];
				}
				break;
			}
			default:
			{
				// unknown data type
				assert(false);
			}
		}

		// advance index of tensor 't' by t->dim[t->ndim - 1] elements
		next_tensor_index(t->ndim - 1, t->dim, index_t);
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
/// \brief Scalar multiply and add two tensors: t = alpha*s + t; dimensions and data types of s and t must agree,
/// and alpha must be of the same data type as tensor entries.
///
void dense_tensor_scalar_multiply_add(const void* alpha, const struct dense_tensor* restrict s, struct dense_tensor* restrict t)
{
	// data types must agree
	assert(s->dtype == t->dtype);

	const long nelem = dense_tensor_num_elements(s);

	assert(s->ndim == t->ndim);
	assert(nelem == dense_tensor_num_elements(t));

	switch (t->dtype)
	{
		case SINGLE_REAL:
		{
			cblas_saxpy(nelem, *((float*)alpha), s->data, 1, t->data, 1);
			break;
		}
		case DOUBLE_REAL:
		{
			cblas_daxpy(nelem, *((double*)alpha), s->data, 1, t->data, 1);
			break;
		}
		case SINGLE_COMPLEX:
		{
			cblas_caxpy(nelem, alpha, s->data, 1, t->data, 1);
			break;
		}
		case DOUBLE_COMPLEX:
		{
			cblas_zaxpy(nelem, alpha, s->data, 1, t->data, 1);
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
/// \brief Multiply trailing 'ndim_mult' axes in 's' by leading 'ndim_mult' axes in 't', and store result in 'r'.
///
/// Memory will be allocated for 'r'.
///
void dense_tensor_dot(const struct dense_tensor* restrict s, const struct dense_tensor* restrict t, const int ndim_mult, struct dense_tensor* restrict r)
{
	// data types must agree
	assert(s->dtype == t->dtype);

	assert(ndim_mult >= 1);
	assert(s->ndim >= ndim_mult && t->ndim >= ndim_mult);
	for (int i = 0; i < ndim_mult; i++)
	{
		assert(s->dim[s->ndim - ndim_mult + i] == t->dim[i]);
	}

	// dimensions of new tensor 'r'
	long* rdim = aligned_alloc(MEM_DATA_ALIGN, (s->ndim + t->ndim - 2*ndim_mult) * sizeof(long));
	for (int i = 0; i < s->ndim - ndim_mult; i++)
	{
		rdim[i] = s->dim[i];
	}
	for (int i = ndim_mult; i < t->ndim; i++)
	{
		rdim[s->ndim + i - 2*ndim_mult] = t->dim[i];
	}
	// create new tensor 'r'
	allocate_dense_tensor(s->dtype, s->ndim + t->ndim - 2*ndim_mult, rdim, r);
	aligned_free(rdim);

	// leading dimension of 's' as a matrix
	const long lds = integer_product(s->dim, s->ndim - ndim_mult);
	// trailing dimension of 's' as a matrix
	const long tds = integer_product(&s->dim[s->ndim - ndim_mult], ndim_mult);

	// leading dimension of 't' as a matrix
	const long ldt = integer_product(t->dim, ndim_mult);
	// trailing dimension of 't' as a matrix
	const long tdt = integer_product(&t->dim[ndim_mult], t->ndim - ndim_mult);

	assert(tds == ldt);

	// matrix-matrix multiplication
	switch (s->dtype)
	{
		case SINGLE_REAL:
		{
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lds, tdt, ldt, 1.f, s->data, tds, t->data, tdt, 0.f, r->data, tdt);
			break;
		}
		case DOUBLE_REAL:
		{
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lds, tdt, ldt, 1.0, s->data, tds, t->data, tdt, 0.0, r->data, tdt);
			break;
		}
		case SINGLE_COMPLEX:
		{
			const scomplex one  = 1;
			const scomplex zero = 0;
			cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lds, tdt, ldt, &one, s->data, tds, t->data, tdt, &zero, r->data, tdt);
			break;
		}
		case DOUBLE_COMPLEX:
		{
			const dcomplex one  = 1;
			const dcomplex zero = 0;
			cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lds, tdt, ldt, &one, s->data, tds, t->data, tdt, &zero, r->data, tdt);
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
/// \brief Multiply trailing 'ndim_mult' axes in 's' by leading 'ndim_mult' axes in 't', scale by 'alpha' and
/// add result to 'r' scaled by beta: r <- alpha * s @ t + beta * r
///
/// Assuming that 'r' has the appropriate dimensions.
///
void dense_tensor_dot_update(const void* alpha, const struct dense_tensor* restrict s, const struct dense_tensor* restrict t, const int ndim_mult, struct dense_tensor* restrict r, const void* beta)
{
	// data types must agree
	assert(s->dtype == t->dtype);

	assert(ndim_mult >= 1);
	assert(s->ndim >= ndim_mult && t->ndim >= ndim_mult);
	for (int i = 0; i < ndim_mult; i++)
	{
		assert(s->dim[s->ndim - ndim_mult + i] == t->dim[i]);
	}
	for (int i = 0; i < s->ndim - ndim_mult; i++)
	{
		assert(r->dim[i] == s->dim[i]);
	}
	for (int i = ndim_mult; i < t->ndim; i++)
	{
		assert(r->dim[s->ndim + i - 2*ndim_mult] == t->dim[i]);
	}

	// leading dimension of 's' as a matrix
	const long lds = integer_product(s->dim, s->ndim - ndim_mult);
	// trailing dimension of 's' as a matrix
	const long tds = integer_product(&s->dim[s->ndim - ndim_mult], ndim_mult);

	// leading dimension of 't' as a matrix
	const long ldt = integer_product(t->dim, ndim_mult);
	// trailing dimension of 't' as a matrix
	const long tdt = integer_product(&t->dim[ndim_mult], t->ndim - ndim_mult);

	assert(tds == ldt);

	// matrix-matrix multiplication
	switch (s->dtype)
	{
		case SINGLE_REAL:
		{
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lds, tdt, ldt, *((float*)alpha), s->data, tds, t->data, tdt, *((float*)beta), r->data, tdt);
			break;
		}
		case DOUBLE_REAL:
		{
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lds, tdt, ldt, *((double*)alpha), s->data, tds, t->data, tdt, *((double*)beta), r->data, tdt);
			break;
		}
		case SINGLE_COMPLEX:
		{
			cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lds, tdt, ldt, alpha, s->data, tds, t->data, tdt, beta, r->data, tdt);
			break;
		}
		case DOUBLE_COMPLEX:
		{
			cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lds, tdt, ldt, alpha, s->data, tds, t->data, tdt, beta, r->data, tdt);
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
/// \brief Compute the Kronecker product of two tensors; the tensors must have the same degree.
///
void dense_tensor_kronecker_product(const struct dense_tensor* restrict s, const struct dense_tensor* restrict t, struct dense_tensor* restrict r)
{
	// data types must agree
	assert(s->dtype == t->dtype);

	// tensors must have the same degree
	assert(s->ndim == t->ndim);

	if (s->ndim == 0)
	{
		allocate_dense_tensor(s->dtype, 0, NULL, r);

		// multiply single scalar entries
		switch (s->dtype)
		{
			case SINGLE_REAL:
			{
				const float* sdata = s->data;
				const float* tdata = t->data;
				float*       rdata = r->data;
				rdata[0] = sdata[0] * tdata[0];
				break;
			}
			case DOUBLE_REAL:
			{
				const double* sdata = s->data;
				const double* tdata = t->data;
				double*       rdata = r->data;
				rdata[0] = sdata[0] * tdata[0];
				break;
			}
			case SINGLE_COMPLEX:
			{
				const scomplex* sdata = s->data;
				const scomplex* tdata = t->data;
				scomplex*       rdata = r->data;
				rdata[0] = sdata[0] * tdata[0];
				break;
			}
			case DOUBLE_COMPLEX:
			{
				const dcomplex* sdata = s->data;
				const dcomplex* tdata = t->data;
				dcomplex*       rdata = r->data;
				rdata[0] = sdata[0] * tdata[0];
				break;
			}
			default:
			{
				// unknown data type
				assert(false);
			}
		}

		return;
	}

	// interleave dimensions of 's' and 't'
	long* rdim_il = aligned_alloc(MEM_DATA_ALIGN, (s->ndim + t->ndim) * sizeof(long));
	for (int i = 0; i < s->ndim; i++)
	{
		rdim_il[2*i  ] = s->dim[i];
		rdim_il[2*i+1] = t->dim[i];
	}
	allocate_dense_tensor(s->dtype, s->ndim + t->ndim, rdim_il, r);
	aligned_free(rdim_il);

	long* index_s = aligned_calloc(MEM_DATA_ALIGN, s->ndim, sizeof(long));
	long* index_t = aligned_calloc(MEM_DATA_ALIGN, t->ndim, sizeof(long));
	long* index_r = aligned_calloc(MEM_DATA_ALIGN, r->ndim, sizeof(long));

	const long last_dim_s = s->dim[s->ndim - 1];
	const long last_dim_t = t->dim[t->ndim - 1];

	const long nelem = dense_tensor_num_elements(r);

	// advance index of tensor 'r' by last_dim_s*last_dim_t elements in each iteration
	for (long or = 0; or < nelem; or += last_dim_s*last_dim_t, next_tensor_index(r->ndim - 2, r->dim, index_r))
	{
		// decode indices for 's' and 't'
		for (int i = 0; i < s->ndim; i++)
		{
			index_s[i] = index_r[2*i  ];
			index_t[i] = index_r[2*i+1];
		}
		const long os = tensor_index_to_offset(s->ndim, s->dim, index_s);
		const long ot = tensor_index_to_offset(t->ndim, t->dim, index_t);

		// outer product; r->data must have been initialized with zeros
		switch (s->dtype)
		{
			case SINGLE_REAL:
			{
				cblas_sger(CblasRowMajor, last_dim_s, last_dim_t, 1.f, (const float*)s->data + os, 1, (const float*)t->data + ot, 1, (float*)r->data + or, last_dim_t);
				break;
			}
			case DOUBLE_REAL:
			{
				cblas_dger(CblasRowMajor, last_dim_s, last_dim_t, 1.0, (const double*)s->data + os, 1, (const double*)t->data + ot, 1, (double*)r->data + or, last_dim_t);
				break;
			}
			case SINGLE_COMPLEX:
			{
				const scomplex one = 1;
				cblas_cgeru(CblasRowMajor, last_dim_s, last_dim_t, &one, (const scomplex*)s->data + os, 1, (const scomplex*)t->data + ot, 1, (scomplex*)r->data + or, last_dim_t);
				break;
			}
			case DOUBLE_COMPLEX:
			{
				const dcomplex one = 1;
				cblas_zgeru(CblasRowMajor, last_dim_s, last_dim_t, &one, (const dcomplex*)s->data + os, 1, (const dcomplex*)t->data + ot, 1, (dcomplex*)r->data + or, last_dim_t);
				break;
			}
			default:
			{
				// unknown data type
				assert(false);
			}
		}
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
	// create new tensor 'b' of same data type and degree as 't'
	allocate_dense_tensor(t->dtype, t->ndim, bdim, b);

	if (t->ndim == 0)
	{
		// copy single scalar entry
		memcpy(b->data, t->data, sizeof_numeric_type(t->dtype));
		return;
	}

	const long nelem = dense_tensor_num_elements(b);

	long* index_t = aligned_calloc(MEM_DATA_ALIGN, t->ndim, sizeof(long));
	long* index_b = aligned_calloc(MEM_DATA_ALIGN, b->ndim, sizeof(long));

	// map first index of tensor 'b' to index of tensor 't', except for last dimension (will be handled in copy loop)
	for (int i = 0; i < t->ndim - 1; i++)
	{
		assert(t->dim[i] > 0 && b->dim[i] > 0);
		index_t[i] = idx[i][0];
	}
	// convert back to offset of tensor 't'
	long ot = tensor_index_to_offset(t->ndim, t->dim, index_t);

	const long* last_idx = idx[b->ndim - 1];

	for (long ob = 0; ob < nelem; ob += b->dim[b->ndim - 1])
	{
		// main copy loop along last dimension
		// (treating last dimension separately to avoid calling index conversion function for each single element)
		const long n = b->dim[b->ndim - 1];
		switch (t->dtype)
		{
			case SINGLE_REAL:
			{
				const float* tdata = (const float*)t->data + ot;
				float*       bdata =       (float*)b->data + ob;
				////assume_aligned(tdata);
				////assume_aligned(bdata);
				#pragma ivdep
				for (long j = 0; j < n; j++)
				{
					bdata[j] = tdata[last_idx[j]];
				}
				break;
			}
			case DOUBLE_REAL:
			{
				const double* tdata = (const double*)t->data + ot;
				double*       bdata =       (double*)b->data + ob;
				////assume_aligned(tdata);
				////assume_aligned(bdata);
				#pragma ivdep
				for (long j = 0; j < n; j++)
				{
					bdata[j] = tdata[last_idx[j]];
				}
				break;
			}
			case SINGLE_COMPLEX:
			{
				const scomplex* tdata = (const scomplex*)t->data + ot;
				scomplex*       bdata =       (scomplex*)b->data + ob;
				////assume_aligned(tdata);
				////assume_aligned(bdata);
				#pragma ivdep
				for (long j = 0; j < n; j++)
				{
					bdata[j] = tdata[last_idx[j]];
				}
				break;
			}
			case DOUBLE_COMPLEX:
			{
				const dcomplex* tdata = (const dcomplex*)t->data + ot;
				dcomplex*       bdata =       (dcomplex*)b->data + ob;
				////assume_aligned(tdata);
				////assume_aligned(bdata);
				#pragma ivdep
				for (long j = 0; j < n; j++)
				{
					bdata[j] = tdata[last_idx[j]];
				}
				break;
			}
			default:
			{
				// unknown data type
				assert(false);
			}
		}

		// advance index of tensor 'b' by b->dim[b->ndim - 1] elements
		next_tensor_index(b->ndim - 1, b->dim, index_b);
		// map index of tensor 'b' to index of tensor 't', except for last dimension (will be handled in copy loop)
		for (int i = 0; i < t->ndim - 1; i++) {
			index_t[i] = idx[i][index_b[i]];
		}
		// convert back to offset of tensor 't'
		ot = tensor_index_to_offset(t->ndim, t->dim, index_t);
	}

	// clean up
	aligned_free(index_b);
	aligned_free(index_t);
}
