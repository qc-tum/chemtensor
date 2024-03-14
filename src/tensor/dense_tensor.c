/// \file dense_tensor.c
/// \brief Dense tensor structure, using row-major storage convention.

#include <stdbool.h>
#include <math.h>
#include <memory.h>
#include <complex.h>
#include <cblas.h>
#include <lapacke.h>
#include "dense_tensor.h"
#include "aligned_memory.h"


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
/// \brief Compute the 2-norm (Frobenius norm) of the tensor.
///
/// Result is returned as double also for single-precision tensor entries.
///
double dense_tensor_norm2(const struct dense_tensor* t)
{
	const long nelem = dense_tensor_num_elements(t);

	switch (t->dtype)
	{
		case SINGLE_REAL:
		{
			return cblas_snrm2(nelem, t->data, 1);
		}
		case DOUBLE_REAL:
		{
			return cblas_dnrm2(nelem, t->data, 1);
		}
		case SINGLE_COMPLEX:
		{
			return cblas_scnrm2(nelem, t->data, 1);
		}
		case DOUBLE_COMPLEX:
		{
			return cblas_dznrm2(nelem, t->data, 1);
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
	// TODO: consider using an optimized library like https://github.com/springer13/hptt,
	// and fuse neighboring to-be transposed axes

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
				////#pragma ivdep
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
				////#pragma ivdep
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
				////#pragma ivdep
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
				////#pragma ivdep
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
/// \brief Slice an axis of the tensor by selecting logical indices 'ind' along this axis.
///
/// Indices 'ind' can be duplicate and need not be sorted.
/// Memory will be allocated for 'r'.
///
void dense_tensor_slice(const struct dense_tensor* restrict t, const int i_ax, const long* ind, const long nind, struct dense_tensor* restrict r)
{
	assert(0 <= i_ax && i_ax < t->ndim);
	assert(nind > 0);

	long* rdim = aligned_alloc(MEM_DATA_ALIGN, t->ndim * sizeof(long));
	memcpy(rdim, t->dim, t->ndim * sizeof(long));
	rdim[i_ax] = nind;
	allocate_dense_tensor(t->dtype, t->ndim, rdim, r);
	aligned_free(rdim);

	dense_tensor_slice_fill(t, i_ax, ind, nind, r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Slice an axis of the tensor by selecting logical indices 'ind' along this axis.
///
/// Indices 'ind' can be duplicate and need not be sorted.
/// 'r' must have been allocated beforehand.
///
void dense_tensor_slice_fill(const struct dense_tensor* restrict t, const int i_ax, const long* ind, const long nind, struct dense_tensor* restrict r)
{
	assert(0 <= i_ax && i_ax < t->ndim);
	assert(nind > 0);
	assert(r->dim[i_ax] == nind);

	// leading dimension of 't' and 'r'
	const long ld = integer_product(t->dim, i_ax);
	// trailing dimension of 't' and 'r'
	const long td = integer_product(&t->dim[i_ax + 1], t->ndim - (i_ax + 1));

	const long stride = td * sizeof_numeric_type(t->dtype);
	for (long j = 0; j < ld; j++)
	{
		for (long k = 0; k < nind; k++)
		{
			assert(0 <= ind[k] && ind[k] < t->dim[i_ax]);
			// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
			memcpy((int8_t*)r->data + (j * nind + k) * stride, (int8_t*)t->data + (j * t->dim[i_ax] + ind[k]) * stride, stride);
		}
	}
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
/// \brief Pointwise multiplication of two tensors, using broadcasting.
/// The output tensor 'r' has the same data type and dimension as 's'.
/// The dimensions of 's' and 't' (starting from the leading or trailing dimension depending on 'axrange') must match.
///
/// Memory will be allocated for 'r'.
///
void dense_tensor_multiply_pointwise(const struct dense_tensor* restrict s, const struct dense_tensor* restrict t, const enum tensor_axis_range axrange, struct dense_tensor* restrict r)
{
	// allocate 'r' with same type and dimension as 's'
	allocate_dense_tensor(s->dtype, s->ndim, s->dim, r);

	dense_tensor_multiply_pointwise_fill(s, t, axrange, r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Pointwise multiplication of two tensors, using broadcasting.
/// The output tensor 'r' has the same data type and dimension as 's'.
/// The dimensions of 's' and 't' (starting from the leading or trailing dimension depending on 'axrange') must match.
///
/// 'r' must have been allocated beforehand.
///
void dense_tensor_multiply_pointwise_fill(const struct dense_tensor* restrict s, const struct dense_tensor* restrict t, const enum tensor_axis_range axrange, struct dense_tensor* restrict r)
{
	assert(s->dtype == t->dtype || numeric_real_type(s->dtype) == t->dtype);
	assert(s->dtype == r->dtype);
	assert(s->ndim  == r->ndim);
	for (int i = 0; i < s->ndim; i++) {
		assert(s->dim[i] == r->dim[i]);
	}

	const int ndim_diff = s->ndim - t->ndim;
	assert(ndim_diff >= 0);

	if (axrange == TENSOR_AXIS_RANGE_LEADING)
	{
		// check dimension compatibility
		for (int i = 0; i < t->ndim; i++) {
			assert(t->dim[i] == s->dim[i]);
		}

		const long nt = dense_tensor_num_elements(t);
		const long tds = integer_product(&s->dim[t->ndim], ndim_diff);

		switch (s->dtype)
		{
			case SINGLE_REAL:
			{
				const float* sdata = s->data;
				const float* tdata = t->data;
				float*       rdata = r->data;
				for (long j = 0; j < nt; j++)
				{
					for (long k = 0; k < tds; k++)
					{
						rdata[j*tds + k] = sdata[j*tds + k] * tdata[j];
					}
				}
				break;
			}
			case DOUBLE_REAL:
			{
				const double* sdata = s->data;
				const double* tdata = t->data;
				double*       rdata = r->data;
				for (long j = 0; j < nt; j++)
				{
					for (long k = 0; k < tds; k++)
					{
						rdata[j*tds + k] = sdata[j*tds + k] * tdata[j];
					}
				}
				break;
			}
			case SINGLE_COMPLEX:
			{
				const scomplex* sdata = s->data;
				scomplex*       rdata = r->data;
				if (t->dtype == SINGLE_COMPLEX)
				{
					const scomplex* tdata = t->data;
					for (long j = 0; j < nt; j++)
					{
						for (long k = 0; k < tds; k++)
						{
							rdata[j*tds + k] = sdata[j*tds + k] * tdata[j];
						}
					}
				}
				else
				{
					assert(t->dtype == SINGLE_REAL);
					const float* tdata = t->data;
					for (long j = 0; j < nt; j++)
					{
						for (long k = 0; k < tds; k++)
						{
							rdata[j*tds + k] = sdata[j*tds + k] * tdata[j];
						}
					}
				}
				break;
			}
			case DOUBLE_COMPLEX:
			{
				const dcomplex* sdata = s->data;
				dcomplex*       rdata = r->data;
				if (t->dtype == DOUBLE_COMPLEX)
				{
					const dcomplex* tdata = t->data;
					for (long j = 0; j < nt; j++)
					{
						for (long k = 0; k < tds; k++)
						{
							rdata[j*tds + k] = sdata[j*tds + k] * tdata[j];
						}
					}
				}
				else
				{
					assert(t->dtype == DOUBLE_REAL);
					const double* tdata = t->data;
					for (long j = 0; j < nt; j++)
					{
						for (long k = 0; k < tds; k++)
						{
							rdata[j*tds + k] = sdata[j*tds + k] * tdata[j];
						}
					}
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
	else
	{
		assert(axrange == TENSOR_AXIS_RANGE_TRAILING);

		// check dimension compatibility
		for (int i = 0; i < t->ndim; i++) {
			assert(t->dim[i] == s->dim[ndim_diff + i]);
		}

		const long nt = dense_tensor_num_elements(t);
		const long lds = integer_product(s->dim, ndim_diff);

		switch (s->dtype)
		{
			case SINGLE_REAL:
			{
				const float* sdata = s->data;
				const float* tdata = t->data;
				float*       rdata = r->data;
				for (long j = 0; j < lds; j++)
				{
					for (long k = 0; k < nt; k++)
					{
						rdata[j*nt + k] = sdata[j*nt + k] * tdata[k];
					}
				}
				break;
			}
			case DOUBLE_REAL:
			{
				const double* sdata = s->data;
				const double* tdata = t->data;
				double*       rdata = r->data;
				for (long j = 0; j < lds; j++)
				{
					for (long k = 0; k < nt; k++)
					{
						rdata[j*nt + k] = sdata[j*nt + k] * tdata[k];
					}
				}
				break;
			}
			case SINGLE_COMPLEX:
			{
				const scomplex* sdata = s->data;
				scomplex*       rdata = r->data;
				if (t->dtype == SINGLE_COMPLEX)
				{
					const scomplex* tdata = t->data;
					for (long j = 0; j < lds; j++)
					{
						for (long k = 0; k < nt; k++)
						{
							rdata[j*nt + k] = sdata[j*nt + k] * tdata[k];
						}
					}
				}
				else
				{
					assert(t->dtype == SINGLE_REAL);
					const float* tdata = t->data;
					for (long j = 0; j < lds; j++)
					{
						for (long k = 0; k < nt; k++)
						{
							rdata[j*nt + k] = sdata[j*nt + k] * tdata[k];
						}
					}
				}
				break;
			}
			case DOUBLE_COMPLEX:
			{
				const dcomplex* sdata = s->data;
				dcomplex*       rdata = r->data;
				if (t->dtype == DOUBLE_COMPLEX)
				{
					const dcomplex* tdata = t->data;
					for (long j = 0; j < lds; j++)
					{
						for (long k = 0; k < nt; k++)
						{
							rdata[j*nt + k] = sdata[j*nt + k] * tdata[k];
						}
					}
				}
				else
				{
					assert(t->dtype == DOUBLE_REAL);
					const double* tdata = t->data;
					for (long j = 0; j < lds; j++)
					{
						for (long k = 0; k < nt; k++)
						{
							rdata[j*nt + k] = sdata[j*nt + k] * tdata[k];
						}
					}
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
}


//________________________________________________________________________________________________________________________
///
/// \brief Multiply (leading or trailing) 'ndim_mult' axes in 's' by 'ndim_mult' axes in 't', and store result in 'r'.
/// Whether to use leading or trailing axes is specified by axis range.
///
/// Memory will be allocated for 'r'.
///
void dense_tensor_dot(const struct dense_tensor* restrict s, const enum tensor_axis_range axrange_s, const struct dense_tensor* restrict t, const enum tensor_axis_range axrange_t, const int ndim_mult, struct dense_tensor* restrict r)
{
	// data types must agree
	assert(s->dtype == t->dtype);

	assert(ndim_mult >= 1);
	assert(s->ndim >= ndim_mult && t->ndim >= ndim_mult);
	for (int i = 0; i < ndim_mult; i++)
	{
		assert(s->dim[(axrange_s == TENSOR_AXIS_RANGE_LEADING ? 0 : s->ndim - ndim_mult) + i]
		    == t->dim[(axrange_t == TENSOR_AXIS_RANGE_LEADING ? 0 : t->ndim - ndim_mult) + i]);
	}

	// dimensions of new tensor 'r'
	const int ndimr = s->ndim + t->ndim - 2*ndim_mult;
	long* rdim = aligned_alloc(MEM_DATA_ALIGN, ndimr * sizeof(long));
	const int offset_s = (axrange_s == TENSOR_AXIS_RANGE_LEADING ? ndim_mult : 0);
	const int offset_t = (axrange_t == TENSOR_AXIS_RANGE_LEADING ? ndim_mult : 0);
	for (int i = 0; i < s->ndim - ndim_mult; i++)
	{
		rdim[i] = s->dim[offset_s + i];
	}
	for (int i = 0; i < t->ndim - ndim_mult; i++)
	{
		rdim[(s->ndim - ndim_mult) + i] = t->dim[offset_t + i];
	}
	// create new tensor 'r'
	allocate_dense_tensor(s->dtype, ndimr, rdim, r);
	aligned_free(rdim);

	const int nlds = (axrange_s == TENSOR_AXIS_RANGE_LEADING ? ndim_mult : s->ndim - ndim_mult);
	const int nldt = (axrange_t == TENSOR_AXIS_RANGE_LEADING ? ndim_mult : t->ndim - ndim_mult);

	// leading dimension of 's' as a matrix
	const long lds = integer_product(s->dim, nlds);
	// trailing dimension of 's' as a matrix
	const long tds = integer_product(&s->dim[nlds], s->ndim - nlds);

	// leading dimension of 't' as a matrix
	const long ldt = integer_product(t->dim, nldt);
	// trailing dimension of 't' as a matrix
	const long tdt = integer_product(&t->dim[nldt], t->ndim - nldt);

	const CBLAS_TRANSPOSE transa = (axrange_s == TENSOR_AXIS_RANGE_LEADING ? CblasTrans : CblasNoTrans);
	const CBLAS_TRANSPOSE transb = (axrange_t == TENSOR_AXIS_RANGE_LEADING ? CblasNoTrans : CblasTrans);

	// matrix-matrix multiplication
	const long m = (axrange_s == TENSOR_AXIS_RANGE_LEADING ? tds : lds);
	const long n = (axrange_t == TENSOR_AXIS_RANGE_LEADING ? tdt : ldt);
	const long k = (axrange_s == TENSOR_AXIS_RANGE_LEADING ? lds : tds);
	assert(k == (axrange_t == TENSOR_AXIS_RANGE_LEADING ? ldt : tdt));
	switch (s->dtype)
	{
		case SINGLE_REAL:
		{
			cblas_sgemm(CblasRowMajor, transa, transb, m, n, k, 1.f, s->data, tds, t->data, tdt, 0.f, r->data, n);
			break;
		}
		case DOUBLE_REAL:
		{
			cblas_dgemm(CblasRowMajor, transa, transb, m, n, k, 1.0, s->data, tds, t->data, tdt, 0.0, r->data, n);
			break;
		}
		case SINGLE_COMPLEX:
		{
			const scomplex one  = 1;
			const scomplex zero = 0;
			cblas_cgemm(CblasRowMajor, transa, transb, m, n, k, &one, s->data, tds, t->data, tdt, &zero, r->data, n);
			break;
		}
		case DOUBLE_COMPLEX:
		{
			const dcomplex one  = 1;
			const dcomplex zero = 0;
			cblas_zgemm(CblasRowMajor, transa, transb, m, n, k, &one, s->data, tds, t->data, tdt, &zero, r->data, n);
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
/// \brief Multiply (leading or trailing) 'ndim_mult' axes in 's' by 'ndim_mult' axes in 't', scale by 'alpha' and
/// add result to 'r' scaled by beta: r <- alpha * s @ t + beta * r.
/// Whether to use leading or trailing axes is specified by axis range.
///
/// Assuming that 'r' has the appropriate dimensions.
///
void dense_tensor_dot_update(const void* alpha, const struct dense_tensor* restrict s, const enum tensor_axis_range axrange_s, const struct dense_tensor* restrict t, const enum tensor_axis_range axrange_t, const int ndim_mult, struct dense_tensor* restrict r, const void* beta)
{
	// data types must agree
	assert(s->dtype == t->dtype);

	assert(ndim_mult >= 1);
	assert(s->ndim >= ndim_mult && t->ndim >= ndim_mult);
	for (int i = 0; i < ndim_mult; i++)
	{
		assert(s->dim[(axrange_s == TENSOR_AXIS_RANGE_LEADING ? 0 : s->ndim - ndim_mult) + i] ==
		       t->dim[(axrange_t == TENSOR_AXIS_RANGE_LEADING ? 0 : t->ndim - ndim_mult) + i]);
	}
	assert(r->ndim == s->ndim + t->ndim - 2*ndim_mult);
	const int offset_s = (axrange_s == TENSOR_AXIS_RANGE_LEADING ? ndim_mult : 0);
	const int offset_t = (axrange_t == TENSOR_AXIS_RANGE_LEADING ? ndim_mult : 0);
	for (int i = 0; i < s->ndim - ndim_mult; i++)
	{
		assert(r->dim[i] == s->dim[offset_s + i]);
	}
	for (int i = 0; i < t->ndim - ndim_mult; i++)
	{
		assert(r->dim[(s->ndim - ndim_mult) + i] == t->dim[offset_t + i]);
	}

	const int nlds = (axrange_s == TENSOR_AXIS_RANGE_LEADING ? ndim_mult : s->ndim - ndim_mult);
	const int nldt = (axrange_t == TENSOR_AXIS_RANGE_LEADING ? ndim_mult : t->ndim - ndim_mult);

	// leading dimension of 's' as a matrix
	const long lds = integer_product(s->dim, nlds);
	// trailing dimension of 's' as a matrix
	const long tds = integer_product(&s->dim[nlds], s->ndim - nlds);

	// leading dimension of 't' as a matrix
	const long ldt = integer_product(t->dim, nldt);
	// trailing dimension of 't' as a matrix
	const long tdt = integer_product(&t->dim[nldt], t->ndim - nldt);

	const CBLAS_TRANSPOSE transa = (axrange_s == TENSOR_AXIS_RANGE_LEADING ? CblasTrans : CblasNoTrans);
	const CBLAS_TRANSPOSE transb = (axrange_t == TENSOR_AXIS_RANGE_LEADING ? CblasNoTrans : CblasTrans);

	// matrix-matrix multiplication
	const long m = (axrange_s == TENSOR_AXIS_RANGE_LEADING ? tds : lds);
	const long n = (axrange_t == TENSOR_AXIS_RANGE_LEADING ? tdt : ldt);
	const long k = (axrange_s == TENSOR_AXIS_RANGE_LEADING ? lds : tds);
	assert(k == (axrange_t == TENSOR_AXIS_RANGE_LEADING ? ldt : tdt));

	// matrix-matrix multiplication
	switch (s->dtype)
	{
		case SINGLE_REAL:
		{
			cblas_sgemm(CblasRowMajor, transa, transb, m, n, k, *((float*)alpha), s->data, tds, t->data, tdt, *((float*)beta), r->data, n);
			break;
		}
		case DOUBLE_REAL:
		{
			cblas_dgemm(CblasRowMajor, transa, transb, m, n, k, *((double*)alpha), s->data, tds, t->data, tdt, *((double*)beta), r->data, n);
			break;
		}
		case SINGLE_COMPLEX:
		{
			cblas_cgemm(CblasRowMajor, transa, transb, m, n, k, alpha, s->data, tds, t->data, tdt, beta, r->data, n);
			break;
		}
		case DOUBLE_COMPLEX:
		{
			cblas_zgemm(CblasRowMajor, transa, transb, m, n, k, alpha, s->data, tds, t->data, tdt, beta, r->data, n);
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
/// \brief Minimum of two integers.
///
static inline long minl(const long a, const long b)
{
	if (a <= b) {
		return a;
	}
	else {
		return b;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the QR decomposition of the matrix 'a', and store the result in 'q' and 'r' (will be allocated).
/// The matrix dimension between 'q' and 'r' is the minimum of the dimensions of 'a'.
///
int dense_tensor_qr(const struct dense_tensor* restrict a, struct dense_tensor* restrict q, struct dense_tensor* restrict r)
{
	// require a matrix
	assert(a->ndim == 2);

	const long m = a->dim[0];
	const long n = a->dim[1];
	const long k = minl(m, n);

	const long dim_q[2] = { m, k };
	const long dim_r[2] = { k, n };
	allocate_dense_tensor(a->dtype, 2, dim_q, q);
	allocate_dense_tensor(a->dtype, 2, dim_r, r);

	return dense_tensor_qr_fill(a, q, r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the QR decomposition of the matrix 'a', and store the result in 'q' and 'r', which must have been allocated beforehand.
/// The matrix dimension between 'q' and 'r' is the minimum of the dimensions of 'a'.
///
int dense_tensor_qr_fill(const struct dense_tensor* restrict a, struct dense_tensor* restrict q, struct dense_tensor* restrict r)
{
	assert(q->dtype == a->dtype);
	assert(r->dtype == a->dtype);

	// require matrices
	assert(a->ndim == 2);
	assert(q->ndim == 2);
	assert(r->ndim == 2);

	const long m = a->dim[0];
	const long n = a->dim[1];
	const long k = minl(m, n);

	assert(q->dim[0] == m);
	assert(q->dim[1] == k);
	assert(r->dim[0] == k);
	assert(r->dim[1] == n);

	switch (a->dtype)
	{
		case SINGLE_REAL:
		{
			float* tau = aligned_alloc(MEM_DATA_ALIGN, k * sizeof(float));

			if (m >= n)
			{
				// copy 'a' into 'q' (as temporary matrix)
				memcpy(q->data, a->data, m*n * sizeof(float));

				// data entries of 'q' are overwritten
				int info = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, m, n, q->data, n, tau);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'sgeqrf()' failed, return value: %i\n", info);
					return -1;
				}

				// copy entries in upper triangular part into 'r' matrix
				float* qdata = q->data;
				float* rdata = r->data;
				for (long l = 0; l < k; l++)
				{
					for (long j = 0; j < l; j++) {
						rdata[l*n + j] = 0;
					}
					for (long j = l; j < k; j++) {
						rdata[l*n + j] = qdata[l*k + j];
					}
				}
			}
			else  // m < n
			{
				// copy 'a' into 'r' (as temporary matrix)
				memcpy(r->data, a->data, m*n * sizeof(float));

				// data entries of 'r' are overwritten
				int info = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, m, n, r->data, n, tau);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'sgeqrf()' failed, return value: %i\n", info);
					return -1;
				}

				// transfer entries in first k columns below diagonal to 'q'
				float* qdata = q->data;
				float* rdata = r->data;
				for (long l = 0; l < k; l++)
				{
					for (long j = 0; j < l; j++) {
						qdata[l*k + j] = rdata[l*n + j];
						rdata[l*n + j] = 0;
					}
					// set other entries to zero (to avoid NaN test failures)
					for (long j = l; j < k; j++) {
						qdata[l*k + j] = 0;
					}
				}
			}

			// generate the final 'q' matrix
			int info = LAPACKE_sorgqr(LAPACK_ROW_MAJOR, m, k, k, q->data, k, tau);
			if (info != 0) {
				fprintf(stderr, "LAPACK function 'sorgqr()' failed, return value: %i\n", info);
				return -2;
			}

			aligned_free(tau);

			break;
		}
		case DOUBLE_REAL:
		{
			double* tau = aligned_alloc(MEM_DATA_ALIGN, k * sizeof(double));

			if (m >= n)
			{
				// copy 'a' into 'q' (as temporary matrix)
				memcpy(q->data, a->data, m*n * sizeof(double));

				// data entries of 'q' are overwritten
				int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, q->data, n, tau);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'dgeqrf()' failed, return value: %i\n", info);
					return -1;
				}

				// copy entries in upper triangular part into 'r' matrix
				double* qdata = q->data;
				double* rdata = r->data;
				for (long l = 0; l < k; l++)
				{
					for (long j = 0; j < l; j++) {
						rdata[l*n + j] = 0;
					}
					for (long j = l; j < k; j++) {
						rdata[l*n + j] = qdata[l*k + j];
					}
				}
			}
			else  // m < n
			{
				// copy 'a' into 'r' (as temporary matrix)
				memcpy(r->data, a->data, m*n * sizeof(double));

				// data entries of 'r' are overwritten
				int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, r->data, n, tau);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'dgeqrf()' failed, return value: %i\n", info);
					return -1;
				}

				// transfer entries in first k columns below diagonal to 'q'
				double* qdata = q->data;
				double* rdata = r->data;
				for (long l = 0; l < k; l++)
				{
					for (long j = 0; j < l; j++) {
						qdata[l*k + j] = rdata[l*n + j];
						rdata[l*n + j] = 0;
					}
					// set other entries to zero (to avoid NaN test failures)
					for (long j = l; j < k; j++) {
						qdata[l*k + j] = 0;
					}
				}
			}

			// generate the final 'q' matrix
			int info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, k, k, q->data, k, tau);
			if (info != 0) {
				fprintf(stderr, "LAPACK function 'dorgqr()' failed, return value: %i\n", info);
				return -2;
			}

			aligned_free(tau);

			break;
		}
		case SINGLE_COMPLEX:
		{
			scomplex* tau = aligned_alloc(MEM_DATA_ALIGN, k * sizeof(scomplex));

			if (m >= n)
			{
				// copy 'a' into 'q' (as temporary matrix)
				memcpy(q->data, a->data, m*n * sizeof(scomplex));

				// data entries of 'q' are overwritten
				int info = LAPACKE_cgeqrf(LAPACK_ROW_MAJOR, m, n, q->data, n, tau);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'cgeqrf()' failed, return value: %i\n", info);
					return -1;
				}

				// copy entries in upper triangular part into 'r' matrix
				scomplex* qdata = q->data;
				scomplex* rdata = r->data;
				for (long l = 0; l < k; l++)
				{
					for (long j = 0; j < l; j++) {
						rdata[l*n + j] = 0;
					}
					for (long j = l; j < k; j++) {
						rdata[l*n + j] = qdata[l*k + j];
					}
				}
			}
			else  // m < n
			{
				// copy 'a' into 'r' (as temporary matrix)
				memcpy(r->data, a->data, m*n * sizeof(scomplex));

				// data entries of 'r' are overwritten
				int info = LAPACKE_cgeqrf(LAPACK_ROW_MAJOR, m, n, r->data, n, tau);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'cgeqrf()' failed, return value: %i\n", info);
					return -1;
				}

				// transfer entries in first k columns below diagonal to 'q'
				scomplex* qdata = q->data;
				scomplex* rdata = r->data;
				for (long l = 0; l < k; l++)
				{
					for (long j = 0; j < l; j++) {
						qdata[l*k + j] = rdata[l*n + j];
						rdata[l*n + j] = 0;
					}
					// set other entries to zero (to avoid NaN test failures)
					for (long j = l; j < k; j++) {
						qdata[l*k + j] = 0;
					}
				}
			}

			// generate the final 'q' matrix
			int info = LAPACKE_cungqr(LAPACK_ROW_MAJOR, m, k, k, q->data, k, tau);
			if (info != 0) {
				fprintf(stderr, "LAPACK function 'cungqr()' failed, return value: %i\n", info);
				return -2;
			}

			aligned_free(tau);

			break;
		}
		case DOUBLE_COMPLEX:
		{
			dcomplex* tau = aligned_alloc(MEM_DATA_ALIGN, k * sizeof(dcomplex));

			if (m >= n)
			{
				// copy 'a' into 'q' (as temporary matrix)
				memcpy(q->data, a->data, m*n * sizeof(dcomplex));

				// data entries of 'q' are overwritten
				int info = LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, m, n, q->data, n, tau);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'zgeqrf()' failed, return value: %i\n", info);
					return -1;
				}

				// copy entries in upper triangular part into 'r' matrix
				dcomplex* qdata = q->data;
				dcomplex* rdata = r->data;
				for (long l = 0; l < k; l++)
				{
					for (long j = 0; j < l; j++) {
						rdata[l*n + j] = 0;
					}
					for (long j = l; j < k; j++) {
						rdata[l*n + j] = qdata[l*k + j];
					}
				}
			}
			else  // m < n
			{
				// copy 'a' into 'r' (as temporary matrix)
				memcpy(r->data, a->data, m*n * sizeof(dcomplex));

				// data entries of 'r' are overwritten
				int info = LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, m, n, r->data, n, tau);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'zgeqrf()' failed, return value: %i\n", info);
					return -1;
				}

				// transfer entries in first k columns below diagonal to 'q'
				dcomplex* qdata = q->data;
				dcomplex* rdata = r->data;
				for (long l = 0; l < k; l++)
				{
					for (long j = 0; j < l; j++) {
						qdata[l*k + j] = rdata[l*n + j];
						rdata[l*n + j] = 0;
					}
					// set other entries to zero (to avoid NaN test failures)
					for (long j = l; j < k; j++) {
						qdata[l*k + j] = 0;
					}
				}
			}

			// generate the final 'q' matrix
			int info = LAPACKE_zungqr(LAPACK_ROW_MAJOR, m, k, k, q->data, k, tau);
			if (info != 0) {
				fprintf(stderr, "LAPACK function 'zungqr()' failed, return value: %i\n", info);
				return -2;
			}

			aligned_free(tau);

			break;
		}
		default:
		{
			// unknown data type
			assert(false);
		}
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the RQ decomposition (upper triangular times isometry) of the matrix 'a', and store the result in 'r' and 'q' (will be allocated).
/// The matrix dimension between 'r' and 'q' is the minimum of the dimensions of 'a'.
///
int dense_tensor_rq(const struct dense_tensor* restrict a, struct dense_tensor* restrict r, struct dense_tensor* restrict q)
{
	// require a matrix
	assert(a->ndim == 2);

	const long m = a->dim[0];
	const long n = a->dim[1];
	const long k = minl(m, n);

	const long dim_r[2] = { m, k };
	const long dim_q[2] = { k, n };
	allocate_dense_tensor(a->dtype, 2, dim_r, r);
	allocate_dense_tensor(a->dtype, 2, dim_q, q);

	return dense_tensor_rq_fill(a, r, q);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the RQ decomposition (upper triangular times isometry) of the matrix 'a', and store the result in 'r' and 'q', which must have been allocated beforehand.
/// The matrix dimension between 'r' and 'q' is the minimum of the dimensions of 'a'.
///
int dense_tensor_rq_fill(const struct dense_tensor* restrict a, struct dense_tensor* restrict r, struct dense_tensor* restrict q)
{
	assert(q->dtype == a->dtype);
	assert(r->dtype == a->dtype);

	// require matrices
	assert(a->ndim == 2);
	assert(r->ndim == 2);
	assert(q->ndim == 2);

	const long m = a->dim[0];
	const long n = a->dim[1];
	const long k = minl(m, n);

	assert(r->dim[0] == m);
	assert(r->dim[1] == k);
	assert(q->dim[0] == k);
	assert(q->dim[1] == n);

	switch (a->dtype)
	{
		case SINGLE_REAL:
		{
			float* tau = aligned_alloc(MEM_DATA_ALIGN, k * sizeof(float));

			if (n >= m)
			{
				// copy 'a' into 'q' (as temporary matrix)
				memcpy(q->data, a->data, m*n * sizeof(float));

				// data entries of 'q' are overwritten
				int info = LAPACKE_sgerqf(LAPACK_ROW_MAJOR, m, n, q->data, n, tau);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'sgerqf()' failed, return value: %i\n", info);
					return -1;
				}

				// copy entries in upper triangular part into 'r' matrix
				float* qdata = q->data;
				float* rdata = r->data;
				for (long l = 0; l < k; l++)
				{
					for (long j = 0; j < l; j++) {
						rdata[l*k + j] = 0;
					}
					for (long j = l; j < k; j++) {
						rdata[l*k + j] = qdata[l*n + (n - k + j)];
					}
				}
			}
			else  // n < m
			{
				// copy 'a' into 'r' (as temporary matrix)
				memcpy(r->data, a->data, m*n * sizeof(float));

				// data entries of 'r' are overwritten
				int info = LAPACKE_sgerqf(LAPACK_ROW_MAJOR, m, n, r->data, n, tau);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'sgerqf()' failed, return value: %i\n", info);
					return -1;
				}

				// transfer entries in first k columns below diagonal to 'q'
				float* qdata = q->data;
				float* rdata = r->data;
				for (long l = 0; l < k; l++)
				{
					for (long j = 0; j < l; j++) {
						qdata[l*n + j] = rdata[(m - k + l)*k + j];
						rdata[(m - k + l)*k + j] = 0;
					}
					// set other entries to zero (to avoid NaN test failures)
					for (long j = l; j < k; j++) {
						qdata[l*n + j] = 0;
					}
				}
			}

			// generate the final 'q' matrix
			int info = LAPACKE_sorgrq(LAPACK_ROW_MAJOR, k, n, k, q->data, n, tau);
			if (info != 0) {
				fprintf(stderr, "LAPACK function 'sorgrq()' failed, return value: %i\n", info);
				return -2;
			}

			aligned_free(tau);

			break;
		}
		case DOUBLE_REAL:
		{
			double* tau = aligned_alloc(MEM_DATA_ALIGN, k * sizeof(double));

			if (n >= m)
			{
				// copy 'a' into 'q' (as temporary matrix)
				memcpy(q->data, a->data, m*n * sizeof(double));

				// data entries of 'q' are overwritten
				int info = LAPACKE_dgerqf(LAPACK_ROW_MAJOR, m, n, q->data, n, tau);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'dgerqf()' failed, return value: %i\n", info);
					return -1;
				}

				// copy entries in upper triangular part into 'r' matrix
				double* qdata = q->data;
				double* rdata = r->data;
				for (long l = 0; l < k; l++)
				{
					for (long j = 0; j < l; j++) {
						rdata[l*k + j] = 0;
					}
					for (long j = l; j < k; j++) {
						rdata[l*k + j] = qdata[l*n + (n - k + j)];
					}
				}
			}
			else  // n < m
			{
				// copy 'a' into 'r' (as temporary matrix)
				memcpy(r->data, a->data, m*n * sizeof(double));

				// data entries of 'r' are overwritten
				int info = LAPACKE_dgerqf(LAPACK_ROW_MAJOR, m, n, r->data, n, tau);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'dgerqf()' failed, return value: %i\n", info);
					return -1;
				}

				// transfer entries in first k columns below diagonal to 'q'
				double* qdata = q->data;
				double* rdata = r->data;
				for (long l = 0; l < k; l++)
				{
					for (long j = 0; j < l; j++) {
						qdata[l*n + j] = rdata[(m - k + l)*k + j];
						rdata[(m - k + l)*k + j] = 0;
					}
					// set other entries to zero (to avoid NaN test failures)
					for (long j = l; j < k; j++) {
						qdata[l*n + j] = 0;
					}
				}
			}

			// generate the final 'q' matrix
			int info = LAPACKE_dorgrq(LAPACK_ROW_MAJOR, k, n, k, q->data, n, tau);
			if (info != 0) {
				fprintf(stderr, "LAPACK function 'dorgrq()' failed, return value: %i\n", info);
				return -2;
			}

			aligned_free(tau);

			break;
		}
		case SINGLE_COMPLEX:
		{
			scomplex* tau = aligned_alloc(MEM_DATA_ALIGN, k * sizeof(scomplex));

			if (n >= m)
			{
				// copy 'a' into 'q' (as temporary matrix)
				memcpy(q->data, a->data, m*n * sizeof(scomplex));

				// data entries of 'q' are overwritten
				int info = LAPACKE_cgerqf(LAPACK_ROW_MAJOR, m, n, q->data, n, tau);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'cgerqf()' failed, return value: %i\n", info);
					return -1;
				}

				// copy entries in upper triangular part into 'r' matrix
				scomplex* qdata = q->data;
				scomplex* rdata = r->data;
				for (long l = 0; l < k; l++)
				{
					for (long j = 0; j < l; j++) {
						rdata[l*k + j] = 0;
					}
					for (long j = l; j < k; j++) {
						rdata[l*k + j] = qdata[l*n + (n - k + j)];
					}
				}
			}
			else  // n < m
			{
				// copy 'a' into 'r' (as temporary matrix)
				memcpy(r->data, a->data, m*n * sizeof(scomplex));

				// data entries of 'r' are overwritten
				int info = LAPACKE_cgerqf(LAPACK_ROW_MAJOR, m, n, r->data, n, tau);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'cgerqf()' failed, return value: %i\n", info);
					return -1;
				}

				// transfer entries in first k columns below diagonal to 'q'
				scomplex* qdata = q->data;
				scomplex* rdata = r->data;
				for (long l = 0; l < k; l++)
				{
					for (long j = 0; j < l; j++) {
						qdata[l*n + j] = rdata[(m - k + l)*k + j];
						rdata[(m - k + l)*k + j] = 0;
					}
					// set other entries to zero (to avoid NaN test failures)
					for (long j = l; j < k; j++) {
						qdata[l*n + j] = 0;
					}
				}
			}

			// generate the final 'q' matrix
			int info = LAPACKE_cungrq(LAPACK_ROW_MAJOR, k, n, k, q->data, n, tau);
			if (info != 0) {
				fprintf(stderr, "LAPACK function 'cungrq()' failed, return value: %i\n", info);
				return -2;
			}

			aligned_free(tau);

			break;
		}
		case DOUBLE_COMPLEX:
		{
			dcomplex* tau = aligned_alloc(MEM_DATA_ALIGN, k * sizeof(dcomplex));

			if (n >= m)
			{
				// copy 'a' into 'q' (as temporary matrix)
				memcpy(q->data, a->data, m*n * sizeof(dcomplex));

				// data entries of 'q' are overwritten
				int info = LAPACKE_zgerqf(LAPACK_ROW_MAJOR, m, n, q->data, n, tau);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'zgerqf()' failed, return value: %i\n", info);
					return -1;
				}

				// copy entries in upper triangular part into 'r' matrix
				dcomplex* qdata = q->data;
				dcomplex* rdata = r->data;
				for (long l = 0; l < k; l++)
				{
					for (long j = 0; j < l; j++) {
						rdata[l*k + j] = 0;
					}
					for (long j = l; j < k; j++) {
						rdata[l*k + j] = qdata[l*n + (n - k + j)];
					}
				}
			}
			else  // n < m
			{
				// copy 'a' into 'r' (as temporary matrix)
				memcpy(r->data, a->data, m*n * sizeof(dcomplex));

				// data entries of 'r' are overwritten
				int info = LAPACKE_zgerqf(LAPACK_ROW_MAJOR, m, n, r->data, n, tau);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'zgerqf()' failed, return value: %i\n", info);
					return -1;
				}

				// transfer entries in first k columns below diagonal to 'q'
				dcomplex* qdata = q->data;
				dcomplex* rdata = r->data;
				for (long l = 0; l < k; l++)
				{
					for (long j = 0; j < l; j++) {
						qdata[l*n + j] = rdata[(m - k + l)*k + j];
						rdata[(m - k + l)*k + j] = 0;
					}
					// set other entries to zero (to avoid NaN test failures)
					for (long j = l; j < k; j++) {
						qdata[l*n + j] = 0;
					}
				}
			}

			// generate the final 'q' matrix
			int info = LAPACKE_zungrq(LAPACK_ROW_MAJOR, k, n, k, q->data, n, tau);
			if (info != 0) {
				fprintf(stderr, "LAPACK function 'zungrq()' failed, return value: %i\n", info);
				return -2;
			}

			aligned_free(tau);

			break;
		}
		default:
		{
			// unknown data type
			assert(false);
		}
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the "economical" SVD decomposition of the matrix 'a', and store the result in 'u', 's' and 'vh' (will be allocated).
/// The singular values 's' are returned as vector.
///
int dense_tensor_svd(const struct dense_tensor* restrict a, struct dense_tensor* restrict u, struct dense_tensor* restrict s, struct dense_tensor* restrict vh)
{
	// require a matrix
	assert(a->ndim == 2);

	const long m = a->dim[0];
	const long n = a->dim[1];
	const long k = minl(m, n);

	const long dim_u[2]  = { m, k };
	const long dim_s[1]  = { k };
	const long dim_vh[2] = { k, n };
	allocate_dense_tensor(a->dtype, 2, dim_u, u);
	allocate_dense_tensor(numeric_real_type(a->dtype), 1, dim_s, s);
	allocate_dense_tensor(a->dtype, 2, dim_vh, vh);

	return dense_tensor_svd_fill(a, u, s, vh);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the "economical" SVD decomposition of the matrix 'a', and store the result in 'u', 's' and 'vh',
/// which must have been allocated beforehand. The singular values 's' are returned as vector.
///
int dense_tensor_svd_fill(const struct dense_tensor* restrict a, struct dense_tensor* restrict u, struct dense_tensor* restrict s, struct dense_tensor* restrict vh)
{
	assert( u->dtype == a->dtype);
	assert( s->dtype == numeric_real_type(a->dtype));
	assert(vh->dtype == a->dtype);

	assert( a->ndim == 2);
	assert( u->ndim == 2);
	assert( s->ndim == 1);
	assert(vh->ndim == 2);

	const long m = a->dim[0];
	const long n = a->dim[1];
	const long k = minl(m, n);

	assert( u->dim[0] == m);
	assert( u->dim[1] == k);
	assert( s->dim[0] == k);
	assert(vh->dim[0] == k);
	assert(vh->dim[1] == n);

	void* superb = aligned_alloc(MEM_DATA_ALIGN, (k - 1) * sizeof_numeric_type(numeric_real_type(a->dtype)));

	switch (a->dtype)
	{
		case SINGLE_REAL:
		{
			if (m >= n)
			{
				// copy 'a' into 'u'
				memcpy(u->data, a->data, m*n * sizeof(float));

				// data entries of 'u' are overwritten with result
				int info = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'O', 'S', m, n, u->data, k, s->data, NULL, k, vh->data, n, superb);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'sgesvd()' failed, return value: %i\n", info);
					return -1;
				}
			}
			else  // m < n
			{
				// copy 'a' into 'vh'
				memcpy(vh->data, a->data, m*n * sizeof(float));

				// data entries of 'vh' are overwritten with result
				int info = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'S', 'O', m, n, vh->data, n, s->data, u->data, k, NULL, n, superb);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'sgesvd()' failed, return value: %i\n", info);
					return -1;
				}
			}

			break;
		}
		case DOUBLE_REAL:
		{
			if (m >= n)
			{
				// copy 'a' into 'u'
				memcpy(u->data, a->data, m*n * sizeof(double));

				// data entries of 'u' are overwritten with result
				int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'O', 'S', m, n, u->data, k, s->data, NULL, k, vh->data, n, superb);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'dgesvd()' failed, return value: %i\n", info);
					return -1;
				}
			}
			else  // m < n
			{
				// copy 'a' into 'vh'
				memcpy(vh->data, a->data, m*n * sizeof(double));

				// data entries of 'vh' are overwritten with result
				int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'O', m, n, vh->data, n, s->data, u->data, k, NULL, n, superb);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'dgesvd()' failed, return value: %i\n", info);
					return -1;
				}
			}

			break;
		}
		case SINGLE_COMPLEX:
		{
			if (m >= n)
			{
				// copy 'a' into 'u'
				memcpy(u->data, a->data, m*n * sizeof(scomplex));

				// data entries of 'u' are overwritten with result
				int info = LAPACKE_cgesvd(LAPACK_ROW_MAJOR, 'O', 'S', m, n, u->data, k, s->data, NULL, k, vh->data, n, superb);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'cgesvd()' failed, return value: %i\n", info);
					return -1;
				}
			}
			else  // m < n
			{
				// copy 'a' into 'vh'
				memcpy(vh->data, a->data, m*n * sizeof(scomplex));

				// data entries of 'vh' are overwritten with result
				int info = LAPACKE_cgesvd(LAPACK_ROW_MAJOR, 'S', 'O', m, n, vh->data, n, s->data, u->data, k, NULL, n, superb);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'cgesvd()' failed, return value: %i\n", info);
					return -1;
				}
			}

			break;
		}
		case DOUBLE_COMPLEX:
		{
			if (m >= n)
			{
				// copy 'a' into 'u'
				memcpy(u->data, a->data, m*n * sizeof(dcomplex));

				// data entries of 'u' are overwritten with result
				int info = LAPACKE_zgesvd(LAPACK_ROW_MAJOR, 'O', 'S', m, n, u->data, k, s->data, NULL, k, vh->data, n, superb);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'zgesvd()' failed, return value: %i\n", info);
					return -1;
				}
			}
			else  // m < n
			{
				// copy 'a' into 'vh'
				memcpy(vh->data, a->data, m*n * sizeof(dcomplex));

				// data entries of 'vh' are overwritten with result
				int info = LAPACKE_zgesvd(LAPACK_ROW_MAJOR, 'S', 'O', m, n, vh->data, n, s->data, u->data, k, NULL, n, superb);
				if (info != 0) {
					fprintf(stderr, "LAPACK function 'zgesvd()' failed, return value: %i\n", info);
					return -1;
				}
			}

			break;
		}
		default:
		{
			// unknown data type
			assert(false);
		}
	}

	aligned_free(superb);

	return 0;
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
				////#pragma ivdep
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
				////#pragma ivdep
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
				////#pragma ivdep
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
				////#pragma ivdep
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


//________________________________________________________________________________________________________________________
///
/// \brief Test whether two dense tensors agree elementwise within tolerance 'tol'.
///
bool dense_tensor_allclose(const struct dense_tensor* restrict s, const struct dense_tensor* restrict t, const double tol)
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
		if (s->dim[i] != t->dim[i]) {
			return false;
		}
	}

	// compare entries
	const long nelem = dense_tensor_num_elements(s);
	if (uniform_distance(s->dtype, nelem, s->data, t->data) > tol) {
		return false;
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Test whether a dense tensors is close to the identity matrix within tolerance 'tol'.
///
bool dense_tensor_is_identity(const struct dense_tensor* t, const double tol)
{
	// must be a matrix
	if (t->ndim != 2) {
		return false;
	}

	// must be a square matrix
	if (t->dim[0] != t->dim[1]) {
		return false;
	}

	// entries
	const long n = t->dim[0] * t->dim[1];
	switch (t->dtype)
	{
		case SINGLE_REAL:
		{
			const float* data = t->data;
			for (long i = 0; i < n; i++)
			{
				const float ref = (i % (t->dim[0] + 1) == 0 ? 1 : 0);
				if (fabsf(data[i] - ref) > tol) {
					return false;
				}
			}
			break;
		}
		case DOUBLE_REAL:
		{
			const double* data = t->data;
			for (long i = 0; i < n; i++)
			{
				const double ref = (i % (t->dim[0] + 1) == 0 ? 1 : 0);
				if (fabs(data[i] - ref) > tol) {
					return false;
				}
			}
			break;
		}
		case SINGLE_COMPLEX:
		{
			const scomplex* data = t->data;
			for (long i = 0; i < n; i++)
			{
				const scomplex ref = (i % (t->dim[0] + 1) == 0 ? 1 : 0);
				if (cabsf(data[i] - ref) > tol) {
					return false;
				}
			}
			break;
		}
		case DOUBLE_COMPLEX:
		{
			const dcomplex* data = t->data;
			for (long i = 0; i < n; i++)
			{
				const dcomplex ref = (i % (t->dim[0] + 1) == 0 ? 1 : 0);
				if (cabs(data[i] - ref) > tol) {
					return false;
				}
			}
			break;
		}
		default:
		{
			// unknown data type
			assert(false);
			return false;
		}
	}

	return true;
}
