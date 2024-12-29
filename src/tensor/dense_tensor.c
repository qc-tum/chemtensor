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
		t->dim = ct_malloc(ndim * sizeof(long));
		memcpy(t->dim, dim, ndim * sizeof(long));
	}
	else    // ndim == 0
	{
		// ct_malloc(0) not guaranteed to return NULL
		t->dim = NULL;
	}

	const long nelem = dense_tensor_num_elements(t);
	// dimensions must be strictly positive
	assert(nelem > 0);
	t->data = ct_calloc(nelem, sizeof_numeric_type(dtype));
	assert(t->data != NULL);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a dense tensor (free memory).
///
void delete_dense_tensor(struct dense_tensor* t)
{
	ct_free(t->data);
	t->data = NULL;

	if (t->ndim > 0)
	{
		ct_free(t->dim);
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
		case CT_SINGLE_REAL:
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
		case CT_DOUBLE_REAL:
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
		case CT_SINGLE_COMPLEX:
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
		case CT_DOUBLE_COMPLEX:
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
/// \brief Compute the "cyclic" partial trace, by tracing out the 'ndim_trace' leading with the 'ndim_trace' trailing axes.
///
void dense_tensor_cyclic_partial_trace(const struct dense_tensor* t, const int ndim_trace, struct dense_tensor* r)
{
	assert(ndim_trace >= 0);
	assert(t->ndim >= 2 * ndim_trace);
	for (int i = 0; i < ndim_trace; i++) {
		assert(t->dim[i] == t->dim[t->ndim - ndim_trace + i]);
	}

	// construct new tensor 'r'
	allocate_dense_tensor(t->dtype, t->ndim - 2 * ndim_trace, t->dim + ndim_trace, r);

	dense_tensor_cyclic_partial_trace_update(t, ndim_trace, r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the "cyclic" partial trace, by tracing out the 'ndim_trace' leading with the 'ndim_trace' trailing axes,
/// and add result to 'r'.
///
/// 'r' must have been allocated beforehand.
///
void dense_tensor_cyclic_partial_trace_update(const struct dense_tensor* t, const int ndim_trace, struct dense_tensor* r)
{
	assert(ndim_trace >= 0);
	assert(t->dtype == r->dtype);
	assert(r->ndim == t->ndim - 2 * ndim_trace);
	for (int i = 0; i < ndim_trace; i++) {
		assert(t->dim[i] == t->dim[t->ndim - ndim_trace + i]);
	}
	for (int i = 0; i < r->ndim; i++) {
		assert(t->dim[ndim_trace + i] == r->dim[i]);
	}

	const long stride = integer_product(t->dim, ndim_trace);
	const long nr = dense_tensor_num_elements(r);

	switch (t->dtype)
	{
		case CT_SINGLE_REAL:
		{
			const float* tdata = t->data;
			float* rdata = r->data;
			for (long k = 0; k < stride; k++)
			{
				for (long j = 0; j < nr; j++)
				{
					rdata[j] += tdata[(k*nr + j)*stride + k];
				}
			}
			break;
		}
		case CT_DOUBLE_REAL:
		{
			const double* tdata = t->data;
			double* rdata = r->data;
			for (long k = 0; k < stride; k++)
			{
				for (long j = 0; j < nr; j++)
				{
					rdata[j] += tdata[(k*nr + j)*stride + k];
				}
			}
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			const scomplex* tdata = t->data;
			scomplex* rdata = r->data;
			for (long k = 0; k < stride; k++)
			{
				for (long j = 0; j < nr; j++)
				{
					rdata[j] += tdata[(k*nr + j)*stride + k];
				}
			}
			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			const dcomplex* tdata = t->data;
			dcomplex* rdata = r->data;
			for (long k = 0; k < stride; k++)
			{
				for (long j = 0; j < nr; j++)
				{
					rdata[j] += tdata[(k*nr + j)*stride + k];
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
		case CT_SINGLE_REAL:
		{
			return cblas_snrm2(nelem, t->data, 1);
		}
		case CT_DOUBLE_REAL:
		{
			return cblas_dnrm2(nelem, t->data, 1);
		}
		case CT_SINGLE_COMPLEX:
		{
			return cblas_scnrm2(nelem, t->data, 1);
		}
		case CT_DOUBLE_COMPLEX:
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
		case CT_SINGLE_REAL:
		{
			cblas_sscal(dense_tensor_num_elements(t), *((float*)alpha), t->data, 1);
			break;
		}
		case CT_DOUBLE_REAL:
		{
			cblas_dscal(dense_tensor_num_elements(t), *((double*)alpha), t->data, 1);
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			cblas_cscal(dense_tensor_num_elements(t), alpha, t->data, 1);
			break;
		}
		case CT_DOUBLE_COMPLEX:
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
		case CT_SINGLE_REAL:
		{
			cblas_sscal(dense_tensor_num_elements(t), *((float*)alpha), t->data, 1);
			break;
		}
		case CT_DOUBLE_REAL:
		{
			cblas_dscal(dense_tensor_num_elements(t), *((double*)alpha), t->data, 1);
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			cblas_csscal(dense_tensor_num_elements(t), *((float*)alpha), t->data, 1);
			break;
		}
		case CT_DOUBLE_COMPLEX:
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
	ct_free(t->dim);
	t->dim = ct_malloc(ndim * sizeof(long));
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
		case CT_SINGLE_REAL:
		case CT_DOUBLE_REAL:
		{
			// no effect
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			const long nelem = dense_tensor_num_elements(t);
			scomplex* data = t->data;
			for (long i = 0; i < nelem; i++)
			{
				data[i] = conjf(data[i]);
			}
			break;
		}
		case CT_DOUBLE_COMPLEX:
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
		case CT_SINGLE_REAL:
		{
			float* data = t->data;
			memset(data, 0, n*dp * sizeof(float));
			for (long j = 0; j < n; j++)
			{
				data[j*stride] = 1;
			}
			break;
		}
		case CT_DOUBLE_REAL:
		{
			double* data = t->data;
			memset(data, 0, n*dp * sizeof(double));
			for (long j = 0; j < n; j++)
			{
				data[j*stride] = 1;
			}
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			scomplex* data = t->data;
			memset(data, 0, n*dp * sizeof(scomplex));
			for (long j = 0; j < n; j++)
			{
				data[j*stride] = 1;
			}
			break;
		}
		case CT_DOUBLE_COMPLEX:
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
/// \brief Fill the entries of a dense tensor with random normal entries.
///
void dense_tensor_fill_random_normal(const void* alpha, const void* shift, struct rng_state* rng_state, struct dense_tensor* t)
{
	const long nelem = dense_tensor_num_elements(t);

	switch (t->dtype)
	{
		case CT_SINGLE_REAL:
		{
			float* data = t->data;
			const float a = *((float*)alpha);
			const float s = *((float*)shift);
			for (long j = 0; j < nelem; j++) {
				data[j] = a * randnf(rng_state) + s;
			}
			break;
		}
		case CT_DOUBLE_REAL:
		{
			double* data = t->data;
			const double a = *((double*)alpha);
			const double s = *((double*)shift);
			for (long j = 0; j < nelem; j++) {
				data[j] = a * randn(rng_state) + s;
			}
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			scomplex* data = t->data;
			const scomplex a = *((scomplex*)alpha);
			const scomplex s = *((scomplex*)shift);
			for (long j = 0; j < nelem; j++) {
				data[j] = a * crandnf(rng_state) + s;
			}
			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			dcomplex* data = t->data;
			const dcomplex a = *((dcomplex*)alpha);
			const dcomplex s = *((dcomplex*)shift);
			for (long j = 0; j < nelem; j++) {
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
/// \brief Temporary data structure storing tensor dimensions and a permutation of them.
///
struct dimension_permutation
{
	long* dim;  //!< dimensions
	int* perm;  //!< permutation
	int ndim;   //!< number of dimensions (degree)
};


//________________________________________________________________________________________________________________________
///
/// \brief "Squeeze" dimensions by removing dimensions equal to one.
///
static void squeeze_dimensions(const struct dimension_permutation* restrict dp, struct dimension_permutation* restrict dp_eff)
{
	assert(dp->ndim >= 1);

	// map from original to effective axis
	int* axis_map = ct_malloc(dp->ndim * sizeof(int));
	dp_eff->ndim = 0;
	for (int i = 0; i < dp->ndim; i++)
	{
		assert(dp->dim[i] >= 1);
		if (dp->dim[i] == 1) {
			axis_map[i] = -1;
		}
		else {
			axis_map[i] = dp_eff->ndim;
			dp_eff->ndim++;
		}
	}
	if (dp_eff->ndim == 0) {
		// special case: all dimensions are 1
		ct_free(axis_map);
		return;
	}

	// effective dimensions
	dp_eff->dim = ct_malloc(dp_eff->ndim * sizeof(long));
	for (int i = 0; i < dp->ndim; i++) {
		if (dp->dim[i] != 1) {
			dp_eff->dim[axis_map[i]] = dp->dim[i];
		}
	}

	// effective permutation
	dp_eff->perm = ct_malloc(dp_eff->ndim * sizeof(int));
	int c = 0;
	for (int i = 0; i < dp->ndim; i++) {
		if (dp->dim[dp->perm[i]] != 1) {
			dp_eff->perm[c] = axis_map[dp->perm[i]];
			c++;
		}
	}
	assert(c == dp_eff->ndim);

	ct_free(axis_map);
}


//________________________________________________________________________________________________________________________
///
/// \brief Fuse axes which remain neighbors after a permutation.
///
static void fuse_permutation_axes(const struct dimension_permutation* restrict dp, struct dimension_permutation* restrict dp_eff)
{
	assert(dp->ndim >= 1);

	// whether to fuse with previous axis
	int* fuse_map = ct_calloc(dp->ndim, sizeof(int));
	for (int i = 1; i < dp->ndim; i++) {
		if (dp->perm[i] == dp->perm[i - 1] + 1) {
			fuse_map[dp->perm[i]] = true;
		}
	}

	// map from original to effective axis
	int* axis_map = ct_malloc(dp->ndim * sizeof(int));
	dp_eff->ndim = 0;
	for (int i = 0; i < dp->ndim; i++) {
		if (fuse_map[i]) {
			axis_map[i] = axis_map[i - 1];
		}
		else {
			axis_map[i] = dp_eff->ndim;
			dp_eff->ndim++;
		}
	}
	assert(dp_eff->ndim >= 1);

	// effective dimensions
	dp_eff->dim = ct_malloc(dp_eff->ndim * sizeof(long));
	for (int i = 0; i < dp_eff->ndim; i++) {
		dp_eff->dim[i] = 1;
	}
	for (int i = 0; i < dp->ndim; i++) {
		assert(dp->dim[i] >= 1);
		dp_eff->dim[axis_map[i]] *= dp->dim[i];
	}

	// effective permutation
	dp_eff->perm = ct_malloc(dp_eff->ndim * sizeof(int));
	int c = 0;
	for (int i = 0; i < dp->ndim; i++) {
		if (!fuse_map[dp->perm[i]]) {
			dp_eff->perm[c] = axis_map[dp->perm[i]];
			c++;
		}
	}
	assert(c == dp_eff->ndim);

	ct_free(fuse_map);
	ct_free(axis_map);
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

	// dimensions of new tensor 'r'
	long* rdim = ct_malloc(t->ndim * sizeof(long));
	for (int i = 0; i < t->ndim; i++) {
		rdim[i] = t->dim[perm[i]];
	}
	// create new tensor 'r'
	allocate_dense_tensor(t->dtype, t->ndim, rdim, r);
	ct_free(rdim);

	// effective dimensions and permutation
	const struct dimension_permutation dp = { .dim = t->dim, .perm = (int*)perm, .ndim = t->ndim };
	struct dimension_permutation dp_squeeze;
	squeeze_dimensions(&dp, &dp_squeeze);
	if (dp_squeeze.ndim == 0)
	{
		// special case: all dimensions are 1
		// copy single scalar entry
		memcpy(r->data, t->data, sizeof_numeric_type(t->dtype));
		return;
	}
	struct dimension_permutation dp_eff;
	fuse_permutation_axes(&dp_squeeze, &dp_eff);
	ct_free(dp_squeeze.perm);
	ct_free(dp_squeeze.dim);

	// effective dimensions of 'r'
	long* rdim_eff = ct_malloc(dp_eff.ndim * sizeof(long));
	for (int i = 0; i < dp_eff.ndim; i++) {
		rdim_eff[i] = dp_eff.dim[dp_eff.perm[i]];
	}

	// stride (offset between successive elements) in new tensor 'r' corresponding to original last axis
	int ax_r_last = -1;
	for (int i = 0; i < dp_eff.ndim; i++) {
		if (dp_eff.perm[i] == dp_eff.ndim - 1) {
			ax_r_last = i;
			break;
		}
	}
	assert(ax_r_last != -1);
	const long stride = integer_product(rdim_eff + ax_r_last + 1, dp_eff.ndim - ax_r_last - 1);

	const long nelem = integer_product(dp_eff.dim, dp_eff.ndim);
	assert(nelem == dense_tensor_num_elements(t));

	long* index_t = ct_calloc(dp_eff.ndim,  sizeof(long));
	long* index_r = ct_malloc(dp_eff.ndim * sizeof(long));

	for (long ot = 0; ot < nelem; ot += dp_eff.dim[dp_eff.ndim - 1])
	{
		// map index of tensor 't' to index of tensor 'r'
		for (int i = 0; i < dp_eff.ndim; i++) {
			index_r[i] = index_t[dp_eff.perm[i]];
		}
		// convert back to offset of tensor 'r'
		const long or = tensor_index_to_offset(dp_eff.ndim, rdim_eff, index_r);

		// main copy loop
		const long n = dp_eff.dim[dp_eff.ndim - 1];
		switch (t->dtype)
		{
			case CT_SINGLE_REAL:
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
			case CT_DOUBLE_REAL:
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
			case CT_SINGLE_COMPLEX:
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
			case CT_DOUBLE_COMPLEX:
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

		// advance index of tensor 't' by dp_eff.dim[dp_eff.ndim - 1] elements
		next_tensor_index(dp_eff.ndim - 1, dp_eff.dim, index_t);
	}

	// clean up
	ct_free(rdim_eff);
	ct_free(dp_eff.perm);
	ct_free(dp_eff.dim);
	ct_free(index_r);
	ct_free(index_t);
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

	long* rdim = ct_malloc(t->ndim * sizeof(long));
	memcpy(rdim, t->dim, t->ndim * sizeof(long));
	rdim[i_ax] = nind;
	allocate_dense_tensor(t->dtype, t->ndim, rdim, r);
	ct_free(rdim);

	dense_tensor_slice_fill(t, i_ax, ind, nind, r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Slice an axis of the tensor by selecting indices 'ind' along this axis.
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
/// \brief Pad a tensor with zeros according to the specified leading and trailing padding width for each axis.
///
void dense_tensor_pad_zeros(const struct dense_tensor* restrict t, const long* restrict pad_before, const long* restrict pad_after, struct dense_tensor* restrict r)
{
	int ndim_eff = 0;
	for (int i = 0; i < t->ndim; i++)
	{
		assert(pad_before[i] >= 0 && pad_after[i] >= 0);
		if (pad_before[i] > 0 || pad_after[i] > 0) {
			ndim_eff = i + 1;
		}
	}
	if (ndim_eff == 0) {
		// nothing to pad
		copy_dense_tensor(t, r);
		return;
	}

	// dimensions of new tensor
	long* rdim = ct_malloc(t->ndim * sizeof(long));
	for (int i = 0; i < t->ndim; i++) {
		rdim[i] = pad_before[i] + t->dim[i] + pad_after[i];
	}

	allocate_dense_tensor(t->dtype, t->ndim, rdim, r);

	ct_free(rdim);

	// leading dimensions
	const long ld = integer_product(t->dim, ndim_eff - 1);
	// trailing dimensions times data type size
	const long tdd = integer_product(&t->dim[ndim_eff], t->ndim - ndim_eff) * sizeof_numeric_type(t->dtype);

	const long stride = t->dim[ndim_eff - 1] * tdd;

	long* index_t = ct_calloc(ndim_eff, sizeof(long));
	long* index_r = ct_calloc(ndim_eff, sizeof(long));
	for (long ot = 0; ot < ld; ot++, next_tensor_index(ndim_eff - 1, t->dim, index_t))
	{
		for (int i = 0; i < ndim_eff; i++) {
			index_r[i] = pad_before[i] + index_t[i];
		}
		const long or = tensor_index_to_offset(ndim_eff, r->dim, index_r);
		memcpy((int8_t*)r->data + or * tdd,
		       (int8_t*)t->data + ot * stride, stride);
	}

	ct_free(index_r);
	ct_free(index_t);
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
		case CT_SINGLE_REAL:
		{
			cblas_saxpy(nelem, *((float*)alpha), s->data, 1, t->data, 1);
			break;
		}
		case CT_DOUBLE_REAL:
		{
			cblas_daxpy(nelem, *((double*)alpha), s->data, 1, t->data, 1);
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			cblas_caxpy(nelem, alpha, s->data, 1, t->data, 1);
			break;
		}
		case CT_DOUBLE_COMPLEX:
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
			case CT_SINGLE_REAL:
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
			case CT_DOUBLE_REAL:
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
			case CT_SINGLE_COMPLEX:
			{
				const scomplex* sdata = s->data;
				scomplex*       rdata = r->data;
				if (t->dtype == CT_SINGLE_COMPLEX)
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
					assert(t->dtype == CT_SINGLE_REAL);
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
			case CT_DOUBLE_COMPLEX:
			{
				const dcomplex* sdata = s->data;
				dcomplex*       rdata = r->data;
				if (t->dtype == CT_DOUBLE_COMPLEX)
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
					assert(t->dtype == CT_DOUBLE_REAL);
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
			case CT_SINGLE_REAL:
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
			case CT_DOUBLE_REAL:
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
			case CT_SINGLE_COMPLEX:
			{
				const scomplex* sdata = s->data;
				scomplex*       rdata = r->data;
				if (t->dtype == CT_SINGLE_COMPLEX)
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
					assert(t->dtype == CT_SINGLE_REAL);
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
			case CT_DOUBLE_COMPLEX:
			{
				const dcomplex* sdata = s->data;
				dcomplex*       rdata = r->data;
				if (t->dtype == CT_DOUBLE_COMPLEX)
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
					assert(t->dtype == CT_DOUBLE_REAL);
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
/// \brief Multiply the 'i_ax' axis of 's' with the leading or trailing axis of 't', preserving the overall dimension ordering of 's'.
///
void dense_tensor_multiply_axis(const struct dense_tensor* restrict s, const int i_ax, const struct dense_tensor* restrict t, const enum tensor_axis_range axrange_t, struct dense_tensor* restrict r)
{
	// data types must agree
	assert(s->dtype == t->dtype);
	// 't' must have a degree of at least 1
	assert(t->ndim >= 1);
	// 'i_ax' must be a valid axis index for 's'
	assert(0 <= i_ax && i_ax < s->ndim);
	// to-be contracted dimensions must match
	assert(s->dim[i_ax] == t->dim[axrange_t == TENSOR_AXIS_RANGE_LEADING ? 0 : t->ndim - 1]);

	// allocate output tensor
	{
		long* rdim = ct_malloc((s->ndim + t->ndim - 2) * sizeof(long));
		memcpy( rdim, s->dim, i_ax * sizeof(long));
		memcpy(&rdim[i_ax], &t->dim[axrange_t == TENSOR_AXIS_RANGE_LEADING ? 1 : 0], (t->ndim - 1) * sizeof(long));
		memcpy(&rdim[i_ax + t->ndim - 1], &s->dim[i_ax + 1], (s->ndim - i_ax - 1) * sizeof(long));
		allocate_dense_tensor(s->dtype, s->ndim + t->ndim - 2, rdim, r);
		ct_free(rdim);
	}

	dense_tensor_multiply_axis_update(numeric_one(s->dtype), s, i_ax, t, axrange_t, numeric_zero(s->dtype), r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Multiply the 'i_ax' axis of 's' with the leading or trailing axis of 't', preserving the overall dimension ordering of 's',
/// scale by 'alpha' and add result to 'r' scaled by beta: r <- alpha * s @ t + beta * r.
///
/// Assuming that 'r' has the appropriate dimensions.
///
void dense_tensor_multiply_axis_update(const void* alpha, const struct dense_tensor* restrict s, const int i_ax, const struct dense_tensor* restrict t, const enum tensor_axis_range axrange_t, const void* beta, struct dense_tensor* restrict r)
{
	// data types must agree
	assert(s->dtype == t->dtype);
	assert(s->dtype == r->dtype);
	// 't' must have a degree of at least 1
	assert(t->ndim >= 1);
	// 'i_ax' must be a valid axis index for 's'
	assert(0 <= i_ax && i_ax < s->ndim);
	// to-be contracted dimensions must match
	assert(s->dim[i_ax] == t->dim[axrange_t == TENSOR_AXIS_RANGE_LEADING ? 0 : t->ndim - 1]);
	// 'r' must have appropriate degree and dimensions
	assert(r->ndim == s->ndim + t->ndim - 2);
	for (int i = 0; i < i_ax; i++) {
		assert(r->dim[i] == s->dim[i]);
	}
	for (int i = 0; i < t->ndim - 1; i++) {
		assert(r->dim[i_ax + i] == t->dim[(axrange_t == TENSOR_AXIS_RANGE_LEADING ? 1 : 0) + i]);
	}
	for (int i = i_ax + 1; i < s->ndim; i++) {
		assert(r->dim[t->ndim - 2 + i] == s->dim[i]);
	}

	// outer dimension of 's' and 'r' as a matrix
	const long dim_outer = integer_product(s->dim, i_ax);
	// inner dimension of 's' as a matrix
	const long dim_inner_s = integer_product(&s->dim[i_ax], s->ndim - i_ax);
	// inner dimension of 'r' as a matrix
	const long dim_inner_r = integer_product(&r->dim[i_ax], r->ndim - i_ax);
	// trailing dimension of 't' as a matrix
	const long tdt = (axrange_t == TENSOR_AXIS_RANGE_LEADING ? integer_product(&t->dim[1], t->ndim - 1) : t->dim[t->ndim - 1]);

	const CBLAS_TRANSPOSE transa = (axrange_t == TENSOR_AXIS_RANGE_LEADING ? CblasTrans : CblasNoTrans);

	const long m = integer_product(&r->dim[i_ax], t->ndim - 1);
	const long k = s->dim[i_ax];
	const long n = integer_product(&s->dim[i_ax + 1], s->ndim - i_ax - 1);

	switch (s->dtype)
	{
		case CT_SINGLE_REAL:
		{
			const float* sdata = s->data;
			float*       rdata = r->data;
			for (long j = 0; j < dim_outer; j++)
			{
				cblas_sgemm(CblasRowMajor, transa, CblasNoTrans, m, n, k, *((float*)alpha), t->data, tdt, &sdata[j*dim_inner_s], n, *((float*)beta), &rdata[j*dim_inner_r], n);
			}
			break;
		}
		case CT_DOUBLE_REAL:
		{
			const double* sdata = s->data;
			double*       rdata = r->data;
			for (long j = 0; j < dim_outer; j++)
			{
				cblas_dgemm(CblasRowMajor, transa, CblasNoTrans, m, n, k, *((double*)alpha), t->data, tdt, &sdata[j*dim_inner_s], n, *((double*)beta), &rdata[j*dim_inner_r], n);
			}
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			const scomplex* sdata = s->data;
			scomplex*       rdata = r->data;
			for (long j = 0; j < dim_outer; j++)
			{
				cblas_cgemm(CblasRowMajor, transa, CblasNoTrans, m, n, k, alpha, t->data, tdt, &sdata[j*dim_inner_s], n, beta, &rdata[j*dim_inner_r], n);
			}
			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			const dcomplex* sdata = s->data;
			dcomplex*       rdata = r->data;
			for (long j = 0; j < dim_outer; j++)
			{
				cblas_zgemm(CblasRowMajor, transa, CblasNoTrans, m, n, k, alpha, t->data, tdt, &sdata[j*dim_inner_s], n, beta, &rdata[j*dim_inner_r], n);
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
	long* rdim = ct_malloc(ndimr * sizeof(long));
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
	ct_free(rdim);

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
		case CT_SINGLE_REAL:
		{
			cblas_sgemm(CblasRowMajor, transa, transb, m, n, k, 1.f, s->data, tds, t->data, tdt, 0.f, r->data, n);
			break;
		}
		case CT_DOUBLE_REAL:
		{
			cblas_dgemm(CblasRowMajor, transa, transb, m, n, k, 1.0, s->data, tds, t->data, tdt, 0.0, r->data, n);
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			const scomplex one  = 1;
			const scomplex zero = 0;
			cblas_cgemm(CblasRowMajor, transa, transb, m, n, k, &one, s->data, tds, t->data, tdt, &zero, r->data, n);
			break;
		}
		case CT_DOUBLE_COMPLEX:
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
void dense_tensor_dot_update(const void* alpha, const struct dense_tensor* restrict s, const enum tensor_axis_range axrange_s, const struct dense_tensor* restrict t, const enum tensor_axis_range axrange_t, const int ndim_mult, const void* beta, struct dense_tensor* restrict r)
{
	// data types must agree
	assert(s->dtype == t->dtype);
	assert(s->dtype == r->dtype);

	assert(ndim_mult >= 1);
	assert(s->ndim >= ndim_mult && t->ndim >= ndim_mult);
	for (int i = 0; i < ndim_mult; i++)
	{
		assert(s->dim[(axrange_s == TENSOR_AXIS_RANGE_LEADING ? 0 : s->ndim - ndim_mult) + i] ==
		       t->dim[(axrange_t == TENSOR_AXIS_RANGE_LEADING ? 0 : t->ndim - ndim_mult) + i]);
	}
	assert(r->ndim == s->ndim + t->ndim - 2*ndim_mult);
	#ifndef NDEBUG
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
	#endif

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
		case CT_SINGLE_REAL:
		{
			cblas_sgemm(CblasRowMajor, transa, transb, m, n, k, *((float*)alpha), s->data, tds, t->data, tdt, *((float*)beta), r->data, n);
			break;
		}
		case CT_DOUBLE_REAL:
		{
			cblas_dgemm(CblasRowMajor, transa, transb, m, n, k, *((double*)alpha), s->data, tds, t->data, tdt, *((double*)beta), r->data, n);
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			cblas_cgemm(CblasRowMajor, transa, transb, m, n, k, alpha, s->data, tds, t->data, tdt, beta, r->data, n);
			break;
		}
		case CT_DOUBLE_COMPLEX:
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
			case CT_SINGLE_REAL:
			{
				const float* sdata = s->data;
				const float* tdata = t->data;
				float*       rdata = r->data;
				rdata[0] = sdata[0] * tdata[0];
				break;
			}
			case CT_DOUBLE_REAL:
			{
				const double* sdata = s->data;
				const double* tdata = t->data;
				double*       rdata = r->data;
				rdata[0] = sdata[0] * tdata[0];
				break;
			}
			case CT_SINGLE_COMPLEX:
			{
				const scomplex* sdata = s->data;
				const scomplex* tdata = t->data;
				scomplex*       rdata = r->data;
				rdata[0] = sdata[0] * tdata[0];
				break;
			}
			case CT_DOUBLE_COMPLEX:
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
	long* rdim_il = ct_malloc((s->ndim + t->ndim) * sizeof(long));
	for (int i = 0; i < s->ndim; i++)
	{
		rdim_il[2*i  ] = s->dim[i];
		rdim_il[2*i+1] = t->dim[i];
	}
	allocate_dense_tensor(s->dtype, s->ndim + t->ndim, rdim_il, r);
	ct_free(rdim_il);

	long* index_s = ct_calloc(s->ndim, sizeof(long));
	long* index_t = ct_calloc(t->ndim, sizeof(long));
	long* index_r = ct_calloc(r->ndim, sizeof(long));

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
			case CT_SINGLE_REAL:
			{
				cblas_sger(CblasRowMajor, last_dim_s, last_dim_t, 1.f, (const float*)s->data + os, 1, (const float*)t->data + ot, 1, (float*)r->data + or, last_dim_t);
				break;
			}
			case CT_DOUBLE_REAL:
			{
				cblas_dger(CblasRowMajor, last_dim_s, last_dim_t, 1.0, (const double*)s->data + os, 1, (const double*)t->data + ot, 1, (double*)r->data + or, last_dim_t);
				break;
			}
			case CT_SINGLE_COMPLEX:
			{
				const scomplex one = 1;
				cblas_cgeru(CblasRowMajor, last_dim_s, last_dim_t, &one, (const scomplex*)s->data + os, 1, (const scomplex*)t->data + ot, 1, (scomplex*)r->data + or, last_dim_t);
				break;
			}
			case CT_DOUBLE_COMPLEX:
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

	ct_free(index_r);
	ct_free(index_t);
	ct_free(index_s);

	// actual dimensions of 'r'
	long* rdim = ct_malloc(s->ndim * sizeof(long));
	for (int i = 0; i < s->ndim; i++)
	{
		rdim[i] = s->dim[i] * t->dim[i];
	}
	reshape_dense_tensor(s->ndim, rdim, r);
	ct_free(rdim);
}


//________________________________________________________________________________________________________________________
///
/// \brief Concatenate tensors along the specified axis. All other dimensions must respectively agree.
///
void dense_tensor_concatenate(const struct dense_tensor* restrict tlist, const int num_tensors, const int i_ax, struct dense_tensor* restrict r)
{
	// consistency and compatibility checks
	assert(num_tensors >= 1);
	assert(0 <= i_ax && i_ax < tlist[0].ndim);
	for (int j = 0; j < num_tensors - 1; j++)
	{
		// data types must match
		assert(tlist[j].dtype == tlist[j + 1].dtype);
		// degrees must match
		assert(tlist[j].ndim == tlist[j + 1].ndim);
	}

	// dimensions of new tensor
	long dim_concat = 0;
	for (int j = 0; j < num_tensors; j++) {
		dim_concat += tlist[j].dim[i_ax];
	}
	long* rdim = ct_malloc(tlist[0].ndim * sizeof(long));
	for (int i = 0; i < tlist[0].ndim; i++)
	{
		rdim[i] = (i == i_ax ? dim_concat : tlist[0].dim[i]);
	}

	// allocate new tensor
	allocate_dense_tensor(tlist[0].dtype, tlist[0].ndim, rdim, r);

	ct_free(rdim);

	dense_tensor_concatenate_fill(tlist, num_tensors, i_ax, r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Concatenate tensors along the specified axis. All other dimensions must respectively agree.
///
/// 'r' must have been allocated beforehand with the correct data type and dimensions.
///
void dense_tensor_concatenate_fill(const struct dense_tensor* restrict tlist, const int num_tensors, const int i_ax, struct dense_tensor* restrict r)
{
	// consistency and compatibility checks
	assert(num_tensors >= 1);
	assert(r->dtype == tlist[0].dtype);
	assert(r->ndim  == tlist[0].ndim);
	assert(0 <= i_ax && i_ax < r->ndim);
	for (int j = 0; j < num_tensors - 1; j++)
	{
		// data types must match
		assert(tlist[j].dtype == tlist[j + 1].dtype);
		// degrees must match
		assert(tlist[j].ndim == tlist[j + 1].ndim);
		for (int i = 0; i < tlist[j].ndim; i++) {
			if (i != i_ax) {
				// other dimensions must match
				assert(tlist[j].dim[i] == tlist[j + 1].dim[i]);
			}
		}
	}
	long dim_concat = 0;
	for (int j = 0; j < num_tensors; j++) {
		dim_concat += tlist[j].dim[i_ax];
	}
	for (int i = 0; i < r->ndim; i++)
	{
		assert(r->dim[i] == (i == i_ax ? dim_concat : tlist[0].dim[i]));
	}

	// leading dimensions
	const long ld = integer_product(r->dim, i_ax);
	// trailing dimensions times data type size
	const long tdd = integer_product(&r->dim[i_ax + 1], r->ndim - (i_ax + 1)) * sizeof_numeric_type(r->dtype);

	long offset = 0;
	for (int j = 0; j < num_tensors; j++)
	{
		const long stride = tlist[j].dim[i_ax] * tdd;

		for (long i = 0; i < ld; i++) {
			memcpy((int8_t*)r->data + (i * r->dim[i_ax] + offset) * tdd,
			       (int8_t*)tlist[j].data + i * stride, stride);
		}

		offset += tlist[j].dim[i_ax];
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compose tensors along the specified axes, creating a block-diagonal pattern. All other dimensions must respectively agree.
///
void dense_tensor_block_diag(const struct dense_tensor* restrict tlist, const int num_tensors, const int* i_ax, const int ndim_block, struct dense_tensor* restrict r)
{
	// consistency and compatibility checks
	assert(num_tensors >= 1);
	const int ndim = tlist[0].ndim;
	assert(1 <= ndim_block && ndim_block <= ndim);
	for (int j = 0; j < num_tensors - 1; j++)
	{
		// data types must match
		assert(tlist[j].dtype == tlist[j + 1].dtype);
		// degrees must match
		assert(tlist[j].ndim == tlist[j + 1].ndim);
	}

	bool* i_ax_indicator = ct_calloc(ndim, sizeof(bool));
	for (int i = 0; i < ndim_block; i++)
	{
		assert(0 <= i_ax[i] && i_ax[i] < ndim);
		assert(!i_ax_indicator[i_ax[i]]);  // axis index can only appear once
		i_ax_indicator[i_ax[i]] = true;
	}

	// dimensions of new tensor
	long* rdim = ct_malloc(ndim * sizeof(long));
	for (int i = 0; i < ndim; i++)
	{
		if (i_ax_indicator[i]) {
			long dsum = 0;
			for (int j = 0; j < num_tensors; j++) {
				dsum += tlist[j].dim[i];
			}
			rdim[i] = dsum;
		}
		else {
			rdim[i] = tlist[0].dim[i];
		}
	}

	// allocate new tensor
	allocate_dense_tensor(tlist[0].dtype, ndim, rdim, r);

	ct_free(rdim);
	ct_free(i_ax_indicator);

	dense_tensor_block_diag_fill(tlist, num_tensors, i_ax, ndim_block, r);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compose tensors along the specified axes, creating a block-diagonal pattern. All other dimensions must respectively agree.
///
/// 'r' must have been allocated beforehand with the correct data type and dimensions.
///
void dense_tensor_block_diag_fill(const struct dense_tensor* restrict tlist, const int num_tensors, const int* i_ax, const int ndim_block, struct dense_tensor* restrict r)
{
	// consistency and compatibility checks
	assert(num_tensors >= 1);
	assert(r->dtype == tlist[0].dtype);
	assert(r->ndim  == tlist[0].ndim);
	assert(1 <= ndim_block && ndim_block <= r->ndim);

	bool* i_ax_indicator = ct_calloc(r->ndim, sizeof(bool));
	for (int i = 0; i < ndim_block; i++)
	{
		assert(0 <= i_ax[i] && i_ax[i] < r->ndim);
		assert(!i_ax_indicator[i_ax[i]]);  // axis index can only appear once
		i_ax_indicator[i_ax[i]] = true;
	}

	for (int j = 0; j < num_tensors - 1; j++)
	{
		// data types must match
		assert(tlist[j].dtype == tlist[j + 1].dtype);
		// degrees must match
		assert(tlist[j].ndim == tlist[j + 1].ndim);
		for (int i = 0; i < tlist[j].ndim; i++) {
			if (!i_ax_indicator[i]) {
				// other dimensions must match
				assert(tlist[j].dim[i] == tlist[j + 1].dim[i]);
			}
		}
	}
	for (int i = 0; i < r->ndim; i++)
	{
		if (i_ax_indicator[i]) {
			long dsum = 0;
			for (int j = 0; j < num_tensors; j++) {
				dsum += tlist[j].dim[i];
			}
			assert(r->dim[i] == dsum);
		}
		else {
			assert(r->dim[i] == tlist[0].dim[i]);
		}
	}

	// effective dimensions used for indexing tensor slices
	int ndim_eff = 0;
	for (int i = 0; i < r->ndim; i++) {
		if (i_ax_indicator[i]) {
			ndim_eff = i + 1;
		}
	}
	assert(ndim_eff >= 1);

	// trailing dimensions times data type size
	const long tdd = integer_product(&r->dim[ndim_eff], r->ndim - ndim_eff) * sizeof_numeric_type(r->dtype);

	long* offset = ct_calloc(ndim_eff, sizeof(long));

	long* index_t = ct_calloc(ndim_eff, sizeof(long));
	long* index_r = ct_calloc(ndim_eff, sizeof(long));

	for (int j = 0; j < num_tensors; j++)
	{
		// leading dimensions
		const long ld = integer_product(tlist[j].dim, ndim_eff - 1);

		const long stride = tlist[j].dim[ndim_eff - 1] * tdd;

		memset(index_t, 0, ndim_eff * sizeof(long));
		for (long ot = 0; ot < ld; ot++, next_tensor_index(ndim_eff - 1, tlist[j].dim, index_t))
		{
			for (int i = 0; i < ndim_eff; i++) {
				index_r[i] = offset[i] + index_t[i];
				assert(index_r[i] < r->dim[i]);
			}
			const long or = tensor_index_to_offset(ndim_eff, r->dim, index_r);
			memcpy((int8_t*)r->data + or * tdd,
			       (int8_t*)tlist[j].data + ot * stride, stride);
		}

		for (int i = 0; i < ndim_eff; i++) {
			if (i_ax_indicator[i]) {
				offset[i] += tlist[j].dim[i];
			}
		}
	}

	ct_free(index_r);
	ct_free(index_t);
	ct_free(offset);
	ct_free(i_ax_indicator);
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
	const long k = lmin(m, n);

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
	const long k = lmin(m, n);

	assert(q->dim[0] == m);
	assert(q->dim[1] == k);
	assert(r->dim[0] == k);
	assert(r->dim[1] == n);

	switch (a->dtype)
	{
		case CT_SINGLE_REAL:
		{
			float* tau = ct_malloc(k * sizeof(float));

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

			ct_free(tau);

			break;
		}
		case CT_DOUBLE_REAL:
		{
			double* tau = ct_malloc(k * sizeof(double));

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

			ct_free(tau);

			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			scomplex* tau = ct_malloc(k * sizeof(scomplex));

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

			ct_free(tau);

			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			dcomplex* tau = ct_malloc(k * sizeof(dcomplex));

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

			ct_free(tau);

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
	const long k = lmin(m, n);

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
	const long k = lmin(m, n);

	assert(r->dim[0] == m);
	assert(r->dim[1] == k);
	assert(q->dim[0] == k);
	assert(q->dim[1] == n);

	switch (a->dtype)
	{
		case CT_SINGLE_REAL:
		{
			float* tau = ct_malloc(k * sizeof(float));

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

			ct_free(tau);

			break;
		}
		case CT_DOUBLE_REAL:
		{
			double* tau = ct_malloc(k * sizeof(double));

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

			ct_free(tau);

			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			scomplex* tau = ct_malloc(k * sizeof(scomplex));

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

			ct_free(tau);

			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			dcomplex* tau = ct_malloc(k * sizeof(dcomplex));

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

			ct_free(tau);

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
/// \brief Compute the eigenvalues and -vectors of a (real symmetric or complex Hermitian) matrix 'a',
/// and store the result in 'u' and 'lambda' (will be allocated).
/// The eigenvalues are real and returned as vector.
///
int dense_tensor_eigh(const struct dense_tensor* restrict a, struct dense_tensor* restrict u, struct dense_tensor* restrict lambda)
{
	// require a square matrix
	assert(a->ndim == 2);
	assert(a->dim[0] == a->dim[1]);

	const long dim_lambda[1]  = { a->dim[0] };
	allocate_dense_tensor(a->dtype, a->ndim, a->dim, u);
	allocate_dense_tensor(numeric_real_type(a->dtype), 1, dim_lambda, lambda);

	return dense_tensor_eigh_fill(a, u, lambda);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the eigenvalues and -vectors of a (real symmetric or complex Hermitian) matrix 'a',
/// and store the result in 'u' and 'lambda', which must have been allocated beforehand.
/// The eigenvalues are real and returned as vector.
///
int dense_tensor_eigh_fill(const struct dense_tensor* restrict a, struct dense_tensor* restrict u, struct dense_tensor* restrict lambda)
{
	assert(u->dtype == a->dtype);
	assert(lambda->dtype == numeric_real_type(a->dtype));

	assert(a->ndim == 2);
	assert(u->ndim == 2);
	assert(lambda->ndim == 1);

	assert(a->dim[0] == a->dim[1]);
	const long n = a->dim[0];

	assert(u->dim[0] == n);
	assert(u->dim[1] == n);
	assert(lambda->dim[0] == n);

	switch (a->dtype)
	{
		case CT_SINGLE_REAL:
		{
			// copy 'a' into 'u'
			memcpy(u->data, a->data, n*n * sizeof(float));

			// data entries of 'u' are overwritten with result
			int info = LAPACKE_ssyevd(LAPACK_ROW_MAJOR, 'V', 'U', n, u->data, n, lambda->data);
			if (info != 0) {
				fprintf(stderr, "LAPACK function 'ssyevd()' failed, return value: %i\n", info);
				return -1;
			}

			break;
		}
		case CT_DOUBLE_REAL:
		{
			// copy 'a' into 'u'
			memcpy(u->data, a->data, n*n * sizeof(double));

			// data entries of 'u' are overwritten with result
			int info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', n, u->data, n, lambda->data);
			if (info != 0) {
				fprintf(stderr, "LAPACK function 'dsyevd()' failed, return value: %i\n", info);
				return -1;
			}

			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			// copy 'a' into 'u'
			memcpy(u->data, a->data, n*n * sizeof(scomplex));

			// data entries of 'u' are overwritten with result
			int info = LAPACKE_cheevd(LAPACK_ROW_MAJOR, 'V', 'U', n, u->data, n, lambda->data);
			if (info != 0) {
				fprintf(stderr, "LAPACK function 'cheevd()' failed, return value: %i\n", info);
				return -1;
			}

			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			// copy 'a' into 'u'
			memcpy(u->data, a->data, n*n * sizeof(dcomplex));

			// data entries of 'u' are overwritten with result
			int info = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'U', n, u->data, n, lambda->data);
			if (info != 0) {
				fprintf(stderr, "LAPACK function 'zheevd()' failed, return value: %i\n", info);
				return -1;
			}

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
	const long k = lmin(m, n);

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
	const long k = lmin(m, n);

	assert( u->dim[0] == m);
	assert( u->dim[1] == k);
	assert( s->dim[0] == k);
	assert(vh->dim[0] == k);
	assert(vh->dim[1] == n);

	// 'lmax' to avoid invalid memory allocation of zero bytes
	void* superb = ct_malloc(lmax((k - 1) * sizeof_numeric_type(numeric_real_type(a->dtype)), 1));

	switch (a->dtype)
	{
		case CT_SINGLE_REAL:
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
		case CT_DOUBLE_REAL:
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
		case CT_SINGLE_COMPLEX:
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
		case CT_DOUBLE_COMPLEX:
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

	ct_free(superb);

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

	long* index_t = ct_calloc(t->ndim, sizeof(long));
	long* index_b = ct_calloc(b->ndim, sizeof(long));

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
			case CT_SINGLE_REAL:
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
			case CT_DOUBLE_REAL:
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
			case CT_SINGLE_COMPLEX:
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
			case CT_DOUBLE_COMPLEX:
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
	ct_free(index_b);
	ct_free(index_t);
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
/// \brief Test whether a dense tensors is entrywise zero within tolerance 'tol'.
///
/// To test whether the tensor is exactly zero, set 'tol = 0.0'.
///
bool dense_tensor_is_zero(const struct dense_tensor* t, const double tol)
{
	const long nelem = dense_tensor_num_elements(t);

	switch (t->dtype)
	{
		case CT_SINGLE_REAL:
		{
			const float* data = t->data;
			for (long i = 0; i < nelem; i++) {
				if (fabsf(data[i]) > tol) {
					return false;
				}
			}
			break;
		}
		case CT_DOUBLE_REAL:
		{
			const double* data = t->data;
			for (long i = 0; i < nelem; i++) {
				if (fabs(data[i]) > tol) {
					return false;
				}
			}
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			const scomplex* data = t->data;
			for (long i = 0; i < nelem; i++) {
				if (cabsf(data[i]) > tol) {
					return false;
				}
			}
			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			const dcomplex* data = t->data;
			for (long i = 0; i < nelem; i++) {
				if (cabs(data[i]) > tol) {
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
		case CT_SINGLE_REAL:
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
		case CT_DOUBLE_REAL:
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
		case CT_SINGLE_COMPLEX:
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
		case CT_DOUBLE_COMPLEX:
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


//________________________________________________________________________________________________________________________
///
/// \brief Test whether a dense tensors is self-adjoint (a real symmetric or complex Hermitian matrix) within tolerance 'tol'.
///
bool dense_tensor_is_self_adjoint(const struct dense_tensor* t, const double tol)
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
	switch (t->dtype)
	{
		case CT_SINGLE_REAL:
		{
			const float* data = t->data;
			for (long i = 0; i < t->dim[0]; i++) {
				for (long j = i + 1; j < t->dim[1]; j++) {
					if (fabsf(data[i*t->dim[1] + j] - data[j*t->dim[1] + i]) > tol) {
						return false;
					}
				}
			}
			break;
		}
		case CT_DOUBLE_REAL:
		{
			const double* data = t->data;
			for (long i = 0; i < t->dim[0]; i++) {
				for (long j = i + 1; j < t->dim[1]; j++) {
					if (fabs(data[i*t->dim[1] + j] - data[j*t->dim[1] + i]) > tol) {
						return false;
					}
				}
			}
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			const scomplex* data = t->data;
			for (long i = 0; i < t->dim[0]; i++) {
				for (long j = i; j < t->dim[1]; j++) {
					if (cabsf(data[i*t->dim[1] + j] - conjf(data[j*t->dim[1] + i])) > tol) {
						return false;
					}
				}
			}
			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			const dcomplex* data = t->data;
			for (long i = 0; i < t->dim[0]; i++) {
				for (long j = i; j < t->dim[1]; j++) {
					if (cabs(data[i*t->dim[1] + j] - conj(data[j*t->dim[1] + i])) > tol) {
						return false;
					}
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


//________________________________________________________________________________________________________________________
///
/// \brief Test whether a dense tensors is an isometry within tolerance 'tol'.
///
bool dense_tensor_is_isometry(const struct dense_tensor* t, const double tol, const bool transpose)
{
	// must be a matrix
	if (t->ndim != 2) {
		return false;
	}

	bool is_isometry;
	if (t->dtype == CT_SINGLE_REAL || t->dtype == CT_DOUBLE_REAL)
	{
		struct dense_tensor t2;
		if (!transpose) {
			dense_tensor_dot(t, TENSOR_AXIS_RANGE_LEADING, t, TENSOR_AXIS_RANGE_LEADING, 1, &t2);
		}
		else {
			dense_tensor_dot(t, TENSOR_AXIS_RANGE_TRAILING, t, TENSOR_AXIS_RANGE_TRAILING, 1, &t2);
		}

		is_isometry = dense_tensor_is_identity(&t2, tol);

		delete_dense_tensor(&t2);
	}
	else
	{
		struct dense_tensor tc;
		copy_dense_tensor(t, &tc);
		conjugate_dense_tensor(&tc);

		struct dense_tensor t2;
		if (!transpose) {
			dense_tensor_dot(&tc, TENSOR_AXIS_RANGE_LEADING, t, TENSOR_AXIS_RANGE_LEADING, 1, &t2);
		}
		else {
			dense_tensor_dot(t, TENSOR_AXIS_RANGE_TRAILING, &tc, TENSOR_AXIS_RANGE_TRAILING, 1, &t2);
		}

		is_isometry = dense_tensor_is_identity(&t2, tol);

		delete_dense_tensor(&t2);
		delete_dense_tensor(&tc);
	}

	return is_isometry;
}
