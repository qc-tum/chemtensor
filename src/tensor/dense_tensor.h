/// \file dense_tensor.h
/// \brief Dense tensor structure, using row-major storage convention.

#pragma once

#include <stdbool.h>
#include <assert.h>
#include "numeric.h"
#include "util.h"
#include "rng.h"


//________________________________________________________________________________________________________________________
///
/// \brief General dense tensor structure.
///
struct dense_tensor
{
	void* data;               //!< data entries, array of dimension dim[0] x ... x dim[ndim-1]
	long* dim;                //!< dimensions (width, height, ...)
	enum numeric_type dtype;  //!< numeric data type
	int ndim;                 //!< number of dimensions (degree)
};


//________________________________________________________________________________________________________________________
//

// allocation and construction

void allocate_dense_tensor(const enum numeric_type dtype, const int ndim, const long* restrict dim, struct dense_tensor* restrict t);

void delete_dense_tensor(struct dense_tensor* t);

void copy_dense_tensor(const struct dense_tensor* restrict src, struct dense_tensor* restrict dst);

void move_dense_tensor_data(struct dense_tensor* restrict src, struct dense_tensor* restrict dst);


//________________________________________________________________________________________________________________________
///
/// \brief Calculate the number of elements of a dense tensor.
///
static inline long dense_tensor_num_elements(const struct dense_tensor* t)
{
	long nelem = integer_product(t->dim, t->ndim);
	assert(nelem > 0);

	return nelem;
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert tensor index to data offset.
///
static inline long tensor_index_to_offset(const int ndim, const long* restrict dim, const long* restrict index)
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
/// \brief Convert data offset to tensor index.
///
static inline void offset_to_tensor_index(const int ndim, const long* restrict dim, const long offset, long* restrict index)
{
	long n = offset;
	for (int i = ndim - 1; i >= 0; i--)
	{
		index[i] = n % dim[i];
		n /= dim[i];
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the lexicographically next tensor index.
///
static inline void next_tensor_index(const int ndim, const long* restrict dim, long* restrict index)
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
//

// trace and norm

void dense_tensor_trace(const struct dense_tensor* t, void* ret);

void dense_tensor_cyclic_partial_trace(const struct dense_tensor* t, const int ndim_trace, struct dense_tensor* r);

void dense_tensor_cyclic_partial_trace_update(const struct dense_tensor* t, const int ndim_trace, struct dense_tensor* r);

double dense_tensor_norm2(const struct dense_tensor* t);


//________________________________________________________________________________________________________________________
//

// in-place manipulation

void scale_dense_tensor(const void* alpha, struct dense_tensor* t);

void rscale_dense_tensor(const void* alpha, struct dense_tensor* t);

void reshape_dense_tensor(const int ndim, const long* dim, struct dense_tensor* t);

void conjugate_dense_tensor(struct dense_tensor* t);

void dense_tensor_flatten_axes(struct dense_tensor* t, const int i_ax);

void dense_tensor_set_identity(struct dense_tensor* t);

void dense_tensor_fill_random_normal(const void* alpha, const void* shift, struct rng_state* rng_state, struct dense_tensor* t);


//________________________________________________________________________________________________________________________
//

// transposition

void transpose_dense_tensor(const int* restrict perm, const struct dense_tensor* restrict t, struct dense_tensor* restrict r);

void conjugate_transpose_dense_tensor(const int* restrict perm, const struct dense_tensor* restrict t, struct dense_tensor* restrict r);


//________________________________________________________________________________________________________________________
//

// slicing

void dense_tensor_slice(const struct dense_tensor* restrict t, const int i_ax, const long* ind, const long nind, struct dense_tensor* restrict r);

void dense_tensor_slice_fill(const struct dense_tensor* restrict t, const int i_ax, const long* ind, const long nind, struct dense_tensor* restrict r);


//________________________________________________________________________________________________________________________
//

// padding

void dense_tensor_pad_zeros(const struct dense_tensor* restrict t, const long* restrict pad_before, const long* restrict pad_after, struct dense_tensor* restrict r);


//________________________________________________________________________________________________________________________
///
/// \brief Tensor axis alignment, used to specify multiplication.
///
enum tensor_axis_range
{
	TENSOR_AXIS_RANGE_LEADING  = 0,  //!< leading tensor axes
	TENSOR_AXIS_RANGE_TRAILING = 1,  //!< trailing tensor axes
	TENSOR_AXIS_RANGE_NUM      = 2,  //!< number of tensor axis alignment modes
};


//________________________________________________________________________________________________________________________
//

// binary operations

void dense_tensor_scalar_multiply_add(const void* alpha, const struct dense_tensor* restrict s, struct dense_tensor* restrict t);

void dense_tensor_multiply_pointwise(const struct dense_tensor* restrict s, const struct dense_tensor* restrict t, const enum tensor_axis_range axrange, struct dense_tensor* restrict r);

void dense_tensor_multiply_pointwise_fill(const struct dense_tensor* restrict s, const struct dense_tensor* restrict t, const enum tensor_axis_range axrange, struct dense_tensor* restrict r);

void dense_tensor_multiply_axis(const struct dense_tensor* restrict s, const int i_ax, const struct dense_tensor* restrict t, const enum tensor_axis_range axrange_t, struct dense_tensor* restrict r);

void dense_tensor_multiply_axis_update(const void* alpha, const struct dense_tensor* restrict s, const int i_ax, const struct dense_tensor* restrict t, const enum tensor_axis_range axrange_t, const void* beta, struct dense_tensor* restrict r);

void dense_tensor_dot(const struct dense_tensor* restrict s, const enum tensor_axis_range axrange_s, const struct dense_tensor* restrict t, const enum tensor_axis_range axrange_t, const int ndim_mult, struct dense_tensor* restrict r);

void dense_tensor_dot_update(const void* alpha, const struct dense_tensor* restrict s, const enum tensor_axis_range axrange_s, const struct dense_tensor* restrict t, const enum tensor_axis_range axrange_t, const int ndim_mult, const void* beta, struct dense_tensor* restrict r);

void dense_tensor_kronecker_product(const struct dense_tensor* restrict s, const struct dense_tensor* restrict t, struct dense_tensor* restrict r);

void dense_tensor_concatenate(const struct dense_tensor* restrict tlist, const int num_tensors, const int i_ax, struct dense_tensor* restrict r);

void dense_tensor_concatenate_fill(const struct dense_tensor* restrict tlist, const int num_tensors, const int i_ax, struct dense_tensor* restrict r);

void dense_tensor_block_diag(const struct dense_tensor* restrict tlist, const int num_tensors, const int* i_ax, const int ndim_block, struct dense_tensor* restrict r);

void dense_tensor_block_diag_fill(const struct dense_tensor* restrict tlist, const int num_tensors, const int* i_ax, const int ndim_block, struct dense_tensor* restrict r);


//________________________________________________________________________________________________________________________
//

// QR and RQ decomposition

int dense_tensor_qr(const struct dense_tensor* restrict a, struct dense_tensor* restrict q, struct dense_tensor* restrict r);

int dense_tensor_qr_fill(const struct dense_tensor* restrict a, struct dense_tensor* restrict q, struct dense_tensor* restrict r);


int dense_tensor_rq(const struct dense_tensor* restrict a, struct dense_tensor* restrict r, struct dense_tensor* restrict q);

int dense_tensor_rq_fill(const struct dense_tensor* restrict a, struct dense_tensor* restrict r, struct dense_tensor* restrict q);


//________________________________________________________________________________________________________________________
//

// spectral decomposition

int dense_tensor_eigh(const struct dense_tensor* restrict a, struct dense_tensor* restrict u, struct dense_tensor* restrict lambda);

int dense_tensor_eigh_fill(const struct dense_tensor* restrict a, struct dense_tensor* restrict u, struct dense_tensor* restrict lambda);


//________________________________________________________________________________________________________________________
//

// singular value decomposition

int dense_tensor_svd(const struct dense_tensor* restrict a, struct dense_tensor* restrict u, struct dense_tensor* restrict s, struct dense_tensor* restrict vh);

int dense_tensor_svd_fill(const struct dense_tensor* restrict a, struct dense_tensor* restrict u, struct dense_tensor* restrict s, struct dense_tensor* restrict vh);


//________________________________________________________________________________________________________________________
//

// extract a sub-block

void dense_tensor_block(const struct dense_tensor* restrict t, const long* restrict sdim, const long* restrict* idx, struct dense_tensor* restrict r);


//________________________________________________________________________________________________________________________
//

// comparison

bool dense_tensor_allclose(const struct dense_tensor* restrict s, const struct dense_tensor* restrict t, const double tol);

bool dense_tensor_is_zero(const struct dense_tensor* t, const double tol);

bool dense_tensor_is_identity(const struct dense_tensor* t, const double tol);

bool dense_tensor_is_self_adjoint(const struct dense_tensor* t, const double tol);

bool dense_tensor_is_isometry(const struct dense_tensor* t, const double tol, const bool transpose);
