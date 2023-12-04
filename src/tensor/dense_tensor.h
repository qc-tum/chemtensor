/// \file dense_tensor.h
/// \brief Dense tensor structure, using row-major storage convention.

#pragma once

#include <assert.h>
#include "config.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief General dense tensor structure.
///
struct dense_tensor
{
	numeric* data;  //!< data entries, array of size dim[0] x ... x dim[ndim-1]
	long* dim;      //!< dimensions (width, height, ...)
	int ndim;       //!< number of dimensions (degree)
};


//________________________________________________________________________________________________________________________
//

// allocation and construction

void allocate_dense_tensor(const int ndim, const long* restrict dim, struct dense_tensor* restrict t);

void delete_dense_tensor(struct dense_tensor* restrict t);

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
//

// trace

numeric dense_tensor_trace(const struct dense_tensor* t);


//________________________________________________________________________________________________________________________
//

// in-place manipulation

void scale_dense_tensor(const double alpha, struct dense_tensor* t);

void reshape_dense_tensor(const int ndim, const long* restrict dim, struct dense_tensor* restrict t);

void conjugate_dense_tensor(struct dense_tensor* t);

void dense_tensor_set_identity(struct dense_tensor* t);


//________________________________________________________________________________________________________________________
//

// transposition

void transpose_dense_tensor(const int* restrict perm, const struct dense_tensor* restrict t, struct dense_tensor* restrict r);

void conjugate_transpose_dense_tensor(const int* restrict perm, const struct dense_tensor* restrict t, struct dense_tensor* restrict r);


//________________________________________________________________________________________________________________________
//

// binary operations

void dense_tensor_scalar_multiply_add(const numeric alpha, const struct dense_tensor* restrict s, struct dense_tensor* restrict t);

void dense_tensor_dot(const struct dense_tensor* restrict s, const struct dense_tensor* restrict t, const int ndim_mult, struct dense_tensor* restrict r);

void dense_tensor_kronecker_product(const struct dense_tensor* restrict s, const struct dense_tensor* restrict t, struct dense_tensor* restrict r);


//________________________________________________________________________________________________________________________
//

// extract a sub-block

void dense_tensor_block(const struct dense_tensor* restrict t, const long* restrict sdim, const long* restrict* idx, struct dense_tensor* restrict r);
