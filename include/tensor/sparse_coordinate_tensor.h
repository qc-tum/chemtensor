/// \file sparse_coordinate_tensor.h
/// \brief Sparse coordinate tensor structure.

#pragma once

#include <stdbool.h>
#include "dense_tensor.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Sparse coordinate tensor structure.
///
struct sparse_coordinate_tensor
{
	void* values;             //!< numerical values of non-zero data entries, array of length 'nnz'
	ct_long* coords;          //!< linearized sorted coordinates (indices) of non-zero entries, array of length 'nnz'
	ct_long* dim;             //!< logical dimensions
	ct_long nnz;              //!< number of non-zero entries
	enum numeric_type dtype;  //!< numeric data type
	int ndim;                 //!< number of dimensions (degree)
};


//________________________________________________________________________________________________________________________
//

// allocation and construction

void allocate_sparse_coordinate_tensor(const enum numeric_type dtype, const int ndim, const ct_long nnz, const ct_long* dim, struct sparse_coordinate_tensor* t);

void delete_sparse_coordinate_tensor(struct sparse_coordinate_tensor* t);


//________________________________________________________________________________________________________________________
//

// internal consistency checking

bool sparse_coordinate_tensor_is_consistent(const struct sparse_coordinate_tensor* t);


//________________________________________________________________________________________________________________________
//

// transposition

void transpose_sparse_coordinate_tensor(const int* perm, const struct sparse_coordinate_tensor* restrict t, struct sparse_coordinate_tensor* restrict r);


//________________________________________________________________________________________________________________________
//

// binary operations

void sparse_coordinate_tensor_dot(const struct sparse_coordinate_tensor* restrict s, const struct sparse_coordinate_tensor* restrict t, const int ndim_mult, struct sparse_coordinate_tensor* restrict r);


//________________________________________________________________________________________________________________________
//

// conversion between dense and sparse coordinate tensors

void sparse_coordinate_to_dense_tensor(const struct sparse_coordinate_tensor* s, struct dense_tensor* t);

void dense_to_sparse_coordinate_tensor(const struct dense_tensor* t, struct sparse_coordinate_tensor* s);
