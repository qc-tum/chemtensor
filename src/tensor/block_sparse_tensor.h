/// \file block_sparse_tensor.h
/// \brief Block-sparse tensor structure.

#pragma once

#include <assert.h>
#include "dense_tensor.h"
#include "qnumber.h"
#include "config.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Block-sparse tensor structure.
///
/// Sparsity structure is imposed by additive quantum numbers.
///
struct block_sparse_tensor
{
	struct dense_tensor** blocks;           //!< dense blocks, array of size dim_blocks[0] x ... x dim_blocks[ndim-1]; contains NULL pointers for non-conserved quantum numbers
	long* dim_blocks;                       //!< block dimensions
	long* dim_logical;                      //!< logical dimensions of the overall tensor
	enum tensor_axis_direction* axis_dir;   //!< tensor axis directions
	qnumber** qnums_blocks;                 //!< block quantum numbers along each dimension (must be sorted and pairwise different)
	qnumber** qnums_logical;                //!< logical quantum numbers along each dimension (not necessarily sorted, can be duplicate)
	int ndim;                               //!< number of dimensions (degree)
};


//________________________________________________________________________________________________________________________
//

// allocation and construction

void allocate_block_sparse_tensor(const int ndim, const long* restrict dim, const enum tensor_axis_direction* axis_dir, const qnumber** restrict qnums, struct block_sparse_tensor* restrict t);

void delete_block_sparse_tensor(struct block_sparse_tensor* t);


//________________________________________________________________________________________________________________________
//


struct dense_tensor* block_sparse_tensor_get_block(const struct block_sparse_tensor* t, const qnumber* qnums);


//________________________________________________________________________________________________________________________
//

// in-place manipulation

void scale_block_sparse_tensor(const double alpha, struct block_sparse_tensor* t);

void conjugate_block_sparse_tensor(struct block_sparse_tensor* t);


//________________________________________________________________________________________________________________________
//

// conversion between dense and block-sparse tensors

void block_sparse_to_dense_tensor(const struct block_sparse_tensor* restrict s, struct dense_tensor* restrict t);

void dense_to_block_sparse_tensor(const struct dense_tensor* restrict t, const enum tensor_axis_direction* axis_dir, const qnumber** restrict qnums, struct block_sparse_tensor* restrict s);


//________________________________________________________________________________________________________________________
//

// transposition

void transpose_block_sparse_tensor(const int* restrict perm, const struct block_sparse_tensor* restrict t, struct block_sparse_tensor* restrict r);

void conjugate_transpose_block_sparse_tensor(const int* restrict perm, const struct block_sparse_tensor* restrict t, struct block_sparse_tensor* restrict r);


//________________________________________________________________________________________________________________________
//


void flatten_block_sparse_tensor_axes(const struct block_sparse_tensor* restrict t, const long i_ax, const enum tensor_axis_direction new_axis_dir, struct block_sparse_tensor* restrict r);


//________________________________________________________________________________________________________________________
//

// binary operations

void block_sparse_tensor_dot(const struct block_sparse_tensor* restrict s, const struct block_sparse_tensor* restrict t, const int ndim_mult, struct block_sparse_tensor* restrict r);
