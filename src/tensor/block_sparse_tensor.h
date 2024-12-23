/// \file block_sparse_tensor.h
/// \brief Block-sparse tensor structure.

#pragma once

#include <assert.h>
#include "dense_tensor.h"
#include "qnumber.h"
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
	enum numeric_type dtype;                //!< numeric data type
	int ndim;                               //!< number of dimensions (degree)
};


//________________________________________________________________________________________________________________________
//

// allocation and construction

void allocate_block_sparse_tensor(const enum numeric_type dtype, const int ndim, const long* restrict dim, const enum tensor_axis_direction* axis_dir, const qnumber** restrict qnums, struct block_sparse_tensor* restrict t);

void allocate_block_sparse_tensor_like(const struct block_sparse_tensor* restrict s, struct block_sparse_tensor* restrict t);

void delete_block_sparse_tensor(struct block_sparse_tensor* t);

void copy_block_sparse_tensor(const struct block_sparse_tensor* restrict src, struct block_sparse_tensor* restrict dst);

void move_block_sparse_tensor_data(struct block_sparse_tensor* restrict src, struct block_sparse_tensor* restrict dst);


//________________________________________________________________________________________________________________________
//


struct dense_tensor* block_sparse_tensor_get_block(const struct block_sparse_tensor* t, const qnumber* qnums);


//________________________________________________________________________________________________________________________
//

// trace and norm

void block_sparse_tensor_cyclic_partial_trace(const struct block_sparse_tensor* restrict t, const int ndim_trace, struct block_sparse_tensor* restrict r);

double block_sparse_tensor_norm2(const struct block_sparse_tensor* t);


//________________________________________________________________________________________________________________________
//

// in-place manipulation

void block_sparse_tensor_reverse_axis_directions(struct block_sparse_tensor* t);

void scale_block_sparse_tensor(const void* alpha, struct block_sparse_tensor* t);

void rscale_block_sparse_tensor(const void* alpha, struct block_sparse_tensor* t);

void conjugate_block_sparse_tensor(struct block_sparse_tensor* t);

void block_sparse_tensor_fill_random_normal(const void* alpha, const void* shift, struct rng_state* rng_state, struct block_sparse_tensor* t);


//________________________________________________________________________________________________________________________
//

// conversion between dense and block-sparse tensors

void block_sparse_to_dense_tensor(const struct block_sparse_tensor* restrict s, struct dense_tensor* restrict t);

void dense_to_block_sparse_tensor(const struct dense_tensor* restrict t, const enum tensor_axis_direction* axis_dir, const qnumber** restrict qnums, struct block_sparse_tensor* restrict s);

void dense_to_block_sparse_tensor_entries(const struct dense_tensor* restrict t, struct block_sparse_tensor* restrict s);


//________________________________________________________________________________________________________________________
//

// transposition

void transpose_block_sparse_tensor(const int* restrict perm, const struct block_sparse_tensor* restrict t, struct block_sparse_tensor* restrict r);

void conjugate_transpose_block_sparse_tensor(const int* restrict perm, const struct block_sparse_tensor* restrict t, struct block_sparse_tensor* restrict r);


//________________________________________________________________________________________________________________________
//

// reshaping

void flatten_block_sparse_tensor_axes(const struct block_sparse_tensor* restrict t, const int i_ax, const enum tensor_axis_direction new_axis_dir, struct block_sparse_tensor* restrict r);

void split_block_sparse_tensor_axis(const struct block_sparse_tensor* restrict t, const int i_ax, const long new_dim_logical[2], const enum tensor_axis_direction new_axis_dir[2], const qnumber* new_qnums_logical[2], struct block_sparse_tensor* restrict r);


//________________________________________________________________________________________________________________________
//

// slicing

void block_sparse_tensor_slice(const struct block_sparse_tensor* restrict t, const int i_ax, const long* ind, const long nind, struct block_sparse_tensor* restrict r);


//________________________________________________________________________________________________________________________
//

// binary operations

void block_sparse_tensor_scalar_multiply_add(const void* alpha, const struct block_sparse_tensor* restrict s, struct block_sparse_tensor* restrict t);

void block_sparse_tensor_multiply_pointwise_vector(const struct block_sparse_tensor* restrict s, const struct dense_tensor* restrict t, const enum tensor_axis_range axrange, struct block_sparse_tensor* restrict r);

void block_sparse_tensor_multiply_axis(const struct block_sparse_tensor* restrict s, const int i_ax, const struct block_sparse_tensor* restrict t, const enum tensor_axis_range axrange_t, struct block_sparse_tensor* restrict r);

void block_sparse_tensor_dot(const struct block_sparse_tensor* restrict s, const enum tensor_axis_range axrange_s, const struct block_sparse_tensor* restrict t, const enum tensor_axis_range axrange_t, const int ndim_mult, struct block_sparse_tensor* restrict r);

void block_sparse_tensor_concatenate(const struct block_sparse_tensor* restrict tlist, const int num_tensors, const int i_ax, struct block_sparse_tensor* restrict r);

void block_sparse_tensor_block_diag(const struct block_sparse_tensor* restrict tlist, const int num_tensors, const int* i_ax, const int ndim_block, struct block_sparse_tensor* restrict r);


//________________________________________________________________________________________________________________________
//

// QR and RQ decomposition

int block_sparse_tensor_qr(const struct block_sparse_tensor* restrict a, struct block_sparse_tensor* restrict q, struct block_sparse_tensor* restrict r);

int block_sparse_tensor_rq(const struct block_sparse_tensor* restrict a, struct block_sparse_tensor* restrict r, struct block_sparse_tensor* restrict q);


//________________________________________________________________________________________________________________________
//

// singular value decomposition

int block_sparse_tensor_svd(const struct block_sparse_tensor* restrict a, struct block_sparse_tensor* restrict u, struct dense_tensor* restrict s, struct block_sparse_tensor* restrict vh);


//________________________________________________________________________________________________________________________
//

// comparison

bool block_sparse_tensor_allclose(const struct block_sparse_tensor* restrict s, const struct block_sparse_tensor* restrict t, const double tol);

bool block_sparse_tensor_is_identity(const struct block_sparse_tensor* t, const double tol);

bool block_sparse_tensor_is_isometry(const struct block_sparse_tensor* t, const double tol, const bool transpose);


//________________________________________________________________________________________________________________________
//

// (de)serialization

long block_sparse_tensor_num_elements_blocks(const struct block_sparse_tensor* t);

void block_sparse_tensor_serialize_entries(const struct block_sparse_tensor* t, void* entries);

void block_sparse_tensor_deserialize_entries(struct block_sparse_tensor* t, const void* entries);


//________________________________________________________________________________________________________________________
///
/// \brief Entry access utility data structure for a block-sparse tensor.
///
/// Entry-wise access is comparatively slow.
///
struct block_sparse_tensor_entry_accessor
{
	const struct block_sparse_tensor* tensor;  //!< reference to tensor
	long** index_map_blocks;                   //!< map from logical index to corresponding block index
	long** index_map_block_entries;            //!< map from logical index to corresponding block entry index
};


void create_block_sparse_tensor_entry_accessor(const struct block_sparse_tensor* t, struct block_sparse_tensor_entry_accessor* acc);

void delete_block_sparse_tensor_entry_accessor(struct block_sparse_tensor_entry_accessor* acc);

void* block_sparse_tensor_get_entry(const struct block_sparse_tensor_entry_accessor* acc, const long* index);
