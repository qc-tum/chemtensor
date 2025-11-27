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
	ct_long* dim_blocks;                    //!< block dimensions
	ct_long* dim_logical;                   //!< logical dimensions of the overall tensor
	enum tensor_axis_direction* axis_dir;   //!< tensor axis directions
	qnumber** qnums_blocks;                 //!< block quantum numbers along each dimension (must be sorted and pairwise different)
	qnumber** qnums_logical;                //!< logical quantum numbers along each dimension (not necessarily sorted, can be duplicate)
	enum numeric_type dtype;                //!< numeric data type
	int ndim;                               //!< number of dimensions (degree)
};


//________________________________________________________________________________________________________________________
//

// allocation and construction

void allocate_block_sparse_tensor(const enum numeric_type dtype, const int ndim, const ct_long* dim, const enum tensor_axis_direction* axis_dir, const qnumber** qnums, struct block_sparse_tensor* t);

void allocate_block_sparse_tensor_like(const struct block_sparse_tensor* s, struct block_sparse_tensor* t);

void delete_block_sparse_tensor(struct block_sparse_tensor* t);

void copy_block_sparse_tensor(const struct block_sparse_tensor* src, struct block_sparse_tensor* dst);


//________________________________________________________________________________________________________________________
//


struct dense_tensor* block_sparse_tensor_get_block(const struct block_sparse_tensor* t, const qnumber* qnums);


//________________________________________________________________________________________________________________________
//

// trace and norm

void block_sparse_tensor_cyclic_partial_trace(const struct block_sparse_tensor* t, const int ndim_trace, struct block_sparse_tensor* r);

double block_sparse_tensor_norm2(const struct block_sparse_tensor* t);


//________________________________________________________________________________________________________________________
//

// in-place manipulation

void block_sparse_tensor_reverse_axis_directions(struct block_sparse_tensor* t);

void block_sparse_tensor_invert_axis_quantum_numbers(struct block_sparse_tensor* t, const int i_ax);

void scale_block_sparse_tensor(const void* alpha, struct block_sparse_tensor* t);

void rscale_block_sparse_tensor(const void* alpha, struct block_sparse_tensor* t);

void conjugate_block_sparse_tensor(struct block_sparse_tensor* t);

void block_sparse_tensor_set_identity_blocks(struct block_sparse_tensor* t);

void block_sparse_tensor_fill_random_normal(const void* alpha, const void* shift, struct rng_state* rng_state, struct block_sparse_tensor* t);


//________________________________________________________________________________________________________________________
//

// conversion between dense and block-sparse tensors

void block_sparse_to_dense_tensor(const struct block_sparse_tensor* s, struct dense_tensor* t);

void dense_to_block_sparse_tensor(const struct dense_tensor* t, const enum tensor_axis_direction* axis_dir, const qnumber** qnums, struct block_sparse_tensor* s);

void dense_to_block_sparse_tensor_entries(const struct dense_tensor* t, struct block_sparse_tensor* s);


//________________________________________________________________________________________________________________________
//

// transposition

void transpose_block_sparse_tensor(const int* perm, const struct block_sparse_tensor* t, struct block_sparse_tensor* r);

void conjugate_transpose_block_sparse_tensor(const int* perm, const struct block_sparse_tensor* t, struct block_sparse_tensor* r);


//________________________________________________________________________________________________________________________
//

// reshaping

void block_sparse_tensor_flatten_axes(const struct block_sparse_tensor* t, const int i_ax, const enum tensor_axis_direction new_axis_dir, struct block_sparse_tensor* r);

void block_sparse_tensor_split_axis(const struct block_sparse_tensor* t, const int i_ax, const ct_long new_dim_logical[2], const enum tensor_axis_direction new_axis_dir[2], const qnumber* new_qnums_logical[2], struct block_sparse_tensor* r);


//________________________________________________________________________________________________________________________
///
/// \brief Temporary data structure for recording the dimensions and quantum numbers before flattening two axes.
///
struct block_sparse_tensor_flatten_axes_record
{
	ct_long dim_logical[2];                  //!< logical dimensions
	enum tensor_axis_direction axis_dir[2];  //!< tensor axis directions
	qnumber* qnums_logical[2];               //!< logical quantum numbers
};

//________________________________________________________________________________________________________________________
///
/// \brief Dimension, quantum number and axis ordering information describing the axis matricization of a tensor.
/// This information is required to undo the matricization.
///
struct block_sparse_tensor_axis_matricization_info
{
	struct block_sparse_tensor_flatten_axes_record* records;  //!< axes flattening records; array of length 'ndim - 2'
	int i_ax_tns;                                             //!< isolated axis index of the original tensor
	int i_ax_mat;                                             //!< corresponding axis index of the matrix (whether 'i_ax' becomes the leading or trailing dimension)
	int ndim;                                                 //!< number of dimensions (degree) of the original tensor
};

void delete_block_sparse_tensor_axis_matricization_info(struct block_sparse_tensor_axis_matricization_info* info);


void block_sparse_tensor_matricize_axis(const struct block_sparse_tensor* t, const int i_ax_tns,
	const int i_ax_mat, const enum tensor_axis_direction flattened_axes_dir,
	struct block_sparse_tensor* mat, struct block_sparse_tensor_axis_matricization_info* info);

void block_sparse_tensor_dematricize_axis(const struct block_sparse_tensor* mat, const struct block_sparse_tensor_axis_matricization_info* info, struct block_sparse_tensor* t);


//________________________________________________________________________________________________________________________
//

// slicing

void block_sparse_tensor_slice(const struct block_sparse_tensor* t, const int i_ax, const ct_long* ind, const ct_long nind, struct block_sparse_tensor* r);


//________________________________________________________________________________________________________________________
//

// binary operations

void block_sparse_tensor_scalar_multiply_add(const void* alpha, const struct block_sparse_tensor* s, struct block_sparse_tensor* t);

void block_sparse_tensor_multiply_pointwise_vector(const struct block_sparse_tensor* s, const struct dense_tensor* t, const enum tensor_axis_range axrange, struct block_sparse_tensor* r);

void block_sparse_tensor_multiply_axis(const struct block_sparse_tensor* s, const int i_ax, const struct block_sparse_tensor* t, const enum tensor_axis_range axrange_t, struct block_sparse_tensor* r);

void block_sparse_tensor_dot(const struct block_sparse_tensor* s, const enum tensor_axis_range axrange_s, const struct block_sparse_tensor* t, const enum tensor_axis_range axrange_t, const int ndim_mult, struct block_sparse_tensor* r);

void block_sparse_tensor_concatenate(const struct block_sparse_tensor* tlist, const int num_tensors, const int i_ax, struct block_sparse_tensor* r);

void block_sparse_tensor_block_diag(const struct block_sparse_tensor* tlist, const int num_tensors, const int* i_ax, const int ndim_block, struct block_sparse_tensor* r);


//________________________________________________________________________________________________________________________
//

// QR and RQ decomposition

int block_sparse_tensor_qr(const struct block_sparse_tensor* a, const enum qr_mode mode, struct block_sparse_tensor* q, struct block_sparse_tensor* r);

int block_sparse_tensor_rq(const struct block_sparse_tensor* a, const enum qr_mode mode, struct block_sparse_tensor* r, struct block_sparse_tensor* q);


//________________________________________________________________________________________________________________________
//

// singular value decomposition

int block_sparse_tensor_svd(const struct block_sparse_tensor* a, struct block_sparse_tensor* u, struct dense_tensor* s, struct block_sparse_tensor* vh);


//________________________________________________________________________________________________________________________
//

// comparison

bool block_sparse_tensor_allclose(const struct block_sparse_tensor* s, const struct block_sparse_tensor* t, const double tol);
bool dense_block_sparse_tensor_allclose(const struct dense_tensor* s, const struct block_sparse_tensor* t, const double tol);

bool block_sparse_tensor_is_identity(const struct block_sparse_tensor* t, const double tol);

bool block_sparse_tensor_is_isometry(const struct block_sparse_tensor* t, const double tol, const bool transpose);


//________________________________________________________________________________________________________________________
//

// augmentation

void block_sparse_tensor_augment_identity_blocks(const struct block_sparse_tensor* t, const bool transpose, struct block_sparse_tensor* ret);


//________________________________________________________________________________________________________________________
//

// (de)serialization

ct_long block_sparse_tensor_num_elements_blocks(const struct block_sparse_tensor* t);

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
	ct_long** index_map_blocks;                //!< map from logical index to corresponding block index
	ct_long** index_map_block_entries;         //!< map from logical index to corresponding block entry index
};


void create_block_sparse_tensor_entry_accessor(const struct block_sparse_tensor* t, struct block_sparse_tensor_entry_accessor* acc);

void delete_block_sparse_tensor_entry_accessor(struct block_sparse_tensor_entry_accessor* acc);

void* block_sparse_tensor_get_entry(const struct block_sparse_tensor_entry_accessor* acc, const ct_long* index);
