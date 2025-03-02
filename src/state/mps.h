/// \file mps.h
/// \brief Matrix product state (MPS) data structure.

#pragma once

#include "block_sparse_tensor.h"
#include "bond_ops.h"


//________________________________________________________________________________________________________________________
///
/// \brief Matrix product state (MPS) data structure.
///
struct mps
{
	struct block_sparse_tensor* a;  //!< tensors associated with sites, with dimensions \f$D_i \times d \times D_{i+1}\f$; array of length 'nsites'
	qnumber* qsite;                 //!< physical quantum numbers at each site
	long d;                         //!< local physical dimension of each site
	int nsites;                     //!< number of sites
};


//________________________________________________________________________________________________________________________
//

// allocation and construction

void allocate_empty_mps(const int nsites, const long d, const qnumber* qsite, struct mps* mps);

void allocate_mps(const enum numeric_type dtype, const int nsites, const long d, const qnumber* qsite, const long* dim_bonds, const qnumber** qbonds, struct mps* mps);

void delete_mps(struct mps* mps);

void copy_mps(const struct mps* restrict src, struct mps* restrict dst);

void move_mps_data(struct mps* restrict src, struct mps* restrict dst);

void construct_random_mps(const enum numeric_type dtype, const int nsites, const long d, const qnumber* qsite, const qnumber qnum_sector, const long max_vdim, struct rng_state* rng_state, struct mps* mps);

bool mps_is_consistent(const struct mps* mps);

bool mps_equals(const struct mps* a, const struct mps* b);


//________________________________________________________________________________________________________________________
///
/// \brief Dimension of i-th virtual bond of a matrix product state, starting with the leftmost (dummy) bond.
///
static inline long mps_bond_dim(const struct mps* mps, const int i)
{
	return (i < mps->nsites ? mps->a[i].dim_logical[0] : mps->a[mps->nsites - 1].dim_logical[2]);
}


//________________________________________________________________________________________________________________________
//

// inner product and norm

void mps_vdot(const struct mps* chi, const struct mps* psi, void* ret);

double mps_norm(const struct mps* psi);


//________________________________________________________________________________________________________________________
//

// logical addition

void mps_add(const struct mps* chi, const struct mps* psi, struct mps* ret);


//________________________________________________________________________________________________________________________
//

// orthonormalization and canonical forms

/// \brief MPS orthonormalization mode.
enum mps_orthonormalization_mode
{
	MPS_ORTHONORMAL_LEFT  = 0,  //!< left-orthonormal
	MPS_ORTHONORMAL_RIGHT = 1,  //!< right-orthonormal
};

void mps_local_orthonormalize_qr(struct block_sparse_tensor* restrict a, struct block_sparse_tensor* restrict a_next);

void mps_local_orthonormalize_rq(struct block_sparse_tensor* restrict a, struct block_sparse_tensor* restrict a_prev);

double mps_orthonormalize_qr(struct mps* mps, const enum mps_orthonormalization_mode mode);


int mps_local_orthonormalize_left_svd(const double tol, const long max_vdim, const bool renormalize,
	struct block_sparse_tensor* restrict a, struct block_sparse_tensor* restrict a_next, struct trunc_info* info);

int mps_local_orthonormalize_right_svd(const double tol, const long max_vdim, const bool renormalize,
	struct block_sparse_tensor* restrict a, struct block_sparse_tensor* restrict a_prev, struct trunc_info* info);

int mps_compress(const double tol, const long max_vdim, const enum mps_orthonormalization_mode mode,
	struct mps* mps, double* restrict norm, double* restrict trunc_scale, struct trunc_info* info);

int mps_compress_rescale(const double tol, const long max_vdim, const enum mps_orthonormalization_mode mode,
	struct mps* mps, double* trunc_scale, struct trunc_info* info);


//________________________________________________________________________________________________________________________
//

// splitting and merging

int mps_split_tensor_svd(const struct block_sparse_tensor* restrict a, const long d[2], const qnumber* new_qsite[2],
	const double tol, const long max_vdim, const bool renormalize, const enum singular_value_distr svd_distr,
	struct block_sparse_tensor* restrict a0, struct block_sparse_tensor* restrict a1, struct trunc_info* info);


void mps_merge_tensor_pair(const struct block_sparse_tensor* restrict a0, const struct block_sparse_tensor* restrict a1, struct block_sparse_tensor* restrict a);


//________________________________________________________________________________________________________________________
//

// conversion to a statevector (intended for testing)

void mps_to_statevector(const struct mps* mps, struct block_sparse_tensor* vec);
