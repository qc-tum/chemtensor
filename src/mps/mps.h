/// \file mps.h
/// \brief Matrix product state (MPS) data structure.

#pragma once

#include "block_sparse_tensor.h"


//________________________________________________________________________________________________________________________
///
/// \brief Matrix product state (MPS) data structure.
///
struct mps
{
	struct block_sparse_tensor* a;  //!< tensors associated with sites, with dimensions D_i x d x D_{i+1}; array of length 'nsites'
	long d;                         //!< local physical dimension of each site
	qnumber* qsite;                 //!< physical quantum numbers at each site
	int nsites;                     //!< number of sites
};


//________________________________________________________________________________________________________________________
//

// allocation and construction

void allocate_mps(const enum numeric_type dtype, const int nsites, const long d, const qnumber* qsite, const long* dim_bonds, const qnumber** qbonds, struct mps* mps);

void delete_mps(struct mps* mps);

bool mps_is_consistent(const struct mps* mps);


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

// conversion to a statevector (intended for testing)

void mps_merge_tensor_pair(const struct block_sparse_tensor* restrict a0, const struct block_sparse_tensor* restrict a1, struct block_sparse_tensor* restrict a);

void mps_to_statevector(const struct mps* mps, struct block_sparse_tensor* a);
