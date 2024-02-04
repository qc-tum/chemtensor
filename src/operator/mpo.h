/// \file mpo.h
/// \brief Matrix product operator (MPO) data structure and functions.

#pragma once

#include "block_sparse_tensor.h"
#include "mpo_graph.h"


//________________________________________________________________________________________________________________________
///
/// \brief Matrix product operator (MPO) data structure.
///
struct mpo
{
	struct block_sparse_tensor* a;  //!< tensors associated with sites, with dimensions D_i x d x d x D_{i+1}; array of length 'nsites'
	long d;                         //!< local physical dimension of each site
	qnumber* qsite;                 //!< physical quantum numbers at each site
	int nsites;                     //!< number of sites
};


//________________________________________________________________________________________________________________________
//

// allocation and construction

void mpo_from_graph(const enum numeric_type dtype, const long d, const qnumber* qsite, const struct mpo_graph* graph, const struct dense_tensor* opmap, struct mpo* mpo);

void delete_mpo(struct mpo* mpo);

bool mpo_is_consistent(const struct mpo* mpo);


//________________________________________________________________________________________________________________________
///
/// \brief Dimension of i-th virtual bond of a matrix product operator, starting with the leftmost (dummy) bond.
///
static inline long mpo_bond_dim(const struct mpo* mpo, const int i)
{
	return (i < mpo->nsites ? mpo->a[i].dim_logical[0] : mpo->a[mpo->nsites - 1].dim_logical[3]);
}


//________________________________________________________________________________________________________________________
//

// conversion to a matrix (intended for testing)

void mpo_merge_tensor_pair(const struct block_sparse_tensor* restrict a0, const struct block_sparse_tensor* restrict a1, struct block_sparse_tensor* restrict a);

void mpo_to_matrix(const struct mpo* mpo, struct block_sparse_tensor* mat);
