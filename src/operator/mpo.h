/// \file mpo.h
/// \brief Matrix product operator (MPO) data structure and functions.

#pragma once

#include "block_sparse_tensor.h"
#include "mpo_graph.h"


//________________________________________________________________________________________________________________________
///
/// \brief Matrix product operator (MPO) assembly: structural description which can be used to construct an MPO.
///
struct mpo_assembly
{
	struct mpo_graph graph;         //!< MPO graph data structure
	struct dense_tensor* opmap;     //!< local operator map (look-up table)
	void* coeffmap;                 //!< coefficient map (look-up table)
	qnumber* qsite;                 //!< physical quantum numbers at each site
	long d;                         //!< local physical dimension of each site
	enum numeric_type dtype;        //!< data type of local operators and coefficients
	int num_local_ops;              //!< number of local operators (length of 'opmap' array)
	int num_coeffs;                 //!< number of coefficients (length of 'coeffmap' array)
};


void delete_mpo_assembly(struct mpo_assembly* assembly);


//________________________________________________________________________________________________________________________
///
/// \brief Matrix product operator (MPO) data structure.
///
struct mpo
{
	struct block_sparse_tensor* a;  //!< tensors associated with sites, with dimensions \f$D_i \times d \times d \times D_{i+1}\f$; array of length 'nsites'
	qnumber* qsite;                 //!< physical quantum numbers at each site
	long d;                         //!< local physical dimension of each site
	int nsites;                     //!< number of sites
};


//________________________________________________________________________________________________________________________
//

// allocation and construction

void allocate_mpo(const enum numeric_type dtype, const int nsites, const long d, const qnumber* qsite, const long* dim_bonds, const qnumber** qbonds, struct mpo* mpo);

void mpo_from_assembly(const struct mpo_assembly* assembly, struct mpo* mpo);

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
