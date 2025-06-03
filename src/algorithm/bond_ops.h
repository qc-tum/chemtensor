/// \file bond_ops.h
/// \brief Auxiliary data structures and functions concerning virtual bonds.

#pragma once

#include <stdbool.h>
#include "block_sparse_tensor.h"


double von_neumann_entropy(const double* sigma, const long n);


//________________________________________________________________________________________________________________________
///
/// \brief Singular value truncation information relevant for two-site MPS and MPO operations.
///
struct trunc_info
{
	double norm_sigma;  //!< norm of the retained singular values
	double entropy;     //!< von Neumann entropy
	double tol_eff;     //!< tolerance (truncation weight), can be larger than input tolerance due to maximum bond dimension
};


//________________________________________________________________________________________________________________________
///
/// \brief Index list, used for selecting singular values.
///
struct index_list
{
	long* ind;  //!< indices
	long num;   //!< number of indices
};

void delete_index_list(struct index_list* list);


void retained_bond_indices(const double* sigma, const long n, const double tol, const bool relative_thresh, const long max_vdim,
	struct index_list* list, struct trunc_info* info);


//________________________________________________________________________________________________________________________
///
/// \brief Singular value distribution mode.
///
enum singular_value_distr
{
	SVD_DISTR_LEFT  = 0,
	SVD_DISTR_RIGHT = 1,
};


int split_block_sparse_matrix_svd(const struct block_sparse_tensor* restrict a,
	const double tol, const bool relative_thresh, const long max_vdim, const bool renormalize, const enum singular_value_distr svd_distr,
	struct block_sparse_tensor* restrict a0, struct block_sparse_tensor* restrict a1, struct trunc_info* info);

int split_block_sparse_matrix_svd_isometry(const struct block_sparse_tensor* restrict a, const double tol, const bool relative_thresh, const long max_vdim,
	struct block_sparse_tensor* restrict u, struct trunc_info* info);
