/// \file bond_ops.h
/// \brief Auxiliary data structures and functions concerning virtual bonds.

#pragma once

#include <stdbool.h>
#include "truncation.h"
#include "block_sparse_tensor.h"


//________________________________________________________________________________________________________________________
///
/// \brief Singular value distribution mode.
///
enum singular_value_distr
{
	SVD_DISTR_LEFT  = 0,
	SVD_DISTR_RIGHT = 1,
};


int split_block_sparse_matrix_svd(const struct block_sparse_tensor* a,
	const double tol, const bool relative_thresh, const ct_long max_vdim, const bool renormalize, const enum singular_value_distr svd_distr,
	struct block_sparse_tensor* a0, struct block_sparse_tensor* a1, struct trunc_info* info);

int split_block_sparse_matrix_svd_isometry(const struct block_sparse_tensor* a, const double tol, const bool relative_thresh, const ct_long max_vdim,
	struct block_sparse_tensor* u, struct trunc_info* info);
