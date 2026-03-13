/// \file su2_bond_ops.h
/// \brief Auxiliary data structures and functions concerning virtual bonds for SU(2) symmetric tensors.

#pragma once

#include <stdbool.h>
#include "truncation.h"
#include "su2_tensor.h"


//________________________________________________________________________________________________________________________
///
/// \brief Singular value distribution mode for SU(2) tensors.
///
enum su2_singular_value_distr
{
	SU2_SVD_DISTR_LEFT  = 0,
	SU2_SVD_DISTR_RIGHT = 1,
};


int split_su2_matrix_svd(const struct su2_tensor* a,
	const double tol, const bool relative_thresh, const ct_long max_vdim,
	const enum su2_singular_value_distr svd_distr, const bool copy_tree_left,
	struct su2_tensor* a0, struct su2_tensor* a1, struct trunc_info* info);
