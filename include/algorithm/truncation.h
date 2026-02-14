/// \file truncation.h
/// \brief Utility functions for singular value truncation.

#pragma once

#include <stdbool.h>
#include "util.h"


double von_neumann_entropy(const double* sigma, const ct_long n);

double von_neumann_entropy_multiplicities(const double* sigma, const int* multiplicities, const ct_long n);


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
	ct_long* ind;  //!< indices
	ct_long num;   //!< number of indices
};

void delete_index_list(struct index_list* list);


void retained_bond_indices(const double* sigma, const ct_long n, const double tol, const bool relative_thresh, const ct_long max_vdim,
	struct index_list* list, struct trunc_info* info);

void retained_bond_indices_multiplicities(const double* sigma, const int* multiplicities, const ct_long n, const double tol, const bool relative_thresh, const ct_long max_vdim,
	struct index_list* list, struct trunc_info* info);
