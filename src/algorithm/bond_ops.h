/// \file bond_ops.h
/// \brief Auxiliary data structures and functions concerning virtual bonds.

#pragma once


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


void retained_bond_indices(const double* sigma, const long n, const double tol, const long max_vdim,
	struct index_list* list, struct trunc_info* info);
