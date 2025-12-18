/// \file su2_mpo.h
/// \brief SU(2) symmetric matrix product operator (MPO) data structure and functions.

#pragma once

#include "su2_tensor.h"


//________________________________________________________________________________________________________________________
///
/// \brief SU(2) symmetric matrix product operator (MPO) data structure.
///
struct su2_mpo
{
	struct su2_tensor* a;  //!< tensors associated with sites, with logical axis ordering "left virtual", "physical output", "physical input", "right virtual"; array of length 'nsites'
	int nsites;            //!< number of sites
};


//________________________________________________________________________________________________________________________
//

// allocation and construction

void allocate_su2_mpo(
	const enum numeric_type dtype, const int nsites,
	const struct su2_irreducible_list* site_irreps, const ct_long* site_dim_degen,
	const struct su2_irreducible_list* bond_irreps, const ct_long** bond_dim_degen,
	struct su2_mpo* mpo);

void delete_su2_mpo(struct su2_mpo* mpo);

bool su2_mpo_is_consistent(const struct su2_mpo* mpo);


//________________________________________________________________________________________________________________________
//

// conversion to a single tensor (intended for testing)

void su2_mpo_to_tensor(const struct su2_mpo* mpo, struct su2_tensor* mat);
