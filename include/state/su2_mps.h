/// \file su2_mps.h
/// \brief SU(2) symmetric matrix product state (MPS) data structure.

#pragma once

#include "su2_tensor.h"
#include "bond_ops.h"


//________________________________________________________________________________________________________________________
///
/// \brief SU(2) symmetric matrix product state (MPS) data structure.
///
struct su2_mps
{
	struct su2_tensor* a;   //!< tensors associated with sites, with axis ordering "left virtual, physical, right virtual", and a fusion-splitting tree with a single node; array of length 'nsites'
	int nsites;             //!< number of sites
};


//________________________________________________________________________________________________________________________
//

// allocation and construction

void allocate_empty_su2_mps(const int nsites, struct su2_mps* mps);

void allocate_su2_mps(
	const enum numeric_type dtype, const int nsites,
	const struct su2_irreducible_list* site_irreps, const ct_long* site_dim_degen,
	const struct su2_irreducible_list* bond_irreps, const ct_long** bond_dim_degen,
	struct su2_mps* mps);

void delete_su2_mps(struct su2_mps* mps);

void construct_random_su2_mps(
	const enum numeric_type dtype, const int nsites,
	const struct su2_irreducible_list* site_irreps, const ct_long* site_dim_degen, const qnumber irrep_sector,
	const qnumber max_bond_irrep, const ct_long max_bond_dim_degen,
	struct rng_state* rng_state, struct su2_mps* mps);

bool su2_mps_is_consistent(const struct su2_mps* mps);


//________________________________________________________________________________________________________________________
//

// orthonormalization and canonical forms

/// \brief SU(2) MPS orthonormalization mode.
enum su2_mps_orthonormalization_mode
{
	SU2_MPS_ORTHONORMAL_LEFT  = 0,  //!< left-orthonormal
	SU2_MPS_ORTHONORMAL_RIGHT = 1,  //!< right-orthonormal
};

void su2_mps_local_orthonormalize_qr(struct su2_tensor* a, struct su2_tensor* a_next);

void su2_mps_local_orthonormalize_rq(struct su2_tensor* a, struct su2_tensor* a_prev);

void su2_mps_create_dummy_head_tensor(const struct su2_tensor* a0, struct su2_tensor* a_head);
void su2_mps_create_dummy_tail_tensor(const struct su2_tensor* al, struct su2_tensor* a_tail);

double su2_mps_orthonormalize_qr(struct su2_mps* mps, const enum su2_mps_orthonormalization_mode mode);


//________________________________________________________________________________________________________________________
//

// splitting and merging

void su2_mps_contract_tensor_pair(const struct su2_tensor* a0, const struct su2_tensor* a1, struct su2_tensor* a);

void su2_mps_merge_tensor_pair(const struct su2_tensor* a0, const struct su2_tensor* a1, struct su2_tensor* a);


//________________________________________________________________________________________________________________________
//

// conversion to a statevector (intended for testing)

void su2_mps_to_statevector(const struct su2_mps* mps, struct su2_tensor* vec);
