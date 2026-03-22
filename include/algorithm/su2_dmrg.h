/// \file su2_dmrg.h
/// \brief DMRG algorithm for SU(2) symmetric tensors.

#pragma once

#include "su2_mps.h"
#include "su2_mpo.h"


int su2_dmrg_singlesite(const struct su2_mpo* hamiltonian, const int num_sweeps, const int maxiter_lanczos, struct su2_mps* psi, double* en_sweeps);

int su2_dmrg_twosite(const struct su2_mpo* hamiltonian, const int num_sweeps, const int maxiter_lanczos, const double tol_split, const ct_long max_vdim,
	struct su2_mps* psi, double* en_sweeps, double* entropy);
