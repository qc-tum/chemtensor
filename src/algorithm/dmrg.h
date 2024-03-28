/// \file dmrg.h
/// \brief DMRG algorithm.

#pragma once

#include "mps.h"
#include "mpo.h"


int dmrg_singlesite(const struct mpo* hamiltonian, const int num_sweeps, const int maxiter_lanczos, struct mps* psi, double* en_sweeps);
