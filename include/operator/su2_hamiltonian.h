/// \file su2_hamiltonian.h
/// \brief Construction of common SU(2) symmetric quantum Hamiltonians.

#pragma once

#include "su2_mpo.h"


void construct_heisenberg_1d_su2_mpo(const int nsites, const double J, struct su2_mpo* mpo);
