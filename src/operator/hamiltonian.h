/// \file hamiltonian.h
/// \brief Construction of common quantum Hamiltonians.

#pragma once

#include "mpo.h"


void construct_ising_1d_mpo_assembly(const int nsites, const double J, const double h, const double g, struct mpo_assembly* assembly);

void construct_heisenberg_xxz_1d_mpo_assembly(const int nsites, const double J, const double D, const double h, struct mpo_assembly* assembly);

void construct_bose_hubbard_1d_mpo_assembly(const int nsites, const long d, const double t, const double u, const double mu, struct mpo_assembly* assembly);

void construct_fermi_hubbard_1d_mpo_assembly(const int nsites, const double t, const double u, const double mu, struct mpo_assembly* assembly);

void construct_molecular_hamiltonian_mpo_assembly(const struct dense_tensor* restrict tkin, const struct dense_tensor* restrict vint, const bool optimize, struct mpo_assembly* assembly);

void construct_spin_molecular_hamiltonian_mpo_assembly(const struct dense_tensor* restrict tkin, const struct dense_tensor* restrict vint, const bool optimize, struct mpo_assembly* assembly);

void construct_quadratic_fermionic_mpo_assembly(const int nsites, const double* coeffc, const double* coeffa, struct mpo_assembly* assembly);

void construct_quadratic_spin_fermionic_mpo_assembly(const int nsites, const double* coeffc, const double* coeffa, const int sigma, struct mpo_assembly* assembly);
