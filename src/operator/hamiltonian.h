/// \file hamiltonian.h
/// \brief Construction of common quantum Hamiltonians.

#pragma once

#include "mpo.h"


void construct_ising_1d_mpo(const int nsites, const double J, const double h, const double g, struct mpo* mpo);

void construct_heisenberg_xxz_1d_mpo(const int nsites, const double J, const double D, const double h, struct mpo* mpo);

void construct_bose_hubbard_1d_mpo(const int nsites, const long d, const double t, const double u, const double mu, struct mpo* mpo);

void construct_fermi_hubbard_1d_mpo(const int nsites, const double t, const double u, const double mu, struct mpo* mpo);

void construct_molecular_hamiltonian_mpo(const struct dense_tensor* restrict tkin, const struct dense_tensor* restrict vint, struct mpo* mpo);
void construct_molecular_hamiltonian_mpo_opt(const struct dense_tensor* restrict tkin, const struct dense_tensor* restrict vint, struct mpo* mpo);
