/// \file hamiltonian.h
/// \brief Construction of common quantum Hamiltonians.

#pragma once

#include "mpo.h"


void construct_ising_1d_mpo_assembly(const int nsites, const double J, const double h, const double g, struct mpo_assembly* assembly);

void construct_heisenberg_xxz_1d_mpo_assembly(const int nsites, const double J, const double D, const double h, struct mpo_assembly* assembly);

void construct_bose_hubbard_1d_mpo_assembly(const int nsites, const long d, const double t, const double u, const double mu, struct mpo_assembly* assembly);

//________________________________________________________________________________________________________________________
///
/// \brief Encode a particle number and spin quantum number for the Fermi-Hubbard Hamiltonian into a single quantum number.
///
static inline qnumber fermi_hubbard_encode_quantum_numbers(const qnumber q_pnum, const qnumber q_spin)
{
	return (q_pnum << 16) + q_spin;
}

//________________________________________________________________________________________________________________________
///
/// \brief Decode a quantum number of the Fermi-Hubbard Hamiltonian into separate particle number and spin quantum numbers.
///
static inline void fermi_hubbard_decode_quantum_numbers(const qnumber qnum, qnumber* q_pnum, qnumber* q_spin)
{
	qnumber qs = qnum % (1 << 16);
	if (qs >= (1 << 15)) {
		qs -= (1 << 16);
	}
	else if (qs < -(1 << 15)) {
		qs += (1 << 16);
	}
	qnumber qp = (qnum - qs) >> 16;

	(*q_pnum) = qp;
	(*q_spin) = qs;
}

void construct_fermi_hubbard_1d_mpo_assembly(const int nsites, const double t, const double u, const double mu, struct mpo_assembly* assembly);

void construct_molecular_hamiltonian_mpo_assembly(const struct dense_tensor* restrict tkin, const struct dense_tensor* restrict vint, struct mpo_assembly* assembly);
void construct_molecular_hamiltonian_mpo_assembly_opt(const struct dense_tensor* restrict tkin, const struct dense_tensor* restrict vint, struct mpo_assembly* assembly);
