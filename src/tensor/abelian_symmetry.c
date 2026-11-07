/// \file abelian_symmetry.c
/// \brief Abelian symmetry description, used by block-sparse tensors.

#include "abelian_symmetry.h"


//________________________________________________________________________________________________________________________
///
/// \brief "Zero" quantum number addition function, always yielding zero.
///
static qnumber qadd_zero(const qnumber base, const int sign, const qnumber qnum)
{
	// suppress unused parameter warnings
	(void)base;
	(void)sign;
	(void)qnum;
	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief "Default" (affine) quantum number addition function.
///
static qnumber qadd_default(const qnumber base, const int sign, const qnumber qnum)
{
	return base + sign * qnum;
}


//________________________________________________________________________________________________________________________
///
/// \brief Parity (Z_2) quantum number addition function.
///
static qnumber qadd_parity(const qnumber base, const int sign, const qnumber qnum)
{
	// unsigned modulo 2
	return (base + sign * qnum) & 1;
}


//________________________________________________________________________________________________________________________
///
/// \brief Table of abelian quantum number addition functions.
///
/// Entries must match the indices of 'enum abelian_symmetry'.
///
abelian_qnumber_addition_func* abelian_qnumber_addition_functions[NUM_ABELIAN_SYMMETRIES] =
{
	qadd_zero,     // ABELIAN_SYMMETRY_NONE
	qadd_default,  // ABELIAN_SYMMETRY_U1
	qadd_parity,   // ABELIAN_SYMMETRY_Z2
	qadd_default,  // ABELIAN_SYMMETRY_U1_U1, assuming that quantum number tuple is encoded as a single integer
};


//________________________________________________________________________________________________________________________
///
/// \brief Enumerate all pairwise sums of tuples formed from the provided quantum number arrays.
///
void qnumber_outer_sum(const enum abelian_symmetry symm, const int sign0, const qnumber* restrict qnums0, const ct_long n0, const int sign1, const qnumber* restrict qnums1, const ct_long n1, qnumber* restrict ret)
{
	assert(0 <= symm && symm < NUM_ABELIAN_SYMMETRIES);

	for (ct_long i = 0; i < n0; i++) {
		for (ct_long j = 0; j < n1; j++) {
			ret[i*n1 + j] =
				abelian_qnumber_addition_functions[symm](
				abelian_qnumber_addition_functions[symm](
					0, sign0, qnums0[i]),
					   sign1, qnums1[j]);
		}
	}
}
