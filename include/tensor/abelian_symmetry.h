/// \file abelian_symmetry.h
/// \brief Abelian symmetry description, used by block-sparse tensors.

#pragma once

#include "qnumber.h"


//________________________________________________________________________________________________________________________
///
/// \brief Abelian symmetries.
///
enum abelian_symmetry
{
	ABELIAN_SYMMETRY_NONE  = 0,  //!< without any symmetry; all quantum numbers set to zero
	ABELIAN_SYMMETRY_U1    = 1,  //!< U(1) symmetry, e.g., particle or charge conservation
	ABELIAN_SYMMETRY_Z2    = 2,  //!< Z_2 symmetry, e.g., parity conservation
	ABELIAN_SYMMETRY_U1_U1 = 3,  //!< U(1) x U(1) symmetry, e.g., simultaneous particle and spin conservation
	NUM_ABELIAN_SYMMETRIES = 4,  //!< number of abelian symmetries
};


//________________________________________________________________________________________________________________________
///
/// \brief Quantum number addition function type for abelian symmetries.
///
typedef qnumber abelian_qnumber_addition_func(const qnumber base, const int sign, const qnumber qnum);


extern abelian_qnumber_addition_func* abelian_qnumber_addition_functions[NUM_ABELIAN_SYMMETRIES];


//________________________________________________________________________________________________________________________
//


void qnumber_outer_sum(const enum abelian_symmetry symm, const int sign0, const qnumber* qnums0, const ct_long n0, const int sign1, const qnumber* qnums1, const ct_long n1, qnumber* ret);
