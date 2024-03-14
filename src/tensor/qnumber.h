/// \file qnumber.h
/// \brief Quantum numbers and corresponding utility functions.

#pragma once

#include <stdbool.h>


//________________________________________________________________________________________________________________________
///
/// \brief Quantum number type for exploiting U(1) symmetries (like particle number or spin conservation).
///
/// For half-integer quantum numbers like spins, we store the physical quantum number times 2 to avoid rounding issues.
///
typedef int qnumber;


bool qnumber_all_equal(const long n, const qnumber* restrict qnums0, const qnumber* restrict qnums1);


//________________________________________________________________________________________________________________________
///
/// \brief Tensor axis direction (orientation), equivalent to endowing quantum numbers with a sign factor.
///
enum tensor_axis_direction
{
	TENSOR_AXIS_IN  = -1,
	TENSOR_AXIS_OUT =  1,
};
