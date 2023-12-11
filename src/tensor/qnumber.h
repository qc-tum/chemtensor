/// \file qnumber.h
/// \brief Quantum numbers and corresponding utility functions.

#pragma once


//________________________________________________________________________________________________________________________
///
/// \brief Quantum number type for exploiting U(1) symmetries (like particle number or spin conservation).
///
/// For half-integer quantum numbers like spins, we store the physical quantum number times 2 to avoid rounding issues.
///
typedef int qnumber;


//________________________________________________________________________________________________________________________
///
/// \brief Tensor axis direction (orientation), equivalent to endowing quantum numbers with a sign factor.
///
enum tensor_axis_direction
{
	TENSOR_AXIS_IN  =  1,
	TENSOR_AXIS_OUT = -1
};
