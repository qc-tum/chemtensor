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
/// \brief Encode a pair of quantum numbers into a single quantum number.
///
static inline qnumber encode_quantum_number_pair(const qnumber qa, const qnumber qb)
{
	return (qa << 16) + qb;
}

//________________________________________________________________________________________________________________________
///
/// \brief Decode a quantum number into two separate quantum numbers.
///
static inline void decode_quantum_number_pair(const qnumber qnum, qnumber* qa, qnumber* qb)
{
	qnumber q1 = qnum % (1 << 16);
	if (q1 >= (1 << 15)) {
		q1 -= (1 << 16);
	}
	else if (q1 < -(1 << 15)) {
		q1 += (1 << 16);
	}
	qnumber q0 = (qnum - q1) >> 16;

	(*qa) = q0;
	(*qb) = q1;
}


//________________________________________________________________________________________________________________________
///
/// \brief Tensor axis direction (orientation), equivalent to endowing quantum numbers with a sign factor.
///
enum tensor_axis_direction
{
	TENSOR_AXIS_IN  = -1,
	TENSOR_AXIS_OUT =  1,
};


void qnumber_outer_sum(const int sign0, const qnumber* restrict qnums0, const long n0, const int sign1, const qnumber* restrict qnums1, const long n1, qnumber* restrict ret);
