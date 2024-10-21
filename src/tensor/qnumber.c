/// \file qnumber.c
/// \brief Quantum numbers and corresponding utility functions.

#include "qnumber.h"


//________________________________________________________________________________________________________________________
///
/// \brief Test whether two arrays of quantum numbers match entrywise.
///
bool qnumber_all_equal(const long n, const qnumber* restrict qnums0, const qnumber* restrict qnums1)
{
	for (long i = 0; i < n; i++)
	{
		if (qnums0[i] != qnums1[i]) {
			return false;
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Enumerate all pairwise sums of tuples formed from the provided quantum number arrays.
///
void qnumber_outer_sum(const int sign0, const qnumber* restrict qnums0, const long n0, const int sign1, const qnumber* restrict qnums1, const long n1, qnumber* restrict ret)
{
	for (long i = 0; i < n0; i++) {
		for (long j = 0; j < n1; j++) {
			ret[i*n1 + j] = sign0 * qnums0[i] + sign1 * qnums1[j];
		}
	}
}
