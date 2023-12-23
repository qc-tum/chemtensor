/// \file qnumber.c
/// \brief Quantum numbers and corresponding utility functions.

#include "qnumber.h"


//________________________________________________________________________________________________________________________
///
/// \brief Test whether two arrays of quantum numbers match entrywise.
///
bool qnumber_all_equal(const long n, const qnumber* restrict qnums0, const qnumber* restrict qnums1)
{
	for (int i = 0; i < n; i++)
	{
		if (qnums0[i] != qnums1[i]) {
			return false;
		}
	}

	return true;
}
