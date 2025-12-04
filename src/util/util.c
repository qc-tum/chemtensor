/// \file util.c
/// \brief Generic utility functions.

#include <math.h>
#include <complex.h>
#include <assert.h>
#include "util.h"
#include "cblas_ct.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Calculate the product of a list of integer numbers.
///
ct_long integer_product(const ct_long* x, const int n)
{
	assert(n >= 0); // n == 0 is still reasonable

	ct_long prod = 1;
	for (int i = 0; i < n; i++)
	{
		prod *= x[i];
	}

	return prod;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the integer power `base^exp`.
///
ct_long ipow(ct_long base, int exp)
{
	assert(exp >= 0);

	ct_long result = 1;
	while (exp != 0)
	{
		if ((exp & 1) == 1) {
			result *= base;
		}
		exp >>= 1;
		base *= base;
	}
	return result;
}


//________________________________________________________________________________________________________________________
///
/// \brief Test whether an integer map is a permutation of the list [0, ..., n - 1].
///
bool is_permutation(const int* map, const int n)
{
	for (int i = 0; i < n; i++) {
		if (map[i] < 0 || map[i] >= n) {
			return false;
		}
	}

	bool* indicator = ct_calloc(n, sizeof(bool));
	for (int i = 0; i < n; i++) {
		indicator[map[i]] = true;
	}
	for (int i = 0; i < n; i++) {
		if (!indicator[i])
		{
			ct_free(indicator);
			return false;
		}
	}
	ct_free(indicator);

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Whether a permutation is the identity permutation.
///
bool is_identity_permutation(const int* perm, const int n)
{
	for (int i = 0; i < n; i++) {
		if (perm[i] != i) {
			return false;
		}
	}
	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Uniform distance (infinity norm) between 'x' and 'y'.
///
/// The result is cast to double format (even if the actual entries are of single precision).
///
double uniform_distance(const enum numeric_type dtype, const ct_long n, const void* restrict x, const void* restrict y)
{
	switch (dtype)
	{
		case CT_SINGLE_REAL:
		{
			const float* xv = x;
			const float* yv = y;
			float d = 0;
			for (ct_long i = 0; i < n; i++)
			{
				d = fmaxf(d, fabsf(xv[i] - yv[i]));
			}
			return d;
		}
		case CT_DOUBLE_REAL:
		{
			const double* xv = x;
			const double* yv = y;
			double d = 0;
			for (ct_long i = 0; i < n; i++)
			{
				d = fmax(d, fabs(xv[i] - yv[i]));
			}
			return d;
		}
		case CT_SINGLE_COMPLEX:
		{
			const scomplex* xv = x;
			const scomplex* yv = y;
			float d = 0;
			for (ct_long i = 0; i < n; i++)
			{
				d = fmaxf(d, cabsf(xv[i] - yv[i]));
			}
			return d;
		}
		case CT_DOUBLE_COMPLEX:
		{
			const dcomplex* xv = x;
			const dcomplex* yv = y;
			double d = 0;
			for (ct_long i = 0; i < n; i++)
			{
				d = fmax(d, cabs(xv[i] - yv[i]));
			}
			return d;
		}
		default:
		{
			// unknown data type
			assert(false);
			return 0;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Euclidean norm of a vector.
///
double norm2(const enum numeric_type dtype, const ct_long n, const void* x)
{
	assert(n >= 0);

	switch (dtype)
	{
		case CT_SINGLE_REAL:
		{
			return cblas_snrm2(n, x, 1);
		}
		case CT_DOUBLE_REAL:
		{
			return cblas_dnrm2(n, x, 1);
		}
		case CT_SINGLE_COMPLEX:
		{
			return cblas_scnrm2(n, x, 1);
		}
		case CT_DOUBLE_COMPLEX:
		{
			return cblas_dznrm2(n, x, 1);
		}
		default:
		{
			// unknown data type
			assert(false);
			return 0;
		}
	}
}
