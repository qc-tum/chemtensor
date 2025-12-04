/// \file util.h
/// \brief Generic utility functions.

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "numeric.h"


/// \brief Universal "long" integer, used for tensor dimensions.
typedef int64_t ct_long;


//________________________________________________________________________________________________________________________
///
/// \brief Minimum of two integers.
///
static inline int imin(const int a, const int b)
{
	return (a <= b) ? a : b;
}


//________________________________________________________________________________________________________________________
///
/// \brief Maximum of two integers.
///
static inline int imax(const int a, const int b)
{
	return (a >= b) ? a : b;
}


//________________________________________________________________________________________________________________________
///
/// \brief Minimum of two long integers.
///
static inline ct_long lmin(const ct_long a, const ct_long b)
{
	return (a <= b) ? a : b;
}


//________________________________________________________________________________________________________________________
///
/// \brief Maximum of two long integers.
///
static inline ct_long lmax(const ct_long a, const ct_long b)
{
	return (a >= b) ? a : b;
}


//________________________________________________________________________________________________________________________
///
/// \brief Square function x -> x^2.
///
static inline double square(const double x)
{
	return x*x;
}


//________________________________________________________________________________________________________________________
//


ct_long integer_product(const ct_long* x, const int n);


ct_long ipow(ct_long base, int exp);


//________________________________________________________________________________________________________________________
//


bool is_permutation(const int* map, const int n);

bool is_identity_permutation(const int* perm, const int n);


//________________________________________________________________________________________________________________________
//


double uniform_distance(const enum numeric_type dtype, const ct_long n, const void* x, const void* y);


double norm2(const enum numeric_type dtype, const ct_long n, const void* x);
