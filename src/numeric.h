/// \file numeric.h
/// \brief Numeric data types.

#pragma once

#include <stdlib.h>
#include <assert.h>


//________________________________________________________________________________________________________________________
///
/// \brief Numeric data type identifiers.
///
enum numeric_type
{
	SINGLE_REAL,     //!< float
	DOUBLE_REAL,     //!< double
	SINGLE_COMPLEX,  //!< float complex
	DOUBLE_COMPLEX,  //!< double complex
};


/// \brief Single-precision complex data type.
typedef float _Complex scomplex;

/// \brief Double-precision complex data type.
typedef double _Complex dcomplex;


//________________________________________________________________________________________________________________________
///
/// \brief Numeric data type size in bytes.
///
static inline size_t sizeof_numeric_type(const enum numeric_type dtype)
{
	switch (dtype)
	{
		case SINGLE_REAL:
		{
			return sizeof(float);
		}
		case DOUBLE_REAL:
		{
			return sizeof(double);
		}
		case SINGLE_COMPLEX:
		{
			return sizeof(scomplex);
		}
		case DOUBLE_COMPLEX:
		{
			return sizeof(dcomplex);
		}
		default:
		{
			// unknown data type
			assert(0);
			return 0;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Real numeric data type of the same precision.
///
static inline enum numeric_type numeric_real_type(const enum numeric_type dtype)
{
	switch (dtype)
	{
		case SINGLE_REAL:
		{
			return SINGLE_REAL;
		}
		case DOUBLE_REAL:
		{
			return DOUBLE_REAL;
		}
		case SINGLE_COMPLEX:
		{
			return SINGLE_REAL;
		}
		case DOUBLE_COMPLEX:
		{
			return DOUBLE_REAL;
		}
		default:
		{
			// unknown data type
			assert(0);
			return DOUBLE_REAL;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Return a pointer to a static constant variable of the provided data type representing one.
///
static inline const void* numeric_one(const enum numeric_type dtype)
{
	switch (dtype)
	{
		case SINGLE_REAL:
		{
			static const float one = 1;
			return &one;
		}
		case DOUBLE_REAL:
		{
			static const double one = 1;
			return &one;
		}
		case SINGLE_COMPLEX:
		{
			static const scomplex one = 1;
			return &one;
		}
		case DOUBLE_COMPLEX:
		{
			static const dcomplex one = 1;
			return &one;
		}
		default:
		{
			// unknown data type
			assert(0);
			return NULL;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Return a pointer to a static constant variable of the provided data type representing zero.
///
static inline const void* numeric_zero(const enum numeric_type dtype)
{
	switch (dtype)
	{
		case SINGLE_REAL:
		{
			static const float zero = 0;
			return &zero;
		}
		case DOUBLE_REAL:
		{
			static const double zero = 0;
			return &zero;
		}
		case SINGLE_COMPLEX:
		{
			static const scomplex zero = 0;
			return &zero;
		}
		case DOUBLE_COMPLEX:
		{
			static const dcomplex zero = 0;
			return &zero;
		}
		default:
		{
			// unknown data type
			assert(0);
			return NULL;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Return a pointer to a static constant variable of the provided data type representing -1.
///
static inline const void* numeric_neg_one(const enum numeric_type dtype)
{
	switch (dtype)
	{
		case SINGLE_REAL:
		{
			static const float neg_one = -1;
			return &neg_one;
		}
		case DOUBLE_REAL:
		{
			static const double neg_one = -1;
			return &neg_one;
		}
		case SINGLE_COMPLEX:
		{
			static const scomplex neg_one = -1;
			return &neg_one;
		}
		case DOUBLE_COMPLEX:
		{
			static const dcomplex neg_one = -1;
			return &neg_one;
		}
		default:
		{
			// unknown data type
			assert(0);
			return NULL;
		}
	}
}
