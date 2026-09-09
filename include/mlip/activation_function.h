/// \file activation_function.h
/// \brief Neural network activation functions.

#pragma once

#include <math.h>
#include <complex.h>
#include "numeric.h"


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the ReLU (rectified linear unit) function for single-precision real numbers.
///
static inline float reluf(const float x)
{
	return fmaxf(0.f, x);
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the ReLU (rectified linear unit) function for double-precision real numbers.
///
static inline double relu(const double x)
{
	return fmax(0.0, x);
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the ReLU (rectified linear unit) function for single-precision complex numbers.
///
static inline scomplex creluf(const scomplex x)
{
	return fmaxf(0.f, crealf(x)) + _Complex_I * fmaxf(0.f, cimagf(x));
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the ReLU (rectified linear unit) function for double-precision complex numbers.
///
static inline dcomplex crelu(const dcomplex x)
{
	return fmax(0.0, creal(x)) + _Complex_I * fmax(0.0, cimag(x));
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the SiLU (sigmoid linear unit) function for single-precision real numbers.
///
static inline float siluf(const float x)
{
	return x / (1.f + expf(-x));
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the SiLU (sigmoid linear unit) function for double-precision real numbers.
///
static inline double silu(const double x)
{
	return x / (1.0 + exp(-x));
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the SiLU (sigmoid linear unit) function for single-precision complex numbers.
///
static inline scomplex csiluf(const scomplex x)
{
	return x / (1.f + cexpf(-x));
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the SiLU (sigmoid linear unit) function for double-precision complex numbers.
///
static inline dcomplex csilu(const dcomplex x)
{
	return x / (1.0 + cexp(-x));
}
