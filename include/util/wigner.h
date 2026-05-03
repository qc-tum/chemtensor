/// \file wigner.h
/// \brief Evaluate the Wigner D-matrix.

#pragma once

#include "qnumber.h"
#include "numeric.h"


void wigner_small_d(const qnumber j, const double theta, double* w);

void wigner_d(const qnumber j, const double psi, const double theta, const double phi, dcomplex* w);

void real_wigner_d(const qnumber j, const double psi, const double theta, const double phi, double* w);
