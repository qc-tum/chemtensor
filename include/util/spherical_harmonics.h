/// \file spherical_harmonics.h
/// \brief Evaluate spherical harmonics.

#pragma once

#include "qnumber.h"


void real_spherical_harmonics_f(const qnumber l, const float v[3], float* yv);

void real_spherical_harmonics(const qnumber l, const double v[3], double* yv);
