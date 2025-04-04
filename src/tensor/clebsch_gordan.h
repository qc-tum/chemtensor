/// \file clebsch_gordan.h
/// \brief Clebsch-Gordan coefficient tables.
///
/// 'j' quantum numbers are represented times 2, to support half-integers.
/// 'm' quantum numbers are enumerated as (j, j - 1, ..., -j).
///
/// This file has been generated by 'clebsch_gordan_gen.py'.

#pragma once

#include "qnumber.h"


double clebsch_gordan(const qnumber j1, const qnumber j2, const qnumber j3, const int im1, const int im2, const int im3);
