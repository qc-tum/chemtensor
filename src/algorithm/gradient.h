/// \file gradient.h
/// \brief Gradient computation for operators.

#pragma once

#include "mps.h"
#include "mpo.h"


void operator_average_coefficient_gradient(const struct mpo_assembly* assembly, const struct mps* psi, const struct mps* chi, void* avr, void* dcoeff);
