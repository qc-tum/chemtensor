/// \file gradient.h
/// \brief Gradient computation for operators.

#pragma once

#include "mps.h"
#include "mpo_graph.h"


void operator_average_coefficient_gradient(const struct mpo_graph* graph, const struct dense_tensor* opmap, const void* coeffmap, const long num_coeffs, const struct mps* psi, const struct mps* chi, void* avr, void* dcoeff);
