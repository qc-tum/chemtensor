/// \file su2_chain_ops.h
/// \brief Higher-level tensor network operations on a chain topology for SU(2) symmetric tensors.

#pragma once

#include "su2_mps.h"
#include "su2_mpo.h"


void su2_contraction_operator_step_right(const struct su2_tensor* a, const struct su2_tensor* b,
	const struct su2_tensor* w, const struct su2_tensor* r, struct su2_tensor* r_next);

void su2_contraction_operator_step_left(const struct su2_tensor* a, const struct su2_tensor* b,
	const struct su2_tensor* w, const struct su2_tensor* l, struct su2_tensor* l_next);


void su2_mpo_inner_product(const struct su2_mps* chi, const struct su2_mpo* op, const struct su2_mps* psi, struct su2_tensor* ret);
