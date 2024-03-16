/// \file operation.h
/// \brief Higher-level tensor network operations.

#pragma once

#include "mps.h"
#include "mpo.h"


void mps_vdot(const struct mps* chi, const struct mps* psi, void* ret);


void operator_inner_product(const struct mps* chi, const struct mpo* op, const struct mps* psi, void* ret);
