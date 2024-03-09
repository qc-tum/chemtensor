/// \file operation.h
/// \brief Higher-level tensor network operations.

#pragma once

#include "mps.h"


void mps_vdot(const struct mps* chi, const struct mps* psi, void* ret);
