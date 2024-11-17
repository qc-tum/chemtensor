/// \file tree_ops.h
/// \brief Higher-level tensor network operations on a tree topology.

#pragma once

#include "ttns.h"
#include "ttno.h"


void ttno_inner_product(const struct ttns* chi, const struct ttno* op, const struct ttns* psi, void* ret);
