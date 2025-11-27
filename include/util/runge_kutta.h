/// \file runge_kutta.h
/// \brief Runge-Kutta methods.

#pragma once

#include "block_sparse_tensor.h"


typedef void ode_func_block_sparse(const double t, const struct block_sparse_tensor* restrict y, const void* restrict data, struct block_sparse_tensor* restrict ret);


void runge_kutta_4_block_sparse(const double t, const struct block_sparse_tensor* restrict y, ode_func_block_sparse func, const void* fdata, const double h, struct block_sparse_tensor* restrict y_next);
