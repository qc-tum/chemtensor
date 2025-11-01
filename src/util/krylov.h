/// \file krylov.h
/// \brief Krylov subspace algorithms.

#pragma once

#include "numeric.h"
#include "util.h"


typedef void lanczos_linear_func_d(const ct_long n, const void* restrict data, const double* restrict v, double* restrict ret);

typedef void lanczos_linear_func_z(const ct_long n, const void* restrict data, const dcomplex* restrict v, dcomplex* restrict ret);


void lanczos_iteration_d(const ct_long n, lanczos_linear_func_d afunc, const void* restrict adata, const double* restrict vstart, const int maxiter,
	double* restrict alpha, double* restrict beta, double* restrict v, int* restrict numiter);

void lanczos_iteration_z(const ct_long n, lanczos_linear_func_z afunc, const void* restrict adata, const dcomplex* restrict vstart, const int maxiter,
	double* restrict alpha, double* restrict beta, dcomplex* restrict v, int* restrict numiter);

//________________________________________________________________________________________________________________________
//

int eigensystem_krylov_symmetric(const ct_long n, lanczos_linear_func_d afunc, const void* restrict adata,
	const double* restrict vstart, const int maxiter, const int numeig,
	double* restrict lambda, double* restrict u_ritz);

int eigensystem_krylov_hermitian(const ct_long n, lanczos_linear_func_z afunc, const void* restrict adata,
	const dcomplex* restrict vstart, const int maxiter, const int numeig,
	double* restrict lambda, dcomplex* restrict u_ritz);
