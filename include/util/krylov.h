/// \file krylov.h
/// \brief Krylov subspace algorithms.

#pragma once

#include "numeric.h"
#include "util.h"


typedef void lanczos_linear_func_d(const ct_long n, const void* data, const double* v, double* ret);

typedef void lanczos_linear_func_z(const ct_long n, const void* data, const dcomplex* v, dcomplex* ret);


void lanczos_iteration_d(const ct_long n, lanczos_linear_func_d afunc, const void* adata, const double* vstart, const int maxiter,
	double* alpha, double* beta, double* v, int* numiter);

void lanczos_iteration_z(const ct_long n, lanczos_linear_func_z afunc, const void* adata, const dcomplex* vstart, const int maxiter,
	double* alpha, double* beta, dcomplex* v, int* numiter);

//________________________________________________________________________________________________________________________
//

int eigensystem_krylov_symmetric(const ct_long n, lanczos_linear_func_d afunc, const void* adata,
	const double* vstart, const int maxiter, const int numeig,
	double* lambda, double* u_ritz);

int eigensystem_krylov_hermitian(const ct_long n, lanczos_linear_func_z afunc, const void* adata,
	const dcomplex* vstart, const int maxiter, const int numeig,
	double* lambda, dcomplex* u_ritz);
