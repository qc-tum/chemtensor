/// \file krylov.h
/// \brief Krylov subspace algorithms.

#pragma once

#include "numeric.h"
#include "util.h"


typedef void lanczos_linear_func_s(const ct_long n, const void* data, const float* v, float* ret);

typedef void lanczos_linear_func_d(const ct_long n, const void* data, const double* v, double* ret);

typedef void lanczos_linear_func_c(const ct_long n, const void* data, const scomplex* v, scomplex* ret);

typedef void lanczos_linear_func_z(const ct_long n, const void* data, const dcomplex* v, dcomplex* ret);


void lanczos_iteration_s(const ct_long n, lanczos_linear_func_s afunc, const void* adata, const float* vstart, const int maxiter,
	float* alpha, float* beta, float* v, int* numiter);

void lanczos_iteration_d(const ct_long n, lanczos_linear_func_d afunc, const void* adata, const double* vstart, const int maxiter,
	double* alpha, double* beta, double* v, int* numiter);

void lanczos_iteration_c(const ct_long n, lanczos_linear_func_c afunc, const void* adata, const scomplex* vstart, const int maxiter,
	float* alpha, float* beta, scomplex* v, int* numiter);

void lanczos_iteration_z(const ct_long n, lanczos_linear_func_z afunc, const void* adata, const dcomplex* vstart, const int maxiter,
	double* alpha, double* beta, dcomplex* v, int* numiter);

//________________________________________________________________________________________________________________________
//

int eigensystem_krylov_symmetric_s(const ct_long n, lanczos_linear_func_s afunc, const void* adata,
	const float* vstart, const int maxiter, const int numeig,
	float* lambda, float* u_ritz);

int eigensystem_krylov_symmetric_d(const ct_long n, lanczos_linear_func_d afunc, const void* adata,
	const double* vstart, const int maxiter, const int numeig,
	double* lambda, double* u_ritz);

int eigensystem_krylov_symmetric_c(const ct_long n, lanczos_linear_func_c afunc, const void* adata,
	const scomplex* vstart, const int maxiter, const int numeig,
	float* lambda, scomplex* u_ritz);

int eigensystem_krylov_symmetric_z(const ct_long n, lanczos_linear_func_z afunc, const void* adata,
	const dcomplex* vstart, const int maxiter, const int numeig,
	double* lambda, dcomplex* u_ritz);
