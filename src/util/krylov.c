/// \file krylov.c
/// \brief Krylov subspace algorithms.

#include <complex.h>
#include <cblas.h>
#include <lapacke.h>
#include <assert.h>
#include "krylov.h"
#include "aligned_memory.h"
#include "util.h"


#define DBL_EPSILON 2.2204460492503131e-16


//________________________________________________________________________________________________________________________
///
/// \brief Perform a "matrix free" Lanczos iteration for real-valued double precision vectors.
///
void lanczos_iteration_d(const long n, lanczos_linear_func_d afunc, const void* restrict adata, const double* restrict vstart, const int maxiter,
	double* restrict alpha, double* restrict beta, double* restrict v, int* restrict numiter)
{
	double* w = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(double));

	// set first "v" vector to normalized starting vector
	memcpy(v, vstart, n * sizeof(double));
	double nrm = cblas_dnrm2(n, v, 1);
	assert(nrm > 0);
	cblas_dscal(n, 1/nrm, v, 1);

	for (int j = 0; j < maxiter - 1; j++)
	{
		// w' = A v_j
		afunc(n, adata, &v[j*n], w);

		// alpha_j = <w', v_j>
		alpha[j] = cblas_ddot(n, w, 1, &v[j*n], 1);

		// w = w' - alpha_j v_j - beta_j v_{j-1}
		if (j > 0) {
			for (long i = 0; i < n; i++) {
				w[i] -= alpha[j] * v[j*n + i] + beta[j - 1] * v[(j - 1)*n + i];
			}
		}
		else {
			for (long i = 0; i < n; i++) {
				w[i] -= alpha[j] * v[j*n + i];
			}
		}

		// beta_{j+1} = ||w||
		beta[j] = cblas_dnrm2(n, w, 1);

		if (beta[j] < 100 * n * DBL_EPSILON)
		{
			// premature end of iterations
			(*numiter) = j + 1;
			aligned_free(w);
			return;
		}

		// v_{j+1} = w / beta[j+1]
		assert(beta[j] > 0);
		const double inv_beta = 1 / beta[j];
		for (long i = 0; i < n; i++)
		{
			v[(j + 1)*n + i] = inv_beta * w[i];
		}
	}

	// complete final iteration
	{
		int j = maxiter - 1;

		// w' = A v_j
		afunc(n, adata, &v[j*n], w);

		// alpha_j = <w', v_j>
		alpha[j] = cblas_ddot(n, w, 1, &v[j*n], 1);
	}

	aligned_free(w);

	(*numiter) = maxiter;
}


//________________________________________________________________________________________________________________________
///
/// \brief Perform a "matrix free" Lanczos iteration for complex-valued double precision vectors.
///
void lanczos_iteration_z(const long n, lanczos_linear_func_z afunc, const void* restrict adata, const dcomplex* restrict vstart, const int maxiter,
	double* restrict alpha, double* restrict beta, dcomplex* restrict v, int* restrict numiter)
{
	dcomplex* w = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(dcomplex));

	// set first "v" vector to normalized starting vector
	memcpy(v, vstart, n * sizeof(dcomplex));
	double nrm = cblas_dznrm2(n, v, 1);
	assert(nrm > 0);
	cblas_zdscal(n, 1/nrm, v, 1);

	for (int j = 0; j < maxiter - 1; j++)
	{
		// w' = A v_j
		afunc(n, adata, &v[j*n], w);

		// alpha_j = <w', v_j>
		dcomplex t;
		cblas_zdotc_sub(n, w, 1, &v[j*n], 1, &t);
		alpha[j] = creal(t);  // should be real for self-adjoint linear operation

		// w = w' - alpha_j v_j - beta_j v_{j-1}
		if (j > 0) {
			for (long i = 0; i < n; i++) {
				w[i] -= alpha[j] * v[j*n + i] + beta[j - 1] * v[(j - 1)*n + i];
			}
		}
		else {
			for (long i = 0; i < n; i++) {
				w[i] -= alpha[j] * v[j*n + i];
			}
		}

		// beta_{j+1} = ||w||
		beta[j] = cblas_dznrm2(n, w, 1);

		if (beta[j] < 100 * n * DBL_EPSILON)
		{
			// premature end of iterations
			(*numiter) = j + 1;
			aligned_free(w);
			return;
		}

		// v_{j+1} = w / beta[j+1]
		assert(beta[j] > 0);
		const double inv_beta = 1 / beta[j];
		for (long i = 0; i < n; i++)
		{
			v[(j + 1)*n + i] = inv_beta * w[i];
		}
	}

	// complete final iteration
	{
		int j = maxiter - 1;

		// w' = A v_j
		afunc(n, adata, &v[j*n], w);

		// alpha_j = <w', v_j>
		dcomplex t;
		cblas_zdotc_sub(n, w, 1, &v[j*n], 1, &t);
		alpha[j] = creal(t);  // should be real for self-adjoint linear operation
	}

	aligned_free(w);

	(*numiter) = maxiter;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute Krylov subspace approximation of eigenvalues and vectors, real symmetric case.
///
int eigensystem_krylov_symmetric(const long n, lanczos_linear_func_d afunc, const void* restrict adata,
	const double* restrict vstart, const int maxiter, const int numeig,
	double* restrict lambda, double* restrict u_ritz)
{
	assert(numeig <= maxiter);

	double* alpha = aligned_alloc(MEM_DATA_ALIGN, maxiter       * sizeof(double));
	double* beta  = aligned_alloc(MEM_DATA_ALIGN, (maxiter - 1) * sizeof(double));
	double* v     = aligned_alloc(MEM_DATA_ALIGN, maxiter*n     * sizeof(double));
	int numiter;
	lanczos_iteration_d(n, afunc, adata, vstart, maxiter, alpha, beta, v, &numiter);

	if (numiter < numeig) {
		fprintf(stderr, "Lanczos iteration stopped after %i iterations, cannot compute %i eigenvalues\n", numiter, numeig);
		return -1;
	}

	// diagonalize Hessenberg matrix
	double* u = aligned_alloc(MEM_DATA_ALIGN, numiter*numiter * sizeof(double));
	lapack_int info = LAPACKE_dsteqr(LAPACK_ROW_MAJOR, 'I', numiter, alpha, beta, u, numiter);
	if (info != 0) {
		fprintf(stderr, "LAPACK function 'dsteqr()' failed, return value: %i\n", info);
		return -2;
	}

	// 'alpha' now contains the eigenvalues
	memcpy(lambda, alpha, numeig * sizeof(double));

	// compute Ritz eigenvectors
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, numeig, numiter, 1., v, n, u, numiter, 0., u_ritz, numeig);

	// clean up
	aligned_free(u);
	aligned_free(v);
	aligned_free(beta);
	aligned_free(alpha);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute Krylov subspace approximation of eigenvalues and vectors, complex Hermitian case.
///
int eigensystem_krylov_hermitian(const long n, lanczos_linear_func_z afunc, const void* restrict adata,
	const dcomplex* restrict vstart, const int maxiter, const int numeig,
	double* restrict lambda, dcomplex* restrict u_ritz)
{
	assert(numeig <= maxiter);

	double* alpha = aligned_alloc(MEM_DATA_ALIGN, maxiter       * sizeof(double));
	double* beta  = aligned_alloc(MEM_DATA_ALIGN, (maxiter - 1) * sizeof(double));
	dcomplex* v   = aligned_alloc(MEM_DATA_ALIGN, maxiter*n     * sizeof(dcomplex));
	int numiter;
	lanczos_iteration_z(n, afunc, adata, vstart, maxiter, alpha, beta, v, &numiter);

	if (numiter < numeig) {
		fprintf(stderr, "Lanczos iteration stopped after %i iterations, cannot compute %i eigenvalues\n", numiter, numeig);
		return -1;
	}

	// diagonalize Hessenberg matrix
	double* u = aligned_alloc(MEM_DATA_ALIGN, numiter*numiter * sizeof(double));
	lapack_int info = LAPACKE_dsteqr(LAPACK_ROW_MAJOR, 'I', numiter, alpha, beta, u, numiter);
	if (info != 0) {
		fprintf(stderr, "LAPACK function 'dsteqr()' failed, return value: %i\n", info);
		return -2;
	}

	// 'alpha' now contains the eigenvalues
	memcpy(lambda, alpha, numeig * sizeof(double));

	// compute Ritz eigenvectors
	// require complex 'u' entries for matrix multiplication
	dcomplex* uz = aligned_alloc(MEM_DATA_ALIGN, numiter*numiter * sizeof(dcomplex));
	for (int i = 0; i < numiter*numiter; i++) {
		uz[i] = (dcomplex)u[i];
	}
	const dcomplex one  = 1;
	const dcomplex zero = 0;
	cblas_zgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, numeig, numiter, &one, v, n, uz, numiter, &zero, u_ritz, numeig);

	// clean up
	aligned_free(uz);
	aligned_free(u);
	aligned_free(v);
	aligned_free(beta);
	aligned_free(alpha);

	return 0;
}
