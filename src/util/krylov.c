/// \file krylov.c
/// \brief Krylov subspace algorithms.

#include <complex.h>
#include <assert.h>
#include <cblas.h>
#include "lapack_ct.h"
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
	double* w = ct_malloc(n * sizeof(double));

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
			ct_free(w);
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

	ct_free(w);

	(*numiter) = maxiter;
}


//________________________________________________________________________________________________________________________
///
/// \brief Perform a "matrix free" Lanczos iteration for complex-valued double precision vectors.
///
void lanczos_iteration_z(const long n, lanczos_linear_func_z afunc, const void* restrict adata, const dcomplex* restrict vstart, const int maxiter,
	double* restrict alpha, double* restrict beta, dcomplex* restrict v, int* restrict numiter)
{
	dcomplex* w = ct_malloc(n * sizeof(dcomplex));

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
			ct_free(w);
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

	ct_free(w);

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

	double* alpha = ct_malloc(maxiter       * sizeof(double));
	double* beta  = ct_malloc((maxiter - 1) * sizeof(double));
	double* v     = ct_malloc(maxiter*n     * sizeof(double));
	int numiter;
	lanczos_iteration_d(n, afunc, adata, vstart, maxiter, alpha, beta, v, &numiter);

	if (numiter < numeig) {
		fprintf(stderr, "Lanczos iteration stopped after %i iterations, cannot compute %i eigenvalues\n", numiter, numeig);
		return -1;
	}

	// diagonalize Hessenberg matrix
	double* u = ct_malloc(numiter*numiter * sizeof(double));
	double* work = ct_malloc(imax(1, 2*numiter - 2) * sizeof(double));
	lapack_int info;
	LAPACK_dsteqr("I", &numiter, alpha, beta, u, &numiter, work, &info);
	ct_free(work);
	if (info != 0) {
		fprintf(stderr, "LAPACK function 'dsteqr' failed, return value: %i\n", info);
		return -2;
	}

	// 'alpha' now contains the eigenvalues
	memcpy(lambda, alpha, numeig * sizeof(double));

	// compute Ritz eigenvectors
	// using transposition of 'u' for implicit column- to row-major order conversion
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, n, numeig, numiter, 1., v, n, u, numiter, 0., u_ritz, numeig);

	// clean up
	ct_free(u);
	ct_free(v);
	ct_free(beta);
	ct_free(alpha);

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

	double* alpha = ct_malloc(maxiter       * sizeof(double));
	double* beta  = ct_malloc((maxiter - 1) * sizeof(double));
	dcomplex* v   = ct_malloc(maxiter*n     * sizeof(dcomplex));
	int numiter;
	lanczos_iteration_z(n, afunc, adata, vstart, maxiter, alpha, beta, v, &numiter);

	if (numiter < numeig) {
		fprintf(stderr, "Lanczos iteration stopped after %i iterations, cannot compute %i eigenvalues\n", numiter, numeig);
		return -1;
	}

	// diagonalize Hessenberg matrix
	double* u = ct_malloc(numiter*numiter * sizeof(double));
	double* work = ct_malloc(imax(1, 2*numiter - 2) * sizeof(double));
	lapack_int info;
	LAPACK_dsteqr("I", &numiter, alpha, beta, u, &numiter, work, &info);
	ct_free(work);
	if (info != 0) {
		fprintf(stderr, "LAPACK function 'dsteqr' failed, return value: %i\n", info);
		return -2;
	}

	// 'alpha' now contains the eigenvalues
	memcpy(lambda, alpha, numeig * sizeof(double));

	// compute Ritz eigenvectors
	// require complex 'u' entries for matrix multiplication
	dcomplex* uz = ct_malloc(numiter*numiter * sizeof(dcomplex));
	for (int i = 0; i < numiter*numiter; i++) {
		uz[i] = (dcomplex)u[i];
	}
	const dcomplex one  = 1;
	const dcomplex zero = 0;
	// using transposition of 'uz' for implicit column- to row-major order conversion
	cblas_zgemm(CblasRowMajor, CblasTrans, CblasTrans, n, numeig, numiter, &one, v, n, uz, numiter, &zero, u_ritz, numeig);

	// clean up
	ct_free(uz);
	ct_free(u);
	ct_free(v);
	ct_free(beta);
	ct_free(alpha);

	return 0;
}
