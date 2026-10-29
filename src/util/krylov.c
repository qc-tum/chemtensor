/// \file krylov.c
/// \brief Krylov subspace algorithms.

#include <math.h>
#include <complex.h>
#include <assert.h>
#include <stdio.h>
#include "krylov.h"
#include "cblas_ct.h"
#include "lapack_ct.h"
#include "aligned_memory.h"
#include "util.h"


#ifndef FLT_EPSILON
// 2^{-23}
#define FLT_EPSILON 1.1920928955078125e-7
#endif

#ifndef DBL_EPSILON
// 2^{-52}
#define DBL_EPSILON 2.2204460492503131e-16
#endif


//________________________________________________________________________________________________________________________
///
/// \brief Perform a "matrix free" Lanczos iteration for real-valued single precision vectors.
///
void lanczos_iteration_s(const ct_long n, lanczos_linear_func_s afunc, const void* restrict adata, const float* restrict vstart, const int maxiter,
	float* restrict alpha, float* restrict beta, float* restrict v, int* restrict numiter)
{
	float* w = ct_malloc(n * sizeof(float));

	// set first "v" vector to normalized starting vector
	memcpy(v, vstart, n * sizeof(float));
	float nrm = cblas_snrm2(n, v, 1);
	assert(nrm > 0);
	cblas_sscal(n, 1/nrm, v, 1);

	for (int j = 0; j < maxiter - 1; j++)
	{
		// w' = A v_j
		afunc(n, adata, &v[j*n], w);

		// alpha_j = <w', v_j>
		alpha[j] = cblas_sdot(n, w, 1, &v[j*n], 1);

		// w = w' - alpha_j v_j - beta_j v_{j-1}
		if (j > 0) {
			for (ct_long i = 0; i < n; i++) {
				w[i] -= alpha[j] * v[j*n + i] + beta[j - 1] * v[(j - 1)*n + i];
			}
		}
		else {
			for (ct_long i = 0; i < n; i++) {
				w[i] -= alpha[j] * v[j*n + i];
			}
		}

		// beta_{j+1} = ||w||
		beta[j] = cblas_snrm2(n, w, 1);

		if (beta[j] < 100 * n * (float)FLT_EPSILON)
		{
			// premature end of iterations
			(*numiter) = j + 1;
			ct_free(w);
			return;
		}

		// v_{j+1} = w / beta[j+1]
		assert(beta[j] > 0);
		const float inv_beta = 1 / beta[j];
		for (ct_long i = 0; i < n; i++)
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
		alpha[j] = cblas_sdot(n, w, 1, &v[j*n], 1);
	}

	ct_free(w);

	(*numiter) = maxiter;
}


//________________________________________________________________________________________________________________________
///
/// \brief Perform a "matrix free" Lanczos iteration for real-valued double precision vectors.
///
void lanczos_iteration_d(const ct_long n, lanczos_linear_func_d afunc, const void* restrict adata, const double* restrict vstart, const int maxiter,
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
			for (ct_long i = 0; i < n; i++) {
				w[i] -= alpha[j] * v[j*n + i] + beta[j - 1] * v[(j - 1)*n + i];
			}
		}
		else {
			for (ct_long i = 0; i < n; i++) {
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
		for (ct_long i = 0; i < n; i++)
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
/// \brief Perform a "matrix free" Lanczos iteration for complex-valued single precision vectors.
///
void lanczos_iteration_c(const ct_long n, lanczos_linear_func_c afunc, const void* restrict adata, const scomplex* restrict vstart, const int maxiter,
	float* restrict alpha, float* restrict beta, scomplex* restrict v, int* restrict numiter)
{
	scomplex* w = ct_malloc(n * sizeof(scomplex));

	// set first "v" vector to normalized starting vector
	memcpy(v, vstart, n * sizeof(scomplex));
	float nrm = cblas_scnrm2(n, v, 1);
	assert(nrm > 0);
	cblas_csscal(n, 1/nrm, v, 1);

	for (int j = 0; j < maxiter - 1; j++)
	{
		// w' = A v_j
		afunc(n, adata, &v[j*n], w);

		// alpha_j = <w', v_j>
		scomplex t;
		cblas_cdotc_sub(n, w, 1, &v[j*n], 1, &t);
		alpha[j] = crealf(t);  // should be real for self-adjoint linear operation

		// w = w' - alpha_j v_j - beta_j v_{j-1}
		if (j > 0) {
			for (ct_long i = 0; i < n; i++) {
				w[i] -= alpha[j] * v[j*n + i] + beta[j - 1] * v[(j - 1)*n + i];
			}
		}
		else {
			for (ct_long i = 0; i < n; i++) {
				w[i] -= alpha[j] * v[j*n + i];
			}
		}

		// beta_{j+1} = ||w||
		beta[j] = cblas_scnrm2(n, w, 1);

		if (beta[j] < 100 * n * (float)FLT_EPSILON)
		{
			// premature end of iterations
			(*numiter) = j + 1;
			ct_free(w);
			return;
		}

		// v_{j+1} = w / beta[j+1]
		assert(beta[j] > 0);
		const float inv_beta = 1 / beta[j];
		for (ct_long i = 0; i < n; i++)
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
		scomplex t;
		cblas_cdotc_sub(n, w, 1, &v[j*n], 1, &t);
		alpha[j] = crealf(t);  // should be real for self-adjoint linear operation
	}

	ct_free(w);

	(*numiter) = maxiter;
}


//________________________________________________________________________________________________________________________
///
/// \brief Perform a "matrix free" Lanczos iteration for complex-valued double precision vectors.
///
void lanczos_iteration_z(const ct_long n, lanczos_linear_func_z afunc, const void* restrict adata, const dcomplex* restrict vstart, const int maxiter,
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
			for (ct_long i = 0; i < n; i++) {
				w[i] -= alpha[j] * v[j*n + i] + beta[j - 1] * v[(j - 1)*n + i];
			}
		}
		else {
			for (ct_long i = 0; i < n; i++) {
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
		for (ct_long i = 0; i < n; i++)
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
/// \brief Compute the Krylov subspace approximation of eigenvalues and vectors, real symmetric single precision version.
///
int eigensystem_krylov_symmetric_s(const ct_long n, lanczos_linear_func_s afunc, const void* restrict adata,
	const float* restrict vstart, const int maxiter, const int numeig,
	float* restrict lambda, float* restrict u_ritz)
{
	assert(numeig <= maxiter);

	#ifndef NDEBUG
	{
		// test symmetry of the matrix representation of 'afunc'

		float* h_dns_t = ct_malloc(n * n * sizeof(float));
		float* unitvec = ct_malloc(n * sizeof(float));
		for (ct_long j = 0; j < n; j++)
		{
			memset(unitvec, 0, n * sizeof(float));
			unitvec[j] = 1;
			afunc(n, adata, unitvec, &h_dns_t[j * n]);
		}
		ct_free(unitvec);

		const float tol = 1e-6;
		bool is_symm = true;
		float err = 0;
		for (ct_long i = 0; i < n; i++)
		{
			for (ct_long j = i + 1; j < n; j++)
			{
				const float err_loc = fabsf(h_dns_t[i*n + j] - h_dns_t[j*n + i]);
				err += err_loc;
				if (err_loc > tol) {
					is_symm = false;
				}
			}
		}
		ct_free(h_dns_t);

		if (!is_symm) {
			fprintf(stderr, "Warning: linear function 'afunc' in 'eigensystem_krylov_symmetric_s' is not symmetric; overall error: %g\n", err);
		}
	}
	#endif

	float* alpha = ct_malloc(maxiter       * sizeof(float));
	float* beta  = ct_malloc((maxiter - 1) * sizeof(float));
	float* v     = ct_malloc(maxiter*n     * sizeof(float));
	int numiter;
	lanczos_iteration_s(n, afunc, adata, vstart, maxiter, alpha, beta, v, &numiter);

	if (numiter < numeig) {
		fprintf(stderr, "Lanczos iteration stopped after %i iterations, cannot compute %i eigenvalues\n", numiter, numeig);
		return -1;
	}

	// diagonalize Hessenberg matrix
	float* u = ct_malloc(numiter*numiter * sizeof(float));
	float* work = ct_malloc(imax(1, 2*numiter - 2) * sizeof(float));
	lapack_int numiterlp = numiter;
	lapack_int info;
	LAPACK_ssteqr("I", &numiterlp, alpha, beta, u, &numiterlp, work, &info);
	ct_free(work);
	if (info != 0) {
		fprintf(stderr, "LAPACK function 'ssteqr' failed, return value: %i\n", info);
		return -2;
	}

	// 'alpha' now contains the eigenvalues
	memcpy(lambda, alpha, numeig * sizeof(float));

	// compute Ritz eigenvectors
	// using transposition of 'u' for implicit column- to row-major order conversion
	cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, n, numeig, numiter, 1.f, v, n, u, numiter, 0.f, u_ritz, numeig);

	// clean up
	ct_free(u);
	ct_free(v);
	ct_free(beta);
	ct_free(alpha);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the Krylov subspace approximation of eigenvalues and vectors, real symmetric double precision version.
///
int eigensystem_krylov_symmetric_d(const ct_long n, lanczos_linear_func_d afunc, const void* restrict adata,
	const double* restrict vstart, const int maxiter, const int numeig,
	double* restrict lambda, double* restrict u_ritz)
{
	assert(numeig <= maxiter);

	#ifndef NDEBUG
	{
		// test symmetry of the matrix representation of 'afunc'

		double* h_dns_t = ct_malloc(n * n * sizeof(double));
		double* unitvec = ct_malloc(n * sizeof(double));
		for (ct_long j = 0; j < n; j++)
		{
			memset(unitvec, 0, n * sizeof(double));
			unitvec[j] = 1;
			afunc(n, adata, unitvec, &h_dns_t[j * n]);
		}
		ct_free(unitvec);

		const double tol = 1e-12;
		bool is_symm = true;
		double err = 0;
		for (ct_long i = 0; i < n; i++)
		{
			for (ct_long j = i + 1; j < n; j++)
			{
				const double err_loc = fabs(h_dns_t[i*n + j] - h_dns_t[j*n + i]);
				err += err_loc;
				if (err_loc > tol) {
					is_symm = false;
				}
			}
		}
		ct_free(h_dns_t);

		if (!is_symm) {
			fprintf(stderr, "Warning: linear function 'afunc' in 'eigensystem_krylov_symmetric_d' is not symmetric; overall error: %g\n", err);
		}
	}
	#endif

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
	lapack_int numiterlp = numiter;
	lapack_int info;
	LAPACK_dsteqr("I", &numiterlp, alpha, beta, u, &numiterlp, work, &info);
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
/// \brief Compute the Krylov subspace approximation of eigenvalues and vectors, complex Hermitian single precision version.
///
int eigensystem_krylov_symmetric_c(const ct_long n, lanczos_linear_func_c afunc, const void* restrict adata,
	const scomplex* restrict vstart, const int maxiter, const int numeig,
	float* restrict lambda, scomplex* restrict u_ritz)
{
	assert(numeig <= maxiter);

	#ifndef NDEBUG
	{
		// test symmetry of the matrix representation of 'afunc'

		scomplex* h_dns_t = ct_malloc(n * n * sizeof(scomplex));
		scomplex* unitvec = ct_malloc(n * sizeof(scomplex));
		for (ct_long j = 0; j < n; j++)
		{
			memset(unitvec, 0, n * sizeof(scomplex));
			unitvec[j] = 1;
			afunc(n, adata, unitvec, &h_dns_t[j * n]);
		}
		ct_free(unitvec);

		const float tol = 1e-6;
		bool is_symm = true;
		float err = 0;
		for (ct_long i = 0; i < n; i++)
		{
			for (ct_long j = i; j < n; j++)  // starting at j == i since diagonal entries must be real
			{
				const float err_loc = cabsf(h_dns_t[i*n + j] - conjf(h_dns_t[j*n + i]));
				err += err_loc;
				if (err_loc > tol) {
					is_symm = false;
				}
			}
		}
		ct_free(h_dns_t);

		if (!is_symm) {
			fprintf(stderr, "Warning: linear function 'afunc' in 'eigensystem_krylov_symmetric_c' is not Hermitian; overall error: %g\n", err);
		}
	}
	#endif

	float* alpha = ct_malloc(maxiter       * sizeof(float));
	float* beta  = ct_malloc((maxiter - 1) * sizeof(float));
	scomplex* v  = ct_malloc(maxiter*n     * sizeof(scomplex));
	int numiter;
	lanczos_iteration_c(n, afunc, adata, vstart, maxiter, alpha, beta, v, &numiter);

	if (numiter < numeig) {
		fprintf(stderr, "Lanczos iteration stopped after %i iterations, cannot compute %i eigenvalues\n", numiter, numeig);
		return -1;
	}

	// diagonalize Hessenberg matrix
	float* u = ct_malloc(numiter*numiter * sizeof(float));
	float* work = ct_malloc(imax(1, 2*numiter - 2) * sizeof(float));
	lapack_int numiterlp = numiter;
	lapack_int info;
	LAPACK_ssteqr("I", &numiterlp, alpha, beta, u, &numiterlp, work, &info);
	ct_free(work);
	if (info != 0) {
		fprintf(stderr, "LAPACK function 'dsteqr' failed, return value: %i\n", info);
		return -2;
	}

	// 'alpha' now contains the eigenvalues
	memcpy(lambda, alpha, numeig * sizeof(float));

	// compute Ritz eigenvectors
	// require complex 'u' entries for matrix multiplication
	scomplex* uz = ct_malloc(numiter*numiter * sizeof(scomplex));
	for (int i = 0; i < numiter*numiter; i++) {
		uz[i] = (scomplex)u[i];
	}
	const scomplex one  = 1;
	const scomplex zero = 0;
	// using transposition of 'uz' for implicit column- to row-major order conversion
	cblas_cgemm(CblasRowMajor, CblasTrans, CblasTrans, n, numeig, numiter, &one, v, n, uz, numiter, &zero, u_ritz, numeig);

	// clean up
	ct_free(uz);
	ct_free(u);
	ct_free(v);
	ct_free(beta);
	ct_free(alpha);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the Krylov subspace approximation of eigenvalues and vectors, complex Hermitian double precision version.
///
int eigensystem_krylov_symmetric_z(const ct_long n, lanczos_linear_func_z afunc, const void* restrict adata,
	const dcomplex* restrict vstart, const int maxiter, const int numeig,
	double* restrict lambda, dcomplex* restrict u_ritz)
{
	assert(numeig <= maxiter);

	#ifndef NDEBUG
	{
		// test symmetry of the matrix representation of 'afunc'

		dcomplex* h_dns_t = ct_malloc(n * n * sizeof(dcomplex));
		dcomplex* unitvec = ct_malloc(n * sizeof(dcomplex));
		for (ct_long j = 0; j < n; j++)
		{
			memset(unitvec, 0, n * sizeof(dcomplex));
			unitvec[j] = 1;
			afunc(n, adata, unitvec, &h_dns_t[j * n]);
		}
		ct_free(unitvec);

		const double tol = 1e-12;
		bool is_symm = true;
		double err = 0;
		for (ct_long i = 0; i < n; i++)
		{
			for (ct_long j = i; j < n; j++)  // starting at j == i since diagonal entries must be real
			{
				const double err_loc = cabs(h_dns_t[i*n + j] - conj(h_dns_t[j*n + i]));
				err += err_loc;
				if (err_loc > tol) {
					is_symm = false;
				}
			}
		}
		ct_free(h_dns_t);

		if (!is_symm) {
			fprintf(stderr, "Warning: linear function 'afunc' in 'eigensystem_krylov_symmetric_z' is not Hermitian; overall error: %g\n", err);
		}
	}
	#endif

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
	lapack_int numiterlp = numiter;
	lapack_int info;
	LAPACK_dsteqr("I", &numiterlp, alpha, beta, u, &numiterlp, work, &info);
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
