#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "krylov.h"
#include "aligned_memory.h"
#include "util.h"


static void multiply_matrix_vector_d(const long n, const void* restrict data, const double* restrict v, double* restrict ret)
{
	const double* a = (double*)data;

	// perform matrix-vector multiplication
	cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1., a, n, v, 1, 0., ret, 1);
}


static void multiply_matrix_vector_z(const long n, const void* restrict data, const dcomplex* restrict v, dcomplex* restrict ret)
{
	const dcomplex* a = (dcomplex*)data;

	// perform matrix-vector multiplication
	const dcomplex one  = 1;
	const dcomplex zero = 0;
	cblas_zgemv(CblasRowMajor, CblasNoTrans, n, n, &one, a, n, v, 1, &zero, ret, 1);
}


char* test_lanczos_iteration_d()
{
	hid_t file = H5Fopen("../test/util/data/test_lanczos_iteration_d.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_lanczos_iteration_d failed";
	}

	// "large" matrix dimension
	const long n = 319;

	// maximum number of iterations
	const int maxiter = 24;

	// load 'a' matrix from disk
	double* a = ct_malloc(n*n * sizeof(double));
	if (read_hdf5_dataset(file, "a", H5T_NATIVE_DOUBLE, a) < 0) {
		return "reading matrix entries from disk failed";
	}

	// load starting vector from disk
	double* vstart = ct_malloc(n * sizeof(double));
	if (read_hdf5_dataset(file, "vstart", H5T_NATIVE_DOUBLE, vstart) < 0) {
		return "reading starting vector from disk failed";
	}

	double* alpha = ct_malloc(maxiter       * sizeof(double));
	double* beta  = ct_malloc((maxiter - 1) * sizeof(double));
	double* v     = ct_malloc(n*maxiter     * sizeof(double));

	// perform Lanczos iteration
	int numiter;
	lanczos_iteration_d(n, multiply_matrix_vector_d, a, vstart, maxiter, alpha, beta, v, &numiter);

	if (numiter != maxiter) {
		return "number of reported iterations does not match expected number";
	}

	// load reference data from disk
	double* alpha_ref = ct_malloc(maxiter       * sizeof(double));
	double* beta_ref  = ct_malloc((maxiter - 1) * sizeof(double));
	double* v_ref     = ct_malloc(maxiter*n     * sizeof(double));
	if (read_hdf5_dataset(file, "alpha", H5T_NATIVE_DOUBLE, alpha_ref) < 0) {
		return "reading 'alpha' vector from disk failed";
	}
	if (read_hdf5_dataset(file, "beta", H5T_NATIVE_DOUBLE, beta_ref) < 0) {
		return "reading 'beta' vector from disk failed";
	}
	if (read_hdf5_dataset(file, "v", H5T_NATIVE_DOUBLE, v_ref) < 0) {
		return "reading 'v' matrix from disk failed";
	}

	// compare
	if (uniform_distance(CT_DOUBLE_REAL, maxiter, alpha, alpha_ref) > 1e-13) {
		return "'alpha' vector does not match reference";
	}
	if (uniform_distance(CT_DOUBLE_REAL, maxiter - 1, beta, beta_ref) > 1e-13) {
		return "'beta' vector does not match reference";
	}
	if (uniform_distance(CT_DOUBLE_REAL, maxiter*n, v, v_ref) > 1e-13) {
		return "'v' matrix does not match reference";
	}

	// clean up
	ct_free(v_ref);
	ct_free(beta_ref);
	ct_free(alpha_ref);
	ct_free(v);
	ct_free(beta);
	ct_free(alpha);
	ct_free(vstart);
	ct_free(a);

	H5Fclose(file);

	return 0;
}


char* test_lanczos_iteration_z()
{
	hid_t file = H5Fopen("../test/util/data/test_lanczos_iteration_z.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_lanczos_iteration_z failed";
	}

	const hid_t hdf5_dcomplex_id = construct_hdf5_double_complex_dtype(false);

	// "large" matrix dimension
	const long n = 173;

	// maximum number of iterations
	const int maxiter = 24;

	// load 'a' matrix from disk
	dcomplex* a = ct_malloc(n*n * sizeof(dcomplex));
	if (read_hdf5_dataset(file, "a", hdf5_dcomplex_id, a) < 0) {
		return "reading matrix entries from disk failed";
	}

	// load starting vector from disk
	dcomplex* vstart = ct_malloc(n * sizeof(dcomplex));
	if (read_hdf5_dataset(file, "vstart", hdf5_dcomplex_id, vstart) < 0) {
		return "reading starting vector from disk failed";
	}

	double* alpha = ct_malloc(maxiter       * sizeof(double));
	double* beta  = ct_malloc((maxiter - 1) * sizeof(double));
	dcomplex* v   = ct_malloc(maxiter*n     * sizeof(dcomplex));

	// perform Lanczos iteration
	int numiter;
	lanczos_iteration_z(n, multiply_matrix_vector_z, a, vstart, maxiter, alpha, beta, v, &numiter);

	if (numiter != maxiter) {
		return "number of reported iterations does not match expected number";
	}

	// load reference data from disk
	double* alpha_ref = ct_malloc(maxiter       * sizeof(double));
	double* beta_ref  = ct_malloc((maxiter - 1) * sizeof(double));
	dcomplex* v_ref   = ct_malloc(maxiter*n     * sizeof(dcomplex));
	if (read_hdf5_dataset(file, "alpha", H5T_NATIVE_DOUBLE, alpha_ref) < 0) {
		return "reading 'alpha' vector from disk failed";
	}
	if (read_hdf5_dataset(file, "beta", H5T_NATIVE_DOUBLE, beta_ref) < 0) {
		return "reading 'beta' vector from disk failed";
	}
	if (read_hdf5_dataset(file, "v", hdf5_dcomplex_id, v_ref) < 0) {
		return "reading 'v' matrix from disk failed";
	}

	// compare
	if (uniform_distance(CT_DOUBLE_REAL, maxiter, alpha, alpha_ref) > 1e-13) {
		return "'alpha' vector does not match reference";
	}
	if (uniform_distance(CT_DOUBLE_REAL, maxiter - 1, beta, beta_ref) > 1e-13) {
		return "'beta' vector does not match reference";
	}
	if (uniform_distance(CT_DOUBLE_COMPLEX, maxiter*n, v, v_ref) > 1e-13) {
		return "'v' matrix does not match reference";
	}

	// clean up
	ct_free(v_ref);
	ct_free(beta_ref);
	ct_free(alpha_ref);
	ct_free(v);
	ct_free(beta);
	ct_free(alpha);
	ct_free(vstart);
	ct_free(a);

	H5Tclose(hdf5_dcomplex_id);
	H5Fclose(file);

	return 0;
}


char* test_eigensystem_krylov_symmetric()
{
	hid_t file = H5Fopen("../test/util/data/test_eigensystem_krylov_symmetric.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_eigensystem_krylov_symmetric failed";
	}

	// "large" matrix dimension
	const long n = 197;

	// maximum number of iterations
	const int maxiter = 35;

	// number of eigenvalues and -vectors
	const int numeig = 5;

	// load 'a' matrix from disk
	double* a = ct_malloc(n*n * sizeof(double));
	if (read_hdf5_dataset(file, "a", H5T_NATIVE_DOUBLE, a) < 0) {
		return "reading matrix entries from disk failed";
	}

	// load starting vector from disk
	double* vstart = ct_malloc(n * sizeof(double));
	if (read_hdf5_dataset(file, "vstart", H5T_NATIVE_DOUBLE, vstart) < 0) {
		return "reading starting vector from disk failed";
	}

	double* lambda = ct_malloc(numeig   * sizeof(double));
	double* u_ritz = ct_malloc(n*numeig * sizeof(double));
	int ret = eigensystem_krylov_symmetric(n, multiply_matrix_vector_d, a, vstart, maxiter, numeig, lambda, u_ritz);
	if (ret < 0) {
		return "'eigensystem_krylov_symmetric' failed internally";
	}

	// load reference data from disk
	double* lambda_ref = ct_malloc(numeig   * sizeof(double));
	double* u_ritz_ref = ct_malloc(n*numeig * sizeof(double));
	if (read_hdf5_dataset(file, "lambda", H5T_NATIVE_DOUBLE, lambda_ref) < 0) {
		return "reading Ritz eigenvalues from disk failed";
	}
	if (read_hdf5_dataset(file, "u_ritz", H5T_NATIVE_DOUBLE, u_ritz_ref) < 0) {
		return "reading Ritz eigenvectors from disk failed";
	}

	// compare
	if (uniform_distance(CT_DOUBLE_REAL, numeig, lambda, lambda_ref) > 1e-13) {
		return "Ritz eigenvalues do not match reference";
	}
	// Ritz eigenvectors can differ by sign factors
	double d = 0;
	for (int i = 0; i < numeig; i++)
	{
		double dv0 = 0;
		for (long j = 0; j < n; j++)
		{
			dv0 = fmax(dv0, fabs(u_ritz[j*numeig + i] - u_ritz_ref[j*numeig + i]));
		}
		// include (-1) factor
		double dv1 = 0;
		for (long j = 0; j < n; j++)
		{
			dv1 = fmax(dv1, fabs(u_ritz[j*numeig + i] + u_ritz_ref[j*numeig + i]));
		}
		d = fmax(d, fmin(dv0, dv1));
	}
	if (d > 1e-12) {
		return "Ritz eigenvectors do not match reference";
	}

	ct_free(u_ritz_ref);
	ct_free(lambda_ref);
	ct_free(u_ritz);
	ct_free(lambda);
	ct_free(vstart);
	ct_free(a);

	H5Fclose(file);

	return 0;
}


char* test_eigensystem_krylov_hermitian()
{
	hid_t file = H5Fopen("../test/util/data/test_eigensystem_krylov_hermitian.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_eigensystem_krylov_hermitian failed";
	}

	const hid_t hdf5_dcomplex_id = construct_hdf5_double_complex_dtype(false);

	// "large" matrix dimension
	const long n = 185;

	// maximum number of iterations
	const int maxiter = 37;

	// number of eigenvalues and -vectors
	const int numeig = 6;

	// load 'a' matrix from disk
	dcomplex* a = ct_malloc(n*n * sizeof(dcomplex));
	if (read_hdf5_dataset(file, "a", hdf5_dcomplex_id, a) < 0) {
		return "reading matrix entries from disk failed";
	}

	// load starting vector from disk
	dcomplex* vstart = ct_malloc(n * sizeof(dcomplex));
	if (read_hdf5_dataset(file, "vstart", hdf5_dcomplex_id, vstart) < 0) {
		return "reading starting vector from disk failed";
	}

	double* lambda   = ct_malloc(numeig   * sizeof(double));
	dcomplex* u_ritz = ct_malloc(n*numeig * sizeof(dcomplex));
	int ret = eigensystem_krylov_hermitian(n, multiply_matrix_vector_z, a, vstart, maxiter, numeig, lambda, u_ritz);
	if (ret < 0) {
		return "'eigensystem_krylov_hermitian' failed internally";
	}

	// load reference data from disk
	double*   lambda_ref = ct_malloc(numeig   * sizeof(double));
	dcomplex* u_ritz_ref = ct_malloc(n*numeig * sizeof(dcomplex));
	if (read_hdf5_dataset(file, "lambda", H5T_NATIVE_DOUBLE, lambda_ref) < 0) {
		return "reading Ritz eigenvalues from disk failed";
	}
	if (read_hdf5_dataset(file, "u_ritz", hdf5_dcomplex_id, u_ritz_ref) < 0) {
		return "reading Ritz eigenvectors from disk failed";
	}

	// compare
	if (uniform_distance(CT_DOUBLE_REAL, numeig, lambda, lambda_ref) > 1e-13) {
		return "Ritz eigenvalues do not match reference";
	}
	// Ritz eigenvectors can differ by phase factors
	double d = 0;
	for (int i = 0; i < numeig; i++)
	{
		double dv0 = 0;
		for (long j = 0; j < n; j++)
		{
			dv0 = fmax(dv0, cabs(u_ritz[j*numeig + i] - u_ritz_ref[j*numeig + i]));
		}
		// include (-1) factor
		double dv1 = 0;
		for (long j = 0; j < n; j++)
		{
			dv1 = fmax(dv1, cabs(u_ritz[j*numeig + i] + u_ritz_ref[j*numeig + i]));
		}
		d = fmax(d, fmin(dv0, dv1));
	}
	if (d > 1e-12) {
		return "Ritz eigenvectors do not match reference";
	}

	ct_free(u_ritz_ref);
	ct_free(lambda_ref);
	ct_free(u_ritz);
	ct_free(lambda);
	ct_free(vstart);
	ct_free(a);

	H5Tclose(hdf5_dcomplex_id);
	H5Fclose(file);

	return 0;
}
