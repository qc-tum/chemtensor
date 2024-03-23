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
	hid_t file = H5Fopen("../test/data/test_lanczos_iteration_d.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_lanczos_iteration_d failed";
	}

	// "large" matrix dimension
	const long n = 319;

	// maximum number of iterations
	const int maxiter = 24;

	// load 'a' matrix from disk
	double* a = aligned_alloc(MEM_DATA_ALIGN, n*n * sizeof(double));
	if (read_hdf5_dataset(file, "a", H5T_NATIVE_DOUBLE, a) < 0) {
		return "reading matrix entries from disk failed";
	}

	// load starting vector from disk
	double* vstart = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(double));
	if (read_hdf5_dataset(file, "vstart", H5T_NATIVE_DOUBLE, vstart) < 0) {
		return "reading starting vector from disk failed";
	}

	double* alpha = aligned_alloc(MEM_DATA_ALIGN, maxiter       * sizeof(double));
	double* beta  = aligned_alloc(MEM_DATA_ALIGN, (maxiter - 1) * sizeof(double));
	double* v     = aligned_alloc(MEM_DATA_ALIGN, n*maxiter     * sizeof(double));

	// perform Lanczos iteration
	int numiter;
	lanczos_iteration_d(n, multiply_matrix_vector_d, a, vstart, maxiter, alpha, beta, v, &numiter);

	if (numiter != maxiter) {
		return "number of reported iterations does not match expected number";
	}

	// load reference data from disk
	double* alpha_ref = aligned_alloc(MEM_DATA_ALIGN, maxiter       * sizeof(double));
	double* beta_ref  = aligned_alloc(MEM_DATA_ALIGN, (maxiter - 1) * sizeof(double));
	double* v_ref     = aligned_alloc(MEM_DATA_ALIGN, maxiter*n     * sizeof(double));
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
	if (uniform_distance(DOUBLE_REAL, maxiter, alpha, alpha_ref) > 1e-13) {
		return "'alpha' vector does not match reference";
	}
	if (uniform_distance(DOUBLE_REAL, maxiter - 1, beta, beta_ref) > 1e-13) {
		return "'beta' vector does not match reference";
	}
	if (uniform_distance(DOUBLE_REAL, maxiter*n, v, v_ref) > 1e-13) {
		return "'v' matrix does not match reference";
	}

	// clean up
	aligned_free(v_ref);
	aligned_free(beta_ref);
	aligned_free(alpha_ref);
	aligned_free(v);
	aligned_free(beta);
	aligned_free(alpha);
	aligned_free(vstart);
	aligned_free(a);

	H5Fclose(file);

	return 0;
}


char* test_lanczos_iteration_z()
{
	hid_t file = H5Fopen("../test/data/test_lanczos_iteration_z.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_lanczos_iteration_z failed";
	}

	// "large" matrix dimension
	const long n = 173;

	// maximum number of iterations
	const int maxiter = 24;

	// load 'a' matrix from disk
	dcomplex* a = aligned_alloc(MEM_DATA_ALIGN, n*n * sizeof(dcomplex));
	if (read_hdf5_dataset(file, "a", H5T_NATIVE_DOUBLE, a) < 0) {
		return "reading matrix entries from disk failed";
	}

	// load starting vector from disk
	dcomplex* vstart = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(dcomplex));
	if (read_hdf5_dataset(file, "vstart", H5T_NATIVE_DOUBLE, vstart) < 0) {
		return "reading starting vector from disk failed";
	}

	double* alpha = aligned_alloc(MEM_DATA_ALIGN, maxiter       * sizeof(double));
	double* beta  = aligned_alloc(MEM_DATA_ALIGN, (maxiter - 1) * sizeof(double));
	dcomplex* v   = aligned_alloc(MEM_DATA_ALIGN, n*maxiter     * sizeof(dcomplex));

	// perform Lanczos iteration
	int numiter;
	lanczos_iteration_z(n, multiply_matrix_vector_z, a, vstart, maxiter, alpha, beta, v, &numiter);

	if (numiter != maxiter) {
		return "number of reported iterations does not match expected number";
	}

	// load reference data from disk
	double* alpha_ref = aligned_alloc(MEM_DATA_ALIGN, maxiter       * sizeof(double));
	double* beta_ref  = aligned_alloc(MEM_DATA_ALIGN, (maxiter - 1) * sizeof(double));
	dcomplex* v_ref   = aligned_alloc(MEM_DATA_ALIGN, maxiter*n     * sizeof(dcomplex));
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
	if (uniform_distance(DOUBLE_REAL, maxiter, alpha, alpha_ref) > 1e-13) {
		return "'alpha' vector does not match reference";
	}
	if (uniform_distance(DOUBLE_REAL, maxiter - 1, beta, beta_ref) > 1e-13) {
		return "'beta' vector does not match reference";
	}
	if (uniform_distance(DOUBLE_COMPLEX, maxiter*n, v, v_ref) > 1e-13) {
		return "'v' matrix does not match reference";
	}

	// clean up
	aligned_free(v_ref);
	aligned_free(beta_ref);
	aligned_free(alpha_ref);
	aligned_free(v);
	aligned_free(beta);
	aligned_free(alpha);
	aligned_free(vstart);
	aligned_free(a);

	H5Fclose(file);

	return 0;
}
