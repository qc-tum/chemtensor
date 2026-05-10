#define _USE_MATH_DEFINES
#include <math.h>
#include "wigner.h"
#include "dense_tensor.h"
#include "hdf5_util.h"


char* test_wigner_small_d()
{
	hid_t file = H5Fopen("../test/util/data/test_wigner_small_d.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_wigner_small_d failed";
	}

	// rotation angle
	double theta;
	if (read_hdf5_attribute(file, "theta", H5T_NATIVE_DOUBLE, &theta) < 0) {
		return "reading 'theta' angle from disk failed";
	}

	for (qnumber j = 0; j <= 6; j++)
	{
		struct dense_tensor w;
		const ct_long dim[2] = { j + 1, j + 1 };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &w);

		wigner_small_d(j, theta, w.data);

		// Wigner d-matrices must be orthogonal
		if (!dense_tensor_is_isometry(&w, 1e-14, false)) {
			return "Wigner \"small\" d-matrix is not orthogonal";
		}

		struct dense_tensor w_ref;
		{
			allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &w_ref);
			char varname[1024];
			sprintf(varname, "w%i", j);
			if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, w_ref.data) < 0) {
				return "reading Wigner \"small\" d-matrix entries from disk failed";
			}
		}

		// compare
		if (!dense_tensor_allclose(&w, &w_ref, 1e-13)) {
			return "Wigner \"small\" d-matrix does not match reference";
		}

		delete_dense_tensor(&w_ref);
		delete_dense_tensor(&w);
	}

	H5Fclose(file);

	return 0;
}


char* test_wigner_d()
{
	hid_t file = H5Fopen("../test/util/data/test_wigner_d.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_wigner_d failed";
	}

	const hid_t hdf5_dcomplex_id = construct_hdf5_double_complex_dtype(false);

	// rotation angles
	double psi;
	if (read_hdf5_attribute(file, "psi", H5T_NATIVE_DOUBLE, &psi) < 0) {
		return "reading 'psi' angle from disk failed";
	}
	double theta;
	if (read_hdf5_attribute(file, "theta", H5T_NATIVE_DOUBLE, &theta) < 0) {
		return "reading 'theta' angle from disk failed";
	}
	double phi;
	if (read_hdf5_attribute(file, "phi", H5T_NATIVE_DOUBLE, &phi) < 0) {
		return "reading 'phi' angle from disk failed";
	}

	for (qnumber j = 0; j <= 5; j++)
	{
		struct dense_tensor w;
		const ct_long dim[2] = { j + 1, j + 1 };
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 2, dim, &w);

		wigner_d(j, psi, theta, phi, w.data);

		// Wigner D-matrices must be unitary
		if (!dense_tensor_is_isometry(&w, 1e-14, false)) {
			return "Wigner D-matrix is not unitary";
		}

		struct dense_tensor w_ref;
		{
			allocate_dense_tensor(CT_DOUBLE_COMPLEX, 2, dim, &w_ref);
			char varname[1024];
			sprintf(varname, "w%i", j);
			if (read_hdf5_dataset(file, varname, hdf5_dcomplex_id, w_ref.data) < 0) {
				return "reading Wigner D-matrix entries from disk failed";
			}
		}

		// compare
		if (!dense_tensor_allclose(&w, &w_ref, 1e-13)) {
			return "Wigner D-matrix does not match reference";
		}

		delete_dense_tensor(&w_ref);
		delete_dense_tensor(&w);
	}

	H5Tclose(hdf5_dcomplex_id);
	H5Fclose(file);

	return 0;
}


char* test_real_wigner_d()
{
	hid_t file = H5Fopen("../test/util/data/test_real_wigner_d.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_real_wigner_d failed";
	}

	// rotation angles
	double psi;
	if (read_hdf5_attribute(file, "psi", H5T_NATIVE_DOUBLE, &psi) < 0) {
		return "reading 'psi' angle from disk failed";
	}
	double theta;
	if (read_hdf5_attribute(file, "theta", H5T_NATIVE_DOUBLE, &theta) < 0) {
		return "reading 'theta' angle from disk failed";
	}
	double phi;
	if (read_hdf5_attribute(file, "phi", H5T_NATIVE_DOUBLE, &phi) < 0) {
		return "reading 'phi' angle from disk failed";
	}

	for (qnumber j = 0; j <= 4; j += 2)
	{
		struct dense_tensor w;
		const ct_long dim[2] = { j + 1, j + 1 };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &w);

		real_wigner_d(j, psi, theta, phi, w.data);

		// real-valued Wigner D-matrices must be orthogonal
		if (!dense_tensor_is_isometry(&w, 1e-14, false)) {
			return "real-valued Wigner D-matrix is not orthogonal";
		}

		struct dense_tensor w_ref;
		{
			allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &w_ref);
			char varname[1024];
			sprintf(varname, "w%i", j);
			if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, w_ref.data) < 0) {
				return "reading real-valued Wigner D-matrix entries from disk failed";
			}
		}

		// compare
		if (!dense_tensor_allclose(&w, &w_ref, 1e-13)) {
			return "real-valued Wigner D-matrix does not match reference";
		}

		delete_dense_tensor(&w_ref);
		delete_dense_tensor(&w);
	}

	H5Fclose(file);

	return 0;
}
