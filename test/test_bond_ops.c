#include <math.h>
#include "bond_ops.h"
#include "util.h"
#include "aligned_memory.h"


char* test_retained_bond_indices()
{
	hid_t file = H5Fopen("../test/data/test_retained_bond_indices.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_retained_bond_indices failed";
	}

	// singular values
	hsize_t sigma_dims[1];
	if (get_hdf5_dataset_dims(file, "sigma", sigma_dims)) {
		return "reading number of singular values from disk failed";
	}
	const long n = sigma_dims[0];
	double* sigma = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(double));
	if (read_hdf5_dataset(file, "sigma", H5T_NATIVE_DOUBLE, sigma) < 0) {
		return "reading singular values from disk failed";
	}

	// truncation tolerance
	double tol;
	if (read_hdf5_attribute(file, "tol", H5T_NATIVE_DOUBLE, &tol) < 0) {
		return "reading truncation tolerance from disk failed";
	}

	// load reference data from disk
	hsize_t ind_ref_dims[1];
	if (get_hdf5_dataset_dims(file, "ind", ind_ref_dims)) {
		return "reading number of reference indices from disk failed";
	}
	long* ind_ref = aligned_alloc(MEM_DATA_ALIGN, ind_ref_dims[0] * sizeof(long));
	if (read_hdf5_dataset(file, "ind", H5T_NATIVE_LONG, ind_ref) < 0) {
		return "reading reference indices from disk failed";
	}
	double norm_sigma_ref;
	if (read_hdf5_dataset(file, "norm_sigma", H5T_NATIVE_DOUBLE, &norm_sigma_ref) < 0) {
		return "reading reference norm of singular values from disk failed";
	}
	double entropy_ref;
	if (read_hdf5_dataset(file, "entropy", H5T_NATIVE_DOUBLE, &entropy_ref) < 0) {
		return "reading reference entropy from disk failed";
	}

	// two versions: truncation determined by tolerance, and truncation determined by length cut-off
	for (int i = 0; i < 2; i++)
	{
		struct index_list list;
		struct trunc_info info;
		retained_bond_indices(sigma, n, i == 0 ? tol : 1e-5, i == 0 ? n : (long)ind_ref_dims[0], &list, &info);

		// compare indices
		if (list.num != (long)ind_ref_dims[0]) {
			return "number of retained singular values does not match reference";
		}
		for (int i = 0; i < list.num; i++) {
			if (list.ind[i] != ind_ref[i]) {
				return "indices of retained singular values do not match reference";
			}
		}
		// compare norm of retained singular values
		if (fabs(info.norm_sigma - norm_sigma_ref) > 1e-13) {
			return "norm of retained singular values does not match reference";
		}
		// compare von Neumann entropy
		if (fabs(info.entropy - entropy_ref) > 1e-13) {
			return "entropy of retained singular values does not match reference";
		}

		delete_index_list(&list);
	}

	aligned_free(ind_ref);
	aligned_free(sigma);

	H5Fclose(file);

	return 0;
}
