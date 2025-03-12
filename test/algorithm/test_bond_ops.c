#include <math.h>
#include "bond_ops.h"
#include "util.h"
#include "aligned_memory.h"


char* test_retained_bond_indices()
{
	hid_t file = H5Fopen("../test/algorithm/data/test_retained_bond_indices.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_retained_bond_indices failed";
	}

	// singular values
	hsize_t sigma_dims[1];
	if (get_hdf5_dataset_dims(file, "sigma", sigma_dims)) {
		return "reading number of singular values from disk failed";
	}
	const long n = sigma_dims[0];
	double* sigma = ct_malloc(n * sizeof(double));
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
	long* ind_ref = ct_malloc(ind_ref_dims[0] * sizeof(long));
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

	ct_free(ind_ref);
	ct_free(sigma);

	H5Fclose(file);

	return 0;
}


char* test_split_block_sparse_matrix_svd()
{
	hid_t file = H5Fopen("../test/algorithm/data/test_split_block_sparse_matrix_svd.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_split_block_sparse_matrix_svd failed";
	}

	const hid_t hdf5_scomplex_id = construct_hdf5_single_complex_dtype(false);

	const long dim[2] = { 181, 191 };
	const long max_vdim = 200;

	// read dense tensor from disk
	struct dense_tensor a_dns;
	allocate_dense_tensor(CT_SINGLE_COMPLEX, 2, dim, &a_dns);
	if (read_hdf5_dataset(file, "a", hdf5_scomplex_id, a_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction axis_dir[2];
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber* qnums[2];
	for (int i = 0; i < 2; i++)
	{
		qnums[i] = ct_malloc(dim[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qnums%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qnums[i]) < 0) {
			return "reading quantum numbers from disk failed";
		}
	}

	// convert dense to block-sparse tensor
	struct block_sparse_tensor a;
	dense_to_block_sparse_tensor(&a_dns, axis_dir, (const qnumber**)qnums, &a);

	double tol;
	if (read_hdf5_attribute(file, "tol", H5T_NATIVE_DOUBLE, &tol) < 0) {
		return "reading tolerance from disk failed";
	}

	long num_retained_ref;
	if (read_hdf5_attribute(file, "num_retained", H5T_NATIVE_LONG, &num_retained_ref) < 0) {
		return "reading number of retained singular values from disk failed";
	}

	// reference reassembled matrix after splitting
	// without renormalization
	struct dense_tensor a_trunc_plain_ref;
	allocate_dense_tensor(CT_SINGLE_COMPLEX, 2, dim, &a_trunc_plain_ref);
	if (read_hdf5_dataset(file, "a_trunc_plain", hdf5_scomplex_id, a_trunc_plain_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	// with renormalization
	struct dense_tensor a_trunc_renrm_ref;
	allocate_dense_tensor(CT_SINGLE_COMPLEX, 2, dim, &a_trunc_renrm_ref);
	if (read_hdf5_dataset(file, "a_trunc_renrm", hdf5_scomplex_id, a_trunc_renrm_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// renormalization
	for (int r = 0; r < 2; r++)
	{
		// singular value distribution mode
		for (int d = 0; d < 2; d++)
		{
			const enum singular_value_distr svd_distr = (d == 0 ? SVD_DISTR_LEFT : SVD_DISTR_RIGHT);

			struct block_sparse_tensor a0, a1;
			struct trunc_info info;
			if (split_block_sparse_matrix_svd(&a, tol, max_vdim, r == 0 ? false : true, svd_distr, &a0, &a1, &info) < 0) {
				return "'split_block_sparse_matrix_svd' failed internally";
			}

			if (a0.dim_logical[1] != num_retained_ref || a1.dim_logical[0] != num_retained_ref) {
				return "number of retained singular values does not match reference";
			}

			if (d == 0)
			{
				// a1 must be an isometry
				if (!block_sparse_tensor_is_isometry(&a1, 5e-6, true)) {
					return "'a1' matrix is not an isometry";
				}
			}
			else
			{
				// a0 must be an isometry
				if (!block_sparse_tensor_is_isometry(&a0, 5e-6, false)) {
					return "'a0' matrix is not an isometry";
				}
			}

			// reassemble matrix after splitting, for comparison with reference
			struct block_sparse_tensor a_trunc;
			block_sparse_tensor_dot(&a0, TENSOR_AXIS_RANGE_TRAILING, &a1, TENSOR_AXIS_RANGE_LEADING, 1, &a_trunc);
			struct dense_tensor a_trunc_dns;
			block_sparse_to_dense_tensor(&a_trunc, &a_trunc_dns);
			// compare
			if (!dense_tensor_allclose(&a_trunc_dns, r == 0 ? &a_trunc_plain_ref : &a_trunc_renrm_ref, 2e-6)) {
				return "merge matrix after truncation does not match reference";
			}

			delete_dense_tensor(&a_trunc_dns);
			delete_block_sparse_tensor(&a_trunc);
			delete_block_sparse_tensor(&a0);
			delete_block_sparse_tensor(&a1);
		}
	}

	delete_dense_tensor(&a_trunc_renrm_ref);
	delete_dense_tensor(&a_trunc_plain_ref);
	delete_block_sparse_tensor(&a);
	for (int i = 0; i < 2; i++) {
		ct_free(qnums[i]);
	}
	delete_dense_tensor(&a_dns);

	H5Tclose(hdf5_scomplex_id);
	H5Fclose(file);

	return 0;
}


char* test_split_block_sparse_matrix_svd_zero()
{
	const long dim[2] = { 7, 5 };
	const long max_vdim = 10;

	// incompatible axis direction and quantum number combination
	enum tensor_axis_direction axis_dir[2] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT };
	const qnumber qnums0[7] = { 1, 3, 2, 2, 1, 2, 3 };
	const qnumber qnums1[5] = { 0, 2, 1, 0, 1 };
	const qnumber* qnums[2] = { qnums0, qnums1 };

	struct block_sparse_tensor a;
	allocate_block_sparse_tensor(CT_SINGLE_COMPLEX, 2, dim, axis_dir, qnums, &a);
	if (block_sparse_tensor_norm2(&a) != 0) {
		return "block-sparse matrix should be logically zero";
	}

	const double tol = 0.1;

	// singular value distribution mode
	for (int d = 0; d < 2; d++)
	{
		const enum singular_value_distr svd_distr = (d == 0 ? SVD_DISTR_LEFT : SVD_DISTR_RIGHT);

		struct block_sparse_tensor a0, a1;
		struct trunc_info info;
		int ret = split_block_sparse_matrix_svd(&a, tol, max_vdim, true, svd_distr, &a0, &a1, &info);
		if (ret < 0) {
			return "'split_block_sparse_matrix_svd' failed internally";
		}

		// dummy bond
		if (a0.dim_logical[1] != 1 || a1.dim_logical[0] != 1) {
			return "number of retained singular values does not match reference";
		}

		if (info.norm_sigma != 0) {
			return "norm of singular values should be zero";
		}

		if (block_sparse_tensor_norm2(&a1) != 0) {
			return "expecting a zero block-sparse matrix";
		}

		// a1 cannot be an isometry for SVD_DISTR_LEFT due to quantum number incompatibility

		if (d == 1)
		{
			// a0 must be an isometry
			if (!block_sparse_tensor_is_isometry(&a0, 5e-6, false)) {
				return "'a0' matrix is not an isometry";
			}
		}

		// reassemble matrix after splitting, to ensure dimension compatibilty
		struct block_sparse_tensor a_trunc;
		block_sparse_tensor_dot(&a0, TENSOR_AXIS_RANGE_TRAILING, &a1, TENSOR_AXIS_RANGE_LEADING, 1, &a_trunc);
		if (block_sparse_tensor_norm2(&a_trunc) != 0) {
			return "truncated matrix should be logically zero";
		}

		delete_block_sparse_tensor(&a_trunc);
		delete_block_sparse_tensor(&a0);
		delete_block_sparse_tensor(&a1);
	}

	delete_block_sparse_tensor(&a);

	return 0;
}
