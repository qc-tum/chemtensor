#include "thc.h"


char* test_thc_spin_molecular_hamiltonian_to_matrix()
{
	hid_t file = H5Fopen("../test/algorithm/data/test_thc_spin_molecular_hamiltonian_to_matrix.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_thc_spin_molecular_hamiltonian_to_matrix failed";
	}

	// number of spin-endowed lattice sites
	const int nsites = 4;
	// THC rank
	const int thc_rank = 11;

	// read coefficient matrices from disk
	struct dense_tensor tkin;
	const long dim_tkin[2] = { nsites, nsites };
	allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim_tkin, &tkin);
	if (read_hdf5_dataset(file, "tkin", H5T_NATIVE_DOUBLE, tkin.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	struct dense_tensor thc_kernel;
	const long dim_thc_kernel[2] = { thc_rank, thc_rank };
	allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim_thc_kernel, &thc_kernel);
	if (read_hdf5_dataset(file, "thc_kernel", H5T_NATIVE_DOUBLE, thc_kernel.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	struct dense_tensor thc_transform;
	const long dim_thc_transform[2] = { nsites, thc_rank };
	allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim_thc_transform, &thc_transform);
	if (read_hdf5_dataset(file, "thc_transform", H5T_NATIVE_DOUBLE, thc_transform.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// construct Hamiltonian
	struct thc_spin_molecular_hamiltonian hamiltonian;
	if (construct_thc_spin_molecular_hamiltonian(&tkin, &thc_kernel, &thc_transform, &hamiltonian) < 0) {
		return "'construct_thc_spin_molecular_hamiltonian' failed internally";
	}

	// convert to a sparse matrix
	struct block_sparse_tensor hmat;
	if (thc_spin_molecular_hamiltonian_to_matrix(&hamiltonian, &hmat) < 0) {
		return "'thc_spin_molecular_hamiltonian_to_matrix' failed internally";
	}

	// convert to dense matrix for comparison with reference
	struct dense_tensor hmat_dns;
	block_sparse_to_dense_tensor(&hmat, &hmat_dns);

	// reference matrix for checking
	hsize_t dim_ref_hsize[2];
	if (get_hdf5_dataset_dims(file, "hmat", dim_ref_hsize) < 0) {
		return "obtaining dimensions of reference Hamiltonian failed";
	}
	// include dummy virtual bond dimensions
	const long dim_ref[4] = { 1, dim_ref_hsize[0], dim_ref_hsize[1], 1 };
	struct dense_tensor hmat_ref;
	allocate_dense_tensor(CT_DOUBLE_REAL, 4, dim_ref, &hmat_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "hmat", H5T_NATIVE_DOUBLE, hmat_ref.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&hmat_dns, &hmat_ref, 1e-13)) {
		return "matrix representation of tensor hypercontraction representation of a molecular Hamiltonian does not match reference";
	}

	delete_dense_tensor(&hmat_ref);
	delete_dense_tensor(&hmat_dns);
	delete_block_sparse_tensor(&hmat);
	delete_thc_spin_molecular_hamiltonian(&hamiltonian);
	delete_dense_tensor(&thc_transform);
	delete_dense_tensor(&thc_kernel);
	delete_dense_tensor(&tkin);

	H5Fclose(file);

	return 0;
}
