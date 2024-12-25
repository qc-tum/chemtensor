#include "thc.h"
#include "aligned_memory.h"


char* test_apply_thc_spin_molecular_hamiltonian()
{
	hid_t file = H5Fopen("../test/algorithm/data/test_apply_thc_spin_molecular_hamiltonian.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_thc_spin_molecular_hamiltonian failed";
	}

	// number of spin-endowed lattice sites
	const int nsites = 5;
	// THC rank
	const int thc_rank = 7;

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

	// input statevector as MPS

	// physical particle number and spin quantum numbers (encoded as single integer)
	const qnumber qn[4] = { 0,  1,  1,  2 };
	const qnumber qs[4] = { 0, -1,  1,  0 };
	const qnumber qsite[4] = {
		encode_quantum_number_pair(qn[0], qs[0]),
		encode_quantum_number_pair(qn[1], qs[1]),
		encode_quantum_number_pair(qn[2], qs[2]),
		encode_quantum_number_pair(qn[3], qs[3]),
	};

	// virtual bond quantum numbers
	const long psi_dim_bonds[6] = { 1, 19, 39, 41, 23, 1 };
	qnumber** psi_qbonds = ct_malloc((nsites + 1) * sizeof(qnumber*));
	for (int i = 0; i < nsites + 1; i++)
	{
		psi_qbonds[i] = ct_malloc(psi_dim_bonds[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "psi_qbond%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, psi_qbonds[i]) < 0) {
			return "reading virtual bond quantum numbers from disk failed";
		}
	}

	struct mps psi;
	allocate_mps(CT_DOUBLE_REAL, nsites, 4, qsite, psi_dim_bonds, (const qnumber**)psi_qbonds, &psi);

	// read MPS tensors from disk
	for (int i = 0; i < nsites; i++)
	{
		// read dense tensors from disk
		struct dense_tensor a_dns;
		allocate_dense_tensor(psi.a[i].dtype, psi.a[i].ndim, psi.a[i].dim_logical, &a_dns);
		char varname[1024];
		sprintf(varname, "psi_a%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, a_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		dense_to_block_sparse_tensor_entries(&a_dns, &psi.a[i]);

		delete_dense_tensor(&a_dns);
	}

	if (!mps_is_consistent(&psi)) {
		return "internal MPS consistency check failed";
	}

	// reference vector
	// include dummy virtual bond dimensions
	const long dim_h_psi_ref[3] = { 1, ipow(4, nsites), 1 };
	struct dense_tensor h_psi_ref;
	allocate_dense_tensor(CT_DOUBLE_REAL, 3, dim_h_psi_ref, &h_psi_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "h_psi", H5T_NATIVE_DOUBLE, h_psi_ref.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	for (int j = 0; j < 2; j++)  // determines tolerance
	{
		const double tol = (j == 0 ? 0 : 1e-3);
		const long max_vdim = 1024;
		struct mps h_psi;
		if (apply_thc_spin_molecular_hamiltonian(&hamiltonian, &psi, tol, max_vdim, &h_psi) < 0) {
			return "'apply_thc_spin_molecular_hamiltonian' failed internally";
		}

		struct block_sparse_tensor h_psi_vec;
		mps_to_statevector(&h_psi, &h_psi_vec);

		struct dense_tensor h_psi_vec_dns;
		block_sparse_to_dense_tensor(&h_psi_vec, &h_psi_vec_dns);

		// compare vectors
		if (!dense_tensor_allclose(&h_psi_vec_dns, &h_psi_ref, j == 0 ? 1e-13 : 0.005)) {
			return "applying a THC spin molecular Hamiltonian does not match reference";
		}

		delete_dense_tensor(&h_psi_vec_dns);
		delete_block_sparse_tensor(&h_psi_vec);
		delete_mps(&h_psi);
	}

	delete_dense_tensor(&h_psi_ref);
	delete_mps(&psi);
	for (int i = 0; i < nsites + 1; i++) {
		ct_free(psi_qbonds[i]);
	}
	ct_free(psi_qbonds);
	delete_thc_spin_molecular_hamiltonian(&hamiltonian);
	delete_dense_tensor(&thc_transform);
	delete_dense_tensor(&thc_kernel);
	delete_dense_tensor(&tkin);

	H5Fclose(file);

	return 0;
}


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
