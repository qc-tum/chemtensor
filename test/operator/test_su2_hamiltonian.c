#include "su2_hamiltonian.h"
#include "hdf5_util.h"


char* test_heisenberg_1d_su2_mpo()
{
	hid_t file = H5Fopen("../test/operator/data/test_heisenberg_1d_su2_mpo.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_heisenberg_1d_su2_mpo failed";
	}

	// number of lattice sites
	const int nsites = 5;

	// Hamiltonian parameters
	const double J = 11./7;

	struct su2_mpo heisenberg_1d_mpo;
	construct_heisenberg_1d_su2_mpo(nsites, J, &heisenberg_1d_mpo);
	if (!su2_mpo_is_consistent(&heisenberg_1d_mpo)) {
		return "internal consistency check for SU(2) symmetric Heisenberg Hamiltonian MPO failed";
	}

	// convert to a matrix
	struct dense_tensor heisenberg_1d_mat;
	{
		struct su2_tensor heisenberg_1d_tensor;
		su2_mpo_to_tensor(&heisenberg_1d_mpo, &heisenberg_1d_tensor);
		if (heisenberg_1d_tensor.ndim_logical != 2 * nsites + 2) {
			return "expecting the SU(2) tensor representation of an SU(2) MPO to have logical degree '2 * nsites + 2'";
		}
		su2_to_dense_tensor(&heisenberg_1d_tensor, &heisenberg_1d_mat);
		delete_su2_tensor(&heisenberg_1d_tensor);
		// reshape into a matrix
		const ct_long hspace_dim = ipow(2, nsites);
		const ct_long dim[2] = { hspace_dim, hspace_dim };
		reshape_dense_tensor(2, dim, &heisenberg_1d_mat);
	}

	// reference matrix for checking
	struct dense_tensor heisenberg_1d_mat_ref;
	{
		hsize_t dim_hsize[2];
		if (get_hdf5_dataset_dims(file, "heisenberg_1d_mat", dim_hsize) < 0) {
			return "obtaining dimensions of reference Hamiltonian failed";
		}
		const ct_long dim[2] = { dim_hsize[0], dim_hsize[1] };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &heisenberg_1d_mat_ref);
		// read values from disk
		if (read_hdf5_dataset(file, "heisenberg_1d_mat", H5T_NATIVE_DOUBLE, heisenberg_1d_mat_ref.data) < 0) {
			return "reading matrix entries from disk failed";
		}
	}

	// compare
	if (!dense_tensor_allclose(&heisenberg_1d_mat, &heisenberg_1d_mat_ref, 1e-13)) {
		return "matrix representation of SU(2) symmetric Heisenberg Hamiltonian based on MPO form does not match reference";
	}

	// clean up
	delete_dense_tensor(&heisenberg_1d_mat_ref);
	delete_dense_tensor(&heisenberg_1d_mat);
	delete_su2_mpo(&heisenberg_1d_mpo);

	H5Fclose(file);

	return 0;
}
