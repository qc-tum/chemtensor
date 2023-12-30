#include "mps.h"
#include "test_dense_tensor.h"
#include "config.h"


char* test_mps_to_statevector()
{
	hid_t file = H5Fopen("../test/data/test_mps_to_statevector.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_mps_to_statevector failed";
	}

	// number of lattice sites
	const int nsites = 5;
	// local physical dimension
	const long d = 3;

	// physical quantum numbers
	qnumber* qsite = aligned_alloc(MEM_DATA_ALIGN, d * sizeof(qnumber));
	if (read_hdf5_attribute(file, "qsite", H5T_NATIVE_INT, qsite) < 0) {
		return "reading physical quantum numbers from disk failed";
	}

	// virtual bond quantum numbers
	const long dim_bonds[6] = { 1, 7, 10, 11, 5, 1 };
	qnumber** qbonds = aligned_alloc(MEM_DATA_ALIGN, (nsites + 1) * sizeof(qnumber*));
	for (int i = 0; i < nsites + 1; i++)
	{
		qbonds[i] = aligned_alloc(MEM_DATA_ALIGN, dim_bonds[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qbond%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds[i]) < 0) {
			return "reading virtual bond quantum numbers from disk failed";
		}
	}

	struct mps mps;
	allocate_mps(DOUBLE_COMPLEX, nsites, d, qsite, dim_bonds, (const qnumber**)qbonds, &mps);

	// read MPS tensors from disk
	for (int i = 0; i < nsites; i++)
	{
		// read dense tensors from disk
		struct dense_tensor a_dns;
		allocate_dense_tensor(mps.a[i].dtype, mps.a[i].ndim, mps.a[i].dim_logical, &a_dns);
		char varname[1024];
		sprintf(varname, "a%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, a_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		dense_to_block_sparse_tensor_entries(&a_dns, &mps.a[i]);

		delete_dense_tensor(&a_dns);
	}

	if (!mps_is_consistent(&mps)) {
		return "internal MPS consistency check failed";
	}

	for (int i = 0; i < nsites + 1; i++) {
		if (mps_bond_dim(&mps, i) != dim_bonds[i]) {
			return "MPS virtual bond dimension does not match reference";
		}
	}

	// convert to state vector
	struct block_sparse_tensor vec;
	mps_to_statevector(&mps, &vec);

	// convert to dense vector (for comparison with reference vector)
	struct dense_tensor vec_dns;
	block_sparse_to_dense_tensor(&vec, &vec_dns);

	// read reference state vector from disk
	struct dense_tensor vec_ref;
	const long dim_vec_ref[3] = { 1, 243, 1 };  // include dummy virtual bond dimensions
	allocate_dense_tensor(DOUBLE_COMPLEX, 3, dim_vec_ref, &vec_ref);
	if (read_hdf5_dataset(file, "vec", H5T_NATIVE_DOUBLE, vec_ref.data) < 0) {
		return "reading state vector entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&vec_dns, &vec_ref, 1e-13)) {
		return "state vector obtained from MPS does not match reference";
	}

	delete_dense_tensor(&vec_ref);
	delete_dense_tensor(&vec_dns);
	delete_block_sparse_tensor(&vec);
	delete_mps(&mps);
	for (int i = 0; i < nsites + 1; i++)
	{
		aligned_free(qbonds[i]);
	}
	aligned_free(qbonds);
	aligned_free(qsite);

	return 0;
}
