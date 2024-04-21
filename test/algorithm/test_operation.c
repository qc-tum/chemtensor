#include <complex.h>
#include "operation.h"
#include "aligned_memory.h"


char* test_operator_inner_product()
{
	hid_t file = H5Fopen("../test/algorithm/data/test_operator_inner_product.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_operator_inner_product failed";
	}

	// number of lattice sites
	const int nsites = 5;
	// local physical dimension
	const long d = 2;

	// physical quantum numbers
	qnumber* qsite = aligned_alloc(MEM_DATA_ALIGN, d * sizeof(qnumber));
	if (read_hdf5_attribute(file, "qsite", H5T_NATIVE_INT, qsite) < 0) {
		return "reading physical quantum numbers from disk failed";
	}

	// virtual bond quantum numbers for 'psi'
	const long dim_bonds_psi[6] = { 1, 10, 19, 26, 7, 1 };
	qnumber** qbonds_psi = aligned_alloc(MEM_DATA_ALIGN, (nsites + 1) * sizeof(qnumber*));
	for (int i = 0; i < nsites + 1; i++)
	{
		qbonds_psi[i] = aligned_alloc(MEM_DATA_ALIGN, dim_bonds_psi[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qbond_psi_%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds_psi[i]) < 0) {
			return "reading virtual bond quantum numbers from disk failed";
		}
	}

	// virtual bond quantum numbers for 'chi'
	const long dim_bonds_chi[6] = { 1, 17, 23, 12, 13, 1 };
	qnumber** qbonds_chi = aligned_alloc(MEM_DATA_ALIGN, (nsites + 1) * sizeof(qnumber*));
	for (int i = 0; i < nsites + 1; i++)
	{
		qbonds_chi[i] = aligned_alloc(MEM_DATA_ALIGN, dim_bonds_chi[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qbond_chi_%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds_chi[i]) < 0) {
			return "reading virtual bond quantum numbers from disk failed";
		}
	}

	// virtual bond quantum numbers for 'op'
	const long dim_bonds_op[6] = { 1, 6, 15, 29, 14, 1 };
	qnumber** qbonds_op = aligned_alloc(MEM_DATA_ALIGN, (nsites + 1) * sizeof(qnumber*));
	for (int i = 0; i < nsites + 1; i++)
	{
		qbonds_op[i] = aligned_alloc(MEM_DATA_ALIGN, dim_bonds_op[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qbond_op_%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds_op[i]) < 0) {
			return "reading virtual bond quantum numbers from disk failed";
		}
	}

	struct mps psi;
	allocate_mps(SINGLE_COMPLEX, nsites, d, qsite, dim_bonds_psi, (const qnumber**)qbonds_psi, &psi);

	// read MPS tensors from disk
	for (int i = 0; i < nsites; i++)
	{
		// read dense tensors from disk
		struct dense_tensor a_dns;
		allocate_dense_tensor(psi.a[i].dtype, psi.a[i].ndim, psi.a[i].dim_logical, &a_dns);
		char varname[1024];
		sprintf(varname, "psi_a%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_FLOAT, a_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		dense_to_block_sparse_tensor_entries(&a_dns, &psi.a[i]);

		delete_dense_tensor(&a_dns);
	}

	if (!mps_is_consistent(&psi)) {
		return "internal MPS consistency check failed";
	}

	struct mps chi;
	allocate_mps(SINGLE_COMPLEX, nsites, d, qsite, dim_bonds_chi, (const qnumber**)qbonds_chi, &chi);

	// read MPS tensors from disk
	for (int i = 0; i < nsites; i++)
	{
		// read dense tensors from disk
		struct dense_tensor a_dns;
		allocate_dense_tensor(chi.a[i].dtype, chi.a[i].ndim, chi.a[i].dim_logical, &a_dns);
		char varname[1024];
		sprintf(varname, "chi_a%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_FLOAT, a_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		dense_to_block_sparse_tensor_entries(&a_dns, &chi.a[i]);

		delete_dense_tensor(&a_dns);
	}

	if (!mps_is_consistent(&chi)) {
		return "internal MPS consistency check failed";
	}

	struct mpo op;
	allocate_mpo(SINGLE_COMPLEX, nsites, d, qsite, dim_bonds_op, (const qnumber**)qbonds_op, &op);

	// read MPS tensors from disk
	for (int i = 0; i < nsites; i++)
	{
		// read dense tensors from disk
		struct dense_tensor a_dns;
		allocate_dense_tensor(op.a[i].dtype, op.a[i].ndim, op.a[i].dim_logical, &a_dns);
		char varname[1024];
		sprintf(varname, "op_a%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_FLOAT, a_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		dense_to_block_sparse_tensor_entries(&a_dns, &op.a[i]);

		delete_dense_tensor(&a_dns);
	}

	if (!mpo_is_consistent(&op)) {
		return "internal MPO consistency check failed";
	}

	// compute operator inner product
	scomplex s;
	operator_inner_product(&chi, &op, &psi, &s);

	scomplex s_ref;
	if (read_hdf5_dataset(file, "s", H5T_NATIVE_FLOAT, &s_ref) < 0) {
		return "reading dot product reference value from disk failed";
	}

	// compare
	if (cabsf(s - s_ref) / cabsf(s_ref) > 1e-6) {
		return "operator inner product does not match reference value";
	}

	delete_mpo(&op);
	delete_mps(&chi);
	delete_mps(&psi);
	for (int i = 0; i < nsites + 1; i++)
	{
		aligned_free(qbonds_op[i]);
		aligned_free(qbonds_chi[i]);
		aligned_free(qbonds_psi[i]);
	}
	aligned_free(qbonds_op);
	aligned_free(qbonds_chi);
	aligned_free(qbonds_psi);
	aligned_free(qsite);

	H5Fclose(file);

	return 0;
}
