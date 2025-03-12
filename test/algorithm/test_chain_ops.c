#include <complex.h>
#include "chain_ops.h"
#include "aligned_memory.h"


char* test_mpo_inner_product()
{
	hid_t file = H5Fopen("../test/algorithm/data/test_mpo_inner_product.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_mpo_inner_product failed";
	}

	const hid_t hdf5_scomplex_id = construct_hdf5_single_complex_dtype(false);

	// number of lattice sites
	const int nsites = 5;
	// local physical dimension
	const long d = 2;

	// physical quantum numbers
	qnumber* qsite = ct_malloc(d * sizeof(qnumber));
	if (read_hdf5_attribute(file, "qsite", H5T_NATIVE_INT, qsite) < 0) {
		return "reading physical quantum numbers from disk failed";
	}

	// virtual bond quantum numbers for 'psi'
	const long dim_bonds_psi[6] = { 1, 10, 19, 26, 7, 1 };
	qnumber** qbonds_psi = ct_malloc((nsites + 1) * sizeof(qnumber*));
	for (int i = 0; i < nsites + 1; i++)
	{
		qbonds_psi[i] = ct_malloc(dim_bonds_psi[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qbond_psi_%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds_psi[i]) < 0) {
			return "reading virtual bond quantum numbers from disk failed";
		}
	}

	// virtual bond quantum numbers for 'chi'
	const long dim_bonds_chi[6] = { 1, 17, 23, 12, 13, 1 };
	qnumber** qbonds_chi = ct_malloc((nsites + 1) * sizeof(qnumber*));
	for (int i = 0; i < nsites + 1; i++)
	{
		qbonds_chi[i] = ct_malloc(dim_bonds_chi[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qbond_chi_%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds_chi[i]) < 0) {
			return "reading virtual bond quantum numbers from disk failed";
		}
	}

	// virtual bond quantum numbers for 'op'
	const long dim_bonds_op[6] = { 1, 6, 15, 29, 14, 1 };
	qnumber** qbonds_op = ct_malloc((nsites + 1) * sizeof(qnumber*));
	for (int i = 0; i < nsites + 1; i++)
	{
		qbonds_op[i] = ct_malloc(dim_bonds_op[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qbond_op_%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds_op[i]) < 0) {
			return "reading virtual bond quantum numbers from disk failed";
		}
	}

	struct mps psi;
	allocate_mps(CT_SINGLE_COMPLEX, nsites, d, qsite, dim_bonds_psi, (const qnumber**)qbonds_psi, &psi);

	// read MPS tensors from disk
	for (int i = 0; i < nsites; i++)
	{
		// read dense tensors from disk
		struct dense_tensor a_dns;
		allocate_dense_tensor(psi.a[i].dtype, psi.a[i].ndim, psi.a[i].dim_logical, &a_dns);
		char varname[1024];
		sprintf(varname, "psi_a%i", i);
		if (read_hdf5_dataset(file, varname, hdf5_scomplex_id, a_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		dense_to_block_sparse_tensor_entries(&a_dns, &psi.a[i]);

		delete_dense_tensor(&a_dns);
	}

	if (!mps_is_consistent(&psi)) {
		return "internal MPS consistency check failed";
	}

	struct mps chi;
	allocate_mps(CT_SINGLE_COMPLEX, nsites, d, qsite, dim_bonds_chi, (const qnumber**)qbonds_chi, &chi);

	// read MPS tensors from disk
	for (int i = 0; i < nsites; i++)
	{
		// read dense tensors from disk
		struct dense_tensor a_dns;
		allocate_dense_tensor(chi.a[i].dtype, chi.a[i].ndim, chi.a[i].dim_logical, &a_dns);
		char varname[1024];
		sprintf(varname, "chi_a%i", i);
		if (read_hdf5_dataset(file, varname, hdf5_scomplex_id, a_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		dense_to_block_sparse_tensor_entries(&a_dns, &chi.a[i]);

		delete_dense_tensor(&a_dns);
	}

	if (!mps_is_consistent(&chi)) {
		return "internal MPS consistency check failed";
	}

	struct mpo op;
	allocate_mpo(CT_SINGLE_COMPLEX, nsites, d, qsite, dim_bonds_op, (const qnumber**)qbonds_op, &op);

	// read MPO tensors from disk
	for (int i = 0; i < nsites; i++)
	{
		// read dense tensors from disk
		struct dense_tensor a_dns;
		allocate_dense_tensor(op.a[i].dtype, op.a[i].ndim, op.a[i].dim_logical, &a_dns);
		char varname[1024];
		sprintf(varname, "op_a%i", i);
		if (read_hdf5_dataset(file, varname, hdf5_scomplex_id, a_dns.data) < 0) {
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
	mpo_inner_product(&chi, &op, &psi, &s);

	scomplex s_ref;
	if (read_hdf5_dataset(file, "s", hdf5_scomplex_id, &s_ref) < 0) {
		return "reading operator inner product reference value from disk failed";
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
		ct_free(qbonds_op[i]);
		ct_free(qbonds_chi[i]);
		ct_free(qbonds_psi[i]);
	}
	ct_free(qbonds_op);
	ct_free(qbonds_chi);
	ct_free(qbonds_psi);
	ct_free(qsite);

	H5Tclose(hdf5_scomplex_id);
	H5Fclose(file);

	return 0;
}


char* test_apply_mpo()
{
	hid_t file = H5Fopen("../test/algorithm/data/test_apply_mpo.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_mpo failed";
	}

	const hid_t hdf5_dcomplex_id = construct_hdf5_double_complex_dtype(false);

	// number of lattice sites
	const int nsites = 6;
	// local physical dimension
	const long d = 3;

	// physical quantum numbers
	qnumber* qsite = ct_malloc(d * sizeof(qnumber));
	if (read_hdf5_attribute(file, "qsite", H5T_NATIVE_INT, qsite) < 0) {
		return "reading physical quantum numbers from disk failed";
	}

	// virtual bond quantum numbers for 'psi'
	const long dim_bonds_psi[7] = { 1, 7, 21, 25, 11, 4, 1 };
	qnumber** qbonds_psi = ct_malloc((nsites + 1) * sizeof(qnumber*));
	for (int i = 0; i < nsites + 1; i++)
	{
		qbonds_psi[i] = ct_malloc(dim_bonds_psi[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qbond_psi_%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds_psi[i]) < 0) {
			return "reading virtual bond quantum numbers from disk failed";
		}
	}

	// virtual bond quantum numbers for 'op'
	const long dim_bonds_op[7] = { 1, 6, 15, 33, 29, 14, 1 };
	qnumber** qbonds_op = ct_malloc((nsites + 1) * sizeof(qnumber*));
	for (int i = 0; i < nsites + 1; i++)
	{
		qbonds_op[i] = ct_malloc(dim_bonds_op[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qbond_op_%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds_op[i]) < 0) {
			return "reading virtual bond quantum numbers from disk failed";
		}
	}

	struct mps psi;
	allocate_mps(CT_DOUBLE_COMPLEX, nsites, d, qsite, dim_bonds_psi, (const qnumber**)qbonds_psi, &psi);

	// read MPS tensors from disk
	for (int i = 0; i < nsites; i++)
	{
		// read dense tensors from disk
		struct dense_tensor a_dns;
		allocate_dense_tensor(psi.a[i].dtype, psi.a[i].ndim, psi.a[i].dim_logical, &a_dns);
		char varname[1024];
		sprintf(varname, "psi_a%i", i);
		if (read_hdf5_dataset(file, varname, hdf5_dcomplex_id, a_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		dense_to_block_sparse_tensor_entries(&a_dns, &psi.a[i]);

		delete_dense_tensor(&a_dns);
	}

	if (!mps_is_consistent(&psi)) {
		return "internal MPS consistency check failed";
	}

	struct mpo op;
	allocate_mpo(CT_DOUBLE_COMPLEX, nsites, d, qsite, dim_bonds_op, (const qnumber**)qbonds_op, &op);

	// read MPO tensors from disk
	for (int i = 0; i < nsites; i++)
	{
		// read dense tensors from disk
		struct dense_tensor a_dns;
		allocate_dense_tensor(op.a[i].dtype, op.a[i].ndim, op.a[i].dim_logical, &a_dns);
		char varname[1024];
		sprintf(varname, "op_a%i", i);
		if (read_hdf5_dataset(file, varname, hdf5_dcomplex_id, a_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		dense_to_block_sparse_tensor_entries(&a_dns, &op.a[i]);

		delete_dense_tensor(&a_dns);
	}

	if (!mpo_is_consistent(&op)) {
		return "internal MPO consistency check failed";
	}

	// apply operator
	struct mps op_psi;
	apply_mpo(&op, &psi, &op_psi);

	if (!mps_is_consistent(&op_psi)) {
		return "internal MPS consistency check failed";
	}

	// convert to vector and matrix and perform matrix-vector multiplication, as reference
	struct block_sparse_tensor op_psi_ref;
	{
		struct block_sparse_tensor psi_vec;
		mps_to_statevector(&psi, &psi_vec);
		assert(psi_vec.ndim == 3);  // includes dummy virtual bonds

		struct block_sparse_tensor op_mat;
		mpo_to_matrix(&op, &op_mat);
		assert(op_mat.ndim == 4);  // includes dummy virtual bonds

		// move physical input axis of 'op_mat' to the end
		const int perm_op[4] = { 0, 1, 3, 2 };
		struct block_sparse_tensor r;
		transpose_block_sparse_tensor(perm_op, &op_mat, &r);

		// move physical axis of 'psi_vec' to the beginning
		const int perm_psi[3] = { 1, 0, 2 };
		struct block_sparse_tensor s;
		transpose_block_sparse_tensor(perm_psi, &psi_vec, &s);

		// perform logical matrix-vector multiplication
		struct block_sparse_tensor t;
		block_sparse_tensor_dot(&r, TENSOR_AXIS_RANGE_TRAILING, &s, TENSOR_AXIS_RANGE_LEADING, 1, &t);
		delete_block_sparse_tensor(&r);
		delete_block_sparse_tensor(&s);

		// reorder axes
		const int perm_ax[5] = { 0, 3, 1, 2, 4 };
		transpose_block_sparse_tensor(perm_ax, &t, &r);
		delete_block_sparse_tensor(&t);

		// flatten left and right dummy virtual bonds
		block_sparse_tensor_flatten_axes(&r, 0, TENSOR_AXIS_OUT, &s);
		delete_block_sparse_tensor(&r);
		block_sparse_tensor_flatten_axes(&s, 2, TENSOR_AXIS_IN, &op_psi_ref);
		delete_block_sparse_tensor(&s);

		delete_block_sparse_tensor(&op_mat);
		delete_block_sparse_tensor(&psi_vec);
	}
	struct block_sparse_tensor op_psi_vec;
	mps_to_statevector(&op_psi, &op_psi_vec);

	// compare
	if (!block_sparse_tensor_allclose(&op_psi_vec, &op_psi_ref, 1e-13)) {
		return "applying an MPO to an MPS does not match reference vector based on matrix-vector multiplication";
	}

	delete_block_sparse_tensor(&op_psi_ref);
	delete_block_sparse_tensor(&op_psi_vec);
	delete_mps(&op_psi);
	delete_mpo(&op);
	delete_mps(&psi);
	for (int i = 0; i < nsites + 1; i++)
	{
		ct_free(qbonds_op[i]);
		ct_free(qbonds_psi[i]);
	}
	ct_free(qbonds_op);
	ct_free(qbonds_psi);
	ct_free(qsite);

	H5Tclose(hdf5_dcomplex_id);
	H5Fclose(file);

	return 0;
}
