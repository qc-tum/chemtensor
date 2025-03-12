#include <complex.h>
#include "mpo_graph.h"
#include "aligned_memory.h"


char* test_mpo_graph_from_opchains_basic()
{
	hid_t file = H5Fopen("../test/operator/data/test_mpo_graph_from_opchains_basic.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_mpo_graph_from_opchains_basic failed";
	}

	const hid_t hdf5_dcomplex_id = construct_hdf5_double_complex_dtype(false);

	// local physical dimension
	const long d = 3;
	// number of sites
	const int nsites = 5;

	const long dim_full = ipow(d, nsites);

	const int nchains = 6;
	int oids[6][5] = {
		{ 10,  8,  6,  3,  1 },
		{      9,  6,  3,  1 },
		{ 10,  8,  7,  4,  1 },
		{      9,  7,  4,  1 },
		{ 10,  8,  7,  5,  2 },
		{      9,  7,  5,  2 },
	};
	int qnums[6][6] = {
		{  0,  1, -1,  0, -1,  1 },
		{      0, -1,  0, -1,  1 },
		{  0,  1, -1,  1, -1,  1 },
		{      0, -1,  1, -1,  1 },
		{  0,  1, -1,  1,  0,  1 },
		{      0, -1,  1,  0,  1 },
	};
	const struct op_chain chains[6] = {
		{ .oids = oids[0], .qnums = qnums[0], .cid = 4, .length = 5, .istart = 0 },
		{ .oids = oids[1], .qnums = qnums[1], .cid = 2, .length = 4, .istart = 1 },
		{ .oids = oids[2], .qnums = qnums[2], .cid = 5, .length = 5, .istart = 0 },
		{ .oids = oids[3], .qnums = qnums[3], .cid = 4, .length = 4, .istart = 1 },
		{ .oids = oids[4], .qnums = qnums[4], .cid = 3, .length = 5, .istart = 0 },
		{ .oids = oids[5], .qnums = qnums[5], .cid = 6, .length = 4, .istart = 1 },
	};

	struct mpo_graph mpo_graph;
	if (mpo_graph_from_opchains(chains, nchains, nsites, &mpo_graph) < 0) {
		return "'mpo_graph_from_opchains' failed internally";
	}

	if (!mpo_graph_is_consistent(&mpo_graph)) {
		return "MPO graph is not consistent";
	}

	int bond_dims_ref[6];
	if (read_hdf5_attribute(file, "bond_dims", H5T_NATIVE_INT, bond_dims_ref) < 0) {
		return "reading virtual bond dimensions from disk failed";
	}
	for (int l = 0; l < nsites + 1; l++){
		if (mpo_graph.num_verts[l] != bond_dims_ref[l]) {
			return "MPO graph virtual bond dimensions do not match reference";
		}
	}

	// read local operator map from disk
	const int num_local_ops = 12;
	struct dense_tensor opmap_tensor;
	const long dim_opmt[3] = { num_local_ops, d, d };
	allocate_dense_tensor(CT_DOUBLE_COMPLEX, 3, dim_opmt, &opmap_tensor);
	// read values from disk
	if (read_hdf5_dataset(file, "opmap", hdf5_dcomplex_id, opmap_tensor.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	// copy individual operators
	struct dense_tensor* opmap = ct_malloc(num_local_ops * sizeof(struct dense_tensor));
	for (int i = 0; i < num_local_ops; i++)
	{
		const long dim[2] = { d, d };
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 2, dim, &opmap[i]);
		const dcomplex* data = opmap_tensor.data;
		memcpy(opmap[i].data, &data[i * d*d], d*d * sizeof(dcomplex));
	}
	delete_dense_tensor(&opmap_tensor);

	// coefficient map; first two entries must always be 0 and 1
	const dcomplex coeffmap[7] = { 0, 1, -0.7 + 0.1i, -1.8 + 0.5i, 3.2 - 1.1i, 1.3 + 0.4i, 0.2 - 0.3i };

	// convert to a dense (full) tensor
	struct dense_tensor a;
	mpo_graph_to_matrix(&mpo_graph, opmap, coeffmap, CT_DOUBLE_COMPLEX, &a);

	// reference matrix representation
	struct dense_tensor a_ref;
	const long dim_a_ref[2] = { dim_full, dim_full };
	allocate_dense_tensor(CT_DOUBLE_COMPLEX, 2, dim_a_ref, &a_ref);
	if (read_hdf5_dataset(file, "mat", hdf5_dcomplex_id, a_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&a, &a_ref, 1e-13)) {
		return "matrix representation of MPO graph does not match reference";
	}

	// sum matrix representations of individual operator chains
	struct dense_tensor a_chains;
	const long dim_a_chains[2] = { dim_full, dim_full };
	allocate_dense_tensor(CT_DOUBLE_COMPLEX, 2, dim_a_chains, &a_chains);
	for (int i = 0; i < nchains; i++)
	{
		struct dense_tensor c;
		op_chain_to_matrix(&chains[i], d, nsites, opmap, coeffmap, CT_DOUBLE_COMPLEX, &c);
		dense_tensor_scalar_multiply_add(numeric_one(CT_DOUBLE_COMPLEX), &c, &a_chains);
		delete_dense_tensor(&c);
	}

	// compare
	if (!dense_tensor_allclose(&a, &a_chains, 1e-13)) {
		return "matrix representation of MPO graph does not match sum of individual chains";
	}

	delete_dense_tensor(&a_ref);
	delete_dense_tensor(&a_chains);
	delete_dense_tensor(&a);
	for (int i = 0; i < num_local_ops; i++)
	{
		delete_dense_tensor(&opmap[i]);
	}
	ct_free(opmap);
	delete_mpo_graph(&mpo_graph);

	H5Tclose(hdf5_dcomplex_id);
	H5Fclose(file);

	return 0;
}


char* test_mpo_graph_from_opchains_advanced()
{
	hid_t file = H5Fopen("../test/operator/data/test_mpo_graph_from_opchains_advanced.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_mpo_graph_from_opchains_advanced failed";
	}

	const hid_t hdf5_scomplex_id = construct_hdf5_single_complex_dtype(false);

	// local physical dimension
	const long d = 4;
	// number of sites
	const int nsites = 5;

	const long dim_full = ipow(d, nsites);

	const int nchains = 7;
	struct op_chain* chains = ct_malloc(nchains * sizeof(struct op_chain));
	for (int i = 0; i < nchains; i++)
	{
		char varname[1024];
		sprintf(varname, "/chain%i/length", i);
		int length;
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, &length) < 0) {
			return "reading operator chain length from disk failed";
		}
		allocate_op_chain(length, &chains[i]);

		sprintf(varname, "/chain%i/oids", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, chains[i].oids) < 0) {
			return "reading operator chain IDs from disk failed";
		}
		sprintf(varname, "/chain%i/qnums", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, chains[i].qnums) < 0) {
			return "reading operator chain quantum numbers from disk failed";
		}
		sprintf(varname, "/chain%i/cid", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, &chains[i].cid) < 0) {
			return "reading operator chain coefficient index from disk failed";
		}
		sprintf(varname, "/chain%i/istart", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, &chains[i].istart) < 0) {
			return "reading operator chain starting site index from disk failed";
		}
	}

	struct mpo_graph mpo_graph;
	if (mpo_graph_from_opchains(chains, nchains, nsites, &mpo_graph) < 0) {
		return "'mpo_graph_from_opchains' failed internally";
	}

	if (!mpo_graph_is_consistent(&mpo_graph)) {
		return "MPO graph is not consistent";
	}

	int bond_dims_ref[6];
	if (read_hdf5_attribute(file, "bond_dims", H5T_NATIVE_INT, bond_dims_ref) < 0) {
		return "reading virtual bond dimensions from disk failed";
	}
	for (int l = 0; l < nsites + 1; l++){
		if (mpo_graph.num_verts[l] != bond_dims_ref[l]) {
			return "MPO graph virtual bond dimensions do not match reference";
		}
	}

	// read local operator map from disk
	const int num_local_ops = 17;
	struct dense_tensor opmap_tensor;
	const long dim_opmt[3] = { num_local_ops, d, d };
	allocate_dense_tensor(CT_SINGLE_COMPLEX, 3, dim_opmt, &opmap_tensor);
	// read values from disk
	if (read_hdf5_dataset(file, "opmap", hdf5_scomplex_id, opmap_tensor.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	// copy individual operators
	struct dense_tensor* opmap = ct_malloc(num_local_ops * sizeof(struct dense_tensor));
	for (int i = 0; i < num_local_ops; i++)
	{
		const long dim[2] = { d, d };
		allocate_dense_tensor(CT_SINGLE_COMPLEX, 2, dim, &opmap[i]);
		const scomplex* data = opmap_tensor.data;
		memcpy(opmap[i].data, &data[i * d*d], d*d * sizeof(scomplex));
	}
	delete_dense_tensor(&opmap_tensor);

	// coefficient map
	scomplex* coeffmap = ct_malloc(8 * sizeof(scomplex));
	if (read_hdf5_dataset(file, "coeffmap", hdf5_scomplex_id, coeffmap) < 0) {
		return "reading coefficient map from disk failed";
	}

	// convert to a dense (full) tensor
	struct dense_tensor a;
	mpo_graph_to_matrix(&mpo_graph, opmap, coeffmap, CT_SINGLE_COMPLEX, &a);

	// sum matrix representations of individual operator chains, as reference
	struct dense_tensor a_chains;
	const long dim_a_chains[2] = { dim_full, dim_full };
	allocate_dense_tensor(CT_SINGLE_COMPLEX, 2, dim_a_chains, &a_chains);
	for (int i = 0; i < nchains; i++)
	{
		struct dense_tensor c;
		op_chain_to_matrix(&chains[i], d, nsites, opmap, coeffmap, CT_SINGLE_COMPLEX, &c);
		dense_tensor_scalar_multiply_add(numeric_one(CT_SINGLE_COMPLEX), &c, &a_chains);
		delete_dense_tensor(&c);
	}

	// compare
	if (!dense_tensor_allclose(&a, &a_chains, 5e-6)) {
		return "matrix representation of MPO graph does not match sum of individual chains";
	}

	delete_dense_tensor(&a_chains);
	delete_dense_tensor(&a);
	ct_free(coeffmap);
	for (int i = 0; i < num_local_ops; i++) {
		delete_dense_tensor(&opmap[i]);
	}
	ct_free(opmap);
	for (int i = 0; i < nchains; i++) {
		delete_op_chain(&chains[i]);
	}
	ct_free(chains);
	delete_mpo_graph(&mpo_graph);

	H5Tclose(hdf5_scomplex_id);
	H5Fclose(file);

	return 0;
}
