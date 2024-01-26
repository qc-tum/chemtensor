#include "mpo_graph.h"
#include "aligned_memory.h"


char* test_mpo_graph_from_opchains_basic()
{
	hid_t file = H5Fopen("../test/data/test_mpo_graph_from_opchains_basic.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_mpo_graph_from_opchains_basic failed";
	}

	// local physical dimension
	const long d = 3;
	// number of sites
	const int nsites = 5;

	const long dim_full = ipow(d, nsites);

	const int oid_identity = 11;

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
		{ .oids = oids[0], .qnums = qnums[0], .coeff =  3.2, .length = 5, .istart = 0 },
		{ .oids = oids[1], .qnums = qnums[1], .coeff = -0.7, .length = 4, .istart = 1 },
		{ .oids = oids[2], .qnums = qnums[2], .coeff =  1.3, .length = 5, .istart = 0 },
		{ .oids = oids[3], .qnums = qnums[3], .coeff =  2.7, .length = 4, .istart = 1 },
		{ .oids = oids[4], .qnums = qnums[4], .coeff = -1.8, .length = 5, .istart = 0 },
		{ .oids = oids[5], .qnums = qnums[5], .coeff =  0.2, .length = 4, .istart = 1 },
	};

	struct mpo_graph mpo_graph;
	mpo_graph_from_opchains(chains, nchains, nsites, oid_identity, &mpo_graph);

	if (!mpo_graph_is_consistent(&mpo_graph)) {
		return "MPO graph is not consistent";
	}

	int bond_dims_ref[6];
	if (read_hdf5_attribute(file, "bond_dims", H5T_NATIVE_INT, bond_dims_ref) < 0) {
		return "reading virtual bond dimensions from disk failed";
	}
	for (int l = 0; l < nsites + 1; l++){
		if (mpo_graph.num_nodes[l] != bond_dims_ref[l]) {
			return "MPO graph virtual bond dimensions do not match reference";
		}
	}

	// read local operator map from disk
	const int num_local_ops = 12;
	struct dense_tensor opmap_tensor;
	const long dim_opmt[3] = { num_local_ops, d, d };
	allocate_dense_tensor(DOUBLE_COMPLEX, 3, dim_opmt, &opmap_tensor);
	// read values from disk
	if (read_hdf5_dataset(file, "opmap", H5T_NATIVE_DOUBLE, opmap_tensor.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	// copy individual operators
	struct dense_tensor* opmap = aligned_alloc(MEM_DATA_ALIGN, num_local_ops * sizeof(struct dense_tensor));
	for (int i = 0; i < num_local_ops; i++)
	{
		const long dim[2] = { d, d };
		allocate_dense_tensor(DOUBLE_COMPLEX, 2, dim, &opmap[i]);
		const dcomplex* data = opmap_tensor.data;
		memcpy(opmap[i].data, &data[i * d*d], d*d * sizeof(dcomplex));
	}
	delete_dense_tensor(&opmap_tensor);

	// convert to a dense (full) tensor
	struct dense_tensor a;
	mpo_graph_to_matrix(&mpo_graph, opmap, DOUBLE_COMPLEX, &a);

	// reference matrix representation
	struct dense_tensor a_ref;
	const long dim_a_ref[2] = { dim_full, dim_full };
	allocate_dense_tensor(DOUBLE_COMPLEX, 2, dim_a_ref, &a_ref);
	if (read_hdf5_dataset(file, "mat", H5T_NATIVE_DOUBLE, a_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&a, &a_ref, 1e-13)) {
		return "matrix representation of MPO graph does not match reference";
	}

	// sum matrix representations of individual operator chains
	struct dense_tensor a_chains;
	const long dim_a_chains[2] = { dim_full, dim_full };
	allocate_dense_tensor(DOUBLE_COMPLEX, 2, dim_a_chains, &a_chains);
	for (int i = 0; i < nchains; i++)
	{
		struct dense_tensor c;
		op_chain_to_matrix(&chains[i], d, nsites, opmap, DOUBLE_COMPLEX, &c);
		dense_tensor_scalar_multiply_add(numeric_one(DOUBLE_COMPLEX), &c, &a_chains);
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
	aligned_free(opmap);
	delete_mpo_graph(&mpo_graph);

	H5Fclose(file);

	return 0;
}


char* test_mpo_graph_from_opchains_advanced()
{
	hid_t file = H5Fopen("../test/data/test_mpo_graph_from_opchains_advanced.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_mpo_graph_from_opchains_advanced failed";
	}

	// local physical dimension
	const long d = 4;
	// number of sites
	const int nsites = 5;

	const long dim_full = ipow(d, nsites);

	const int oid_identity = 0;

	const int nchains = 7;
	struct op_chain* chains = aligned_alloc(MEM_DATA_ALIGN, nchains * sizeof(struct op_chain));
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
		sprintf(varname, "/chain%i/coeff", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_DOUBLE, &chains[i].coeff) < 0) {
			return "reading operator chain coefficient from disk failed";
		}
		sprintf(varname, "/chain%i/istart", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, &chains[i].istart) < 0) {
			return "reading operator chain starting site index from disk failed";
		}
	}

	struct mpo_graph mpo_graph;
	mpo_graph_from_opchains(chains, nchains, nsites, oid_identity, &mpo_graph);

	if (!mpo_graph_is_consistent(&mpo_graph)) {
		return "MPO graph is not consistent";
	}

	int bond_dims_ref[6];
	if (read_hdf5_attribute(file, "bond_dims", H5T_NATIVE_INT, bond_dims_ref) < 0) {
		return "reading virtual bond dimensions from disk failed";
	}
	for (int l = 0; l < nsites + 1; l++){
		if (mpo_graph.num_nodes[l] != bond_dims_ref[l]) {
			return "MPO graph virtual bond dimensions do not match reference";
		}
	}

	// read local operator map from disk
	const int num_local_ops = 17;
	struct dense_tensor opmap_tensor;
	const long dim_opmt[3] = { num_local_ops, d, d };
	allocate_dense_tensor(SINGLE_COMPLEX, 3, dim_opmt, &opmap_tensor);
	// read values from disk
	if (read_hdf5_dataset(file, "opmap", H5T_NATIVE_FLOAT, opmap_tensor.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	// copy individual operators
	struct dense_tensor* opmap = aligned_alloc(MEM_DATA_ALIGN, num_local_ops * sizeof(struct dense_tensor));
	for (int i = 0; i < num_local_ops; i++)
	{
		const long dim[2] = { d, d };
		allocate_dense_tensor(SINGLE_COMPLEX, 2, dim, &opmap[i]);
		const scomplex* data = opmap_tensor.data;
		memcpy(opmap[i].data, &data[i * d*d], d*d * sizeof(scomplex));
	}
	delete_dense_tensor(&opmap_tensor);

	// convert to a dense (full) tensor
	struct dense_tensor a;
	mpo_graph_to_matrix(&mpo_graph, opmap, SINGLE_COMPLEX, &a);

	// reference matrix representation
	struct dense_tensor a_ref;
	const long dim_a_ref[2] = { dim_full, dim_full };
	allocate_dense_tensor(SINGLE_COMPLEX, 2, dim_a_ref, &a_ref);
	if (read_hdf5_dataset(file, "mat", H5T_NATIVE_FLOAT, a_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&a, &a_ref, 1e-6)) {
		return "matrix representation of MPO graph does not match reference";
	}

	// sum matrix representations of individual operator chains
	struct dense_tensor a_chains;
	const long dim_a_chains[2] = { dim_full, dim_full };
	allocate_dense_tensor(SINGLE_COMPLEX, 2, dim_a_chains, &a_chains);
	for (int i = 0; i < nchains; i++)
	{
		struct dense_tensor c;
		op_chain_to_matrix(&chains[i], d, nsites, opmap, SINGLE_COMPLEX, &c);
		dense_tensor_scalar_multiply_add(numeric_one(SINGLE_COMPLEX), &c, &a_chains);
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
	aligned_free(opmap);
	for (int i = 0; i < nchains; i++)
	{
		delete_op_chain(&chains[i]);
	}
	aligned_free(chains);
	delete_mpo_graph(&mpo_graph);

	H5Fclose(file);

	return 0;
}
