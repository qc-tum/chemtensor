#include "ttno_graph.h"
#include "ttno.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


char* test_ttno_graph_from_opchains()
{
	hid_t file = H5Fopen("../test/operator/data/test_ttno_graph_from_opchains.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_ttno_graph_from_opchains failed";
	}

	// number of lattice sites
	const int nsites = 8;
	// local physical dimension
	const long d = 3;

	const long dim_full = ipow(d, nsites);

	// tree topology:
	//
	//  4           6
	//    \       /
	//      \   /
	//        0
	//        |
	//        |
	//  2 --- 3 --- 1 --- 7
	//        |
	//        |
	//        5
	//
	int neigh0[] = { 3, 4, 6 };
	int neigh1[] = { 3, 7 };
	int neigh2[] = { 3 };
	int neigh3[] = { 0, 1, 2, 5 };
	int neigh4[] = { 0 };
	int neigh5[] = { 3 };
	int neigh6[] = { 0 };
	int neigh7[] = { 1 };
	int* neighbor_map[8] = {
		neigh0, neigh1, neigh2, neigh3, neigh4, neigh5, neigh6, neigh7,
	};
	int num_neighbors[8] = {
		ARRLEN(neigh0), ARRLEN(neigh1), ARRLEN(neigh2), ARRLEN(neigh3), ARRLEN(neigh4), ARRLEN(neigh5), ARRLEN(neigh6), ARRLEN(neigh7),
	};
	struct abstract_graph topology = {
		.neighbor_map  = neighbor_map,
		.num_neighbors = num_neighbors,
		.num_nodes     = nsites,
	};
	assert(abstract_graph_is_connected_tree(&topology));

	const int oid_identity = 5;

	const int nchains = 9;
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

	struct ttno_graph graph;
	if (ttno_graph_from_opchains(chains, nchains, &topology, oid_identity, &graph) < 0) {
		return "'ttno_graph_from_opchains' failed internally";
	}
	if (!ttno_graph_is_consistent(&graph)) {
		return "constructed TTNO graph is inconsistent";
	}

	// bond dimensions
	int rank_046_12357;
	if (read_hdf5_attribute(file, "rank_046_12357", H5T_NATIVE_INT, &rank_046_12357) < 0) {
		return "reading operator partitioning rank from disk failed";
	}
	int rank_17_023456;
	if (read_hdf5_attribute(file, "rank_17_023456", H5T_NATIVE_INT, &rank_17_023456) < 0) {
		return "reading operator partitioning rank from disk failed";
	}
	int rank_2_0134567;
	if (read_hdf5_attribute(file, "rank_2_0134567", H5T_NATIVE_INT, &rank_2_0134567) < 0) {
		return "reading operator partitioning rank from disk failed";
	}
	int rank_6_0123457;
	if (read_hdf5_attribute(file, "rank_6_0123457", H5T_NATIVE_INT, &rank_6_0123457) < 0) {
		return "reading operator partitioning rank from disk failed";
	}

	if (graph.num_verts[0*nsites + 3] != rank_046_12357) {
		return "virtual bond dimension does not match expected partitioning matrix rank";
	}
	if (graph.num_verts[1*nsites + 3] != rank_17_023456) {
		return "virtual bond dimension does not match expected partitioning matrix rank";
	}
	if (graph.num_verts[2*nsites + 3] != rank_2_0134567) {
		return "virtual bond dimension does not match expected partitioning matrix rank";
	}
	if (graph.num_verts[0*nsites + 6] != rank_6_0123457) {
		return "virtual bond dimension does not match expected partitioning matrix rank";
	}

	// read local operator map from disk
	const int num_local_ops = 18;
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

	// convert TTNO graph to a (dense) matrix
	struct dense_tensor mat;
	ttno_graph_to_matrix(&graph, opmap, &mat);

	// sum matrix representations of individual operator chains, as reference
	struct dense_tensor mat_ref;
	const long dim_mat_ref[2] = { dim_full, dim_full };
	allocate_dense_tensor(DOUBLE_COMPLEX, 2, dim_mat_ref, &mat_ref);
	for (int i = 0; i < nchains; i++)
	{
		struct dense_tensor c;
		op_chain_to_matrix(&chains[i], d, nsites, opmap, DOUBLE_COMPLEX, &c);
		dense_tensor_scalar_multiply_add(numeric_one(DOUBLE_COMPLEX), &c, &mat_ref);
		delete_dense_tensor(&c);
	}

	// compare
	if (!dense_tensor_allclose(&mat, &mat_ref, 1e-13)) {
		return "matrix representation of TTNO graph does not match reference matrix";
	}

	// clean up
	delete_dense_tensor(&mat_ref);
	delete_dense_tensor(&mat);
	for (int i = 0; i < num_local_ops; i++) {
		delete_dense_tensor(&opmap[i]);
	}
	aligned_free(opmap);
	for (int i = 0; i < nchains; i++)
	{
		delete_op_chain(&chains[i]);
	}
	aligned_free(chains);
	delete_ttno_graph(&graph);

	H5Fclose(file);

	return 0;
}
