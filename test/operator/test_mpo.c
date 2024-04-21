#include <math.h>
#include "mpo.h"
#include "aligned_memory.h"


char* test_mpo_from_graph()
{
	hid_t file = H5Fopen("../test/operator/data/test_mpo_from_graph.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_mpo_from_graph failed";
	}

	// number of lattice sites
	const int nsites = 5;
	// local physical dimension
	const long d = 4;

	const qnumber qsite[4] = { -1, 0, 2, 0 };

	int n0_eids_1[2] = { 0, 1 };
	int n1_eids_0[1] = { 0 };
	int n1_eids_1[1] = { 0 };
	int n2_eids_0[1] = { 1 };
	int n2_eids_1[1] = { 1 };
	int n3_eids_0[2] = { 0, 1 };
	int n3_eids_1[2] = { 0, 1 };
	int n4_eids_0[1] = { 0 };
	int n4_eids_1[1] = { 0 };
	int n5_eids_0[1] = { 1 };
	int n5_eids_1[2] = { 1, 2 };
	int n6_eids_0[2] = { 0, 1 };
	int n6_eids_1[2] = { 0, 2 };
	int n7_eids_0[1] = { 2 };
	int n7_eids_1[1] = { 1 };
	int n8_eids_0[3] = { 0, 1, 2 };
	struct mpo_graph_node node_list[] = {
		{ .qnum =  0, .eids = { NULL,      n0_eids_1 }, .num_edges = { 0, 2 } },
		{ .qnum =  1, .eids = { n1_eids_0, n1_eids_1 }, .num_edges = { 1, 1 } },
		{ .qnum =  0, .eids = { n2_eids_0, n2_eids_1 }, .num_edges = { 1, 1 } },
		{ .qnum = -1, .eids = { n3_eids_0, n3_eids_1 }, .num_edges = { 2, 2 } },
		{ .qnum =  0, .eids = { n4_eids_0, n4_eids_1 }, .num_edges = { 1, 1 } },
		{ .qnum =  1, .eids = { n5_eids_0, n5_eids_1 }, .num_edges = { 1, 2 } },
		{ .qnum = -1, .eids = { n6_eids_0, n6_eids_1 }, .num_edges = { 2, 2 } },
		{ .qnum =  0, .eids = { n7_eids_0, n7_eids_1 }, .num_edges = { 1, 1 } },
		{ .qnum =  1, .eids = { n8_eids_0, NULL      }, .num_edges = { 3, 0 } },
	};
	struct mpo_graph_node* graph_nodes[] = {
		&node_list[0],
		&node_list[1],
		&node_list[3],
		&node_list[4],
		&node_list[6],
		&node_list[8],
	};
	int graph_num_nodes[6] = { 1, 2, 1, 2, 2, 1 };

	struct local_op_ref  e0_opics[1] = { { .oid =  2, .coeff = -0.6 }, };
	struct local_op_ref  e1_opics[2] = { { .oid =  5, .coeff =  1.3 }, { .oid = 11, .coeff = -0.4 }, };
	struct local_op_ref  e2_opics[1] = { { .oid =  1, .coeff =  0.4 }, };
	struct local_op_ref  e3_opics[1] = { { .oid =  4, .coeff = -1.2 }, };
	struct local_op_ref  e4_opics[1] = { { .oid =  7, .coeff =  0.7 }, };
	struct local_op_ref  e5_opics[2] = { { .oid =  0, .coeff =  0.5 }, { .oid = 10, .coeff =  0.6 }, };
	struct local_op_ref  e6_opics[3] = { { .oid =  3, .coeff = -1.6 }, { .oid = 12, .coeff = -2.1 }, { .oid =  4, .coeff =  0.5 }, };
	struct local_op_ref  e7_opics[1] = { { .oid =  6, .coeff =  0.8 }, };
	struct local_op_ref  e8_opics[1] = { { .oid =  8, .coeff = -0.3 }, };
	struct local_op_ref  e9_opics[1] = { { .oid = 10, .coeff =  0.9 }, };
	struct local_op_ref e10_opics[1] = { { .oid =  9, .coeff = -0.2 }, };
	struct local_op_ref e11_opics[2] = { { .oid = 13, .coeff =  1.2 }, { .oid = 14, .coeff = -0.6 }};
	struct mpo_graph_edge edge_list[] = {
		{ .nids = { 0, 0 }, .opics =  e0_opics, .nopics = 1 },
		{ .nids = { 0, 1 }, .opics =  e1_opics, .nopics = 2 },
		{ .nids = { 0, 0 }, .opics =  e2_opics, .nopics = 1 },
		{ .nids = { 1, 0 }, .opics =  e3_opics, .nopics = 1 },
		{ .nids = { 0, 0 }, .opics =  e4_opics, .nopics = 1 },
		{ .nids = { 0, 1 }, .opics =  e5_opics, .nopics = 2 },
		{ .nids = { 0, 0 }, .opics =  e6_opics, .nopics = 3 },
		{ .nids = { 1, 0 }, .opics =  e7_opics, .nopics = 1 },
		{ .nids = { 1, 1 }, .opics =  e8_opics, .nopics = 1 },
		{ .nids = { 0, 0 }, .opics =  e9_opics, .nopics = 1 },
		{ .nids = { 1, 0 }, .opics = e10_opics, .nopics = 1 },
		{ .nids = { 0, 0 }, .opics = e11_opics, .nopics = 2 },
	};
	struct mpo_graph_edge* graph_edges[] = {
		&edge_list[0],
		&edge_list[2],
		&edge_list[4],
		&edge_list[6],
		&edge_list[9],
	};
	int graph_num_edges[5] = { 2, 2, 2, 3, 3 };

	// construct MPO graph
	struct mpo_graph graph = {
		.nodes     = graph_nodes,
		.edges     = graph_edges,
		.num_nodes = graph_num_nodes,
		.num_edges = graph_num_edges,
		.nsites    = nsites
	};
	if (!mpo_graph_is_consistent(&graph)) {
		return "internal MPO graph construction is inconsistent";
	}

	// read local operator map from disk
	const int num_local_ops = 15;
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

	// construct MPO from an MPO graph
	struct mpo mpo;
	mpo_from_graph(DOUBLE_COMPLEX, d, qsite, &graph, opmap, &mpo);

	if (!mpo_is_consistent(&mpo)) {
		return "internal MPO consistency check failed";
	}

	const long dim_bonds[6] = { 1, 2, 1, 2, 2, 1 };
	for (int i = 0; i < nsites + 1; i++) {
		if (mpo_bond_dim(&mpo, i) != dim_bonds[i]) {
			return "MPO virtual bond dimension does not match reference";
		}
	}

	// convert to a matrix
	struct block_sparse_tensor mat;
	mpo_to_matrix(&mpo, &mat);
	// convert to dense matrix (for comparison with reference matrix)
	struct dense_tensor mat_dns;
	block_sparse_to_dense_tensor(&mat, &mat_dns);

	// convert MPO graph to a (dense) matrix, as reference
	struct dense_tensor mat_ref;
	mpo_graph_to_matrix(&graph, opmap, DOUBLE_COMPLEX, &mat_ref);
	// include dummy virtual bond dimensions
	const long dim_mat_graph[4] = { 1, mat_ref.dim[0], mat_ref.dim[1], 1};
	reshape_dense_tensor(4, dim_mat_graph, &mat_ref);

	// compare
	if (!dense_tensor_allclose(&mat_dns, &mat_ref, 1e-13)) {
		return "matrix representation of MPO does not match corresponding matrix obtained from MPO graph";
	}

	delete_dense_tensor(&mat_ref);
	delete_dense_tensor(&mat_dns);
	delete_block_sparse_tensor(&mat);
	delete_mpo(&mpo);
	for (int i = 0; i < num_local_ops; i++)
	{
		delete_dense_tensor(&opmap[i]);
	}
	aligned_free(opmap);

	H5Fclose(file);

	return 0;
}
