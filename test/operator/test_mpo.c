#include <math.h>
#include "mpo.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


char* test_mpo_from_assembly()
{
	hid_t file = H5Fopen("../test/operator/data/test_mpo_from_assembly.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_mpo_from_assembly failed";
	}

	const hid_t hdf5_dcomplex_id = construct_hdf5_double_complex_dtype(false);

	// number of lattice sites
	const int nsites = 5;
	// local physical dimension
	const long d = 4;

	const qnumber qsite[4] = { -1, 0, 2, 0 };

	int v0_eids_1[] = { 0, 1 };
	int v1_eids_0[] = { 0 };
	int v1_eids_1[] = { 0 };
	int v2_eids_0[] = { 1 };
	int v2_eids_1[] = { 1 };
	int v3_eids_0[] = { 0, 1 };
	int v3_eids_1[] = { 0, 1 };
	int v4_eids_0[] = { 0 };
	int v4_eids_1[] = { 0 };
	int v5_eids_0[] = { 1 };
	int v5_eids_1[] = { 1, 2 };
	int v6_eids_0[] = { 0, 1 };
	int v6_eids_1[] = { 0, 2 };
	int v7_eids_0[] = { 2 };
	int v7_eids_1[] = { 1 };
	int v8_eids_0[] = { 0, 1, 2 };
	struct mpo_graph_vertex vertex_list[] = {
		{ .eids = { NULL,      v0_eids_1 }, .num_edges = { 0,                 ARRLEN(v0_eids_1) }, .qnum =  0 },
		{ .eids = { v1_eids_0, v1_eids_1 }, .num_edges = { ARRLEN(v1_eids_0), ARRLEN(v1_eids_1) }, .qnum =  1 },
		{ .eids = { v2_eids_0, v2_eids_1 }, .num_edges = { ARRLEN(v2_eids_0), ARRLEN(v2_eids_1) }, .qnum =  0 },
		{ .eids = { v3_eids_0, v3_eids_1 }, .num_edges = { ARRLEN(v3_eids_0), ARRLEN(v3_eids_1) }, .qnum = -1 },
		{ .eids = { v4_eids_0, v4_eids_1 }, .num_edges = { ARRLEN(v4_eids_0), ARRLEN(v4_eids_1) }, .qnum =  0 },
		{ .eids = { v5_eids_0, v5_eids_1 }, .num_edges = { ARRLEN(v5_eids_0), ARRLEN(v5_eids_1) }, .qnum =  1 },
		{ .eids = { v6_eids_0, v6_eids_1 }, .num_edges = { ARRLEN(v6_eids_0), ARRLEN(v6_eids_1) }, .qnum = -1 },
		{ .eids = { v7_eids_0, v7_eids_1 }, .num_edges = { ARRLEN(v7_eids_0), ARRLEN(v7_eids_1) }, .qnum =  0 },
		{ .eids = { v8_eids_0, NULL      }, .num_edges = { ARRLEN(v8_eids_0), 0                 }, .qnum =  1 },
	};
	struct mpo_graph_vertex* graph_vertices[] = {
		&vertex_list[0],
		&vertex_list[1],
		&vertex_list[3],
		&vertex_list[4],
		&vertex_list[6],
		&vertex_list[8],
	};
	int graph_num_vertices[6] = { 1, 2, 1, 2, 2, 1 };

	struct local_op_ref  e0_opics[] = { { .oid =  2, .cid =  5 }, };
	struct local_op_ref  e1_opics[] = { { .oid =  5, .cid = 15 }, { .oid = 11, .cid = 10 }, };
	struct local_op_ref  e2_opics[] = { { .oid =  1, .cid = 12 }, };
	struct local_op_ref  e3_opics[] = { { .oid =  4, .cid =  4 }, };
	struct local_op_ref  e4_opics[] = { { .oid =  7, .cid =  7 }, };
	struct local_op_ref  e5_opics[] = { { .oid =  0, .cid =  9 }, { .oid = 10, .cid =  3 }, };
	struct local_op_ref  e6_opics[] = { { .oid =  3, .cid =  2 }, { .oid = 12, .cid =  8 }, { .oid =  4, .cid =  9 }, };
	struct local_op_ref  e7_opics[] = { { .oid =  6, .cid = 13 }, };
	struct local_op_ref  e8_opics[] = { { .oid =  8, .cid =  6 }, };
	struct local_op_ref  e9_opics[] = { { .oid = 10, .cid = 16 }, };
	struct local_op_ref e10_opics[] = { { .oid =  9, .cid = 14 }, };
	struct local_op_ref e11_opics[] = { { .oid = 13, .cid = 11 }, { .oid = 14, .cid =  5 }};
	struct mpo_graph_edge edge_list[] = {
		{ .vids = { 0, 0 }, .opics =  e0_opics, .nopics = ARRLEN( e0_opics) },
		{ .vids = { 0, 1 }, .opics =  e1_opics, .nopics = ARRLEN( e1_opics) },
		{ .vids = { 0, 0 }, .opics =  e2_opics, .nopics = ARRLEN( e2_opics) },
		{ .vids = { 1, 0 }, .opics =  e3_opics, .nopics = ARRLEN( e3_opics) },
		{ .vids = { 0, 0 }, .opics =  e4_opics, .nopics = ARRLEN( e4_opics) },
		{ .vids = { 0, 1 }, .opics =  e5_opics, .nopics = ARRLEN( e5_opics) },
		{ .vids = { 0, 0 }, .opics =  e6_opics, .nopics = ARRLEN( e6_opics) },
		{ .vids = { 1, 0 }, .opics =  e7_opics, .nopics = ARRLEN( e7_opics) },
		{ .vids = { 1, 1 }, .opics =  e8_opics, .nopics = ARRLEN( e8_opics) },
		{ .vids = { 0, 0 }, .opics =  e9_opics, .nopics = ARRLEN( e9_opics) },
		{ .vids = { 1, 0 }, .opics = e10_opics, .nopics = ARRLEN(e10_opics) },
		{ .vids = { 0, 0 }, .opics = e11_opics, .nopics = ARRLEN(e11_opics) },
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
		.verts     = graph_vertices,
		.edges     = graph_edges,
		.num_verts = graph_num_vertices,
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
	const dcomplex coeffmap[17] = { 0, 1, -1.6, 0.6, -1.2, -0.6, -0.3 - 0.1i, 0.7, -2.1, 0.5, -0.4 + 0.3i, 1.2, 0.4, 0.8, -0.2, 1.3, 0.9 + 0.5i };

	struct mpo_assembly assembly = {
		.graph         = graph,
		.opmap         = opmap,
		.coeffmap      = (dcomplex*)coeffmap,
		.qsite         = (qnumber*)qsite,
		.d             = d,
		.dtype         = CT_DOUBLE_COMPLEX,
		.num_local_ops = num_local_ops,
		.num_coeffs    = ARRLEN(coeffmap),
	};

	// construct MPO from an MPO assembly
	struct mpo mpo;
	mpo_from_assembly(&assembly, &mpo);

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
	mpo_graph_to_matrix(&graph, opmap, coeffmap, CT_DOUBLE_COMPLEX, &mat_ref);
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
	for (int i = 0; i < num_local_ops; i++) {
		delete_dense_tensor(&opmap[i]);
	}
	ct_free(opmap);

	H5Tclose(hdf5_dcomplex_id);
	H5Fclose(file);

	return 0;
}
