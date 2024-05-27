#include <math.h>
#include "ttno.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


char* test_ttno_from_graph()
{
	hid_t file = H5Fopen("../test/operator/data/test_ttno_from_graph.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_ttno_from_graph failed";
	}

	// number of lattice sites
	const int nsites = 7;
	// local physical dimension
	const long d = 3;

	const qnumber qsite[3] = { 0, -1, 1 };

	// tree topology:
	//
	//  4           6
	//    \       /
	//      \   /
	//        0
	//        |
	//        |
	//  2 --- 3 --- 1
	//        |
	//        |
	//        5
	//
	int neigh0[] = { 3, 4, 6 };
	int neigh1[] = { 3 };
	int neigh2[] = { 3 };
	int neigh3[] = { 0, 1, 2, 5 };
	int neigh4[] = { 0 };
	int neigh5[] = { 3 };
	int neigh6[] = { 0 };
	int* neighbor_map[7] = {
		neigh0, neigh1, neigh2, neigh3, neigh4, neigh5, neigh6,
	};
	int num_neighbors[7] = {
		ARRLEN(neigh0), ARRLEN(neigh1), ARRLEN(neigh2), ARRLEN(neigh3), ARRLEN(neigh4), ARRLEN(neigh5), ARRLEN(neigh6),
	};
	struct abstract_graph topology = {
		.neighbor_map  = neighbor_map,
		.num_neighbors = num_neighbors,
		.num_nodes     = nsites,
	};

	// vertex groups, corresponding to virtual bonds and indexed by site indices (i, j) with i < j
	int v03_0_eids_0[] = { 0, 2, 3 };  int v03_0_eids_1[] = { 1, 2 };
	int v03_1_eids_0[] = { 1 };        int v03_1_eids_1[] = { 0 };
	struct ttno_graph_vertex vl03[] = {
		{ .eids = { v03_0_eids_0, v03_0_eids_1 }, .num_edges = { ARRLEN(v03_0_eids_0), ARRLEN(v03_0_eids_1) }, .qnum = -1 },
		{ .eids = { v03_1_eids_0, v03_1_eids_1 }, .num_edges = { ARRLEN(v03_1_eids_0), ARRLEN(v03_1_eids_1) }, .qnum =  0 },
	};
	int v04_0_eids_0[] = { 1, 2 };     int v04_0_eids_1[] = { 0 };
	int v04_1_eids_0[] = { 0, 3 };     int v04_1_eids_1[] = { 1 };
	struct ttno_graph_vertex vl04[] = {
		{ .eids = { v04_0_eids_0, v04_0_eids_1 }, .num_edges = { ARRLEN(v04_0_eids_0), ARRLEN(v04_0_eids_1) }, .qnum =  1 },
		{ .eids = { v04_1_eids_0, v04_1_eids_1 }, .num_edges = { ARRLEN(v04_1_eids_0), ARRLEN(v04_1_eids_1) }, .qnum =  0 },
	};
	int v06_0_eids_0[] = { 0, 2 };     int v06_0_eids_1[] = { 1, 3 };
	int v06_1_eids_0[] = { 3 };        int v06_1_eids_1[] = { 0 };
	int v06_2_eids_0[] = { 1 };        int v06_2_eids_1[] = { 2 };
	struct ttno_graph_vertex vl06[] = {
		{ .eids = { v06_0_eids_0, v06_0_eids_1 }, .num_edges = { ARRLEN(v06_0_eids_0), ARRLEN(v06_0_eids_1) }, .qnum =  0 },
		{ .eids = { v06_1_eids_0, v06_1_eids_1 }, .num_edges = { ARRLEN(v06_1_eids_0), ARRLEN(v06_1_eids_1) }, .qnum = -1 },
		{ .eids = { v06_2_eids_0, v06_2_eids_1 }, .num_edges = { ARRLEN(v06_2_eids_0), ARRLEN(v06_2_eids_1) }, .qnum =  1 },
	};
	int v13_0_eids_0[] = { 0, 2 };     int v13_0_eids_1[] = { 0 };
	int v13_1_eids_0[] = { 1 };        int v13_1_eids_1[] = { 1, 2 };
	struct ttno_graph_vertex vl13[] = {
		{ .eids = { v13_0_eids_0, v13_0_eids_1 }, .num_edges = { ARRLEN(v13_0_eids_0), ARRLEN(v13_0_eids_1) }, .qnum =  1 },
		{ .eids = { v13_1_eids_0, v13_1_eids_1 }, .num_edges = { ARRLEN(v13_1_eids_0), ARRLEN(v13_1_eids_1) }, .qnum =  0 },
	};
	int v23_0_eids_0[] = { 1 };        int v23_0_eids_1[] = { 0 };
	int v23_1_eids_0[] = { 0 };        int v23_1_eids_1[] = { 1, 2 };
	struct ttno_graph_vertex vl23[] = {
		{ .eids = { v23_0_eids_0, v23_0_eids_1 }, .num_edges = { ARRLEN(v23_0_eids_0), ARRLEN(v23_0_eids_1) }, .qnum = -1 },
		{ .eids = { v23_1_eids_0, v23_1_eids_1 }, .num_edges = { ARRLEN(v23_1_eids_0), ARRLEN(v23_1_eids_1) }, .qnum =  0 },
	};
	int v35_0_eids_0[] = { 0, 1, 2 };  int v35_0_eids_1[] = { 0 };
	struct ttno_graph_vertex vl35[] = {
		{ .eids = { v35_0_eids_0, v35_0_eids_1 }, .num_edges = { ARRLEN(v35_0_eids_0), ARRLEN(v35_0_eids_1) }, .qnum =  0 },
	};
	struct ttno_graph_vertex* vertices[7 * 7] = {
		// 0     1     2     3     4     5     6
		   NULL, NULL, NULL, vl03, vl04, NULL, vl06,  // 0
		   NULL, NULL, NULL, vl13, NULL, NULL, NULL,  // 1
		   NULL, NULL, NULL, vl23, NULL, NULL, NULL,  // 2
		   NULL, NULL, NULL, NULL, NULL, vl35, NULL,  // 3
		   NULL, NULL, NULL, NULL, NULL, NULL, NULL,  // 4
		   NULL, NULL, NULL, NULL, NULL, NULL, NULL,  // 5
		   NULL, NULL, NULL, NULL, NULL, NULL, NULL,  // 6
	};
	int num_vertices[7 * 7] = {
		// 0             1             2             3             4             5             6
		   0,            0,            0,            ARRLEN(vl03), ARRLEN(vl04), 0,            ARRLEN(vl06),  // 0
		   0,            0,            0,            ARRLEN(vl13), 0,            0,            0,             // 1
		   0,            0,            0,            ARRLEN(vl23), 0,            0,            0,             // 2
		   0,            0,            0,            0,            0,            ARRLEN(vl35), 0,             // 3
		   0,            0,            0,            0,            0,            0,            0,             // 4
		   0,            0,            0,            0,            0,            0,            0,             // 5
		   0,            0,            0,            0,            0,            0,            0,             // 6
	};

	// edges
	struct local_op_ref e0_0_opics[] = { { .oid =  0, .coeff =  0.7 }, };
	struct local_op_ref e0_1_opics[] = { { .oid =  4, .coeff = -1.2 }, { .oid =  1, .coeff =  0.4 }, };
	struct local_op_ref e0_2_opics[] = { { .oid =  5, .coeff =  0.6 }, { .oid =  6, .coeff = -1.1 }, };
	struct local_op_ref e0_3_opics[] = { { .oid =  7, .coeff = -0.1 }, };
	int e0_0_vids[] = { 0, 1, 0 };
	int e0_1_vids[] = { 1, 0, 2 };
	int e0_2_vids[] = { 0, 0, 0 };
	int e0_3_vids[] = { 0, 1, 1 };
	struct ttno_graph_hyperedge el0[] = {
		{ .vids = e0_0_vids, .order = ARRLEN(e0_0_vids), .opics = e0_0_opics, .nopics = ARRLEN(e0_0_opics) },
		{ .vids = e0_1_vids, .order = ARRLEN(e0_1_vids), .opics = e0_1_opics, .nopics = ARRLEN(e0_1_opics) },
		{ .vids = e0_2_vids, .order = ARRLEN(e0_2_vids), .opics = e0_2_opics, .nopics = ARRLEN(e0_2_opics) },
		{ .vids = e0_3_vids, .order = ARRLEN(e0_3_vids), .opics = e0_3_opics, .nopics = ARRLEN(e0_3_opics) },
	};
	struct local_op_ref e1_0_opics[] = { { .oid =  8, .coeff =  0.6 }, };
	struct local_op_ref e1_1_opics[] = { { .oid =  9, .coeff = -0.7 }, };
	struct local_op_ref e1_2_opics[] = { { .oid = 10, .coeff =  0.2 }, { .oid = 11, .coeff =  0.8 }, };
	int e1_0_vids[] = { 0 };
	int e1_1_vids[] = { 1 };
	int e1_2_vids[] = { 0 };
	struct ttno_graph_hyperedge el1[] = {
		{ .vids = e1_0_vids, .order = ARRLEN(e1_0_vids), .opics = e1_0_opics, .nopics = ARRLEN(e1_0_opics) },
		{ .vids = e1_1_vids, .order = ARRLEN(e1_1_vids), .opics = e1_1_opics, .nopics = ARRLEN(e1_1_opics) },
		{ .vids = e1_2_vids, .order = ARRLEN(e1_2_vids), .opics = e1_2_opics, .nopics = ARRLEN(e1_2_opics) },
	};
	struct local_op_ref e2_0_opics[] = { { .oid = 12, .coeff =  0.4 }, };
	struct local_op_ref e2_1_opics[] = { { .oid = 13, .coeff = -1.3 }, };
	int e2_0_vids[] = { 1 };
	int e2_1_vids[] = { 0 };
	struct ttno_graph_hyperedge el2[] = {
		{ .vids = e2_0_vids, .order = ARRLEN(e2_0_vids), .opics = e2_0_opics, .nopics = ARRLEN(e2_0_opics) },
		{ .vids = e2_1_vids, .order = ARRLEN(e2_1_vids), .opics = e2_1_opics, .nopics = ARRLEN(e2_1_opics) },
	};
	struct local_op_ref e3_0_opics[] = { { .oid = 14, .coeff =  0.5 }, };
	struct local_op_ref e3_1_opics[] = { { .oid = 10, .coeff =  1.8 }, };
	struct local_op_ref e3_2_opics[] = { { .oid = 15, .coeff = -0.1 }, { .oid = 16, .coeff = -0.9 }, };
	int e3_0_vids[] = { 1, 0, 0, 0 };
	int e3_1_vids[] = { 0, 1, 1, 0 };
	int e3_2_vids[] = { 0, 1, 1, 0 };
	struct ttno_graph_hyperedge el3[] = {
		{ .vids = e3_0_vids, .order = ARRLEN(e3_0_vids), .opics = e3_0_opics, .nopics = ARRLEN(e3_0_opics) },
		{ .vids = e3_1_vids, .order = ARRLEN(e3_1_vids), .opics = e3_1_opics, .nopics = ARRLEN(e3_1_opics) },
		{ .vids = e3_2_vids, .order = ARRLEN(e3_2_vids), .opics = e3_2_opics, .nopics = ARRLEN(e3_2_opics) },
	};
	struct local_op_ref e4_0_opics[] = { { .oid = 17, .coeff = -0.3 }, };
	struct local_op_ref e4_1_opics[] = { { .oid = 14, .coeff =  0.7 }, };
	int e4_0_vids[] = { 0 };
	int e4_1_vids[] = { 1 };
	struct ttno_graph_hyperedge el4[] = {
		{ .vids = e4_0_vids, .order = ARRLEN(e4_0_vids), .opics = e4_0_opics, .nopics = ARRLEN(e4_0_opics) },
		{ .vids = e4_1_vids, .order = ARRLEN(e4_1_vids), .opics = e4_1_opics, .nopics = ARRLEN(e4_1_opics) },
	};
	struct local_op_ref e5_0_opics[] = { { .oid = 18, .coeff =  1.0 }, { .oid = 19, .coeff = -0.4 }, { .oid = 20, .coeff =  0.7 }, };
	int e5_0_vids[] = { 0 };
	struct ttno_graph_hyperedge el5[] = {
		{ .vids = e5_0_vids, .order = ARRLEN(e5_0_vids), .opics = e5_0_opics, .nopics = ARRLEN(e5_0_opics) },
	};
	struct local_op_ref e6_0_opics[] = { { .oid = 21, .coeff =  0.5 }, };
	struct local_op_ref e6_1_opics[] = { { .oid = 22, .coeff =  0.3 }, };
	struct local_op_ref e6_2_opics[] = { { .oid = 23, .coeff = -0.2 }, { .oid = 17, .coeff =  0.9 }, };
	struct local_op_ref e6_3_opics[] = { { .oid = 24, .coeff = -1.4 }, };
	int e6_0_vids[] = { 1 };
	int e6_1_vids[] = { 0 };
	int e6_2_vids[] = { 2 };
	int e6_3_vids[] = { 0 };
	struct ttno_graph_hyperedge el6[] = {
		{ .vids = e6_0_vids, .order = ARRLEN(e6_0_vids), .opics = e6_0_opics, .nopics = ARRLEN(e6_0_opics) },
		{ .vids = e6_1_vids, .order = ARRLEN(e6_1_vids), .opics = e6_1_opics, .nopics = ARRLEN(e6_1_opics) },
		{ .vids = e6_2_vids, .order = ARRLEN(e6_2_vids), .opics = e6_2_opics, .nopics = ARRLEN(e6_2_opics) },
		{ .vids = e6_3_vids, .order = ARRLEN(e6_3_vids), .opics = e6_3_opics, .nopics = ARRLEN(e6_3_opics) },
	};
	struct ttno_graph_hyperedge* edges[7] = {
		el0, el1, el2, el3, el4, el5, el6,
	};
	int num_edges[7] = {
		ARRLEN(el0), ARRLEN(el1), ARRLEN(el2), ARRLEN(el3), ARRLEN(el4), ARRLEN(el5), ARRLEN(el6),
	};

	// construct the TTNO graph
	struct ttno_graph graph = {
		.topology  = topology,
		.verts     = vertices,
		.edges     = edges,
		.num_verts = num_vertices,
		.num_edges = num_edges,
		.nsites    = nsites,
	};
	if (!ttno_graph_is_consistent(&graph)) {
		return "internal TTNO graph construction is inconsistent";
	}

	// read local operator map from disk
	const int num_local_ops = 25;
	struct dense_tensor opmap_tensor;
	const long dim_opmt[3] = { num_local_ops, d, d };
	allocate_dense_tensor(DOUBLE_REAL, 3, dim_opmt, &opmap_tensor);
	// read values from disk
	if (read_hdf5_dataset(file, "opmap", H5T_NATIVE_DOUBLE, opmap_tensor.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	// copy individual operators
	struct dense_tensor* opmap = aligned_alloc(MEM_DATA_ALIGN, num_local_ops * sizeof(struct dense_tensor));
	for (int i = 0; i < num_local_ops; i++)
	{
		const long dim[2] = { d, d };
		allocate_dense_tensor(DOUBLE_REAL, 2, dim, &opmap[i]);
		const double* data = opmap_tensor.data;
		memcpy(opmap[i].data, &data[i * d*d], d*d * sizeof(double));
	}
	delete_dense_tensor(&opmap_tensor);

	// construct a TTNO from a TTNO graph
	struct ttno ttno;
	ttno_from_graph(DOUBLE_REAL, d, qsite, &graph, opmap, &ttno);

	if (!ttno_is_consistent(&ttno)) {
		return "internal TTNO consistency check failed";
	}

	// convert to a matrix
	struct block_sparse_tensor mat;
	ttno_to_matrix(&ttno, &mat);
	// convert to dense matrix (for comparison with reference matrix)
	struct dense_tensor mat_dns;
	block_sparse_to_dense_tensor(&mat, &mat_dns);

	// convert TTNO graph to a (dense) matrix, as reference
	struct dense_tensor mat_ref;
	ttno_graph_to_matrix(&graph, opmap, &mat_ref);

	// compare
	if (!dense_tensor_allclose(&mat_dns, &mat_ref, 1e-13)) {
		return "matrix representation of TTNO does not match corresponding matrix obtained from TTNO graph";
	}

	delete_dense_tensor(&mat_ref);
	delete_dense_tensor(&mat_dns);
	delete_block_sparse_tensor(&mat);
	delete_ttno(&ttno);
	for (int i = 0; i < num_local_ops; i++) {
		delete_dense_tensor(&opmap[i]);
	}
	aligned_free(opmap);

	H5Fclose(file);

	return 0;
}
