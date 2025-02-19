#include <math.h>
#include "ttno.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


char* test_ttno_from_assembly()
{
	hid_t file = H5Fopen("../test/operator/data/test_ttno_from_assembly.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_ttno_from_assembly failed";
	}

	// number of physical and branching lattice sites
	const int nsites_physical  = 6;
	const int nsites_branching = 2;
	// local physical dimension
	const long d = 3;

	const qnumber qsite[3] = { 0, -1, 1 };

	// tree topology:
	//
	//     4     0
	//      ╲   ╱
	//       ╲ ╱
	//        6
	//        │
	//        │
	//  2 ─── 5 ─── 1 ─── 7
	//        │
	//        │
	//        3
	//
	int neigh0[] = { 6 };
	int neigh1[] = { 5, 7 };
	int neigh2[] = { 5 };
	int neigh3[] = { 5 };
	int neigh4[] = { 6 };
	int neigh5[] = { 1, 2, 3, 6 };
	int neigh6[] = { 0, 4, 5 };
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
		.num_nodes     = nsites_physical + nsites_branching,
	};

	// vertex groups, corresponding to virtual bonds and indexed by site indices (i, j) with i < j
	int v06_0_eids_0[] = { 1, 3 };     int v06_0_eids_1[] = { 2, 3 };
	int v06_1_eids_0[] = { 0 };        int v06_1_eids_1[] = { 1 };
	int v06_2_eids_0[] = { 2 };        int v06_2_eids_1[] = { 0 };
	struct ttno_graph_vertex vl06[] = {
		{ .eids = { v06_0_eids_0, v06_0_eids_1 }, .num_edges = { ARRLEN(v06_0_eids_0), ARRLEN(v06_0_eids_1) }, .qnum =  0 },
		{ .eids = { v06_1_eids_0, v06_1_eids_1 }, .num_edges = { ARRLEN(v06_1_eids_0), ARRLEN(v06_1_eids_1) }, .qnum =  1 },
		{ .eids = { v06_2_eids_0, v06_2_eids_1 }, .num_edges = { ARRLEN(v06_2_eids_0), ARRLEN(v06_2_eids_1) }, .qnum = -1 },
	};
	int v15_0_eids_0[] = { 0, 2 };     int v15_0_eids_1[] = { 0 };
	int v15_1_eids_0[] = { 1 };        int v15_1_eids_1[] = { 1, 2 };
	struct ttno_graph_vertex vl15[] = {
		{ .eids = { v15_0_eids_0, v15_0_eids_1 }, .num_edges = { ARRLEN(v15_0_eids_0), ARRLEN(v15_0_eids_1) }, .qnum =  1 },
		{ .eids = { v15_1_eids_0, v15_1_eids_1 }, .num_edges = { ARRLEN(v15_1_eids_0), ARRLEN(v15_1_eids_1) }, .qnum =  0 },
	};
	int v17_0_eids_0[] = { 0, 1, 2 };  int v17_0_eids_1[] = { 0, 1 };
	struct ttno_graph_vertex vl17[] = {
		{ .eids = { v17_0_eids_0, v17_0_eids_1 }, .num_edges = { ARRLEN(v17_0_eids_0), ARRLEN(v17_0_eids_1) }, .qnum =  0 },
	};
	int v25_0_eids_0[] = { 1 };        int v25_0_eids_1[] = { 0 };
	int v25_1_eids_0[] = { 0 };        int v25_1_eids_1[] = { 1, 2 };
	struct ttno_graph_vertex vl25[] = {
		{ .eids = { v25_0_eids_0, v25_0_eids_1 }, .num_edges = { ARRLEN(v25_0_eids_0), ARRLEN(v25_0_eids_1) }, .qnum = -1 },
		{ .eids = { v25_1_eids_0, v25_1_eids_1 }, .num_edges = { ARRLEN(v25_1_eids_0), ARRLEN(v25_1_eids_1) }, .qnum =  0 },
	};
	int v35_0_eids_0[] = { 0 };        int v35_0_eids_1[] = { 0, 1, 2 };
	struct ttno_graph_vertex vl35[] = {
		{ .eids = { v35_0_eids_0, v35_0_eids_1 }, .num_edges = { ARRLEN(v35_0_eids_0), ARRLEN(v35_0_eids_1) }, .qnum =  0 },
	};
	int v46_0_eids_0[] = { 0 };        int v46_0_eids_1[] = { 1, 2 };
	int v46_1_eids_0[] = { 1 };        int v46_1_eids_1[] = { 0, 3 };
	struct ttno_graph_vertex vl46[] = {
		{ .eids = { v46_0_eids_0, v46_0_eids_1 }, .num_edges = { ARRLEN(v46_0_eids_0), ARRLEN(v46_0_eids_1) }, .qnum = -1 },
		{ .eids = { v46_1_eids_0, v46_1_eids_1 }, .num_edges = { ARRLEN(v46_1_eids_0), ARRLEN(v46_1_eids_1) }, .qnum =  0 },
	};
	int v56_0_eids_0[] = { 1, 2 };     int v56_0_eids_1[] = { 0, 2 };
	int v56_1_eids_0[] = { 0 };        int v56_1_eids_1[] = { 1, 3 };
	struct ttno_graph_vertex vl56[] = {
		{ .eids = { v56_0_eids_0, v56_0_eids_1 }, .num_edges = { ARRLEN(v56_0_eids_0), ARRLEN(v56_0_eids_1) }, .qnum =  1 },
		{ .eids = { v56_1_eids_0, v56_1_eids_1 }, .num_edges = { ARRLEN(v56_1_eids_0), ARRLEN(v56_1_eids_1) }, .qnum =  0 },
	};
	struct ttno_graph_vertex* vertices[8 * 8] = {
		// 0     1     2     3     4     5     6     7
		   NULL, NULL, NULL, NULL, NULL, NULL, vl06, NULL,  // 0
		   NULL, NULL, NULL, NULL, NULL, vl15, NULL, vl17,  // 1
		   NULL, NULL, NULL, NULL, NULL, vl25, NULL, NULL,  // 2
		   NULL, NULL, NULL, NULL, NULL, vl35, NULL, NULL,  // 3
		   NULL, NULL, NULL, NULL, NULL, NULL, vl46, NULL,  // 4
		   NULL, NULL, NULL, NULL, NULL, NULL, vl56, NULL,  // 5
		   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,  // 6
		   NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,  // 7
	};
	int num_vertices[8 * 8] = {
		// 0             1             2             3             4             5             6             7
		   0,            0,            0,            0,            0,            0,            ARRLEN(vl06), 0,             // 0
		   0,            0,            0,            0,            0,            ARRLEN(vl15), 0,            ARRLEN(vl17),  // 1
		   0,            0,            0,            0,            0,            ARRLEN(vl25), 0,            0,             // 2
		   0,            0,            0,            0,            0,            ARRLEN(vl35), 0,            0,             // 3
		   0,            0,            0,            0,            0,            0,            ARRLEN(vl46), 0,             // 4
		   0,            0,            0,            0,            0,            0,            ARRLEN(vl56), 0,             // 5
		   0,            0,            0,            0,            0,            0,            0,            0,             // 6
		   0,            0,            0,            0,            0,            0,            0,            0,             // 7
	};

	// edges
	struct local_op_ref e0_0_opics[] = { { .oid =  4, .cid = 15 }, };
	struct local_op_ref e0_1_opics[] = { { .oid = 22, .cid = 17 }, };
	struct local_op_ref e0_2_opics[] = { { .oid =  7, .cid =  5 }, { .oid = 17, .cid = 16 }, };
	struct local_op_ref e0_3_opics[] = { { .oid = 24, .cid = 19 }, };
	int e0_0_vids[] = { 1 };
	int e0_1_vids[] = { 0 };
	int e0_2_vids[] = { 2 };
	int e0_3_vids[] = { 0 };
	struct ttno_graph_hyperedge el0[] = {
		{ .vids = e0_0_vids, .order = ARRLEN(e0_0_vids), .opics = e0_0_opics, .nopics = ARRLEN(e0_0_opics) },
		{ .vids = e0_1_vids, .order = ARRLEN(e0_1_vids), .opics = e0_1_opics, .nopics = ARRLEN(e0_1_opics) },
		{ .vids = e0_2_vids, .order = ARRLEN(e0_2_vids), .opics = e0_2_opics, .nopics = ARRLEN(e0_2_opics) },
		{ .vids = e0_3_vids, .order = ARRLEN(e0_3_vids), .opics = e0_3_opics, .nopics = ARRLEN(e0_3_opics) },
	};
	struct local_op_ref e1_0_opics[] = { { .oid =  8, .cid =  9 }, };
	struct local_op_ref e1_1_opics[] = { { .oid =  9, .cid =  2 }, };
	struct local_op_ref e1_2_opics[] = { { .oid = 10, .cid = 14 }, { .oid = 11, .cid =  4 }, };
	int e1_0_vids[] = { 0, 0 };
	int e1_1_vids[] = { 1, 0 };
	int e1_2_vids[] = { 0, 0 };
	struct ttno_graph_hyperedge el1[] = {
		{ .vids = e1_0_vids, .order = ARRLEN(e1_0_vids), .opics = e1_0_opics, .nopics = ARRLEN(e1_0_opics) },
		{ .vids = e1_1_vids, .order = ARRLEN(e1_1_vids), .opics = e1_1_opics, .nopics = ARRLEN(e1_1_opics) },
		{ .vids = e1_2_vids, .order = ARRLEN(e1_2_vids), .opics = e1_2_opics, .nopics = ARRLEN(e1_2_opics) },
	};
	struct local_op_ref e2_0_opics[] = { { .oid = 12, .cid = 10 }, };
	struct local_op_ref e2_1_opics[] = { { .oid = 13, .cid =  6 }, };
	int e2_0_vids[] = { 1 };
	int e2_1_vids[] = { 0 };
	struct ttno_graph_hyperedge el2[] = {
		{ .vids = e2_0_vids, .order = ARRLEN(e2_0_vids), .opics = e2_0_opics, .nopics = ARRLEN(e2_0_opics) },
		{ .vids = e2_1_vids, .order = ARRLEN(e2_1_vids), .opics = e2_1_opics, .nopics = ARRLEN(e2_1_opics) },
	};
	struct local_op_ref e3_0_opics[] = { { .oid = 18, .cid =  1 }, { .oid = 19, .cid = 18 }, { .oid = 20, .cid = 11 }, };
	int e3_0_vids[] = { 0 };
	struct ttno_graph_hyperedge el3[] = {
		{ .vids = e3_0_vids, .order = ARRLEN(e3_0_vids), .opics = e3_0_opics, .nopics = ARRLEN(e3_0_opics) },
	};
	struct local_op_ref e4_0_opics[] = { { .oid = 17, .cid =  8 }, };
	struct local_op_ref e4_1_opics[] = { { .oid = 14, .cid = 11 }, };
	int e4_0_vids[] = { 0 };
	int e4_1_vids[] = { 1 };
	struct ttno_graph_hyperedge el4[] = {
		{ .vids = e4_0_vids, .order = ARRLEN(e4_0_vids), .opics = e4_0_opics, .nopics = ARRLEN(e4_0_opics) },
		{ .vids = e4_1_vids, .order = ARRLEN(e4_1_vids), .opics = e4_1_opics, .nopics = ARRLEN(e4_1_opics) },
	};
	struct local_op_ref e5_0_opics[] = { { .oid = 14, .cid = 15 }, };
	struct local_op_ref e5_1_opics[] = { { .oid = 10, .cid = 13 }, };
	struct local_op_ref e5_2_opics[] = { { .oid = 15, .cid =  3 }, { .oid = 16, .cid = 20 }, };
	int e5_0_vids[] = { 0, 0, 0, 1 };
	int e5_1_vids[] = { 1, 1, 0, 0 };
	int e5_2_vids[] = { 1, 1, 0, 0 };
	struct ttno_graph_hyperedge el5[] = {
		{ .vids = e5_0_vids, .order = ARRLEN(e5_0_vids), .opics = e5_0_opics, .nopics = ARRLEN(e5_0_opics) },
		{ .vids = e5_1_vids, .order = ARRLEN(e5_1_vids), .opics = e5_1_opics, .nopics = ARRLEN(e5_1_opics) },
		{ .vids = e5_2_vids, .order = ARRLEN(e5_2_vids), .opics = e5_2_opics, .nopics = ARRLEN(e5_2_opics) },
	};
	struct local_op_ref e6_0_opics[] = { { .oid = OID_NOP, .cid = 11 }, };
	struct local_op_ref e6_1_opics[] = { { .oid = OID_NOP, .cid =  7 }, { .oid = OID_NOP, .cid = 10 }, };
	struct local_op_ref e6_2_opics[] = { { .oid = OID_NOP, .cid =  9 }, { .oid = OID_NOP, .cid = 12 }, };
	struct local_op_ref e6_3_opics[] = { { .oid = OID_NOP, .cid =  3 }, };
	int e6_0_vids[] = { 2, 1, 0 };
	int e6_1_vids[] = { 1, 0, 1 };
	int e6_2_vids[] = { 0, 0, 0 };
	int e6_3_vids[] = { 0, 1, 1 };
	struct ttno_graph_hyperedge el6[] = {
		{ .vids = e6_0_vids, .order = ARRLEN(e6_0_vids), .opics = e6_0_opics, .nopics = ARRLEN(e6_0_opics) },
		{ .vids = e6_1_vids, .order = ARRLEN(e6_1_vids), .opics = e6_1_opics, .nopics = ARRLEN(e6_1_opics) },
		{ .vids = e6_2_vids, .order = ARRLEN(e6_2_vids), .opics = e6_2_opics, .nopics = ARRLEN(e6_2_opics) },
		{ .vids = e6_3_vids, .order = ARRLEN(e6_3_vids), .opics = e6_3_opics, .nopics = ARRLEN(e6_3_opics) },
	};
	struct local_op_ref e7_0_opics[] = { { .oid = OID_NOP, .cid = 21 }, };
	struct local_op_ref e7_1_opics[] = { { .oid = OID_NOP, .cid =  7 }, };
	int e7_0_vids[] = { 0 };
	int e7_1_vids[] = { 0 };
	struct ttno_graph_hyperedge el7[] = {
		{ .vids = e7_0_vids, .order = ARRLEN(e7_0_vids), .opics = e7_0_opics, .nopics = ARRLEN(e7_0_opics) },
		{ .vids = e7_1_vids, .order = ARRLEN(e7_1_vids), .opics = e7_1_opics, .nopics = ARRLEN(e7_1_opics) },
	};
	struct ttno_graph_hyperedge* edges[8] = {
		el0, el1, el2, el3, el4, el5, el6, el7,
	};
	int num_edges[8] = {
		ARRLEN(el0), ARRLEN(el1), ARRLEN(el2), ARRLEN(el3), ARRLEN(el4), ARRLEN(el5), ARRLEN(el6), ARRLEN(el7),
	};

	// construct the TTNO graph
	struct ttno_graph graph = {
		.topology         = topology,
		.verts            = vertices,
		.edges            = edges,
		.num_verts        = num_vertices,
		.num_edges        = num_edges,
		.nsites_physical  = nsites_physical,
		.nsites_branching = nsites_branching,
	};
	if (!ttno_graph_is_consistent(&graph)) {
		return "internal TTNO graph construction is inconsistent";
	}

	// read local operator map from disk
	const int num_local_ops = 25;
	struct dense_tensor opmap_tensor;
	const long dim_opmt[3] = { num_local_ops, d, d };
	allocate_dense_tensor(CT_DOUBLE_REAL, 3, dim_opmt, &opmap_tensor);
	// read values from disk
	if (read_hdf5_dataset(file, "opmap", H5T_NATIVE_DOUBLE, opmap_tensor.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	// copy individual operators
	struct dense_tensor* opmap = ct_malloc(num_local_ops * sizeof(struct dense_tensor));
	for (int i = 0; i < num_local_ops; i++)
	{
		const long dim[2] = { d, d };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, &opmap[i]);
		const double* data = opmap_tensor.data;
		memcpy(opmap[i].data, &data[i * d*d], d*d * sizeof(double));
	}
	delete_dense_tensor(&opmap_tensor);

	// coefficient map; first two entries must always be 0 and 1
	const double coeffmap[] = { 0, 1, -0.7, -0.1, 0.8, -0.2, -1.3, -1.2, -0.3, 0.6, 0.4, 0.7, -1.1, 1.8, 0.2, 0.5, 0.9, 0.3, -0.4, -1.4, -0.9, 1.5 };

	struct ttno_assembly assembly = {
		.graph         = graph,
		.opmap         = opmap,
		.coeffmap      = (double*)coeffmap,
		.qsite         = (qnumber*)qsite,
		.d             = d,
		.dtype         = CT_DOUBLE_REAL,
		.num_local_ops = num_local_ops,
		.num_coeffs    = ARRLEN(coeffmap),
	};

	// construct a TTNO from a TTNO assembly
	struct ttno ttno;
	ttno_from_assembly(&assembly, &ttno);

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
	ttno_graph_to_matrix(&graph, opmap, coeffmap, &mat_ref);

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
	ct_free(opmap);

	H5Fclose(file);

	return 0;
}
