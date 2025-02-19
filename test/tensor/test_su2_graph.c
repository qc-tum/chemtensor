#include "su2_graph.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


char* test_su2_graph_to_fuse_split_tree()
{
	const int ndim = 11;

	// construct the fuse and split tree
	//
	//    4    3   0
	//     ╲  ╱   ╱
	//      ╲╱   ╱         fuse
	//      7╲  ╱
	//        ╲╱
	//        │
	//        │9
	//        │
	//        ╱╲
	//       ╱  ╲
	//      ╱    ╲         split
	//    8╱      ╲10
	//    ╱╲      ╱╲
	//   ╱  ╲    ╱  ╲
	//  5    1  6    2
	//
	struct su2_tree_node j0  = { .i_ax =  0, .c = { NULL, NULL } };
	struct su2_tree_node j1  = { .i_ax =  1, .c = { NULL, NULL } };
	struct su2_tree_node j2  = { .i_ax =  2, .c = { NULL, NULL } };
	struct su2_tree_node j3  = { .i_ax =  3, .c = { NULL, NULL } };
	struct su2_tree_node j4  = { .i_ax =  4, .c = { NULL, NULL } };
	struct su2_tree_node j5  = { .i_ax =  5, .c = { NULL, NULL } };
	struct su2_tree_node j6  = { .i_ax =  6, .c = { NULL, NULL } };
	struct su2_tree_node j7  = { .i_ax =  7, .c = { &j4,  &j3  } };
	struct su2_tree_node j8  = { .i_ax =  8, .c = { &j5,  &j1  } };
	struct su2_tree_node j10 = { .i_ax = 10, .c = { &j6,  &j2  } };
	struct su2_tree_node j9f = { .i_ax =  9, .c = { &j7,  &j0  } };
	struct su2_tree_node j9s = { .i_ax =  9, .c = { &j8,  &j10 } };

	struct su2_fuse_split_tree tree = { .tree_fuse = &j9f, .tree_split = &j9s, .ndim = ndim };

	if (!su2_fuse_split_tree_is_consistent(&tree)) {
		return "internal consistency check for the fuse and split tree failed";
	}

	// convert the tree to a graph
	struct su2_graph graph;
	su2_graph_from_fuse_split_tree(&tree, &graph);

	if (!su2_graph_is_consistent(&graph)) {
		return "internal consistency check for the SU(2) graph failed";
	}

	if (!su2_graph_has_fuse_split_tree_topology(&graph)) {
		return "SU(2) graph is expected to have fuse and split tree topology";
	}

	// convert back to tree
	struct su2_fuse_split_tree tree2;
	su2_graph_to_fuse_split_tree(&graph, &tree2);

	if (!su2_fuse_split_tree_is_consistent(&tree2)) {
		return "internal consistency check for the fuse and split tree failed";
	}

	if (!su2_fuse_split_tree_equal(&tree, &tree2)) {
		return "fuse and split tree after intermediate conversion to a graph does not agree with original tree";
	}

	delete_su2_fuse_split_tree(&tree2);
	delete_su2_graph(&graph);

	return 0;
}


char* test_su2_graph_yoga_to_simple_subtree()
{
	// construct the graph
	//
	//   4       2    6
	//    ╲       ╲  ╱
	//     ╲       ╲╱
	//      ╲      ╱3
	//       ╲    ╱
	//        ╲ 1╱╲
	//         ╲╱  ╲
	//         ╱    ╲
	//        ╱      ╲
	//       0        5
	//
	struct su2_graph_node nodes[] = {
		{ .eid_parent = 3, .eid_child = { 2, 6 }, .type = SU2_GRAPH_NODE_FUSE  },
		{ .eid_parent = 3, .eid_child = { 1, 5 }, .type = SU2_GRAPH_NODE_SPLIT },
		{ .eid_parent = 0, .eid_child = { 4, 1 }, .type = SU2_GRAPH_NODE_FUSE  },
	};
	struct su2_graph_edge edges[] = {
		{ .nid = {  2, -1 } },  // 0
		{ .nid = {  1,  2 } },  // 1
		{ .nid = { -1,  0 } },  // 2
		{ .nid = {  0,  1 } },  // 3
		{ .nid = { -1,  2 } },  // 4
		{ .nid = {  1, -1 } },  // 5
		{ .nid = { -1,  0 } },  // 6
	};

	struct su2_graph graph = {
		.nodes = nodes,
		.edges = edges,
		.num_nodes = ARRLEN(nodes),
		.num_edges = ARRLEN(edges),
	};

	if (!su2_graph_is_consistent(&graph)) {
		return "internal consistency check for the SU(2) symmetry graph failed";
	}

	if (su2_graph_has_fuse_split_tree_topology(&graph)) {
		return "SU(2) symmetry graph should not have fuse and split tree topology";
	}

	su2_graph_yoga_to_simple_subtree(&graph, 1);

	if (!su2_graph_is_consistent(&graph)) {
		return "internal consistency check for the SU(2) symmetry graph failed";
	}

	if (!su2_graph_has_fuse_split_tree_topology(&graph)) {
		return "SU(2) symmetry graph does not have expected fuse and split tree topology";
	}

	// reference fuse and split tree
	//
	//  4   2    6
	//   ╲   ╲  ╱
	//    ╲   ╲╱     fuse
	//     ╲  ╱3
	//      ╲╱
	//      │
	//     1│
	//      │
	//      ╱╲
	//     ╱  ╲      split
	//    0    5
	//
	struct su2_tree_node j0  = { .i_ax =  0, .c = { NULL, NULL } };
	struct su2_tree_node j2  = { .i_ax =  2, .c = { NULL, NULL } };
	struct su2_tree_node j4  = { .i_ax =  4, .c = { NULL, NULL } };
	struct su2_tree_node j5  = { .i_ax =  5, .c = { NULL, NULL } };
	struct su2_tree_node j6  = { .i_ax =  6, .c = { NULL, NULL } };
	struct su2_tree_node j3  = { .i_ax =  3, .c = { &j2,  &j6  } };
	struct su2_tree_node j1f = { .i_ax =  1, .c = { &j4,  &j3  } };
	struct su2_tree_node j1s = { .i_ax =  1, .c = { &j0,  &j5  } };

	struct su2_fuse_split_tree tree_ref = { .tree_fuse = &j1f, .tree_split = &j1s, .ndim = 7 };
	assert(su2_fuse_split_tree_is_consistent(&tree_ref));

	// convert graph to a tree
	struct su2_fuse_split_tree tree;
	su2_graph_to_fuse_split_tree(&graph, &tree);

	if (!su2_fuse_split_tree_equal(&tree, &tree_ref)) {
		return "converting a SU(2) symmetry graph does not result in the expected fuse and split tree";
	}

	delete_su2_fuse_split_tree(&tree);

	return 0;
}


char* test_su2_graph_connect()
{
	// construct the graph
	//
	//       6    2
	//        ╲  ╱
	//         ╲╱
	//         ╱5
	//        ╱   0
	//       ╱╲1 ╱
	//      ╱  ╲╱
	//     4    ╲
	//           ╲
	//            3
	//
	struct su2_graph_node nodes_f[] = {
		{ .eid_parent = 5, .eid_child = { 6, 2 }, .type = SU2_GRAPH_NODE_FUSE  },
		{ .eid_parent = 5, .eid_child = { 4, 1 }, .type = SU2_GRAPH_NODE_SPLIT },
		{ .eid_parent = 3, .eid_child = { 1, 0 }, .type = SU2_GRAPH_NODE_FUSE  },
	};
	struct su2_graph_edge edges_f[] = {
		{ .nid = { -1,  2 } },  // 0
		{ .nid = {  1,  2 } },  // 1
		{ .nid = { -1,  0 } },  // 2
		{ .nid = {  2, -1 } },  // 3
		{ .nid = {  1, -1 } },  // 4
		{ .nid = {  0,  1 } },  // 5
		{ .nid = { -1,  0 } },  // 6
	};

	struct su2_graph f = {
		.nodes = nodes_f,
		.edges = edges_f,
		.num_nodes = ARRLEN(nodes_f),
		.num_edges = ARRLEN(edges_f),
	};

	if (!su2_graph_is_consistent(&f)) {
		return "internal consistency check for the SU(2) symmetry graph failed";
	}

	// construct the graph
	//
	//              3
	//             ╱
	//       7    ╱
	//        ╲ 1╱╲
	//         ╲╱  ╲
	//         9╲   ╲0
	//           ╲   ╲
	//           ╱╲2 ╱
	//          ╱  ╲╱
	//         5   4╲
	//               ╲
	//               ╱╲
	//              ╱  ╲
	//             6    8
	//
	struct su2_graph_node nodes_g[] = {
		{ .eid_parent = 3, .eid_child = { 1, 0 }, .type = SU2_GRAPH_NODE_SPLIT },
		{ .eid_parent = 9, .eid_child = { 7, 1 }, .type = SU2_GRAPH_NODE_FUSE  },
		{ .eid_parent = 9, .eid_child = { 5, 2 }, .type = SU2_GRAPH_NODE_SPLIT },
		{ .eid_parent = 4, .eid_child = { 2, 0 }, .type = SU2_GRAPH_NODE_FUSE  },
		{ .eid_parent = 4, .eid_child = { 6, 8 }, .type = SU2_GRAPH_NODE_SPLIT },
	};
	struct su2_graph_edge edges_g[] = {
		{ .nid = {  0,  3 } },  // 0
		{ .nid = {  0,  1 } },  // 1
		{ .nid = {  2,  3 } },  // 2
		{ .nid = { -1,  0 } },  // 3
		{ .nid = {  3,  4 } },  // 4
		{ .nid = {  2, -1 } },  // 5
		{ .nid = {  4, -1 } },  // 6
		{ .nid = { -1,  1 } },  // 7
		{ .nid = {  4, -1 } },  // 8
		{ .nid = {  1,  2 } },  // 9
	};

	struct su2_graph g = {
		.nodes = nodes_g,
		.edges = edges_g,
		.num_nodes = ARRLEN(nodes_g),
		.num_edges = ARRLEN(edges_g),
	};

	if (!su2_graph_is_consistent(&g)) {
		return "internal consistency check for the SU(2) symmetry graph failed";
	}

	// connect the two graphs
	const int edge_map_f[] = { 4, 0, 1, 10, 9, 5, 7 };
	const int edge_map_g[] = { 6, 12, 13, 8, 2, 1, 11, 10, 4, 3 };
	struct su2_graph connected_graph;
	su2_graph_connect(&f, edge_map_f, &g, edge_map_g, &connected_graph);

	if (!su2_graph_is_consistent(&connected_graph)) {
		return "internal consistency check for the SU(2) symmetry graph failed";
	}

	if (connected_graph.num_edges != 14) {
		return "connected graph does not have number of expected edges";
	}

	// flipping f <-> g should result in the same connected graph
	struct su2_graph connected_graph_alt;
	su2_graph_connect(&g, edge_map_g, &f, edge_map_f, &connected_graph_alt);

	if (!su2_graph_is_consistent(&connected_graph_alt)) {
		return "internal consistency check for the SU(2) symmetry graph failed";
	}

	if (!su2_graph_equal(&connected_graph, &connected_graph_alt)) {
		return "connecting two graphs must be independent of ordering";
	}

	delete_su2_graph(&connected_graph_alt);
	delete_su2_graph(&connected_graph);

	return 0;
}
