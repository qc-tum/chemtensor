/// \file su2_graph.h
/// \brief Internal temporary graph data structure for SU(2) symmetric tensors.

#pragma once

#include <stdbool.h>
#include "su2_tree.h"


//________________________________________________________________________________________________________________________
///
/// \brief SU(2) symmetry graph node type.
///
enum su2_graph_node_type
{
	SU2_GRAPH_NODE_FUSE  = 0,  //!< fusion node
	SU2_GRAPH_NODE_SPLIT = 1,  //!< splitting node
};


//________________________________________________________________________________________________________________________
///
/// \brief SU(2) symmetry graph node.
///
/// Convention for orientations:
///
/// fusion node:
///  c[0]  c[1]
///      ╲╱
///      │
///      p
///
/// splitting node:
///      p
///      │
///      ╱╲
///  c[0]  c[1]
///
struct su2_graph_node
{
	int eid_parent;                 //!< index of parent edge
	int eid_child[2];               //!< indices of left and right child edges
	enum su2_graph_node_type type;  //!< node type (fusion or splitting)
};


//________________________________________________________________________________________________________________________
///
/// \brief SU(2) symmetry graph edge.
///
struct su2_graph_edge
{
	int nid[2];  //!< index of connected upstream and downstream node, or -1 if not connected
};


//________________________________________________________________________________________________________________________
///
/// \brief SU(2) symmetry graph: temporary data structure for representing the structural tensor.
///
/// The graph can contain loops.
///
struct su2_graph
{
	struct su2_graph_node* nodes;  //!< nodes
	struct su2_graph_edge* edges;  //!< edges; array indices correspond to logical axis indices
	int num_nodes;                 //!< overall number of nodes
	int num_edges;                 //!< overall number of edges (corresponds to logical number of dimensions)
};

void copy_su2_graph(const struct su2_graph* src, struct su2_graph* dst);

void delete_su2_graph(struct su2_graph* graph);

bool su2_graph_is_consistent(const struct su2_graph* graph);

bool su2_graph_equal(const struct su2_graph* restrict f, const struct su2_graph* restrict g);


//________________________________________________________________________________________________________________________
//

// conversion from and to a tree

void su2_graph_from_fuse_split_tree(const struct su2_fuse_split_tree* tree, struct su2_graph* graph);

bool su2_graph_has_fuse_split_tree_topology(const struct su2_graph* graph);

void su2_graph_to_fuse_split_tree(const struct su2_graph* graph, struct su2_fuse_split_tree* tree);


//________________________________________________________________________________________________________________________
//

// elementary yoga subtrees

bool su2_graph_is_yoga_edge(const struct su2_graph* graph, const int eid);

void su2_graph_yoga_to_simple_subtree(struct su2_graph* graph, const int eid);


//________________________________________________________________________________________________________________________
//

// connecting two graphs

void su2_graph_connect(
	const struct su2_graph* restrict f, const int* restrict edge_map_f,
	const struct su2_graph* restrict g, const int* restrict edge_map_g,
	struct su2_graph* restrict connected_graph);
