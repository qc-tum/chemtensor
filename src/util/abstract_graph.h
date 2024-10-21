/// \file abstract_graph.h
/// \brief Abstract (symbolic) graph data structure and utility functions.

#pragma once

#include <stdbool.h>


//________________________________________________________________________________________________________________________
///
/// \brief Abstract undirected graph, defined by neighborhood map.
///
struct abstract_graph
{
	int** neighbor_map;  //!< neighborhood map (ordered list of neighbors of each node), defining undirected graph topology
	int* num_neighbors;  //!< number of neighbors for each node
	int num_nodes;       //!< number of nodes
};

void copy_abstract_graph(const struct abstract_graph* restrict src, struct abstract_graph* restrict dst);

void delete_abstract_graph(struct abstract_graph* graph);

bool abstract_graph_is_consistent(const struct abstract_graph* graph);


bool abstract_graph_equal(const struct abstract_graph* restrict graph0, const struct abstract_graph* restrict graph1);


bool abstract_graph_is_connected_tree(const struct abstract_graph* graph);


//________________________________________________________________________________________________________________________
///
/// \brief Temporary data structure for enumerating graph nodes and their distances to a designated root node.
///
struct graph_node_distance_tuple
{
	int i_node;    //!< node index
	int i_parent;  //!< parent node index
	int distance;  //!< distance from root
};

void enumerate_graph_node_distance_tuples(const struct abstract_graph* graph, const int i_root, struct graph_node_distance_tuple* tuples);
