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


bool abstract_graph_is_connected_tree(const struct abstract_graph* graph);
