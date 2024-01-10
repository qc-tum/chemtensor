/// \file bipartite_graph.h
/// \brief Bipartite graph utility functions, including an implementation of the Hopcroft-Karp algorithm based on https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm

#pragma once

#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Data structure representing a bipartite graph G = ((U, V), E),
/// where 'U' and 'V' are the vertices in the left and right partition, respectively,
/// and 'E' the edges.
///
/// Vertices in 'U' and 'V' are assumed to be sequentially indexed: 0, 1, ...
///
struct bipartite_graph
{
	int** adj_u;     //!< adjacent 'V' vertices for each 'U' vertex
	int** adj_v;     //!< adjacent 'U' vertices for each 'V' vertex
	int* num_adj_u;  //!< number of adjacent 'V' vertices for each 'U' vertex
	int* num_adj_v;  //!< number of adjacent 'U' vertices for each 'V' vertex
	int num_u;       //!< number of 'U' vertices
	int num_v;       //!< number of 'V' vertices
};


//________________________________________________________________________________________________________________________
///
/// \brief Bipartite graph edge.
///
struct bipartite_graph_edge
{
	int u;  //!< 'U' vertex
	int v;  //!< 'V' vertex
};


void init_bipartite_graph(const int num_u, const int num_v, const struct bipartite_graph_edge* edges, const int nedges, struct bipartite_graph* graph);

void delete_bipartite_graph(struct bipartite_graph* graph);


//________________________________________________________________________________________________________________________
///
/// \brief Bipartite graph matching.
///
struct bipartite_graph_matching
{
	struct bipartite_graph_edge* edges;  //!< edges of the matching
	int nedges;                          //!< number of edges
};

void bipartite_graph_maximum_cardinality_matching(const struct bipartite_graph* graph, struct bipartite_graph_matching* matching);


//________________________________________________________________________________________________________________________
//


void bipartite_graph_minimum_vertex_cover(const struct bipartite_graph* graph, bool* restrict u_cover, bool* restrict v_cover);
