/// \file abstract_graph.c
/// \brief Abstract (symbolic) graph data structure and utility functions.

#include "abstract_graph.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Copy an abstract graph.
///
void copy_abstract_graph(const struct abstract_graph* restrict src, struct abstract_graph* restrict dst)
{
	dst->num_nodes = src->num_nodes;

	dst->num_neighbors = ct_malloc(src->num_nodes * sizeof(int));
	memcpy(dst->num_neighbors, src->num_neighbors, src->num_nodes * sizeof(int));

	dst->neighbor_map = ct_malloc(src->num_nodes * sizeof(int*));
	for (int l = 0; l < src->num_nodes; l++)
	{
		dst->neighbor_map[l] = ct_malloc(src->num_neighbors[l] * sizeof(int));
		memcpy(dst->neighbor_map[l], src->neighbor_map[l], src->num_neighbors[l] * sizeof(int));
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an abstract graph (free memory).
///
void delete_abstract_graph(struct abstract_graph* graph)
{
	for (int l = 0; l < graph->num_nodes; l++)
	{
		ct_free(graph->neighbor_map[l]);
	}
	ct_free(graph->neighbor_map);
	ct_free(graph->num_neighbors);

	graph->num_nodes = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Determine whether the neighborhood map of an abstract graph is consistent.
///
bool abstract_graph_is_consistent(const struct abstract_graph* graph)
{
	if (graph->num_nodes < 0) {
		return false;
	}

	for (int l = 0; l < graph->num_nodes; l++)
	{
		if (graph->num_neighbors[l] < 0) {
			return false;
		}

		for (int i = 0; i < graph->num_neighbors[l]; i++)
		{
			// check neighbor index range
			if (graph->neighbor_map[l][i] < 0 || graph->neighbor_map[l][i] >= graph->num_nodes) {
				return false;
			}

			// a node cannot be a neighbor of itself
			if (graph->neighbor_map[l][i] == l) {
				return false;
			}

			// list of neighboring nodes must be sorted
			if (i > 0 && graph->neighbor_map[l][i - 1] >= graph->neighbor_map[l][i]) {
				return false;
			}

			// neighbor must refer back to current node
			const int k = graph->neighbor_map[l][i];
			bool found = false;
			for (int j = 0; j < graph->num_neighbors[k]; j++) {
				if (graph->neighbor_map[k][j] == l) {
					found = true;
					break;
				}
			}
			if (!found) {
				return false;
			}
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Depth-first traversal of an abstract graph, marking visited nodes and returning false when encountering a loop.
///
static bool traverse_abstract_graph(const struct abstract_graph* graph, const int i_start, const int i_parent, bool* visited)
{
	visited[i_start] = true;

	bool loop_free = true;
	for (int j = 0; j < graph->num_neighbors[i_start]; j++)
	{
		const int i = graph->neighbor_map[i_start][j];

		if (i == i_parent) {
			continue;
		}

		if (visited[i]) {
			// encountered a loop
			loop_free = false;
			continue;
		}

		if (!traverse_abstract_graph(graph, i, i_start, visited)) {
			loop_free = false;
		}
	}

	return loop_free;
}


//________________________________________________________________________________________________________________________
///
/// \brief Determine whether an abstract graph is a connected tree.
///
bool abstract_graph_is_connected_tree(const struct abstract_graph* graph)
{
	if (graph->num_nodes < 1) {
		return false;
	}

	bool* visited = ct_calloc(graph->num_nodes, sizeof(bool));
	if (!traverse_abstract_graph(graph, 0, -1, visited)) {
		ct_free(visited);
		return false;
	}

	// all nodes must have been visited, i.e., graph must be connected
	for (int i = 0; i < graph->num_nodes; i++) {
		if (!visited[i]) {
			ct_free(visited);
			return false;
		}
	}

	ct_free(visited);

	return true;
}
