/// \file bipartite_graph.c
/// \brief Bipartite graph utility functions, including an implementation of the Hopcroft-Karp algorithm based on https://en.wikipedia.org/wiki/Hopcroft%E2%80%93Karp_algorithm

#include "bipartite_graph.h"
#include "queue.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Allocate and initialize a bipartite graph based on the provided edge list.
///
void init_bipartite_graph(const int num_u, const int num_v, const struct bipartite_graph_edge* edges, const int nedges, struct bipartite_graph* graph)
{
	assert(num_u >= 1);
	assert(num_v >= 1);
	graph->num_u = num_u;
	graph->num_v = num_v;
	graph->num_adj_u = ct_calloc(num_u, sizeof(int));
	graph->num_adj_v = ct_calloc(num_v, sizeof(int));
	// upper bounds for memory allocation
	int* num_adj_u_max = ct_calloc(num_u, sizeof(int));
	int* num_adj_v_max = ct_calloc(num_v, sizeof(int));
	for (int i = 0; i < nedges; i++)
	{
		int u = edges[i].u;
		int v = edges[i].v;
		assert(0 <= u && u < num_u);
		assert(0 <= v && v < num_v);
		num_adj_u_max[u]++;
		num_adj_v_max[v]++;
	}
	// construct adjacency maps
	graph->adj_u = ct_calloc(num_u, sizeof(int*));
	graph->adj_v = ct_calloc(num_v, sizeof(int*));
	for (int u = 0; u < num_u; u++) {
		graph->adj_u[u] = ct_calloc(num_adj_u_max[u], sizeof(int));
	}
	for (int v = 0; v < num_v; v++) {
		graph->adj_v[v] = ct_calloc(num_adj_v_max[v], sizeof(int));
	}
	for (int i = 0; i < nedges; i++)
	{
		int u = edges[i].u;
		int v = edges[i].v;
		// test whether edge is already registered in 'adj_u'
		bool contains_uv = false;
		for (int j = 0; j < graph->num_adj_u[u]; j++) {
			if (graph->adj_u[u][j] == v) {
				contains_uv = true;
				break;
			}
		}
		// add edge
		if (!contains_uv) {
			graph->adj_u[u][graph->num_adj_u[u]] = v;
			graph->num_adj_u[u]++;
		}
		assert(graph->num_adj_u[u] <= num_adj_u_max[u]);
		// test whether edge is already registered in 'adj_v'
		bool contains_vu = false;
		for (int j = 0; j < graph->num_adj_v[v]; j++) {
			if (graph->adj_v[v][j] == u) {
				contains_vu = true;
				break;
			}
		}
		// add edge
		if (!contains_vu) {
			graph->adj_v[v][graph->num_adj_v[v]] = u;
			graph->num_adj_v[v]++;
		}
		assert(graph->num_adj_v[v] <= num_adj_v_max[v]);
	}

	ct_free(num_adj_u_max);
	ct_free(num_adj_v_max);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a bipartite graph (free memory).
///
void delete_bipartite_graph(struct bipartite_graph* graph)
{
	for (int u = 0; u < graph->num_u; u++) {
		ct_free(graph->adj_u[u]);
	}
	for (int v = 0; v < graph->num_v; v++) {
		ct_free(graph->adj_v[v]);
	}
	ct_free(graph->adj_u);
	ct_free(graph->adj_v);
	graph->adj_u = NULL;
	graph->adj_v = NULL;
	ct_free(graph->num_adj_u);
	ct_free(graph->num_adj_v);
	graph->num_adj_u = NULL;
	graph->num_adj_v = NULL;
	graph->num_u = 0;
	graph->num_v = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Temporary data for running the Hopcroft-Karp algorithm to find a maximum-cardinality matching.
///
struct hopcroft_karp_data
{
	const struct bipartite_graph* graph;  //!< store a reference to the graph
	int* matched_pairs_u;                 //!< matched pair of each 'U' vertex, or NIL if vertex is not matched
	int* matched_pairs_v;                 //!< matched pair of each 'V' vertex, or NIL if vertex is not matched
	int* dist;                            //!< distance of 'U' vertices along augmenting path
};


//________________________________________________________________________________________________________________________
///
/// \brief Allocate and initialize the temporary data for running the Hopcroft-Karp algorithm.
///
static void init_hopcroft_karp_data(const struct bipartite_graph* graph, struct hopcroft_karp_data* data)
{
	// store a reference to the graph
	data->graph = graph;

	data->matched_pairs_u = ct_malloc(graph->num_u * sizeof(int));
	data->matched_pairs_v = ct_malloc(graph->num_v * sizeof(int));
	// NIL vertex is indexed by -1
	for (int u = 0; u < graph->num_u; u++) {
		data->matched_pairs_u[u] = -1;
	}
	for (int v = 0; v < graph->num_v; v++) {
		data->matched_pairs_v[v] = -1;
	}

	// formally "infinite" distance
	const int inf_dist = graph->num_u + 1;

	data->dist = ct_malloc((graph->num_u + 1) * sizeof(int));
	data->dist++;  // advance pointer since NIL vertex is indexed by -1
	for (int u = -1; u < graph->num_u; u++) {
		data->dist[u] = inf_dist;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete the Hopcroft-Karp data (free memory).
///
static void delete_hopcroft_karp_data(struct hopcroft_karp_data* data)
{
	// restore original memory pointer
	data->dist--;
	ct_free(data->dist);
	ct_free(data->matched_pairs_v);
	ct_free(data->matched_pairs_u);
	data->graph = NULL;
}


//________________________________________________________________________________________________________________________
///
/// \brief Find a path of minimal length connecting currently unmatched vertices in 'U' to currently unmatched vertices in 'V'
/// via a breadth-first search.
///
static bool hopcroft_karp_connect_unmatched_vertices(struct hopcroft_karp_data* data)
{
	// formally "infinite" distance
	const int inf_dist = data->graph->num_u + 1;

	struct queue queue = { 0 };
	for (int u = 0; u < data->graph->num_u; u++)
	{
		if (data->matched_pairs_u[u] == -1) {
			// 'u' has not been matched yet
			data->dist[u] = 0;
			enqueue(&queue, (void*)u);
		}
		else {
			data->dist[u] = inf_dist;
		}
	}
	// initialize distance to NIL as infinite
	data->dist[-1] = inf_dist;

	while (!queue_is_empty(&queue))
	{
		int u = (int)dequeue(&queue);
		// whether this node can provide a shorter path to NIL
		if (data->dist[u] < data->dist[-1])
		{
			for (int j = 0; j < data->graph->num_adj_u[u]; j++)
			{
				int v = data->graph->adj_u[u][j];
				// if pair of v has not been considered so far...
				if (data->dist[data->matched_pairs_v[v]] == inf_dist)
				{
					data->dist[data->matched_pairs_v[v]] = data->dist[u] + 1;
					enqueue(&queue, (void*)data->matched_pairs_v[v]);
				}
			}
		}
	}
	// if we can return to NIL using alternating path of distinct vertices
	// then there is an augmenting path
	return data->dist[-1] != inf_dist;
}


//________________________________________________________________________________________________________________________
///
/// \brief Add an augmenting path to the matching by performing a depth-first search.
///
static bool hopcroft_karp_add_augmenting_path(struct hopcroft_karp_data* data, const int u)
{
	// formally "infinite" distance
	const int inf_dist = data->graph->num_u + 1;

	if (u != -1)
	{
		for (int j = 0; j < data->graph->num_adj_u[u]; j++)
		{
			int v = data->graph->adj_u[u][j];
			// follow the distances from connecting unmatched vertices
			if (data->dist[data->matched_pairs_v[v]] == data->dist[u] + 1)
			{
				if (hopcroft_karp_add_augmenting_path(data, data->matched_pairs_v[v])) {
					data->matched_pairs_v[v] = u;
					data->matched_pairs_u[u] = v;
					return true;
				}
			}
		}
		// do not visit the same vertex multiple times
		data->dist[u] = inf_dist;
		return false;
	}
	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Run the Hopcroft-Karp algorithm to find a maximum-cardinality matching.
///
void bipartite_graph_maximum_cardinality_matching(const struct bipartite_graph* graph, struct bipartite_graph_matching* matching)
{
	struct hopcroft_karp_data data;
	init_hopcroft_karp_data(graph, &data);

	// outer loop of the algorithm
	while (hopcroft_karp_connect_unmatched_vertices(&data))
	{
		for (int u = 0; u < graph->num_u; u++)
		{
			if (data.matched_pairs_u[u] == -1) {
				hopcroft_karp_add_augmenting_path(&data, u);
			}
		}
	}

	// collect matched edges
	matching->nedges = 0;
	for (int u = 0; u < graph->num_u; u++)
	{
		if (data.matched_pairs_u[u] != -1) {
			matching->nedges++;
		}
	}
	matching->edges = ct_malloc(matching->nedges * sizeof(struct bipartite_graph_edge));
	int c = 0;
	for (int u = 0; u < graph->num_u; u++)
	{
		if (data.matched_pairs_u[u] != -1) {
			matching->edges[c].u = u;
			matching->edges[c].v = data.matched_pairs_u[u];
			c++;
		}
	}

	delete_hopcroft_karp_data(&data);
}


//________________________________________________________________________________________________________________________
///
/// \brief Whether an edge is contained in the bipartite graph matching.
///
static bool edge_in_matching(const struct bipartite_graph_matching* matching, const struct bipartite_graph_edge* edge)
{
	for (int i = 0; i < matching->nedges; i++) {
		if (matching->edges[i].u == edge->u && matching->edges[i].v == edge->v) {
			return true;
		}
	}
	return false;
}


//________________________________________________________________________________________________________________________
///
/// \brief Explore alternating paths originating from 'u_start' by a depth-first search.
///
static void explore_alternating_paths(const struct bipartite_graph* graph, const struct bipartite_graph_matching* matching,
	const int u_start, bool* restrict u_visited, bool* restrict v_visited)
{
	assert(0 <= u_start && u_start < graph->num_u);

	if (u_visited[u_start]) {
		return;
	}

	u_visited[u_start] = true;

	for (int j = 0; j < graph->num_adj_u[u_start]; j++)
	{
		int v = graph->adj_u[u_start][j];

		// traverse only unmatched edges
		struct bipartite_graph_edge edge = { u_start, v };
		if (edge_in_matching(matching, &edge)) {
			continue;
		}
		if (v_visited[v]) {
			continue;
		}
		v_visited[v] = true;
		for (int i = 0; i < graph->num_adj_v[v]; i++)
		{
			int u = graph->adj_v[v][i];

			// traverse only matched edges
			struct bipartite_graph_edge edge = { u, v };
			if (edge_in_matching(matching, &edge)) {
				explore_alternating_paths(graph, matching, u, u_visited, v_visited);
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Find a minimum vertex cover based on Kőnig's theorem.
///
void bipartite_graph_minimum_vertex_cover(const struct bipartite_graph* graph, bool* restrict u_cover, bool* restrict v_cover)
{
	// maximum matching
	struct bipartite_graph_matching matching = { 0 };
	bipartite_graph_maximum_cardinality_matching(graph, &matching);

	// unmatched vertices in 'U'
	bool* alist = ct_malloc(graph->num_u * sizeof(bool));
	for (int u = 0; u < graph->num_u; u++) {
		alist[u] = true;
	}
	for (int i = 0; i < matching.nedges; i++) {
		alist[matching.edges[i].u] = false;
	}
	// initialize cover vertices
	for (int u = 0; u < graph->num_u; u++) {
		u_cover[u] = true;
	}
	for (int v = 0; v < graph->num_v; v++) {
		v_cover[v] = false;
	}

	bool* u_visited = ct_malloc(graph->num_u * sizeof(bool));
	bool* v_visited = ct_malloc(graph->num_v * sizeof(bool));

	// vertices which are either in 'alist' or are connected to 'alist' by alternating paths
	for (int u = 0; u < graph->num_u; u++)
	{
		if (!alist[u]) {
			continue;
		}

		// initialize to false
		memset(u_visited, 0, graph->num_u * sizeof(bool));
		memset(v_visited, 0, graph->num_v * sizeof(bool));
		explore_alternating_paths(graph, &matching, u, u_visited, v_visited);

		// remove 'u_visited' from cover set
		for (int w = 0; w < graph->num_u; w++) {
			if (u_visited[w]) {
				u_cover[w] = false;
			}
		}
		for (int w = 0; w < graph->num_v; w++) {
			if (v_visited[w]) {
				v_cover[w] = true;
			}
		}
	}

	// number of vertices in minimum vertex cover must agree with
	// maximum-cardinality matching according to Kőnig's theorem
	#ifndef NDEBUG
	int num_cover = 0;
	for (int u = 0; u < graph->num_u; u++) {
		if (u_cover[u]) {
			num_cover++;
		}
	}
	for (int v = 0; v < graph->num_v; v++) {
		if (v_cover[v]) {
			num_cover++;
		}
	}
	assert(num_cover == matching.nedges);
	#endif

	// clean up
	ct_free(v_visited);
	ct_free(u_visited);
	ct_free(alist);
	ct_free(matching.edges);
}
