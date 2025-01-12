/// \file su2_graph.c
/// \brief Internal temporary graph data structure for SU(2) symmetric tensors.

#include <stdlib.h>
#include "su2_graph.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Copy an SU(2) symmetry graph.
///
void copy_su2_graph(const struct su2_graph* src, struct su2_graph* dst)
{
	dst->nodes = ct_malloc(src->num_nodes * sizeof(struct su2_graph_node));
	dst->edges = ct_malloc(src->num_edges * sizeof(struct su2_graph_edge));

	memcpy(dst->nodes, src->nodes, src->num_nodes * sizeof(struct su2_graph_node));
	memcpy(dst->edges, src->edges, src->num_edges * sizeof(struct su2_graph_edge));

	dst->num_nodes = src->num_nodes;
	dst->num_edges = src->num_edges;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an SU(2) symmetry graph (free memory).
///
void delete_su2_graph(struct su2_graph* graph)
{
	ct_free(graph->edges);
	ct_free(graph->nodes);

	graph->num_edges = 0;
	graph->num_nodes = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Internal consistency check of the SU(2) symmetry graph data structure.
///
bool su2_graph_is_consistent(const struct su2_graph* graph)
{
	if (graph->num_nodes <= 0) {
		return false;
	}
	if (graph->num_edges <= 0) {
		return false;
	}

	int* edge_counter = ct_calloc(graph->num_edges, sizeof(int));

	for (int i = 0; i < graph->num_nodes; i++)
	{
		const struct su2_graph_node* node = &graph->nodes[i];
		const int eids[3] = { node->eid_parent, node->eid_child[0], node->eid_child[1] };
		// must be pairwise different
		if ((eids[0] == eids[1]) || (eids[1] == eids[2]) || (eids[2] == eids[0]))
		{
			ct_free(edge_counter);
			return false;
		}

		const int d = (node->type == SU2_GRAPH_NODE_FUSE ? 0 : 1);

		for (int j = 0; j < 3; j++)
		{
			if (eids[j] < 0 || eids[j] >= graph->num_edges) {
				ct_free(edge_counter);
				return false;
			}

			edge_counter[eids[j]]++;

			const struct su2_graph_edge* edge = &graph->edges[eids[j]];

			// edge must refer back to node
			if (edge->nid[j == 0 ? d : 1 - d] != i)
			{
				ct_free(edge_counter);
				return false;
			}
		}
	}

	for (int i = 0; i < graph->num_edges; i++)
	{
		// edges must be connected to one or two nodes
		if (edge_counter[i] < 1 || edge_counter[i] > 2)
		{
			ct_free(edge_counter);
			return false;
		}
	}

	ct_free(edge_counter);

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct an SU(2) symmetry (sub-)graph from a tree, assuming that sufficient memory for 'nodes' array has been allocated.
///
static int su2_subgraph_from_tree(const struct su2_tree_node* tree, const enum su2_graph_node_type type, struct su2_graph* graph)
{
	assert(tree != NULL);

	if (su2_tree_node_is_leaf(tree)) {
		return -1;
	}

	const int d = (type == SU2_GRAPH_NODE_FUSE ? 0 : 1);

	const int nid = graph->num_nodes;
	struct su2_graph_node* node = &graph->nodes[nid];
	graph->num_nodes++;  // can further increase during recursive function calls
	node->eid_parent = tree->i_ax;
	node->type = type;

	for (int j = 0; j < 2; j++)
	{
		const int eid = tree->c[j]->i_ax;
		assert(0 <= eid && eid < graph->num_edges);

		node->eid_child[j] = eid;

		struct su2_graph_edge* edge = &graph->edges[eid];
		edge->nid[1 - d] = nid;  // refer back to node
		edge->nid[d] = su2_subgraph_from_tree(tree->c[j], type, graph);
	}

	return nid;
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct an SU(2) symmetry graph from a fuse and split tree.
///
void su2_graph_from_fuse_split_tree(const struct su2_fuse_split_tree* tree, struct su2_graph* graph)
{
	graph->num_nodes = 0;
	graph->num_edges = tree->ndim;

	// use number of edges as upper bound for number of required nodes
	graph->nodes = ct_malloc(graph->num_edges * sizeof(struct su2_graph_node));
	graph->edges = ct_malloc(graph->num_edges * sizeof(struct su2_graph_edge));

	assert(tree->tree_fuse->i_ax == tree->tree_split->i_ax);
	struct su2_graph_edge* edge = &graph->edges[tree->tree_fuse->i_ax];
	edge->nid[0] = su2_subgraph_from_tree(tree->tree_fuse,  SU2_GRAPH_NODE_FUSE,  graph);
	edge->nid[1] = su2_subgraph_from_tree(tree->tree_split, SU2_GRAPH_NODE_SPLIT, graph);

	assert(graph->num_nodes > 0);
	assert(graph->num_nodes <= graph->num_edges);
}


//________________________________________________________________________________________________________________________
///
/// \brief Whether the node 'nid' is a fusion node or corresponds to a leaf.
///
static inline bool su2_graph_node_fuse_or_leaf(const struct su2_graph* graph, const int nid)
{
	return (nid < 0) || (graph->nodes[nid].type == SU2_GRAPH_NODE_FUSE);
}


//________________________________________________________________________________________________________________________
///
/// \brief Whether the node 'nid' is a splitting node or corresponds to a leaf.
///
static inline bool su2_graph_node_split_or_leaf(const struct su2_graph* graph, const int nid)
{
	return (nid < 0) || (graph->nodes[nid].type == SU2_GRAPH_NODE_SPLIT);
}


//________________________________________________________________________________________________________________________
///
/// \brief Whether an SU(2) graph node is the root of an upstream fuse tree.
///
static bool su2_subgraph_is_fuse_tree(const struct su2_graph* graph, const int nid)
{
	if (nid < 0) {
		return true;
	}

	assert(nid < graph->num_nodes);

	const struct su2_graph_node* node = &graph->nodes[nid];
	if (node->type == SU2_GRAPH_NODE_SPLIT) {
		return false;
	}

	return su2_subgraph_is_fuse_tree(graph, graph->edges[node->eid_child[0]].nid[0]) &&
	       su2_subgraph_is_fuse_tree(graph, graph->edges[node->eid_child[1]].nid[0]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Whether an SU(2) graph node is the root of a downstream split tree.
///
static bool su2_subgraph_is_split_tree(const struct su2_graph* graph, const int nid)
{
	if (nid < 0) {
		return true;
	}

	assert(nid < graph->num_nodes);

	const struct su2_graph_node* node = &graph->nodes[nid];
	if (node->type == SU2_GRAPH_NODE_FUSE) {
		return false;
	}

	return su2_subgraph_is_split_tree(graph, graph->edges[node->eid_child[0]].nid[1]) &&
	       su2_subgraph_is_split_tree(graph, graph->edges[node->eid_child[1]].nid[1]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Whether an SU(2) symmetry graph has the same topology as a fuse and split tree.
///
bool su2_graph_has_fuse_split_tree_topology(const struct su2_graph* graph)
{
	assert(graph->num_edges > 0);

	// follow edge towards potential center of fusion and splitting tree
	const struct su2_graph_edge* edge = &graph->edges[0];
	while (su2_graph_node_fuse_or_leaf(graph, edge->nid[0]) && !su2_graph_node_split_or_leaf(graph, edge->nid[1]))
	{
		edge = &graph->edges[graph->nodes[edge->nid[1]].eid_parent];
	}
	while (su2_graph_node_split_or_leaf(graph, edge->nid[1]) && !su2_graph_node_fuse_or_leaf(graph, edge->nid[0]))
	{
		edge = &graph->edges[graph->nodes[edge->nid[0]].eid_parent];
	}

	return su2_subgraph_is_fuse_tree (graph, edge->nid[0]) &&
	       su2_subgraph_is_split_tree(graph, edge->nid[1]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert an SU(2) symmetry (sub-)graph with tree topology to an actual tree.
///
static struct su2_tree_node* su2_subgraph_to_tree(const struct su2_graph* graph, const int eid, const int direction)
{
	assert(0 <= eid && eid < graph->num_edges);

	struct su2_tree_node* tree = ct_malloc(sizeof(struct su2_tree_node));
	tree->i_ax = eid;

	const struct su2_graph_edge* edge = &graph->edges[eid];

	const int nid = edge->nid[direction];

	if (nid < 0)
	{
		tree->c[0] = NULL;
		tree->c[1] = NULL;
	}
	else
	{
		assert(nid < graph->num_nodes);
		assert(eid == graph->nodes[nid].eid_parent);
		tree->c[0] = su2_subgraph_to_tree(graph, graph->nodes[nid].eid_child[0], direction);
		tree->c[1] = su2_subgraph_to_tree(graph, graph->nodes[nid].eid_child[1], direction);
	}

	return tree;
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert an SU(2) symmetry graph with fuse and split tree topology to an actual tree.
///
void su2_graph_to_fuse_split_tree(const struct su2_graph* graph, struct su2_fuse_split_tree* tree)
{
	assert(su2_graph_has_fuse_split_tree_topology(graph));

	// follow edge towards center of fusion and splitting tree
	int eid = 0;
	while (su2_graph_node_fuse_or_leaf(graph, graph->edges[eid].nid[0]) && !su2_graph_node_split_or_leaf(graph, graph->edges[eid].nid[1]))
	{
		eid = graph->nodes[graph->edges[eid].nid[1]].eid_parent;
	}
	while (su2_graph_node_split_or_leaf(graph, graph->edges[eid].nid[1]) && !su2_graph_node_fuse_or_leaf(graph, graph->edges[eid].nid[0]))
	{
		eid = graph->nodes[graph->edges[eid].nid[0]].eid_parent;
	}

	tree->tree_fuse  = su2_subgraph_to_tree(graph, eid, 0);
	tree->tree_split = su2_subgraph_to_tree(graph, eid, 1);
	tree->ndim = graph->num_edges;
}


//________________________________________________________________________________________________________________________
///
/// \brief Whether an SU(2) symmetry graph edge is at the center of an elementary yoga subtree.
///
bool su2_graph_is_yoga_edge(const struct su2_graph* graph, const int eid)
{
	assert(0 <= eid && eid < graph->num_edges);
	const struct su2_graph_edge* edge = &graph->edges[eid];

	if ((edge->nid[0] < 0) || (edge->nid[1] < 0)) {
		return false;
	}

	if ((graph->nodes[edge->nid[0]].type == SU2_GRAPH_NODE_FUSE) || (graph->nodes[edge->nid[1]].type == SU2_GRAPH_NODE_SPLIT)) {
		return false;
	}

	// requiring that nodes refer back to edge
	assert((eid == graph->nodes[edge->nid[0]].eid_child[0]) != (eid == graph->nodes[edge->nid[0]].eid_child[1]));
	assert((eid == graph->nodes[edge->nid[1]].eid_child[0]) != (eid == graph->nodes[edge->nid[1]].eid_child[1]));

	return (eid == graph->nodes[edge->nid[0]].eid_child[0]) == (eid == graph->nodes[edge->nid[1]].eid_child[1]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert an internal elementary yoga subtree to a simple subtree.
///
void su2_graph_yoga_to_simple_subtree(struct su2_graph* graph, const int eid)
{
	assert(su2_graph_is_yoga_edge(graph, eid));

	struct su2_graph_edge* edge = &graph->edges[eid];

	struct su2_graph_node* node0 = &graph->nodes[edge->nid[0]];
	struct su2_graph_node* node1 = &graph->nodes[edge->nid[1]];

	graph->edges[node0->eid_parent].nid[1] = edge->nid[1];
	graph->edges[node1->eid_parent].nid[0] = edge->nid[0];

	const int i = (eid == node0->eid_child[0]) ? 0 : 1;
	node0->eid_child[i    ] = node1->eid_parent;
	node1->eid_child[1 - i] = node0->eid_parent;

	node0->eid_parent = eid;
	node1->eid_parent = eid;

	// swap upstream and downstream node indices of edge
	int tmp = edge->nid[0];
	edge->nid[0] = edge->nid[1];
	edge->nid[1] = tmp;
}
