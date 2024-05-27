/// \file ttno_graph.c
/// \brief Tree tensor network operator (TTNO) graph internal data structure.

#include "ttno_graph.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Add an edge reference to a TTNO graph vertex.
///
void ttno_graph_vertex_add_edge(const int direction, const int eid, struct ttno_graph_vertex* vertex)
{
	assert(0 <= direction && direction < 2);

	if (vertex->num_edges[direction] == 0)
	{
		vertex->eids[direction] = aligned_alloc(MEM_DATA_ALIGN, sizeof(int));
		vertex->eids[direction][0] = eid;
	}
	else
	{
		// re-allocate memory for indices
		int* eids_prev = vertex->eids[direction];
		const int num = vertex->num_edges[direction];
		vertex->eids[direction] = aligned_alloc(MEM_DATA_ALIGN, (num + 1) * sizeof(int));
		memcpy(vertex->eids[direction], eids_prev, num * sizeof(int));
		aligned_free(eids_prev);
		vertex->eids[direction][num] = eid;
	}

	vertex->num_edges[direction]++;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a TTNO graph vertex (free memory).
///
static void delete_ttno_graph_vertex(struct ttno_graph_vertex* vertex)
{
	aligned_free(vertex->eids[0]);
	aligned_free(vertex->eids[1]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a TTNO graph hyperedge (free memory).
///
static void delete_ttno_graph_hyperedge(struct ttno_graph_hyperedge* edge)
{
	aligned_free(edge->opics);
	edge->nopics = 0;
	aligned_free(edge->vids);
	edge->order = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a TTNO graph (free memory).
///
void delete_ttno_graph(struct ttno_graph* graph)
{
	// edges
	for (int l = 0; l < graph->nsites; l++)
	{
		for (int i = 0; i < graph->num_edges[l]; i++)
		{
			delete_ttno_graph_hyperedge(&graph->edges[l][i]);
		}
		aligned_free(graph->edges[l]);
	}
	aligned_free(graph->edges);
	aligned_free(graph->num_edges);

	// vertices
	for (int l = 0; l < graph->nsites; l++)
	{
		for (int n = 0; n < graph->topology.num_neighbors[l]; n++)
		{
			const int k = graph->topology.neighbor_map[l][n];
			if (k > l) {
				continue;
			}
			const int iv = k*graph->nsites + l;
			for (int i = 0; i < graph->num_verts[iv]; i++) {
				delete_ttno_graph_vertex(&graph->verts[iv][i]);
			}
			aligned_free(graph->verts[iv]);
		}
	}
	aligned_free(graph->verts);
	aligned_free(graph->num_verts);

	delete_abstract_graph(&graph->topology);
}


//________________________________________________________________________________________________________________________
///
/// \brief Internal consistency check of a TTNO graph.
///
bool ttno_graph_is_consistent(const struct ttno_graph* graph)
{
	if (graph->nsites <= 0) {
		return false;
	}

	// topology
	if (graph->topology.num_nodes != graph->nsites) {
		return false;
	}
	if (!abstract_graph_is_consistent(&graph->topology)) {
		return false;
	}
	// verify tree topology
	if (!abstract_graph_is_connected_tree(&graph->topology)) {
		return false;
	}

	// compatibility of vertex numbers with tree topology
	for (int l = 0; l < graph->nsites; l++)
	{
		for (int k = l; k < graph->nsites; k++) {
			if (graph->num_verts[k*graph->nsites + l] != 0) {
				return false;
			}
		}

		// require at least one vertex for each edge in tree topology
		for (int k = 0; k < l; k++) {
			bool is_neighbor = false;
			for (int n = 0; n < graph->topology.num_neighbors[l]; n++) {
				if (graph->topology.neighbor_map[l][n] == k) {
					is_neighbor = true;
					break;
				}
			}
			if (is_neighbor) {
				if (graph->num_verts[k*graph->nsites + l] <= 0) {
					return false;
				}
			}
			else {
				if (graph->num_verts[k*graph->nsites + l] != 0) {
					return false;
				}
			}
		}
	}

	// edges indexed by a vertex must point back to same vertex
	for (int l = 0; l < graph->nsites; l++)
	{
		for (int n = 0; n < graph->topology.num_neighbors[l]; n++)
		{
			const int k = graph->topology.neighbor_map[l][n];
			if (k > l) {
				continue;
			}
			int m = 0;
			for (; m < graph->topology.num_neighbors[k]; m++) {
				if (graph->topology.neighbor_map[k][m] == l) {
					break;
				}
			}
			assert(m < graph->topology.num_neighbors[k]);

			const int iv = k*graph->nsites + l;

			for (int i = 0; i < graph->num_verts[iv]; i++)
			{
				const struct ttno_graph_vertex* vertex = &graph->verts[iv][i];
				for (int j = 0; j < vertex->num_edges[0]; j++)
				{
					if (vertex->eids[0][j] < 0 || vertex->eids[0][j] >= graph->num_edges[k]) {
						return false;
					}
					const struct ttno_graph_hyperedge* edge = &graph->edges[k][vertex->eids[0][j]];
					if (edge->vids[m] != i) {
						return false;
					}
				}
				for (int j = 0; j < vertex->num_edges[1]; j++)
				{
					if (vertex->eids[1][j] < 0 || vertex->eids[1][j] >= graph->num_edges[l]) {
						return false;
					}
					const struct ttno_graph_hyperedge* edge = &graph->edges[l][vertex->eids[1][j]];
					if (edge->vids[n] != i) {
						return false;
					}
				}
			}
		}
	}

	// vertices indexed by an edge must point back to same edge
	for (int l = 0; l < graph->nsites; l++)
	{
		for (int j = 0; j < graph->num_edges[l]; j++)
		{
			const struct ttno_graph_hyperedge* edge = &graph->edges[l][j];
			if (edge->order != graph->topology.num_neighbors[l]) {
				return false;
			}
			for (int n = 0; n < edge->order; n++)
			{
				int k = graph->topology.neighbor_map[l][n];
				int iv = (k < l ? k*graph->nsites + l : l*graph->nsites + k);
				const struct ttno_graph_vertex* vertex = &graph->verts[iv][edge->vids[n]];
				bool edge_ref = false;
				for (int i = 0; i < vertex->num_edges[l < k ? 0 : 1]; i++) {
					if (vertex->eids[l < k ? 0 : 1][i] == j) {
						edge_ref = true;
						break;
					}
				}
				if (!edge_ref) {
					return false;
				}
			}
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Contracted subtree of a TTNO graph (auxiliary data structure used for contraction).
///
struct ttno_graph_contracted_subtree
{
	struct dense_tensor* blocks;   //!< overall logical operators, for each upstream virtual bond
	int* i_sites;                  //!< logical site indices of contracted subtree
	int nblocks;                   //!< number of blocks
	int nsites;                    //!< number of sites
};


//________________________________________________________________________________________________________________________
///
/// \brief Delete a contracted subtree (free memory).
///
static void delete_ttno_graph_contracted_subtree(struct ttno_graph_contracted_subtree* subtree)
{
	for (int i = 0; i < subtree->nblocks; i++) {
		delete_dense_tensor(&subtree->blocks[i]);
	}
	aligned_free(subtree->blocks);
	aligned_free(subtree->i_sites);
}


//________________________________________________________________________________________________________________________
///
/// \brief Temporary data structure for sorting by site index.
///
struct indexed_site_index
{
	int i_site;
	int index;
};

//________________________________________________________________________________________________________________________
///
/// \brief Comparison function for sorting.
///
static int compare_indexed_site_index(const void* a, const void* b)
{
	const struct indexed_site_index* x = (const struct indexed_site_index*)a;
	const struct indexed_site_index* y = (const struct indexed_site_index*)b;

	if (x->i_site < y->i_site) {
		return -1;
	}
	if (x->i_site > y->i_site) {
		return 1;
	}
	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Recursively contract a subtree of a TTNO graph starting from 'i_site'.
///
/// 'i_parent == -1' indicates root node.
///
static void ttno_graph_contract_subtree(const struct ttno_graph* graph, const int i_site, const int i_parent, const struct dense_tensor* opmap, struct ttno_graph_contracted_subtree* contracted)
{
	assert(i_site != i_parent);

	// local dimension
	const long d = opmap[0].dim[0];

	// construct blocks for connections to child nodes
	struct ttno_graph_contracted_subtree* children = aligned_alloc(MEM_DATA_ALIGN, graph->topology.num_neighbors[i_site] * sizeof(struct ttno_graph_contracted_subtree));
	for (int n = 0; n < graph->topology.num_neighbors[i_site]; n++)
	{
		const int k = graph->topology.neighbor_map[i_site][n];
		if (k == i_parent) {
			continue;
		}
		ttno_graph_contract_subtree(graph, k, i_site, opmap, &children[n]);
	}

	// determine collection of site indices of to-be contracted subtree
	contracted->i_sites = aligned_alloc(MEM_DATA_ALIGN, sizeof(int));
	contracted->i_sites[0] = i_site;
	contracted->nsites = 1;
	for (int n = 0; n < graph->topology.num_neighbors[i_site]; n++)
	{
		const int k = graph->topology.neighbor_map[i_site][n];
		if (k == i_parent) {
			continue;
		}

		int* i_sites_new = aligned_alloc(MEM_DATA_ALIGN, (contracted->nsites + children[n].nsites) * sizeof(int));
		if (k < i_site) {
			memcpy( i_sites_new, children[n].i_sites, children[n].nsites * sizeof(int));
			memcpy(&i_sites_new[children[n].nsites], contracted->i_sites, contracted->nsites * sizeof(int));
		}
		else {
			memcpy( i_sites_new, contracted->i_sites, contracted->nsites * sizeof(int));
			memcpy(&i_sites_new[contracted->nsites], children[n].i_sites, children[n].nsites * sizeof(int));
		}
		aligned_free(contracted->i_sites);
		contracted->i_sites = i_sites_new;
		contracted->nsites += children[n].nsites;
	}

	if (i_parent < 0)  // root node
	{
		contracted->nblocks = 1;
		contracted->blocks = aligned_alloc(MEM_DATA_ALIGN, contracted->nblocks * sizeof(struct dense_tensor));

		assert(graph->num_edges[i_site] > 0);
		for (int j = 0; j < graph->num_edges[i_site]; j++)
		{
			const struct ttno_graph_hyperedge* edge = &graph->edges[i_site][j];

			// local operator
			struct dense_tensor op;
			construct_local_operator(edge->opics, edge->nopics, opmap, &op);
			assert(op.ndim == 2);
			assert(op.dim[0] == d && op.dim[1] == d);

			// compute Kronecker products with child blocks indexed by current edge
			for (int n = 0; n < graph->topology.num_neighbors[i_site]; n++)
			{
				const int k = graph->topology.neighbor_map[i_site][n];

				struct dense_tensor t;
				if (k < i_site) {
					dense_tensor_kronecker_product(&children[n].blocks[edge->vids[n]], &op, &t);
				}
				else {
					dense_tensor_kronecker_product(&op, &children[n].blocks[edge->vids[n]], &t);
				}
				delete_dense_tensor(&op);
				move_dense_tensor_data(&t, &op);
			}

			// accumulate Kronecker product in current block
			if (j == 0) {
				move_dense_tensor_data(&op, &contracted->blocks[0]);
			}
			else {
				dense_tensor_scalar_multiply_add(numeric_one(op.dtype), &op, &contracted->blocks[0]);
				delete_dense_tensor(&op);
			}
		}
	}
	else  // not root node
	{
		const int iv = (i_site < i_parent ? i_site*graph->nsites + i_parent : i_parent*graph->nsites + i_site);
		assert(graph->verts[iv] != NULL);

		contracted->nblocks = graph->num_verts[iv];
		contracted->blocks = aligned_alloc(MEM_DATA_ALIGN, contracted->nblocks * sizeof(struct dense_tensor));

		for (int i = 0; i < graph->num_verts[iv]; i++)
		{
			const struct ttno_graph_vertex* vertex = &graph->verts[iv][i];

			const int dir = (i_site < i_parent ? 0 : 1);
			assert(vertex->num_edges[dir] > 0);
			for (int j = 0; j < vertex->num_edges[dir]; j++)
			{
				assert(0 <= vertex->eids[dir][j] && vertex->eids[dir][j] < graph->num_edges[i_site]);

				const struct ttno_graph_hyperedge* edge = &graph->edges[i_site][vertex->eids[dir][j]];

				// local operator
				struct dense_tensor op;
				construct_local_operator(edge->opics, edge->nopics, opmap, &op);
				assert(op.ndim == 2);
				assert(op.dim[0] == d && op.dim[1] == d);

				// compute Kronecker products with child blocks indexed by current edge
				for (int n = 0; n < graph->topology.num_neighbors[i_site]; n++)
				{
					const int k = graph->topology.neighbor_map[i_site][n];
					if (k == i_parent) {
						assert(edge->vids[n] == i);
						continue;
					}

					struct dense_tensor t;
					if (k < i_site) {
						dense_tensor_kronecker_product(&children[n].blocks[edge->vids[n]], &op, &t);
					}
					else {
						dense_tensor_kronecker_product(&op, &children[n].blocks[edge->vids[n]], &t);
					}
					delete_dense_tensor(&op);
					move_dense_tensor_data(&t, &op);
				}

				// accumulate Kronecker product in current block
				if (j == 0) {
					move_dense_tensor_data(&op, &contracted->blocks[i]);
				}
				else {
					dense_tensor_scalar_multiply_add(numeric_one(op.dtype), &op, &contracted->blocks[i]);
					delete_dense_tensor(&op);
				}
			}
		}
	}

	for (int n = 0; n < graph->topology.num_neighbors[i_site]; n++)
	{
		const int k = graph->topology.neighbor_map[i_site][n];
		if (k == i_parent) {
			continue;
		}
		delete_ttno_graph_contracted_subtree(&children[n]);
	}
	aligned_free(children);

	// sort axes by site indices
	struct indexed_site_index* indexed_sites = aligned_alloc(MEM_DATA_ALIGN, contracted->nsites * sizeof(struct indexed_site_index));
	for (int j = 0; j < contracted->nsites; j++) {
		indexed_sites[j].i_site = contracted->i_sites[j];
		indexed_sites[j].index = j;
	}
	qsort(indexed_sites, contracted->nsites, sizeof(struct indexed_site_index), compare_indexed_site_index);
	int* perm = aligned_alloc(MEM_DATA_ALIGN, 2 * contracted->nsites * sizeof(int));
	for (int j = 0; j < contracted->nsites; j++)
	{
		contracted->i_sites[j] = indexed_sites[j].i_site;
		perm[j] = indexed_sites[j].index;
	}
	aligned_free(indexed_sites);
	// skip permutation operations in case of an identity permutation
	bool is_identity_perm = true;
	for (int j = 0; j < contracted->nsites; j++) {
		if (perm[j] != j) {
			is_identity_perm = false;
			break;
		}
	}
	if (!is_identity_perm)
	{
		// permute global row and column dimensions simultaneously
		for (int j = 0; j < contracted->nsites; j++) {
			perm[contracted->nsites + j] = contracted->nsites + perm[j];
		}
		long* dim = aligned_alloc(MEM_DATA_ALIGN, 2 * contracted->nsites * sizeof(long));
		for (int j = 0; j < 2 * contracted->nsites; j++) {
			dim[j] = d;
		}
		for (int i = 0; i < contracted->nblocks; i++)
		{
			assert(contracted->blocks[i].ndim == 2);
			const long orig_dim[2] = { contracted->blocks[i].dim[0], contracted->blocks[i].dim[1] };

			reshape_dense_tensor(2 * contracted->nsites, dim, &contracted->blocks[i]);

			struct dense_tensor t;
			transpose_dense_tensor(perm, &contracted->blocks[i], &t);
			delete_dense_tensor(&contracted->blocks[i]);
			move_dense_tensor_data(&t, &contracted->blocks[i]);

			reshape_dense_tensor(2, orig_dim, &contracted->blocks[i]);
		}
		aligned_free(dim);
	}
	aligned_free(perm);
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct the full matrix representation of the TTNO graph.
///
void ttno_graph_to_matrix(const struct ttno_graph* graph, const struct dense_tensor* opmap, struct dense_tensor* mat)
{
	assert(graph->nsites >= 1);

	// select site with maximum number of neighbors as root for contraction
	int i_root = 0;
	for (int l = 1; l < graph->nsites; l++) {
		if (graph->topology.num_neighbors[l] > graph->topology.num_neighbors[i_root]) {
			i_root = l;
		}
	}

	// contract full tree
	// set parent index to -1 for root node
	struct ttno_graph_contracted_subtree contracted;
	ttno_graph_contract_subtree(graph, i_root, -1, opmap, &contracted);
	assert(contracted.nsites == graph->nsites);
	assert(contracted.nblocks == 1);
	for (int l = 0; l < contracted.nsites; l++) {
		assert(contracted.i_sites[l] == l);
	}
	move_dense_tensor_data(&contracted.blocks[0], mat);
	aligned_free(contracted.blocks);
	aligned_free(contracted.i_sites);
}
