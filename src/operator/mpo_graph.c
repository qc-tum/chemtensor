/// \file mpo_graph.c
/// \brief MPO graph internal data structure for generating MPO representations.

#include <stdio.h>
#include <assert.h>
#include "mpo_graph.h"
#include "hash_table.h"
#include "linked_list.h"
#include "bipartite_graph.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Add an edge reference to an MPO graph vertex.
///
void mpo_graph_vertex_add_edge(const int direction, const int eid, struct mpo_graph_vertex* vertex)
{
	assert(0 <= direction && direction < 2);

	if (vertex->num_edges[direction] == 0)
	{
		vertex->eids[direction] = ct_malloc(sizeof(int));
		vertex->eids[direction][0] = eid;
	}
	else
	{
		// re-allocate memory for indices
		int* eids_prev = vertex->eids[direction];
		const int num = vertex->num_edges[direction];
		vertex->eids[direction] = ct_malloc((num + 1) * sizeof(int));
		memcpy(vertex->eids[direction], eids_prev, num * sizeof(int));
		ct_free(eids_prev);
		vertex->eids[direction][num] = eid;
	}

	vertex->num_edges[direction]++;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an MPO graph vertex (free memory).
///
static void delete_mpo_graph_vertex(struct mpo_graph_vertex* vertex)
{
	ct_free(vertex->eids[0]);
	ct_free(vertex->eids[1]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate an MPO graph edge.
///
static void allocate_mpo_graph_edge(const int nopics, struct mpo_graph_edge* edge)
{
	edge->vids[0] = -1;
	edge->vids[1] = -1;
	edge->opics = ct_calloc(nopics, sizeof(struct local_op_ref));
	edge->nopics = nopics;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an MPO graph edge (free memory).
///
static void delete_mpo_graph_edge(struct mpo_graph_edge* edge)
{
	ct_free(edge->opics);
	edge->nopics = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Operator half-chain, temporary data structure for building an operator graph from a list of operator chains.
///
struct op_halfchain
{
	int* oids;       //!< list of local op_i operator IDs
	qnumber* qnums;  //!< interleaved bond quantum numbers, including a leading and trailing quantum number
	int length;      //!< length (number of local operators)
	int vidl;        //!< index of left-connected vertex
};


//________________________________________________________________________________________________________________________
///
/// \brief Allocate an operator half-chain.
///
static void allocate_op_halfchain(const int length, struct op_halfchain* chain)
{
	chain->oids   = ct_calloc(length, sizeof(int));
	chain->qnums  = ct_calloc(length + 1, sizeof(qnumber));
	chain->length = length;
	chain->vidl   = -1;
}


//________________________________________________________________________________________________________________________
///
/// \brief Copy an operator half-chain, allocating memory for the copy.
///
static void copy_op_halfchain(const struct op_halfchain* restrict src, struct op_halfchain* restrict dst)
{
	allocate_op_halfchain(src->length, dst);
	memcpy(dst->oids,  src->oids,   src->length      * sizeof(int));
	memcpy(dst->qnums, src->qnums, (src->length + 1) * sizeof(qnumber));
	dst->vidl = src->vidl;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an operator half-chain (free memory).
///
static void delete_op_halfchain(struct op_halfchain* chain)
{
	ct_free(chain->qnums);
	ct_free(chain->oids);
	chain->length = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Test equality of two operator half-chains.
///
static bool op_halfchain_equal(const void* c1, const void* c2)
{
	const struct op_halfchain* chain1 = c1;
	const struct op_halfchain* chain2 = c2;

	if (chain1->length != chain2->length) {
		return false;
	}
	if (chain1->vidl != chain2->vidl) {
		return false;
	}
	for (int i = 0; i < chain1->length; i++) {
		if (chain1->oids[i] != chain2->oids[i]) {
			return false;
		}
	}
	for (int i = 0; i < chain1->length + 1; i++) {
		if (chain1->qnums[i] != chain2->qnums[i]) {
			return false;
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the hash value of an operator half-chain.
///
static hash_type op_halfchain_hash_func(const void* c)
{
	const struct op_halfchain* chain = c;

	// Fowler-Noll-Vo FNV-1a (64-bit) hash function, see
	// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
	const uint64_t offset = 14695981039346656037U;
	const uint64_t prime  = 1099511628211U;
	hash_type hash = offset;
	for (int i = 0; i < chain->length; i++) {
		hash = (hash ^ chain->oids[i]) * prime;
	}
	for (int i = 0; i < chain->length + 1; i++) {
		hash = (hash ^ chain->qnums[i]) * prime;
	}
	hash = (hash ^ chain->vidl) * prime;
	return hash;
}


//________________________________________________________________________________________________________________________
///
/// \brief Store the information of a 'U' node.
///
/// Temporary data structure for building an operator graph from a list of operator chains.
///
struct u_node
{
	int oid;        //!< local operator ID
	qnumber qnum0;  //!< left quantum number
	qnumber qnum1;  //!< right quantum number
	int vidl;       //!< index of left-connected vertex
};


//________________________________________________________________________________________________________________________
///
/// \brief Test equality of two 'U' nodes.
///
static bool u_node_equal(const void* n1, const void* n2)
{
	const struct u_node* node1 = n1;
	const struct u_node* node2 = n2;

	return (node1->oid   == node2->oid  ) &&
	       (node1->qnum0 == node2->qnum0) &&
	       (node1->qnum1 == node2->qnum1) &&
	       (node1->vidl  == node2->vidl);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the hash value of a 'U' node.
///
static hash_type u_node_hash_func(const void* n)
{
	const struct u_node* node = n;

	// Fowler-Noll-Vo FNV-1a (64-bit) hash function, see
	// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
	const uint64_t offset = 14695981039346656037U;
	const uint64_t prime  = 1099511628211U;
	hash_type hash = offset;
	hash = (hash ^ node->oid)   * prime;
	hash = (hash ^ node->qnum0) * prime;
	hash = (hash ^ node->qnum1) * prime;
	hash = (hash ^ node->vidl)  * prime;
	return hash;
}


//________________________________________________________________________________________________________________________
///
/// \brief Weighted edge temporary data structure.
///
struct weighted_edge
{
	int i, j;  //!< indices
	int cid;   //!< coefficient index
};


//________________________________________________________________________________________________________________________
///
/// \brief Local site and half-chain partition auxiliary data structure.
///
struct site_halfchain_partition
{
	struct u_node*       ulist;  //!< array of 'U' nodes
	struct op_halfchain* vlist;  //!< array of 'V' half-chains
	int* gamma;                  //!< matrix of gamma coefficient indices, of dimension 'num_u x num_v'
	int num_u;                   //!< number of 'U' nodes
	int num_v;                   //!< number of 'V' half-chains
};


//________________________________________________________________________________________________________________________
///
/// \brief Delete a local site and half-chain partition (free memory).
///
static void delete_site_halfchain_partition(struct site_halfchain_partition* partition)
{
	ct_free(partition->gamma);
	for (int j = 0; j < partition->num_v; j++) {
		delete_op_halfchain(&partition->vlist[j]);
	}
	ct_free(partition->vlist);
	ct_free(partition->ulist);
}


//________________________________________________________________________________________________________________________
///
/// \brief Repartition half-chains after splitting off the local operators acting on the leftmost site.
///
static void site_partition_halfchains(const struct op_halfchain* chains, const int* cids, const int nchains, struct site_halfchain_partition* partition)
{
	memset(partition, 0, sizeof(struct site_halfchain_partition));
	// upper bound on required memory
	partition->ulist = ct_calloc(nchains, sizeof(struct u_node));
	partition->vlist = ct_calloc(nchains, sizeof(struct op_halfchain));

	// use a linked list for intermediate storage of gamma coefficients
	struct linked_list gamma_list = { 0 };

	// use hash tables for fast look-up
	struct hash_table u_ht, v_ht;
	create_hash_table(u_node_equal, u_node_hash_func, sizeof(struct u_node), 4*nchains, &u_ht);
	create_hash_table(op_halfchain_equal, op_halfchain_hash_func, sizeof(struct op_halfchain), 4*nchains, &v_ht);

	for (int k = 0; k < nchains; k++)
	{
		const struct op_halfchain* chain = &chains[k];
		assert(chain->length >= 1);

		// U_i node
		struct u_node u = { .oid = chain->oids[0], .qnum0 = chain->qnums[0], .qnum1 = chain->qnums[1], .vidl = chain->vidl };
		int* i = hash_table_get(&u_ht, &u);
		if (i == NULL)
		{
			// insert node into array
			memcpy(&partition->ulist[partition->num_u], &u, sizeof(struct u_node));
			// insert (node, array index) into hash table
			i = ct_malloc(sizeof(int));
			*i = partition->num_u;
			hash_table_insert(&u_ht, &u, i);
			partition->num_u++;
		}
		else
		{
			assert(*i < partition->num_u);
		}

		// V_j node: remainder of input half-chain
		struct op_halfchain v;
		allocate_op_halfchain(chain->length - 1, &v);
		memcpy(v.oids,  chain->oids  + 1, (chain->length - 1) * sizeof(int));
		memcpy(v.qnums, chain->qnums + 1,  chain->length      * sizeof(qnumber));
		int* j = hash_table_get(&v_ht, &v);
		if (j == NULL)
		{
			// insert half-chain into array
			memcpy(&partition->vlist[partition->num_v], &v, sizeof(struct op_halfchain));
			// insert (half-chain, array index) into hash table
			j = ct_malloc(sizeof(int));
			*j = partition->num_v;
			hash_table_insert(&v_ht, &v, j);
			partition->num_v++;
		}
		else
		{
			// half-chain already exists
			delete_op_halfchain(&v);
			assert(*j < partition->num_v);
		}

		// record gamma coefficient index
		struct weighted_edge* edge = ct_malloc(sizeof(struct weighted_edge));
		edge->i = (*i);
		edge->j = (*j);
		edge->cid = cids[k];
		linked_list_append(&gamma_list, edge);
	}

	// copy entries into final gamma matrix
	partition->gamma = ct_calloc(partition->num_u*partition->num_v, sizeof(int));  // using that CID_ZERO == 0
	struct linked_list_node* node = gamma_list.head;
	while (node != NULL)
	{
		const struct weighted_edge* edge = node->data;
		// half-chains must be unique; if gamma coefficient index has been set, an input half-chain appeared twice
		assert(partition->gamma[edge->i*partition->num_v + edge->j] == CID_ZERO);
		partition->gamma[edge->i*partition->num_v + edge->j] = edge->cid;
		node = node->next;
	}
	delete_linked_list(&gamma_list, ct_free);

	delete_hash_table(&v_ht, ct_free);
	delete_hash_table(&u_ht, ct_free);
}


//________________________________________________________________________________________________________________________
///
/// \brief Comparison function for sorting.
///
static int compare_hashes(const void* a, const void* b)
{
	const hash_type x = *((hash_type*)a);
	const hash_type y = *((hash_type*)b);

	if (x < y) {
		return -1;
	}
	else if (x == y) {
		return 0;
	}
	else {
		return 1;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct an MPO operator graph from a list of operator chains, implementing the algorithm in:
///   Jiajun Ren, Weitang Li, Tong Jiang, Zhigang Shuai
///   A general automatic method for optimal construction of matrix product operators using bipartite graph theory
///   J. Chem. Phys. 153, 084118 (2020)
///
int mpo_graph_from_opchains(const struct op_chain* chains, const int nchains, const int nsites, struct mpo_graph* mpo_graph)
{
	// need at least one site
	assert(nsites > 0);
	// list of operator chains cannot be empty
	assert(nchains > 0);

	// pad identities and filter out chains with zero coefficients
	int nhalfchains = 0;
	struct op_chain* chains_full = ct_calloc(nchains, sizeof(struct op_chain));
	for (int k = 0; k < nchains; k++)
	{
		if (chains[k].cid != CID_ZERO)
		{
			op_chain_pad_identities(&chains[k], nsites, &chains_full[nhalfchains]);
			assert(chains_full[nhalfchains].length == nsites);
			nhalfchains++;
		}
	}
	// require at least one non-zero half-chain
	assert(nhalfchains > 0);

	// convert to half-chains and add a dummy identity operator
	struct op_halfchain* vlist_next = ct_malloc(nhalfchains * sizeof(struct op_halfchain));
	int* cids_next                  = ct_malloc(nhalfchains * sizeof(int));
	hash_type* halfchain_hashes     = ct_malloc(nhalfchains * sizeof(hash_type));
	for (int k = 0; k < nhalfchains; k++)
	{
		// dummy identity operator at the end
		allocate_op_halfchain(chains_full[k].length + 1, &vlist_next[k]);
		memcpy(vlist_next[k].oids,  chains_full[k].oids,   chains_full[k].length      * sizeof(int));
		memcpy(vlist_next[k].qnums, chains_full[k].qnums, (chains_full[k].length + 1) * sizeof(qnumber));
		vlist_next[k].oids[chains_full[k].length] = OID_IDENTITY;
		vlist_next[k].vidl = 0;

		cids_next[k] = chains_full[k].cid;

		halfchain_hashes[k] = op_halfchain_hash_func(&vlist_next[k]);
	}

	for (int k = 0; k < nhalfchains; k++) {
		delete_op_chain(&chains_full[k]);
	}
	ct_free(chains_full);

	// half-chains must be unique (to avoid repeated bipartite graph edges)
	qsort(halfchain_hashes, nhalfchains, sizeof(hash_type), compare_hashes);
	for (int k = 0; k < nhalfchains - 1; k++) {
		if (halfchain_hashes[k] == halfchain_hashes[k + 1]) {
			fprintf(stderr, "operator chains input to 'mpo_graph_from_opchains' are most likely not unique\n");
			return -1;
		}
	}
	ct_free(halfchain_hashes);

	mpo_graph->nsites = nsites;
	mpo_graph->verts     = ct_calloc(nsites + 1, sizeof(struct mpo_graph_vertex*));
	mpo_graph->edges     = ct_calloc(nsites,     sizeof(struct mpo_graph_edge*));
	mpo_graph->num_verts = ct_calloc(nsites + 1, sizeof(int));
	mpo_graph->num_edges = ct_calloc(nsites,     sizeof(int));
	// left start vertex
	mpo_graph->num_verts[0] = 1;
	mpo_graph->verts[0] = ct_calloc(1, sizeof(struct mpo_graph_vertex));

	// sweep from left to right
	for (int l = 0; l < nsites; l++)
	{
		struct site_halfchain_partition partition;
		site_partition_halfchains(vlist_next, cids_next, nhalfchains, &partition);

		// extract edges
		int nedges = 0;
		for (int i = 0; i < partition.num_u; i++) {
			for (int j = 0; j < partition.num_v; j++) {
				if (partition.gamma[i*partition.num_v + j] != CID_ZERO) {
					nedges++;
				}
			}
		}
		struct bipartite_graph_edge* edges = ct_malloc(nedges * sizeof(struct bipartite_graph_edge));
		int c = 0;
		for (int i = 0; i < partition.num_u; i++) {
			for (int j = 0; j < partition.num_v; j++) {
				if (partition.gamma[i*partition.num_v + j] != CID_ZERO) {
					edges[c].u = i;
					edges[c].v = j;
					c++;
				}
			}
		}
		assert(c == nedges);
		// construct bipartite graph and find a minimum vertex cover
		struct bipartite_graph bigraph;
		init_bipartite_graph(partition.num_u, partition.num_v, edges, nedges, &bigraph);
		ct_free(edges);
		bool* u_cover = ct_calloc(bigraph.num_u, sizeof(bool));
		bool* v_cover = ct_calloc(bigraph.num_v, sizeof(bool));
		if (partition.num_v == 1) {
			// for the special case of a single V vertex, make this the cover vertex
			// to ensure that coefficients are not passed on beyond last site
			v_cover[0] = true;
		}
		else {
			bipartite_graph_minimum_vertex_cover(&bigraph, u_cover, v_cover);
		}
		int num_u_cover = 0;
		for (int i = 0; i < bigraph.num_u; i++) {
			if (u_cover[i]) {
				num_u_cover++;
			}
		}
		int num_v_cover = 0;
		for (int j = 0; j < bigraph.num_v; j++) {
			if (v_cover[j]) {
				num_v_cover++;
			}
		}

		ct_free(cids_next);
		for (int k = 0; k < nhalfchains; k++) {
			delete_op_halfchain(&vlist_next[k]);
		}
		ct_free(vlist_next);
		nhalfchains = 0;

		// allocate memory for next iteration, using bipartite graph 'nedges' as upper bound for number of required half-chains
		vlist_next = ct_malloc(nedges * sizeof(struct op_halfchain));
		cids_next  = ct_malloc(nedges * sizeof(int));

		// using bipartite graph 'nedges' as upper bound for number of required MPO graph edges
		mpo_graph->edges[l] = ct_calloc(nedges, sizeof(struct mpo_graph_edge));
		mpo_graph->verts[l + 1] = ct_calloc(num_u_cover + num_v_cover, sizeof(struct mpo_graph_vertex));

		// current edge and vertex counter
		int ce = 0;
		int cv = 0;

		for (int i = 0; i < bigraph.num_u; i++)
		{
			if (!u_cover[i]) {
				continue;
			}

			const struct u_node* u = &partition.ulist[i];
			assert(mpo_graph->verts[l][u->vidl].qnum == u->qnum0);

			// add a new edge
			struct mpo_graph_edge* edge = &mpo_graph->edges[l][ce];
			allocate_mpo_graph_edge(1, edge);
			edge->vids[0] = u->vidl;
			edge->opics[0].oid = u->oid;
			edge->opics[0].cid = CID_ONE;
			// connect edge to previous vertex
			mpo_graph_vertex_add_edge(1, ce, &mpo_graph->verts[l][u->vidl]);

			// add a new vertex
			struct mpo_graph_vertex* vertex = &mpo_graph->verts[l + 1][cv];
			vertex->qnum = u->qnum1;
			mpo_graph_vertex_add_edge(0, ce, vertex);
			edge->vids[1] = cv;

			// assemble operator half-chains for next iteration
			for (int k = 0; k < bigraph.num_adj_u[i]; k++)
			{
				int j = bigraph.adj_u[i][k];
				// copy half-chain
				assert(partition.vlist[j].length == nsites - l);
				copy_op_halfchain(&partition.vlist[j], &vlist_next[nhalfchains]);
				vlist_next[nhalfchains].vidl = cv;
				// pass gamma coefficient index
				cids_next[nhalfchains] = partition.gamma[i*partition.num_v + j];
				nhalfchains++;
				// avoid double-counting
				partition.gamma[i*partition.num_v + j] = CID_ZERO;
			}

			ce++;
			cv++;
		}

		for (int j = 0; j < bigraph.num_v; j++)
		{
			if (!v_cover[j]) {
				continue;
			}

			// add a new vertex
			struct mpo_graph_vertex* vertex = &mpo_graph->verts[l + 1][cv];
			vertex->qnum = partition.vlist[j].qnums[0];

			// add operator half-chain for next iteration with reference to vertex
			assert(partition.vlist[j].length == nsites - l);
			copy_op_halfchain(&partition.vlist[j], &vlist_next[nhalfchains]);
			vlist_next[nhalfchains].vidl = cv;
			cids_next[nhalfchains] = CID_ONE;
			nhalfchains++;

			// create a "complementary operator"
			for (int k = 0; k < bigraph.num_adj_v[j]; k++)
			{
				int i = bigraph.adj_v[j][k];

				if (partition.gamma[i*partition.num_v + j] == CID_ZERO) {
					continue;
				}

				const struct u_node* u = &partition.ulist[i];
				assert(u->qnum1 == vertex->qnum);
				// add a new edge
				struct mpo_graph_edge* edge = &mpo_graph->edges[l][ce];
				allocate_mpo_graph_edge(1, edge);
				edge->vids[0] = u->vidl;
				edge->vids[1] = cv;
				edge->opics[0].oid = u->oid;
				edge->opics[0].cid = partition.gamma[i*partition.num_v + j];
				// keep track of handled edges
				partition.gamma[i*partition.num_v + j] = CID_ZERO;
				// connect edge to previous vertex
				mpo_graph_vertex_add_edge(1, ce, &mpo_graph->verts[l][u->vidl]);
				assert(mpo_graph->verts[l][u->vidl].qnum == u->qnum0);
				// connect edge to new vertex
				mpo_graph_vertex_add_edge(0, ce, vertex);
				ce++;
			}

			cv++;
		}

		// ensure that we have handled all edges
		for (int i = 0; i < partition.num_u; i++) {
			for (int j = 0; j < partition.num_v; j++) {
				assert(partition.gamma[i*partition.num_v + j] == CID_ZERO);
			}
		}

		// ensure that we had allocated sufficient memory
		assert(nhalfchains <= nedges);

		assert(cv == num_u_cover + num_v_cover);
		mpo_graph->num_verts[l + 1] = cv;
		mpo_graph->num_edges[l]     = ce;

		ct_free(v_cover);
		ct_free(u_cover);
		delete_bipartite_graph(&bigraph);
		delete_site_halfchain_partition(&partition);
	}

	// dummy trailing half-chain
	assert(nhalfchains == 1);
	assert(cids_next[0] == CID_ONE);
	ct_free(cids_next);
	for (int k = 0; k < nhalfchains; k++) {
		delete_op_halfchain(&vlist_next[k]);
	}
	ct_free(vlist_next);
	nhalfchains = 0;

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an MPO graph (free memory).
///
void delete_mpo_graph(struct mpo_graph* mpo_graph)
{
	for (int l = 0; l < mpo_graph->nsites; l++)
	{
		for (int i = 0; i < mpo_graph->num_edges[l]; i++)
		{
			delete_mpo_graph_edge(&mpo_graph->edges[l][i]);
		}
		ct_free(mpo_graph->edges[l]);
	}
	ct_free(mpo_graph->edges);
	ct_free(mpo_graph->num_edges);

	for (int l = 0; l < mpo_graph->nsites + 1; l++)
	{
		for (int i = 0; i < mpo_graph->num_verts[l]; i++)
		{
			delete_mpo_graph_vertex(&mpo_graph->verts[l][i]);
		}
		ct_free(mpo_graph->verts[l]);
	}
	ct_free(mpo_graph->verts);
	ct_free(mpo_graph->num_verts);
}


//________________________________________________________________________________________________________________________
///
/// \brief Internal consistency check of an MPO graph.
///
bool mpo_graph_is_consistent(const struct mpo_graph* mpo_graph)
{
	if (mpo_graph->nsites <= 0) {
		return false;
	}

	// edges indexed by a vertex must point back to same vertex
	for (int l = 0; l < mpo_graph->nsites + 1; l++)
	{
		for (int i = 0; i < mpo_graph->num_verts[l]; i++)
		{
			const struct mpo_graph_vertex* vertex = &mpo_graph->verts[l][i];
			if (l == 0 && vertex->num_edges[0] != 0) {
				return false;
			}
			if (l == mpo_graph->nsites && vertex->num_edges[1] != 0) {
				return false;
			}
			for (int j = 0; j < vertex->num_edges[0]; j++)
			{
				if (vertex->eids[0][j] < 0 || vertex->eids[0][j] >= mpo_graph->num_edges[l - 1]) {
					return false;
				}
				const struct mpo_graph_edge* edge = &mpo_graph->edges[l - 1][vertex->eids[0][j]];
				if (edge->vids[1] != i) {
					return false;
				}
			}
			for (int j = 0; j < vertex->num_edges[1]; j++)
			{
				if (vertex->eids[1][j] < 0 || vertex->eids[1][j] >= mpo_graph->num_edges[l]) {
					return false;
				}
				const struct mpo_graph_edge* edge = &mpo_graph->edges[l][vertex->eids[1][j]];
				if (edge->vids[0] != i) {
					return false;
				}
			}
		}
	}

	// vertices indexed by an edge must point back to same edge
	for (int l = 0; l < mpo_graph->nsites; l++)
	{
		for (int j = 0; j < mpo_graph->num_edges[l]; j++)
		{
			const struct mpo_graph_edge* edge = &mpo_graph->edges[l][j];
			for (int dir = 0; dir < 2; dir++)
			{
				const struct mpo_graph_vertex* vertex = &mpo_graph->verts[l + dir][edge->vids[dir]];
				bool edge_ref = false;
				for (int i = 0; i < vertex->num_edges[1 - dir]; i++) {
					if (vertex->eids[1 - dir][i] == j) {
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
/// \brief Construct the full matrix representation of the MPO graph.
///
void mpo_graph_to_matrix(const struct mpo_graph* mpo_graph, const struct dense_tensor* opmap, const void* coeffmap, const enum numeric_type dtype, struct dense_tensor* mat)
{
	const int nsites = mpo_graph->nsites;
	assert(nsites >= 1);
	assert(mpo_graph->num_verts[0]      == 1);
	assert(mpo_graph->num_verts[nsites] == 1);

	assert(coefficient_map_is_valid(dtype, coeffmap));

	struct dense_tensor* blocks[2];

	// initial 1x1 tensor
	blocks[0] = ct_malloc(sizeof(struct dense_tensor));
	const ct_long dim_init[2] = { 1, 1 };
	allocate_dense_tensor(dtype, 2, dim_init, blocks[0]);
	dense_tensor_set_identity(blocks[0]);

	// sweep from left to right
	for (int l = 0; l < nsites; l++)
	{
		blocks[(l + 1) % 2] = ct_malloc(mpo_graph->num_verts[l + 1] * sizeof(struct dense_tensor));

		for (int i = 0; i < mpo_graph->num_verts[l + 1]; i++)
		{
			const struct mpo_graph_vertex* vertex = &mpo_graph->verts[l + 1][i];
			assert(vertex->num_edges[0] > 0);
			for (int j = 0; j < vertex->num_edges[0]; j++)
			{
				const struct mpo_graph_edge* edge = &mpo_graph->edges[l][vertex->eids[0][j]];
				assert(edge->vids[1] == i);

				struct dense_tensor op;
				construct_local_operator(edge->opics, edge->nopics, opmap, coeffmap, &op);

				// ensure that left-connected vertex index is valid
				assert(0 <= edge->vids[0] && edge->vids[0] < mpo_graph->num_verts[l]);
				if (j == 0)
				{
					dense_tensor_kronecker_product(&blocks[l % 2][edge->vids[0]], &op, &blocks[(l + 1) % 2][i]);
				}
				else
				{
					struct dense_tensor tmp;
					dense_tensor_kronecker_product(&blocks[l % 2][edge->vids[0]], &op, &tmp);
					dense_tensor_scalar_multiply_add(numeric_one(tmp.dtype), &tmp, &blocks[(l + 1) % 2][i]);
					delete_dense_tensor(&tmp);
				}

				delete_dense_tensor(&op);
			}
		}

		for (int i = 0; i < mpo_graph->num_verts[l]; i++)
		{
			delete_dense_tensor(&blocks[l % 2][i]);
		}
		ct_free(blocks[l % 2]);
	}

	// final single block contains result
	assert(mpo_graph->num_verts[nsites] == 1);
	*mat = blocks[nsites % 2][0];  // copy internal data pointers
	ct_free(blocks[nsites % 2]);
}
