/// \file mpo_graph.c
/// \brief MPO graph internal data structure for generating MPO representations.

#include <assert.h>
#include "mpo_graph.h"
#include "hash_table.h"
#include "bipartite_graph.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Add an edge reference to an MPO graph node.
///
static void mpo_graph_node_add_edge(const int direction, const int eid, struct mpo_graph_node* node)
{
	assert(0 <= direction && direction < 2);

	if (node->num_edges[direction] == 0)
	{
		node->eids[direction] = aligned_alloc(MEM_DATA_ALIGN, sizeof(int));
		node->eids[direction][0] = eid;
	}
	else
	{
		// re-allocate memory for indices
		int* eids_prev = node->eids[direction];
		const int num = node->num_edges[direction];
		node->eids[direction] = aligned_alloc(MEM_DATA_ALIGN, (num + 1) * sizeof(int));
		memcpy(node->eids[direction], eids_prev, num * sizeof(int));
		aligned_free(eids_prev);
		node->eids[direction][num] = eid;
	}

	node->num_edges[direction]++;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an MPO graph node (free memory).
///
static void delete_mpo_graph_node(struct mpo_graph_node* node)
{
	aligned_free(node->eids[0]);
	aligned_free(node->eids[1]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate an MPO graph edge.
///
static void allocate_mpo_graph_edge(const int nopics, struct mpo_graph_edge* edge)
{
	edge->nids[0] = -1;
	edge->nids[1] = -1;
	edge->opics = aligned_calloc(MEM_DATA_ALIGN, nopics, sizeof(struct local_op_ref));
	edge->nopics = nopics;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an MPO graph edge (free memory).
///
static void delete_mpo_graph_edge(struct mpo_graph_edge* edge)
{
	aligned_free(edge->opics);
	edge->nopics = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct the local operator (as dense matrix) of an MPO graph edge.
///
void mpo_graph_edge_local_op(const struct mpo_graph_edge* edge, const struct dense_tensor* opmap, struct dense_tensor* op)
{
	assert(edge->nopics > 0);

	// first summand
	copy_dense_tensor(&opmap[edge->opics[0].oid], op);
	if (op->dtype == SINGLE_REAL || op->dtype == SINGLE_COMPLEX) {
		float alpha = (float)edge->opics[0].coeff;
		rscale_dense_tensor(&alpha, op);
	}
	else {
		assert(op->dtype == DOUBLE_REAL || op->dtype == DOUBLE_COMPLEX);
		rscale_dense_tensor(&edge->opics[0].coeff, op);
	}

	// add the other summands
	for (int i = 1; i < edge->nopics; i++)
	{
		// ensure that 'alpha' is large enough to store any numeric type
		dcomplex alpha;
		numeric_from_double(edge->opics[i].coeff, op->dtype, &alpha);
		dense_tensor_scalar_multiply_add(&alpha, &opmap[edge->opics[i].oid], op);
	}
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
	int nidl;        //!< index of left-connected node
};


//________________________________________________________________________________________________________________________
///
/// \brief Allocate an operator half-chain.
///
static void allocate_op_halfchain(const int length, struct op_halfchain* chain)
{
	chain->oids   = aligned_calloc(MEM_DATA_ALIGN, length, sizeof(int));
	chain->qnums  = aligned_calloc(MEM_DATA_ALIGN, length + 1, sizeof(qnumber));
	chain->length = length;
	chain->nidl   = -1;
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
	dst->nidl = src->nidl;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an operator half-chain (free memory).
///
static void delete_op_halfchain(struct op_halfchain* chain)
{
	aligned_free(chain->qnums);
	aligned_free(chain->oids);
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
	if (chain1->nidl != chain2->nidl) {
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
	hash = (hash ^ chain->nidl) * prime;
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
	int nidl;       //!< index of left-connected node
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
	       (node1->nidl  == node2->nidl);
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
	hash = (hash ^ node->nidl)  * prime;
	return hash;
}


//________________________________________________________________________________________________________________________
///
/// \brief Local site and half-chain partition auxiliary data structure.
///
struct site_halfchain_partition
{
	struct u_node*       ulist;  //!< array of 'U' nodes
	struct op_halfchain* vlist;  //!< array of 'V' half-chains
	double* gamma;               //!< matrix of gamma coefficients, of dimension 'num_u x num_v'
	int num_u;                   //!< number of 'U' nodes
	int num_v;                   //!< number of 'V' half-chains
};


//________________________________________________________________________________________________________________________
///
/// \brief Delete a local site and half-chain partition (free memory).
///
static void delete_site_halfchain_partition(struct site_halfchain_partition* partition)
{
	aligned_free(partition->gamma);
	for (int j = 0; j < partition->num_v; j++) {
		delete_op_halfchain(&partition->vlist[j]);
	}
	aligned_free(partition->vlist);
	aligned_free(partition->ulist);
}


//________________________________________________________________________________________________________________________
///
/// \brief Repartition half-chains after splitting off the local operators acting on the leftmost site.
///
static void site_partition_halfchains(const struct op_halfchain* chains, const double* coeffs, const int nchains, struct site_halfchain_partition* partition)
{
	memset(partition, 0, sizeof(struct site_halfchain_partition));
	// upper bound on required memory
	partition->ulist = aligned_calloc(MEM_DATA_ALIGN, nchains, sizeof(struct u_node));
	partition->vlist = aligned_calloc(MEM_DATA_ALIGN, nchains, sizeof(struct op_halfchain));

	double* gamma = aligned_calloc(MEM_DATA_ALIGN, nchains*nchains, sizeof(double));

	// use hash tables for fast look-up
	struct hash_table u_ht, v_ht;
	create_hash_table(u_node_equal, u_node_hash_func, sizeof(struct u_node), 4*nchains, &u_ht);
	create_hash_table(op_halfchain_equal, op_halfchain_hash_func, sizeof(struct op_halfchain), 4*nchains, &v_ht);

	for (int k = 0; k < nchains; k++)
	{
		const struct op_halfchain* chain = &chains[k];
		assert(chain->length >= 1);

		// U_i node
		struct u_node u = { .oid = chain->oids[0], .qnum0 = chain->qnums[0], .qnum1 = chain->qnums[1], .nidl = chain->nidl };
		int* i = hash_table_get(&u_ht, &u);
		if (i == NULL)
		{
			// insert node into array
			memcpy(&partition->ulist[partition->num_u], &u, sizeof(struct u_node));
			// insert (node, array index) into hash table
			i = aligned_alloc(MEM_DATA_ALIGN, sizeof(int));
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
			j = aligned_alloc(MEM_DATA_ALIGN, sizeof(int));
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

		// record gamma coefficient
		gamma[(*i)*nchains + (*j)] += coeffs[k];
	}

	// copy entries into final gamma matrix
	partition->gamma = aligned_calloc(MEM_DATA_ALIGN, partition->num_u*partition->num_v, sizeof(double));
	for (int i = 0; i < partition->num_u; i++) {
		memcpy(&partition->gamma[i*partition->num_v], &gamma[i*nchains], partition->num_v * sizeof(double));
	}
	aligned_free(gamma);

	delete_hash_table(&v_ht, aligned_free);
	delete_hash_table(&u_ht, aligned_free);
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct an MPO operator graph from a list of operator chains, implementing the algorithm in:
///   Jiajun Ren, Weitang Li, Tong Jiang, Zhigang Shuai
///   A general automatic method for optimal construction of matrix product operators using bipartite graph theory
///   J. Chem. Phys. 153, 084118 (2020)
///
void mpo_graph_from_opchains(const struct op_chain* chains, const int nchains, const int nsites, const int oid_identity, struct mpo_graph* mpo_graph)
{
	// need at least one site
	assert(nsites > 0);
	// list of operator chains cannot be empty
	assert(nchains > 0);

	mpo_graph->nsites = nsites;
	mpo_graph->nodes     = aligned_calloc(MEM_DATA_ALIGN, nsites + 1, sizeof(struct mpo_graph_node*));
	mpo_graph->edges     = aligned_calloc(MEM_DATA_ALIGN, nsites,     sizeof(struct mpo_graph_edge*));
	mpo_graph->num_nodes = aligned_calloc(MEM_DATA_ALIGN, nsites + 1, sizeof(int));
	mpo_graph->num_edges = aligned_calloc(MEM_DATA_ALIGN, nsites,     sizeof(int));
	// left start node
	mpo_graph->num_nodes[0] = 1;
	mpo_graph->nodes[0] = aligned_calloc(MEM_DATA_ALIGN, 1, sizeof(struct mpo_graph_node));

	// pad identities and filter out chains with zero coefficients
	int nhalfchains = 0;
	struct op_chain* chains_full = aligned_calloc(MEM_DATA_ALIGN, nchains, sizeof(struct op_chain));
	for (int k = 0; k < nchains; k++)
	{
		if (chains[k].coeff != 0)
		{
			op_chain_pad_identities(&chains[k], nsites, oid_identity, &chains_full[nhalfchains]);
			assert(chains_full[nhalfchains].length == nsites);
			nhalfchains++;
		}
	}
	// require at least one non-zero half-chain
	assert(nhalfchains > 0);

	// convert to half-chains and add a dummy identity operator
	struct op_halfchain* vlist_next = aligned_alloc(MEM_DATA_ALIGN, nhalfchains * sizeof(struct op_halfchain));
	double* coeffs_next             = aligned_alloc(MEM_DATA_ALIGN, nhalfchains * sizeof(double));
	for (int k = 0; k < nhalfchains; k++)
	{
		// dummy identity operator at the end
		allocate_op_halfchain(chains_full[k].length + 1, &vlist_next[k]);
		memcpy(vlist_next[k].oids,  chains_full[k].oids,   chains_full[k].length      * sizeof(int));
		memcpy(vlist_next[k].qnums, chains_full[k].qnums, (chains_full[k].length + 1) * sizeof(qnumber));
		vlist_next[k].oids[chains_full[k].length] = oid_identity;
		vlist_next[k].nidl = 0;

		coeffs_next[k] = chains_full[k].coeff;
	}

	for (int k = 0; k < nhalfchains; k++) {
		delete_op_chain(&chains_full[k]);
	}
	aligned_free(chains_full);

	// sweep from left to right
	for (int l = 0; l < nsites; l++)
	{
		struct site_halfchain_partition partition;
		site_partition_halfchains(vlist_next, coeffs_next, nhalfchains, &partition);

		// extract edges
		int nedges = 0;
		for (int i = 0; i < partition.num_u; i++) {
			for (int j = 0; j < partition.num_v; j++) {
				if (partition.gamma[i*partition.num_v + j] != 0) {
					nedges++;
				}
			}
		}
		struct bipartite_graph_edge* edges = aligned_alloc(MEM_DATA_ALIGN, nedges * sizeof(struct bipartite_graph_edge));
		int c = 0;
		for (int i = 0; i < partition.num_u; i++) {
			for (int j = 0; j < partition.num_v; j++) {
				if (partition.gamma[i*partition.num_v + j] != 0) {
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
		aligned_free(edges);
		bool* u_cover = aligned_calloc(MEM_DATA_ALIGN, bigraph.num_u, sizeof(bool));
		bool* v_cover = aligned_calloc(MEM_DATA_ALIGN, bigraph.num_v, sizeof(bool));
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

		aligned_free(coeffs_next);
		for (int k = 0; k < nhalfchains; k++) {
			delete_op_halfchain(&vlist_next[k]);
		}
		aligned_free(vlist_next);
		nhalfchains = 0;

		// allocate memory for next iteration, using bipartite graph 'nedges' as upper bound for number of required half-chains
		vlist_next  = aligned_alloc(MEM_DATA_ALIGN, nedges * sizeof(struct op_halfchain));
		coeffs_next = aligned_alloc(MEM_DATA_ALIGN, nedges * sizeof(double));

		// using bipartite graph 'nedges' as upper bound for number of required MPO graph edges
		mpo_graph->edges[l] = aligned_calloc(MEM_DATA_ALIGN, nedges, sizeof(struct mpo_graph_edge));
		mpo_graph->nodes[l + 1] = aligned_calloc(MEM_DATA_ALIGN, num_u_cover + num_v_cover, sizeof(struct mpo_graph_node));

		// current edge and node counter
		int ce = 0;
		int cn = 0;

		for (int i = 0; i < bigraph.num_u; i++)
		{
			if (!u_cover[i]) {
				continue;
			}

			const struct u_node* u = &partition.ulist[i];
			assert(mpo_graph->nodes[l][u->nidl].qnum == u->qnum0);

			// add a new edge
			struct mpo_graph_edge* edge = &mpo_graph->edges[l][ce];
			allocate_mpo_graph_edge(1, edge);
			edge->nids[0] = u->nidl;
			edge->opics[0].oid = u->oid;
			edge->opics[0].coeff = 1;
			// connect edge to previous node
			mpo_graph_node_add_edge(1, ce, &mpo_graph->nodes[l][u->nidl]);

			// add a new node
			struct mpo_graph_node* node = &mpo_graph->nodes[l + 1][cn];
			node->qnum = u->qnum1;
			mpo_graph_node_add_edge(0, ce, node);
			edge->nids[1] = cn;

			// assemble operator half-chains for next iteration
			for (int k = 0; k < bigraph.num_adj_u[i]; k++)
			{
				int j = bigraph.adj_u[i][k];
				// copy half-chain
				assert(partition.vlist[j].length == nsites - l);
				copy_op_halfchain(&partition.vlist[j], &vlist_next[nhalfchains]);
				vlist_next[nhalfchains].nidl = cn;
				// pass gamma coefficient
				coeffs_next[nhalfchains] = partition.gamma[i*partition.num_v + j];
				nhalfchains++;
				// avoid double-counting
				partition.gamma[i*partition.num_v + j] = 0;
			}

			ce++;
			cn++;
		}

		for (int j = 0; j < bigraph.num_v; j++)
		{
			if (!v_cover[j]) {
				continue;
			}

			// add a new node
			struct mpo_graph_node* node = &mpo_graph->nodes[l + 1][cn];
			node->qnum = partition.vlist[j].qnums[0];

			// add operator half-chain for next iteration with reference to node
			assert(partition.vlist[j].length == nsites - l);
			copy_op_halfchain(&partition.vlist[j], &vlist_next[nhalfchains]);
			vlist_next[nhalfchains].nidl = cn;
			coeffs_next[nhalfchains] = 1;
			nhalfchains++;

			// create a "complementary operator"
			for (int k = 0; k < bigraph.num_adj_v[j]; k++)
			{
				int i = bigraph.adj_v[j][k];

				if (partition.gamma[i*partition.num_v + j] == 0) {
					continue;
				}

				const struct u_node* u = &partition.ulist[i];
				assert(u->qnum1 == node->qnum);
				// add a new edge
				struct mpo_graph_edge* edge = &mpo_graph->edges[l][ce];
				allocate_mpo_graph_edge(1, edge);
				edge->nids[0] = u->nidl;
				edge->nids[1] = cn;
				edge->opics[0].oid = u->oid;
				edge->opics[0].coeff = partition.gamma[i*partition.num_v + j];
				// keep track of handled edges
				partition.gamma[i*partition.num_v + j] = 0;
				// connect edge to previous node
				mpo_graph_node_add_edge(1, ce, &mpo_graph->nodes[l][u->nidl]);
				assert(mpo_graph->nodes[l][u->nidl].qnum == u->qnum0);
				// connect edge to new node
				mpo_graph_node_add_edge(0, ce, node);
				ce++;
			}

			cn++;
		}

		// ensure that we have handled all edges
		for (int i = 0; i < partition.num_u; i++) {
			for (int j = 0; j < partition.num_v; j++) {
				assert(partition.gamma[i*partition.num_v + j] == 0);
			}
		}

		// ensure that we had allocated sufficient memory
		assert(nhalfchains <= nedges);

		assert(cn == num_u_cover + num_v_cover);
		mpo_graph->num_nodes[l + 1] = cn;
		mpo_graph->num_edges[l]     = ce;

		aligned_free(v_cover);
		aligned_free(u_cover);
		delete_bipartite_graph(&bigraph);
		delete_site_halfchain_partition(&partition);
	}

	// dummy trailing half-chain
	assert(nhalfchains == 1);
	assert(coeffs_next[0] == 1);
	aligned_free(coeffs_next);
	for (int k = 0; k < nhalfchains; k++) {
		delete_op_halfchain(&vlist_next[k]);
	}
	aligned_free(vlist_next);
	nhalfchains = 0;
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
		aligned_free(mpo_graph->edges[l]);
	}
	aligned_free(mpo_graph->edges);
	aligned_free(mpo_graph->num_edges);

	for (int l = 0; l < mpo_graph->nsites + 1; l++)
	{
		for (int i = 0; i < mpo_graph->num_nodes[l]; i++)
		{
			delete_mpo_graph_node(&mpo_graph->nodes[l][i]);
		}
		aligned_free(mpo_graph->nodes[l]);
	}
	aligned_free(mpo_graph->nodes);
	aligned_free(mpo_graph->num_nodes);
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

	// edges indexed by a node must point back to same node
	for (int l = 0; l < mpo_graph->nsites + 1; l++)
	{
		for (int i = 0; i < mpo_graph->num_nodes[l]; i++)
		{
			const struct mpo_graph_node* node = &mpo_graph->nodes[l][i];
			if (l == 0 && node->num_edges[0] != 0) {
				return false;
			}
			if (l == mpo_graph->nsites && node->num_edges[1] != 0) {
				return false;
			}
			for (int j = 0; j < node->num_edges[0]; j++)
			{
				const struct mpo_graph_edge* edge = &mpo_graph->edges[l - 1][node->eids[0][j]];
				if (edge->nids[1] != i) {
					return false;
				}
			}
			for (int j = 0; j < node->num_edges[1]; j++)
			{
				const struct mpo_graph_edge* edge = &mpo_graph->edges[l][node->eids[1][j]];
				if (edge->nids[0] != i) {
					return false;
				}
			}
		}
	}

	// nodes indexed by an edge must point back to same edge
	for (int l = 0; l < mpo_graph->nsites; l++)
	{
		for (int j = 0; j < mpo_graph->num_edges[l]; j++)
		{
			const struct mpo_graph_edge* edge = &mpo_graph->edges[l][j];
			for (int dir = 0; dir < 2; dir++)
			{
				const struct mpo_graph_node* node = &mpo_graph->nodes[l + dir][edge->nids[dir]];
				bool edge_ref = false;
				for (int i = 0; i < node->num_edges[1 - dir]; i++) {
					if (node->eids[1 - dir][i] == j) {
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
void mpo_graph_to_matrix(const struct mpo_graph* mpo_graph, const struct dense_tensor* opmap, const enum numeric_type dtype, struct dense_tensor* a)
{
	const int nsites = mpo_graph->nsites;
	assert(nsites >= 1);
	assert(mpo_graph->num_nodes[0]      == 1);
	assert(mpo_graph->num_nodes[nsites] == 1);

	struct dense_tensor* blocks[2];

	// initial 1x1 tensor
	blocks[0] = aligned_alloc(MEM_DATA_ALIGN, sizeof(struct dense_tensor));
	const long dim_init[2] = { 1, 1 };
	allocate_dense_tensor(dtype, 2, dim_init, blocks[0]);
	dense_tensor_set_identity(blocks[0]);

	// sweep from left to right
	for (int l = 0; l < nsites; l++)
	{
		blocks[(l + 1) % 2] = aligned_alloc(MEM_DATA_ALIGN, mpo_graph->num_nodes[l + 1] * sizeof(struct dense_tensor));

		for (int i = 0; i < mpo_graph->num_nodes[l + 1]; i++)
		{
			const struct mpo_graph_node* node = &mpo_graph->nodes[l + 1][i];
			assert(node->num_edges[0] > 0);
			for (int j = 0; j < node->num_edges[0]; j++)
			{
				const struct mpo_graph_edge* edge = &mpo_graph->edges[l][node->eids[0][j]];
				assert(edge->nids[1] == i);

				struct dense_tensor op;
				mpo_graph_edge_local_op(edge, opmap, &op);

				// ensure that left-connected node index is valid
				assert(0 <= edge->nids[0] && edge->nids[0] < mpo_graph->num_nodes[l]);
				if (j == 0)
				{
					dense_tensor_kronecker_product(&blocks[l % 2][edge->nids[0]], &op, &blocks[(l + 1) % 2][i]);
				}
				else
				{
					struct dense_tensor tmp;
					dense_tensor_kronecker_product(&blocks[l % 2][edge->nids[0]], &op, &tmp);
					dense_tensor_scalar_multiply_add(numeric_one(tmp.dtype), &tmp, &blocks[(l + 1) % 2][i]);
					delete_dense_tensor(&tmp);
				}

				delete_dense_tensor(&op);
			}
		}

		for (int i = 0; i < mpo_graph->num_nodes[l]; i++)
		{
			delete_dense_tensor(&blocks[l % 2][i]);
		}
		aligned_free(blocks[l % 2]);
	}

	// final single block contains result
	assert(mpo_graph->num_nodes[nsites] == 1);
	move_dense_tensor_data(&blocks[nsites % 2][0], a);
	aligned_free(blocks[nsites % 2]);
}
