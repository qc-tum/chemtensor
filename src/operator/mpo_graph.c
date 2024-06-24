/// \file mpo_graph.c
/// \brief MPO graph internal data structure for generating MPO representations.

#include <assert.h>
#include "mpo_graph.h"
#include "hash_table.h"
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
/// \brief Delete an MPO graph vertex (free memory).
///
static void delete_mpo_graph_vertex(struct mpo_graph_vertex* vertex)
{
	aligned_free(vertex->eids[0]);
	aligned_free(vertex->eids[1]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate an MPO graph edge.
///
static void allocate_mpo_graph_edge(const int nopics, struct mpo_graph_edge* edge)
{
	edge->vids[0] = -1;
	edge->vids[1] = -1;
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
	chain->oids   = aligned_calloc(MEM_DATA_ALIGN, length, sizeof(int));
	chain->qnums  = aligned_calloc(MEM_DATA_ALIGN, length + 1, sizeof(qnumber));
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
/// \brief Temporary integer tuple data structure.
///
struct integer_tuple
{
	int i;  //!< first integer
	int j;  //!< second integer
};


//________________________________________________________________________________________________________________________
///
/// \brief Test equality of two integer tuples.
///
static bool integer_tuple_equal(const void* t1, const void* t2)
{
	const struct integer_tuple* tuple1 = t1;
	const struct integer_tuple* tuple2 = t2;

	return (tuple1->i == tuple2->i) && (tuple1->j == tuple2->j);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the hash value of an integer tuple.
///
static hash_type integer_tuple_hash_func(const void* t)
{
	const struct integer_tuple* tuple = t;

	// Fowler-Noll-Vo FNV-1a (64-bit) hash function, see
	// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
	const uint64_t offset = 14695981039346656037U;
	const uint64_t prime  = 1099511628211U;
	hash_type hash = offset;
	hash = (hash ^ tuple->i) * prime;
	hash = (hash ^ tuple->j) * prime;
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

	// use a hash table for intermediate storage of gamma coefficients
	struct hash_table gamma_ht;
	create_hash_table(integer_tuple_equal, integer_tuple_hash_func, sizeof(struct integer_tuple), 4*nchains, &gamma_ht);

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
		struct integer_tuple ij = { .i = (*i), .j = (*j) };
		double* gamma = hash_table_get(&gamma_ht, &ij);
		if (gamma == NULL)
		{
			// insert coefficient into table
			gamma = aligned_alloc(MEM_DATA_ALIGN, sizeof(double));
			(*gamma) = coeffs[k];
			hash_table_insert(&gamma_ht, &ij, gamma);
		}
		else
		{
			// (i, j) tuple already exists
			(*gamma) += coeffs[k];
		}
	}

	// copy entries into final gamma matrix
	partition->gamma = aligned_calloc(MEM_DATA_ALIGN, partition->num_u*partition->num_v, sizeof(double));
	struct hash_table_iterator iter;
	for (init_hash_table_iterator(&gamma_ht, &iter); hash_table_iterator_is_valid(&iter); hash_table_iterator_next(&iter))
	{
		const struct integer_tuple* tuple = hash_table_iterator_get_key(&iter);
		const double* gamma = hash_table_iterator_get_value(&iter);
		partition->gamma[tuple->i*partition->num_v + tuple->j] = (*gamma);
	}
	delete_hash_table(&gamma_ht, aligned_free);

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
	mpo_graph->verts     = aligned_calloc(MEM_DATA_ALIGN, nsites + 1, sizeof(struct mpo_graph_vertex*));
	mpo_graph->edges     = aligned_calloc(MEM_DATA_ALIGN, nsites,     sizeof(struct mpo_graph_edge*));
	mpo_graph->num_verts = aligned_calloc(MEM_DATA_ALIGN, nsites + 1, sizeof(int));
	mpo_graph->num_edges = aligned_calloc(MEM_DATA_ALIGN, nsites,     sizeof(int));
	// left start vertex
	mpo_graph->num_verts[0] = 1;
	mpo_graph->verts[0] = aligned_calloc(MEM_DATA_ALIGN, 1, sizeof(struct mpo_graph_vertex));

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
		vlist_next[k].vidl = 0;

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
		mpo_graph->verts[l + 1] = aligned_calloc(MEM_DATA_ALIGN, num_u_cover + num_v_cover, sizeof(struct mpo_graph_vertex));

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
			edge->opics[0].coeff = 1;
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
				// pass gamma coefficient
				coeffs_next[nhalfchains] = partition.gamma[i*partition.num_v + j];
				nhalfchains++;
				// avoid double-counting
				partition.gamma[i*partition.num_v + j] = 0;
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
				assert(u->qnum1 == vertex->qnum);
				// add a new edge
				struct mpo_graph_edge* edge = &mpo_graph->edges[l][ce];
				allocate_mpo_graph_edge(1, edge);
				edge->vids[0] = u->vidl;
				edge->vids[1] = cv;
				edge->opics[0].oid = u->oid;
				edge->opics[0].coeff = partition.gamma[i*partition.num_v + j];
				// keep track of handled edges
				partition.gamma[i*partition.num_v + j] = 0;
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
				assert(partition.gamma[i*partition.num_v + j] == 0);
			}
		}

		// ensure that we had allocated sufficient memory
		assert(nhalfchains <= nedges);

		assert(cv == num_u_cover + num_v_cover);
		mpo_graph->num_verts[l + 1] = cv;
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
		for (int i = 0; i < mpo_graph->num_verts[l]; i++)
		{
			delete_mpo_graph_vertex(&mpo_graph->verts[l][i]);
		}
		aligned_free(mpo_graph->verts[l]);
	}
	aligned_free(mpo_graph->verts);
	aligned_free(mpo_graph->num_verts);
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
void mpo_graph_to_matrix(const struct mpo_graph* mpo_graph, const struct dense_tensor* opmap, const enum numeric_type dtype, struct dense_tensor* mat)
{
	const int nsites = mpo_graph->nsites;
	assert(nsites >= 1);
	assert(mpo_graph->num_verts[0]      == 1);
	assert(mpo_graph->num_verts[nsites] == 1);

	struct dense_tensor* blocks[2];

	// initial 1x1 tensor
	blocks[0] = aligned_alloc(MEM_DATA_ALIGN, sizeof(struct dense_tensor));
	const long dim_init[2] = { 1, 1 };
	allocate_dense_tensor(dtype, 2, dim_init, blocks[0]);
	dense_tensor_set_identity(blocks[0]);

	// sweep from left to right
	for (int l = 0; l < nsites; l++)
	{
		blocks[(l + 1) % 2] = aligned_alloc(MEM_DATA_ALIGN, mpo_graph->num_verts[l + 1] * sizeof(struct dense_tensor));

		for (int i = 0; i < mpo_graph->num_verts[l + 1]; i++)
		{
			const struct mpo_graph_vertex* vertex = &mpo_graph->verts[l + 1][i];
			assert(vertex->num_edges[0] > 0);
			for (int j = 0; j < vertex->num_edges[0]; j++)
			{
				const struct mpo_graph_edge* edge = &mpo_graph->edges[l][vertex->eids[0][j]];
				assert(edge->vids[1] == i);

				struct dense_tensor op;
				construct_local_operator(edge->opics, edge->nopics, opmap, &op);

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
		aligned_free(blocks[l % 2]);
	}

	// final single block contains result
	assert(mpo_graph->num_verts[nsites] == 1);
	move_dense_tensor_data(&blocks[nsites % 2][0], mat);
	aligned_free(blocks[nsites % 2]);
}
