/// \file ttno_graph.c
/// \brief Tree tensor network operator (TTNO) graph internal data structure.

#include "ttno_graph.h"
#include "hash_table.h"
#include "integer_linear_algebra.h"
#include "bipartite_graph.h"
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
/// \brief Allocate memory for a TTNO graph hyperedge.
///
static void allocate_ttno_graph_hyperedge(const int order, const int nopics, struct ttno_graph_hyperedge* edge)
{
	edge->vids   = aligned_alloc(MEM_DATA_ALIGN, order  * sizeof(int));
	edge->opics  = aligned_alloc(MEM_DATA_ALIGN, nopics * sizeof(struct local_op_ref));
	edge->order  = order;
	edge->nopics = nopics;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a TTNO graph hyperedge (free memory).
///
static void delete_ttno_graph_hyperedge(struct ttno_graph_hyperedge* edge)
{
	aligned_free(edge->opics);
	aligned_free(edge->vids);
	edge->nopics = 0;
	edge->order  = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Operator cluster (list of local operators acting on a subset of sites); temporary data structure for building a TTNO graph from a list of operator chains.
///
struct op_cluster
{
	int* oids;       //!< list of local op_i operator IDs, enumerated by increasing site index
	qnumber* qnums;  //!< bond quantum numbers
	int* vids;       //!< indices of connected vertices (towards inactive sites)
	int size;        //!< number of local operators
	int nbonds;      //!< number of bonds
	int nverts;      //!< number of connected vertices
};


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for an operator cluster.
///
static void allocate_op_cluster(const int size, const int nbonds, const int nverts, struct op_cluster* cluster)
{
	cluster->oids  = aligned_alloc(MEM_DATA_ALIGN, size   * sizeof(int));
	cluster->qnums = aligned_alloc(MEM_DATA_ALIGN, nbonds * sizeof(qnumber));
	cluster->vids  = aligned_alloc(MEM_DATA_ALIGN, nverts * sizeof(int));

	cluster->size   = size;
	cluster->nbonds = nbonds;
	cluster->nverts = nverts;
}


//________________________________________________________________________________________________________________________
///
/// \brief Copy an operator cluster, allocating memory for the copy.
///
static void copy_op_cluster(const struct op_cluster* restrict src, struct op_cluster* restrict dst)
{
	allocate_op_cluster(src->size, src->nbonds, src->nverts, dst);
	memcpy(dst->oids,  src->oids,  src->size   * sizeof(int));
	memcpy(dst->qnums, src->qnums, src->nbonds * sizeof(qnumber));
	memcpy(dst->vids,  src->vids,  src->nverts * sizeof(int));
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an operator cluster (free memory).
///
static void delete_op_cluster(struct op_cluster* cluster)
{
	aligned_free(cluster->oids);
	aligned_free(cluster->qnums);
	aligned_free(cluster->vids);
}


//________________________________________________________________________________________________________________________
///
/// \brief Test equality of two operator clusters.
///
static bool op_cluster_equal(const void* c1, const void* c2)
{
	const struct op_cluster* cluster1 = c1;
	const struct op_cluster* cluster2 = c2;

	if (cluster1->size != cluster2->size) {
		return false;
	}
	if (cluster1->nbonds != cluster2->nbonds) {
		return false;
	}
	if (cluster1->nverts != cluster2->nverts) {
		return false;
	}
	for (int i = 0; i < cluster1->size; i++) {
		if (cluster1->oids[i] != cluster2->oids[i]) {
			return false;
		}
	}
	for (int i = 0; i < cluster1->nbonds; i++) {
		if (cluster1->qnums[i] != cluster2->qnums[i]) {
			return false;
		}
	}
	for (int i = 0; i < cluster1->nverts; i++) {
		if (cluster1->vids[i] != cluster2->vids[i]) {
			return false;
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the hash value of an operator half-chain.
///
static hash_type op_cluster_hash_func(const void* c)
{
	const struct op_cluster* cluster = c;

	// Fowler-Noll-Vo FNV-1a (64-bit) hash function, see
	// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
	const uint64_t offset = 14695981039346656037U;
	const uint64_t prime  = 1099511628211U;
	hash_type hash = offset;
	for (int i = 0; i < cluster->size; i++) {
		hash = (hash ^ cluster->oids[i]) * prime;
	}
	for (int i = 0; i < cluster->nbonds; i++) {
		hash = (hash ^ cluster->qnums[i]) * prime;
	}
	for (int i = 0; i < cluster->nverts; i++) {
		hash = (hash ^ cluster->vids[i]) * prime;
	}
	return hash;
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert an edge tuple index (i, j) to a linear vertex array index.
///
static inline int edge_to_vertex_index(const int nsites, const int i, const int j)
{
	assert(i != j);
	return i < j ? (i*nsites + j) : (j*nsites + i);
}


//________________________________________________________________________________________________________________________
///
/// \brief Operator cluster assembly: list of operator clusters and corresponding meta-information.
///
struct op_cluster_assembly
{
	struct op_cluster* clusters;            //!< array of operator clusters
	const struct abstract_graph* topology;  //!< reference to overall graph topology
	bool* active_sites;                     //!< truth table of active sites acted on by the local operators
	int* qnum_index_map;                    //!< map from edges indexed by site tuples (i, j) to bond quantum number array index
	int* vid_index_map;                     //!< map from edges indexed by site tuples (i, j) to connected vertex ID array index
	int nclusters;                          //!< number of clusters
};


//________________________________________________________________________________________________________________________
///
/// \brief Delete an operator cluster assembly (free memory).
///
static void delete_op_cluster_assembly(struct op_cluster_assembly* assembly)
{
	for (int i = 0; i < assembly->nclusters; i++) {
		delete_op_cluster(&assembly->clusters[i]);
	}
	aligned_free(assembly->clusters);
	assembly->nclusters = 0;

	aligned_free(assembly->active_sites);
	aligned_free(assembly->qnum_index_map);
	aligned_free(assembly->vid_index_map);
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct the map from an edge indexed by site tuples (i, j) to a bond quantum number array index, given a subset of active sites.
///
static int construct_qnumber_index_map(const struct abstract_graph* topology, const bool* active_sites, int* map)
{
	for (int iv = 0; iv < topology->num_nodes * topology->num_nodes; iv++) {
		map[iv] = -1;
	}

	int c = 0;
	for (int l = 0; l < topology->num_nodes; l++)
	{
		if (!active_sites[l]) {
			continue;
		}

		for (int n = 0; n < topology->num_neighbors[l]; n++)
		{
			const int k = topology->neighbor_map[l][n];
			// avoid double-counting
			if (k > l && active_sites[k]) {
				continue;
			}

			const int iv = edge_to_vertex_index(topology->num_nodes, k, l);
			assert(map[iv] == -1);
			map[iv] = c++;
		}
	}

	return c;
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct the map from an edge indexed by site tuples (i, j) to the connected vertex ID array index, given a subset of active sites.
///
static int construct_vid_index_map(const struct abstract_graph* topology, const bool* active_sites, int* map)
{
	for (int iv = 0; iv < topology->num_nodes * topology->num_nodes; iv++) {
		map[iv] = -1;
	}

	int c = 0;
	for (int l = 0; l < topology->num_nodes; l++)
	{
		if (!active_sites[l]) {
			continue;
		}

		for (int n = 0; n < topology->num_neighbors[l]; n++)
		{
			const int k = topology->neighbor_map[l][n];
			// only take edges between active and inactive sites into account
			if (active_sites[k]) {
				continue;
			}

			const int iv = edge_to_vertex_index(topology->num_nodes, k, l);
			assert(map[iv] == -1);
			map[iv] = c++;
		}
	}

	return c;
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct an operator cluster assembly from a list of operator chains.
///
static int cluster_assembly_from_opchains(const struct op_chain* chains, const int nchains, const int oid_identity, const struct abstract_graph* topology, struct op_cluster_assembly* assembly, double* restrict coeffs)
{
	// need at least one site
	assert(topology->num_nodes >= 1);
	// expecting a tree
	assert(abstract_graph_is_connected_tree(topology));
	// list of operator chains cannot be empty
	assert(nchains >= 1);

	// copy pointer to topology
	assembly->topology = topology;

	const int nsites = topology->num_nodes;

	// mark all sites as active
	assembly->active_sites = aligned_alloc(MEM_DATA_ALIGN, nsites * sizeof(bool));
	for (int l = 0; l < nsites; l++) {
		assembly->active_sites[l] = true;
	}

	// transform operator chain quantum numbers to quantum numbers of tree layout
	int c = 0;
	int* m_qnums = aligned_calloc(MEM_DATA_ALIGN, nsites * (nsites - 1), sizeof(int));
	// note: require same enumeration order as in 'construct_qnumber_index_map'
	for (int l = 0; l < nsites; l++)
	{
		for (int n = 0; n < topology->num_neighbors[l]; n++)
		{
			const int k = topology->neighbor_map[l][n];
			// avoid double-counting
			if (k > l) {
				continue;
			}

			m_qnums[l*(nsites - 1) + c] = TENSOR_AXIS_OUT;
			m_qnums[k*(nsites - 1) + c] = TENSOR_AXIS_IN;
			c++;
		}
	}
	assert(c == nsites - 1);

	// ignoring last row (otherwise linear system would be over-determined)
	int* h_qnums = aligned_alloc(MEM_DATA_ALIGN, (nsites - 1) * (nsites - 1) * sizeof(int));
	int* u_qnums = aligned_alloc(MEM_DATA_ALIGN, (nsites - 1) * (nsites - 1) * sizeof(int));
	if (integer_hermite_normal_form(nsites - 1, m_qnums, h_qnums, u_qnums) < 0) {
		fprintf(stderr, "Hermite decomposition of quantum number map for given tree topology encountered singular matrix\n");
		return -1;
	}

	assembly->clusters = aligned_calloc(MEM_DATA_ALIGN, nchains, sizeof(struct op_cluster));
	c = 0;
	for (int k = 0; k < nchains; k++)
	{
		// filter out chains with zero coefficients
		if (chains[k].coeff == 0) {
			continue;
		}

		// pad identities
		struct op_chain pad_chain;
		op_chain_pad_identities(&chains[k], nsites, oid_identity, &pad_chain);
		assert(pad_chain.length == nsites);
		// require zero leading and trailing quantum numbers
		assert(pad_chain.qnums[0] == 0 && pad_chain.qnums[nsites] == 0);

		// copy operator IDs
		assembly->clusters[c].oids = aligned_alloc(MEM_DATA_ALIGN, nsites * sizeof(int));
		memcpy(assembly->clusters[c].oids, pad_chain.oids, nsites * sizeof(int));
		assembly->clusters[c].size = nsites;

		// quantum numbers
		qnumber* dq = aligned_alloc(MEM_DATA_ALIGN, nsites * sizeof(qnumber));
		for (int l = 0; l < nsites; l++) {
			// local quantum number for site 'l'
			dq[l] = pad_chain.qnums[l] - pad_chain.qnums[l + 1];
		}
		qnumber* u_dq = aligned_alloc(MEM_DATA_ALIGN, (nsites - 1) * sizeof(qnumber));
		// skipping last quantum number difference to avoid over-determined linear system
		integer_gemv(nsites - 1, nsites - 1, u_qnums, dq, u_dq);
		assembly->clusters[c].qnums = aligned_alloc(MEM_DATA_ALIGN, (nsites - 1) * sizeof(qnumber));
		if (integer_backsubstitute(h_qnums, nsites - 1, u_dq, assembly->clusters[c].qnums) < 0) {
			fprintf(stderr, "integer backsubstitution for obtaining quantum number assignment for given tree topology failed\n");
			return -1;
		}
		aligned_free(u_dq);
		aligned_free(dq);
		assembly->clusters[c].nbonds = nsites - 1;

		// no inactive sites, hence no connected vertices
		assembly->clusters[c].vids = aligned_alloc(MEM_DATA_ALIGN, sizeof(int));  // allocate a dummy block
		assembly->clusters[c].nverts = 0;

		// record coefficient
		coeffs[c] = pad_chain.coeff;

		c++;

		delete_op_chain(&pad_chain);
	}
	// require at least one non-zero chain
	assert(c > 0);
	assembly->nclusters = c;

	// meta-information
	assembly->qnum_index_map = aligned_alloc(MEM_DATA_ALIGN, nsites*nsites * sizeof(int));
	assembly->vid_index_map  = aligned_alloc(MEM_DATA_ALIGN, nsites*nsites * sizeof(int));
	construct_qnumber_index_map(topology, assembly->active_sites, assembly->qnum_index_map);
	construct_vid_index_map(topology, assembly->active_sites, assembly->vid_index_map);

	aligned_free(u_qnums);
	aligned_free(h_qnums);
	aligned_free(m_qnums);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Store the information of a 'U' node.
///
/// Temporary data structure for building a TTNO graph from a list of operator chains.
///
struct u_node
{
	int oid;         //!< local operator ID
	qnumber* qnums;  //!< quantum numbers (array of length 'order', sorted by neighbor site index)
	int* vids;       //!< indices of connected vertices (array of length 'order - 1')
	int order;       //!< order (number of neighboring sites)
};


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for a 'U' node.
///
static void allocate_u_node(const int order, struct u_node* node)
{
	assert(order >= 1);
	node->qnums = aligned_alloc(MEM_DATA_ALIGN, order * sizeof(qnumber));
	node->vids  = aligned_alloc(MEM_DATA_ALIGN, (order - 1) * sizeof(int));
	node->order = order;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a 'U' node (free memory).
///
static void delete_u_node(struct u_node* node)
{
	aligned_free(node->vids);
	aligned_free(node->qnums);
	node->order = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Test equality of two 'U' nodes.
///
static bool u_node_equal(const void* n1, const void* n2)
{
	const struct u_node* node1 = n1;
	const struct u_node* node2 = n2;

	if (node1->oid != node2->oid) {
		return false;
	}
	if (node1->order != node2->order) {
		return false;
	}
	for (int i = 0; i < node1->order; i++) {
		if (node1->qnums[i] != node2->qnums[i]) {
			return false;
		}
	}
	for (int i = 0; i < node1->order - 1; i++) {
		if (node1->vids[i] != node2->vids[i]) {
			return false;
		}
	}

	return true;
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
	hash = (hash ^ node->oid) * prime;
	for (int i = 0; i < node->order; i++) {
		hash = (hash ^ node->qnums[i]) * prime;
	}
	for (int i = 0; i < node->order - 1; i++) {
		hash = (hash ^ node->vids[i]) * prime;
	}
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
/// \brief Local site and cluster partition auxiliary data structure.
///
struct site_cluster_partition
{
	struct u_node* ulist;                      //!< array of 'U' nodes
	int num_u;                                 //!< number of 'U' nodes
	struct op_cluster_assembly part_assembly;  //!< operator cluster assembly on subset of sites
	double* gamma;                             //!< matrix of gamma coefficients, of dimension 'num U x num V'
};


//________________________________________________________________________________________________________________________
///
/// \brief Delete a local site and cluster partition (free memory).
///
static void delete_site_cluster_partition(struct site_cluster_partition* partition)
{
	aligned_free(partition->gamma);
	delete_op_cluster_assembly(&partition->part_assembly);
	for (int i = 0; i < partition->num_u; i++) {
		delete_u_node(&partition->ulist[i]);
	}
	aligned_free(partition->ulist);
}


//________________________________________________________________________________________________________________________
///
/// \brief Repartition operator clusters after splitting off the local operators acting on a single site.
///
static void site_partition_clusters(const struct op_cluster_assembly* assembly, const double* coeffs, const int i_site, struct site_cluster_partition* partition)
{
	// must be an active site
	assert(assembly->active_sites[i_site]);

	const struct abstract_graph* topology = assembly->topology;
	const int nsites = topology->num_nodes;

	// size (number of active sites) in assembly
	int size = 0;
	for (int l = 0; l < nsites; l++) {
		if (assembly->active_sites[l]) {
			size++;
		}
	}
	assert(size >= 2);

	// operator array index corresponding to 'i_site' inside cluster
	int i_site_cluster = 0;
	for (int l = 0; l < i_site; l++) {
		if (assembly->active_sites[l]) {
			i_site_cluster++;
		}
	}

	memset(partition, 0, sizeof(struct site_cluster_partition));
	// upper bound on required memory
	partition->ulist = aligned_calloc(MEM_DATA_ALIGN, assembly->nclusters, sizeof(struct u_node));
	partition->part_assembly.clusters = aligned_calloc(MEM_DATA_ALIGN, assembly->nclusters, sizeof(struct op_cluster));

	// copy pointer to topology
	partition->part_assembly.topology = topology;
	// new active sites
	partition->part_assembly.active_sites = aligned_calloc(MEM_DATA_ALIGN, nsites, sizeof(bool));
	for (int l = 0; l < nsites; l++) {
		if (assembly->active_sites[l] && l != i_site) {
			partition->part_assembly.active_sites[l] = true;
		}
	}
	// meta-information
	partition->part_assembly.qnum_index_map = aligned_alloc(MEM_DATA_ALIGN, nsites*nsites * sizeof(int));
	partition->part_assembly.vid_index_map  = aligned_alloc(MEM_DATA_ALIGN, nsites*nsites * sizeof(int));
	const int part_nbonds = construct_qnumber_index_map(topology, partition->part_assembly.active_sites, partition->part_assembly.qnum_index_map);
	const int part_nverts = construct_vid_index_map(topology, partition->part_assembly.active_sites, partition->part_assembly.vid_index_map);

	// use a hash table for intermediate storage of gamma coefficients
	struct hash_table gamma_ht;
	create_hash_table(integer_tuple_equal, integer_tuple_hash_func, sizeof(struct integer_tuple), 4*assembly->nclusters, &gamma_ht);

	// use hash tables for fast look-up
	struct hash_table u_ht, v_ht;
	create_hash_table(u_node_equal, u_node_hash_func, sizeof(struct u_node), 4*assembly->nclusters, &u_ht);
	create_hash_table(op_cluster_equal, op_cluster_hash_func, sizeof(struct op_cluster), 4*assembly->nclusters, &v_ht);

	for (int m = 0; m < assembly->nclusters; m++)
	{
		const struct op_cluster* cluster = &assembly->clusters[m];
		assert(cluster->size == size);

		// U_i node
		struct u_node u;
		allocate_u_node(topology->num_neighbors[i_site], &u);
		u.oid = cluster->oids[i_site_cluster];
		int c = 0;
		for (int n = 0; n < topology->num_neighbors[i_site]; n++)
		{
			const int k = topology->neighbor_map[i_site][n];
			const int iv = edge_to_vertex_index(nsites, k, i_site);

			assert(assembly->qnum_index_map[iv] >= 0);
			u.qnums[n] = cluster->qnums[assembly->qnum_index_map[iv]];

			if (!assembly->active_sites[k]) {
				// only take edges between active and inactive sites into account
				u.vids[c++] = cluster->vids[assembly->vid_index_map[iv]];
			}
		}
		// 'u' must be connected to exactly one active site
		assert(c == u.order - 1);
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
			// node already exists
			delete_u_node(&u);
			assert(*i < partition->num_u);
		}

		// V_j node: remainder of input cluster
		struct op_cluster v;
		allocate_op_cluster(size - 1, part_nbonds, part_nverts, &v);
		c = 0;
		for (int l = 0; l < size; l++) {
			if (l != i_site_cluster) {
				v.oids[c++] = cluster->oids[l];
			}
		}
		assert(c == size - 1);
		for (int l = 0; l < nsites; l++)
		{
			if (!partition->part_assembly.active_sites[l]) {
				continue;
			}
			for (int n = 0; n < topology->num_neighbors[l]; n++)
			{
				const int k = topology->neighbor_map[l][n];
				const int iv = edge_to_vertex_index(topology->num_nodes, k, l);
				v.qnums[partition->part_assembly.qnum_index_map[iv]] = cluster->qnums[assembly->qnum_index_map[iv]];
				if (!partition->part_assembly.active_sites[k])
				{
					// edge between active and inactive sites -> assign vertex index
					// if 'k' corresponds to the split-off node, the vertex index will be assigned later, and we set it to -1 for now
					v.vids[partition->part_assembly.vid_index_map[iv]] = (k == i_site ? -1 : cluster->vids[assembly->vid_index_map[iv]]);
				}
			}
		}
		int* j = hash_table_get(&v_ht, &v);
		if (j == NULL)
		{
			// insert cluster into array
			memcpy(&partition->part_assembly.clusters[partition->part_assembly.nclusters], &v, sizeof(struct op_cluster));
			// insert (cluster, array index) into hash table
			j = aligned_alloc(MEM_DATA_ALIGN, sizeof(int));
			*j = partition->part_assembly.nclusters;
			hash_table_insert(&v_ht, &v, j);
			partition->part_assembly.nclusters++;
		}
		else
		{
			// cluster already exists
			delete_op_cluster(&v);
			assert(*j < partition->part_assembly.nclusters);
		}

		// record gamma coefficient
		struct integer_tuple ij = { .i = (*i), .j = (*j) };
		double* gamma = hash_table_get(&gamma_ht, &ij);
		if (gamma == NULL)
		{
			// insert coefficient into table
			gamma = aligned_alloc(MEM_DATA_ALIGN, sizeof(double));
			(*gamma) = coeffs[m];
			hash_table_insert(&gamma_ht, &ij, gamma);
		}
		else
		{
			// (i, j) tuple already exists
			(*gamma) += coeffs[m];
		}
	}

	// copy entries into final gamma matrix
	partition->gamma = aligned_calloc(MEM_DATA_ALIGN, partition->num_u*partition->part_assembly.nclusters, sizeof(double));
	struct hash_table_iterator iter;
	for (init_hash_table_iterator(&gamma_ht, &iter); hash_table_iterator_is_valid(&iter); hash_table_iterator_next(&iter))
	{
		const struct integer_tuple* tuple = hash_table_iterator_get_key(&iter);
		const double* gamma = hash_table_iterator_get_value(&iter);
		partition->gamma[tuple->i*partition->part_assembly.nclusters + tuple->j] = (*gamma);
	}
	delete_hash_table(&gamma_ht, aligned_free);

	delete_hash_table(&v_ht, aligned_free);
	delete_hash_table(&u_ht, aligned_free);
}


//________________________________________________________________________________________________________________________
///
/// \brief Temporary data structure for sorting sites by distance from (temporary) root.
///
struct site_distance_tuple
{
	int i_site;    //!< site index
	int i_parent;  //!< parent site index
	int distance;  //!< distance from root
};


//________________________________________________________________________________________________________________________
///
/// \brief Comparison function for sorting.
///
static int compare_site_distance_tuples(const void* a, const void* b)
{
	const struct site_distance_tuple* x = (const struct site_distance_tuple*)a;
	const struct site_distance_tuple* y = (const struct site_distance_tuple*)b;

	// sort by distance in descending order
	if (x->distance > y->distance) {
		return -1;
	}
	if (x->distance < y->distance) {
		return 1;
	}
	// distances are equal; sort by site index
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
/// \brief Enumerate site-distance tuples by a recursive tree traversal.
///
static void enumerate_site_distance_tuples(const struct abstract_graph* topology, const int i_site, const int i_parent, const int distance, struct site_distance_tuple* tuples)
{
	// current tuple
	tuples[i_site].i_site   = i_site;
	tuples[i_site].i_parent = i_parent;
	tuples[i_site].distance = distance;

	for (int n = 0; n < topology->num_neighbors[i_site]; n++)
	{
		const int k = topology->neighbor_map[i_site][n];
		if (k == i_parent) {
			continue;
		}

		// recurse function call with parent 'i_site'
		enumerate_site_distance_tuples(topology, k, i_site, distance + 1, tuples);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct a TTNO operator graph from a list of operator chains.
///
int ttno_graph_from_opchains(const struct op_chain* chains, const int nchains, const struct abstract_graph* topology, const int oid_identity, struct ttno_graph* ttno_graph)
{
	// need at least one site
	assert(topology->num_nodes > 0);
	// expecting a tree
	assert(abstract_graph_is_connected_tree(topology));
	// list of operator chains cannot be empty
	assert(nchains > 0);

	const int nsites = topology->num_nodes;

	copy_abstract_graph(topology, &ttno_graph->topology);
	ttno_graph->edges     = aligned_calloc(MEM_DATA_ALIGN, nsites,        sizeof(struct ttno_graph_edge*));
	ttno_graph->verts     = aligned_calloc(MEM_DATA_ALIGN, nsites*nsites, sizeof(struct ttno_graph_vertex*));
	ttno_graph->num_edges = aligned_calloc(MEM_DATA_ALIGN, nsites,        sizeof(int));
	ttno_graph->num_verts = aligned_calloc(MEM_DATA_ALIGN, nsites*nsites, sizeof(int));
	ttno_graph->nsites = nsites;

	// initial operator cluster assembly
	struct op_cluster_assembly assembly;
	double* coeffs_next = aligned_alloc(MEM_DATA_ALIGN, nchains * sizeof(double));
	if (cluster_assembly_from_opchains(chains, nchains, oid_identity, topology, &assembly, coeffs_next) < 0) {
		return -1;
	}

	// select site with maximum number of neighbors as root
	int i_root = 0;
	for (int l = 1; l < topology->num_nodes; l++) {
		if (topology->num_neighbors[l] > topology->num_neighbors[i_root]) {
			i_root = l;
		}
	}
	struct site_distance_tuple* sd = aligned_alloc(MEM_DATA_ALIGN, nsites * sizeof(struct site_distance_tuple));
	enumerate_site_distance_tuples(topology, i_root, -1, 0, sd);
	// sort by distance from root in descending order
	qsort(sd, nsites, sizeof(struct site_distance_tuple), compare_site_distance_tuples);
	assert(sd[nsites - 1].i_site == i_root);

	// iterate over sites by decreasing distance from root
	for (int l = 0; l < nsites - 1; l++)
	{
		const int i_site   = sd[l].i_site;
		const int i_parent = sd[l].i_parent;

		struct site_cluster_partition partition;
		site_partition_clusters(&assembly, coeffs_next, i_site, &partition);

		// extract bipartite graph edges
		int nedges = 0;
		for (int i = 0; i < partition.num_u; i++) {
			for (int j = 0; j < partition.part_assembly.nclusters; j++) {
				if (partition.gamma[i*partition.part_assembly.nclusters + j] != 0) {
					nedges++;
				}
			}
		}
		struct bipartite_graph_edge* edges = aligned_alloc(MEM_DATA_ALIGN, nedges * sizeof(struct bipartite_graph_edge));
		int c = 0;
		for (int i = 0; i < partition.num_u; i++) {
			for (int j = 0; j < partition.part_assembly.nclusters; j++) {
				if (partition.gamma[i*partition.part_assembly.nclusters + j] != 0) {
					edges[c].u = i;
					edges[c].v = j;
					c++;
				}
			}
		}
		assert(c == nedges);
		// construct bipartite graph and find a minimum vertex cover
		struct bipartite_graph bigraph;
		init_bipartite_graph(partition.num_u, partition.part_assembly.nclusters, edges, nedges, &bigraph);
		aligned_free(edges);
		bool* u_cover = aligned_calloc(MEM_DATA_ALIGN, bigraph.num_u, sizeof(bool));
		bool* v_cover = aligned_calloc(MEM_DATA_ALIGN, bigraph.num_v, sizeof(bool));
		if (partition.part_assembly.nclusters == 1) {
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
		delete_op_cluster_assembly(&assembly);

		// allocate memory for next iteration, using bipartite graph 'nedges' as upper bound for number of required operator clusters
		assembly.clusters  = aligned_alloc(MEM_DATA_ALIGN, nedges * sizeof(struct op_cluster));
		assembly.nclusters = 0;
		// copy meta-information from partition
		assembly.topology = topology;
		assembly.active_sites   = aligned_alloc(MEM_DATA_ALIGN, nsites * sizeof(bool));
		assembly.qnum_index_map = aligned_alloc(MEM_DATA_ALIGN, nsites*nsites * sizeof(int));
		assembly.vid_index_map  = aligned_alloc(MEM_DATA_ALIGN, nsites*nsites * sizeof(int));
		memcpy(assembly.active_sites,   partition.part_assembly.active_sites,   nsites * sizeof(bool));
		memcpy(assembly.qnum_index_map, partition.part_assembly.qnum_index_map, nsites*nsites * sizeof(int));
		memcpy(assembly.vid_index_map,  partition.part_assembly.vid_index_map,  nsites*nsites * sizeof(int));

		coeffs_next = aligned_alloc(MEM_DATA_ALIGN, nedges * sizeof(double));

		// using bipartite graph 'nedges' as upper bound for number of required TTNO graph hyperedges
		ttno_graph->edges[i_site] = aligned_calloc(MEM_DATA_ALIGN, nedges, sizeof(struct ttno_graph_hyperedge));
		const int iv = edge_to_vertex_index(nsites, i_site, i_parent);
		ttno_graph->verts[iv] = aligned_calloc(MEM_DATA_ALIGN, num_u_cover + num_v_cover, sizeof(struct ttno_graph_vertex));

		// current edge and vertex counter
		int ce = 0;
		int cv = 0;

		for (int i = 0; i < bigraph.num_u; i++)
		{
			if (!u_cover[i]) {
				continue;
			}

			const struct u_node* u = &partition.ulist[i];
			assert(u->order == topology->num_neighbors[i_site]);

			// add a new TTNO hyperedge and vertex
			struct ttno_graph_hyperedge* edge = &ttno_graph->edges[i_site][ce];
			allocate_ttno_graph_hyperedge(u->order, 1, edge);
			struct ttno_graph_vertex* vertex = &ttno_graph->verts[iv][cv];
			for (int n = 0; n < topology->num_neighbors[i_site]; n++)
			{
				const int k = topology->neighbor_map[i_site][n];
				if (k == i_parent) {
					edge->vids[n] = cv;
					vertex->qnum = u->qnums[n];
				}
				else {
					edge->vids[n] = u->vids[k < i_parent ? n : n - 1];
					// connect edge to neighboring vertex (away from root)
					const int ivd = edge_to_vertex_index(nsites, k, i_site);
					assert(edge->vids[n] < ttno_graph->num_verts[ivd]);
					assert(ttno_graph->verts[ivd][edge->vids[n]].qnum == u->qnums[n]);
					ttno_graph_vertex_add_edge(k < i_site ? 1 : 0, ce, &ttno_graph->verts[ivd][edge->vids[n]]);
				}
			}
			edge->opics[0].oid = u->oid;
			edge->opics[0].coeff = 1;
			ttno_graph_vertex_add_edge(i_site < i_parent ? 0 : 1, ce, vertex);

			// assemble operator clusters for next iteration
			for (int k = 0; k < bigraph.num_adj_u[i]; k++)
			{
				int j = bigraph.adj_u[i][k];
				// copy cluster
				copy_op_cluster(&partition.part_assembly.clusters[j], &assembly.clusters[assembly.nclusters]);
				assembly.clusters[assembly.nclusters].vids[assembly.vid_index_map[iv]] = cv;
				// pass gamma coefficient
				coeffs_next[assembly.nclusters] = partition.gamma[i*partition.part_assembly.nclusters + j];
				assembly.nclusters++;
				// avoid double-counting
				partition.gamma[i*partition.part_assembly.nclusters + j] = 0;
			}

			ce++;
			cv++;
		}

		int neigh_parent_index = -1;
		for (int n = 0; n < topology->num_neighbors[i_site]; n++)
		{
			const int k = topology->neighbor_map[i_site][n];
			if (k == i_parent) {
				neigh_parent_index = n;
				break;
			}
		}
		assert(neigh_parent_index != -1);

		for (int j = 0; j < bigraph.num_v; j++)
		{
			if (!v_cover[j]) {
				continue;
			}

			// add a new TTNO vertex
			struct ttno_graph_vertex* vertex = &ttno_graph->verts[iv][cv];
			vertex->qnum = partition.part_assembly.clusters[j].qnums[partition.part_assembly.qnum_index_map[iv]];

			// add operator cluster for next iteration with reference to vertex
			copy_op_cluster(&partition.part_assembly.clusters[j], &assembly.clusters[assembly.nclusters]);
			assembly.clusters[assembly.nclusters].vids[assembly.vid_index_map[iv]] = cv;
			coeffs_next[assembly.nclusters] = 1;
			assembly.nclusters++;

			// create a "complementary operator"
			for (int m = 0; m < bigraph.num_adj_v[j]; m++)
			{
				int i = bigraph.adj_v[j][m];

				if (partition.gamma[i*partition.part_assembly.nclusters + j] == 0) {
					continue;
				}

				const struct u_node* u = &partition.ulist[i];
				assert(u->order == topology->num_neighbors[i_site]);
				// vertex quantum number must agree with the corresponding one stored in connected 'U' node
				assert(vertex->qnum == u->qnums[neigh_parent_index]);

				// add a new TTNO hyperedge
				struct ttno_graph_hyperedge* edge = &ttno_graph->edges[i_site][ce];
				allocate_ttno_graph_hyperedge(u->order, 1, edge);
				for (int n = 0; n < u->order; n++)
				{
					const int k = topology->neighbor_map[i_site][n];
					if (k == i_parent) {
						edge->vids[n] = cv;
					}
					else {
						edge->vids[n] = u->vids[k < i_parent ? n : n - 1];
						// connect edge to neighboring vertex (away from root)
						const int ivd = edge_to_vertex_index(nsites, k, i_site);
						assert(edge->vids[n] < ttno_graph->num_verts[ivd]);
						assert(ttno_graph->verts[ivd][edge->vids[n]].qnum == u->qnums[n]);
						ttno_graph_vertex_add_edge(k < i_site ? 1 : 0, ce, &ttno_graph->verts[ivd][edge->vids[n]]);
					}
				}
				edge->opics[0].oid = u->oid;
				edge->opics[0].coeff = partition.gamma[i*partition.part_assembly.nclusters + j];
				// keep track of handled bipartite graph edges
				partition.gamma[i*partition.part_assembly.nclusters + j] = 0;
				// connect edge to new vertex
				ttno_graph_vertex_add_edge(i_site < i_parent ? 0 : 1, ce, vertex);
				ce++;
			}

			cv++;
		}

		// ensure that we have handled all edges
		for (int i = 0; i < partition.num_u; i++) {
			for (int j = 0; j < partition.part_assembly.nclusters; j++) {
				assert(partition.gamma[i*partition.part_assembly.nclusters + j] == 0);
			}
		}

		// ensure that we had allocated sufficient memory
		assert(assembly.nclusters <= nedges);

		assert(cv == num_u_cover + num_v_cover);
		ttno_graph->num_verts[iv]     = cv;
		ttno_graph->num_edges[i_site] = ce;

		aligned_free(v_cover);
		aligned_free(u_cover);
		delete_bipartite_graph(&bigraph);
		delete_site_cluster_partition(&partition);
	}

	// hyperedges for root node
	assert(assembly.active_sites[i_root]);
	ttno_graph->edges[i_root] = aligned_calloc(MEM_DATA_ALIGN, assembly.nclusters, sizeof(struct ttno_graph_hyperedge));
	for (int ce = 0; ce < assembly.nclusters; ce++)
	{
		const struct op_cluster* cluster = &assembly.clusters[ce];
		assert(cluster->size == 1);
		// add a new TTNO hyperedge
		struct ttno_graph_hyperedge* edge = &ttno_graph->edges[i_root][ce];
		allocate_ttno_graph_hyperedge(topology->num_neighbors[i_root], 1, edge);
		edge->opics[0].oid = cluster->oids[0];
		edge->opics[0].coeff = coeffs_next[ce];
		// connect to vertices
		for (int n = 0; n < topology->num_neighbors[i_root]; n++)
		{
			const int k = topology->neighbor_map[i_root][n];
			const int iv = edge_to_vertex_index(nsites, k, i_root);
			edge->vids[n] = cluster->vids[assembly.vid_index_map[iv]];
			ttno_graph_vertex_add_edge(i_root < k ? 0 : 1, ce, &ttno_graph->verts[iv][edge->vids[n]]);
		}
	}
	ttno_graph->num_edges[i_root] = assembly.nclusters;

	aligned_free(coeffs_next);
	delete_op_cluster_assembly(&assembly);

	aligned_free(sd);

	return 0;
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
			const int iv = edge_to_vertex_index(graph->nsites, k, l);
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
		// entries in 'num_verts' must be zero for k >= l
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
			const int iv = edge_to_vertex_index(graph->nsites, k, l);
			if (is_neighbor) {
				if (graph->num_verts[iv] <= 0) {
					return false;
				}
			}
			else {
				if (graph->num_verts[iv] != 0) {
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

			const int iv = edge_to_vertex_index(graph->nsites, k, l);

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
				int iv = edge_to_vertex_index(graph->nsites, k, l);
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
	int i_site;  //!< site index
	int index;   //!< general index
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
		const int iv = edge_to_vertex_index(graph->nsites, i_site, i_parent);
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
