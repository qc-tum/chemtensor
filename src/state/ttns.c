/// \file ttns.c
/// \brief Tree tensor network state (TTNS) data structure.

#include <math.h>
#include "ttns.h"
#include "bond_ops.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for a tree tensor network state. 'dim_bonds' and 'qbonds' are indexed by site index tuples (i, j) with i < j.
///
void allocate_ttns(const enum numeric_type dtype, const int nsites_physical, const struct abstract_graph* topology, const ct_long* d, const qnumber** qsite, const qnumber qnum_sector, const ct_long* dim_bonds, const qnumber** qbonds, struct ttns* ttns)
{
	assert(nsites_physical >= 1);
	assert(nsites_physical <= topology->num_nodes);
	const int nsites = topology->num_nodes;
	ttns->nsites_physical  = nsites_physical;
	ttns->nsites_branching = nsites - nsites_physical;

	// tree topology
	copy_abstract_graph(topology, &ttns->topology);

	// allocate tensors at each site
	ttns->a = ct_calloc(nsites, sizeof(struct block_sparse_tensor));
	for (int l = 0; l < nsites; l++)
	{
		assert(d[l] >= 1);

		if (l >= ttns->nsites_physical)
		{
			// local dimension for branching sites must be 1 (dummy physical dimension)
			assert(d[l] == 1);
			assert(qsite[l][0] == 0);
		}

		const int ndim = 1 + ttns->topology.num_neighbors[l] + (l == nsites - 1 ? 1 : 0);  // last term counts auxiliary dimension

		ct_long* dim = ct_calloc(ndim, sizeof(ct_long));
		enum tensor_axis_direction* axis_dir = ct_calloc(ndim, sizeof(enum tensor_axis_direction));
		const qnumber** qnums = ct_calloc(ndim, sizeof(qnumber*));

		// virtual bonds
		for (int n = 0; n < ttns->topology.num_neighbors[l]; n++)
		{
			if (n > 0) {
				assert(ttns->topology.neighbor_map[l][n - 1] < ttns->topology.neighbor_map[l][n]);
			}
			int k = ttns->topology.neighbor_map[l][n];
			assert(k != l);
			if (k < l)
			{
				assert(dim_bonds[k*nsites + l] > 0);
				dim[n]      = dim_bonds[k*nsites + l];
				axis_dir[n] = TENSOR_AXIS_OUT;
				qnums[n]    = qbonds[k*nsites + l];  // copy the pointer
			}
			else  // l < k
			{
				assert(dim_bonds[l*nsites + k] > 0);
				dim[n + 1]      = dim_bonds[l*nsites + k];
				axis_dir[n + 1] = TENSOR_AXIS_IN;
				qnums[n + 1]    = qbonds[l*nsites + k];  // copy the pointer
			}
		}

		// physical axis
		#ifndef NDEBUG
		bool site_info_set = false;
		#endif
		for (int i = 0; i < ndim; i++)
		{
			if (dim[i] == 0)
			{
				dim[i]      = d[l];
				qnums[i]    = qsite[l];
				axis_dir[i] = TENSOR_AXIS_OUT;
				#ifndef NDEBUG
				site_info_set = true;
				#endif
				break;
			}
		}
		#ifndef NDEBUG
		assert(site_info_set);
		#endif

		// auxiliary axis on the last site, storing the quantum number sector
		const qnumber qnum_aux[1] = { qnum_sector };
		if (l == nsites - 1)
		{
			assert(dim[ndim - 1] == 0);
			dim[ndim - 1]      = 1;
			qnums[ndim - 1]    = qnum_aux;
			axis_dir[ndim - 1] = TENSOR_AXIS_IN;
		}

		allocate_block_sparse_tensor(dtype, ndim, dim, axis_dir, qnums, &ttns->a[l]);

		ct_free(qnums);
		ct_free(axis_dir);
		ct_free(dim);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a tree tensor network state (free memory).
///
void delete_ttns(struct ttns* ttns)
{
	// overall number of sites
	const int nsites = ttns->nsites_physical + ttns->nsites_branching;

	for (int l = 0; l < nsites; l++) {
		delete_block_sparse_tensor(&ttns->a[l]);
	}
	ct_free(ttns->a);
	ttns->a = NULL;

	delete_abstract_graph(&ttns->topology);

	ttns->nsites_physical  = 0;
	ttns->nsites_branching = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Copy a tree tensor network state and its block sparse tensors.
///
void copy_ttns(const struct ttns* restrict src, struct ttns* restrict dst)
{
	dst->nsites_physical  = src->nsites_physical;
	dst->nsites_branching = src->nsites_branching;

	// tree topology
	copy_abstract_graph(&src->topology, &dst->topology);

	// copy tensors at each site
	const int nsites = src->topology.num_nodes;
	assert(nsites == src->nsites_physical + src->nsites_branching);
	dst->a = ct_malloc(nsites * sizeof(struct block_sparse_tensor));
	for (int l = 0; l < nsites; l++) {
		copy_block_sparse_tensor(&src->a[l], &dst->a[l]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert an edge tuple index (i, j) to a linear virtual bond array index.
///
static inline int edge_to_bond_index(const int nsites, const int i, const int j)
{
	assert(i != j);
	return i < j ? (i*nsites + j) : (j*nsites + i);
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct a tree tensor network state with random normal tensor entries, given a maximum virtual bond dimension.
///
void construct_random_ttns(const enum numeric_type dtype, const int nsites_physical, const struct abstract_graph* topology, const ct_long* d, const qnumber** qsite, const qnumber qnum_sector, const ct_long max_vdim, struct rng_state* rng_state, struct ttns* ttns)
{
	assert(nsites_physical >= 1);
	assert(nsites_physical <= topology->num_nodes);
	const int nsites = topology->num_nodes;

	// select site with maximum number of neighbors as root
	int i_root = 0;
	for (int l = 1; l < topology->num_nodes; l++) {
		if (topology->num_neighbors[l] > topology->num_neighbors[i_root]) {
			i_root = l;
		}
	}
	struct graph_node_distance_tuple* sd = ct_malloc(nsites * sizeof(struct graph_node_distance_tuple));
	enumerate_graph_node_distance_tuples(topology, i_root, sd);
	assert(sd[0].i_node == i_root);

	// virtual bond dimensions and quantum numbers
	ct_long* dim_bonds = ct_calloc(nsites * nsites, sizeof(ct_long));
	qnumber** qbonds   = ct_calloc(nsites * nsites, sizeof(qnumber*));

	// iterate over sites by decreasing distance from root (omitting the root node itself)
	for (int l = nsites - 1; l > 0; l--)
	{
		assert(sd[l - 1].distance <= sd[l].distance);

		const int i_site   = sd[l].i_node;
		const int i_parent = sd[l].i_parent;

		assert(d[i_site] >= 1);

		if (i_site >= nsites_physical)
		{
			// local dimension for branching sites must be 1 (dummy physical dimension)
			assert(d[i_site] == 1);
			assert(qsite[i_site][0] == 0);
		}

		// enumerate all combinations of bond quantum numbers to more distant nodes and local physical quantum numbers
		ct_long dim_full = d[i_site];
		qnumber* qnums_full = ct_malloc(dim_full * sizeof(qnumber));
		memcpy(qnums_full, qsite[i_site], d[i_site] * sizeof(qnumber));
		// auxiliary axis of last site contains overall quantum number sector
		if (i_site == nsites - 1) {
			for (ct_long i = 0; i < d[i_site]; i++) {
				qnums_full[i] -= qnum_sector;
			}
		}
		for (int n = 0; n < topology->num_neighbors[i_site]; n++)
		{
			const int k = topology->neighbor_map[i_site][n];
			if (k == i_parent) {
				continue;
			}

			const int ib = edge_to_bond_index(nsites, i_site, k);
			assert(dim_bonds[ib] > 0);
			assert(qbonds[ib] != NULL);
			qnumber* qnums_full_next = ct_malloc(dim_full * dim_bonds[ib] * sizeof(qnumber));
			// outer sum
			qnumber_outer_sum(1, qnums_full, dim_full, k < i_site ? TENSOR_AXIS_OUT : TENSOR_AXIS_IN, qbonds[ib], dim_bonds[ib], qnums_full_next);
			ct_free(qnums_full);
			qnums_full = qnums_full_next;
			dim_full *= dim_bonds[ib];
			if (dim_full > max_vdim)
			{
				// randomly select quantum numbers
				qnumber* qnums_select = ct_malloc(max_vdim * sizeof(qnumber));
				uint64_t* idx = ct_malloc(max_vdim * sizeof(uint64_t));
				rand_choice(dim_full, max_vdim, rng_state, idx);
				for (ct_long i = 0; i < max_vdim; i++) {
					qnums_select[i] = qnums_full[idx[i]];
				}
				ct_free(idx);
				ct_free(qnums_full);
				qnums_full = qnums_select;
				dim_full = max_vdim;
			}
		}

		// define virtual bond quantum numbers on bond connected to parent node
		const int ib = edge_to_bond_index(nsites, i_site, i_parent);
		assert(dim_bonds[ib] == 0);
		assert(qbonds[ib] == NULL);
		if (i_parent < i_site) {
			// tensor axis direction points outwards -> flip sign
			for (ct_long i = 0; i < dim_full; i++) {
				qnums_full[i] *= -1;
			}
		}
		qbonds[ib] = qnums_full;
		dim_bonds[ib] = dim_full;
	}

	ct_free(sd);

	allocate_ttns(dtype, nsites_physical, topology, d, qsite, qnum_sector, dim_bonds, (const qnumber**)qbonds, ttns);

	for (int l = 0; l < nsites * nsites; l++) {
		if (qbonds[l] != NULL) {
			assert(dim_bonds[l] > 0);
			ct_free(qbonds[l]);
		}
		else {
			assert(dim_bonds[l] == 0);
		}
	}
	ct_free(qbonds);
	ct_free(dim_bonds);

	// fill TTNS tensor entries with pseudo-random numbers, scaled by 1 / sqrt("number of entries")
	for (int l = 0; l < nsites; l++)
	{
		// logical number of entries in TTNS tensor
		const ct_long nelem = integer_product(ttns->a[l].dim_logical, ttns->a[l].ndim);
		// ensure that 'alpha' is large enough to store any numeric type
		dcomplex alpha;
		assert(ttns->a[l].dtype == dtype);
		numeric_from_double(1.0 / sqrt(nelem), dtype, &alpha);
		block_sparse_tensor_fill_random_normal(&alpha, numeric_zero(dtype), rng_state, &ttns->a[l]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Internal consistency check of the TTNS data structure.
///
bool ttns_is_consistent(const struct ttns* ttns)
{
	if (ttns->nsites_physical <= 0) {
		return false;
	}
	if (ttns->nsites_branching < 0) {
		return false;
	}
	// overall number of sites
	const int nsites = ttns->nsites_physical + ttns->nsites_branching;

	// topology
	if (ttns->topology.num_nodes != nsites) {
		return false;
	}
	if (!abstract_graph_is_consistent(&ttns->topology)) {
		return false;
	}
	// verify tree topology
	if (!abstract_graph_is_connected_tree(&ttns->topology)) {
		return false;
	}

	struct ttns_tensor_axis_desc** axis_desc = ct_malloc(nsites * sizeof(struct ttns_tensor_axis_desc*));
	for (int l = 0; l < nsites; l++)
	{
		axis_desc[l] = ct_malloc(ttns->a[l].ndim * sizeof(struct ttns_tensor_axis_desc));
		ttns_tensor_get_axis_desc(&ttns->topology, l, axis_desc[l]);
	}

	for (int l = 0; l < nsites; l++)
	{
		if (ttns->a[l].ndim != 1 + ttns->topology.num_neighbors[l] + (l == nsites - 1 ? 1 : 0)) {
			return false;
		}

		// quantum numbers for physical legs of individual tensors
		for (int i = 0; i < ttns->a[l].ndim; i++)
		{
			if (axis_desc[l][i].type == TTNS_TENSOR_AXIS_PHYSICAL)
			{
				// expecting dummy physical axis for a branching tensor
				if (l >= ttns->nsites_physical)
				{
					if (ttns->a[l].dim_logical[i] != 1) {
						return false;
					}
					if (ttns->a[l].qnums_logical[i][0] != 0) {
						return false;
					}
				}
				if (ttns->a[l].dim_logical[i] != ttns_local_dimension(ttns, l)) {
					return false;
				}
				if (ttns->a[l].axis_dir[i] != TENSOR_AXIS_OUT) {
					return false;
				}
			}
		}

		// auxiliary axis of last site
		if (l == nsites - 1)
		{
			if (ttns->a[l].ndim <= 1) {
				return false;
			}
			int i = ttns->a[l].ndim - 1;
			if (ttns->a[l].dim_logical[i] != 1) {
				return false;
			}
			if (ttns->a[l].axis_dir[i] != TENSOR_AXIS_IN) {
				return false;
			}
			if (axis_desc[l][i].type != TTNS_TENSOR_AXIS_AUXILIARY) {
				return false;
			}
		}

		// virtual bond quantum numbers and axis directions must match
		for (int n = 0; n < ttns->topology.num_neighbors[l]; n++)
		{
			const int k = ttns->topology.neighbor_map[l][n];
			assert(k != l);

			// find respective axis indices
			int i_ax_neigh_lk = -1;
			for (int i = 0; i < ttns->a[l].ndim; i++) {
				if (axis_desc[l][i].type == TTNS_TENSOR_AXIS_VIRTUAL && axis_desc[l][i].index == k) {
					i_ax_neigh_lk = i;
					break;
				}
			}
			if (i_ax_neigh_lk == -1) {
				return false;
			}
			int i_ax_neigh_kl = -1;
			for (int i = 0; i < ttns->a[k].ndim; i++) {
				if (axis_desc[k][i].type == TTNS_TENSOR_AXIS_VIRTUAL && axis_desc[k][i].index == l) {
					i_ax_neigh_kl = i;
					break;
				}
			}
			if (i_ax_neigh_kl == -1) {
				return false;
			}

			if (ttns->a[l].dim_logical[i_ax_neigh_lk] != ttns->a[k].dim_logical[i_ax_neigh_kl]) {
				return false;
			}
			if (!qnumber_all_equal(ttns->a[l].dim_logical[i_ax_neigh_lk],
					ttns->a[l].qnums_logical[i_ax_neigh_lk],
					ttns->a[k].qnums_logical[i_ax_neigh_kl])) {
				return false;
			}
			if (ttns->a[l].axis_dir[i_ax_neigh_lk] != (k < l ? TENSOR_AXIS_OUT : TENSOR_AXIS_IN)) {
				return false;
			}
			if (ttns->a[k].axis_dir[i_ax_neigh_kl] != (l < k ? TENSOR_AXIS_OUT : TENSOR_AXIS_IN)) {
				return false;
			}
		}
	}

	for (int l = 0; l < nsites; l++) {
		ct_free(axis_desc[l]);
	}
	ct_free(axis_desc);

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Get the local physical dimension at 'i_site'.
///
ct_long ttns_local_dimension(const struct ttns* ttns, const int i_site)
{
	assert(ttns->a[i_site].ndim == 1 + ttns->topology.num_neighbors[i_site] + (i_site == ttns->topology.num_nodes - 1 ? 1 : 0));

	// count virtual bonds preceeding physical axis
	int n = 0;
	for (; n < ttns->topology.num_neighbors[i_site]; n++)
	{
		if (n > 0) {
			assert(ttns->topology.neighbor_map[i_site][n - 1] < ttns->topology.neighbor_map[i_site][n]);
		}
		int k = ttns->topology.neighbor_map[i_site][n];
		assert(k != i_site);
		if (k > i_site) {
			break;
		}
	}

	return ttns->a[i_site].dim_logical[n];
}


//________________________________________________________________________________________________________________________
///
/// \brief Get the maximum virtual bond dimension of the TTNS.
///
ct_long ttns_maximum_bond_dimension(const struct ttns* ttns)
{
	// return 1 for the special case that the TTNS topology consists of a single site only
	ct_long m = 1;

	for (int l = 0; l < ttns->topology.num_nodes; l++)
	{
		for (int n = 0; n < ttns->topology.num_neighbors[l]; n++)
		{
			const int k = ttns->topology.neighbor_map[l][n];
			assert(k != l);

			// virtual bond axis index
			const int i_ax = (k < l ? n : n + 1);

			assert(i_ax < ttns->a[l].ndim);
			m = lmax(m, ttns->a[l].dim_logical[i_ax]);
		}
	}

	return m;
}


//________________________________________________________________________________________________________________________
///
/// \brief Get the tensor axis index of the virtual bond at site 'i_site' to neighbor 'i_neigh'.
///
int ttns_tensor_bond_axis_index(const struct abstract_graph* topology, const int i_site, const int i_neigh)
{
	assert(0 <= i_site  && i_site  < topology->num_nodes);
	assert(0 <= i_neigh && i_neigh < topology->num_nodes);
	assert(i_site != i_neigh);

	for (int n = 0; n < topology->num_neighbors[i_site]; n++)
	{
		if (topology->neighbor_map[i_site][n] == i_neigh)
		{
			return (i_neigh < i_site ? n : n + 1);
		}
	}

	// not found
	return -1;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the dot (scalar) product `<chi | psi>` of two TTNS, complex conjugating `chi`.
///
void ttns_vdot(const struct ttns* chi, const struct ttns* psi, void* ret)
{
	// topology must agree
	assert(psi->nsites_physical == chi->nsites_physical);
	assert(psi->nsites_physical >= 1);
	assert(abstract_graph_equal(&chi->topology, &psi->topology));

	// data types must match
	assert(chi->a[0].dtype == psi->a[0].dtype);

	if (ttns_quantum_number_sector(chi) != ttns_quantum_number_sector(psi))
	{
		// inner product is zero if quantum number sectors disagree
		memcpy(ret, numeric_zero(psi->a[0].dtype), sizeof_numeric_type(psi->a[0].dtype));
		return;
	}

	const int nsites = psi->topology.num_nodes;

	// select site with maximum number of neighbors as root
	int i_root = 0;
	for (int l = 1; l < psi->topology.num_nodes; l++) {
		if (psi->topology.num_neighbors[l] > psi->topology.num_neighbors[i_root]) {
			i_root = l;
		}
	}
	struct graph_node_distance_tuple* sd = ct_malloc(nsites * sizeof(struct graph_node_distance_tuple));
	enumerate_graph_node_distance_tuples(&psi->topology, i_root, sd);
	assert(sd[0].i_node == i_root);

	// matrices on virtual bonds corresponding to contracted subtrees
	struct block_sparse_tensor* inner_bonds = ct_malloc(nsites * sizeof(struct block_sparse_tensor));

	// iterate over sites by decreasing distance from root
	for (int l = nsites - 1; l >= 0; l--)
	{
		if (l > 0) {
			assert(sd[l - 1].distance <= sd[l].distance);
		}

		const int i_site   = sd[l].i_node;
		const int i_parent = sd[l].i_parent;

		local_ttns_inner_product(&chi->a[i_site], &psi->a[i_site], &psi->topology, i_site, i_parent, inner_bonds);
	}

	assert(inner_bonds[i_root].ndim == 0);
	assert(inner_bonds[i_root].blocks[0] != NULL);
	// copy scalar entry
	memcpy(ret, inner_bonds[i_root].blocks[0]->data, sizeof_numeric_type(inner_bonds[i_root].dtype));

	for (int l = 0; l < nsites; l++) {
		delete_block_sparse_tensor(&inner_bonds[l]);
	}
	ct_free(inner_bonds);

	ct_free(sd);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the local inner product `<chi | psi>` at 'i_site' given the averages of the connected child nodes.
/// The virtual bonds towards the parent node (if any) remain open.
/// The output is stored in inner_bonds[i_site].
///
void local_ttns_inner_product(const struct block_sparse_tensor* restrict chi, const struct block_sparse_tensor* restrict psi,
	const struct abstract_graph* topology, const int i_site, const int i_parent, struct block_sparse_tensor* restrict inner_bonds)
{
	int i_ax_p = -1;

	struct block_sparse_tensor psi_envs;
	copy_block_sparse_tensor(psi, &psi_envs);
	for (int n = 0; n < topology->num_neighbors[i_site]; n++)
	{
		const int k = topology->neighbor_map[i_site][n];
		assert(k != i_site);

		const int i_ax = (k < i_site ? n : n + 1);

		if (k == i_parent)
		{
			i_ax_p = i_ax;
			continue;
		}

		assert(inner_bonds[k].ndim == 2);
		struct block_sparse_tensor tmp;
		block_sparse_tensor_multiply_axis(&psi_envs, i_ax, &inner_bonds[k], TENSOR_AXIS_RANGE_LEADING, &tmp);
		delete_block_sparse_tensor(&psi_envs);
		psi_envs = tmp;  // copy internal data pointers
	}

	struct block_sparse_tensor chi_conj;
	copy_block_sparse_tensor(chi, &chi_conj);
	conjugate_block_sparse_tensor(&chi_conj);
	block_sparse_tensor_reverse_axis_directions(&chi_conj);

	if (i_parent == -1)
	{
		assert(i_ax_p == -1);
		// contract all axes
		block_sparse_tensor_dot(&psi_envs, TENSOR_AXIS_RANGE_TRAILING, &chi_conj, TENSOR_AXIS_RANGE_TRAILING, psi_envs.ndim, &inner_bonds[i_site]);
	}
	else
	{
		assert(i_ax_p != -1);

		if (i_ax_p == 0)
		{
			block_sparse_tensor_dot(&psi_envs, TENSOR_AXIS_RANGE_TRAILING, &chi_conj, TENSOR_AXIS_RANGE_TRAILING, psi_envs.ndim - 1, &inner_bonds[i_site]);
		}
		else if (i_ax_p == psi_envs.ndim - 1)
		{
			block_sparse_tensor_dot(&psi_envs, TENSOR_AXIS_RANGE_LEADING, &chi_conj, TENSOR_AXIS_RANGE_LEADING, psi_envs.ndim - 1, &inner_bonds[i_site]);
		}
		else
		{
			// move virtual parent bond axis to the end

			int* perm = ct_malloc(psi_envs.ndim * sizeof(int));
			for (int i = 0; i < psi_envs.ndim - 1; i++) {
				perm[i] = (i < i_ax_p ? i : i + 1);
			}
			perm[psi_envs.ndim - 1] = i_ax_p;

			struct block_sparse_tensor psi_envs_perm, chi_conj_perm;
			transpose_block_sparse_tensor(perm, &psi_envs, &psi_envs_perm);
			transpose_block_sparse_tensor(perm, &chi_conj, &chi_conj_perm);

			block_sparse_tensor_dot(&psi_envs_perm, TENSOR_AXIS_RANGE_LEADING, &chi_conj_perm, TENSOR_AXIS_RANGE_LEADING, psi_envs_perm.ndim - 1, &inner_bonds[i_site]);

			delete_block_sparse_tensor(&chi_conj_perm);
			delete_block_sparse_tensor(&psi_envs_perm);
			ct_free(perm);
		}
		assert(inner_bonds[i_site].ndim == 2);
	}

	delete_block_sparse_tensor(&psi_envs);
	delete_block_sparse_tensor(&chi_conj);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the Euclidean norm of the TTNS.
///
/// Result is returned as double also for single-precision tensor entries.
///
double ttns_norm(const struct ttns* psi)
{
	assert(psi->nsites_physical > 0);

	switch (psi->a[0].dtype)
	{
		case CT_SINGLE_REAL:
		{
			float nrm2;
			ttns_vdot(psi, psi, &nrm2);
			assert(nrm2 >= 0);
			return sqrt(nrm2);
		}
		case CT_DOUBLE_REAL:
		{
			double nrm2;
			ttns_vdot(psi, psi, &nrm2);
			assert(nrm2 >= 0);
			return sqrt(nrm2);
		}
		case CT_SINGLE_COMPLEX:
		{
			scomplex vdot;
			ttns_vdot(psi, psi, &vdot);
			float nrm2 = crealf(vdot);
			assert(nrm2 >= 0);
			return sqrt(nrm2);
		}
		case CT_DOUBLE_COMPLEX:
		{
			dcomplex vdot;
			ttns_vdot(psi, psi, &vdot);
			double nrm2 = creal(vdot);
			assert(nrm2 >= 0);
			return sqrt(nrm2);
		}
		default:
		{
			// unknown data type
			assert(false);
			return 0;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Fill the axis descriptions of a TTNS tensor; 'desc' must point to an array of the same length as the degree of the tensor at 'i_site'.
///
void ttns_tensor_get_axis_desc(const struct abstract_graph* topology, const int i_site, struct ttns_tensor_axis_desc* desc)
{
	assert(0 <= i_site && i_site < topology->num_nodes);

	const int ndim = 1 + topology->num_neighbors[i_site] + (i_site == topology->num_nodes - 1 ? 1 : 0);

	// physical axis description; all entries but one will be overwritten
	for (int i = 0; i < ndim; i++) {
		desc[i].type  = TTNS_TENSOR_AXIS_PHYSICAL;
		desc[i].index = i_site;
	}

	// virtual bonds to neighbors
	for (int n = 0; n < topology->num_neighbors[i_site]; n++)
	{
		if (n > 0) {
			assert(topology->neighbor_map[i_site][n - 1] < topology->neighbor_map[i_site][n]);
		}
		int k = topology->neighbor_map[i_site][n];
		assert(k != i_site);
		desc[k < i_site ? n : n + 1].type  = TTNS_TENSOR_AXIS_VIRTUAL;
		desc[k < i_site ? n : n + 1].index = k;
	}

	// auxiliary axis of last site
	if (i_site == topology->num_nodes - 1) {
		desc[ndim - 1].type = TTNS_TENSOR_AXIS_AUXILIARY;
		desc[ndim - 1].index = i_site;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Orthonormalize (a subtree of) the TTNS in-place using QR decompositions,
/// and return the normalization factor if 'i_site' is the root node (i.e., 'i_parent' == -1).
/// 'i_parent' is the parent node index (or -1 if no parent exists).
///
static double ttns_orthonormalize_qr_subtree(const int i_site, const int i_parent, struct ttns* ttns, struct block_sparse_tensor* r)
{
	int i_ax_p = -1;

	for (int n = 0; n < ttns->topology.num_neighbors[i_site]; n++)
	{
		const int k = ttns->topology.neighbor_map[i_site][n];
		assert(k != i_site);

		// corresponding tensor axis index
		const int i_ax = (k < i_site ? n : n + 1);

		if (k == i_parent)
		{
			i_ax_p = i_ax;
			continue;
		}

		struct block_sparse_tensor r_child;
		ttns_orthonormalize_qr_subtree(k, i_site, ttns, &r_child);

		struct block_sparse_tensor tmp;
		block_sparse_tensor_multiply_axis(&ttns->a[i_site], i_ax, &r_child, TENSOR_AXIS_RANGE_TRAILING, &tmp);
		delete_block_sparse_tensor(&ttns->a[i_site]);
		ttns->a[i_site] = tmp;  // copy internal data pointers

		delete_block_sparse_tensor(&r_child);
	}

	if (i_parent == -1)
	{
		assert(i_ax_p == -1);

		double nrm = block_sparse_tensor_norm2(&ttns->a[i_site]);

		if (numeric_real_type(ttns->a[i_site].dtype) == CT_DOUBLE_REAL)
		{
			const double alpha = 1. / nrm;
			rscale_block_sparse_tensor(&alpha, &ttns->a[i_site]);
		}
		else
		{
			assert(numeric_real_type(ttns->a[i_site].dtype) == CT_SINGLE_REAL);
			const float alpha = (float)(1. / nrm);
			rscale_block_sparse_tensor(&alpha, &ttns->a[i_site]);
		}

		// note: 'r' is not modified

		return nrm;
	}

	assert(i_ax_p != -1);

	struct block_sparse_tensor ai_mat;
	struct block_sparse_tensor_axis_matricization_info mat_info;
	block_sparse_tensor_matricize_axis(&ttns->a[i_site], i_ax_p, 1, -ttns->a[i_site].axis_dir[i_ax_p], &ai_mat, &mat_info);
	delete_block_sparse_tensor(&ttns->a[i_site]);

	// perform QR decomposition
	struct block_sparse_tensor q;
	block_sparse_tensor_qr(&ai_mat, QR_REDUCED, &q, r);
	delete_block_sparse_tensor(&ai_mat);

	// updated site-local tensor is the reshaped 'q'
	block_sparse_tensor_dematricize_axis(&q, &mat_info, &ttns->a[i_site]);
	delete_block_sparse_tensor(&q);
	delete_block_sparse_tensor_axis_matricization_info(&mat_info);

	return 1;
}


//________________________________________________________________________________________________________________________
///
/// \brief Orthonormalize the TTNS in-place using QR decompositions, and return the normalization factor.
///
double ttns_orthonormalize_qr(const int i_root, struct ttns* ttns)
{
	return ttns_orthonormalize_qr_subtree(i_root, -1, ttns, NULL);
}


//________________________________________________________________________________________________________________________
///
/// \brief Recursively compress the subtree at 'i_site' using higher-order singular value decomposition with truncation.
/// 'i_parent' is the parent node index (or -1 if no parent exists).
/// \returns 0 on success, a negative integer otherwise
///
static int ttns_compress_subtree(const int i_site, const int i_parent, const double tol, const bool relative_thresh, const ct_long max_vdim, struct ttns* ttns)
{
	if (ttns->topology.num_neighbors[i_site] <= 1)
	{
		// leaf node
		return 0;
	}

	struct block_sparse_tensor* u_list = ct_malloc(ttns->topology.num_neighbors[i_site] * sizeof(struct block_sparse_tensor));
	for (int n = 0; n < ttns->topology.num_neighbors[i_site]; n++)
	{
		const int i_child = ttns->topology.neighbor_map[i_site][n];
		assert(i_child != i_site);
		if (i_child == i_parent) {
			continue;
		}

		// corresponding tensor axis index
		const int i_ax = (i_child < i_site ? n : n + 1);

		struct block_sparse_tensor a_mat;
		struct block_sparse_tensor_axis_matricization_info mat_info;
		// opposite axis direction for merged axes
		block_sparse_tensor_matricize_axis(&ttns->a[i_site], i_ax, 0, -ttns->a[i_site].axis_dir[i_ax], &a_mat, &mat_info);
		delete_block_sparse_tensor_axis_matricization_info(&mat_info);

		struct trunc_info info;
		int ret = split_block_sparse_matrix_svd_isometry(&a_mat, tol, relative_thresh, max_vdim, &u_list[n], &info);
		delete_block_sparse_tensor(&a_mat);
		if (ret < 0) {
			return ret;
		}

		// axis index of child node connecting to current site
		int i_ax_c = ttns_tensor_bond_axis_index(&ttns->topology, i_child, i_site);
		assert(i_ax_c != -1);

		// multiply current 'u' with child tensor
		struct block_sparse_tensor tmp;
		block_sparse_tensor_multiply_axis(&ttns->a[i_child], i_ax_c, &u_list[n], TENSOR_AXIS_RANGE_LEADING, &tmp);
		delete_block_sparse_tensor(&ttns->a[i_child]);
		ttns->a[i_child] = tmp;  // copy internal data pointers

		// recursion to children
		ret = ttns_compress_subtree(i_child, i_site, tol, relative_thresh, max_vdim, ttns);
		if (ret < 0) {
			return ret;
		}
	}

	// form the truncated core tensor
	for (int n = 0; n < ttns->topology.num_neighbors[i_site]; n++)
	{
		const int k = ttns->topology.neighbor_map[i_site][n];
		assert(k != i_site);
		if (k == i_parent) {
			continue;
		}

		// corresponding tensor axis index
		const int i_ax = (k < i_site ? n : n + 1);

		// apply u^\dagger to the core tensor
		conjugate_block_sparse_tensor(&u_list[n]);
		block_sparse_tensor_reverse_axis_directions(&u_list[n]);
		struct block_sparse_tensor tmp;
		block_sparse_tensor_multiply_axis(&ttns->a[i_site], i_ax, &u_list[n], TENSOR_AXIS_RANGE_LEADING, &tmp);
		delete_block_sparse_tensor(&ttns->a[i_site]);
		ttns->a[i_site] = tmp;  // copy internal data pointers

		delete_block_sparse_tensor(&u_list[n]);
	}

	ct_free(u_list);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Recursively compress the tree from the specified root node to the leaves.
/// \returns 0 on success, a negative integer otherwise
///
int ttns_compress(const int i_root, const double tol, const bool relative_thresh, const ct_long max_vdim, struct ttns* ttns)
{
	return ttns_compress_subtree(i_root, -1, tol, relative_thresh, max_vdim, ttns);
}


//________________________________________________________________________________________________________________________
///
/// \brief Contracted subtree of a TTNS (auxiliary data structure used for contraction).
///
struct ttns_contracted_subtree
{
	struct block_sparse_tensor tensor;        //!< subtree tensor
	struct ttns_tensor_axis_desc* axis_desc;  //!< axis descriptions
};


//________________________________________________________________________________________________________________________
///
/// \brief In-place transpose (permute) tensor axes of a contracted subtree.
///
static void transpose_ttns_contracted_subtree(const int* perm, struct ttns_contracted_subtree* subtree)
{
	struct block_sparse_tensor t;
	transpose_block_sparse_tensor(perm, &subtree->tensor, &t);
	delete_block_sparse_tensor(&subtree->tensor);
	subtree->tensor = t;  // copy internal data pointers

	// update axis descriptions
	struct ttns_tensor_axis_desc* new_axis_desc = ct_malloc(subtree->tensor.ndim * sizeof(struct ttns_tensor_axis_desc));
	for (int i = 0; i < subtree->tensor.ndim; i++) {
		new_axis_desc[i] = subtree->axis_desc[perm[i]];
	}
	ct_free(subtree->axis_desc);
	subtree->axis_desc = new_axis_desc;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a contracted subtree (free memory).
///
static void delete_ttns_contracted_subtree(struct ttns_contracted_subtree* subtree)
{
	delete_block_sparse_tensor(&subtree->tensor);
	ct_free(subtree->axis_desc);
}


//________________________________________________________________________________________________________________________
///
/// \brief Temporary data structure for sorting axis descriptions by site index and type.
///
struct ttns_indexed_tensor_axis_desc
{
	struct ttns_tensor_axis_desc axis_desc;
	int index;
};

//________________________________________________________________________________________________________________________
///
/// \brief Comparison function for sorting.
///
static int compare_ttns_indexed_tensor_axis_desc(const void* a, const void* b)
{
	const struct ttns_indexed_tensor_axis_desc* x = a;
	const struct ttns_indexed_tensor_axis_desc* y = b;

	if (x->axis_desc.index < y->axis_desc.index) {
		return -1;
	}
	if (x->axis_desc.index > y->axis_desc.index) {
		return 1;
	}
	// x->axis_desc.index == y->axis_desc.index

	if (x->axis_desc.type < y->axis_desc.type) {
		return -1;
	}
	if (x->axis_desc.type > y->axis_desc.type) {
		return 1;
	}
	// x->axis_desc.type == y->axis_desc.type

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Recursively contract a subtree of a TTNS starting from 'i_site'.
///
static void ttns_contract_subtree(const struct ttns* ttns, const int i_site, const int i_parent, struct ttns_contracted_subtree* contracted)
{
	copy_block_sparse_tensor(&ttns->a[i_site], &contracted->tensor);
	contracted->axis_desc = ct_malloc(contracted->tensor.ndim * sizeof(struct ttns_tensor_axis_desc));
	ttns_tensor_get_axis_desc(&ttns->topology, i_site, contracted->axis_desc);

	// merge child subtrees into current subtree
	for (int i = 0; i < ttns->topology.num_neighbors[i_site]; i++)
	{
		const int k = ttns->topology.neighbor_map[i_site][i];
		if (k == i_parent) {
			continue;
		}

		// recursive function call
		struct ttns_contracted_subtree child;
		ttns_contract_subtree(ttns, k, i_site, &child);

		// find axis index of 'contracted' connecting to 'child'
		int i_ax_c = -1;
		for (int j = 0; j < contracted->tensor.ndim; j++)
		{
			if (contracted->axis_desc[j].type  == TTNS_TENSOR_AXIS_VIRTUAL &&
			    contracted->axis_desc[j].index == k) {
				i_ax_c = j;
				break;
			}
		}
		assert(i_ax_c != -1);

		// find axis index of 'child' connecting to 'contracted'
		int i_ax_p = -1;
		for (int j = 0; j < child.tensor.ndim; j++)
		{
			if (child.axis_desc[j].type  == TTNS_TENSOR_AXIS_VIRTUAL &&
			    child.axis_desc[j].index == i_site) {
				i_ax_p = j;
				break;
			}
		}
		assert(i_ax_p != -1);

		enum tensor_axis_range axrange_c;
		if (i_ax_p == 0) {
			axrange_c = TENSOR_AXIS_RANGE_LEADING;
		}
		else if (i_ax_p == child.tensor.ndim - 1) {
			axrange_c = TENSOR_AXIS_RANGE_TRAILING;
		}
		else {
			axrange_c = TENSOR_AXIS_RANGE_TRAILING;

			// transpose child tensor such that to-be contracted axis is the trailing axis
			int* perm = ct_malloc(child.tensor.ndim * sizeof(int));
			for (int j = 0; j < child.tensor.ndim - 1; j++)
			{
				perm[j] = (j < i_ax_p ? j : j + 1);
			}
			perm[child.tensor.ndim - 1] = i_ax_p;

			transpose_ttns_contracted_subtree(perm, &child);

			ct_free(perm);
		}

		// contract current tensor with child
		struct block_sparse_tensor t;
		block_sparse_tensor_multiply_axis(&contracted->tensor, i_ax_c, &child.tensor, axrange_c, &t);

		// update axis descriptions
		struct ttns_tensor_axis_desc* new_axis_desc = ct_malloc(t.ndim * sizeof(struct ttns_tensor_axis_desc));
		memcpy( new_axis_desc, contracted->axis_desc, i_ax_c * sizeof(struct ttns_tensor_axis_desc));
		memcpy(&new_axis_desc[i_ax_c], &child.axis_desc[axrange_c == TENSOR_AXIS_RANGE_LEADING ? 1 : 0], (child.tensor.ndim - 1) * sizeof(struct ttns_tensor_axis_desc));
		memcpy(&new_axis_desc[i_ax_c + child.tensor.ndim - 1], &contracted->axis_desc[i_ax_c + 1], (contracted->tensor.ndim - i_ax_c - 1) * sizeof(struct ttns_tensor_axis_desc));

		delete_block_sparse_tensor(&contracted->tensor);
		contracted->tensor = t;  // copy internal data pointers
		ct_free(contracted->axis_desc);
		contracted->axis_desc = new_axis_desc;

		delete_ttns_contracted_subtree(&child);

		// sort new axes
		// determine corresponding permutation
		struct ttns_indexed_tensor_axis_desc* indexed_axis_desc = ct_malloc(contracted->tensor.ndim * sizeof(struct ttns_indexed_tensor_axis_desc));
		for (int j = 0; j < contracted->tensor.ndim; j++) {
			indexed_axis_desc[j].axis_desc = contracted->axis_desc[j];
			indexed_axis_desc[j].index = j;
		}
		qsort(indexed_axis_desc, contracted->tensor.ndim, sizeof(struct ttns_indexed_tensor_axis_desc), compare_ttns_indexed_tensor_axis_desc);
		int* perm = ct_malloc(contracted->tensor.ndim * sizeof(int));
		for (int j = 0; j < contracted->tensor.ndim; j++) {
			perm[j] = indexed_axis_desc[j].index;
		}
		ct_free(indexed_axis_desc);
		// skip permutation operations in case of an identity permutation
		if (!is_identity_permutation(perm, contracted->tensor.ndim)) {
			transpose_ttns_contracted_subtree(perm, contracted);
		}
		ct_free(perm);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Merge all tensors of a TTNS to obtain the vector representation on the full Hilbert space.
///
/// The output has a physical and (dummy) auxiliary axis of dimension 1 describing the quantum number sector.
///
void ttns_to_statevector(const struct ttns* ttns, struct block_sparse_tensor* vec)
{
	assert(ttns->nsites_physical  >= 1);
	assert(ttns->nsites_branching >= 0);

	// overall number of sites
	const int nsites = ttns->nsites_physical + ttns->nsites_branching;

	// select site with maximum number of neighbors as root for contraction
	int i_root = 0;
	for (int l = 1; l < nsites; l++) {
		if (ttns->topology.num_neighbors[l] > ttns->topology.num_neighbors[i_root]) {
			i_root = l;
		}
	}

	// contract full tree
	// set parent index to -1 for root node
	struct ttns_contracted_subtree contracted;
	ttns_contract_subtree(ttns, i_root, -1, &contracted);
	assert(contracted.tensor.ndim == nsites + 1);  // including auxiliary axis
	for (int l = 0; l < nsites; l++) {
		// sites must be ordered after subtree contraction
		assert(contracted.axis_desc[l].type == TTNS_TENSOR_AXIS_PHYSICAL);
		assert(contracted.axis_desc[l].index == l);
	}
	assert(contracted.axis_desc[nsites].type == TTNS_TENSOR_AXIS_AUXILIARY);
	assert(contracted.axis_desc[nsites].index == nsites - 1);
	ct_free(contracted.axis_desc);

	*vec = contracted.tensor;  // copy internal data pointers
	while (vec->ndim > 2)
	{
		// flatten pairs of physical axes
		struct block_sparse_tensor t;
		block_sparse_tensor_flatten_axes(vec, 0, TENSOR_AXIS_OUT, &t);
		delete_block_sparse_tensor(vec);
		*vec = t;  // copy internal data pointers
	}
	assert(vec->ndim == 2);
	assert(vec->dim_logical[1] == 1);  // auxiliary axis
}
