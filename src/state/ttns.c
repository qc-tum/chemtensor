/// \file ttns.c
/// \brief Tree tensor network state (TTNS) data structure.

#include <math.h>
#include "ttns.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for a tree tensor network state. 'dim_bonds' and 'qbonds' are indexed by site index tuples (i, j) with i < j.
///
void allocate_ttns(const enum numeric_type dtype, const int nsites_physical, const struct abstract_graph* topology, const long d, const qnumber* qsite, const qnumber qnum_sector, const long* dim_bonds, const qnumber** qbonds, struct ttns* ttns)
{
	assert(nsites_physical >= 1);
	assert(nsites_physical <= topology->num_nodes);
	const int nsites = topology->num_nodes;
	ttns->nsites_physical  = nsites_physical;
	ttns->nsites_branching = nsites - nsites_physical;

	assert(d >= 1);
	ttns->d = d;
	ttns->qsite = ct_malloc(d * sizeof(qnumber));
	memcpy(ttns->qsite, qsite, d * sizeof(qnumber));

	// tree topology
	copy_abstract_graph(topology, &ttns->topology);

	// allocate tensors at each site
	ttns->a = ct_calloc(nsites, sizeof(struct block_sparse_tensor));
	for (int l = 0; l < nsites; l++)
	{
		const int offset_phys_aux = (l < ttns->nsites_physical ? (l == 0 ? 2 : 1) : 0);
		const int ndim = ttns->topology.num_neighbors[l] + offset_phys_aux;

		long* dim = ct_calloc(ndim, sizeof(long));
		enum tensor_axis_direction* axis_dir = ct_calloc(ndim, sizeof(enum tensor_axis_direction));
		const qnumber** qnums = ct_calloc(ndim, sizeof(qnumber*));

		// virtual bonds
		for (int i = 0; i < ttns->topology.num_neighbors[l]; i++)
		{
			if (i > 0) {
				assert(ttns->topology.neighbor_map[l][i - 1] < ttns->topology.neighbor_map[l][i]);
			}
			int k = ttns->topology.neighbor_map[l][i];
			assert(k != l);
			if (k < l)
			{
				assert(dim_bonds[k*nsites + l] > 0);
				dim[i]      = dim_bonds[k*nsites + l];
				axis_dir[i] = TENSOR_AXIS_OUT;
				qnums[i]    = qbonds[k*nsites + l];  // copy the pointer
			}
			else  // l < k
			{
				assert(dim_bonds[l*nsites + k] > 0);
				dim[i + offset_phys_aux]      = dim_bonds[l*nsites + k];
				axis_dir[i + offset_phys_aux] = TENSOR_AXIS_IN;
				qnums[i + offset_phys_aux]    = qbonds[l*nsites + k];  // copy the pointer
			}
		}
		// physical and auxiliary axes
		// include the quantum number sector in the auxiliary axis on site 0
		const qnumber qnum_aux[1] = { qnum_sector };
		if (l == 0)
		{
			assert(dim[0] == 0 && dim[1] == 0);
			dim[0] = d;  // physical
			dim[1] = 1;  // auxiliary
			qnums[0] = qsite;
			qnums[1] = qnum_aux;
			axis_dir[0] = TENSOR_AXIS_OUT;
			axis_dir[1] = TENSOR_AXIS_IN;
		}
		else if (l < ttns->nsites_physical)
		{
			#ifndef NDEBUG
			bool site_info_set = false;
			#endif
			for (int i = 0; i < ndim; i++)
			{
				if (dim[i] == 0) {
					dim[i]      = d;
					qnums[i]    = qsite;
					axis_dir[i] = TENSOR_AXIS_OUT;
					#ifndef NDEBUG
					site_info_set = true;
					#endif
					break;
				}
			}
			assert(site_info_set);
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

	for (int l = 0; l < nsites; l++)
	{
		delete_block_sparse_tensor(&ttns->a[l]);
	}
	ct_free(ttns->a);
	ttns->a = NULL;

	delete_abstract_graph(&ttns->topology);

	ttns->nsites_physical  = 0;
	ttns->nsites_branching = 0;

	ct_free(ttns->qsite);
	ttns->qsite = NULL;
	ttns->d = 0;
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
void construct_random_ttns(const enum numeric_type dtype, const int nsites_physical, const struct abstract_graph* topology, const long d, const qnumber* qsite, const qnumber qnum_sector, const long max_vdim, struct rng_state* rng_state, struct ttns* ttns)
{
	assert(nsites_physical >= 1);
	assert(nsites_physical <= topology->num_nodes);
	const int nsites = topology->num_nodes;
	assert(d >= 1);

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
	long* dim_bonds  = ct_calloc(nsites * nsites, sizeof(long));
	qnumber** qbonds = ct_calloc(nsites * nsites, sizeof(qnumber*));

	// iterate over sites by decreasing distance from root (omitting the root node itself)
	for (int l = nsites - 1; l > 0; l--)
	{
		assert(sd[l - 1].distance <= sd[l].distance);

		const int i_site   = sd[l].i_node;
		const int i_parent = sd[l].i_parent;

		// enumerate all combinations of bond quantum numbers to more distant nodes and local physical quantum numbers
		long dim_full = (i_site < nsites_physical ? d : 1);
		qnumber* qnums_full = ct_calloc(dim_full, sizeof(qnumber));
		if (i_site < nsites_physical)
		{
			memcpy(qnums_full, qsite, d * sizeof(qnumber));
			// auxiliary axis of site 0 contains overall quantum number sector
			if (i_site == 0) {
				for (long i = 0; i < d; i++) {
					qnums_full[i] -= qnum_sector;
				}
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
				for (long i = 0; i < max_vdim; i++) {
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
			for (long i = 0; i < dim_full; i++) {
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
		const long nelem = integer_product(ttns->a[l].dim_logical, ttns->a[l].ndim);
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

	if (ttns->d <= 0) {
		return false;
	}

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
		ttns_tensor_get_axis_desc(ttns, l, axis_desc[l]);
	}

	for (int l = 0; l < nsites; l++)
	{
		const int offset_phys_aux = (l < ttns->nsites_physical ? (l == 0 ? 2 : 1) : 0);

		if (ttns->a[l].ndim != ttns->topology.num_neighbors[l] + offset_phys_aux) {
			return false;
		}

		// quantum numbers for physical legs of individual tensors must agree with 'qsite'
		for (int i = 0; i < ttns->a[l].ndim; i++)
		{
			if (axis_desc[l][i].type == TTNS_TENSOR_AXIS_PHYSICAL)
			{
				if (ttns->a[l].dim_logical[i] != ttns->d) {
					return false;
				}
				if (!qnumber_all_equal(ttns->d, ttns->a[l].qnums_logical[i], ttns->qsite)) {
					return false;
				}
				if (ttns->a[l].axis_dir[i] != TENSOR_AXIS_OUT) {
					return false;
				}
			}
		}
		// auxiliary axis of site 0
		{
			if (ttns->a[0].ndim <= 1) {
				return false;
			}
			if (ttns->a[0].dim_logical[1] != 1) {
				return false;
			}
			if (ttns->a[0].axis_dir[1] != TENSOR_AXIS_IN) {
				return false;
			}
			if (axis_desc[0][1].type != TTNS_TENSOR_AXIS_AUXILIARY) {
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
/// \brief Compute the dot (scalar) product `<chi | psi>` of two TTNS, complex conjugating `chi`.
///
void ttns_vdot(const struct ttns* chi, const struct ttns* psi, void* ret)
{
	// topology must agree
	assert(psi->nsites_physical == chi->nsites_physical);
	assert(psi->nsites_physical >= 1);
	assert(abstract_graph_equal(&chi->topology, &psi->topology));

	// physical quantum numbers must agree
	assert(chi->d == psi->d);
	assert(qnumber_all_equal(chi->d, chi->qsite, psi->qsite));

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
	struct block_sparse_tensor** r_bonds = ct_calloc(nsites * nsites, sizeof(struct block_sparse_tensor*));

	// iterate over sites by decreasing distance from root
	for (int l = nsites - 1; l >= 0; l--)
	{
		if (l > 0) {
			assert(sd[l - 1].distance <= sd[l].distance);
		}

		const int i_site   = sd[l].i_node;
		const int i_parent = sd[l].i_parent;

		const int offset_phys_aux = (i_site < psi->nsites_physical ? (i_site == 0 ? 2 : 1) : 0);

		// contract the local tensor in 'psi' with the matrices on the bonds towards the children
		struct block_sparse_tensor psi_a_bonds;
		copy_block_sparse_tensor(&psi->a[i_site], &psi_a_bonds);
		for (int n = 0; n < psi->topology.num_neighbors[i_site]; n++)
		{
			const int k = psi->topology.neighbor_map[i_site][n];
			assert(k != i_site);
			if (k == i_parent) {
				continue;
			}

			const int ib = edge_to_bond_index(nsites, i_site, k);
			assert(r_bonds[ib] != NULL);
			assert(r_bonds[ib]->ndim == 2);

			struct block_sparse_tensor tmp;
			const int i_ax = (k < i_site ? n : n + offset_phys_aux);
			block_sparse_tensor_multiply_axis(&psi_a_bonds, i_ax, r_bonds[ib], TENSOR_AXIS_RANGE_LEADING, &tmp);
			delete_block_sparse_tensor(&psi_a_bonds);
			move_block_sparse_tensor_data(&tmp, &psi_a_bonds);
		}

		struct block_sparse_tensor chi_a_conj;
		copy_block_sparse_tensor(&chi->a[i_site], &chi_a_conj);
		conjugate_block_sparse_tensor(&chi_a_conj);
		block_sparse_tensor_reverse_axis_directions(&chi_a_conj);

		if (l == 0)  // root node
		{
			// contract all axes
			struct block_sparse_tensor r;
			block_sparse_tensor_dot(&chi_a_conj, TENSOR_AXIS_RANGE_LEADING, &psi_a_bonds, TENSOR_AXIS_RANGE_LEADING, psi_a_bonds.ndim, &r);
			assert(r.ndim == 0);
			assert(r.blocks[0] != NULL);

			// copy scalar entry
			memcpy(ret, r.blocks[0]->data, sizeof_numeric_type(r.dtype));

			delete_block_sparse_tensor(&r);
		}
		else
		{
			// find parent bond axis
			int i_ax_p = -1;
			for (int n = 0; n < psi->topology.num_neighbors[i_site]; n++)
			{
				const int k = psi->topology.neighbor_map[i_site][n];
				if (k == i_parent) {
					i_ax_p = (k < i_site ? n : n + offset_phys_aux);
					break;
				}
			}
			assert(i_ax_p != -1);

			assert(psi_a_bonds.ndim == chi_a_conj.ndim);
			assert(psi_a_bonds.ndim == psi->topology.num_neighbors[i_site] + offset_phys_aux);

			if (i_ax_p != 0)
			{
				// move parent bond to the beginning

				int* perm = ct_malloc(psi_a_bonds.ndim * sizeof(int));
				perm[0] = i_ax_p;
				for (int j = 0; j < psi_a_bonds.ndim - 1; j++) {
					perm[j + 1] = (j < i_ax_p ? j : j + 1);
				}

				struct block_sparse_tensor tmp;

				transpose_block_sparse_tensor(perm, &psi_a_bonds, &tmp);
				delete_block_sparse_tensor(&psi_a_bonds);
				move_block_sparse_tensor_data(&tmp, &psi_a_bonds);

				transpose_block_sparse_tensor(perm, &chi_a_conj, &tmp);
				delete_block_sparse_tensor(&chi_a_conj);
				move_block_sparse_tensor_data(&tmp, &chi_a_conj);

				ct_free(perm);
			}

			if (psi_a_bonds.ndim == 1)
			{
				// special case: single virtual bond axis to parent node and no physical or auxiliary axis
				// -> add dummy leg for contraction

				{
					const qnumber q_zero[1] = { 0 };
					const long new_dim_logical[2]                    = { psi_a_bonds.dim_logical[0],   1               };
					const enum tensor_axis_direction new_axis_dir[2] = { psi_a_bonds.axis_dir[0],      TENSOR_AXIS_OUT };
					const qnumber* new_qnums_logical[2]              = { psi_a_bonds.qnums_logical[0], q_zero          };
					struct block_sparse_tensor tmp;
					split_block_sparse_tensor_axis(&psi_a_bonds, 0, new_dim_logical, new_axis_dir, new_qnums_logical, &tmp);
					delete_block_sparse_tensor(&psi_a_bonds);
					move_block_sparse_tensor_data(&tmp, &psi_a_bonds);
				}

				{
					const qnumber q_zero[1] = { 0 };
					const long new_dim_logical[2]                    = { chi_a_conj.dim_logical[0],   1              };
					const enum tensor_axis_direction new_axis_dir[2] = { chi_a_conj.axis_dir[0],      TENSOR_AXIS_IN };
					const qnumber* new_qnums_logical[2]              = { chi_a_conj.qnums_logical[0], q_zero         };
					struct block_sparse_tensor tmp;
					split_block_sparse_tensor_axis(&chi_a_conj, 0, new_dim_logical, new_axis_dir, new_qnums_logical, &tmp);
					delete_block_sparse_tensor(&chi_a_conj);
					move_block_sparse_tensor_data(&tmp, &chi_a_conj);
				}
			}
			assert(psi_a_bonds.ndim > 1);

			const int ib = edge_to_bond_index(nsites, i_site, i_parent);
			assert(r_bonds[ib] == NULL);

			// contract all other axes
			r_bonds[ib] = ct_malloc(sizeof(struct block_sparse_tensor));
			block_sparse_tensor_dot(&psi_a_bonds, TENSOR_AXIS_RANGE_TRAILING, &chi_a_conj, TENSOR_AXIS_RANGE_TRAILING, psi_a_bonds.ndim - 1, r_bonds[ib]);
		}

		delete_block_sparse_tensor(&chi_a_conj);
		delete_block_sparse_tensor(&psi_a_bonds);
	}

	for (int l = 0; l < nsites * nsites; l++) {
		if (r_bonds[l] != NULL) {
			delete_block_sparse_tensor(r_bonds[l]);
			ct_free(r_bonds[l]);
		}
	}
	ct_free(r_bonds);

	ct_free(sd);
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
void ttns_tensor_get_axis_desc(const struct ttns* ttns, const int i_site, struct ttns_tensor_axis_desc* desc)
{
	// overall number of sites
	#ifndef NDEBUG
	const int nsites = ttns->nsites_physical + ttns->nsites_branching;
	#endif

	const int offset_phys_aux = (i_site < ttns->nsites_physical ? (i_site == 0 ? 2 : 1) : 0);

	assert(0 <= i_site && i_site < nsites);
	assert(ttns->a[i_site].ndim == ttns->topology.num_neighbors[i_site] + offset_phys_aux);

	// set to default values
	for (int i = 0; i < ttns->a[i_site].ndim; i++) {
		desc[i].type  = TTNS_TENSOR_AXIS_PHYSICAL;
		desc[i].index = i_site;
	}

	// auxiliary axis of site 0 (special case)
	if (i_site == 0) {
		assert(ttns->a[0].ndim >= 2);
		assert(ttns->a[0].dim_logical[1] == 1);
		desc[1].type = TTNS_TENSOR_AXIS_AUXILIARY;
		desc[1].index = i_site;
	}

	// virtual bonds to neighbors
	for (int i = 0; i < ttns->topology.num_neighbors[i_site]; i++)
	{
		if (i > 0) {
			assert(ttns->topology.neighbor_map[i_site][i - 1] < ttns->topology.neighbor_map[i_site][i]);
		}
		int k = ttns->topology.neighbor_map[i_site][i];
		assert(k != i_site);
		desc[k < i_site ? i : i + offset_phys_aux].type  = TTNS_TENSOR_AXIS_VIRTUAL;
		desc[k < i_site ? i : i + offset_phys_aux].index = k;
	}
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
	move_block_sparse_tensor_data(&t, &subtree->tensor);

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
	ttns_tensor_get_axis_desc(ttns, i_site, contracted->axis_desc);

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
		move_block_sparse_tensor_data(&t, &contracted->tensor);
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
	assert(contracted.tensor.ndim == ttns->nsites_physical + 1);  // including auxiliary axis
	assert(contracted.axis_desc[0].index == 0 && contracted.axis_desc[0].type == TTNS_TENSOR_AXIS_PHYSICAL);
	assert(contracted.axis_desc[1].index == 0 && contracted.axis_desc[1].type == TTNS_TENSOR_AXIS_AUXILIARY);
	for (int l = 1; l < ttns->nsites_physical; l++) {
		// sites must be ordered after subtree contraction
		assert(contracted.axis_desc[1 + l].index == l && contracted.axis_desc[1 + l].type == TTNS_TENSOR_AXIS_PHYSICAL);
	}
	ct_free(contracted.axis_desc);

	// temporarily flip first physical and auxiliary axis
	struct block_sparse_tensor ctensor_flip_aux;
	{
		int* perm = ct_malloc(contracted.tensor.ndim * sizeof(int));
		perm[0] = 1;
		perm[1] = 0;
		for (int i = 2; i < contracted.tensor.ndim; i++) {
			perm[i] = i;
		}
		transpose_block_sparse_tensor(perm, &contracted.tensor, &ctensor_flip_aux);
		ct_free(perm);
	}
	delete_block_sparse_tensor(&contracted.tensor);

	while (ctensor_flip_aux.ndim > 2)
	{
		// flatten pairs of physical axes
		struct block_sparse_tensor t;
		flatten_block_sparse_tensor_axes(&ctensor_flip_aux, 1, TENSOR_AXIS_OUT, &t);
		delete_block_sparse_tensor(&ctensor_flip_aux);
		move_block_sparse_tensor_data(&t, &ctensor_flip_aux);
		assert(ctensor_flip_aux.dim_logical[0] == 1);  // auxiliary axis
	}
	assert(ctensor_flip_aux.ndim == 2);

	// undo flip and store resulting tensor in 'vec'
	const int perm[2] = { 1, 0 };
	transpose_block_sparse_tensor(perm, &ctensor_flip_aux, vec);
	delete_block_sparse_tensor(&ctensor_flip_aux);
}
