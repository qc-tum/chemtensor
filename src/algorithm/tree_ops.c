/// \file tree_ops.c
/// \brief Higher-level tensor network operations on a tree topology.

#include "tree_ops.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Compute the inner product between a TTNO and two TTNS, `<chi | op | psi>`.
///
void ttno_inner_product(const struct ttns* chi, const struct ttno* op, const struct ttns* psi, void* ret)
{
	// data types must match
	assert(chi->a[0].dtype == psi->a[0].dtype);
	assert( op->a[0].dtype == psi->a[0].dtype);

	if (ttns_quantum_number_sector(chi) != ttns_quantum_number_sector(psi))
	{
		// inner product is zero if quantum number sectors disagree
		memcpy(ret, numeric_zero(psi->a[0].dtype), sizeof_numeric_type(psi->a[0].dtype));
		return;
	}

	// select site with maximum number of neighbors as root
	int i_root = 0;
	for (int l = 1; l < psi->topology.num_nodes; l++) {
		if (psi->topology.num_neighbors[l] > psi->topology.num_neighbors[i_root]) {
			i_root = l;
		}
	}

	struct block_sparse_tensor* avg_bonds = ct_malloc(psi->topology.num_nodes * sizeof(struct block_sparse_tensor));
	ttno_subtrees_inner_products(chi, op, psi, i_root, avg_bonds);

	// copy scalar entry
	assert(avg_bonds[i_root].ndim == 0);
	assert(avg_bonds[i_root].blocks[0] != NULL);
	memcpy(ret, avg_bonds[i_root].blocks[0]->data, sizeof_numeric_type(avg_bonds[i_root].dtype));

	for (int l = 0; l < psi->topology.num_nodes; l++) {
		delete_block_sparse_tensor(&avg_bonds[l]);
	}
	ct_free(avg_bonds);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the inner products on all subtrees oriented towards 'i_root' between a TTNO and two TTNS, `<chi | op | psi>`.
///
/// At input, 'avg_bonds' must be an array of length "number of sites".
/// After the function call, avg_bonds[i] for i != i_root will be a degree 3 tensor
/// with axes corresponding to the virtual bonds towards 'i_root' of the operator sandwich ordered (bond of psi, bond of op, bond of chi),
/// and avg_bonds[i_root] a degree 0 tensor containing the overall inner product value.
///
void ttno_subtrees_inner_products(const struct ttns* chi, const struct ttno* op, const struct ttns* psi, const int i_root, struct block_sparse_tensor* avg_bonds)
{
	// topology must agree
	assert(chi->nsites_physical == psi->nsites_physical);
	assert( op->nsites_physical == psi->nsites_physical);
	assert(psi->nsites_physical >= 1);
	assert(abstract_graph_equal(&chi->topology, &psi->topology));
	assert(abstract_graph_equal( &op->topology, &psi->topology));

	// must be a connected tree
	assert(abstract_graph_is_connected_tree(&psi->topology));

	// data types must match
	assert(chi->a[0].dtype == psi->a[0].dtype);
	assert( op->a[0].dtype == psi->a[0].dtype);

	const int nsites = psi->topology.num_nodes;
	assert(0 <= i_root && i_root < nsites);

	struct graph_node_distance_tuple* sd = ct_malloc(nsites * sizeof(struct graph_node_distance_tuple));
	enumerate_graph_node_distance_tuples(&psi->topology, i_root, sd);
	assert(sd[0].i_node == i_root);

	// iterate over sites by decreasing distance from root
	for (int l = nsites - 1; l >= 0; l--)
	{
		if (l > 0) {
			assert(sd[l - 1].distance <= sd[l].distance);
		}

		const int i_site   = sd[l].i_node;
		const int i_parent = sd[l].i_parent;
		assert(l > 0 || i_parent == -1);

		local_ttno_inner_product(&chi->a[i_site], &op->a[i_site], &psi->a[i_site], &op->topology, i_site, i_parent, avg_bonds);
	}

	ct_free(sd);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the local inner product `<chi | op | psi>` at 'i_site' given the averages of the connected child nodes.
/// The virtual bonds towards the parent node (if any) remain open.
/// The output is stored in avg_bonds[i_site].
///
void local_ttno_inner_product(const struct block_sparse_tensor* restrict chi, const struct block_sparse_tensor* restrict op, const struct block_sparse_tensor* restrict psi,
	const struct abstract_graph* topology, const int i_site, const int i_parent, struct block_sparse_tensor* restrict avg_bonds)
{
	const int offset_phys_aux = (i_site == 0 ? 2 : 1);

	// contract the local tensor 'psi' with the tensors on the bonds towards the children
	struct block_sparse_tensor psi_avg;
	copy_block_sparse_tensor(psi, &psi_avg);

	// using decreasing axis index sequence here to preserve preceeding virtual bond axes indices
	// (relying on the property that the list of neighboring nodes is sorted)
	for (int n = topology->num_neighbors[i_site] - 1; n >= 0; n--)
	{
		const int k = topology->neighbor_map[i_site][n];
		assert(k != i_site);
		if (k == i_parent) {
			continue;
		}

		assert(avg_bonds[k].ndim == 3);
		struct block_sparse_tensor tmp;
		const int i_ax = (k < i_site ? n : n + offset_phys_aux);
		block_sparse_tensor_multiply_axis(&psi_avg, i_ax, &avg_bonds[k], TENSOR_AXIS_RANGE_LEADING, &tmp);
		delete_block_sparse_tensor(&psi_avg);
		psi_avg = tmp;  // copy internal data pointers
	}
	assert(psi_avg.ndim == psi->ndim + topology->num_neighbors[i_site] - (i_parent == -1 ? 0 : 1));

	struct ttns_tensor_axis_desc* axis_desc_op_psi = ct_malloc((op->ndim + psi->ndim) * sizeof(struct ttns_tensor_axis_desc));
	int ca = 0;  // counter

	// move virtual parent bond and physical output axis of the local operator to the front
	struct block_sparse_tensor op_perm;
	{
		struct ttno_tensor_axis_desc* axis_desc_op = ct_malloc(op->ndim * sizeof(struct ttno_tensor_axis_desc));
		ttno_tensor_get_axis_desc(topology, i_site, axis_desc_op);

		int* perm = ct_malloc(op->ndim * sizeof(int));
		int cp = 0;
		for (int i = 0; i < op->ndim; i++)
		{
			if (axis_desc_op[i].type == TTNO_TENSOR_AXIS_VIRTUAL && axis_desc_op[i].index == i_parent) {
				perm[cp++] = i;
				axis_desc_op_psi[ca].type  = TTNS_TENSOR_AXIS_VIRTUAL;
				axis_desc_op_psi[ca].index = ((1 << 16) + i_parent);  // use special encoding for parent bond of 'op', to distinguish it from 'psi'
				ca++;
			}
		}
		for (int i = 0; i < op->ndim; i++)
		{
			if (axis_desc_op[i].type == TTNO_TENSOR_AXIS_PHYS_OUT) {
				perm[cp++] = i;
				axis_desc_op_psi[ca].type  = TTNS_TENSOR_AXIS_PHYSICAL;
				axis_desc_op_psi[ca].index = i_site;
				ca++;
			}
		}
		for (int i = 0; i < op->ndim; i++)
		{
			if (axis_desc_op[i].type == TTNO_TENSOR_AXIS_PHYS_IN ||
				((axis_desc_op[i].type == TTNO_TENSOR_AXIS_VIRTUAL && axis_desc_op[i].index != i_parent))) {
				perm[cp++] = i;
			}
		}
		assert(cp == op->ndim);

		if (is_identity_permutation(perm, op->ndim)) {
			copy_block_sparse_tensor(op, &op_perm);
		}
		else {
			transpose_block_sparse_tensor(perm, op, &op_perm);
		}

		ct_free(perm);
		ct_free(axis_desc_op);
	}

	struct ttns_tensor_axis_desc* axis_desc_state = ct_malloc(psi->ndim * sizeof(struct ttns_tensor_axis_desc));
	ttns_tensor_get_axis_desc(topology, i_site, axis_desc_state);

	// move operator bonds and physical axis in 'psi_avg' to the front
	{
		int* perm = ct_malloc(psi_avg.ndim * sizeof(int));
		int cp = 0;
		int i_ext = 0;
		for (int i = 0; i < psi->ndim; i++)
		{
			if (axis_desc_state[i].type == TTNS_TENSOR_AXIS_PHYSICAL)
			{
				perm[cp++] = i_ext;
				i_ext++;
			}
			else if (axis_desc_state[i].type == TTNS_TENSOR_AXIS_AUXILIARY)
			{
				i_ext++;
			}
			else
			{
				assert(axis_desc_state[i].type == TTNS_TENSOR_AXIS_VIRTUAL);
				if (axis_desc_state[i].index == i_parent) {  // bond to parent site
					i_ext++;
				}
				else {
					perm[cp++] = i_ext;
					i_ext += 2;
				}
			}
		}
		i_ext = 0;
		for (int i = 0; i < psi->ndim; i++)
		{
			if (axis_desc_state[i].type == TTNS_TENSOR_AXIS_PHYSICAL)
			{
				i_ext++;
			}
			else if (axis_desc_state[i].type == TTNS_TENSOR_AXIS_AUXILIARY)
			{
				perm[cp++] = i_ext;
				i_ext++;
				axis_desc_op_psi[ca].type  = TTNS_TENSOR_AXIS_AUXILIARY;
				axis_desc_op_psi[ca].index = i_site;
				ca++;
			}
			else
			{
				assert(axis_desc_state[i].type == TTNS_TENSOR_AXIS_VIRTUAL);
				if (axis_desc_state[i].index == i_parent) {  // bond to parent site
					perm[cp++] = i_ext;
					i_ext++;
				}
				else {
					perm[cp++] = i_ext + 1;
					i_ext += 2;
				}
				axis_desc_op_psi[ca].type  = TTNS_TENSOR_AXIS_VIRTUAL;
				axis_desc_op_psi[ca].index = axis_desc_state[i].index;
				ca++;
			}
		}
		assert(cp == psi_avg.ndim);

		if (!is_identity_permutation(perm, psi_avg.ndim))
		{
			struct block_sparse_tensor tmp;
			transpose_block_sparse_tensor(perm, &psi_avg, &tmp);
			delete_block_sparse_tensor(&psi_avg);
			psi_avg = tmp;  // copy internal data pointers
		}

		ct_free(perm);
	}

	// contract with TTNO tensor
	// number of to-be contracted axes:
	// - physical: 1
	// - virtual:  #neighbors - (1 if not root else 0)
	const int ndim_mult_op = 1 + topology->num_neighbors[i_site] - (i_parent == -1 ? 0 : 1);
	assert(op_perm.ndim + psi_avg.ndim - 2*ndim_mult_op == ca);
	assert(ndim_mult_op >= 1);
	struct block_sparse_tensor op_psi_avg;
	block_sparse_tensor_dot(&op_perm, TENSOR_AXIS_RANGE_TRAILING, &psi_avg, TENSOR_AXIS_RANGE_LEADING, ndim_mult_op, &op_psi_avg);
	delete_block_sparse_tensor(&op_perm);
	delete_block_sparse_tensor(&psi_avg);
	assert(op_psi_avg.ndim == ca);

	// move parent bond axes of 'op_psi_avg' to the front and interleave physical axis
	{
		int* perm = ct_malloc(op_psi_avg.ndim * sizeof(int));
		int cp = 0;
		for (int i = 0; i < op_psi_avg.ndim; i++)
		{
			if (axis_desc_op_psi[i].type == TTNS_TENSOR_AXIS_VIRTUAL && axis_desc_op_psi[i].index == i_parent)
			{
				// virtual bond to parent of original 'psi'
				perm[cp++] = i;
			}
		}
		for (int i = 0; i < op_psi_avg.ndim; i++)
		{
			if (axis_desc_op_psi[i].type == TTNS_TENSOR_AXIS_VIRTUAL && axis_desc_op_psi[i].index == ((1 << 16) + i_parent))
			{
				// virtual bond to parent of original 'op'
				perm[cp++] = i;
			}
		}
		for (int i = 0; i < op_psi_avg.ndim; i++)
		{
			if (axis_desc_op_psi[i].type == TTNS_TENSOR_AXIS_VIRTUAL
				&& axis_desc_op_psi[i].index != i_parent
				&& axis_desc_op_psi[i].index != ((1 << 16) + i_parent)
				&& axis_desc_op_psi[i].index < i_site)
			{
				perm[cp++] = i;
			}
		}
		for (int i = 0; i < op_psi_avg.ndim; i++)
		{
			if (axis_desc_op_psi[i].type == TTNS_TENSOR_AXIS_PHYSICAL || axis_desc_op_psi[i].type == TTNS_TENSOR_AXIS_AUXILIARY)
			{
				assert(axis_desc_op_psi[i].index == i_site);
				perm[cp++] = i;
			}
		}
		for (int i = 0; i < op_psi_avg.ndim; i++)
		{
			if (axis_desc_op_psi[i].type == TTNS_TENSOR_AXIS_VIRTUAL
				&& axis_desc_op_psi[i].index != i_parent
				&& axis_desc_op_psi[i].index != ((1 << 16) + i_parent)
				&& axis_desc_op_psi[i].index > i_site)
			{
				perm[cp++] = i;
			}
		}
		assert(cp == op_psi_avg.ndim);

		if (!is_identity_permutation(perm, op_psi_avg.ndim))
		{
			struct block_sparse_tensor tmp;
			transpose_block_sparse_tensor(perm, &op_psi_avg, &tmp);
			delete_block_sparse_tensor(&op_psi_avg);
			op_psi_avg = tmp;  // copy internal data pointers
		}

		ct_free(perm);
	}

	ct_free(axis_desc_op_psi);

	struct block_sparse_tensor chi_conj;
	copy_block_sparse_tensor(chi, &chi_conj);
	conjugate_block_sparse_tensor(&chi_conj);
	block_sparse_tensor_reverse_axis_directions(&chi_conj);

	if (i_parent != -1)  // not root node
	{
		// move parent bond axis in 'chi_conj' to the front

		int* perm = ct_malloc(chi_conj.ndim * sizeof(int));
		int c = 0;
		for (int i = 0; i < chi->ndim; i++) {
			if (axis_desc_state[i].type == TTNS_TENSOR_AXIS_VIRTUAL && axis_desc_state[i].index == i_parent) {
				perm[c++] = i;
			}
		}
		for (int i = 0; i < chi->ndim; i++) {
			if (axis_desc_state[i].type != TTNS_TENSOR_AXIS_VIRTUAL || axis_desc_state[i].index != i_parent) {
				perm[c++] = i;
			}
		}
		assert(c == chi_conj.ndim);

		if (!is_identity_permutation(perm, chi_conj.ndim))
		{
			struct block_sparse_tensor tmp;
			transpose_block_sparse_tensor(perm, &chi_conj, &tmp);
			delete_block_sparse_tensor(&chi_conj);
			chi_conj = tmp;  // copy internal data pointers
		}

		ct_free(perm);
	}

	ct_free(axis_desc_state);

	const int ndim_mult_chi = chi_conj.ndim - (i_parent == -1 ? 0 : 1);
	assert(ndim_mult_chi >= 1);
	block_sparse_tensor_dot(&op_psi_avg, TENSOR_AXIS_RANGE_TRAILING, &chi_conj, TENSOR_AXIS_RANGE_TRAILING, ndim_mult_chi, &avg_bonds[i_site]);
	delete_block_sparse_tensor(&op_psi_avg);
	delete_block_sparse_tensor(&chi_conj);

	if (i_parent == -1)  // root node
	{
		assert(avg_bonds[i_site].ndim == 0);
		assert(avg_bonds[i_site].blocks[0] != NULL);
	}
	else {
		assert(avg_bonds[i_site].ndim == 3);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply a local operator `op |psi>` at 'i_site' and contract with the environment tensors on the connected bonds towards 'i_site'.
/// The environment tensor on the i-th virtual bond is 'envs[i]'.
///
void apply_local_ttno_tensor(const struct block_sparse_tensor* restrict op, const struct block_sparse_tensor* restrict psi,
	const struct abstract_graph* topology, const int i_site, const struct block_sparse_tensor* restrict envs,
	struct block_sparse_tensor* restrict ret)
{
	const int offset_phys_aux = (i_site == 0 ? 2 : 1);

	// contract the local tensor 'psi' with the environment tensors
	struct block_sparse_tensor psi_envs;
	copy_block_sparse_tensor(psi, &psi_envs);

	// using decreasing axis index sequence here to preserve preceeding virtual bond axes indices
	// (relying on the property that the list of neighboring nodes is sorted)
	for (int n = topology->num_neighbors[i_site] - 1; n >= 0; n--)
	{
		const int k = topology->neighbor_map[i_site][n];
		assert(k != i_site);

		assert(envs[n].ndim == 3);
		struct block_sparse_tensor tmp;
		const int i_ax = (k < i_site ? n : n + offset_phys_aux);
		block_sparse_tensor_multiply_axis(&psi_envs, i_ax, &envs[n], TENSOR_AXIS_RANGE_LEADING, &tmp);
		delete_block_sparse_tensor(&psi_envs);
		psi_envs = tmp;  // copy internal data pointers
	}
	assert(psi_envs.ndim == psi->ndim + topology->num_neighbors[i_site]);

	// move physical output axis of the local operator to the front
	struct block_sparse_tensor op_perm;
	bool op_perm_shallow_copy;
	{
		struct ttno_tensor_axis_desc* axis_desc_op = ct_malloc(op->ndim * sizeof(struct ttno_tensor_axis_desc));
		ttno_tensor_get_axis_desc(topology, i_site, axis_desc_op);

		int i_ax_phys_out = -1;
		for (int i = 0; i < op->ndim; i++)
		{
			if (axis_desc_op[i].type == TTNO_TENSOR_AXIS_PHYS_OUT) {
				i_ax_phys_out = i;
				break;
			}
		}
		assert(i_ax_phys_out != -1);

		if (i_ax_phys_out == 0)
		{
			// copy internal data pointers
			op_perm = *op;
			op_perm_shallow_copy = true;
		}
		else
		{
			int* perm = ct_malloc(op->ndim * sizeof(int));
			perm[0] = i_ax_phys_out;
			for (int i = 1; i < op->ndim; i++) {
				perm[i] = (i <= i_ax_phys_out ? i - 1 : i);
			}

			transpose_block_sparse_tensor(perm, op, &op_perm);
			op_perm_shallow_copy = false;

			ct_free(perm);
		}

		ct_free(axis_desc_op);
	}

	int i_ax_phys = -1;

	// move operator bonds and physical axis in 'psi_envs' to the front
	{
		struct ttns_tensor_axis_desc* axis_desc_state = ct_malloc(psi->ndim * sizeof(struct ttns_tensor_axis_desc));
		ttns_tensor_get_axis_desc(topology, i_site, axis_desc_state);

		int* perm = ct_malloc(psi_envs.ndim * sizeof(int));
		int cp = 0;
		int i_ext = 0;
		for (int i = 0; i < psi->ndim; i++)
		{
			if (axis_desc_state[i].type == TTNS_TENSOR_AXIS_PHYSICAL)
			{
				perm[cp++] = i_ext;
				i_ext++;
				i_ax_phys = i;
			}
			else if (axis_desc_state[i].type == TTNS_TENSOR_AXIS_AUXILIARY)
			{
				i_ext++;
			}
			else
			{
				assert(axis_desc_state[i].type == TTNS_TENSOR_AXIS_VIRTUAL);
				perm[cp++] = i_ext;
				i_ext += 2;
			}
		}
		i_ext = 0;
		for (int i = 0; i < psi->ndim; i++)
		{
			if (axis_desc_state[i].type == TTNS_TENSOR_AXIS_PHYSICAL)
			{
				i_ext++;
			}
			else if (axis_desc_state[i].type == TTNS_TENSOR_AXIS_AUXILIARY)
			{
				perm[cp++] = i_ext;
				i_ext++;
			}
			else
			{
				assert(axis_desc_state[i].type == TTNS_TENSOR_AXIS_VIRTUAL);
				perm[cp++] = i_ext + 1;
				i_ext += 2;
			}
		}
		assert(cp == psi_envs.ndim);

		if (!is_identity_permutation(perm, psi_envs.ndim))
		{
			struct block_sparse_tensor tmp;
			transpose_block_sparse_tensor(perm, &psi_envs, &tmp);
			delete_block_sparse_tensor(&psi_envs);
			psi_envs = tmp;  // copy internal data pointers
		}

		ct_free(perm);
		ct_free(axis_desc_state);
	}

	assert(i_ax_phys != -1);

	// contract with TTNO tensor
	// number of to-be contracted axes:
	// - physical: 1
	// - virtual:  #neighbors
	const int ndim_mult_op = 1 + topology->num_neighbors[i_site];
	block_sparse_tensor_dot(&op_perm, TENSOR_AXIS_RANGE_TRAILING, &psi_envs, TENSOR_AXIS_RANGE_LEADING, ndim_mult_op, ret);
	if (!op_perm_shallow_copy) {
		delete_block_sparse_tensor(&op_perm);
	}
	delete_block_sparse_tensor(&psi_envs);
	assert(ret->ndim == psi->ndim);

	// restore axes ordering if necessary
	if (i_ax_phys != 0)
	{
		int* perm = ct_malloc(ret->ndim * sizeof(int));
		for (int i = 0; i < i_ax_phys; i++) {
			perm[i] = i + 1;
		}
		perm[i_ax_phys] = 0;
		for (int i = i_ax_phys + 1; i < ret->ndim; i++) {
			perm[i] = i;
		}

		struct block_sparse_tensor tmp;
		transpose_block_sparse_tensor(perm, ret, &tmp);
		delete_block_sparse_tensor(ret);
		*ret = tmp;  // copy internal data pointers

		ct_free(perm);
	}
}
