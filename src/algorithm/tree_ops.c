/// \file tree_ops.c
/// \brief Higher-level tensor network operations on a tree topology.

#include "tree_ops.h"
#include "aligned_memory.h"


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
/// \brief Compute the inner product between a TTNO and two TTNS, `<chi | op | psi>`.
///
void ttno_inner_product(const struct ttns* chi, const struct ttno* op, const struct ttns* psi, void* ret)
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

	// degree 3 tensors on virtual bonds corresponding to contracted subtrees
	struct block_sparse_tensor** r_bonds = ct_calloc(nsites * nsites, sizeof(struct block_sparse_tensor*));

	// iterate over sites by decreasing distance from root
	for (int l = nsites - 1; l >= 0; l--)
	{
		if (l > 0) {
			assert(sd[l - 1].distance <= sd[l].distance);
		}

		const int i_site   = sd[l].i_node;
		const int i_parent = sd[l].i_parent;
		assert(l > 0 || i_parent == -1);

		const int offset_phys_aux = (i_site == 0 ? 2 : 1);

		// contract the local tensor in 'psi' with the tensors on the bonds towards the children
		struct block_sparse_tensor psi_a_bonds;
		copy_block_sparse_tensor(&psi->a[i_site], &psi_a_bonds);

		// using decreasing axis index sequence here to preserve preceeding virtual bond axes indices
		// (relying on the property that the list of neighboring nodes is sorted)
		for (int n = psi->topology.num_neighbors[i_site] - 1; n >= 0; n--)
		{
			const int k = psi->topology.neighbor_map[i_site][n];
			assert(k != i_site);
			if (k == i_parent) {
				continue;
			}

			const int ib = edge_to_bond_index(nsites, i_site, k);
			assert(r_bonds[ib] != NULL);
			assert(r_bonds[ib]->ndim == 3);

			struct block_sparse_tensor tmp;
			const int i_ax = (k < i_site ? n : n + offset_phys_aux);
			block_sparse_tensor_multiply_axis(&psi_a_bonds, i_ax, r_bonds[ib], TENSOR_AXIS_RANGE_LEADING, &tmp);
			delete_block_sparse_tensor(&psi_a_bonds);
			psi_a_bonds = tmp;  // copy internal data pointers
		}
		assert(psi_a_bonds.ndim == psi->a[i_site].ndim + psi->topology.num_neighbors[i_site] - (i_parent == -1 ? 0 : 1));

		struct ttns_tensor_axis_desc* axis_desc_op_psi = ct_malloc((op->a[i_site].ndim + psi->a[i_site].ndim) * sizeof(struct ttns_tensor_axis_desc));
		int ca = 0;  // counter

		// move virtual parent bond and physical output axis of the local operator to the front
		struct block_sparse_tensor op_a;
		{
			struct ttno_tensor_axis_desc* axis_desc_op = ct_malloc(op->a[i_site].ndim * sizeof(struct ttno_tensor_axis_desc));
			ttno_tensor_get_axis_desc(op, i_site, axis_desc_op);

			int* perm = ct_malloc(op->a[i_site].ndim * sizeof(int));
			int cp = 0;
			for (int i = 0; i < op->a[i_site].ndim; i++)
			{
				if (axis_desc_op[i].type == TTNO_TENSOR_AXIS_VIRTUAL && axis_desc_op[i].index == i_parent) {
					perm[cp++] = i;
					axis_desc_op_psi[ca].type  = TTNS_TENSOR_AXIS_VIRTUAL;
					axis_desc_op_psi[ca].index = (i_parent << 16);  // use special encoding for parent bond of 'op', to distinguish it from 'psi'
					ca++;
				}
			}
			for (int i = 0; i < op->a[i_site].ndim; i++)
			{
				if (axis_desc_op[i].type == TTNO_TENSOR_AXIS_PHYS_OUT) {
					perm[cp++] = i;
					axis_desc_op_psi[ca].type  = TTNS_TENSOR_AXIS_PHYSICAL;
					axis_desc_op_psi[ca].index = i_site;
					ca++;
				}
			}
			for (int i = 0; i < op->a[i_site].ndim; i++)
			{
				if (axis_desc_op[i].type == TTNO_TENSOR_AXIS_PHYS_IN ||
				  ((axis_desc_op[i].type == TTNO_TENSOR_AXIS_VIRTUAL && axis_desc_op[i].index != i_parent))) {
					perm[cp++] = i;
				}
			}
			assert(cp == op->a[i_site].ndim);

			if (is_identity_permutation(perm, op->a[i_site].ndim)) {
				copy_block_sparse_tensor(&op->a[i_site], &op_a);
			}
			else {
				transpose_block_sparse_tensor(perm, &op->a[i_site], &op_a);
			}

			ct_free(perm);
			ct_free(axis_desc_op);
		}

		// move operator bonds and physical axis in 'psi_a_bonds' to the front
		{
			struct ttns_tensor_axis_desc* axis_desc_psi = ct_malloc(psi->a[i_site].ndim * sizeof(struct ttns_tensor_axis_desc));
			ttns_tensor_get_axis_desc(psi, i_site, axis_desc_psi);

			int* perm = ct_malloc(psi_a_bonds.ndim * sizeof(int));
			int cp = 0;
			int i_ext = 0;
			for (int i = 0; i < psi->a[i_site].ndim; i++)
			{
				if (axis_desc_psi[i].type == TTNS_TENSOR_AXIS_PHYSICAL)
				{
					perm[cp++] = i_ext;
					i_ext++;
				}
				else if (axis_desc_psi[i].type == TTNS_TENSOR_AXIS_AUXILIARY)
				{
					i_ext++;
				}
				else
				{
					assert(axis_desc_psi[i].type == TTNS_TENSOR_AXIS_VIRTUAL);
					if (axis_desc_psi[i].index == i_parent) {  // bond to parent site
						i_ext++;
					}
					else {
						perm[cp++] = i_ext;
						i_ext += 2;
					}
				}
			}
			i_ext = 0;
			for (int i = 0; i < psi->a[i_site].ndim; i++)
			{
				if (axis_desc_psi[i].type == TTNS_TENSOR_AXIS_PHYSICAL)
				{
					i_ext++;
				}
				else if (axis_desc_psi[i].type == TTNS_TENSOR_AXIS_AUXILIARY)
				{
					perm[cp++] = i_ext;
					i_ext++;
					axis_desc_op_psi[ca].type  = TTNS_TENSOR_AXIS_AUXILIARY;
					axis_desc_op_psi[ca].index = i_site;
					ca++;
				}
				else
				{
					assert(axis_desc_psi[i].type == TTNS_TENSOR_AXIS_VIRTUAL);
					if (axis_desc_psi[i].index == i_parent) {  // bond to parent site
						perm[cp++] = i_ext;
						i_ext++;
					}
					else {
						perm[cp++] = i_ext + 1;
						i_ext += 2;
					}
					axis_desc_op_psi[ca].type  = TTNS_TENSOR_AXIS_VIRTUAL;
					axis_desc_op_psi[ca].index = axis_desc_psi[i].index;
					ca++;
				}
			}
			assert(cp == psi_a_bonds.ndim);

			if (!is_identity_permutation(perm, psi_a_bonds.ndim))
			{
				struct block_sparse_tensor tmp;
				transpose_block_sparse_tensor(perm, &psi_a_bonds, &tmp);
				delete_block_sparse_tensor(&psi_a_bonds);
				psi_a_bonds = tmp;  // copy internal data pointers
			}

			ct_free(perm);
			ct_free(axis_desc_psi);
		}

		// contract with TTNO tensor
		// number of to-be contracted axes:
		// - physical: 1
		// - virtual:  #neighbors - (1 if not root else 0)
		const int ndim_mult_op = 1 + psi->topology.num_neighbors[i_site] - (i_parent == -1 ? 0 : 1);
		assert(op_a.ndim + psi_a_bonds.ndim - 2*ndim_mult_op == ca);
		assert(ndim_mult_op >= 1);
		struct block_sparse_tensor op_psi_a_bonds;
		block_sparse_tensor_dot(&op_a, TENSOR_AXIS_RANGE_TRAILING, &psi_a_bonds, TENSOR_AXIS_RANGE_LEADING, ndim_mult_op, &op_psi_a_bonds);
		delete_block_sparse_tensor(&op_a);
		delete_block_sparse_tensor(&psi_a_bonds);
		assert(op_psi_a_bonds.ndim == ca);

		// move parent bond axes of 'op_psi_a_bonds' to the front and interleave physical axis
		{
			int* perm = ct_malloc(op_psi_a_bonds.ndim * sizeof(int));
			int cp = 0;
			for (int i = 0; i < op_psi_a_bonds.ndim; i++)
			{
				if (axis_desc_op_psi[i].type == TTNS_TENSOR_AXIS_VIRTUAL && axis_desc_op_psi[i].index == i_parent)
				{
					// virtual bond to parent of original 'psi'
					perm[cp++] = i;
				}
			}
			for (int i = 0; i < op_psi_a_bonds.ndim; i++)
			{
				if (axis_desc_op_psi[i].type == TTNS_TENSOR_AXIS_VIRTUAL && axis_desc_op_psi[i].index == (i_parent << 16))
				{
					// virtual bond to parent of original 'op'
					perm[cp++] = i;
				}
			}
			for (int i = 0; i < op_psi_a_bonds.ndim; i++)
			{
				if (axis_desc_op_psi[i].type == TTNS_TENSOR_AXIS_VIRTUAL
					&& axis_desc_op_psi[i].index != i_parent
					&& axis_desc_op_psi[i].index != (i_parent << 16)
					&& axis_desc_op_psi[i].index < i_site)
				{
					perm[cp++] = i;
				}
			}
			for (int i = 0; i < op_psi_a_bonds.ndim; i++)
			{
				if (axis_desc_op_psi[i].type == TTNS_TENSOR_AXIS_PHYSICAL || axis_desc_op_psi[i].type == TTNS_TENSOR_AXIS_AUXILIARY)
				{
					assert(axis_desc_op_psi[i].index == i_site);
					perm[cp++] = i;
				}
			}
			for (int i = 0; i < op_psi_a_bonds.ndim; i++)
			{
				if (axis_desc_op_psi[i].type == TTNS_TENSOR_AXIS_VIRTUAL
					&& axis_desc_op_psi[i].index != i_parent
					&& axis_desc_op_psi[i].index != (i_parent << 16)
					&& axis_desc_op_psi[i].index > i_site)
				{
					perm[cp++] = i;
				}
			}
			assert(cp == op_psi_a_bonds.ndim);

			if (!is_identity_permutation(perm, op_psi_a_bonds.ndim))
			{
				struct block_sparse_tensor tmp;
				transpose_block_sparse_tensor(perm, &op_psi_a_bonds, &tmp);
				delete_block_sparse_tensor(&op_psi_a_bonds);
				op_psi_a_bonds = tmp;  // copy internal data pointers
			}

			ct_free(perm);
		}

		ct_free(axis_desc_op_psi);

		struct block_sparse_tensor chi_a_conj;
		copy_block_sparse_tensor(&chi->a[i_site], &chi_a_conj);
		conjugate_block_sparse_tensor(&chi_a_conj);
		block_sparse_tensor_reverse_axis_directions(&chi_a_conj);

		if (l > 0)  // not root node
		{
			assert(i_parent != -1);

			// move parent bond axis in 'chi_a_conj' to the front

			struct ttns_tensor_axis_desc* axis_desc_chi = ct_malloc(chi->a[i_site].ndim * sizeof(struct ttns_tensor_axis_desc));
			ttns_tensor_get_axis_desc(chi, i_site, axis_desc_chi);

			int* perm = ct_malloc(chi_a_conj.ndim * sizeof(int));
			int c = 0;
			for (int i = 0; i < chi->a[i_site].ndim; i++) {
				if (axis_desc_chi[i].type == TTNS_TENSOR_AXIS_VIRTUAL && axis_desc_chi[i].index == i_parent) {
					perm[c++] = i;
				}
			}
			for (int i = 0; i < chi->a[i_site].ndim; i++) {
				if (axis_desc_chi[i].type != TTNS_TENSOR_AXIS_VIRTUAL || axis_desc_chi[i].index != i_parent) {
					perm[c++] = i;
				}
			}
			assert(c == chi_a_conj.ndim);

			if (!is_identity_permutation(perm, chi_a_conj.ndim))
			{
				struct block_sparse_tensor tmp;
				transpose_block_sparse_tensor(perm, &chi_a_conj, &tmp);
				delete_block_sparse_tensor(&chi_a_conj);
				chi_a_conj = tmp;  // copy internal data pointers
			}

			ct_free(perm);
			ct_free(axis_desc_chi);
		}

		struct block_sparse_tensor r;
		const int ndim_mult_chi = chi_a_conj.ndim - (i_parent == -1 ? 0 : 1);
		assert(ndim_mult_chi >= 1);
		block_sparse_tensor_dot(&op_psi_a_bonds, TENSOR_AXIS_RANGE_TRAILING, &chi_a_conj, TENSOR_AXIS_RANGE_TRAILING, ndim_mult_chi/****imax(ndim_mult_chi, 1)****/, &r);
		delete_block_sparse_tensor(&op_psi_a_bonds);
		delete_block_sparse_tensor(&chi_a_conj);

		if (l == 0)  // root node
		{
			assert(r.ndim == 0);
			assert(r.blocks[0] != NULL);

			// copy scalar entry
			memcpy(ret, r.blocks[0]->data, sizeof_numeric_type(r.dtype));

			delete_block_sparse_tensor(&r);
		}
		else
		{
			assert(r.ndim == 3);
			const int ib = edge_to_bond_index(nsites, i_site, i_parent);
			assert(r_bonds[ib] == NULL);
			r_bonds[ib] = ct_malloc(sizeof(struct block_sparse_tensor));
			*r_bonds[ib] = r;  // copy internal data pointers
		}
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
