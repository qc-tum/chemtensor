/// \file su2_tree.c
/// \brief Internal tree data structure for SU(2) symmetric tensors.

#include <math.h>
#include <assert.h>
#include "su2_tree.h"
#include "aligned_memory.h"
#include "queue.h"
#include "linked_list.h"


//________________________________________________________________________________________________________________________
///
/// \brief Delete a charge sector array (free memory).
///
void delete_charge_sectors(struct charge_sectors* sectors)
{
	aligned_free(sectors->jlists);
	sectors->nsec = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Number of nodes in an SU(2) symmetry tree.
///
int su2_tree_num_nodes(const struct su2_tree_node* tree)
{
	if (tree == NULL) {
		return 0;
	}

	return 1 + su2_tree_num_nodes(tree->c[0]) + su2_tree_num_nodes(tree->c[1]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Reset the entry in 'jlist' indexed by 'node'.
///
static inline void su2_tree_node_reset_quantum_number(const struct su2_tree_node* node, const struct su2_irreducible_list* leaf_ranges, qnumber* restrict jlist)
{
	if (su2_tree_node_is_leaf(node))
	{
		assert(leaf_ranges[node->i_ax].num >= 1);
		jlist[node->i_ax] = leaf_ranges[node->i_ax].jlist[0];
	}
	else
	{
		// j_3 = |j_1 - j_2|
		jlist[node->i_ax] = abs(jlist[node->c[0]->i_ax] - jlist[node->c[1]->i_ax]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Enumerate the next "charge sector" (irreducible 'j' quantum number configuration) for a given tree layout.
///
/// The input nodes must be ordered (e.g., by tree height), such that the downstream nodes which a given node depends on appear later in the list.
///
static bool su2_tree_next_charge_sector(const struct su2_tree_node** ordered_nodes, const int num_nodes, const struct su2_irreducible_list* leaf_ranges, qnumber* restrict jlist)
{
	for (int i = 0; i < num_nodes; i++)
	{
		const struct su2_tree_node* node = ordered_nodes[i];

		if (su2_tree_node_is_leaf(node))
		{
			// index of current 'j' quantum number
			int k;
			for (k = 0; k < leaf_ranges[node->i_ax].num; k++)
			{
				if (jlist[node->i_ax] == leaf_ranges[node->i_ax].jlist[k]) {
					break;
				}
			}
			assert(k < leaf_ranges[node->i_ax].num);
			if (k < leaf_ranges[node->i_ax].num - 1)
			{
				jlist[node->i_ax] = leaf_ranges[node->i_ax].jlist[k + 1];

				// reset all preceeding nodes
				for (int j = i - 1; j >= 0; j--) {
					su2_tree_node_reset_quantum_number(ordered_nodes[j], leaf_ranges, jlist);
				}

				return true;
			}
		}
		else // not a leaf node
		{
			if (jlist[node->i_ax] < jlist[node->c[0]->i_ax] + jlist[node->c[1]->i_ax])
			{
				// j_3 < j_1 + j_2, thus can increment j_3
				jlist[node->i_ax] += 2;
				assert(jlist[node->i_ax] <= jlist[node->c[0]->i_ax] + jlist[node->c[1]->i_ax]);

				// reset all preceeding nodes
				for (int j = i - 1; j >= 0; j--) {
					su2_tree_node_reset_quantum_number(ordered_nodes[j], leaf_ranges, jlist);
				}

				return true;
			}
		}
	}

	return false;
}


//________________________________________________________________________________________________________________________
///
/// \brief Utility function for creating a copy of 'data'.
///
static void* allocate_copy(const void* data, const size_t size)
{
	void* p = aligned_alloc(MEM_DATA_ALIGN, size);
	memcpy(p, data, size);
	return p;
}


//________________________________________________________________________________________________________________________
///
/// \brief Enumerate all "charge sectors" ('j' quantum number configurations) for a given tree layout.
/// 'ndim' is the number of 'j' quantum numbers.
/// The output array 'charge_sectors' will be allocated, with dimension 'number of sectors x ndim'.
///
void su2_tree_enumerate_charge_sectors(const struct su2_tree_node* tree, const int ndim, const struct su2_irreducible_list* leaf_ranges, struct charge_sectors* sectors)
{
	assert(tree != NULL);

	const int num_nodes = su2_tree_num_nodes(tree);

	const struct su2_tree_node** ordered_nodes = aligned_alloc(MEM_DATA_ALIGN, num_nodes * sizeof(struct su2_tree_node*));

	// traverse tree breadth-first to assemble ordered nodes
	{
		struct queue q = { 0 };
		enqueue(&q, (void*)tree);
		int c = 0;
		while (!queue_is_empty(&q))
		{
			const struct su2_tree_node* node = dequeue(&q);
			ordered_nodes[c] = node;
			c++;
			if (!su2_tree_node_is_leaf(node))
			{
				enqueue(&q, (void*)node->c[0]);
				enqueue(&q, (void*)node->c[1]);
			}
		}
		assert(c == num_nodes);
	}

	qnumber* jlist = aligned_calloc(MEM_DATA_ALIGN, ndim, sizeof(qnumber));
	// initialize internal 'j' quantum numbers with their smallest possible values
	for (int i = num_nodes - 1; i >= 0; i--)  // start with unrestricted (logical or auxiliary) leaf nodes first
	{
		su2_tree_node_reset_quantum_number(ordered_nodes[i], leaf_ranges, jlist);
	}

	// enumerate and store all charge sectors
	struct linked_list charges = { 0 };
	do {
		linked_list_append(&charges, allocate_copy(jlist, ndim * sizeof(qnumber)));
	}
	while (su2_tree_next_charge_sector(ordered_nodes, num_nodes, leaf_ranges, jlist));

	// copy charge sectors to output array
	sectors->ndim = ndim;
	sectors->nsec = charges.size;
	sectors->jlists = aligned_alloc(MEM_DATA_ALIGN, charges.size * ndim * sizeof(qnumber));
	long i = 0;
	struct linked_list_node* node = charges.head;
	while (node != NULL)
	{
		memcpy(&sectors->jlists[i * ndim], node->data, ndim * sizeof(qnumber));
		node = node->next;
		i++;
	}
	assert(i == charges.size);

	delete_linked_list(&charges, aligned_free);
	aligned_free(jlist);
	aligned_free(ordered_nodes);
}
