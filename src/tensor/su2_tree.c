/// \file su2_tree.c
/// \brief Internal tree data structures for SU(2) symmetric tensors.

#include <math.h>
#include "su2_tree.h"
#include "aligned_memory.h"
#include "queue.h"
#include "linked_list.h"
#include "clebsch_gordan.h"


//________________________________________________________________________________________________________________________
///
/// \brief Make a deep copy of an SU(2) symmetry tree.
///
void copy_su2_tree(const struct su2_tree_node* src, struct su2_tree_node* dst)
{
	assert(src != NULL);

	dst->i_ax = src->i_ax;

	if (src->c[0] != NULL)
	{
		assert(src->c[1] != NULL);
		dst->c[0] = ct_malloc(sizeof(struct su2_tree_node));
		dst->c[1] = ct_malloc(sizeof(struct su2_tree_node));
		copy_su2_tree(src->c[0], dst->c[0]);
		copy_su2_tree(src->c[1], dst->c[1]);
	}
	else
	{
		assert(src->c[1] == NULL);
		dst->c[0] = NULL;
		dst->c[1] = NULL;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an SU(2) symmetry tree (free memory).
///
/// The root node must be deallocated by the calling function.
///
void delete_su2_tree(struct su2_tree_node* tree)
{
	if (tree->c[0] != NULL)
	{
		assert(tree->c[1] != NULL);

		delete_su2_tree(tree->c[0]);
		delete_su2_tree(tree->c[1]);

		ct_free(tree->c[0]);
		ct_free(tree->c[1]);
		tree->c[0] = NULL;
		tree->c[1] = NULL;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Whether two SU(2) symmetry trees are logically equal.
///
bool su2_tree_equal(const struct su2_tree_node* restrict s, const struct su2_tree_node* restrict t)
{
	if (s->i_ax != t->i_ax) {
		return false;
	}

	// test whether both nodes are leaves
	const bool leaf_s = su2_tree_node_is_leaf(s);
	const bool leaf_t = su2_tree_node_is_leaf(t);
	if (leaf_s != leaf_t) {
		return false;
	}
	if (leaf_s) {
		// both nodes are leaves
		return true;
	}

	// test whether subtrees are equal
	return su2_tree_equal(s->c[0], t->c[0]) && su2_tree_equal(s->c[1], t->c[1]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Whether two SU(2) symmetry trees have the same topology.
///
bool su2_tree_equal_topology(const struct su2_tree_node* restrict s, const struct su2_tree_node* restrict t)
{
	// test whether both nodes are leaves
	const bool leaf_s = su2_tree_node_is_leaf(s);
	const bool leaf_t = su2_tree_node_is_leaf(t);
	if (leaf_s != leaf_t) {
		return false;
	}
	if (leaf_s) {
		// both nodes are leaves
		return true;
	}

	// test whether subtrees have the same topology
	return su2_tree_equal_topology(s->c[0], t->c[0]) && su2_tree_equal_topology(s->c[1], t->c[1]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Whether the SU(2) symmetry tree contains a leaf with axis index 'i_ax'.
///
bool su2_tree_contains_leaf(const struct su2_tree_node* tree, const int i_ax)
{
	if (tree == NULL) {
		return false;
	}

	if (su2_tree_node_is_leaf(tree))
	{
		return tree->i_ax == i_ax;
	}

	return su2_tree_contains_leaf(tree->c[0], i_ax) || su2_tree_contains_leaf(tree->c[1], i_ax);
}


//________________________________________________________________________________________________________________________
///
/// \brief Locate the node with axis index 'i_ax', returning NULL if such a node cannot be found.
///
const struct su2_tree_node* su2_tree_find_node(const struct su2_tree_node* tree, const int i_ax)
{
	if (tree == NULL) {
		return NULL;
	}

	if (tree->i_ax == i_ax) {
		return tree;
	}

	// search in left subtree
	const struct su2_tree_node* node = su2_tree_find_node(tree->c[0], i_ax);
	if (node != NULL) {
		return node;
	}
	// search in right subtree
	return su2_tree_find_node(tree->c[1], i_ax);
}


//________________________________________________________________________________________________________________________
///
/// \brief Locate the parent of the node with axis index 'i_ax', returning NULL if parent cannot be found.
///
const struct su2_tree_node* su2_tree_find_parent_node(const struct su2_tree_node* tree, const int i_ax)
{
	if (tree == NULL) {
		return NULL;
	}
	if (su2_tree_node_is_leaf(tree)) {
		return NULL;
	}

	if (tree->c[0]->i_ax == i_ax || tree->c[1]->i_ax == i_ax) {
		return tree;
	}

	// search in left subtree
	const struct su2_tree_node* node = su2_tree_find_parent_node(tree->c[0], i_ax);
	if (node != NULL) {
		return node;
	}
	// search in right subtree
	return su2_tree_find_parent_node(tree->c[1], i_ax);
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
/// \brief Fill the list of axis indices contained in the tree, and return the number of axes.
///
int su2_tree_axes_list(const struct su2_tree_node* tree, int* list)
{
	if (tree == NULL) {
		return 0;
	}

	// in-order traversal
	int n0 = su2_tree_axes_list(tree->c[0], list);
	list[n0] = tree->i_ax;
	int n1 = su2_tree_axes_list(tree->c[1], list + n0 + 1);

	return n0 + 1 + n1;
}


//________________________________________________________________________________________________________________________
///
/// \brief Fill the indicator with the axes indexed by the tree.
///
void su2_tree_axes_indicator(const struct su2_tree_node* tree, bool* indicator)
{
	if (tree == NULL) {
		return;
	}

	assert(tree->i_ax >= 0);
	// must not be assigned yet
	assert(!indicator[tree->i_ax]);
	indicator[tree->i_ax] = true;

	su2_tree_axes_indicator(tree->c[0], indicator);
	su2_tree_axes_indicator(tree->c[1], indicator);
}


//________________________________________________________________________________________________________________________
///
/// \brief Update the axes indices using the provided map.
///
void su2_tree_update_axes_indices(struct su2_tree_node* tree, const int* axis_map)
{
	if (tree == NULL) {
		return;
	}

	assert(tree->i_ax >= 0);
	tree->i_ax = axis_map[tree->i_ax];

	su2_tree_update_axes_indices(tree->c[0], axis_map);
	su2_tree_update_axes_indices(tree->c[1], axis_map);
}


//________________________________________________________________________________________________________________________
///
/// \brief Temporary data structure for finding a subtree with leaf axis indices exactly matching a given list.
///
struct subtree_with_leaf_axes_data
{
	const struct su2_tree_node* subtree;  //!< pointer to found subtree, or NULL if not yet found
	bool all_leaves_in_list;              //!< whether axis indices of all leaves are contained in provided list
	int num_leaves;                       //!< number of leaves in current subtree (if 'all_leaves_in_list' is true)
};


//________________________________________________________________________________________________________________________
///
/// \brief Internal recursive function for finding a subtree with leaf axis indices exactly matching the provided 'i_ax' list.
///
/// Entries in 'i_ax' must be pairwise different.
///
static struct subtree_with_leaf_axes_data su2_tree_search_subtree_with_leaf_axes(const struct su2_tree_node* tree, const int* i_ax, const int num_axes)
{
	assert(tree != NULL);

	if (su2_tree_node_is_leaf(tree))
	{
		struct subtree_with_leaf_axes_data data = { 0 };

		for (int i = 0; i < num_axes; i++) {
			if (tree->i_ax == i_ax[i]) {
				data.all_leaves_in_list = true;
				break;
			}
		}

		if (data.all_leaves_in_list) {
			data.num_leaves = 1;
			if (num_axes == 1) {
				// searching for a single axis, which matches current leaf
				data.subtree = tree;
			}
		}

		return data;
	}

	struct subtree_with_leaf_axes_data data0 = su2_tree_search_subtree_with_leaf_axes(tree->c[0], i_ax, num_axes);
	struct subtree_with_leaf_axes_data data1 = su2_tree_search_subtree_with_leaf_axes(tree->c[1], i_ax, num_axes);

	// fast return if subtree has already been found
	if (data0.subtree != NULL) {
		return data0;
	}
	if (data1.subtree != NULL) {
		return data1;
	}

	struct subtree_with_leaf_axes_data data = { 0 };

	data.all_leaves_in_list = (data0.all_leaves_in_list && data1.all_leaves_in_list);
	if (data.all_leaves_in_list)
	{
		data.num_leaves = data0.num_leaves + data1.num_leaves;
		if (data.num_leaves == num_axes) {
			// found subtree
			data.subtree = tree;
		}
	}

	return data;
}


//________________________________________________________________________________________________________________________
///
/// \brief Find the subtree with the specified leaf axis indices.
///
/// Ordering of axis indices is not relevant. Returns NULL if subtree does not exist.
///
const struct su2_tree_node* su2_subtree_with_leaf_axes(const struct su2_tree_node* tree, const int* i_ax, const int num_axes)
{
	return su2_tree_search_subtree_with_leaf_axes(tree, i_ax, num_axes).subtree;
}


//________________________________________________________________________________________________________________________
///
/// \brief Replace the proper subtree with axis index 'i_ax' by 'subtree', and return a pointer to the old subtree, or NULL if 'i_ax' cannot be found.
///
struct su2_tree_node* su2_tree_replace_subtree(struct su2_tree_node* tree, const int i_ax, struct su2_tree_node* subtree)
{
	if (tree == NULL) {
		return NULL;
	}
	if (su2_tree_node_is_leaf(tree)) {
		return NULL;
	}

	if (tree->c[0]->i_ax == i_ax) {
		struct su2_tree_node* old = tree->c[0];
		tree->c[0] = subtree;
		return old;
	}

	if (tree->c[1]->i_ax == i_ax) {
		struct su2_tree_node* old = tree->c[1];
		tree->c[1] = subtree;
		return old;
	}

	// search in left subtree
	struct su2_tree_node* old = su2_tree_replace_subtree(tree->c[0], i_ax, subtree);
	if (old != NULL) {
		return old;
	}
	// search in right subtree
	return su2_tree_replace_subtree(tree->c[1], i_ax, subtree);
}


//________________________________________________________________________________________________________________________
///
/// \brief Perform an in-place left-rotating F-move rooted at the node pointed to by 'tree', retaining the axis indices.
///
void su2_tree_fmove_left(struct su2_tree_node* tree)
{
	assert(!su2_tree_node_is_leaf(tree));
	struct su2_tree_node* d = tree->c[1];
	assert(!su2_tree_node_is_leaf(d));
	struct su2_tree_node* b = d->c[0];
	d->c[0] = tree->c[0];
	tree->c[0] = d;
	tree->c[1] = d->c[1];
	d->c[1] = b;
}


//________________________________________________________________________________________________________________________
///
/// \brief Perform an in-place right-rotating F-move rooted at the node pointed to by 'tree', retaining the axis indices.
///
void su2_tree_fmove_right(struct su2_tree_node* tree)
{
	assert(!su2_tree_node_is_leaf(tree));
	struct su2_tree_node* d = tree->c[0];
	assert(!su2_tree_node_is_leaf(d));
	struct su2_tree_node* b = d->c[1];
	d->c[1] = tree->c[1];
	tree->c[1] = d;
	tree->c[0] = d->c[0];
	d->c[0] = b;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the SU(2) symmetry tree by contracting it, interpreting the nodes as Clebsch-Gordan coefficients.
/// The 'j' quantum number configuration and the 'm' quantum numbers at the leaves and root are prescribed.
///
double su2_tree_eval_clebsch_gordan(const struct su2_tree_node* tree, const qnumber* restrict jlist, const int* restrict im_leaves, const int im_root)
{
	assert(0 <= im_root && im_root < jlist[tree->i_ax] + 1);

	if (su2_tree_node_is_leaf(tree))
	{
		return (im_leaves[tree->i_ax] == im_root) ? 1 : 0;
	}

	const qnumber j1 = jlist[tree->c[0]->i_ax];
	const qnumber j2 = jlist[tree->c[1]->i_ax];
	const qnumber j3 = jlist[tree->i_ax];

	double* e2_list = ct_malloc((j2 + 1) * sizeof(double));
	for (int im2 = 0; im2 <= j2; im2++) {
		e2_list[im2] = su2_tree_eval_clebsch_gordan(tree->c[1], jlist, im_leaves, im2);
	}

	double v = 0;
	for (int im1 = 0; im1 <= j1; im1++)
	{
		double e1 = su2_tree_eval_clebsch_gordan(tree->c[0], jlist, im_leaves, im1);
		if (e1 == 0) {
			continue;
		}

		for (int im2 = 0; im2 <= j2; im2++)
		{
			if (e2_list[im2] == 0) {
				continue;
			}

			v += e1 * e2_list[im2] * clebsch_gordan(j1, j2, j3, im1, im2, im_root);
		}
	}

	ct_free(e2_list);

	return v;
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
	void* p = ct_malloc(size);
	memcpy(p, data, size);
	return p;
}


//________________________________________________________________________________________________________________________
///
/// \brief Enumerate all "charge sectors" ('j' quantum number configurations) for a given tree layout.
/// 'ndim' is the number of 'j' quantum numbers.
/// The output array 'sectors' will be allocated, with dimension 'number of sectors x ndim'.
///
void su2_tree_enumerate_charge_sectors(const struct su2_tree_node* tree, const int ndim, const struct su2_irreducible_list* leaf_ranges, struct charge_sectors* sectors)
{
	assert(tree != NULL);

	const int num_nodes = su2_tree_num_nodes(tree);

	const struct su2_tree_node** ordered_nodes = ct_malloc(num_nodes * sizeof(struct su2_tree_node*));

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

	qnumber* jlist = ct_calloc(ndim, sizeof(qnumber));
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
	allocate_charge_sectors(charges.size, ndim, sectors);
	long i = 0;
	struct linked_list_node* node = charges.head;
	while (node != NULL)
	{
		memcpy(&sectors->jlists[i * ndim], node->data, ndim * sizeof(qnumber));
		node = node->next;
		i++;
	}
	assert(i == charges.size);

	delete_linked_list(&charges, ct_free);
	ct_free(jlist);
	ct_free(ordered_nodes);
}


//________________________________________________________________________________________________________________________
///
/// \brief Make a deep copy of a fuse and split tree.
///
void copy_su2_fuse_split_tree(const struct su2_fuse_split_tree* src, struct su2_fuse_split_tree* dst)
{
	dst->ndim = src->ndim;

	dst->tree_fuse  = ct_malloc(sizeof(struct su2_tree_node));
	dst->tree_split = ct_malloc(sizeof(struct su2_tree_node));
	copy_su2_tree(src->tree_fuse,  dst->tree_fuse);
	copy_su2_tree(src->tree_split, dst->tree_split);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a fuse and split tree (free memory).
///
void delete_su2_fuse_split_tree(struct su2_fuse_split_tree* tree)
{
	delete_su2_tree(tree->tree_split);
	delete_su2_tree(tree->tree_fuse);
	ct_free(tree->tree_split);
	ct_free(tree->tree_fuse);
}


//________________________________________________________________________________________________________________________
///
/// \brief Internal consistency check of the fuse and split tree data structure.
///
bool su2_fuse_split_tree_is_consistent(const struct su2_fuse_split_tree* tree)
{
	if (tree->ndim <= 0) {
		return false;
	}

	if ((tree->tree_fuse == NULL) || (tree->tree_split == NULL)) {
		return false;
	}

	if (tree->tree_fuse->i_ax != tree->tree_split->i_ax) {
		return false;
	}
	const int i_ax_shared = tree->tree_fuse->i_ax;

	// fusion and splitting trees cannot both be single leaves
	if (su2_tree_node_is_leaf(tree->tree_fuse) && su2_tree_node_is_leaf(tree->tree_split)) {
		return false;
	}

	bool* indicator_fuse  = ct_calloc(tree->ndim, sizeof(bool));
	bool* indicator_split = ct_calloc(tree->ndim, sizeof(bool));
	su2_tree_axes_indicator(tree->tree_fuse,  indicator_fuse);
	su2_tree_axes_indicator(tree->tree_split, indicator_split);
	for (int i = 0; i < tree->ndim; i++)
	{
		if (i == i_ax_shared) {
			if (!(indicator_fuse[i] && indicator_split[i])) {
				ct_free(indicator_split);
				ct_free(indicator_fuse);
				return false;
			}
		}
		else {
			// axis must be assigned either to the fusion or to the splitting tree
			if (indicator_fuse[i] == indicator_split[i]) {
				ct_free(indicator_split);
				ct_free(indicator_fuse);
				return false;
			}
		}
	}

	ct_free(indicator_split);
	ct_free(indicator_fuse);

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Whether two fuse and split trees are logically equal.
///
bool su2_fuse_split_tree_equal(const struct su2_fuse_split_tree* s, const struct su2_fuse_split_tree* t)
{
	if (s->ndim != t->ndim) {
		return false;
	}

	return su2_tree_equal(s->tree_fuse, t->tree_fuse) && su2_tree_equal(s->tree_split, t->tree_split);
}


//________________________________________________________________________________________________________________________
///
/// \brief Flip the fuse and split trees.
///
void su2_fuse_split_tree_flip(struct su2_fuse_split_tree* tree)
{
	struct su2_tree_node* tmp = tree->tree_fuse;
	tree->tree_fuse  = tree->tree_split;
	tree->tree_split = tmp;
}


//________________________________________________________________________________________________________________________
///
/// \brief Update the axes indices of a fuse and split tree using the provided map.
///
void su2_fuse_split_tree_update_axes_indices(struct su2_fuse_split_tree* tree, const int* axis_map)
{
	su2_tree_update_axes_indices(tree->tree_fuse,  axis_map);
	su2_tree_update_axes_indices(tree->tree_split, axis_map);
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the fuse and split tree by contracting both subtrees, interpreting the nodes as Clebsch-Gordan coefficients.
/// The 'j' quantum number configuration and the 'm' quantum numbers at the leaves are prescribed.
///
double su2_fuse_split_tree_eval_clebsch_gordan(const struct su2_fuse_split_tree* tree, const qnumber* restrict jlist, const int* restrict im_leaves)
{
	assert(tree->tree_fuse->i_ax == tree->tree_split->i_ax);

	const qnumber j_root = jlist[tree->tree_fuse->i_ax];

	double v = 0;
	for (int im_root = 0; im_root <= j_root; im_root++) {
		v += (su2_tree_eval_clebsch_gordan(tree->tree_fuse,  jlist, im_leaves, im_root)
		    * su2_tree_eval_clebsch_gordan(tree->tree_split, jlist, im_leaves, im_root));
	}

	return v;
}


//________________________________________________________________________________________________________________________
///
/// \brief Comparison function used by 'qsort'.
///
static int compare_su2_irreducible_lists_wrapper(const void* a, const void* b)
{
	return compare_su2_irreducible_lists(
		(const struct su2_irreducible_list*)a,
		(const struct su2_irreducible_list*)b);
}


//________________________________________________________________________________________________________________________
///
/// \brief Enumerate all "charge sectors" ('j' quantum number configurations) for a given fuse and split tree layout.
///
void su2_fuse_split_tree_enumerate_charge_sectors(const struct su2_fuse_split_tree* tree, const struct su2_irreducible_list* leaf_ranges, struct charge_sectors* sectors)
{
	assert(tree->ndim > 0);
	assert(tree->tree_fuse->i_ax == tree->tree_split->i_ax);
	const int i_ax_shared = tree->tree_fuse->i_ax;

	bool* indicator_fuse  = ct_calloc(tree->ndim, sizeof(bool));
	bool* indicator_split = ct_calloc(tree->ndim, sizeof(bool));
	su2_tree_axes_indicator(tree->tree_fuse,  indicator_fuse);
	su2_tree_axes_indicator(tree->tree_split, indicator_split);
	// consistency check
	for (int i = 0; i < tree->ndim; i++) {
		if (i == i_ax_shared) {
			assert(indicator_fuse[i] && indicator_split[i]);
		}
		else {
			assert(indicator_fuse[i] != indicator_split[i]);
		}
	}

	struct charge_sectors sectors_fuse;
	struct charge_sectors sectors_split;
	su2_tree_enumerate_charge_sectors(tree->tree_fuse,  tree->ndim, leaf_ranges, &sectors_fuse);
	su2_tree_enumerate_charge_sectors(tree->tree_split, tree->ndim, leaf_ranges, &sectors_split);
	assert(sectors_fuse.ndim  == tree->ndim);
	assert(sectors_split.ndim == tree->ndim);

	// merge the charge sectors of the fuse and split trees
	struct su2_irreducible_list* merged_sectors = ct_malloc(sectors_fuse.nsec * sectors_split.nsec * sizeof(struct su2_irreducible_list));
	long c = 0;
	for (long j = 0; j < sectors_fuse.nsec; j++)
	{
		for (long k = 0; k < sectors_split.nsec; k++)
		{
			// quantum number at common root edge must match
			if (sectors_fuse.jlists[j * tree->ndim + i_ax_shared] == sectors_split.jlists[k * tree->ndim + i_ax_shared])
			{
				merged_sectors[c].num = tree->ndim;
				merged_sectors[c].jlist = ct_malloc(tree->ndim * sizeof(qnumber));
				// merge quantum numbers
				for (int i = 0; i < tree->ndim; i++)
				{
					merged_sectors[c].jlist[i] =
						(indicator_fuse[i] ?
							 sectors_fuse.jlists[j * tree->ndim + i] :
							sectors_split.jlists[k * tree->ndim + i]);
				}

				c++;
			}
		}
	}

	// sort lexicographically
	qsort(merged_sectors, c, sizeof(struct su2_irreducible_list), compare_su2_irreducible_lists_wrapper);

	// copy data into output array
	allocate_charge_sectors(c, tree->ndim, sectors);
	for (long i = 0; i < c; i++)
	{
		memcpy(&sectors->jlists[i * tree->ndim], merged_sectors[i].jlist, tree->ndim * sizeof(qnumber));
		ct_free(merged_sectors[i].jlist);
	}

	ct_free(merged_sectors);
	ct_free(indicator_split);
	ct_free(indicator_fuse);
	delete_charge_sectors(&sectors_split);
	delete_charge_sectors(&sectors_fuse);
}
