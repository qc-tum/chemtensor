/// \file su2_tree.h
/// \brief Internal tree data structures for SU(2) symmetric tensors.

#pragma once

#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include "su2_irrep_lists.h"


//________________________________________________________________________________________________________________________
///
/// \brief SU(2) symmetry tree node.
///
struct su2_tree_node
{
	int i_ax;                    //!< tensor axis index (logical or auxiliary for leaf nodes, otherwise internal)
	struct su2_tree_node* c[2];  //!< pointer to left and right child nodes; NULL for a leaf node
};

void copy_su2_tree(const struct su2_tree_node* src, struct su2_tree_node* dst);

void delete_su2_tree(struct su2_tree_node* tree);

//________________________________________________________________________________________________________________________
///
/// \brief Whether an SU(2) symmetry tree node is a leaf.
///
static inline bool su2_tree_node_is_leaf(const struct su2_tree_node* node)
{
	if (node->c[0] == NULL)
	{
		// both pointers must be NULL
		assert(node->c[1] == NULL);
		return true;
	}
	else
	{
		// both pointers must be non-NULL
		assert(node->c[1] != NULL);
		return false;
	}
}

bool su2_tree_equal(const struct su2_tree_node* restrict s, const struct su2_tree_node* restrict t);

bool su2_tree_equal_topology(const struct su2_tree_node* restrict s, const struct su2_tree_node* restrict t);

bool su2_tree_contains_leaf(const struct su2_tree_node* tree, const int i_ax);

const struct su2_tree_node* su2_tree_find_node(const struct su2_tree_node* tree, const int i_ax);

const struct su2_tree_node* su2_tree_find_parent_node(const struct su2_tree_node* tree, const int i_ax);

int su2_tree_num_nodes(const struct su2_tree_node* tree);

int su2_tree_axes_list(const struct su2_tree_node* tree, int* list);

void su2_tree_axes_indicator(const struct su2_tree_node* tree, bool* indicator);

void su2_tree_update_axes_indices(struct su2_tree_node* tree, const int* axis_map);

const struct su2_tree_node* su2_subtree_with_leaf_axes(const struct su2_tree_node* tree, const int* i_ax, const int num_axes);

struct su2_tree_node* su2_tree_replace_subtree(struct su2_tree_node* tree, const int i_ax, struct su2_tree_node* subtree);

void su2_tree_fmove_left(struct su2_tree_node* tree);

void su2_tree_fmove_right(struct su2_tree_node* tree);

double su2_tree_eval_clebsch_gordan(const struct su2_tree_node* tree, const qnumber* restrict jlist, const int* restrict im_leaves, const int im_root);

void su2_tree_enumerate_charge_sectors(const struct su2_tree_node* tree, const int ndim, const struct su2_irreducible_list* leaf_ranges, struct charge_sectors* sectors);


//________________________________________________________________________________________________________________________
///
/// \brief Internal fuse and split tree of an SU(2) symmetric tensor.
///
struct su2_fuse_split_tree
{
	struct su2_tree_node* tree_fuse;    //!< root of fusion tree
	struct su2_tree_node* tree_split;   //!< root of splitting tree
	int ndim;                           //!< overall number of dimensions
};

void copy_su2_fuse_split_tree(const struct su2_fuse_split_tree* src, struct su2_fuse_split_tree* dst);

void delete_su2_fuse_split_tree(struct su2_fuse_split_tree* tree);

bool su2_fuse_split_tree_is_consistent(const struct su2_fuse_split_tree* tree);

bool su2_fuse_split_tree_equal(const struct su2_fuse_split_tree* s, const struct su2_fuse_split_tree* t);

void su2_fuse_split_tree_flip(struct su2_fuse_split_tree* tree);

void su2_fuse_split_tree_update_axes_indices(struct su2_fuse_split_tree* tree, const int* axis_map);

double su2_fuse_split_tree_eval_clebsch_gordan(const struct su2_fuse_split_tree* tree, const qnumber* restrict jlist, const int* restrict im_leaves);

void su2_fuse_split_tree_enumerate_charge_sectors(const struct su2_fuse_split_tree* tree, const struct su2_irreducible_list* leaf_ranges, struct charge_sectors* sectors);
