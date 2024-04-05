/// \file su2_tree.h
/// \brief Internal tree data structure for SU(2) symmetric tensors.

#pragma once

#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include "qnumber.h"


//________________________________________________________________________________________________________________________
///
/// \brief List of irreducible 'j' quantum numbers.
///
struct su2_irreducible_list
{
	qnumber* jlist;  //!< 'j' quantum numbers times 2
	int num;         //!< number of irreducible SU(2) subspaces (length of jlist)
};


//________________________________________________________________________________________________________________________
///
/// \brief Charge sectors ('j' quantum number configurations).
///
struct charge_sectors
{
	qnumber* jlists;  //!< matrix of 'j' quantum numbers times 2, of dimension nsec x ndim
	long nsec;        //!< number of sectors (configurations)
	int ndim;         //!< number of dimensions ('j' quantum numbers in each configuration)
};

void delete_charge_sectors(struct charge_sectors* sectors);


//________________________________________________________________________________________________________________________
///
/// \brief SU(2) symmetry tree node.
///
struct su2_tree_node
{
	int i_ax;                    //!< logical axis index (logical or auxiliary for leaf nodes, otherwise internal)
	struct su2_tree_node* c[2];  //!< pointer to left and right child nodes; NULL for a leaf node
};

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

int su2_tree_num_nodes(const struct su2_tree_node* tree);

void su2_tree_enumerate_charge_sectors(const struct su2_tree_node* tree, const int ndim, const struct su2_irreducible_list* leaf_ranges, struct charge_sectors* sectors);
