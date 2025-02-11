/// \file su2_tensor.h
/// \brief Data structures and functions for SU(2) symmetric tensors.

#pragma once

#include "su2_tree.h"
#include "dense_tensor.h"


//________________________________________________________________________________________________________________________
///
/// \brief SU(2) symmetric tensor (assumed to be described internally by a fusion-splitting tree).
///
/// Axes are enumerated in the order "logical", "auxiliary", "internal"; number of internal dimensions is `ndim_logical + ndim_auxiliary - 3`
///
struct su2_tensor
{
	struct su2_fuse_split_tree tree;            //!< internal fusion-splitting tree
	struct su2_irreducible_list* outer_irreps;  //!< lists of irreducible 'j' quantum numbers times 2, for each outer (logical and auxiliary) axis
	struct charge_sectors charge_sectors;       //!< charge sectors (irreducible logical, auxiliary and internal 'j' quantum number configurations), computed from 'tree' and 'outer_irreps', and sorted lexicographically
	struct dense_tensor** degensors;            //!< dense "degeneracy" tensors, pointer array of length "number of charge sectors"
	long** dim_degen;                           //!< degeneracy dimension for each logical axis; indexed by corresponding 'j' quantum number
	enum numeric_type dtype;                    //!< numeric data type
	int ndim_logical;                           //!< number of logical dimensions
	int ndim_auxiliary;                         //!< number of auxiliary dimensions (dummy outer axes in fusion-splitting tree)
};


//________________________________________________________________________________________________________________________
//

// allocation and construction

void allocate_empty_su2_tensor(const enum numeric_type dtype, const int ndim_logical, int ndim_auxiliary, const struct su2_fuse_split_tree* tree, const struct su2_irreducible_list* outer_irreps, const long** dim_degen, struct su2_tensor* t);

void allocate_su2_tensor(const enum numeric_type dtype, const int ndim_logical, int ndim_auxiliary, const struct su2_fuse_split_tree* tree, const struct su2_irreducible_list* outer_irreps, const long** dim_degen, struct su2_tensor* t);

void delete_su2_tensor(struct su2_tensor* t);


//________________________________________________________________________________________________________________________
///
/// \brief Number of internal dimensions of an SU(2) symmetric tensor.
///
static inline int su2_tensor_ndim_internal(const struct su2_tensor* t)
{
	return t->ndim_logical + t->ndim_auxiliary - 3;
}

//________________________________________________________________________________________________________________________
///
/// \brief Number of overall dimensions of an SU(2) symmetric tensor.
///
static inline int su2_tensor_ndim(const struct su2_tensor* t)
{
	return t->ndim_logical + t->ndim_auxiliary + su2_tensor_ndim_internal(t);
}


long su2_tensor_dim_logical_axis(const struct su2_tensor* t, const int i_ax);


enum tensor_axis_direction su2_tensor_logical_axis_direction(const struct su2_tensor* t, const int i_ax);


//________________________________________________________________________________________________________________________
//

// internal consistency checking

bool su2_tensor_is_consistent(const struct su2_tensor* t);


//________________________________________________________________________________________________________________________
//

// in-place manipulation

static inline void su2_tensor_flip_trees(struct su2_tensor* t)
{
	su2_fuse_split_tree_flip(&t->tree);
}


bool su2_tensor_delete_charge_sector(struct su2_tensor* t, const qnumber* jlist);


//________________________________________________________________________________________________________________________
//

// F-move

void su2_tensor_fmove(const struct su2_tensor* restrict t, const int i_ax, struct su2_tensor* restrict r);


//________________________________________________________________________________________________________________________
//

// axis fusion and splitting

void su2_tensor_fuse_axes(const struct su2_tensor* restrict t, const int i_ax_0, const int i_ax_1, struct su2_tensor* restrict r);

void su2_tensor_split_axis(const struct su2_tensor* restrict t, const int i_ax_split, const int i_ax_add, const bool tree_left_child, const struct su2_irreducible_list outer_irreps[2], const long* dim_degen[2], struct su2_tensor* restrict r);


//________________________________________________________________________________________________________________________
//

// contraction

void su2_tensor_contract_simple(const struct su2_tensor* restrict s, const int* restrict i_ax_s, const struct su2_tensor* restrict t, const int* restrict i_ax_t, const int ndim_mult, struct su2_tensor* restrict r);

void su2_tensor_contract_yoga(const struct su2_tensor* restrict s, const int i_ax_s, const struct su2_tensor* restrict t, const int i_ax_t, struct su2_tensor* restrict r);


//________________________________________________________________________________________________________________________
//

// conversion to a dense tensor

void su2_to_dense_tensor(const struct su2_tensor* restrict s, struct dense_tensor* restrict t);
