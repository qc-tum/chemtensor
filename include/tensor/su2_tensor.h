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
	ct_long** dim_degen;                        //!< degeneracy dimension for each logical axis; indexed by corresponding 'j' quantum number
	enum numeric_type dtype;                    //!< numeric data type
	int ndim_logical;                           //!< number of logical dimensions
	int ndim_auxiliary;                         //!< number of auxiliary dimensions (dummy outer axes in fusion-splitting tree)
};


//________________________________________________________________________________________________________________________
//

// allocation and construction

void allocate_empty_su2_tensor(const enum numeric_type dtype, const int ndim_logical, int ndim_auxiliary, const struct su2_fuse_split_tree* tree, const struct su2_irreducible_list* outer_irreps, const ct_long** dim_degen, struct su2_tensor* t);

void allocate_su2_tensor(const enum numeric_type dtype, const int ndim_logical, int ndim_auxiliary, const struct su2_fuse_split_tree* tree, const struct su2_irreducible_list* outer_irreps, const ct_long** dim_degen, struct su2_tensor* t);

void delete_su2_tensor(struct su2_tensor* t);

void copy_su2_tensor(const struct su2_tensor* src, struct su2_tensor* dst);


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


ct_long su2_tensor_dim_logical_axis(const struct su2_tensor* t, const int i_ax);

ct_long su2_tensor_num_elements_logical(const struct su2_tensor* t);


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

void su2_tensor_delete_charge_sector_by_index(struct su2_tensor* t, const ct_long idx);
bool su2_tensor_delete_charge_sector(struct su2_tensor* t, const qnumber* jlist);

void scale_su2_tensor(const void* alpha, struct su2_tensor* t);

void rscale_su2_tensor(const void* alpha, struct su2_tensor* t);

void conjugate_su2_tensor(struct su2_tensor* t);

void su2_tensor_swap_tree_axes(struct su2_tensor* t, const int i_ax_0, const int i_ax_1);

void su2_tensor_add_auxiliary_axis(struct su2_tensor* t, const int i_ax_insert, const bool insert_left);

void su2_tensor_fill_random_normal(const void* alpha, const void* shift, struct rng_state* rng_state, struct su2_tensor* t);


//________________________________________________________________________________________________________________________
//

// transposition

void su2_tensor_transpose(const int* perm, const struct su2_tensor* t, struct su2_tensor* r);

void su2_tensor_transpose_logical(const int* perm, const struct su2_tensor* t, struct su2_tensor* r);


//________________________________________________________________________________________________________________________
//

// F-move

void su2_tensor_fmove(const struct su2_tensor* t, const int i_ax, struct su2_tensor* r);


//________________________________________________________________________________________________________________________
//

// axis reversal

void su2_tensor_reverse_axis_simple(struct su2_tensor* t, const int i_ax);


//________________________________________________________________________________________________________________________
//

// axis fusion and splitting

void su2_tensor_fuse_axes(const struct su2_tensor* t, const int i_ax_0, const int i_ax_1, struct su2_tensor* r);

void su2_tensor_fuse_axes_add_auxiliary(const struct su2_tensor* t, const int i_ax_0, const int i_ax_1, struct su2_tensor* r);

void su2_tensor_split_axis(const struct su2_tensor* t, const int i_ax_split, const int i_ax_add, const bool tree_left_child, const struct su2_irreducible_list outer_irreps[2], const ct_long* dim_degen[2], struct su2_tensor* r);

void su2_tensor_split_axis_remove_auxiliary(const struct su2_tensor* t, const int i_ax_split, const int i_ax_add, const struct su2_irreducible_list outer_irreps[2], const ct_long* dim_degen[2], struct su2_tensor* r);


//________________________________________________________________________________________________________________________
//

// contraction

void su2_tensor_contract_simple(const struct su2_tensor* s, const int* i_ax_s, const struct su2_tensor* t, const int* i_ax_t, const int ndim_mult, struct su2_tensor* r);

void su2_tensor_contract_yoga(const struct su2_tensor* s, const int i_ax_s, const struct su2_tensor* t, const int i_ax_t, struct su2_tensor* r);


//________________________________________________________________________________________________________________________
//

// conversion to a dense tensor

void su2_to_dense_tensor(const struct su2_tensor* s, struct dense_tensor* t);


//________________________________________________________________________________________________________________________
//

// QR and RQ decomposition

int su2_tensor_qr(const struct su2_tensor* a, const enum qr_mode mode, struct su2_tensor* q, struct su2_tensor* r);

int su2_tensor_rq(const struct su2_tensor* a, const enum qr_mode mode, struct su2_tensor* r, struct su2_tensor* q);


//________________________________________________________________________________________________________________________
//

// comparison

bool su2_tensor_allclose(const struct su2_tensor* s, const struct su2_tensor* t, const double tol);
bool dense_su2_tensor_allclose(const struct dense_tensor* s, const struct su2_tensor* t, const double tol);

bool su2_tensor_is_identity(const struct su2_tensor* t, const double tol);

bool su2_tensor_is_isometry(const struct su2_tensor* t, const double tol, const bool transpose);
