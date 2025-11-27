/// \file su2_irreps.h
/// \brief Irreducible 'j' quantum number configurations for SU(2) symmetric tensors.

#pragma once

#include <stdbool.h>
#include "qnumber.h"


//________________________________________________________________________________________________________________________
///
/// \brief List of irreducible 'j' quantum numbers.
///
struct su2_irreducible_list
{
	qnumber* jlist;  //!< 'j' quantum numbers times 2 (must be unique)
	int num;         //!< number of irreducible SU(2) subspaces (length of jlist)
};

void allocate_su2_irreducible_list(const int num, struct su2_irreducible_list* list);

void copy_su2_irreducible_list(const struct su2_irreducible_list* src, struct su2_irreducible_list* dst);

void delete_su2_irreducible_list(struct su2_irreducible_list* list);

bool su2_irreducible_list_equal(const struct su2_irreducible_list* s, const struct su2_irreducible_list* t);

int compare_su2_irreducible_lists(const struct su2_irreducible_list* s, const struct su2_irreducible_list* t);

void su2_irreps_tensor_product(
	const struct su2_irreducible_list* irreps_a, const ct_long* dim_degen_a,
	const struct su2_irreducible_list* irreps_b, const ct_long* dim_degen_b,
	struct su2_irreducible_list* irreps_prod, ct_long** dim_degen_prod);


//________________________________________________________________________________________________________________________
///
/// \brief Charge sectors ('j' quantum number configurations).
///
struct charge_sectors
{
	qnumber* jlists;  //!< matrix of 'j' quantum numbers times 2, of dimension nsec x ndim; must be sorted lexicographically
	ct_long nsec;     //!< number of sectors (configurations)
	int ndim;         //!< number of dimensions ('j' quantum numbers in each configuration)
};

void allocate_charge_sectors(const ct_long nsec, const int ndim, struct charge_sectors* sectors);

void copy_charge_sectors(const struct charge_sectors* src, struct charge_sectors* dst);

void delete_charge_sectors(struct charge_sectors* sectors);

ct_long charge_sector_index(const struct charge_sectors* sectors, const qnumber* jlist);

bool charge_sectors_equal(const struct charge_sectors* s, const struct charge_sectors* t);


//________________________________________________________________________________________________________________________
///
/// \brief Trie (prefix tree) node for irreducible 'j' quantum number configurations (temporary data structure).
///
struct su2_irrep_trie_node
{
	qnumber* jvals;  //!< irreducible 'j' quantum number values at current level, must be distinct and sorted
	void** cdata;    //!< pointers to child nodes or data values
	int length;      //!< length of 'jvals' and 'cdata' arrays
};

int su2_irrep_trie_num_leaves(const int height, const struct su2_irrep_trie_node* trie);

void** su2_irrep_trie_search_insert(const qnumber* jlist, const int num, struct su2_irrep_trie_node* trie);

void** su2_irrep_trie_enumerate_configurations(const int height, const struct su2_irrep_trie_node* trie, struct charge_sectors* sectors);

void delete_su2_irrep_trie(const int height, struct su2_irrep_trie_node* trie);
