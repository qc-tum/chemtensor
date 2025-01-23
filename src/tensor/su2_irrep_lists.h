/// \file su2_irrep_lists.h
/// \brief Irreducible 'j' quantum number lists and configurations for SU(2) symmetric tensors.

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


//________________________________________________________________________________________________________________________
///
/// \brief Charge sectors ('j' quantum number configurations).
///
struct charge_sectors
{
	qnumber* jlists;  //!< matrix of 'j' quantum numbers times 2, of dimension nsec x ndim; must be sorted lexicographically
	long nsec;        //!< number of sectors (configurations)
	int ndim;         //!< number of dimensions ('j' quantum numbers in each configuration)
};

void allocate_charge_sectors(const long nsec, const int ndim, struct charge_sectors* sectors);

void delete_charge_sectors(struct charge_sectors* sectors);

long charge_sector_index(const struct charge_sectors* sectors, const qnumber* jlist);
