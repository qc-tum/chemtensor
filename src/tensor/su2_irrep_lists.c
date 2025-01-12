/// \file su2_irrep_lists.c
/// \brief Irreducible 'j' quantum number lists and configurations for SU(2) symmetric tensors.

#include <assert.h>
#include "su2_irrep_lists.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Allocate a list of irreducible 'j' quantum numbers.
///
void allocate_su2_irreducible_list(const int num, struct su2_irreducible_list* list)
{
	list->jlist = ct_malloc(num * sizeof(qnumber));
	list->num = num;
}


//________________________________________________________________________________________________________________________
///
/// \brief Copy a list of irreducible 'j' quantum numbers.
///
void copy_su2_irreducible_list(const struct su2_irreducible_list* src, struct su2_irreducible_list* dst)
{
	allocate_su2_irreducible_list(src->num, dst);
	memcpy(dst->jlist, src->jlist, src->num * sizeof(qnumber));
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a list of irreducible 'j' quantum numbers (free memory).
///
void delete_su2_irreducible_list(struct su2_irreducible_list* list)
{
	ct_free(list->jlist);
	list->num = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Whether two irreducible 'j' quantum number lists are logically equal.
///
bool su2_irreducible_list_equal(const struct su2_irreducible_list* s, const struct su2_irreducible_list* t)
{
	if (s->num != t->num) {
		return false;
	}

	for (int k = 0; k < s->num; k++) {
		if (s->jlist[k] != t->jlist[k]) {
			return false;
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Comparison function for lexicographical sorting, assuming that both lists have the same length.
///
int compare_su2_irreducible_lists(const struct su2_irreducible_list* s, const struct su2_irreducible_list* t)
{
	assert(s->num == t->num);
	for (int i = 0; i < s->num; i++)
	{
		if (s->jlist[i] < t->jlist[i]) {
			return -1;
		}
		else if (s->jlist[i] > t->jlist[i]) {
			return 1;
		}
	}

	// lists are equal
	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate a charge sector array.
///
void allocate_charge_sectors(const long nsec, const int ndim, struct charge_sectors* sectors)
{
	sectors->jlists = ct_malloc(nsec * ndim * sizeof(qnumber));
	sectors->nsec = nsec;
	sectors->ndim = ndim;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a charge sector array (free memory).
///
void delete_charge_sectors(struct charge_sectors* sectors)
{
	ct_free(sectors->jlists);
	sectors->nsec = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Find the index of charge sector 'jlist', assuming that the list of charge sectors is sorted.
/// Returns -1 if charge sector cannot be found.
///
long charge_sector_index(const struct charge_sectors* sectors, const qnumber* jlist)
{
	const struct su2_irreducible_list target = {
		.jlist = (qnumber*)jlist,  // cast to avoid compiler warning; we do not modify 'jlist'
		.num   = sectors->ndim,
	};

	// search interval: [lower, upper)
	long lower = 0;
	long upper = sectors->nsec;
	while (true)
	{
		if (lower >= upper) {
			return -1;
		}
		const long i = (lower + upper) / 2;
		const struct su2_irreducible_list current = {
			.jlist = &sectors->jlists[i * sectors->ndim],
			.num   = sectors->ndim,
		};
		const int c = compare_su2_irreducible_lists(&target, &current);
		if (c < 0) {
			// target < current
			upper = i;
		}
		else if (c == 0) {
			// target == current -> found it
			return i;
		}
		else {
			// target > current
			lower = i + 1;
		}
	}
}
