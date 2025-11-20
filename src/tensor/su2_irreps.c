/// \file su2_irreps.c
/// \brief Irreducible 'j' quantum number configurations for SU(2) symmetric tensors.

#include <assert.h>
#include "su2_irreps.h"
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
/// \brief Enumerate all admissible irreducible quantum numbers and corresponding degeneracies
/// when computing the tensor product of two irreducible configurations.
///
void su2_irreps_tensor_product(
	const struct su2_irreducible_list* restrict irreps_a, const ct_long* restrict dim_degen_a,
	const struct su2_irreducible_list* restrict irreps_b, const ct_long* restrict dim_degen_b,
	struct su2_irreducible_list* restrict irreps_prod, ct_long** restrict dim_degen_prod)
{
	qnumber j_max_a = 0;
	for (int k = 0; k < irreps_a->num; k++) {
		j_max_a = qmax(j_max_a, irreps_a->jlist[k]);
	}
	qnumber j_max_b = 0;
	for (int k = 0; k < irreps_b->num; k++) {
		j_max_b = qmax(j_max_b, irreps_b->jlist[k]);
	}
	const qnumber j_max_prod = j_max_a + j_max_b;

	// evaluate "degeneracies"
	(*dim_degen_prod) = ct_calloc(j_max_prod + 1, sizeof(ct_long));
	for (int k = 0; k < irreps_a->num; k++)
	{
		const qnumber ja = irreps_a->jlist[k];
		assert(ja >= 0);
		assert(dim_degen_a[ja] > 0);

		for (int l = 0; l < irreps_b->num; l++)
		{
			const qnumber jb = irreps_b->jlist[l];
			assert(jb >= 0);
			assert(dim_degen_b[jb] > 0);

			for (qnumber jp = abs(ja - jb); jp <= ja + jb; jp += 2)
			{
				(*dim_degen_prod)[jp] += dim_degen_a[ja] * dim_degen_b[jb];
			}
		}
	}

	// enumerate irreducible quantum numbers
	irreps_prod->num = 0;
	irreps_prod->jlist = ct_malloc((j_max_prod + 1) * sizeof(qnumber));  // upper bound on required memory
	for (qnumber j = 0; j <= j_max_prod; j++)
	{
		if ((*dim_degen_prod)[j] > 0)
		{
			irreps_prod->jlist[irreps_prod->num++] = j;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate a charge sector array.
///
void allocate_charge_sectors(const ct_long nsec, const int ndim, struct charge_sectors* sectors)
{
	sectors->jlists = ct_malloc(nsec * ndim * sizeof(qnumber));
	sectors->nsec = nsec;
	sectors->ndim = ndim;
}


//________________________________________________________________________________________________________________________
///
/// \brief Copy a charge sector array.
///
void copy_charge_sectors(const struct charge_sectors* src, struct charge_sectors* dst)
{
	allocate_charge_sectors(src->nsec, src->ndim, dst);
	memcpy(dst->jlists, src->jlists, src->nsec * src->ndim * sizeof(qnumber));
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
ct_long charge_sector_index(const struct charge_sectors* sectors, const qnumber* jlist)
{
	const struct su2_irreducible_list target = {
		.jlist = (qnumber*)jlist,  // cast to avoid compiler warning; we do not modify 'jlist'
		.num   = sectors->ndim,
	};

	// search interval: [lower, upper)
	ct_long lower = 0;
	ct_long upper = sectors->nsec;
	while (true)
	{
		if (lower >= upper) {
			return -1;
		}
		const ct_long i = (lower + upper) / 2;
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


//________________________________________________________________________________________________________________________
///
/// \brief Whether two charge sector arrays are logically equal.
///
bool charge_sectors_equal(const struct charge_sectors* restrict s, const struct charge_sectors* restrict t)
{
	if (s->nsec != t->nsec) {
		return false;
	}

	if (s->ndim != t->ndim) {
		return false;
	}

	for (ct_long i = 0; i < s->nsec * s->ndim; i++) {
		if (s->jlists[i] != t->jlists[i]) {
			return false;
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Count the leaves of an irreducible 'j' quantum number search trie.
///
int su2_irrep_trie_num_leaves(const int height, const struct su2_irrep_trie_node* trie)
{
	assert(height >= 1);

	if (height == 1) {
		return trie->length;
	}

	int n = 0;
	for (int i = 0; i < trie->length; i++)
	{
		n += su2_irrep_trie_num_leaves(height - 1, (const struct su2_irrep_trie_node*)trie->cdata[i]);
	}
	return n;
}


//________________________________________________________________________________________________________________________
///
/// \brief Search or insert an irreducible 'j' quantum number configuration into the trie,
/// returning a reference to the corresponding data pointer.
///
void** su2_irrep_trie_search_insert(const qnumber* jlist, const int num, struct su2_irrep_trie_node* trie)
{
	assert(num >= 1);

	int i;
	for (i = 0; i < trie->length; i++)
	{
		// must be sorted
		assert(i == 0 || (trie->jvals[i - 1] < trie->jvals[i]));

		if (trie->jvals[i] == jlist[0])
		{
			if (num == 1) {
				return &trie->cdata[i];
			}
			else {
				return su2_irrep_trie_search_insert(jlist + 1, num - 1, (struct su2_irrep_trie_node*)trie->cdata[i]);
			}
		}
		else if (trie->jvals[i] > jlist[0]) {
			break;
		}
	}
	// 'jlist[0]' not found; i contains the insert location

	qnumber* jvals_new = ct_malloc((trie->length + 1) * sizeof(qnumber));
	void**   cdata_new = ct_malloc((trie->length + 1) * sizeof(void*));
	for (int k = 0; k < i; k++)
	{
		jvals_new[k] = trie->jvals[k];
		cdata_new[k] = trie->cdata[k];
	}
	jvals_new[i] = jlist[0];
	cdata_new[i] = NULL;
	for (int k = i; k < trie->length; k++)
	{
		jvals_new[k + 1] = trie->jvals[k];
		cdata_new[k + 1] = trie->cdata[k];
	}
	ct_free(trie->jvals);
	ct_free(trie->cdata);
	trie->jvals = jvals_new;
	trie->cdata = cdata_new;
	trie->length++;

	if (num == 1)
	{
		return &trie->cdata[i];
	}

	trie->cdata[i] = ct_calloc(1, sizeof(struct su2_irrep_trie_node));
	return su2_irrep_trie_search_insert(jlist + 1, num - 1, (struct su2_irrep_trie_node*)trie->cdata[i]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Enumerate all irreducible 'j' quantum number configurations stored in a sub-trie,
/// and return the next configuration index.
///
static int su2_irrep_subtrie_configurations(const int height, const struct su2_irrep_trie_node* trie, struct charge_sectors* sectors, int sector_index, void** data)
{
	assert(1 <= height && height <= sectors->ndim);
	assert(0 <= sector_index && sector_index < sectors->nsec);
	assert(trie->length >= 1);

	for (int i = 0; i < trie->length; i++)
	{
		qnumber* jlist = &sectors->jlists[sector_index * sectors->ndim];

		if (i > 0) {
			// copy preceeding 'j' quantum numbers
			memcpy(jlist, jlist - sectors->ndim, (sectors->ndim - height) * sizeof(qnumber));
		}
		jlist[sectors->ndim - height] = trie->jvals[i];

		if (height > 1) {
			sector_index = su2_irrep_subtrie_configurations(height - 1, trie->cdata[i], sectors, sector_index, data);
		}
		else
		{
			data[sector_index] = trie->cdata[i];
			sector_index++;
		}
	}

	return sector_index;
}


//________________________________________________________________________________________________________________________
///
/// \brief Enumerate all irreducible 'j' quantum number configurations stored in the trie,
/// and return the corresponding array of data pointers at the leaves.
///
void** su2_irrep_trie_enumerate_configurations(const int height, const struct su2_irrep_trie_node* trie, struct charge_sectors* sectors)
{
	const int nsec = su2_irrep_trie_num_leaves(height, trie);
	allocate_charge_sectors(nsec, height, sectors);

	void** data = ct_calloc(nsec, sizeof(void*));

	#ifndef NDEBUG
	int sector_index =
	#endif
	su2_irrep_subtrie_configurations(height, trie, sectors, 0, data);
	#ifndef NDEBUG
	assert(sector_index == nsec);
	#endif

	return data;
}


//________________________________________________________________________________________________________________________
///
/// \brief Recursively delete all search trie nodes (free memory).
///
/// This function does not free the data pointers at the leaves.
///
void delete_su2_irrep_trie(const int height, struct su2_irrep_trie_node* trie)
{
	assert(height >= 1);

	if (height > 1)
	{
		for (int i = 0; i < trie->length; i++)
		{
			delete_su2_irrep_trie(height - 1, (struct su2_irrep_trie_node*)trie->cdata[i]);
			ct_free(trie->cdata[i]);
		}
	}

	if (trie->length > 0)
	{
		ct_free(trie->jvals);
		ct_free(trie->cdata);
	}
}
