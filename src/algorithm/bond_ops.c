/// \file bond_ops.c
/// \brief Auxiliary data structures and functions concerning virtual bonds.

#include <math.h>
#include "bond_ops.h"
#include "aligned_memory.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Compute the von Neumann entropy of singular values 'sigma'.
///
double von_neumann_entropy(const double* sigma, const long n)
{
	double s = 0;

	for (long i = 0; i < n; i++)
	{
		if (sigma[i] > 0)
		{
			const double sq = square(sigma[i]);
			s -= sq * log(sq);
		}
	}

	return s;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an index list (free memory).
///
void delete_index_list(struct index_list* list)
{
	if (list->ind != NULL) {
		aligned_free(list->ind);
		list->ind = NULL;
	}

	list->num = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Temporary value-index type.
///
struct val_idx
{
	double v;   //!< value
	long i;     //!< index
};


//________________________________________________________________________________________________________________________
///
/// \brief Compare value-index pairs by value (for 'qsort').
///
static int val_idx_compare_value(const void* p1, const void* p2)
{
	const struct val_idx* x = (const struct val_idx*)p1;
	const struct val_idx* y = (const struct val_idx*)p2;

	if (x->v < y->v)
	{
		return -1;
	}
	else if (y->v < x->v)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Indices of retained singular values based on given tolerance 'tol' and length cut-off 'max_vdim'.
///
/// Singular values need not be sorted at input.
///
void retained_bond_indices(const double* sigma, const long n, const double tol, const long max_vdim,
	struct index_list* list, struct trunc_info* info)
{
	assert(tol >= 0);

	info->tol_eff = tol;

	// store singular values as value-index pairs and sort them by value
	struct val_idx* s_sort = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(struct val_idx));
	for (long i = 0; i < n; i++)
	{
		s_sort[i].v = sigma[i];
		s_sort[i].i = i;
	}
	qsort(s_sort, n, sizeof(struct val_idx), val_idx_compare_value);

	// square and normalize singular values (we sort them first and start with the smallest value to increase accuracy)
	double sqsum = 0;
	for (long i = 0; i < n; i++)
	{
		s_sort[i].v = square(s_sort[i].v);
		sqsum += s_sort[i].v;
	}
	// special case: all singular values zero
	if (sqsum == 0)
	{
		aligned_free(s_sort);

		list->ind = NULL;
		list->num = 0;

		info->norm_sigma = 0;
		info->entropy = 0;
		return;
	}
	for (long i = 0; i < n; i++)
	{
		s_sort[i].v /= sqsum;
	}

	// accumulate squares
	for (long i = 1; i < n; i++)
	{
		s_sort[i].v += s_sort[i - 1].v;
	}

	if (max_vdim < n)
	{
		// effective tolerance: maximum of specified tolerance and accumulated squared singular values which we cut off
		info->tol_eff = fmax(tol, s_sort[n - max_vdim - 1].v);

		// set accumulated squares which are cut-off by 'max_vdim' to zero
		for (long i = 0; i < n - max_vdim; i++)
		{
			s_sort[i].v = 0;
		}
	}

	// restore original ordering of accumulated squares
	double* accum = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(double));
	for (long i = 0; i < n; i++)
	{
		accum[s_sort[i].i] = s_sort[i].v;
	}

	aligned_free(s_sort);

	// indices of accumulated squares larger than tolerance
	// filter out singular values which are (almost) zero to machine precision
	const double tol_mzero = fmax(tol, 1e-28);
	list->ind = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(long));
	list->num = 0;
	for (long i = 0; i < n; i++)
	{
		if (accum[i] > tol_mzero)
		{
			list->ind[list->num] = i;
			list->num++;
		}
	}
	assert(list->num <= max_vdim);
	aligned_free(accum);

	if (list->num == 0)
	{
		// special case: all singular values truncated

		aligned_free(list->ind);
		list->ind = NULL;

		info->norm_sigma = 0;
		info->entropy = 0;
		return;
	}

	// record norm and von Neumann entropy of retained singular values
	double* retained = aligned_alloc(MEM_DATA_ALIGN, list->num * sizeof(double));
	for (long i = 0; i < list->num; i++)
	{
		retained[i] = sigma[list->ind[i]];
	}
	info->norm_sigma = norm2(DOUBLE_REAL, list->num, retained);

	// normalized retained singular values
	for (long i = 0; i < list->num; i++)
	{
		retained[i] /= info->norm_sigma;
	}

	info->entropy = von_neumann_entropy(retained, list->num);

	aligned_free(retained);
}
