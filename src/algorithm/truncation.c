/// \file truncation.c
/// \brief Utility functions for singular value truncation.

#include <math.h>
#include "truncation.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Compute the von Neumann entropy of singular values 'sigma'.
///
double von_neumann_entropy(const double* sigma, const ct_long n)
{
	double s = 0;

	for (ct_long i = 0; i < n; i++)
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
/// \brief Compute the von Neumann entropy of singular values 'sigma', including multiplicities.
///
double von_neumann_entropy_multiplicities(const double* sigma, const int* multiplicities, const ct_long n)
{
	double s = 0;

	for (ct_long i = 0; i < n; i++)
	{
		if (sigma[i] > 0)
		{
			const double sq = square(sigma[i]);
			s -= (double)multiplicities[i] * (sq * log(sq));
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
	if (list->ind != NULL)
	{
		ct_free(list->ind);
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
	ct_long i;  //!< index
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
/// The singular values need not be sorted at input.
/// The parameter 'relative_thresh' indicates whether the cumulative sums of the squared singular values
/// should be renormalized before applying the truncation threshold 'tol'.
///
void retained_bond_indices(const double* sigma, const ct_long n, const double tol, const bool relative_thresh, const ct_long max_vdim,
	struct index_list* list, struct trunc_info* info)
{
	assert(tol >= 0);

	info->tol_eff = tol;

	// store singular values as value-index pairs and sort them by value
	struct val_idx* s_sort = ct_malloc(n * sizeof(struct val_idx));
	for (ct_long i = 0; i < n; i++)
	{
		s_sort[i].v = sigma[i];
		s_sort[i].i = i;
	}
	qsort(s_sort, n, sizeof(struct val_idx), val_idx_compare_value);

	// square (and normalize) singular values (we sort them first and start with the smallest value to increase accuracy)
	double sqsum = 0;
	for (ct_long i = 0; i < n; i++)
	{
		s_sort[i].v = square(s_sort[i].v);
		sqsum += s_sort[i].v;
	}
	// special case: all singular values zero
	if (sqsum == 0)
	{
		ct_free(s_sort);

		list->ind = NULL;
		list->num = 0;

		info->norm_sigma = 0;
		info->entropy = 0;
		return;
	}
	if (relative_thresh)
	{
		for (ct_long i = 0; i < n; i++)
		{
			s_sort[i].v /= sqsum;
		}
	}

	// accumulate squares
	for (ct_long i = 1; i < n; i++)
	{
		s_sort[i].v += s_sort[i - 1].v;
	}

	if (max_vdim < n)
	{
		// effective tolerance: maximum of specified tolerance and accumulated squared singular values which we cut off
		info->tol_eff = fmax(tol, s_sort[n - max_vdim - 1].v);

		// set accumulated squares which are cut-off by 'max_vdim' to zero
		for (ct_long i = 0; i < n - max_vdim; i++)
		{
			s_sort[i].v = 0;
		}
	}

	// restore original ordering of accumulated squares
	double* accum = ct_malloc(n * sizeof(double));
	for (ct_long i = 0; i < n; i++)
	{
		accum[s_sort[i].i] = s_sort[i].v;
	}

	ct_free(s_sort);

	// indices of accumulated squares larger than tolerance
	list->ind = ct_malloc(n * sizeof(ct_long));
	list->num = 0;
	for (ct_long i = 0; i < n; i++)
	{
		if (accum[i] > tol)
		{
			list->ind[list->num] = i;
			list->num++;
		}
	}
	assert(list->num <= max_vdim);
	ct_free(accum);

	if (list->num == 0)
	{
		// special case: all singular values truncated

		ct_free(list->ind);
		list->ind = NULL;

		info->norm_sigma = 0;
		info->entropy = 0;
		return;
	}

	// record norm and von Neumann entropy of retained singular values
	double* retained = ct_malloc(list->num * sizeof(double));
	for (ct_long i = 0; i < list->num; i++)
	{
		retained[i] = sigma[list->ind[i]];
	}
	info->norm_sigma = norm2(CT_DOUBLE_REAL, list->num, retained);

	// normalized retained singular values
	for (ct_long i = 0; i < list->num; i++)
	{
		retained[i] /= info->norm_sigma;
	}

	info->entropy = von_neumann_entropy(retained, list->num);

	ct_free(retained);
}


//________________________________________________________________________________________________________________________
///
/// \brief Indices of retained singular values with specified multiplicities based on given tolerance 'tol' and length cut-off 'max_vdim'.
///
/// The singular values need not be sorted at input.
/// The parameter 'relative_thresh' indicates whether the cumulative sums of the squared singular values
/// should be renormalized before applying the truncation threshold 'tol'.
/// 'max_vdim' is the logical maximum bond dimension (including multiplicities).
///
void retained_bond_indices_multiplicities(const double* sigma, const int* multiplicities, const ct_long n, const double tol, const bool relative_thresh, const ct_long max_vdim,
	struct index_list* list, struct trunc_info* info)
{
	assert(tol >= 0);

	info->tol_eff = tol;

	// store singular values as value-index pairs and sort them by value
	struct val_idx* s_sort = ct_malloc(n * sizeof(struct val_idx));
	for (ct_long i = 0; i < n; i++)
	{
		s_sort[i].v = sigma[i];
		s_sort[i].i = i;
	}
	qsort(s_sort, n, sizeof(struct val_idx), val_idx_compare_value);

	// square (and normalize) singular values (we sort them first and start with the smallest value to increase accuracy)
	double sqsum = 0;
	for (ct_long i = 0; i < n; i++)
	{
		s_sort[i].v = multiplicities[s_sort[i].i] * square(s_sort[i].v);
		sqsum += s_sort[i].v;
	}
	// special case: all singular values zero
	if (sqsum == 0)
	{
		ct_free(s_sort);

		list->ind = NULL;
		list->num = 0;

		info->norm_sigma = 0;
		info->entropy = 0;
		return;
	}
	if (relative_thresh)
	{
		for (ct_long i = 0; i < n; i++)
		{
			s_sort[i].v /= sqsum;
		}
	}

	// accumulate squares
	for (ct_long i = 1; i < n; i++)
	{
		s_sort[i].v += s_sort[i - 1].v;
	}

	// logical dimension (including multiplicities)
	ct_long n_logical = 0;
	for (ct_long i = 0; i < n; i++)
	{
		assert(multiplicities[i] > 0);
		n_logical += multiplicities[i];
	}

	if (max_vdim < n_logical)
	{
		ct_long max_vdim_bare = 0;
		ct_long m = 0;
		for (ct_long i = 0; i < n; i++)
		{
			m += multiplicities[s_sort[i].i];
			if (n_logical - m <= max_vdim)
			{
				max_vdim_bare = n - i - 1;
				break;
			}
		}
		assert(0 <= max_vdim_bare && max_vdim_bare < n);

		// effective tolerance: maximum of specified tolerance and accumulated squared singular values which we cut off
		info->tol_eff = fmax(tol, s_sort[n - max_vdim_bare - 1].v);

		// set accumulated squares which are cut-off by 'max_vdim_bare' to zero
		for (ct_long i = 0; i < n - max_vdim_bare; i++)
		{
			s_sort[i].v = 0;
		}
	}

	// restore original ordering of accumulated squares
	double* accum = ct_malloc(n * sizeof(double));
	for (ct_long i = 0; i < n; i++)
	{
		accum[s_sort[i].i] = s_sort[i].v;
	}

	ct_free(s_sort);

	// indices of accumulated squares larger than tolerance
	list->ind = ct_malloc(n * sizeof(ct_long));
	list->num = 0;
	for (ct_long i = 0; i < n; i++)
	{
		if (accum[i] > tol)
		{
			list->ind[list->num] = i;
			list->num++;
		}
	}
	ct_free(accum);

	if (list->num == 0)
	{
		// special case: all singular values truncated

		ct_free(list->ind);
		list->ind = NULL;

		info->norm_sigma = 0;
		info->entropy = 0;
		return;
	}

	// record norm and von Neumann entropy of retained singular values
	double* retained_sigma = ct_malloc(list->num * sizeof(double));
	int*    retained_multiplicities = ct_malloc(list->num * sizeof(int));
	info->norm_sigma = 0;
	for (ct_long i = 0; i < list->num; i++)
	{
		retained_sigma[i] = sigma[list->ind[i]];
		retained_multiplicities[i] = multiplicities[list->ind[i]];
		info->norm_sigma += retained_multiplicities[i] * square(retained_sigma[i]);
	}
	info->norm_sigma = sqrt(info->norm_sigma);

	// normalized retained singular values
	for (ct_long i = 0; i < list->num; i++)
	{
		retained_sigma[i] /= info->norm_sigma;
	}

	info->entropy = von_neumann_entropy_multiplicities(retained_sigma, retained_multiplicities, list->num);

	ct_free(retained_multiplicities);
	ct_free(retained_sigma);
}
