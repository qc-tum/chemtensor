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
	struct val_idx* s_sort = ct_malloc(n * sizeof(struct val_idx));
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
		ct_free(s_sort);

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
	double* accum = ct_malloc(n * sizeof(double));
	for (long i = 0; i < n; i++)
	{
		accum[s_sort[i].i] = s_sort[i].v;
	}

	ct_free(s_sort);

	// indices of accumulated squares larger than tolerance
	// filter out singular values which are (almost) zero to machine precision
	const double tol_mzero = fmax(tol, 1e-28);
	list->ind = ct_malloc(n * sizeof(long));
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
	for (long i = 0; i < list->num; i++)
	{
		retained[i] = sigma[list->ind[i]];
	}
	info->norm_sigma = norm2(CT_DOUBLE_REAL, list->num, retained);

	// normalized retained singular values
	for (long i = 0; i < list->num; i++)
	{
		retained[i] /= info->norm_sigma;
	}

	info->entropy = von_neumann_entropy(retained, list->num);

	ct_free(retained);
}


//________________________________________________________________________________________________________________________
///
/// \brief Split a block-sparse matrix by singular value decomposition,
/// and truncate small singular values based on tolerance and maximum bond dimension.
///
int split_block_sparse_matrix_svd(const struct block_sparse_tensor* restrict a,
	const double tol, const long max_vdim, const bool renormalize, const enum singular_value_distr svd_distr,
	struct block_sparse_tensor* restrict a0, struct block_sparse_tensor* restrict a1, struct trunc_info* info)
{
	assert(a->ndim == 2);

	struct block_sparse_tensor u, vh;
	struct dense_tensor s;
	int ret = block_sparse_tensor_svd(a, &u, &s, &vh);
	if (ret < 0) {
		return ret;
	}

	// determine retained bond indices
	struct index_list retained;
	if (s.dtype == CT_DOUBLE_REAL)
	{
		retained_bond_indices(s.data, s.dim[0], tol, max_vdim, &retained, info);
	}
	else
	{
		assert(s.dtype == CT_SINGLE_REAL);

		// temporarily convert singular values to double format
		const float* sdata = s.data;
		double* sigma = ct_malloc(s.dim[0] * sizeof(double));
		for (long i = 0; i < s.dim[0]; i++) {
			sigma[i] = (double)sdata[i];
		}

		retained_bond_indices(sigma, s.dim[0], tol, max_vdim, &retained, info);

		ct_free(sigma);
	}

	if (retained.num == 0)
	{
		// use dummy virtual bond dimension 1
		const long ind[1] = { 0 };
		block_sparse_tensor_slice(&u, 1, ind, 1, a0);
		delete_block_sparse_tensor(&u);
		// note: 'vh' is not an isometry in the special case of incompatible quantum numbers in 'a'
		block_sparse_tensor_slice(&vh, 0, ind, 1, a1);
		delete_block_sparse_tensor(&vh);

		// dummy singular value vector with a single entry 0
		struct dense_tensor s_zero;
		const long sdim[1] = { 1 };
		allocate_dense_tensor(s.dtype, 1, sdim, &s_zero);
		delete_dense_tensor(&s);

		if (svd_distr == SVD_DISTR_LEFT)
		{
			struct block_sparse_tensor tmp;
			block_sparse_tensor_multiply_pointwise_vector(a0, &s_zero, TENSOR_AXIS_RANGE_TRAILING, &tmp);
			delete_block_sparse_tensor(a0);
			*a0 = tmp;  // copy internal data pointers
		}
		else
		{
			struct block_sparse_tensor tmp;
			block_sparse_tensor_multiply_pointwise_vector(a1, &s_zero, TENSOR_AXIS_RANGE_LEADING, &tmp);
			delete_block_sparse_tensor(a1);
			*a1 = tmp;  // copy internal data pointers
		}

		delete_dense_tensor(&s_zero);

		return 0;
	}

	// select retained singular values and corresponding matrix slices

	block_sparse_tensor_slice(&u, 1, retained.ind, retained.num, a0);
	delete_block_sparse_tensor(&u);

	block_sparse_tensor_slice(&vh, 0, retained.ind, retained.num, a1);
	delete_block_sparse_tensor(&vh);

	struct dense_tensor s_ret;
	dense_tensor_slice(&s, 0, retained.ind, retained.num, &s_ret);
	if (renormalize)
	{
		// norm of all singular values
		const double norm_sigma_all = dense_tensor_norm2(&s);

		// rescale retained singular values
		assert(info->norm_sigma > 0);
		const double scale = norm_sigma_all / info->norm_sigma;
		if (s_ret.dtype == CT_SINGLE_REAL)
		{
			const float scalef = (float)scale;
			scale_dense_tensor(&scalef, &s_ret);
		}
		else
		{
			assert(s_ret.dtype == CT_DOUBLE_REAL);
			scale_dense_tensor(&scale, &s_ret);
		}
	}
	delete_dense_tensor(&s);

	delete_index_list(&retained);

	// multiply retained singular values with left or right isometry
	if (svd_distr == SVD_DISTR_LEFT)
	{
		struct block_sparse_tensor tmp;
		block_sparse_tensor_multiply_pointwise_vector(a0, &s_ret, TENSOR_AXIS_RANGE_TRAILING, &tmp);
		delete_block_sparse_tensor(a0);
		*a0 = tmp;  // copy internal data pointers
	}
	else
	{
		struct block_sparse_tensor tmp;
		block_sparse_tensor_multiply_pointwise_vector(a1, &s_ret, TENSOR_AXIS_RANGE_LEADING, &tmp);
		delete_block_sparse_tensor(a1);
		*a1 = tmp;  // copy internal data pointers
	}

	delete_dense_tensor(&s_ret);

	return 0;
}
