/// \file bond_ops.c
/// \brief Auxiliary data structures and functions concerning virtual bonds.

#include <math.h>
#include "bond_ops.h"
#include "aligned_memory.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Split a block-sparse matrix by singular value decomposition,
/// and truncate small singular values based on the specified tolerance and maximum bond dimension.
///
int split_block_sparse_matrix_svd(const struct block_sparse_tensor* restrict a,
	const double tol, const bool relative_thresh, const ct_long max_vdim, const bool renormalize, const enum singular_value_distr svd_distr,
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
		retained_bond_indices(s.data, s.dim[0], tol, relative_thresh, max_vdim, &retained, info);
	}
	else
	{
		assert(s.dtype == CT_SINGLE_REAL);

		// temporarily convert singular values to double format
		const float* sdata = s.data;
		double* sigma = ct_malloc(s.dim[0] * sizeof(double));
		for (ct_long i = 0; i < s.dim[0]; i++) {
			sigma[i] = (double)sdata[i];
		}

		retained_bond_indices(sigma, s.dim[0], tol, relative_thresh, max_vdim, &retained, info);

		ct_free(sigma);
	}

	if (retained.num == 0)
	{
		// use dummy virtual bond dimension 1
		const ct_long ind[1] = { 0 };
		block_sparse_tensor_slice(&u, 1, ind, 1, a0);
		delete_block_sparse_tensor(&u);
		// note: 'vh' is not an isometry in the special case of incompatible quantum numbers in 'a'
		block_sparse_tensor_slice(&vh, 0, ind, 1, a1);
		delete_block_sparse_tensor(&vh);

		// dummy singular value vector with a single entry 0
		struct dense_tensor s_zero;
		const ct_long sdim[1] = { 1 };
		allocate_zero_dense_tensor(s.dtype, 1, sdim, &s_zero);
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


//________________________________________________________________________________________________________________________
///
/// \brief Compute the left isometry after splitting a block-sparse matrix by singular value decomposition
/// and truncating small singular values based on the specified tolerance and maximum bond dimension.
///
int split_block_sparse_matrix_svd_isometry(const struct block_sparse_tensor* restrict a, const double tol, const bool relative_thresh, const ct_long max_vdim,
	struct block_sparse_tensor* restrict u, struct trunc_info* info)
{
	assert(a->ndim == 2);

	struct block_sparse_tensor w, vh;
	struct dense_tensor s;
	int ret = block_sparse_tensor_svd(a, &w, &s, &vh);
	delete_block_sparse_tensor(&vh);
	if (ret < 0) {
		return ret;
	}

	// determine retained bond indices
	struct index_list retained;
	if (s.dtype == CT_DOUBLE_REAL)
	{
		retained_bond_indices(s.data, s.dim[0], tol, relative_thresh, max_vdim, &retained, info);
	}
	else
	{
		assert(s.dtype == CT_SINGLE_REAL);

		// temporarily convert singular values to double format
		const float* sdata = s.data;
		double* sigma = ct_malloc(s.dim[0] * sizeof(double));
		for (ct_long i = 0; i < s.dim[0]; i++) {
			sigma[i] = (double)sdata[i];
		}

		retained_bond_indices(sigma, s.dim[0], tol, relative_thresh, max_vdim, &retained, info);

		ct_free(sigma);
	}

	delete_dense_tensor(&s);

	if (retained.num == 0)
	{
		// use dummy virtual bond dimension 1
		const ct_long ind[1] = { 0 };
		block_sparse_tensor_slice(&w, 1, ind, 1, u);
	}
	else
	{
		// select retained singular values and corresponding matrix slices
		block_sparse_tensor_slice(&w, 1, retained.ind, retained.num, u);
	}

	delete_index_list(&retained);
	delete_block_sparse_tensor(&w);

	return 0;
}
