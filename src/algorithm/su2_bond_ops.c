/// \file su2_bond_ops.c
/// \brief Auxiliary data structures and functions concerning virtual bonds for SU(2) symmetric tensors.

#include "su2_bond_ops.h"
#include "aligned_memory.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Split an SU(2) symmetric matrix by singular value decomposition,
/// and truncate small singular values based on the specified tolerance and maximum bond dimension.
///
int split_su2_matrix_svd(const struct su2_tensor* restrict a,
	const double tol, const bool relative_thresh, const ct_long max_vdim,
	const enum su2_singular_value_distr svd_distr, const bool copy_tree_left,
	struct su2_tensor* restrict a0, struct su2_tensor* restrict a1, struct trunc_info* info)
{
	assert(a->ndim_logical == 2);

	// SVD requires at least one charge sector in 'a'
	struct su2_tensor u, vh;
	struct dense_tensor s;
	int* multiplicities;
	int ret = su2_tensor_svd(a, copy_tree_left, &u, &s, &multiplicities, &vh);
	if (ret < 0) {
		return ret;
	}
	assert( u.charge_sectors.nsec > 0);
	assert(vh.charge_sectors.nsec > 0);

	// determine retained bond indices
	struct index_list retained;
	if (s.dtype == CT_DOUBLE_REAL)
	{
		retained_bond_indices_multiplicities(s.data, multiplicities, s.dim[0], tol, relative_thresh, max_vdim, &retained, info);
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

		retained_bond_indices_multiplicities(sigma, multiplicities, s.dim[0], tol, relative_thresh, max_vdim, &retained, info);

		ct_free(sigma);
	}

	ct_free(multiplicities);

	// select retained singular values
	struct dense_tensor s_ret;
	if (retained.num > 0)
	{
		dense_tensor_slice(&s, 0, retained.ind, retained.num, &s_ret);
	}
	else
	{
		assert(retained.ind == NULL);

		// use degeneracy dimension 1 for the smallest 'j' quantum number along the virtual bond
		retained.ind = ct_calloc(1, sizeof(ct_long));  // set index to zero
		retained.num = 1;

		// dummy singular value vector with a single entry 0
		const ct_long sdim[1] = { 1 };
		allocate_zero_dense_tensor(s.dtype, 1, sdim, &s_ret);
	}
	delete_dense_tensor(&s);

	// slice matrices and multiply retained singular values with left or right isometry
	if (svd_distr == SU2_SVD_DISTR_LEFT)
	{
		su2_tensor_slice_scale(&u, 1, retained.ind, &s_ret, a0);
		delete_su2_tensor(&u);

		su2_tensor_slice(&vh, 0, retained.ind, retained.num, a1);
		delete_su2_tensor(&vh);
	}
	else
	{
		su2_tensor_slice(&u, 1, retained.ind, retained.num, a0);
		delete_su2_tensor(&u);

		su2_tensor_slice_scale(&vh, 0, retained.ind, &s_ret, a1);
		delete_su2_tensor(&vh);
	}

	delete_dense_tensor(&s_ret);
	delete_index_list(&retained);

	return 0;
}
