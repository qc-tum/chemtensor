#include <math.h>
#include "sparse_coordinate_tensor.h"
#include "aligned_memory.h"
#include "rng.h"


//________________________________________________________________________________________________________________________
///
/// \brief Set randomly selected entries of a dense tensor to random normal values.
///
static void dense_tensor_fill_sparse_random_normal(const double fill_fraction, struct rng_state* rng_state, struct dense_tensor* t)
{
	const ct_long nelem = dense_tensor_num_elements(t);

	switch (t->dtype)
	{
		case CT_SINGLE_REAL:
		{
			float* tdata = t->data;
			for (ct_long j = 0; j < nelem; j++) {
				if (randu(rng_state) < fill_fraction) {
					tdata[j] = randnf(rng_state);
				}
			}
			break;
		}
		case CT_DOUBLE_REAL:
		{
			double* tdata = t->data;
			for (ct_long j = 0; j < nelem; j++) {
				if (randu(rng_state) < fill_fraction) {
					tdata[j] = randn(rng_state);
				}
			}
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			scomplex* tdata = t->data;
			for (ct_long j = 0; j < nelem; j++) {
				if (randu(rng_state) < fill_fraction) {
					tdata[j] = crandnf(rng_state);
				}
			}
			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			dcomplex* tdata = t->data;
			for (ct_long j = 0; j < nelem; j++) {
				if (randu(rng_state) < fill_fraction) {
					tdata[j] = crandn(rng_state);
				}
			}
			break;
		}
		default:
		{
			// unknown data type
			assert(false);
		}
	}
}


char* test_sparse_coordinate_tensor_transpose()
{
	struct rng_state rng_state;
	seed_rng_state(51, &rng_state);

	// data types
	for (int dtype = 0; dtype < CT_NUM_NUMERIC_TYPES; dtype++)
	{
		// special case of all zero entries, and generic case
		for (int c = 0; c < 2; c++)
		{
			// special case of zero degree, and generic case
			for (int d = 0; d < 2; d++)
			{
				// create a random dense tensor
				struct dense_tensor t_dns;
				if (d == 0)
				{
					allocate_dense_tensor(dtype, 0, NULL, &t_dns);
				}
				else
				{
					const ct_long dim[7] = { 17, 2, 5, 3, 1, 11, 8 };
					allocate_dense_tensor(dtype, 7, dim, &t_dns);
				}

				if (c == 1) {
					// fill with random entries
					dense_tensor_fill_sparse_random_normal(0.5, &rng_state, &t_dns);
				}

				struct sparse_coordinate_tensor t;
				dense_to_sparse_coordinate_tensor(&t_dns, &t);
				if (!sparse_coordinate_tensor_is_consistent(&t)) {
					return "internal consistency check for sparse coordinate tensor failed";
				}

				// generalized transposition
				struct sparse_coordinate_tensor t_tp;
				const int perm[7] = { 6, 3, 2, 0, 4, 5, 1 };
				transpose_sparse_coordinate_tensor(d == 0 ? NULL : perm, &t, &t_tp);
				if (!sparse_coordinate_tensor_is_consistent(&t_tp)) {
					return "internal consistency check for sparse coordinate tensor failed";
				}

				// reference calculation
				struct dense_tensor t_tp_ref;
				transpose_dense_tensor(d == 0 ? NULL : perm, &t_dns, &t_tp_ref);

				struct dense_tensor t_tp_dns;
				sparse_coordinate_to_dense_tensor(&t_tp, &t_tp_dns);

				// compare
				if (!dense_tensor_allclose(&t_tp_dns, &t_tp_ref, 0.)) {
					return "transposed sparse coordinate tensor does not match reference";
				}

				delete_dense_tensor(&t_tp_dns);
				delete_dense_tensor(&t_tp_ref);
				delete_sparse_coordinate_tensor(&t_tp);
				delete_sparse_coordinate_tensor(&t);
				delete_dense_tensor(&t_dns);
			}
		}
	}

	return 0;
}


char* test_sparse_coordinate_tensor_dot()
{
	struct rng_state rng_state;
	seed_rng_state(52, &rng_state);

	// data types
	for (int dtype = 0; dtype < CT_NUM_NUMERIC_TYPES; dtype++)
	{
		// special case of all zero entries in first tensor, and generic case
		for (int cs = 0; cs < 2; cs++)
		{
			// variants of the degree of first tensor
			for (int ds = 0; ds < 2; ds++)
			{
				struct dense_tensor s_dns;
				if (ds == 0)
				{
					const ct_long dim[2] = { 11, 4 };
					allocate_dense_tensor(dtype, 2, dim, &s_dns);
				}
				else
				{
					const ct_long dim[4] = { 7, 9, 11, 4 };
					allocate_dense_tensor(dtype, 4, dim, &s_dns);
				}

				if (cs == 1) {
					// fill with random entries
					dense_tensor_fill_sparse_random_normal(0.2, &rng_state, &s_dns);
				}

				struct sparse_coordinate_tensor s;
				dense_to_sparse_coordinate_tensor(&s_dns, &s);
				if (!sparse_coordinate_tensor_is_consistent(&s)) {
					return "internal consistency check for sparse coordinate tensor failed";
				}

				// special case of all zero entries in second tensor, and generic case
				for (int ct = 0; ct < 2; ct++)
				{
					// variants of the degree of second tensor
					for (int dt = 0; dt < 2; dt++)
					{
						struct dense_tensor t_dns;
						if (dt == 0)
						{
							const ct_long dim[2] = { 11, 4 };
							allocate_dense_tensor(dtype, 2, dim, &t_dns);
						}
						else
						{
							const ct_long dim[5] = { 17, 1, 5, 11, 4 };
							allocate_dense_tensor(dtype, 5, dim, &t_dns);
						}

						if (ct == 1) {
							// fill with random entries
							dense_tensor_fill_sparse_random_normal(0.3, &rng_state, &t_dns);
						}

						struct sparse_coordinate_tensor t;
						dense_to_sparse_coordinate_tensor(&t_dns, &t);
						if (!sparse_coordinate_tensor_is_consistent(&t)) {
							return "internal consistency check for sparse coordinate tensor failed";
						}

						const int ndim_mult = 2;
						struct sparse_coordinate_tensor r;
						sparse_coordinate_tensor_dot(&s, &t, ndim_mult, &r);
						if (!sparse_coordinate_tensor_is_consistent(&r)) {
							return "internal consistency check for sparse coordinate tensor failed";
						}

						// reference calculation
						struct dense_tensor r_ref;
						dense_tensor_dot(&s_dns, TENSOR_AXIS_RANGE_TRAILING, &t_dns, TENSOR_AXIS_RANGE_TRAILING, ndim_mult, &r_ref);

						struct dense_tensor r_dns;
						sparse_coordinate_to_dense_tensor(&r, &r_dns);

						// compare
						const double tol = (numeric_real_type(dtype) == CT_SINGLE_REAL ? 1e-7 : 1e-13);
						if (!dense_tensor_allclose(&r_dns, &r_ref, tol)) {
							return "dot product of two sparse coordinate tensors does not agree with reference";
						}

						delete_dense_tensor(&r_dns);
						delete_dense_tensor(&r_ref);
						delete_sparse_coordinate_tensor(&r);

						delete_sparse_coordinate_tensor(&t);
						delete_dense_tensor(&t_dns);
					}
				}

				delete_sparse_coordinate_tensor(&s);
				delete_dense_tensor(&s_dns);
			}
		}
	}

	return 0;
}


char* test_sparse_coordinate_to_dense_tensor()
{
	struct rng_state rng_state;
	seed_rng_state(53, &rng_state);

	// special case of all zero entries, and generic case
	for (int c = 0; c < 2; c++)
	{
		// special case of zero degree, and generic case
		for (int d = 0; d < 2; d++)
		{
			// data types
			for (int dtype = 0; dtype < CT_NUM_NUMERIC_TYPES; dtype++)
			{
				struct dense_tensor t;
				if (d == 0)
				{
					allocate_dense_tensor(dtype, 0, NULL, &t);
				}
				else
				{
					const int ndim = 5;
					const ct_long dim[5] = { 4, 7, 1, 13, 5 };
					allocate_dense_tensor(dtype, ndim, dim, &t);
				}

				if (c == 1)
				{
					// fill with random entries
					dense_tensor_fill_sparse_random_normal(0.5, &rng_state, &t);
				}

				struct sparse_coordinate_tensor s;
				dense_to_sparse_coordinate_tensor(&t, &s);

				if (!sparse_coordinate_tensor_is_consistent(&s)) {
					return "internal consistency check for sparse coordinate tensor failed";
				}

				if (c == 0 && s.nnz > 0) {
					return "expecting nnz == 0 for a sparse coordinate tensor converted from a dense tensor with all zero entries";
				}

				// convert back to a dense tensor
				struct dense_tensor r;
				sparse_coordinate_to_dense_tensor(&s, &r);
				// compare with original tensor
				if (!dense_tensor_allclose(&t, &r, 0.)) {
					return "converting a dense tensor to a sparse coordinate tensor and back does not agree with original tensor";
				}
				delete_dense_tensor(&r);

				delete_sparse_coordinate_tensor(&s);
				delete_dense_tensor(&t);
			}
		}
	}

	return 0;
}
