/// \file sparse_coordinate_tensor.c
/// \brief Sparse coordinate tensor structure.

#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include "sparse_coordinate_tensor.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for a sparse coordinate tensor, including the data entries.
///
void allocate_sparse_coordinate_tensor(const enum numeric_type dtype, const int ndim, const ct_long nnz, const ct_long* dim, struct sparse_coordinate_tensor* t)
{
	t->dtype = dtype;

	assert(ndim >= 0);
	t->ndim = ndim;

	assert(nnz >= 0);
	assert(nnz <= integer_product(dim, ndim));
	t->nnz = nnz;

	// ensure that 'values' and 'coords' are valid pointers even if nnz == 0
	t->values = ct_calloc(lmax(nnz, 1), sizeof_numeric_type(dtype));
	t->coords = ct_calloc(lmax(nnz, 1), sizeof(ct_long));

	if (ndim == 0)  // special case
	{
		t->dim = NULL;
	}
	else
	{
		t->dim = ct_malloc(ndim * sizeof(ct_long));
		memcpy(t->dim, dim, ndim * sizeof(ct_long));
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a sparse coordinate tensor (free memory).
///
void delete_sparse_coordinate_tensor(struct sparse_coordinate_tensor* t)
{
	ct_free(t->coords);
	t->coords = NULL;

	ct_free(t->values);
	t->values = NULL;

	if (t->ndim > 0)
	{
		ct_free(t->dim);
		t->dim = NULL;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Internal consistency check of a sparse coordinate tensor.
///
bool sparse_coordinate_tensor_is_consistent(const struct sparse_coordinate_tensor* t)
{
	if (t->ndim < 0) {
		return false;
	}

	if (t->nnz < 0) {
		return false;
	}

	const ct_long nelem_max = integer_product(t->dim, t->ndim);
	if (nelem_max <= 0) {
		return false;
	}
	if (t->nnz > nelem_max) {
		return false;
	}

	// coordinates must be sorted and unique
	for (ct_long k = 0; k < t->nnz - 1; k++)
	{
		if (t->coords[k] >= t->coords[k + 1]) {
			return false;
		}
	}

	// check coordinate ranges
	for (ct_long k = 0; k < t->nnz; k++)
	{
		if (t->coords[k] < 0 || t->coords[k] >= nelem_max) {
			return false;
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Temporary data structure storing a coordinate and value index, used for transposing a sparse coordinate tensor.
///
struct coord_value_index_tuple
{
	ct_long coord;  //!< coordinate
	ct_long v_idx;  //!< value index
};


//________________________________________________________________________________________________________________________
///
/// \brief Comparison function for sorting.
///
static int compare_coord_value_index_tuples(const void* a, const void* b)
{
	const struct coord_value_index_tuple* x = a;
	const struct coord_value_index_tuple* y = b;

	if (x->coord < y->coord) {
		return -1;
	}
	else if (x->coord == y->coord) {
		return 0;
	}
	else {
		return 1;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Generalized transpose of a tensor 't' such that
/// the i-th axis in the output tensor 'r' is the perm[i]-th axis of the input tensor 't'.
///
/// Memory will be allocated for 'r'.
///
void sparse_coordinate_tensor_transpose(const int* perm, const struct sparse_coordinate_tensor* restrict t, struct sparse_coordinate_tensor* restrict r)
{
	// ensure that 'perm' is a valid permutation
	assert(is_permutation(perm, t->ndim));

	// dimensions of new tensor 'r'
	ct_long* rdim = ct_malloc(t->ndim * sizeof(ct_long));
	for (int i = 0; i < t->ndim; i++) {
		rdim[i] = t->dim[perm[i]];
	}
	// create new tensor 'r'
	allocate_sparse_coordinate_tensor(t->dtype, t->ndim, t->nnz, rdim, r);
	ct_free(rdim);

	if (t->ndim == 0)
	{
		assert(t->nnz <= 1);
		memcpy(r->values, t->values, t->nnz * sizeof_numeric_type(t->dtype));
		return;
	}

	struct coord_value_index_tuple* tuples = ct_malloc(t->nnz * sizeof(struct coord_value_index_tuple));

	ct_long* index_t = ct_malloc(t->ndim * sizeof(ct_long));
	ct_long* index_r = ct_malloc(r->ndim * sizeof(ct_long));
	for (ct_long k = 0; k < t->nnz; k++)
	{
		offset_to_tensor_index(t->ndim, t->dim, t->coords[k], index_t);

		// permute indices
		for (int i = 0; i < t->ndim; i++) {
			index_r[i] = index_t[perm[i]];
		}

		tuples[k].coord = tensor_index_to_offset(r->ndim, r->dim, index_r);
		tuples[k].v_idx = k;
	}
	ct_free(index_r);
	ct_free(index_t);

	// sort by coordinates
	qsort(tuples, t->nnz, sizeof(struct coord_value_index_tuple), compare_coord_value_index_tuples);

	// copy coordinates
	#pragma omp parallel for
	for (ct_long k = 0; k < t->nnz; k++)
	{
		r->coords[k] = tuples[k].coord;
	}
	// copy values
	switch (t->dtype)
	{
		case CT_SINGLE_REAL:
		{
			const float* t_values = t->values;
			float*       r_values = r->values;
			#pragma omp parallel for
			for (ct_long k = 0; k < t->nnz; k++)
			{
				r_values[k] = t_values[tuples[k].v_idx];
			}
			break;
		}
		case CT_DOUBLE_REAL:
		{
			const double* t_values = t->values;
			double*       r_values = r->values;
			#pragma omp parallel for
			for (ct_long k = 0; k < t->nnz; k++)
			{
				r_values[k] = t_values[tuples[k].v_idx];
			}
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			const scomplex* t_values = t->values;
			scomplex*       r_values = r->values;
			#pragma omp parallel for
			for (ct_long k = 0; k < t->nnz; k++)
			{
				r_values[k] = t_values[tuples[k].v_idx];
			}
			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			const dcomplex* t_values = t->values;
			dcomplex*       r_values = r->values;
			#pragma omp parallel for
			for (ct_long k = 0; k < t->nnz; k++)
			{
				r_values[k] = t_values[tuples[k].v_idx];
			}
			break;
		}
		default:
		{
			// unknown data type
			assert(false);
		}
	}

	ct_free(tuples);
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the ranges of same index values, assuming that 'indices' are sorted in ascending order,
/// and return the number of unique indices.
///
static inline ct_long compute_same_index_ranges(const ct_long* restrict indices, const ct_long num, ct_long* restrict ranges)
{
	// count unique indices
	ct_long count_unique = 0;

	for (ct_long i = 0; i < num; )  // 'i' is incremented in the loop body
	{
		ct_long r = 1;
		while (i + r < num && indices[i] == indices[i + r]) {
			r++;
		}
		ranges[i] = r;

		i += r;
		count_unique++;
	}

	return count_unique;
}


//________________________________________________________________________________________________________________________
///
/// \brief Temporary data structure per tensor used for the multiplication of two sparse coordinate tensors,
/// interpreting the tensor as a matrix.
///
struct sparse_coordinate_tensor_dot_data
{
	ct_long* indices_row;         //!< linearized indices of the leading axes fused into the row dimension, array of length 'nnz'
	ct_long* indices_col;         //!< linearized indices of the trailing axes fused into the column dimension, array of length 'nnz'
	ct_long* indices_row_ranges;  //!< ranges of same index values along the row axis, array of length 'nnz'
	ct_long dim_row;              //!< logical row dimension
	ct_long dim_col;              //!< logical column dimension
	ct_long num_unique_row;       //!< number of unique row indices
};


//________________________________________________________________________________________________________________________
///
/// \brief Fill the temporary data structure used for the multiplication of two sparse coordinate tensors.
///
static inline void sparse_coordinate_tensor_preprocess_dot(const struct sparse_coordinate_tensor* t, const int ndim_mult, struct sparse_coordinate_tensor_dot_data* data)
{
	data->dim_row = integer_product(t->dim, (t->ndim - ndim_mult));
	data->dim_col = integer_product(t->dim + (t->ndim - ndim_mult), ndim_mult);
	assert(data->dim_row >= 1);
	assert(data->dim_col >= 1);

	// row and column indices of 't' interpreted as a matrix
	data->indices_row = ct_malloc(t->nnz * sizeof(ct_long));
	data->indices_col = ct_malloc(t->nnz * sizeof(ct_long));
	#pragma omp parallel for
	for (ct_long i = 0; i < t->nnz; i++)
	{
		data->indices_row[i] = t->coords[i] / data->dim_col;
		data->indices_col[i] = t->coords[i] % data->dim_col;
	}

	// index ranges along the open axis of 't'
	data->indices_row_ranges = ct_calloc(t->nnz, sizeof(ct_long));
	data->num_unique_row = compute_same_index_ranges(data->indices_row, t->nnz, data->indices_row_ranges);
	assert(data->num_unique_row <= data->dim_row);
	assert(t->nnz == 0 || data->num_unique_row > 0);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete the temporary data structure used for the multiplication of two sparse coordinate tensors.
///
static inline void delete_sparse_coordinate_tensor_dot_data(struct sparse_coordinate_tensor_dot_data* data)
{
	ct_free(data->indices_row_ranges);
	ct_free(data->indices_col);
	ct_free(data->indices_row);
}


//________________________________________________________________________________________________________________________
///
/// \brief Multiply the trailing 'ndim_mult' axes in 's' by the trailing 'ndim_mult' axes in 't', and store result in 'r'.
/// To select other axes for multiplication, transpose 's' and 't' beforehand.
///
/// Memory will be allocated for 'r'.
///
void sparse_coordinate_tensor_dot(const struct sparse_coordinate_tensor* restrict s, const struct sparse_coordinate_tensor* restrict t, const int ndim_mult, struct sparse_coordinate_tensor* restrict r)
{
	assert(s->dtype == t->dtype);

	// dimension compatibility checks
	assert(ndim_mult >= 1);
	assert(s->ndim >= ndim_mult && t->ndim >= ndim_mult);
	for (int i = 0; i < ndim_mult; i++) {
		assert(s->dim[s->ndim - ndim_mult + i] ==
		       t->dim[t->ndim - ndim_mult + i]);
	}

	struct sparse_coordinate_tensor_dot_data s_data, t_data;
	sparse_coordinate_tensor_preprocess_dot(s, ndim_mult, &s_data);
	sparse_coordinate_tensor_preprocess_dot(t, ndim_mult, &t_data);

	// allocate new sparse coordinate tensor 'r'
	{
		const int ndimr = s->ndim + t->ndim - 2*ndim_mult;

		// logical dimensions of new tensor 'r'
		ct_long* r_dim = ct_malloc(ndimr * sizeof(ct_long));
		for (int i = 0; i < s->ndim - ndim_mult; i++)
		{
			r_dim[i] = s->dim[i];
		}
		for (int i = 0; i < t->ndim - ndim_mult; i++)
		{
			r_dim[s->ndim - ndim_mult + i] = t->dim[i];
		}

		// upper bound on non-zero entries
		allocate_sparse_coordinate_tensor(t->dtype, ndimr, s_data.num_unique_row * t_data.num_unique_row, r_dim, r);

		ct_free(r_dim);
	}

	// arithmetic operations
	ct_long nnz = 0;
	for (ct_long i = 0; i < s->nnz; )  // 'i' is incremented in the loop body
	{
		assert(s_data.indices_row_ranges[i] > 0);
		const ct_long i_next = i + s_data.indices_row_ranges[i];

		for (ct_long j = 0; j < t->nnz; )  // 'j' is incremented in the loop body
		{
			assert(t_data.indices_row_ranges[j] > 0);
			const ct_long j_next = j + t_data.indices_row_ranges[j];

			switch (s->dtype)
			{
				case CT_SINGLE_REAL:
				{
					const float* s_values = s->values;
					const float* t_values = t->values;
					float accum = 0;
					ct_long k = i;
					ct_long l = j;
					while (k < i_next && l < j_next)
					{
						if (s_data.indices_col[k] == t_data.indices_col[l])
						{
							accum += s_values[k] * t_values[l];
							k++;
							l++;
						}
						else if (s_data.indices_col[k] < t_data.indices_col[l])
						{
							k++;
						}
						else
						{
							l++;
						}
					}
					if (accum != 0)
					{
						// add new entry to 'r'
						r->coords[nnz] = s_data.indices_row[i] * t_data.dim_row + t_data.indices_row[j];
						float* r_values = r->values;
						r_values[nnz] = accum;
						nnz++;
					}
					break;
				}
				case CT_DOUBLE_REAL:
				{
					const double* s_values = s->values;
					const double* t_values = t->values;
					double accum = 0;
					ct_long k = i;
					ct_long l = j;
					while (k < i_next && l < j_next)
					{
						if (s_data.indices_col[k] == t_data.indices_col[l])
						{
							accum += s_values[k] * t_values[l];
							k++;
							l++;
						}
						else if (s_data.indices_col[k] < t_data.indices_col[l])
						{
							k++;
						}
						else
						{
							l++;
						}
					}
					if (accum != 0)
					{
						// add new entry to 'r'
						r->coords[nnz] = s_data.indices_row[i] * t_data.dim_row + t_data.indices_row[j];
						double* r_values = r->values;
						r_values[nnz] = accum;
						nnz++;
					}
					break;
				}
				case CT_SINGLE_COMPLEX:
				{
					const scomplex* s_values = s->values;
					const scomplex* t_values = t->values;
					scomplex accum = 0;
					ct_long k = i;
					ct_long l = j;
					while (k < i_next && l < j_next)
					{
						if (s_data.indices_col[k] == t_data.indices_col[l])
						{
							accum += s_values[k] * t_values[l];
							k++;
							l++;
						}
						else if (s_data.indices_col[k] < t_data.indices_col[l])
						{
							k++;
						}
						else
						{
							l++;
						}
					}
					if (accum != 0)
					{
						// add new entry to 'r'
						r->coords[nnz] = s_data.indices_row[i] * t_data.dim_row + t_data.indices_row[j];
						scomplex* r_values = r->values;
						r_values[nnz] = accum;
						nnz++;
					}
					break;
				}
				case CT_DOUBLE_COMPLEX:
				{
					const dcomplex* s_values = s->values;
					const dcomplex* t_values = t->values;
					dcomplex accum = 0;
					ct_long k = i;
					ct_long l = j;
					while (k < i_next && l < j_next)
					{
						if (s_data.indices_col[k] == t_data.indices_col[l])
						{
							accum += s_values[k] * t_values[l];
							k++;
							l++;
						}
						else if (s_data.indices_col[k] < t_data.indices_col[l])
						{
							k++;
						}
						else
						{
							l++;
						}
					}
					if (accum != 0)
					{
						// add new entry to 'r'
						r->coords[nnz] = s_data.indices_row[i] * t_data.dim_row + t_data.indices_row[j];
						dcomplex* r_values = r->values;
						r_values[nnz] = accum;
						nnz++;
					}
					break;
				}
				default:
				{
					// unknown data type
					assert(false);
				}
			}

			j = j_next;
		}

		i = i_next;
	}
	// actual number of non-zero entries
	assert(nnz <= r->nnz);
	r->nnz = nnz;

	delete_sparse_coordinate_tensor_dot_data(&t_data);
	delete_sparse_coordinate_tensor_dot_data(&s_data);
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert a sparse coordinate tensor to an equivalent dense tensor.
///
void sparse_coordinate_to_dense_tensor(const struct sparse_coordinate_tensor* s, struct dense_tensor* t)
{
	// data entries are initialized to zeros
	allocate_dense_tensor(s->dtype, s->ndim, s->dim, t);

	switch (s->dtype)
	{
		case CT_SINGLE_REAL:
		{
			const float* s_values = s->values;
			float* tdata          = t->data;
			#pragma omp parallel for
			for (ct_long j = 0; j < s->nnz; j++)
			{
				tdata[s->coords[j]] = s_values[j];
			}
			break;
		}
		case CT_DOUBLE_REAL:
		{
			const double* s_values = s->values;
			double* tdata          = t->data;
			#pragma omp parallel for
			for (ct_long j = 0; j < s->nnz; j++)
			{
				tdata[s->coords[j]] = s_values[j];
			}
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			const scomplex* s_values = s->values;
			scomplex* tdata          = t->data;
			#pragma omp parallel for
			for (ct_long j = 0; j < s->nnz; j++)
			{
				tdata[s->coords[j]] = s_values[j];
			}
			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			const dcomplex* s_values = s->values;
			dcomplex* tdata          = t->data;
			#pragma omp parallel for
			for (ct_long j = 0; j < s->nnz; j++)
			{
				tdata[s->coords[j]] = s_values[j];
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


//________________________________________________________________________________________________________________________
///
/// \brief Convert a dense to an equivalent sparse coordinate tensor.
///
void dense_to_sparse_coordinate_tensor(const struct dense_tensor* t, struct sparse_coordinate_tensor* s)
{
	const ct_long nelem = dense_tensor_num_elements(t);

	// number of non-zero entries
	const ct_long nnz = dense_tensor_num_nonzero_elements(t);

	allocate_sparse_coordinate_tensor(t->dtype, t->ndim, nnz, t->dim, s);

	ct_long c = 0;  // counter
	switch (t->dtype)
	{
		case CT_SINGLE_REAL:
		{
			const float* tdata = t->data;
			float* s_values    = s->values;
			for (ct_long j = 0; j < nelem; j++)
			{
				if (tdata[j] == 0) {
					continue;
				}
				s_values[c] = tdata[j];
				s->coords[c] = j;
				c++;
			}
			break;
		}
		case CT_DOUBLE_REAL:
		{
			const double* tdata = t->data;
			double* s_values    = s->values;
			for (ct_long j = 0; j < nelem; j++)
			{
				if (tdata[j] == 0) {
					continue;
				}
				s_values[c] = tdata[j];
				s->coords[c] = j;
				c++;
			}
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			const scomplex* tdata = t->data;
			scomplex* s_values    = s->values;
			for (ct_long j = 0; j < nelem; j++)
			{
				if (tdata[j] == 0) {
					continue;
				}
				s_values[c] = tdata[j];
				s->coords[c] = j;
				c++;
			}
			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			const dcomplex* tdata = t->data;
			dcomplex* s_values    = s->values;
			for (ct_long j = 0; j < nelem; j++)
			{
				if (tdata[j] == 0) {
					continue;
				}
				s_values[c] = tdata[j];
				s->coords[c] = j;
				c++;
			}
			break;
		}
		default:
		{
			// unknown data type
			assert(false);
		}
	}

	assert(c == nnz);
}
