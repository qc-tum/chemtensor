#include <math.h>
#include "block_sparse_tensor.h"
#include "aligned_memory.h"


char* test_block_sparse_tensor_copy()
{
	hid_t file = H5Fopen("../test/tensor/data/test_block_sparse_tensor_copy.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_copy failed";
	}

	const int ndim = 4;
	const long dims[4] = { 5, 13, 4, 7 };

	// read dense tensors from disk
	struct dense_tensor t_dns;
	allocate_dense_tensor(CT_DOUBLE_COMPLEX, 4, dims, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction* axis_dir = ct_malloc(ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = ct_malloc(ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = ct_malloc(dims[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qnums%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qnums[i]) < 0) {
			return "reading quantum numbers from disk failed";
		}
	}

	// convert dense to block-sparse tensor
	struct block_sparse_tensor t;
	dense_to_block_sparse_tensor(&t_dns, axis_dir, (const qnumber**)qnums, &t);

	// copy the block-sparse tensor
	struct block_sparse_tensor t2;
	copy_block_sparse_tensor(&t, &t2);

	// compare
	if (!block_sparse_tensor_allclose(&t2, &t, 0.)) {
		return "copy of a block-sparse tensor does not match reference";
	}

	// clean up
	for (int i = 0; i < ndim; i++)
	{
		ct_free(qnums[i]);
	}
	ct_free(qnums);
	ct_free(axis_dir);
	delete_block_sparse_tensor(&t2);
	delete_block_sparse_tensor(&t);
	delete_dense_tensor(&t_dns);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_get_block()
{
	hid_t file = H5Fopen("../test/tensor/data/test_block_sparse_tensor_get_block.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_get_block failed";
	}

	const int ndim = 5;
	const long dims[5] = { 7, 4, 11, 5, 7 };

	// read dense tensors from disk
	struct dense_tensor t_dns;
	allocate_dense_tensor(CT_DOUBLE_COMPLEX, ndim, dims, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction* axis_dir = ct_malloc(ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = ct_malloc(ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = ct_malloc(dims[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qnums%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qnums[i]) < 0) {
			return "reading quantum numbers from disk failed";
		}
	}

	// convert dense to block-sparse tensor
	struct block_sparse_tensor t;
	dense_to_block_sparse_tensor(&t_dns, axis_dir, (const qnumber**)qnums, &t);

	const long nblocks = integer_product(t.dim_blocks, t.ndim);
	long* index_block = ct_calloc(t.ndim, sizeof(long));
	for (long k = 0; k < nblocks; k++, next_tensor_index(t.ndim, t.dim_blocks, index_block))
	{
		// probe whether quantum numbers in 't' sum to zero
		qnumber qsum = 0;
		for (int i = 0; i < t.ndim; i++)
		{
			qsum += t.axis_dir[i] * t.qnums_blocks[i][index_block[i]];
		}
		if (qsum != 0) {
			continue;
		}

		struct dense_tensor* b_ref = t.blocks[k];
		assert(b_ref != NULL);
		assert(b_ref->ndim == t.ndim);

		qnumber* qnums_block = ct_malloc(ndim * sizeof(qnumber*));
		for (int i = 0; i < ndim; i++)
		{
			qnums_block[i] = t.qnums_blocks[i][index_block[i]];
		}

		struct dense_tensor* b = block_sparse_tensor_get_block(&t, qnums_block);
		// compare pointers
		if (b != b_ref) {
			return "retrieved tensor block based on quantum numbers does not match reference";
		}

		ct_free(qnums_block);
	}
	ct_free(index_block);

	// clean up
	for (int i = 0; i < ndim; i++)
	{
		ct_free(qnums[i]);
	}
	ct_free(qnums);
	ct_free(axis_dir);
	delete_block_sparse_tensor(&t);
	delete_dense_tensor(&t_dns);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_cyclic_partial_trace()
{
	hid_t file = H5Fopen("../test/tensor/data/test_block_sparse_tensor_cyclic_partial_trace.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_cyclic_partial_trace failed";
	}

	const int ndim = 7;
	const int ndim_trace = 2;
	const long dims[7] = { 7, 4, 3, 2, 5, 7, 4 };

	// read dense tensor from disk
	struct dense_tensor t_dns;
	allocate_dense_tensor(CT_SINGLE_COMPLEX, ndim, dims, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_FLOAT, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction* axis_dir = ct_malloc(ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = ct_malloc(ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = ct_malloc(dims[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qnums%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qnums[i]) < 0) {
			return "reading quantum numbers from disk failed";
		}
	}

	// convert dense to block-sparse tensor
	struct block_sparse_tensor t;
	dense_to_block_sparse_tensor(&t_dns, axis_dir, (const qnumber**)qnums, &t);

	// compute cyclic partial trace
	struct block_sparse_tensor t_tr;
	block_sparse_tensor_cyclic_partial_trace(&t, ndim_trace, &t_tr);

	// read reference dense tensor from disk
	struct dense_tensor t_tr_ref_dns;
	allocate_dense_tensor(CT_SINGLE_COMPLEX, ndim - 2*ndim_trace, dims + ndim_trace, &t_tr_ref_dns);
	if (read_hdf5_dataset(file, "t_tr", H5T_NATIVE_FLOAT, t_tr_ref_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// convert dense to block-sparse tensor
	struct block_sparse_tensor t_tr_ref;
	dense_to_block_sparse_tensor(&t_tr_ref_dns, axis_dir + ndim_trace, (const qnumber**)(qnums + ndim_trace), &t_tr_ref);

	// compare
	if (!block_sparse_tensor_allclose(&t_tr, &t_tr_ref, 5e-6)) {
		return "cyclic partial trace tensor does not match reference";
	}

	// clean up
	for (int i = 0; i < ndim; i++)
	{
		ct_free(qnums[i]);
	}
	ct_free(qnums);
	ct_free(axis_dir);
	delete_block_sparse_tensor(&t_tr_ref);
	delete_block_sparse_tensor(&t_tr);
	delete_block_sparse_tensor(&t);
	delete_dense_tensor(&t_tr_ref_dns);
	delete_dense_tensor(&t_dns);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_norm2()
{
	hid_t file = H5Fopen("../test/tensor/data/test_block_sparse_tensor_norm2.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_norm2 failed";
	}

	const int ndim = 4;
	const long dims[4] = { 4, 6, 13, 5 };

	// read dense tensor from disk
	struct dense_tensor t_dns;
	allocate_dense_tensor(CT_DOUBLE_COMPLEX, ndim, dims, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction* axis_dir = ct_malloc(ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = ct_malloc(ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = ct_malloc(dims[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qnums%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qnums[i]) < 0) {
			return "reading quantum numbers from disk failed";
		}
	}

	// convert dense to block-sparse tensor
	struct block_sparse_tensor t;
	dense_to_block_sparse_tensor(&t_dns, axis_dir, (const qnumber**)qnums, &t);

	double nrm = block_sparse_tensor_norm2(&t);

	double nrm_ref;
	if (read_hdf5_dataset(file, "nrm", H5T_NATIVE_DOUBLE, &nrm_ref) < 0) {
		return "reading tensor norm from disk failed";
	}

	// compare
	if (fabs(nrm - nrm_ref) / nrm_ref > 1e-13) {
		return "block-sparse tensor norm does not match reference";
	}

	// clean up
	for (int i = 0; i < ndim; i++)
	{
		ct_free(qnums[i]);
	}
	ct_free(qnums);
	ct_free(axis_dir);
	delete_block_sparse_tensor(&t);
	delete_dense_tensor(&t_dns);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_transpose()
{
	hid_t file = H5Fopen("../test/tensor/data/test_block_sparse_tensor_transpose.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_transpose failed";
	}

	const int ndim = 4;
	const long dim[4] = { 5, 4, 11, 7 };

	// read dense tensors from disk
	struct dense_tensor t_dns;
	allocate_dense_tensor(CT_DOUBLE_REAL, 4, dim, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	struct dense_tensor t_tp_dns_ref;
	const long dim_ref[4] = { 4, 7, 11, 5 };
	allocate_dense_tensor(CT_DOUBLE_REAL, 4, dim_ref, &t_tp_dns_ref);
	if (read_hdf5_dataset(file, "t_tp", H5T_NATIVE_DOUBLE, t_tp_dns_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction* axis_dir = ct_malloc(ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = ct_malloc(ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = ct_malloc(dim[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qnums%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qnums[i]) < 0) {
			return "reading quantum numbers from disk failed";
		}
	}

	// convert dense to block-sparse tensor
	struct block_sparse_tensor t;
	dense_to_block_sparse_tensor(&t_dns, axis_dir, (const qnumber**)qnums, &t);

	// transpose the block-sparse tensor
	const int perm[4] = { 1, 3, 2, 0 };
	struct block_sparse_tensor t_tp;
	transpose_block_sparse_tensor(perm, &t, &t_tp);

	// convert back to a dense tensor
	struct dense_tensor t_tp_dns;
	block_sparse_to_dense_tensor(&t_tp, &t_tp_dns);

	// compare
	if (!dense_tensor_allclose(&t_tp_dns, &t_tp_dns_ref, 0.)) {
		return "transposed block-sparse tensor does not match reference";
	}

	// clean up
	for (int i = 0; i < ndim; i++)
	{
		ct_free(qnums[i]);
	}
	ct_free(qnums);
	ct_free(axis_dir);
	delete_block_sparse_tensor(&t_tp);
	delete_block_sparse_tensor(&t);
	delete_dense_tensor(&t_tp_dns);
	delete_dense_tensor(&t_tp_dns_ref);
	delete_dense_tensor(&t_dns);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_reshape()
{
	hid_t file = H5Fopen("../test/tensor/data/test_block_sparse_tensor_reshape.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_reshape failed";
	}

	const int ndim = 5;
	const long dim[5] = { 5, 7, 4, 11, 3 };

	// read dense tensors from disk
	struct dense_tensor t_dns;
	allocate_dense_tensor(CT_DOUBLE_COMPLEX, 5, dim, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction* axis_dir = ct_malloc(ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = ct_malloc(ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = ct_malloc(dim[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qnums%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qnums[i]) < 0) {
			return "reading quantum numbers from disk failed";
		}
	}

	// convert dense to block-sparse tensor
	struct block_sparse_tensor t;
	dense_to_block_sparse_tensor(&t_dns, axis_dir, (const qnumber**)qnums, &t);

	// flatten axes
	const int i_ax = 1;
	const enum tensor_axis_direction new_axis_dir = TENSOR_AXIS_OUT;
	struct block_sparse_tensor t_flat;
	block_sparse_tensor_flatten_axes(&t, i_ax, new_axis_dir, &t_flat);

	// reshape original dense tensor, as reference
	const long dim_reshape[4] = { 5, 7 * 4, 11, 3 };
	reshape_dense_tensor(ndim - 1, dim_reshape, &t_dns);

	// convert block-sparse back to a dense tensor
	struct dense_tensor t_flat_dns;
	block_sparse_to_dense_tensor(&t_flat, &t_flat_dns);

	// compare
	if (!dense_tensor_allclose(&t_flat_dns, &t_dns, 0.)) {
		return "flattened block-sparse tensor does not match reference";
	}

	// split axis again
	struct block_sparse_tensor s;
	block_sparse_tensor_split_axis(&t_flat, i_ax, t.dim_logical + i_ax, t.axis_dir + i_ax, (const qnumber**)(t.qnums_logical + i_ax), &s);

	// compare
	if (!block_sparse_tensor_allclose(&s, &t, 0.)) {
		return "block-sparse tensor after axes flattening and splitting does not agree with original tensor";
	}

	// convert block-sparse tensor 's' to a dense tensor
	struct dense_tensor s_dns;
	block_sparse_to_dense_tensor(&s, &s_dns);

	// restore shape of original dense tensor, as reference
	reshape_dense_tensor(ndim, dim, &t_dns);

	// compare
	if (!dense_tensor_allclose(&s_dns, &t_dns, 0.)) {
		return "block-sparse tensor with split axes does not match reference";
	}

	// clean up
	delete_dense_tensor(&s_dns);
	delete_dense_tensor(&t_flat_dns);
	delete_block_sparse_tensor(&s);
	delete_block_sparse_tensor(&t_flat);
	delete_block_sparse_tensor(&t);
	for (int i = 0; i < ndim; i++)
	{
		ct_free(qnums[i]);
	}
	ct_free(qnums);
	ct_free(axis_dir);
	delete_dense_tensor(&t_dns);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_slice()
{
	hid_t file = H5Fopen("../test/tensor/data/test_block_sparse_tensor_slice.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_slice failed";
	}

	const int ndim = 4;
	const long dim[4] = { 6, 7, 11, 13 };

	// read dense tensors from disk
	struct dense_tensor t_dns;
	allocate_dense_tensor(CT_SINGLE_REAL, 4, dim, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_FLOAT, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction* axis_dir = ct_malloc(ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = ct_malloc(ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = ct_malloc(dim[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qnums%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qnums[i]) < 0) {
			return "reading quantum numbers from disk failed";
		}
	}

	// convert dense to block-sparse tensor
	struct block_sparse_tensor t;
	dense_to_block_sparse_tensor(&t_dns, axis_dir, (const qnumber**)qnums, &t);

	// slicing indices
	const long nind = 17;
	long *ind = ct_malloc(nind * sizeof(long));
	if (read_hdf5_attribute(file, "ind", H5T_NATIVE_LONG, ind) < 0) {
		return "reading slice indices from disk failed";
	}

	// perform slicing
	struct block_sparse_tensor s;
	block_sparse_tensor_slice(&t, 2, ind, nind, &s);

	// convert block-sparse tensor 's' to a dense tensor
	struct dense_tensor s_dns;
	block_sparse_to_dense_tensor(&s, &s_dns);

	// read reference tensor from disk
	struct dense_tensor s_ref;
	const long dim_s_ref[4] = { 6, 7, nind, 13 };
	allocate_dense_tensor(CT_SINGLE_REAL, 4, dim_s_ref, &s_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "s", H5T_NATIVE_FLOAT, s_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// compare with reference
	if (!dense_tensor_allclose(&s_dns, &s_ref, 0.)) {
		return "block-sparse tensor with split axes does not match reference";
	}

	// clean up
	delete_dense_tensor(&s_ref);
	delete_dense_tensor(&s_dns);
	delete_block_sparse_tensor(&s);
	ct_free(ind);
	delete_block_sparse_tensor(&t);
	for (int i = 0; i < ndim; i++)
	{
		ct_free(qnums[i]);
	}
	ct_free(qnums);
	ct_free(axis_dir);
	delete_dense_tensor(&t_dns);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_multiply_pointwise_vector()
{
	hid_t file = H5Fopen("../test/tensor/data/test_block_sparse_tensor_multiply_pointwise_vector.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_multiply_pointwise_vector failed";
	}

	const int ndim = 4;
	const long dim[4] = { 4, 7, 3, 11 };

	// read dense tensor from disk
	struct dense_tensor s_dns;
	allocate_dense_tensor(CT_SINGLE_COMPLEX, ndim, dim, &s_dns);
	if (read_hdf5_dataset(file, "s", H5T_NATIVE_FLOAT, s_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction* axis_dir = ct_malloc(ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = ct_malloc(ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = ct_malloc(dim[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qnums%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qnums[i]) < 0) {
			return "reading quantum numbers from disk failed";
		}
	}

	// convert dense to block-sparse tensor
	struct block_sparse_tensor s;
	dense_to_block_sparse_tensor(&s_dns, axis_dir, (const qnumber**)qnums, &s);

	for (int i = 0; i < 2; i++)
	{
		// load vector from disk
		struct dense_tensor t;
		const long tdim[1] = { i == 0 ? dim[0] : dim[ndim - 1] };
		allocate_dense_tensor(CT_SINGLE_REAL, 1, tdim, &t);
		// read values from disk
		char varname[1024];
		sprintf(varname, "t%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_FLOAT, t.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		// multiply tensors pointwise
		struct block_sparse_tensor s_mult_t;
		block_sparse_tensor_multiply_pointwise_vector(&s, &t, i == 0 ? TENSOR_AXIS_RANGE_LEADING : TENSOR_AXIS_RANGE_TRAILING, &s_mult_t);

		// convert back to a dense tensor
		struct dense_tensor s_mult_t_dns;
		block_sparse_to_dense_tensor(&s_mult_t, &s_mult_t_dns);

		// reference tensors for checking
		struct dense_tensor s_mult_t_ref;
		allocate_dense_tensor(CT_SINGLE_COMPLEX, ndim, dim, &s_mult_t_ref);
		// read values from disk
		sprintf(varname, "s_mult_t%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_FLOAT, s_mult_t_ref.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		// compare
		if (!dense_tensor_allclose(&s_mult_t_dns, &s_mult_t_ref, 1e-13)) {
			return "pointwise product of tensors does not match reference";
		}

		delete_dense_tensor(&s_mult_t_ref);
		delete_dense_tensor(&s_mult_t_dns);
		delete_block_sparse_tensor(&s_mult_t);
		delete_dense_tensor(&t);
	}

	delete_block_sparse_tensor(&s);
	for (int i = 0; i < ndim; i++)
	{
		ct_free(qnums[i]);
	}
	ct_free(qnums);
	ct_free(axis_dir);
	delete_dense_tensor(&s_dns);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_multiply_axis()
{
	hid_t file = H5Fopen("../test/tensor/data/test_block_sparse_tensor_multiply_axis.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_multiply_axis failed";
	}

	struct block_sparse_tensor s;
	{
		// read dense tensor from disk
		struct dense_tensor s_dns;
		const long s_dim[4] = { 7, 4, 11, 5 };
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 4, s_dim, &s_dns);
		if (read_hdf5_dataset(file, "s", H5T_NATIVE_DOUBLE, s_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		enum tensor_axis_direction* s_axis_dir = ct_malloc(s_dns.ndim * sizeof(enum tensor_axis_direction));
		if (read_hdf5_attribute(file, "s_axis_dir", H5T_NATIVE_INT, s_axis_dir) < 0) {
			return "reading axis directions from disk failed";
		}

		qnumber** s_qnums = ct_malloc(s_dns.ndim * sizeof(qnumber*));
		for (int i = 0; i < s_dns.ndim; i++)
		{
			s_qnums[i] = ct_malloc(s_dns.dim[i] * sizeof(qnumber));
			char varname[1024];
			sprintf(varname, "s_qnums%i", i);
			if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, s_qnums[i]) < 0) {
				return "reading quantum numbers from disk failed";
			}
		}

		// convert dense to block-sparse tensor
		dense_to_block_sparse_tensor(&s_dns, s_axis_dir, (const qnumber**)s_qnums, &s);

		for (int i = 0; i < s_dns.ndim; i++) {
			ct_free(s_qnums[i]);
		}
		ct_free(s_qnums);
		ct_free(s_axis_dir);
		delete_dense_tensor(&s_dns);
	}

	struct block_sparse_tensor t[2];
	for (int i = 0; i < 2; i++)
	{
		char varname[1024];

		struct dense_tensor t_dns;
		const long t_dim[2][3] = { { 11, 6, 9 }, { 8, 3, 11 } };
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 3, t_dim[i], &t_dns);
		sprintf(varname, "t%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, t_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		enum tensor_axis_direction* t_axis_dir = ct_malloc(t_dns.ndim * sizeof(enum tensor_axis_direction));
		sprintf(varname, "t%i_axis_dir", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, t_axis_dir) < 0) {
			return "reading axis directions from disk failed";
		}

		qnumber** t_qnums = ct_malloc(t_dns.ndim * sizeof(qnumber*));
		for (int j = 0; j < t_dns.ndim; j++)
		{
			t_qnums[j] = ct_malloc(t_dns.dim[j] * sizeof(qnumber));
			sprintf(varname, "t%i_qnums%i", i, j);
			if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, t_qnums[j]) < 0) {
				return "reading quantum numbers from disk failed";
			}
		}

		// convert dense to block-sparse tensor
		dense_to_block_sparse_tensor(&t_dns, t_axis_dir, (const qnumber**)t_qnums, &t[i]);

		for (int j = 0; j < t_dns.ndim; j++) {
			ct_free(t_qnums[j]);
		}
		ct_free(t_qnums);
		ct_free(t_axis_dir);
		delete_dense_tensor(&t_dns);
	}

	const int i_ax = 2;

	for (int i = 0; i < 2; i++)
	{
		struct block_sparse_tensor r;
		block_sparse_tensor_multiply_axis(&s, i_ax, &t[i], i == 0 ? TENSOR_AXIS_RANGE_LEADING : TENSOR_AXIS_RANGE_TRAILING, &r);

		// convert back to a dense tensor
		struct dense_tensor r_dns;
		block_sparse_to_dense_tensor(&r, &r_dns);

		// reference tensor
		struct dense_tensor r_dns_ref;
		const long r_ref_dim[2][5] = { { 7, 4, 6, 9, 5 }, { 7, 4, 8, 3, 5 } };
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 5, r_ref_dim[i], &r_dns_ref);
		char varname[1024];
		sprintf(varname, "r%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, r_dns_ref.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		// compare
		if (!dense_tensor_allclose(&r_dns, &r_dns_ref, 1e-13)) {
			return "multiplication along axis for block-sparse tensors does not match reference";
		}

		delete_dense_tensor(&r_dns_ref);
		delete_dense_tensor(&r_dns);
		delete_block_sparse_tensor(&r);
	}

	// clean up
	delete_block_sparse_tensor(&t[1]);
	delete_block_sparse_tensor(&t[0]);
	delete_block_sparse_tensor(&s);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_concatenate()
{
	hid_t file = H5Fopen("../test/tensor/data/test_block_sparse_tensor_concatenate.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_concatenate failed";
	}

	const int num_tensors = 5;

	struct block_sparse_tensor tlist[5];
	const long dims[5][4] = {
		{ 13, 17, 11, 7 },
		{ 13,  4, 11, 7 },
		{ 13, 12, 11, 7 },
		{ 13,  3, 11, 7 },
		{ 13, 19, 11, 7 },
	};
	enum tensor_axis_direction axis_dir[4];
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}
	for (int j = 0; j < num_tensors; j++)
	{
		// read dense tensor from disk
		struct dense_tensor t_dns;
		allocate_dense_tensor(CT_SINGLE_REAL, 4, dims[j], &t_dns);
		char varname[1024];
		sprintf(varname, "t%i", j);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_FLOAT, t_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		qnumber** t_qnums = ct_malloc(t_dns.ndim * sizeof(qnumber*));
		for (int i = 0; i < t_dns.ndim; i++)
		{
			t_qnums[i] = ct_malloc(t_dns.dim[i] * sizeof(qnumber));
			sprintf(varname, "t%i_qnums%i", j, i);
			if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, t_qnums[i]) < 0) {
				return "reading quantum numbers from disk failed";
			}
		}

		// convert dense to block-sparse tensor
		dense_to_block_sparse_tensor(&t_dns, axis_dir, (const qnumber**)t_qnums, &tlist[j]);

		for (int i = 0; i < t_dns.ndim; i++) {
			ct_free(t_qnums[i]);
		}
		ct_free(t_qnums);
		delete_dense_tensor(&t_dns);
	}

	struct block_sparse_tensor r;
	const int i_ax = 1;
	block_sparse_tensor_concatenate(tlist, num_tensors, i_ax, &r);

	// convert back to a dense tensor
	struct dense_tensor r_dns;
	block_sparse_to_dense_tensor(&r, &r_dns);

	// load reference values from disk
	struct dense_tensor r_ref;
	const long refdim[4] = { 13, 55, 11, 7 };
	allocate_dense_tensor(CT_SINGLE_REAL, 4, refdim, &r_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "r", H5T_NATIVE_FLOAT, r_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&r_dns, &r_ref, 0.)) {
		return "concatenated block-sparse tensor does not match reference";
	}

	delete_dense_tensor(&r_ref);
	delete_dense_tensor(&r_dns);
	delete_block_sparse_tensor(&r);
	for (int j = 0; j < num_tensors; j++) {
		delete_block_sparse_tensor(&tlist[j]);
	}

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_block_diag()
{
	hid_t file = H5Fopen("../test/tensor/data/test_block_sparse_tensor_block_diag.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_block_diag failed";
	}

	const int num_tensors = 6;

	struct block_sparse_tensor tlist[6];
	const long dims[6][5] = {
		{ 4,  2,  5,  7,  1 },
		{ 4, 11,  5,  4,  3 },
		{ 4,  3,  5, 10,  5 },
		{ 4,  1,  5,  3,  8 },
		{ 4,  4,  5,  1,  4 },
		{ 4,  5,  5,  2,  7 },
	};
	enum tensor_axis_direction axis_dir[5];
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}
	for (int j = 0; j < num_tensors; j++)
	{
		// read dense tensor from disk
		struct dense_tensor t_dns;
		allocate_dense_tensor(CT_SINGLE_REAL, 5, dims[j], &t_dns);
		char varname[1024];
		sprintf(varname, "t%i", j);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_FLOAT, t_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		qnumber** t_qnums = ct_malloc(t_dns.ndim * sizeof(qnumber*));
		for (int i = 0; i < t_dns.ndim; i++)
		{
			t_qnums[i] = ct_malloc(t_dns.dim[i] * sizeof(qnumber));
			sprintf(varname, "t%i_qnums%i", j, i);
			if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, t_qnums[i]) < 0) {
				return "reading quantum numbers from disk failed";
			}
		}

		// convert dense to block-sparse tensor
		dense_to_block_sparse_tensor(&t_dns, axis_dir, (const qnumber**)t_qnums, &tlist[j]);

		for (int i = 0; i < t_dns.ndim; i++) {
			ct_free(t_qnums[i]);
		}
		ct_free(t_qnums);
		delete_dense_tensor(&t_dns);
	}

	struct block_sparse_tensor r;
	const int i_ax[3] = { 1, 3, 4 };
	block_sparse_tensor_block_diag(tlist, num_tensors, i_ax, 3, &r);

	// convert back to a dense tensor
	struct dense_tensor r_dns;
	block_sparse_to_dense_tensor(&r, &r_dns);

	// load reference values from disk
	struct dense_tensor r_ref;
	const long refdim[5] = { 4, 26, 5, 27, 28 };
	allocate_dense_tensor(CT_SINGLE_REAL, 5, refdim, &r_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "r", H5T_NATIVE_FLOAT, r_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&r_dns, &r_ref, 0.)) {
		return "concatenated block-sparse tensor does not match reference";
	}

	delete_dense_tensor(&r_ref);
	delete_dense_tensor(&r_dns);
	delete_block_sparse_tensor(&r);
	for (int j = 0; j < num_tensors; j++) {
		delete_block_sparse_tensor(&tlist[j]);
	}

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_dot()
{
	hid_t file = H5Fopen("../test/tensor/data/test_block_sparse_tensor_dot.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_dot failed";
	}

	const int ndim = 8;       // overall number of involved dimensions
	const int ndim_mult = 3;  // number of to-be contracted dimensions
	const long dims[8] = { 4, 11, 5, 7, 6, 13, 3, 5 };

	// read dense tensors from disk
	struct dense_tensor s_dns;
	allocate_dense_tensor(CT_DOUBLE_COMPLEX, 5, dims, &s_dns);
	if (read_hdf5_dataset(file, "s", H5T_NATIVE_DOUBLE, s_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	struct dense_tensor t_dns;
	allocate_dense_tensor(CT_DOUBLE_COMPLEX, 6, dims + 2, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	struct dense_tensor r_dns_ref;
	const long rdim_ref[5] = { 4, 11, 13, 3, 5 };
	allocate_dense_tensor(CT_DOUBLE_COMPLEX, 5, rdim_ref, &r_dns_ref);
	if (read_hdf5_dataset(file, "r", H5T_NATIVE_DOUBLE, r_dns_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	assert(s_dns.ndim + t_dns.ndim - ndim_mult == ndim);

	enum tensor_axis_direction* axis_dir = ct_malloc(ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = ct_malloc(ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = ct_malloc(dims[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qnums%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qnums[i]) < 0) {
			return "reading quantum numbers from disk failed";
		}
	}

	// convert dense to block-sparse tensors
	struct block_sparse_tensor s;
	dense_to_block_sparse_tensor(&s_dns, axis_dir, (const qnumber**)qnums, &s);
	// the axis directions of the to-be contracted axes are reversed for the 't' tensor
	enum tensor_axis_direction* axis_dir_t = ct_malloc(t_dns.ndim * sizeof(enum tensor_axis_direction));
	for (int i = 0; i < t_dns.ndim; i++) {
		axis_dir_t[i] = (i < ndim_mult ? -1 : 1) * axis_dir[s_dns.ndim - ndim_mult + i];
	}
	struct block_sparse_tensor t;
	dense_to_block_sparse_tensor(&t_dns, axis_dir_t, (const qnumber**)(qnums + (s_dns.ndim - ndim_mult)), &t);
	ct_free(axis_dir_t);

	for (enum tensor_axis_range axrange_s = 0; axrange_s < TENSOR_AXIS_RANGE_NUM; axrange_s++)
	{
		struct block_sparse_tensor sp;
		if (axrange_s == TENSOR_AXIS_RANGE_TRAILING) {
			copy_block_sparse_tensor(&s, &sp);
		}
		else {
			const int perm[5] = { 2, 3, 4, 0, 1 };
			transpose_block_sparse_tensor(perm, &s, &sp);
		}

		for (enum tensor_axis_range axrange_t = 0; axrange_t < TENSOR_AXIS_RANGE_NUM; axrange_t++)
		{
			struct block_sparse_tensor tp;
			if (axrange_t == TENSOR_AXIS_RANGE_LEADING) {
				copy_block_sparse_tensor(&t, &tp);
			}
			else {
				const int perm[6] = { 3, 4, 5, 0, 1, 2 };
				transpose_block_sparse_tensor(perm, &t, &tp);
			}

			// block-sparse tensor multiplication
			struct block_sparse_tensor r;
			block_sparse_tensor_dot(&sp, axrange_s, &tp, axrange_t, ndim_mult, &r);
			// convert back to a dense tensor
			struct dense_tensor r_dns;
			block_sparse_to_dense_tensor(&r, &r_dns);

			// compare
			if (!dense_tensor_allclose(&r_dns, &r_dns_ref, 1e-13)) {
				return "dot product of block-sparse tensors does not match reference";
			}

			delete_dense_tensor(&r_dns);
			delete_block_sparse_tensor(&r);

			delete_block_sparse_tensor(&tp);
		}

		delete_block_sparse_tensor(&sp);
	}

	// clean up
	delete_block_sparse_tensor(&s);
	delete_block_sparse_tensor(&t);
	for (int i = 0; i < ndim; i++)
	{
		ct_free(qnums[i]);
	}
	ct_free(qnums);
	ct_free(axis_dir);
	delete_dense_tensor(&r_dns_ref);
	delete_dense_tensor(&s_dns);
	delete_dense_tensor(&t_dns);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_qr()
{
	hid_t file = H5Fopen("../test/tensor/data/test_block_sparse_tensor_qr.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_qr failed";
	}

	// generic version and dummy special case
	for (int c = 0; c < 2; c++)
	{
		// read dense tensor from disk
		struct dense_tensor a_dns;
		const long dim[2] = { 173, 105 };
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 2, dim, &a_dns);
		char varname[1024];
		sprintf(varname, "a%i", c);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, a_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		enum tensor_axis_direction axis_dir[2];
		sprintf(varname, "axis_dir%i", c);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, axis_dir) < 0) {
			return "reading axis directions from disk failed";
		}

		qnumber* qnums[2];
		for (int i = 0; i < 2; i++)
		{
			qnums[i] = ct_malloc(dim[i] * sizeof(qnumber));
			char varname[1024];
			sprintf(varname, "qnums%i%i", c, i);
			if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qnums[i]) < 0) {
				return "reading quantum numbers from disk failed";
			}
		}

		// convert dense to block-sparse tensor
		struct block_sparse_tensor a;
		dense_to_block_sparse_tensor(&a_dns, axis_dir, (const qnumber**)qnums, &a);

		// perform QR decomposition
		struct block_sparse_tensor q, r;
		block_sparse_tensor_qr(&a, &q, &r);

		// matrix product 'q r' must be equal to 'a'
		struct block_sparse_tensor qr;
		block_sparse_tensor_dot(&q, TENSOR_AXIS_RANGE_TRAILING, &r, TENSOR_AXIS_RANGE_LEADING, 1, &qr);
		if (!block_sparse_tensor_allclose(&qr, &a, 1e-13)) {
			return "matrix product Q R is not equal to original A matrix";
		}
		delete_block_sparse_tensor(&qr);

		// 'q' must be an isometry
		if (!block_sparse_tensor_is_isometry(&q, 1e-13, false)) {
			return "Q matrix is not an isometry";
		}

		// 'r' only upper triangular after sorting second axis by quantum numbers

		// clean up
		delete_block_sparse_tensor(&r);
		delete_block_sparse_tensor(&q);
		delete_block_sparse_tensor(&a);
		for (int i = 0; i < 2; i++) {
			ct_free(qnums[i]);
		}
		delete_dense_tensor(&a_dns);
	}

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_rq()
{
	hid_t file = H5Fopen("../test/tensor/data/test_block_sparse_tensor_rq.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_rq failed";
	}

	// generic version and dummy special case
	for (int c = 0; c < 2; c++)
	{
		// read dense tensor from disk
		struct dense_tensor a_dns;
		const long dim[2] = { 163, 115 };
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 2, dim, &a_dns);
		char varname[1024];
		sprintf(varname, "a%i", c);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, a_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		enum tensor_axis_direction axis_dir[2];
		sprintf(varname, "axis_dir%i", c);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, axis_dir) < 0) {
			return "reading axis directions from disk failed";
		}

		qnumber* qnums[2];
		for (int i = 0; i < 2; i++)
		{
			qnums[i] = ct_malloc(dim[i] * sizeof(qnumber));
			char varname[1024];
			sprintf(varname, "qnums%i%i", c, i);
			if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qnums[i]) < 0) {
				return "reading quantum numbers from disk failed";
			}
		}

		// convert dense to block-sparse tensor
		struct block_sparse_tensor a;
		dense_to_block_sparse_tensor(&a_dns, axis_dir, (const qnumber**)qnums, &a);

		// perform RQ decomposition
		struct block_sparse_tensor r, q;
		block_sparse_tensor_rq(&a, &r, &q);

		// matrix product 'r q' must be equal to 'a'
		struct block_sparse_tensor rq;
		block_sparse_tensor_dot(&r, TENSOR_AXIS_RANGE_TRAILING, &q, TENSOR_AXIS_RANGE_LEADING, 1, &rq);
		if (!block_sparse_tensor_allclose(&rq, &a, 1e-13)) {
			return "matrix product R Q is not equal to original A matrix";
		}
		delete_block_sparse_tensor(&rq);

		// 'q' must be an isometry
		if (!block_sparse_tensor_is_isometry(&q, 1e-13, true)) {
			return "Q matrix is not an isometry";
		}

		// 'r' only upper triangular after sorting first axis by quantum numbers

		// clean up
		delete_block_sparse_tensor(&r);
		delete_block_sparse_tensor(&q);
		delete_block_sparse_tensor(&a);
		for (int i = 0; i < 2; i++) {
			ct_free(qnums[i]);
		}
		delete_dense_tensor(&a_dns);
	}

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_svd()
{
	hid_t file = H5Fopen("../test/tensor/data/test_block_sparse_tensor_svd.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_svd failed";
	}

	// generic version and dummy special case
	for (int c = 0; c < 2; c++)
	{
		// read dense tensor from disk
		struct dense_tensor a_dns;
		const long dim[2] = { 167, 98 };
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 2, dim, &a_dns);
		char varname[1024];
		sprintf(varname, "a%i", c);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, a_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		enum tensor_axis_direction axis_dir[2];
		sprintf(varname, "axis_dir%i", c);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, axis_dir) < 0) {
			return "reading axis directions from disk failed";
		}

		qnumber* qnums[2];
		for (int i = 0; i < 2; i++)
		{
			qnums[i] = ct_malloc(dim[i] * sizeof(qnumber));
			char varname[1024];
			sprintf(varname, "qnums%i%i", c, i);
			if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qnums[i]) < 0) {
				return "reading quantum numbers from disk failed";
			}
		}

		// convert dense to block-sparse tensor
		struct block_sparse_tensor a;
		dense_to_block_sparse_tensor(&a_dns, axis_dir, (const qnumber**)qnums, &a);

		// perform SVD
		struct block_sparse_tensor u, vh;
		struct dense_tensor s;
		block_sparse_tensor_svd(&a, &u, &s, &vh);

		if (s.dtype != CT_DOUBLE_REAL) {
			return "expecting double data type for singular values";
		}

		// matrix product 'u s vh' must be equal to 'a'
		struct block_sparse_tensor us;
		block_sparse_tensor_multiply_pointwise_vector(&u, &s, TENSOR_AXIS_RANGE_TRAILING, &us);
		struct block_sparse_tensor usvh;
		block_sparse_tensor_dot(&us, TENSOR_AXIS_RANGE_TRAILING, &vh, TENSOR_AXIS_RANGE_LEADING, 1, &usvh);
		delete_block_sparse_tensor(&us);
		if (!block_sparse_tensor_allclose(&usvh, &a, 1e-13)) {
			return "matrix product U S V^dag is not equal to original A matrix";
		}
		delete_block_sparse_tensor(&usvh);

		// 'u' must be an isometry
		if (!block_sparse_tensor_is_isometry(&u, 1e-13, false)) {
			return "U matrix is not an isometry";
		}

		if (c == 0)
		{
			// 'vh' must be an isometry
			if (!block_sparse_tensor_is_isometry(&vh, 1e-13, true)) {
				return "V matrix is not an isometry";
			}
		}
		else
		{
			// 'v' can only be the zero matrix for c == 1 due to quantum number incompatibility
			if (block_sparse_tensor_norm2(&vh) != 0) {
				return "V matrix must be zero for incompatible quantum numbers";
			}
		}

		// clean up
		delete_block_sparse_tensor(&vh);
		delete_dense_tensor(&s);
		delete_block_sparse_tensor(&u);
		delete_block_sparse_tensor(&a);
		for (int i = 0; i < 2; i++) {
			ct_free(qnums[i]);
		}
		delete_dense_tensor(&a_dns);
	}

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_serialize()
{
	hid_t file = H5Fopen("../test/tensor/data/test_block_sparse_tensor_serialize.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_serialize failed";
	}

	const int ndim = 4;
	const long dims[4] = { 11, 3, 5, 8 };

	// read dense tensors from disk
	struct dense_tensor t_dns;
	allocate_dense_tensor(CT_SINGLE_COMPLEX, 4, dims, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_FLOAT, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction* axis_dir = ct_malloc(ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = ct_malloc(ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = ct_malloc(dims[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qnums%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qnums[i]) < 0) {
			return "reading quantum numbers from disk failed";
		}
	}

	// convert dense to block-sparse tensor
	struct block_sparse_tensor t;
	dense_to_block_sparse_tensor(&t_dns, axis_dir, (const qnumber**)qnums, &t);

	// serialize
	const long nelem = block_sparse_tensor_num_elements_blocks(&t);
	scomplex* entries = ct_malloc(nelem * sizeof(scomplex));
	block_sparse_tensor_serialize_entries(&t, entries);

	// create a block-sparse tensor with same dimensions and quantum numbers
	struct block_sparse_tensor s;
	allocate_block_sparse_tensor(CT_SINGLE_COMPLEX, ndim, dims, axis_dir, (const qnumber**)qnums, &s);

	// deserialize
	block_sparse_tensor_deserialize_entries(&s, entries);

	// compare
	if (!block_sparse_tensor_allclose(&s, &t, 0.)) {
		return "block-sparse tensor after serialization and deserialization does not match original tensor";
	}

	// clean up
	for (int i = 0; i < ndim; i++)
	{
		ct_free(qnums[i]);
	}
	ct_free(qnums);
	ct_free(axis_dir);
	ct_free(entries);
	delete_block_sparse_tensor(&s);
	delete_block_sparse_tensor(&t);
	delete_dense_tensor(&t_dns);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_get_entry()
{
	hid_t file = H5Fopen("../test/tensor/data/test_block_sparse_tensor_get_entry.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_get_entry failed";
	}

	const int ndim = 4;
	const long dims[4] = { 7, 6, 13, 4 };

	// read dense tensors from disk
	struct dense_tensor t_dns;
	allocate_dense_tensor(CT_SINGLE_REAL, 4, dims, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_FLOAT, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction* axis_dir = ct_malloc(ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = ct_malloc(ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = ct_malloc(dims[i] * sizeof(qnumber));
		char varname[1024];
		sprintf(varname, "qnums%i", i);
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qnums[i]) < 0) {
			return "reading quantum numbers from disk failed";
		}
	}

	// convert dense to block-sparse tensor
	struct block_sparse_tensor t;
	dense_to_block_sparse_tensor(&t_dns, axis_dir, (const qnumber**)qnums, &t);

	// entry accessor
	struct block_sparse_tensor_entry_accessor acc;
	create_block_sparse_tensor_entry_accessor(&t, &acc);

	// reconstruct dense tensor entry-by-entry
	struct dense_tensor t_reconstr;
	allocate_dense_tensor(t.dtype, t.ndim, t.dim_logical, &t_reconstr);
	long* index = ct_calloc(t.ndim, sizeof(long));
	const long nelem = dense_tensor_num_elements(&t_reconstr);
	float* tdata = t_reconstr.data;
	for (long k = 0; k < nelem; k++, next_tensor_index(t_reconstr.ndim, t_reconstr.dim, index))
	{
		float* pentry = block_sparse_tensor_get_entry(&acc, index);
		if (pentry != NULL) {
			tdata[tensor_index_to_offset(t_reconstr.ndim, t_reconstr.dim, index)] = (*pentry);
		}
	}
	ct_free(index);

	// compare
	if (!dense_tensor_allclose(&t_reconstr, &t_dns, 0.)) {
		return "reconstructing a tensor entry-by-entry from a block-sparse tensor does not match original tensor";
	}

	// clean up
	delete_dense_tensor(&t_reconstr);
	delete_block_sparse_tensor_entry_accessor(&acc);
	delete_block_sparse_tensor(&t);
	for (int i = 0; i < ndim; i++) {
		ct_free(qnums[i]);
	}
	ct_free(qnums);
	ct_free(axis_dir);
	delete_dense_tensor(&t_dns);

	H5Fclose(file);

	return 0;
}
