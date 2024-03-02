#include <math.h>
#include "block_sparse_tensor.h"
#include "aligned_memory.h"


char* test_block_sparse_tensor_copy()
{
	hid_t file = H5Fopen("../test/data/test_block_sparse_tensor_copy.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_copy failed";
	}

	const int ndim = 4;
	const long dims[4] = { 5, 13, 4, 7 };

	// read dense tensors from disk
	struct dense_tensor t_dns;
	allocate_dense_tensor(DOUBLE_COMPLEX, 4, dims, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction* axis_dir = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = aligned_alloc(MEM_DATA_ALIGN, dims[i] * sizeof(qnumber));
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
		aligned_free(qnums[i]);
	}
	aligned_free(qnums);
	aligned_free(axis_dir);
	delete_block_sparse_tensor(&t2);
	delete_block_sparse_tensor(&t);
	delete_dense_tensor(&t_dns);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_get_block()
{
	hid_t file = H5Fopen("../test/data/test_block_sparse_tensor_get_block.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_get_block failed";
	}

	const int ndim = 5;
	const long dims[5] = { 7, 4, 11, 5, 7 };

	// read dense tensors from disk
	struct dense_tensor t_dns;
	allocate_dense_tensor(DOUBLE_COMPLEX, ndim, dims, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction* axis_dir = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = aligned_alloc(MEM_DATA_ALIGN, dims[i] * sizeof(qnumber));
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
	long* index_block = aligned_calloc(MEM_DATA_ALIGN, t.ndim, sizeof(long));
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

		qnumber* qnums_block = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(qnumber*));
		for (int i = 0; i < ndim; i++)
		{
			qnums_block[i] = t.qnums_blocks[i][index_block[i]];
		}

		struct dense_tensor* b = block_sparse_tensor_get_block(&t, qnums_block);
		// compare pointers
		if (b != b_ref) {
			return "retrieved tensor block based on quantum numbers does not match reference";
		}

		aligned_free(qnums_block);
	}
	aligned_free(index_block);

	// clean up
	for (int i = 0; i < ndim; i++)
	{
		aligned_free(qnums[i]);
	}
	aligned_free(qnums);
	aligned_free(axis_dir);
	delete_block_sparse_tensor(&t);
	delete_dense_tensor(&t_dns);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_norm2()
{
	hid_t file = H5Fopen("../test/data/test_block_sparse_tensor_norm2.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_norm2 failed";
	}

	const int ndim = 4;
	const long dims[4] = { 4, 6, 13, 5 };

	// read dense tensors from disk
	struct dense_tensor t_dns;
	allocate_dense_tensor(DOUBLE_COMPLEX, ndim, dims, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction* axis_dir = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = aligned_alloc(MEM_DATA_ALIGN, dims[i] * sizeof(qnumber));
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
		aligned_free(qnums[i]);
	}
	aligned_free(qnums);
	aligned_free(axis_dir);
	delete_block_sparse_tensor(&t);
	delete_dense_tensor(&t_dns);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_transpose()
{
	hid_t file = H5Fopen("../test/data/test_block_sparse_tensor_transpose.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_transpose failed";
	}

	const int ndim = 4;
	const long dim[4] = { 5, 4, 11, 7 };

	// read dense tensors from disk
	struct dense_tensor t_dns;
	allocate_dense_tensor(DOUBLE_REAL, 4, dim, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	struct dense_tensor t_tp_dns_ref;
	const long dim_ref[4] = { 4, 7, 11, 5 };
	allocate_dense_tensor(DOUBLE_REAL, 4, dim_ref, &t_tp_dns_ref);
	if (read_hdf5_dataset(file, "t_tp", H5T_NATIVE_DOUBLE, t_tp_dns_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction* axis_dir = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = aligned_alloc(MEM_DATA_ALIGN, dim[i] * sizeof(qnumber));
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
		aligned_free(qnums[i]);
	}
	aligned_free(qnums);
	aligned_free(axis_dir);
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
	hid_t file = H5Fopen("../test/data/test_block_sparse_tensor_reshape.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_reshape failed";
	}

	const int ndim = 5;
	const long dim[5] = { 5, 7, 4, 11, 3 };

	// read dense tensors from disk
	struct dense_tensor t_dns;
	allocate_dense_tensor(DOUBLE_COMPLEX, 5, dim, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction* axis_dir = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = aligned_alloc(MEM_DATA_ALIGN, dim[i] * sizeof(qnumber));
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
	flatten_block_sparse_tensor_axes(&t, i_ax, new_axis_dir, &t_flat);

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
	split_block_sparse_tensor_axis(&t_flat, i_ax, t.dim_logical + i_ax, t.axis_dir + i_ax, (const qnumber**)(t.qnums_logical + i_ax), &s);

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
		aligned_free(qnums[i]);
	}
	aligned_free(qnums);
	aligned_free(axis_dir);
	delete_dense_tensor(&t_dns);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_slice()
{
	hid_t file = H5Fopen("../test/data/test_block_sparse_tensor_slice.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_slice failed";
	}

	const int ndim = 4;
	const long dim[4] = { 6, 7, 11, 13 };

	// read dense tensors from disk
	struct dense_tensor t_dns;
	allocate_dense_tensor(SINGLE_REAL, 4, dim, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_FLOAT, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction* axis_dir = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = aligned_alloc(MEM_DATA_ALIGN, dim[i] * sizeof(qnumber));
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
	long *ind = aligned_alloc(MEM_DATA_ALIGN, nind * sizeof(long));
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
	allocate_dense_tensor(SINGLE_REAL, 4, dim_s_ref, &s_ref);
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
	aligned_free(ind);
	delete_block_sparse_tensor(&t);
	for (int i = 0; i < ndim; i++)
	{
		aligned_free(qnums[i]);
	}
	aligned_free(qnums);
	aligned_free(axis_dir);
	delete_dense_tensor(&t_dns);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_multiply_pointwise_vector()
{
	hid_t file = H5Fopen("../test/data/test_block_sparse_tensor_multiply_pointwise_vector.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_multiply_pointwise_vector failed";
	}

	const int ndim = 4;
	const long dim[4] = { 4, 7, 3, 11 };

	// read dense tensor from disk
	struct dense_tensor s_dns;
	allocate_dense_tensor(SINGLE_COMPLEX, ndim, dim, &s_dns);
	if (read_hdf5_dataset(file, "s", H5T_NATIVE_FLOAT, s_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	enum tensor_axis_direction* axis_dir = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = aligned_alloc(MEM_DATA_ALIGN, dim[i] * sizeof(qnumber));
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
		allocate_dense_tensor(SINGLE_REAL, 1, tdim, &t);
		// read values from disk
		char varname[1024];
		sprintf(varname, "t%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_FLOAT, t.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		// multiply tensors pointwise
		struct block_sparse_tensor s_mult_t;
		block_sparse_tensor_multiply_pointwise_vector(&s, &t, i == 0 ? TENSOR_AXIS_ALIGN_LEADING : TENSOR_AXIS_ALIGN_TRAILING, &s_mult_t);

		// convert back to a dense tensor
		struct dense_tensor s_mult_t_dns;
		block_sparse_to_dense_tensor(&s_mult_t, &s_mult_t_dns);

		// reference tensors for checking
		struct dense_tensor s_mult_t_ref;
		allocate_dense_tensor(SINGLE_COMPLEX, ndim, dim, &s_mult_t_ref);
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
		aligned_free(qnums[i]);
	}
	aligned_free(qnums);
	aligned_free(axis_dir);
	delete_dense_tensor(&s_dns);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_dot()
{
	hid_t file = H5Fopen("../test/data/test_block_sparse_tensor_dot.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_dot failed";
	}

	const int ndim = 8;       // overall number of involved dimensions
	const int ndim_mult = 3;  // number of to-be contracted dimensions
	const long dims[8] = { 4, 11, 5, 7, 6, 13, 3, 5 };

	// read dense tensors from disk
	struct dense_tensor s_dns;
	const long sdim[5] = { 4, 11, 5, 7, 6 };
	allocate_dense_tensor(DOUBLE_COMPLEX, 5, sdim, &s_dns);
	if (read_hdf5_dataset(file, "s", H5T_NATIVE_DOUBLE, s_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	struct dense_tensor t_dns;
	const long tdim[6] = { 5, 7, 6, 13, 3, 5 };
	allocate_dense_tensor(DOUBLE_COMPLEX, 6, tdim, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	struct dense_tensor r_dns_ref;
	const long rdim_ref[5] = { 4, 11, 13, 3, 5 };
	allocate_dense_tensor(DOUBLE_COMPLEX, 5, rdim_ref, &r_dns_ref);
	if (read_hdf5_dataset(file, "r", H5T_NATIVE_DOUBLE, r_dns_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	assert(s_dns.ndim + t_dns.ndim - ndim_mult == ndim);

	enum tensor_axis_direction* axis_dir = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(enum tensor_axis_direction));
	if (read_hdf5_attribute(file, "axis_dir", H5T_NATIVE_INT, axis_dir) < 0) {
		return "reading axis directions from disk failed";
	}

	qnumber** qnums = aligned_alloc(MEM_DATA_ALIGN, ndim * sizeof(qnumber*));
	for (int i = 0; i < ndim; i++)
	{
		qnums[i] = aligned_alloc(MEM_DATA_ALIGN, dims[i] * sizeof(qnumber));
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
	enum tensor_axis_direction* axis_dir_t = aligned_alloc(MEM_DATA_ALIGN, t_dns.ndim * sizeof(enum tensor_axis_direction));
	for (int i = 0; i < t_dns.ndim; i++) {
		axis_dir_t[i] = (i < ndim_mult ? -1 : 1) * axis_dir[s_dns.ndim - ndim_mult + i];
	}
	struct block_sparse_tensor t;
	dense_to_block_sparse_tensor(&t_dns, axis_dir_t, (const qnumber**)(qnums + (s_dns.ndim - ndim_mult)), &t);
	aligned_free(axis_dir_t);

	// block-sparse tensor multiplication
	struct block_sparse_tensor r;
	block_sparse_tensor_dot(&s, &t, ndim_mult, &r);
	// convert back to a dense tensor
	struct dense_tensor r_dns;
	block_sparse_to_dense_tensor(&r, &r_dns);

	// compare
	if (!dense_tensor_allclose(&r_dns, &r_dns_ref, 1e-13)) {
		return "dot product of block-sparse tensors does not match reference";
	}

	// equivalent dense tensor multiplication
	struct dense_tensor r_dns_alt;
	dense_tensor_dot(&s_dns, &t_dns, ndim_mult, &r_dns_alt);
	// compare
	if (!dense_tensor_allclose(&r_dns_alt, &r_dns_ref, 1e-13)) {
		return "dot product of dense tensors does not match reference";
	}

	// clean up
	delete_dense_tensor(&r_dns_alt);
	delete_dense_tensor(&r_dns);
	delete_block_sparse_tensor(&r);
	delete_block_sparse_tensor(&s);
	delete_block_sparse_tensor(&t);
	for (int i = 0; i < ndim; i++)
	{
		aligned_free(qnums[i]);
	}
	aligned_free(qnums);
	aligned_free(axis_dir);
	delete_dense_tensor(&r_dns_ref);
	delete_dense_tensor(&s_dns);
	delete_dense_tensor(&t_dns);

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_qr()
{
	hid_t file = H5Fopen("../test/data/test_block_sparse_tensor_qr.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_qr failed";
	}

	// generic version and dummy special case
	for (int c = 0; c < 2; c++)
	{
		// read dense tensor from disk
		struct dense_tensor a_dns;
		const long dim[2] = { 173, 105 };
		allocate_dense_tensor(DOUBLE_COMPLEX, 2, dim, &a_dns);
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
			qnums[i] = aligned_alloc(MEM_DATA_ALIGN, dim[i] * sizeof(qnumber));
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
		block_sparse_tensor_dot(&q, &r, 1, &qr);
		if (!block_sparse_tensor_allclose(&qr, &a, 1e-13)) {
			return "matrix product Q R is not equal to original A matrix";
		}
		delete_block_sparse_tensor(&qr);

		// 'q' must be an isometry
		struct block_sparse_tensor qh;
		const int perm[2] = { 1, 0 };
		conjugate_transpose_block_sparse_tensor(perm, &q, &qh);
		// revert tensor axes directions for multiplication
		qh.axis_dir[0] = -qh.axis_dir[0];
		qh.axis_dir[1] = -qh.axis_dir[1];
		struct block_sparse_tensor qhq;
		block_sparse_tensor_dot(&qh, &q, 1, &qhq);
		if (!block_sparse_tensor_is_identity(&qhq, 1e-13)) {
			return "Q matrix is not an isometry";
		}
		delete_block_sparse_tensor(&qhq);
		delete_block_sparse_tensor(&qh);

		// 'r' only upper triangular after sorting second axis by quantum numbers

		// clean up
		delete_block_sparse_tensor(&r);
		delete_block_sparse_tensor(&q);
		delete_block_sparse_tensor(&a);
		for (int i = 0; i < 2; i++) {
			aligned_free(qnums[i]);
		}
		delete_dense_tensor(&a_dns);
	}

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_rq()
{
	hid_t file = H5Fopen("../test/data/test_block_sparse_tensor_rq.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_rq failed";
	}

	// generic version and dummy special case
	for (int c = 0; c < 2; c++)
	{
		// read dense tensor from disk
		struct dense_tensor a_dns;
		const long dim[2] = { 163, 115 };
		allocate_dense_tensor(DOUBLE_COMPLEX, 2, dim, &a_dns);
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
			qnums[i] = aligned_alloc(MEM_DATA_ALIGN, dim[i] * sizeof(qnumber));
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
		block_sparse_tensor_dot(&r, &q, 1, &rq);
		if (!block_sparse_tensor_allclose(&rq, &a, 1e-13)) {
			return "matrix product R Q is not equal to original A matrix";
		}
		delete_block_sparse_tensor(&rq);

		// 'q' must be an isometry
		struct block_sparse_tensor qh;
		const int perm[2] = { 1, 0 };
		conjugate_transpose_block_sparse_tensor(perm, &q, &qh);
		// revert tensor axes directions for multiplication
		qh.axis_dir[0] = -qh.axis_dir[0];
		qh.axis_dir[1] = -qh.axis_dir[1];
		struct block_sparse_tensor qqh;
		block_sparse_tensor_dot(&q, &qh, 1, &qqh);
		if (!block_sparse_tensor_is_identity(&qqh, 1e-13)) {
			return "Q matrix is not an isometry";
		}
		delete_block_sparse_tensor(&qqh);
		delete_block_sparse_tensor(&qh);

		// 'r' only upper triangular after sorting first axis by quantum numbers

		// clean up
		delete_block_sparse_tensor(&r);
		delete_block_sparse_tensor(&q);
		delete_block_sparse_tensor(&a);
		for (int i = 0; i < 2; i++) {
			aligned_free(qnums[i]);
		}
		delete_dense_tensor(&a_dns);
	}

	H5Fclose(file);

	return 0;
}


char* test_block_sparse_tensor_svd()
{
	hid_t file = H5Fopen("../test/data/test_block_sparse_tensor_svd.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_svd failed";
	}

	// generic version and dummy special case
	for (int c = 0; c < 2; c++)
	{
		// read dense tensor from disk
		struct dense_tensor a_dns;
		const long dim[2] = { 167, 98 };
		allocate_dense_tensor(DOUBLE_COMPLEX, 2, dim, &a_dns);
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
			qnums[i] = aligned_alloc(MEM_DATA_ALIGN, dim[i] * sizeof(qnumber));
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

		if (s.dtype != DOUBLE_REAL) {
			return "expecting double data type for singular values";
		}

		// temporarily convert 's' to a block-sparse diagonal matrix
		const long dim_s[2] = { s.dim[0], s.dim[0] };
		struct dense_tensor s_mat;
		allocate_dense_tensor(DOUBLE_COMPLEX, 2, dim_s, &s_mat);
		const double* s_src = s.data;
		dcomplex* s_dst = s_mat.data;
		for (long j = 0; j < s.dim[0]; j++)
		{
			s_dst[(s.dim[0] + 1) * j] = s_src[j];
		}
		struct block_sparse_tensor s_blk;
		const enum tensor_axis_direction axis_dir_s[2] = { -u.axis_dir[1], -vh.axis_dir[0] };
		const qnumber* qnums_s[2] = { u.qnums_logical[1], vh.qnums_logical[0] };
		dense_to_block_sparse_tensor(&s_mat, axis_dir_s, qnums_s, &s_blk);

		// matrix product 'u s vh' must be equal to 'a'
		struct block_sparse_tensor us;
		block_sparse_tensor_dot(&u, &s_blk, 1, &us);
		struct block_sparse_tensor usvh;
		block_sparse_tensor_dot(&us, &vh, 1, &usvh);
		delete_block_sparse_tensor(&us);
		if (!block_sparse_tensor_allclose(&usvh, &a, 1e-13)) {
			return "matrix product U S V^dag is not equal to original A matrix";
		}
		delete_block_sparse_tensor(&usvh);

		delete_block_sparse_tensor(&s_blk);
		delete_dense_tensor(&s_mat);

		// 'u' must be an isometry
		struct block_sparse_tensor uh;
		const int perm_u[2] = { 1, 0 };
		conjugate_transpose_block_sparse_tensor(perm_u, &u, &uh);
		// revert tensor axes directions for multiplication
		uh.axis_dir[0] = -uh.axis_dir[0];
		uh.axis_dir[1] = -uh.axis_dir[1];
		struct block_sparse_tensor uhu;
		block_sparse_tensor_dot(&uh, &u, 1, &uhu);
		if (!block_sparse_tensor_is_identity(&uhu, 1e-13)) {
			return "U matrix is not an isometry";
		}
		delete_block_sparse_tensor(&uhu);
		delete_block_sparse_tensor(&uh);

		if (c == 0)
		{
			// 'vh' must be an isometry
			struct block_sparse_tensor v;
			const int perm_v[2] = { 1, 0 };
			conjugate_transpose_block_sparse_tensor(perm_v, &vh, &v);
			// revert tensor axes directions for multiplication
			v.axis_dir[0] = -v.axis_dir[0];
			v.axis_dir[1] = -v.axis_dir[1];
			struct block_sparse_tensor vhv;
			block_sparse_tensor_dot(&vh, &v, 1, &vhv);
			if (!block_sparse_tensor_is_identity(&vhv, 1e-13)) {
				return "V matrix is not an isometry";
			}
			delete_block_sparse_tensor(&vhv);
			delete_block_sparse_tensor(&v);
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
			aligned_free(qnums[i]);
		}
		delete_dense_tensor(&a_dns);
	}

	H5Fclose(file);

	return 0;
}
