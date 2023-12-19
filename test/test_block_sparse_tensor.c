#include "block_sparse_tensor.h"
#include "test_dense_tensor.h"


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
	allocate_dense_tensor(5, dims, &t_dns);
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
	dense_to_block_sparse_tensor(&t_dns, axis_dir, (const qnumber **)qnums, &t);

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
	allocate_dense_tensor(4, dim, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	struct dense_tensor t_tp_dns_ref;
	const long dim_ref[4] = { 4, 7, 11, 5 };
	allocate_dense_tensor(4, dim_ref, &t_tp_dns_ref);
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
	dense_to_block_sparse_tensor(&t_dns, axis_dir, (const qnumber **)qnums, &t);

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


char* test_block_sparse_tensor_flatten_axes()
{
	hid_t file = H5Fopen("../test/data/test_block_sparse_tensor_flatten_axes.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_block_sparse_tensor_flatten_axes failed";
	}

	const int ndim = 5;
	const long dim[5] = { 5, 7, 4, 11, 3 };

	// read dense tensors from disk
	struct dense_tensor t_dns;
	allocate_dense_tensor(5, dim, &t_dns);
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
	dense_to_block_sparse_tensor(&t_dns, axis_dir, (const qnumber **)qnums, &t);

	// flatten axes
	const int i_ax = 1;
	const enum tensor_axis_direction new_axis_dir = TENSOR_AXIS_OUT;
	struct block_sparse_tensor r;
	flatten_block_sparse_tensor_axes(&t, i_ax, new_axis_dir, &r);

	// reshape original dense tensor, as reference
	const long dim_reshape[4] = { 5, 7 * 4, 11, 3 };
	reshape_dense_tensor(ndim - 1, dim_reshape, &t_dns);

	// convert block-sparse back to a dense tensor
	struct dense_tensor r_dns;
	block_sparse_to_dense_tensor(&r, &r_dns);

	// compare
	if (!dense_tensor_allclose(&r_dns, &t_dns, 0.)) {
		return "flattened block-sparse tensor does not match reference";
	}

	// clean up
	delete_dense_tensor(&r_dns);
	delete_block_sparse_tensor(&r);
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
	allocate_dense_tensor(5, sdim, &s_dns);
	if (read_hdf5_dataset(file, "s", H5T_NATIVE_DOUBLE, s_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	struct dense_tensor t_dns;
	const long tdim[6] = { 5, 7, 6, 13, 3, 5 };
	allocate_dense_tensor(6, tdim, &t_dns);
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t_dns.data) < 0) {
		return "reading tensor entries from disk failed";
	}
	struct dense_tensor r_dns_ref;
	const long rdim_ref[5] = { 4, 11, 13, 3, 5 };
	allocate_dense_tensor(5, rdim_ref, &r_dns_ref);
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
	dense_to_block_sparse_tensor(&s_dns, axis_dir, (const qnumber **)qnums, &s);
	// the axis directions of the to-be contracted axes are reversed for the 't' tensor
	enum tensor_axis_direction* axis_dir_t = aligned_alloc(MEM_DATA_ALIGN, t_dns.ndim * sizeof(enum tensor_axis_direction));
	for (int i = 0; i < t_dns.ndim; i++) {
		axis_dir_t[i] = (i < ndim_mult ? -1 : 1) * axis_dir[s_dns.ndim - ndim_mult + i];
	}
	struct block_sparse_tensor t;
	dense_to_block_sparse_tensor(&t_dns, axis_dir_t, (const qnumber **)(qnums + (s_dns.ndim - ndim_mult)), &t);
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
