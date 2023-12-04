#include <math.h>
#include <memory.h>
#include <stdbool.h>
#include "dense_tensor.h"


//________________________________________________________________________________________________________________________
///
/// \brief Test whether two tensor agree elementwise within tolerance 'tol'.
///
static bool dense_tensor_allclose(const struct dense_tensor* s, const struct dense_tensor* t, double tol)
{
	// compare degrees
	if (s->ndim != t->ndim) {
		return false;
	}

	// compare dimensions
	for (int i = 0; i < s->ndim; i++)
	{
		if (s->dim[i] != t->dim[i]) {
			return false;
		}
	}

	// compare entries
	const long nelem = dense_tensor_num_elements(s);
	for (long j = 0; j < nelem; j++)
	{
		if (cabs(s->data[j] - t->data[j]) > tol) {
			return false;
		}
	}

	return true;
}


char* test_dense_tensor_trace()
{
	hid_t file = H5Fopen("../test/data/test_dense_tensor_trace.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_dense_tensor_trace failed";
	}

	struct dense_tensor t;
	const long tdim[3] = { 5, 5, 5 };
	allocate_dense_tensor(3, tdim, &t);
	// read values from disk
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	numeric tr = dense_tensor_trace(&t);

	// reference value for checking
	numeric tr_ref;
	if (read_hdf5_dataset(file, "tr", H5T_NATIVE_DOUBLE, &tr_ref) < 0) {
		return "reading trace value from disk failed";
	}

	// compare
	if (cabs(tr - tr_ref) > 1e-13) {
		return "tensor trace does not match reference";
	}

	// clean up
	delete_dense_tensor(&t);

	H5Fclose(file);

	return 0;
}


char* test_dense_tensor_transpose()
{
	hid_t file = H5Fopen("../test/data/test_dense_tensor_transpose.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_dense_tensor_transpose failed";
	}

	// create tensor 't'
	struct dense_tensor t;
	const long dim[4] = { 4, 5, 6, 7 };
	allocate_dense_tensor(4, dim,  &t);
	// read values from disk
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// generalized transposition
	const int perm[4] = { 1, 3, 2, 0 };
	struct dense_tensor t_tp;
	transpose_dense_tensor(perm, &t, &t_tp);

	// reference tensor
	const long refdim[4] = { 5, 7, 6, 4 };
	struct dense_tensor t_tp_ref;
	allocate_dense_tensor(4, refdim, &t_tp_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "t_tp", H5T_NATIVE_DOUBLE, t_tp_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&t_tp, &t_tp_ref, 1e-15)) {
		return "transposed tensor does not match reference";
	}

	// clean up
	delete_dense_tensor(&t_tp_ref);
	delete_dense_tensor(&t_tp);
	delete_dense_tensor(&t);

	H5Fclose(file);

	return 0;
}


char* test_dense_tensor_dot()
{
	hid_t file = H5Fopen("../test/data/test_dense_tensor_dot.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_dense_tensor_dot failed";
	}

	// create tensor 't'
	struct dense_tensor t;
	const long dim[4] = { 2, 3, 4, 5 };
	allocate_dense_tensor(4, dim,  &t);
	// read values from disk
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// general dot product
	{
		// create another tensor 's'
		struct dense_tensor s;
		const long sdim[4] = { 4, 5, 7, 6 };
		allocate_dense_tensor(4, sdim, &s);
		// read values from disk
		if (read_hdf5_dataset(file, "s", H5T_NATIVE_DOUBLE, s.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		// multiply tensors and store result in 'r'
		struct dense_tensor t_dot_s;
		dense_tensor_dot(&t, &s, 2, &t_dot_s);

		// reference tensor for checking
		const long refdim[4] = { 2, 3, 7, 6 };
		struct dense_tensor t_dot_s_ref;
		allocate_dense_tensor(4, refdim, &t_dot_s_ref);
		// read values from disk
		if (read_hdf5_dataset(file, "t_dot_s", H5T_NATIVE_DOUBLE, t_dot_s_ref.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		// compare
		if (!dense_tensor_allclose(&t_dot_s, &t_dot_s_ref, 1e-13)) {
			return "dot product of tensors does not match reference";
		}

		// clean up
		delete_dense_tensor(&t_dot_s_ref);
		delete_dense_tensor(&t_dot_s);
		delete_dense_tensor(&s);
	}

	// matrix-vector multiplication
	{
		// create another tensor 'p'
		struct dense_tensor p;
		const long pdim[1] = { 5 };
		allocate_dense_tensor(1, pdim, &p);
		// read values from disk
		if (read_hdf5_dataset(file, "p", H5T_NATIVE_DOUBLE, p.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		// multiply tensors
		struct dense_tensor t_dot_p;
		dense_tensor_dot(&t, &p, 1, &t_dot_p);

		// reference tensor for checking
		const long refdim[3] = { 2, 3, 4 };
		struct dense_tensor t_dot_p_ref;
		allocate_dense_tensor(3, refdim, &t_dot_p_ref);
		// read values from disk
		if (read_hdf5_dataset(file, "t_dot_p", H5T_NATIVE_DOUBLE, t_dot_p_ref.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		// compare
		if (!dense_tensor_allclose(&t_dot_p, &t_dot_p_ref, 1e-13)) {
			return "dot product of tensors does not match reference";
		}

		// clean up
		delete_dense_tensor(&t_dot_p_ref);
		delete_dense_tensor(&t_dot_p);
		delete_dense_tensor(&p);
	}

	// vector-matrix multiplication
	{
		// create another tensor 'q'
		struct dense_tensor q;
		const long qdim[1] = { 2 };
		allocate_dense_tensor(1, qdim, &q);
		// read values from disk
		if (read_hdf5_dataset(file, "q", H5T_NATIVE_DOUBLE, q.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		// multiply tensors
		struct dense_tensor q_dot_t;
		dense_tensor_dot(&q, &t, 1, &q_dot_t);

		// reference tensor for checking
		const long refdim[3] = { 3, 4, 5 };
		struct dense_tensor q_dot_t_ref;
		allocate_dense_tensor(3, refdim, &q_dot_t_ref);
		// read values from disk
		if (read_hdf5_dataset(file, "q_dot_t", H5T_NATIVE_DOUBLE, q_dot_t_ref.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		// compare
		if (!dense_tensor_allclose(&q_dot_t, &q_dot_t_ref, 1e-13)) {
			return "dot product of tensors does not match reference";
		}

		// clean up
		delete_dense_tensor(&q_dot_t_ref);
		delete_dense_tensor(&q_dot_t);
		delete_dense_tensor(&q);
	}

	// vector-vector multiplication (i.e., inner product)
	{
		const long nelem = dense_tensor_num_elements(&t);

		// interpret as vector
		struct dense_tensor t1;
		copy_dense_tensor(&t, &t1);
		reshape_dense_tensor(1, &nelem, &t1);

		// create another tensor 'v'
		struct dense_tensor v;
		allocate_dense_tensor(1, &nelem, &v);
		// read values from disk
		if (read_hdf5_dataset(file, "v", H5T_NATIVE_DOUBLE, v.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		// multiply tensors
		struct dense_tensor t_dot_v;
		dense_tensor_dot(&t1, &v, 1, &t_dot_v);

		// reference tensor for checking
		struct dense_tensor t_dot_v_ref;
		// formally a zero-dimensional tensor
		allocate_dense_tensor(0, NULL, &t_dot_v_ref);
		// read values from disk
		if (read_hdf5_dataset(file, "t_dot_v", H5T_NATIVE_DOUBLE, t_dot_v_ref.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		// compare
		if (!dense_tensor_allclose(&t_dot_v, &t_dot_v_ref, 1e-13)) {
			return "dot product of tensors does not match reference";
		}

		// clean up
		delete_dense_tensor(&t_dot_v_ref);
		delete_dense_tensor(&t_dot_v);
		delete_dense_tensor(&v);
		delete_dense_tensor(&t1);
	}

	// clean up
	delete_dense_tensor(&t);

	H5Fclose(file);

	return 0;
}


char* test_dense_tensor_kronecker_product()
{
	hid_t file = H5Fopen("../test/data/test_dense_tensor_kronecker_product.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_dense_tensor_kronecker_product failed";
	}

	// create tensor 's'
	struct dense_tensor s;
	const long sdim[4] = { 6, 5, 7, 2 };
	allocate_dense_tensor(4, sdim, &s);
	// read values from disk
	if (read_hdf5_dataset(file, "s", H5T_NATIVE_DOUBLE, s.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// create tensor 't'
	struct dense_tensor t;
	const long dim[4] = { 3, 11, 2, 5 };
	allocate_dense_tensor(4, dim,  &t);
	// read values from disk
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	struct dense_tensor r;
	dense_tensor_kronecker_product(&s, &t, &r);

	// load reference values from disk
	struct dense_tensor r_ref;
	const long refdim[4] = { 18, 55, 14, 10 };
	allocate_dense_tensor(4, refdim, &r_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "r", H5T_NATIVE_DOUBLE, r_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&r, &r_ref, 1e-13)) {
		return "Kronecker product of tensors does not match reference";
	}

	// clean up
	delete_dense_tensor(&r_ref);
	delete_dense_tensor(&r);
	delete_dense_tensor(&t);
	delete_dense_tensor(&s);

	H5Fclose(file);

	return 0;
}


char* test_dense_tensor_block()
{
	hid_t file = H5Fopen("../test/data/test_dense_tensor_block.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_dense_tensor_transpose failed";
	}

	// create tensor 't'
	struct dense_tensor t;
	const long dim[4] = { 2, 3, 4, 5 };
	allocate_dense_tensor(4, dim,  &t);
	// read values from disk
	if (read_hdf5_dataset(file, "t", H5T_NATIVE_DOUBLE, t.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	const long bdim[4] = { 1, 2, 4, 3 };

	// indices along each dimension
	const long idx0[1] = { 1 };
	const long idx1[2] = { 0, 2 };
	const long idx2[4] = { 0, 1, 2, 3 };
	const long idx3[3] = { 1, 4, 4 }; // index 4 appears twice
	const long *idx[4] = { idx0, idx1, idx2, idx3 };

	struct dense_tensor b;
	dense_tensor_block(&t, bdim, idx, &b);

	// reference tensor for checking
	struct dense_tensor b_ref;
	allocate_dense_tensor(4, bdim, &b_ref);
	// read values from disk
	if (read_hdf5_dataset(file, "b", H5T_NATIVE_DOUBLE, b_ref.data) < 0) {
		return "reading tensor entries from disk failed";
	}

	// compare
	if (!dense_tensor_allclose(&b, &b_ref, 1e-15)) {
		return "extracted sub-block does not match reference";
	}

	// clean up
	delete_dense_tensor(&b_ref);
	delete_dense_tensor(&b);
	delete_dense_tensor(&t);

	H5Fclose(file);

	return 0;
}
