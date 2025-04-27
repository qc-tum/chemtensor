#define _USE_MATH_DEFINES
#include <math.h>
#include "runge_kutta.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


struct ode_func_data
{
	const struct block_sparse_tensor* lintensor;
	const struct block_sparse_tensor* nonlintensors[3];
};


static void ode_func(const double t, const struct block_sparse_tensor* restrict y, const void* restrict data, struct block_sparse_tensor* restrict ret)
{
	const struct ode_func_data* fdata = data;

	// linear term
	block_sparse_tensor_dot(fdata->lintensor, TENSOR_AXIS_RANGE_TRAILING, y, TENSOR_AXIS_RANGE_LEADING, 2, ret);

	// nonlinear term
	struct block_sparse_tensor s;
	for (int i = 0; i < 3; i++)
	{
		struct block_sparse_tensor t;
		block_sparse_tensor_multiply_axis(i == 0 ? y : &s, i, fdata->nonlintensors[i], TENSOR_AXIS_RANGE_TRAILING, &t);
		if (i > 0) {
			delete_block_sparse_tensor(&s);
		}
		s = t;  // copy internal data pointers
	}
	const int perm[3] = { 2, 0, 1 };
	struct block_sparse_tensor k;
	transpose_block_sparse_tensor(perm, &s, &k);
	delete_block_sparse_tensor(&s);
	// square and scale entries of 'k'
	const double scaling = sin(M_PI * (0.25 + 5*t));
	// for each block in 'k'...
	const long nblocks = integer_product(k.dim_blocks, k.ndim);
	for (long i = 0; i < nblocks; i++)
	{
		struct dense_tensor* block = k.blocks[i];
		if (block == NULL) {
			continue;
		}
		dcomplex* data = block->data;
		const long nelem = dense_tensor_num_elements(block);
		for (long j = 0; j < nelem; j++) {
			data[j] = scaling * (data[j] * data[j]);
		}
	}

	block_sparse_tensor_scalar_multiply_add(numeric_one(k.dtype), &k, ret);

	delete_block_sparse_tensor(&k);
}


char* test_runge_kutta_4_block_sparse()
{
	hid_t file = H5Fopen("../test/util/data/test_runge_kutta_4_block_sparse.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_runge_kutta_4_block_sparse failed";
	}

	const hid_t hdf5_dcomplex_id = construct_hdf5_double_complex_dtype(false);

	const int ndim = 3;
	const long dims[3] = { 14, 13, 17 };

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

	// initial and reference final state
	struct block_sparse_tensor y0, y1_ref;
	{
		// read dense tensors from disk
		struct dense_tensor y0_dns;
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, ndim, dims, &y0_dns);
		if (read_hdf5_dataset(file, "y0", hdf5_dcomplex_id, y0_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}
		struct dense_tensor y1_ref_dns;
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, ndim, dims, &y1_ref_dns);
		if (read_hdf5_dataset(file, "y1", hdf5_dcomplex_id, y1_ref_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		// convert dense to block-sparse tensors
		dense_to_block_sparse_tensor(&y0_dns, axis_dir, (const qnumber**)qnums, &y0);
		dense_to_block_sparse_tensor(&y1_ref_dns, axis_dir, (const qnumber**)qnums, &y1_ref);

		delete_dense_tensor(&y1_ref_dns);
		delete_dense_tensor(&y0_dns);
	}

	// tensor defining linear term of ODE
	struct block_sparse_tensor lintensor;
	{
		const long dims_lin[4] = { dims[0], dims[1], dims[0], dims[1] };
		const enum tensor_axis_direction axis_dir_lin[4] = { axis_dir[0], axis_dir[1], -axis_dir[0], -axis_dir[1] };
		const qnumber* qnums_lin[4] = { qnums[0], qnums[1], qnums[0], qnums[1] };

		// read dense tensor from disk
		struct dense_tensor lintensor_dns;
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 4, dims_lin, &lintensor_dns);
		if (read_hdf5_dataset(file, "lintensor", hdf5_dcomplex_id, lintensor_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		// convert dense to block-sparse tensors
		dense_to_block_sparse_tensor(&lintensor_dns, axis_dir_lin, qnums_lin, &lintensor);

		delete_dense_tensor(&lintensor_dns);
	}

	// tensors defining nonlinear term of ODE
	struct block_sparse_tensor nonlintensors[3];
	for (int i = 0; i < ndim; i++)
	{
		const int i_next = (i + 1) % ndim;
		const long dims_nonlin[2] = { dims[i_next], dims[i] };
		const enum tensor_axis_direction axis_dir_nonlin[2] = { axis_dir[i_next], -axis_dir[i] };
		const qnumber* qnums_nonlin[2] = { qnums[i_next], qnums[i] };

		// read dense tensor from disk
		struct dense_tensor nonlintensor_dns;
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 2, dims_nonlin, &nonlintensor_dns);
		char varname[1024];
		sprintf(varname, "nonlintensor%i", i);
		if (read_hdf5_dataset(file, varname, hdf5_dcomplex_id, nonlintensor_dns.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		// convert dense to block-sparse tensors
		dense_to_block_sparse_tensor(&nonlintensor_dns, axis_dir_nonlin, qnums_nonlin, &nonlintensors[i]);

		delete_dense_tensor(&nonlintensor_dns);
	}

	// overall simulation time
	const double tmax = 0.1;

	struct ode_func_data fdata = {
		.lintensor = &lintensor,
		.nonlintensors = { &nonlintensors[0], &nonlintensors[1], &nonlintensors[2] },
	};

	const int nsteps[] = { 1, 4, 16, 32 };
	double err_list[ARRLEN(nsteps)];
	for (int i = 0; i < (int)ARRLEN(nsteps); i++)
	{
		// time step
		const double dt = tmax / nsteps[i];

		struct block_sparse_tensor y;
		copy_block_sparse_tensor(&y0, &y);
		for (int n = 0; n < nsteps[i]; n++)
		{
			struct block_sparse_tensor y_next;
			runge_kutta_4_block_sparse(n * dt, &y, ode_func, &fdata, dt, &y_next);
			delete_block_sparse_tensor(&y);
			y = y_next;  // copy internal data pointers
		}

		// compute error
		struct block_sparse_tensor diff;
		copy_block_sparse_tensor(&y1_ref, &diff);
		block_sparse_tensor_scalar_multiply_add(numeric_neg_one(y.dtype), &y, &diff);
		err_list[i] = block_sparse_tensor_norm2(&diff);
		delete_block_sparse_tensor(&diff);

		delete_block_sparse_tensor(&y);
	}

	if (err_list[3] > 1e-6) {
		return "too large Runge-Kutta 4 error";
	}

	// expecting convergence order 4
	for (int i = 1; i < (int)ARRLEN(err_list); i++)
	{
		double slope = (log(err_list[i - 1]) - log(err_list[i])) / (log(tmax / nsteps[i - 1]) - log(tmax / nsteps[i]));
		if (fabs(slope - 4) > 0.2) {
			return "observed Runge-Kutta 4 convergence order does not match 4";
		}
	}

	// clean up
	for (int i = 0; i < ndim; i++) {
		delete_block_sparse_tensor(&nonlintensors[i]);
	}
	delete_block_sparse_tensor(&lintensor);
	delete_block_sparse_tensor(&y1_ref);
	delete_block_sparse_tensor(&y0);
	for (int i = 0; i < ndim; i++) {
		ct_free(qnums[i]);
	}
	ct_free(qnums);
	ct_free(axis_dir);

	H5Tclose(hdf5_dcomplex_id);
	H5Fclose(file);

	return 0;
}
