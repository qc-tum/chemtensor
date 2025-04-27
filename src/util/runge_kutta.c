/// \file runge_kutta.c
/// \brief Runge-Kutta methods.

#include "runge_kutta.h"
#include "numeric.h"


//________________________________________________________________________________________________________________________
///
/// \brief Runge-Kutta method of order 4 for block-sparse tensors.
///
void runge_kutta_4_block_sparse(const double t, const struct block_sparse_tensor* restrict y, ode_func_block_sparse func, const void* fdata, const double h, struct block_sparse_tensor* restrict y_next)
{
	// ensure that variable sizes are large enough to store any numeric type
	dcomplex one_h, one_half_h, one_sixth_h, one_third_h;
	numeric_from_double(h,     y->dtype, &one_h);
	numeric_from_double(0.5*h, y->dtype, &one_half_h);
	numeric_from_double(h/6,   y->dtype, &one_sixth_h);
	numeric_from_double(h/3,   y->dtype, &one_third_h);

	// k1 = f(t, y)
	struct block_sparse_tensor k1;
	{
		func(t, y, fdata, &k1);
		assert(k1.dtype == y->dtype);
	}

	// k2 = f(t + 0.5*h, y + 0.5*h*k1)
	struct block_sparse_tensor k2;
	{
		// y + 0.5*h*k1
		struct block_sparse_tensor yk1;
		copy_block_sparse_tensor(y, &yk1);
		block_sparse_tensor_scalar_multiply_add(&one_half_h, &k1, &yk1);
		
		func(t + 0.5*h, &yk1, fdata, &k2);
		assert(k2.dtype == y->dtype);

		delete_block_sparse_tensor(&yk1);
	}

	// k3 = f(t + 0.5*h, y + 0.5*h*k2)
	struct block_sparse_tensor k3;
	{
		// y + 0.5*h*k2
		struct block_sparse_tensor yk2;
		copy_block_sparse_tensor(y, &yk2);
		block_sparse_tensor_scalar_multiply_add(&one_half_h, &k2, &yk2);

		func(t + 0.5*h, &yk2, fdata, &k3);
		assert(k3.dtype == y->dtype);

		delete_block_sparse_tensor(&yk2);
	}

	// k4 = f(t + h, y + h*k3)
	struct block_sparse_tensor k4;
	{
		// y + h*k3
		struct block_sparse_tensor yk3;
		copy_block_sparse_tensor(y, &yk3);
		block_sparse_tensor_scalar_multiply_add(&one_h, &k3, &yk3);

		func(t + h, &yk3, fdata, &k4);
		assert(k4.dtype == y->dtype);

		delete_block_sparse_tensor(&yk3);
	}

	// y_next = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
	copy_block_sparse_tensor(y, y_next);
	block_sparse_tensor_scalar_multiply_add(&one_sixth_h, &k1, y_next);
	block_sparse_tensor_scalar_multiply_add(&one_third_h, &k2, y_next);
	block_sparse_tensor_scalar_multiply_add(&one_third_h, &k3, y_next);
	block_sparse_tensor_scalar_multiply_add(&one_sixth_h, &k4, y_next);

	delete_block_sparse_tensor(&k4);
	delete_block_sparse_tensor(&k3);
	delete_block_sparse_tensor(&k2);
	delete_block_sparse_tensor(&k1);
}
