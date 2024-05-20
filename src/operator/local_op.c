/// \file local_op.c
/// \brief (Site-)local operators.

#include "local_op.h"


//________________________________________________________________________________________________________________________
///
/// \brief Construct the local operator as dense matrix.
///
void construct_local_operator(const struct local_op_ref* opics, const int nopics, const struct dense_tensor* opmap, struct dense_tensor* op)
{
	assert(nopics > 0);

	// first summand
	copy_dense_tensor(&opmap[opics[0].oid], op);
	if (op->dtype == SINGLE_REAL || op->dtype == SINGLE_COMPLEX) {
		float alpha = (float)opics[0].coeff;
		rscale_dense_tensor(&alpha, op);
	}
	else {
		assert(op->dtype == DOUBLE_REAL || op->dtype == DOUBLE_COMPLEX);
		rscale_dense_tensor(&opics[0].coeff, op);
	}

	// add the other summands
	for (int i = 1; i < nopics; i++)
	{
		// ensure that 'alpha' is large enough to store any numeric type
		dcomplex alpha;
		numeric_from_double(opics[i].coeff, op->dtype, &alpha);
		dense_tensor_scalar_multiply_add(&alpha, &opmap[opics[i].oid], op);
	}
}
