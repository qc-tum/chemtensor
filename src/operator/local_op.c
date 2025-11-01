/// \file local_op.c
/// \brief (Site-)local operators.

#include "local_op.h"


//________________________________________________________________________________________________________________________
///
/// \brief Test whether a coefficient map is valid, i.e., contains 0 and 1 as first two entries
///
bool coefficient_map_is_valid(const enum numeric_type dtype, const void* coeffmap)
{
	switch (dtype)
	{
		case CT_SINGLE_REAL:
		{
			const float* cmap = coeffmap;
			return (cmap[0] == 0) && (cmap[1] == 1);
		}
		case CT_DOUBLE_REAL:
		{
			const double* cmap = coeffmap;
			return (cmap[0] == 0) && (cmap[1] == 1);
		}
		case CT_SINGLE_COMPLEX:
		{
			const scomplex* cmap = coeffmap;
			return (cmap[0] == 0) && (cmap[1] == 1);
		}
		case CT_DOUBLE_COMPLEX:
		{
			const dcomplex* cmap = coeffmap;
			return (cmap[0] == 0) && (cmap[1] == 1);
		}
		default:
		{
			// unknown data type
			assert(false);
			return false;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct the local operator as dense matrix.
///
void construct_local_operator(const struct local_op_ref* opics, const int nopics, const struct dense_tensor* opmap, const void* coeffmap, struct dense_tensor* op)
{
	assert(nopics > 0);

	if (opics[0].oid == OID_NOP)
	{
		// construct dummy identity operator
		const ct_long dim[2] = { 1, 1 };
		allocate_dense_tensor(opmap[0].dtype, 2, dim, op);
		dense_tensor_set_identity(op);
		// scale by weight factor
		switch (op->dtype)
		{
			case CT_SINGLE_REAL:
			{
				const float* cmap = coeffmap;
				float weight = 0;
				for (int i = 0; i < nopics; i++) {
					// operators must be consistent
					assert(opics[i].oid == OID_NOP);
					weight += cmap[opics[i].cid];
				}
				scale_dense_tensor(&weight, op);
				break;
			}
			case CT_DOUBLE_REAL:
			{
				const double* cmap = coeffmap;
				double weight = 0;
				for (int i = 0; i < nopics; i++) {
					// operators must be consistent
					assert(opics[i].oid == OID_NOP);
					weight += cmap[opics[i].cid];
				}
				scale_dense_tensor(&weight, op);
				break;
			}
			case CT_SINGLE_COMPLEX:
			{
				const scomplex* cmap = coeffmap;
				scomplex weight = 0;
				for (int i = 0; i < nopics; i++) {
					// operators must be consistent
					assert(opics[i].oid == OID_NOP);
					weight += cmap[opics[i].cid];
				}
				scale_dense_tensor(&weight, op);
				break;
			}
			case CT_DOUBLE_COMPLEX:
			{
				const dcomplex* cmap = coeffmap;
				dcomplex weight = 0;
				for (int i = 0; i < nopics; i++) {
					// operators must be consistent
					assert(opics[i].oid == OID_NOP);
					weight += cmap[opics[i].cid];
				}
				scale_dense_tensor(&weight, op);
				break;
			}
			default:
			{
				// unknown data type
				assert(false);
			}
		}
	}
	else
	{
		// first summand
		copy_dense_tensor(&opmap[opics[0].oid], op);
		switch (op->dtype)
		{
			case CT_SINGLE_REAL:
			{
				const float* cmap = coeffmap;
				scale_dense_tensor(&cmap[opics[0].cid], op);
				// add the other summands
				for (int i = 1; i < nopics; i++) {
					dense_tensor_scalar_multiply_add(&cmap[opics[i].cid], &opmap[opics[i].oid], op);
				}
				break;
			}
			case CT_DOUBLE_REAL:
			{
				const double* cmap = coeffmap;
				scale_dense_tensor(&cmap[opics[0].cid], op);
				// add the other summands
				for (int i = 1; i < nopics; i++) {
					dense_tensor_scalar_multiply_add(&cmap[opics[i].cid], &opmap[opics[i].oid], op);
				}
				break;
			}
			case CT_SINGLE_COMPLEX:
			{
				const scomplex* cmap = coeffmap;
				scale_dense_tensor(&cmap[opics[0].cid], op);
				// add the other summands
				for (int i = 1; i < nopics; i++) {
					dense_tensor_scalar_multiply_add(&cmap[opics[i].cid], &opmap[opics[i].oid], op);
				}
				break;
			}
			case CT_DOUBLE_COMPLEX:
			{
				const dcomplex* cmap = coeffmap;
				scale_dense_tensor(&cmap[opics[0].cid], op);
				// add the other summands
				for (int i = 1; i < nopics; i++) {
					dense_tensor_scalar_multiply_add(&cmap[opics[i].cid], &opmap[opics[i].oid], op);
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
}
