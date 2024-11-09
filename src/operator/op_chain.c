/// \file op_chain.c
/// \brief Symbolic operator chain data structure and utility functions.

#include <assert.h>
#include "op_chain.h"
#include "local_op.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for an operator chain, and initialize coefficient by default value 1.
///
void allocate_op_chain(const int length, struct op_chain* chain)
{
	assert(length > 0);
	chain->oids   = ct_calloc(length, sizeof(int));
	chain->qnums  = ct_calloc(length + 1, sizeof(qnumber));  // includes a leading and trailing quantum number
	chain->length = length;
	chain->cid    = CID_ONE;  // initialize coefficient by default value 1
	chain->istart = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete an operator chain (free memory).
///
void delete_op_chain(struct op_chain* chain)
{
	ct_free(chain->qnums);
	ct_free(chain->oids);
	chain->qnums = NULL;
	chain->oids  = NULL;
	chain->length = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Copy an operator chain, allocating memory for the copy.
///
void copy_op_chain(const struct op_chain* restrict src, struct op_chain* restrict dst)
{
	allocate_op_chain(src->length, dst);

	memcpy(dst->oids,  src->oids,   src->length      * sizeof(int));
	memcpy(dst->qnums, src->qnums, (src->length + 1) * sizeof(qnumber));
	dst->cid    = src->cid;
	dst->istart = src->istart;
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct a new operator chain with identities padded on the left and right.
///
void op_chain_pad_identities(const struct op_chain* restrict chain, const int new_length, struct op_chain* restrict pad_chain)
{
	assert(new_length >= chain->istart + chain->length);

	// initial operator IDs and quantum numbers are set to 0
	allocate_op_chain(new_length, pad_chain);

	// using that OID_IDENTITY == 0 for leading and trailing operator IDs
	memcpy(pad_chain->oids + chain->istart, chain->oids, chain->length * sizeof(int));

	// quantum numbers
	memcpy(pad_chain->qnums + chain->istart, chain->qnums, (chain->length + 1) * sizeof(qnumber));

	pad_chain->cid = chain->cid;

	pad_chain->istart = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct the full matrix representation of the operator chain.
///
void op_chain_to_matrix(const struct op_chain* chain, const long d, const int nsites, const struct dense_tensor* opmap, const void* coeffmap, const enum numeric_type dtype, struct dense_tensor* a)
{
	assert(d >= 1);

	// leading identity matrices times coefficient
	const long d0 = ipow(d, chain->istart);
	const long dim0[2] = { d0, d0 };
	allocate_dense_tensor(dtype, 2, dim0, a);
	dense_tensor_set_identity(a);
	switch (dtype)
	{
		case CT_SINGLE_REAL:
		{
			const float* cmap = coeffmap;
			scale_dense_tensor(&cmap[chain->cid], a);
			break;
		}
		case CT_DOUBLE_REAL:
		{
			const double* cmap = coeffmap;
			scale_dense_tensor(&cmap[chain->cid], a);
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			const scomplex* cmap = coeffmap;
			scale_dense_tensor(&cmap[chain->cid], a);
			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			const dcomplex* cmap = coeffmap;
			scale_dense_tensor(&cmap[chain->cid], a);
			break;
		}
		default:
		{
			// unknown data type
			assert(false);
		}
	}

	for (int l = 0; l < chain->length; l++)
	{
		struct dense_tensor tmp;
		move_dense_tensor_data(a, &tmp);
		const struct dense_tensor* op = &opmap[chain->oids[l]];
		assert(op->ndim == 2);
		assert(op->dim[0] == d && op->dim[1] == d);
		dense_tensor_kronecker_product(&tmp, op, a);
		delete_dense_tensor(&tmp);
	}

	// trailing identity matrices
	const int n_trail = nsites - (chain->istart + chain->length);
	assert(n_trail >= 0);
	const long d1 = ipow(d, n_trail);
	const long dim1[2] = { d1, d1 };
	struct dense_tensor id_trail;
	allocate_dense_tensor(dtype, 2, dim1, &id_trail);
	dense_tensor_set_identity(&id_trail);
	struct dense_tensor tmp;
	move_dense_tensor_data(a, &tmp);
	dense_tensor_kronecker_product(&tmp, &id_trail, a);
	delete_dense_tensor(&tmp);
	delete_dense_tensor(&id_trail);
}
