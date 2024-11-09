/// \file op_chain.h
/// \brief Symbolic operator chain data structure and utility functions.

#pragma once

#include "numeric.h"
#include "qnumber.h"
#include "dense_tensor.h"


//________________________________________________________________________________________________________________________
///
/// \brief Symbolic operator chain `coeff op_i x op_{i+1} x ... x op_{i+n-1}`,
/// with `op_i` acting on lattice site `i`
///
/// A single bond quantum number is interleaved between each `op_i` and `op_{i+1}`;
/// set all quantum numbers to zero to effectively disable them.
///
struct op_chain
{
	int* oids;          //!< list of local op_i operator IDs
	qnumber* qnums;     //!< interleaved bond quantum numbers, including a leading and trailing quantum number
	int cid;            //!< index of coefficient (scalar factor)
	int length;         //!< length of the operator chain (number of local operators)
	int istart;         //!< first lattice site the operator chain acts on
};


void allocate_op_chain(const int length, struct op_chain* chain);

void delete_op_chain(struct op_chain* chain);

void copy_op_chain(const struct op_chain* restrict src, struct op_chain* restrict dst);


void op_chain_pad_identities(const struct op_chain* restrict chain, const int new_length, struct op_chain* restrict pad_chain);


//________________________________________________________________________________________________________________________
//

// conversion to full matrix (intended for testing)

void op_chain_to_matrix(const struct op_chain* chain, const long d, const int nsites, const struct dense_tensor* opmap, const void* coeffmap, const enum numeric_type dtype, struct dense_tensor* a);
