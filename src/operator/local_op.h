/// \file local_op.h
/// \brief (Site-)local operators.

#pragma once

#include "dense_tensor.h"


//________________________________________________________________________________________________________________________
///
/// \brief Convention for operator indices.
///
enum
{
	OID_IDENTITY = 0,  //!< index corresponding to identity operation
};


//________________________________________________________________________________________________________________________
///
/// \brief Convention for coefficient indices.
///
enum
{
	CID_ZERO = 0,  //!< index corresponding to numerical value zero
	CID_ONE  = 1,  //!< index corresponding to numerical value one
};


bool coefficient_map_is_valid(const enum numeric_type dtype, const void* coeffmap);


//________________________________________________________________________________________________________________________
///
/// \brief Local operator ID and corresponding coefficient.
///
struct local_op_ref
{
	int oid;   //!< operator ID
	int cid;   //!< coefficient index
};


void construct_local_operator(const struct local_op_ref* opics, const int nopics, const struct dense_tensor* opmap, const void* coeffmap, struct dense_tensor* op);
