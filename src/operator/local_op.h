/// \file local_op.h
/// \brief (Site-)local operators.

#pragma once

#include "dense_tensor.h"


//________________________________________________________________________________________________________________________
///
/// \brief Local operator ID and corresponding coefficient.
///
struct local_op_ref
{
	int oid;        //!< operator ID
	double coeff;   //!< coefficient
};


void construct_local_operator(const struct local_op_ref* opics, const int nopics, const struct dense_tensor* opmap, struct dense_tensor* op);
