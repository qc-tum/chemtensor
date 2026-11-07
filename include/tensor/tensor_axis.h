/// \file tensor_axis.h
/// \brief Tensor axis description.

#pragma once


//________________________________________________________________________________________________________________________
///
/// \brief Tensor axis direction (orientation), equivalent to endowing quantum numbers with a sign factor.
///
enum tensor_axis_direction
{
	TENSOR_AXIS_IN  = -1,  //!< logical input axis
	TENSOR_AXIS_OUT =  1,  //!< logical output axis
};
