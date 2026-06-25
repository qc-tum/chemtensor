/// \file atomic_configuration.h
/// \brief Atomic configuration (atomic positions and nuclear charges) data structure.

#pragma once

#include "qnumber.h"
#include "dense_tensor.h"


//________________________________________________________________________________________________________________________
///
/// \brief Atomic configuration.
///
struct atomic_configuration
{
	struct dense_tensor positions;    //!< atomic positions, matrix of size "number of atoms" x 3
	qnumber* atomic_numbers;          //!< atomic number (nuclear charge number Z) of each atom
};
