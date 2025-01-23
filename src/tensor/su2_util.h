/// \file su2_util.h
/// \brief Utility functions for SU(2) symmetric tensors.

#pragma once

#include "dense_tensor.h"
#include "su2_irrep_lists.h"
#include "su2_graph.h"


//________________________________________________________________________________________________________________________
///
/// \brief Essential data stored in an SU(2) symmetric tensor.
///
struct su2_tensor_data
{
	struct charge_sectors charge_sectors;  //!< charge sectors (irreducible 'j' quantum number configurations)
	struct dense_tensor** degensors;       //!< dense "degeneracy" tensors, pointer array of length "number of charge sectors"
};


void su2_convert_yoga_to_simple_subtree(const struct su2_tensor_data* restrict data_yoga, struct su2_graph* graph, const int eid, struct su2_tensor_data* restrict data_simple);
