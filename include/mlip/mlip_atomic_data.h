/// \file mlip_atomic_data.h
/// \brief Atomic data structure for machine learning interaction potentials.

#pragma once

#include <stdbool.h>
#include "atomic_configuration.h"
#include "mlip_model.h"
#include "dense_tensor.h"
#include "abstract_graph.h"
#include "qnumber.h"


//________________________________________________________________________________________________________________________
///
/// \brief Atomic data for machine learning interaction potentials, representing neighborhood relations via a graph.
///
struct mlip_atomic_data
{
	struct dense_tensor node_attrs;   //!< node attributes, dimensions "number of nodes" x "number of node features"
	struct dense_tensor positions;    //!< atomic positions, dimensions "number of nodes" x 3
	struct abstract_graph neighbors;  //!< neighbors of each node (within cutoff radius)
};


void mlip_atomic_data_from_configuration(const struct atomic_configuration* config, const struct mlip_model_parameters* params, struct mlip_atomic_data* data);

void delete_mlip_atomic_data(struct mlip_atomic_data* data);
