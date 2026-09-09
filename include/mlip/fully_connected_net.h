/// \file fully_connected_net.h
/// \brief Fully connected artificial neural network, consisting of a sequence of dense layers.

#pragma once

#include "dense_layer.h"


//________________________________________________________________________________________________________________________
///
/// \brief Specification of a fully connected artificial neural network.
///
struct fully_connected_net_spec
{
	struct dense_layer_spec* layers;  //!< individual layer specifications
	int nlayers;                      //!< number of layers
};

void allocate_empty_fully_connected_net_spec(const int nlayers, struct fully_connected_net_spec* spec);

void delete_fully_connected_net_spec(struct fully_connected_net_spec* spec);


//________________________________________________________________________________________________________________________
//


void fully_connected_net_forward(const struct fully_connected_net_spec* spec, const struct dense_tensor* x, struct dense_tensor* out);
