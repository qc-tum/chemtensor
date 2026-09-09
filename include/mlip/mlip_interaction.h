/// \file mlip_interaction.h
/// \brief Machine learning interaction potentials interaction (neighbor summation) layer, based on the MACE architecture.

#pragma once

#include "mlip_model.h"
#include "mlip_embedding.h"


void mlip_interaction_first_layer(const struct mlip_embedding* embedding, const struct mlip_model* model, struct dense_tensor* atomic_base);
