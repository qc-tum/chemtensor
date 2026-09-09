/// \file mlip_model.c
/// \brief Machine learning interaction potentials model parameters and trainable data based on the MACE architecture.

#include "mlip_model.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Internal consistency check of the machine learning interaction potentials model parameters.
///
bool mlip_model_parameters_are_consistent(const struct mlip_model_parameters* params)
{
	if (params->num_species <= 0) {
		return false;
	}
	// atomic numbers must be positive, and the table must be sorted
	for (int i = 0; i < params->num_species - 1; i++)
	{
		if (params->z_table[i] <= 0) {
			return false;
		}
		if (params->z_table[i] >= params->z_table[i + 1]) {
			return false;
		}
	}

	if (!is_real_numeric_type(params->dtype)) {
		return false;
	}

	if (params->dtype == CT_SINGLE_REAL)
	{
		if (*(float*)params->r_cut <= 0) {
			return false;
		}
		if (*(float*)params->avg_num_neighbors <= 0) {
			return false;
		}
	}
	else
	{
		if (*(double*)params->r_cut <= 0) {
			return false;
		}
		if (*(double*)params->avg_num_neighbors <= 0) {
			return false;
		}
	}

	if (params->num_layers <= 0) {
		return false;
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a machine learning interaction potentials model layer (free memory).
///
void delete_mlip_model_layer(struct mlip_model_layer* layer)
{
	delete_dense_tensor(&layer->node_linear);
	delete_fully_connected_net_spec(&layer->radial_mlp);
	delete_su2_tensor(&layer->atomic_base_linear);
}


//________________________________________________________________________________________________________________________
///
/// \brief Internal consistency check of the machine learning interaction potentials model.
///
bool mlip_model_is_consistent(const struct mlip_model* model)
{
	if (!mlip_model_parameters_are_consistent(&model->params)) {
		return false;
	}

	// output dimension of first radial MLP
	const struct fully_connected_net_spec* first_radial_mlp = &model->layers[0].radial_mlp;
	if (first_radial_mlp->layers[first_radial_mlp->nlayers - 1].weight.dim[0]
		!= model->params.num_channels * (model->params.l_max + 1)) {
		return false;
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a machine learning interaction potentials model data (free memory).
///
void delete_mlip_model(struct mlip_model* model)
{
	for (int i = 0; i < model->params.num_layers; i++) {
		delete_mlip_model_layer(&model->layers[i]);
	}
	ct_free(model->layers);
	delete_dense_tensor(&model->species_embedding_weight);
	ct_free(model->params.z_table);
}
