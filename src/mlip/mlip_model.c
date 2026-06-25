/// \file mlip_model.c
/// \brief Machine learning interaction potentials model parameters and trainable data based on the MACE architecture.

#include "mlip_model.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Internal consistency check of the machine learning interaction potentials model.
///
bool mlip_model_is_consistent(const struct mlip_model* model)
{
	if (model->params.num_species <= 0) {
		return false;
	}
	// atomic numbers must be positive, and the table must be sorted
	for (int i = 0; i < model->params.num_species - 1; i++)
	{
		if (model->params.z_table[i] <= 0) {
			return false;
		}
		if (model->params.z_table[i] >= model->params.z_table[i + 1]) {
			return false;
		}
	}

	if (!is_real_numeric_type(model->params.dtype)) {
		return false;
	}

	if (model->params.dtype == CT_SINGLE_REAL)
	{
		if (*(float*)model->params.r_cut <= 0) {
			return false;
		}
	}
	else
	{
		if (*(double*)model->params.r_cut <= 0) {
			return false;
		}
	}

	if (model->params.num_layers <= 0) {
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
	delete_dense_tensor(&model->species_embedding_weight);
	ct_free(model->params.z_table);
	ct_free(model->params.r_cut);
}
