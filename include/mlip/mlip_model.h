/// \file mlip_model.h
/// \brief Machine learning interaction potentials model parameters and trainable data based on the MACE architecture.

#pragma once

#include "qnumber.h"
#include "numeric.h"
#include "dense_tensor.h"
#include "su2_tensor.h"


//________________________________________________________________________________________________________________________
///
/// \brief Machine learning interaction potentials model parameters based on the MACE architecture.
///
struct mlip_model_parameters
{
	qnumber* z_table;         //!< list of atomic numbers
	void* r_cut;              //!< cutoff radius for determining neighborhood relations and radial embedding
	int num_species;          //!< number of atomic species (length of 'z_table')
	int p_cut;                //!< order of the polynomial cutoff function
	int num_bessel_func;      //!< number of Bessel functions for the radial embedding
	qnumber l_max;            //!< maximum angular quantum number for the spherical harmonics
	int num_channels;         //!< number of internal channels
	int num_layers;           //!< number of message passing layers
	enum numeric_type dtype;  //!< numeric data type (single or double precision real)
};


//________________________________________________________________________________________________________________________
///
/// \brief Machine learning interaction potentials model parameters and weights, based on the MACE architecture.
///
struct mlip_model
{
	struct mlip_model_parameters params;           //!< model parameters
	struct dense_tensor species_embedding_weight;  //!< weights for the species embedding, dimensions "number of channels" x "number of node features"
};


bool mlip_model_is_consistent(const struct mlip_model* model);


void delete_mlip_model(struct mlip_model* model);
