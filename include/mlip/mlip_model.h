/// \file mlip_model.h
/// \brief Machine learning interaction potentials model parameters and trainable data based on the MACE architecture.

#pragma once

#include "qnumber.h"
#include "numeric.h"
#include "dense_tensor.h"
#include "su2_tensor.h"
#include "fully_connected_net.h"


//________________________________________________________________________________________________________________________
///
/// \brief Machine learning interaction potentials model parameters based on the MACE architecture.
///
struct mlip_model_parameters
{
	qnumber* z_table;              //!< list of atomic numbers
	ct_realnum r_cut;              //!< cutoff radius for determining neighborhood relations and radial embedding
	int num_species;               //!< number of atomic species (length of 'z_table')
	int p_cut;                     //!< order of the polynomial cutoff function
	int num_bessel_func;           //!< number of Bessel functions for the radial embedding
	qnumber l_max;                 //!< maximum angular quantum number for the spherical harmonics
	int num_channels;              //!< number of internal channels
	int num_layers;                //!< number of message passing layers
	ct_realnum avg_num_neighbors;  //!< average number of neighbors of the atoms, used for internal message normalisation
	enum numeric_type dtype;       //!< numeric data type (single or double precision real)
};


bool mlip_model_parameters_are_consistent(const struct mlip_model_parameters* params);


//________________________________________________________________________________________________________________________
///
/// \brief Machine learning interaction potentials message-passing layer, based on the MACE architecture.
///
struct mlip_model_layer
{
	struct dense_tensor node_linear;             //!< linear transformation of the node features
	struct fully_connected_net_spec radial_mlp;  //!< radial transformation
	struct su2_tensor atomic_base_linear;        //!< linear transformation to compute the 'A' functions
};


void delete_mlip_model_layer(struct mlip_model_layer* layer);


//________________________________________________________________________________________________________________________
///
/// \brief Machine learning interaction potentials model parameters and weights, based on the MACE architecture.
///
struct mlip_model
{
	struct mlip_model_parameters params;           //!< model parameters
	struct dense_tensor species_embedding_weight;  //!< weights for the species embedding, dimensions "number of channels" x "number of node features"
	struct mlip_model_layer* layers;               //!< weights for each message-passing layer
};


bool mlip_model_is_consistent(const struct mlip_model* model);


void delete_mlip_model(struct mlip_model* model);
