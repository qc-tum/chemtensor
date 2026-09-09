/// \file dense_layer.h
/// \brief Transformation effected by a dense neural network layer.

#pragma once

#include <stdbool.h>
#include "dense_tensor.h"


//________________________________________________________________________________________________________________________
///
/// \brief Activation function type of a neural network layer.
///
enum activation_function_type
{
	ACTIVATION_IDENTITY = 0,  //!< identity map
	ACTIVATION_RELU     = 1,  //!< rectified linear unit function: max(0, x)
	ACTIVATION_SILU     = 2,  //!< sigmoid linear unit function: x * sigma(x) with sigma(x) = 1 / (1 + exp(-x))
};


//________________________________________________________________________________________________________________________
///
/// \brief Specification (weights and activation function) of a dense neural network layer, without bias vector.
///
struct dense_layer_spec
{
	struct dense_tensor weight;                     //!< weight matrix
	bool normalize;                                 //!< whether to normalize the input and output data by the factor 'sqrt(num_features)'
	enum activation_function_type activation_func;  //!< activation function
	ct_numeric activation_prefactor;                //!< scalar prefactor for the activation function
};


void delete_dense_layer_spec(struct dense_layer_spec* spec);


//________________________________________________________________________________________________________________________
//


void dense_layer_forward(const struct dense_layer_spec* spec, const struct dense_tensor* restrict x, struct dense_tensor* restrict out);
