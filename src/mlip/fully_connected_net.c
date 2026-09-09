/// \file fully_connected_net.c
/// \brief Fully connected artificial neural network, consisting of a sequence of dense layers.

#include "fully_connected_net.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Allocate memory for the fully connected network specification, without the layer weights.
///
void allocate_empty_fully_connected_net_spec(const int nlayers, struct fully_connected_net_spec* spec)
{
	assert(nlayers > 0);
	spec->nlayers = nlayers;
	spec->layers = ct_calloc(nlayers, sizeof(struct dense_layer_spec));
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete the fully connected network specification (free memory).
///
void delete_fully_connected_net_spec(struct fully_connected_net_spec* spec)
{
	for (int i = 0; i < spec->nlayers; i++) {
		delete_dense_layer_spec(&spec->layers[i]);
	}
	ct_free(spec->layers);
	spec->layers = NULL;
	spec->nlayers = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Forward pass of a fully connected artificial neural network.
///
void fully_connected_net_forward(const struct fully_connected_net_spec* spec, const struct dense_tensor* restrict x, struct dense_tensor* restrict out)
{
	assert(spec->nlayers > 0);

	// copy internal data pointers
	struct dense_tensor tmp = *x;
	for (int i = 0; i < spec->nlayers; i++)
	{
		dense_layer_forward(&spec->layers[i], &tmp, out);
		if (i > 0) {
			delete_dense_tensor(&tmp);
		}
		tmp = *out;
	}
}
