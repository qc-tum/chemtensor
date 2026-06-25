/// \file mlip_embedding.h
/// \brief Machine learning interaction potentials data embedding based on the MACE architecture.

#pragma once

#include "mlip_model.h"
#include "mlip_atomic_data.h"


//________________________________________________________________________________________________________________________
///
/// \brief Machine learning interaction potentials data embedding based on the MACE architecture.
///
struct mlip_embedding
{
	struct dense_tensor node;     //!< node embedding (chemical species embedding), for each node; dimensions "number of nodes" x "number of channels"
	struct dense_tensor radial;   //!< radial embedding (Bessel functions), for each edge; dimensions "number of edges" x "number of Bessel functions"
	struct dense_tensor angular;  //!< angular embedding (spherical harmonics), for each edge; dimensions "number of edges" x (l_max + 1)^2
	int* edge_indices;            //!< edge indices, containing directed edge tuples (i, j); dimensions "number of edges" x 2
};


void mlip_embedding_from_atomic_data(const struct mlip_atomic_data* atomic_data, const struct mlip_model* model, struct mlip_embedding* embedding);

void delete_mlip_embedding(struct mlip_embedding* embedding);


//________________________________________________________________________________________________________________________
///
/// \brief Number of nodes in the MLIP data embedding.
///
static inline ct_long mlip_embedding_num_nodes(const struct mlip_embedding* embedding)
{
	return embedding->node.dim[0];
}


//________________________________________________________________________________________________________________________
///
/// \brief Number of channels for the node features in the MLIP data embedding.
///
static inline ct_long mlip_embedding_num_channels(const struct mlip_embedding* embedding)
{
	return embedding->node.dim[1];
}


//________________________________________________________________________________________________________________________
///
/// \brief Number of directed edges.
///
static inline ct_long mlip_embedding_num_edges(const struct mlip_embedding* embedding)
{
	assert(embedding->radial.dim[0] == embedding->angular.dim[0]);
	return embedding->radial.dim[0];
}
