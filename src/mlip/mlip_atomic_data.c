/// \file mlip_atomic_data.c
/// \brief Atomic data structure for machine learning interaction potentials.

#include <stdlib.h>
#include "mlip_atomic_data.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Compute the atomic data from an atomic configuration, representing neighborhood relations via a graph.
///
void mlip_atomic_data_from_configuration(const struct atomic_configuration* config, const struct mlip_model_parameters* params, struct mlip_atomic_data* data)
{
	// copy atomic positions
	assert(config->positions.dtype == params->dtype);
	assert(config->positions.ndim == 2);
	assert(config->positions.dim[1] == 3);
	copy_dense_tensor(&config->positions, &data->positions);

	// node attributes store the one-hot encoding of the index into the atomic number table
	{
		// using same data type for node attributes as for atomic positions
		const ct_long dim_node_attrs[2] = { data->positions.dim[0], params->num_species };
		allocate_zero_dense_tensor(data->positions.dtype, 2, dim_node_attrs, &data->node_attrs);

		const size_t dtype_size = sizeof_numeric_type(data->node_attrs.dtype);

		for (ct_long i = 0; i < data->node_attrs.dim[0]; i++)
		{
			// find the index in 'z_table'
			int j;
			for (j = 0; j < params->num_species; j++) {
				if (params->z_table[j] == config->atomic_numbers[i]) {
					break;
				}
			}
			assert(j < params->num_species);

			// store a single 1 entry
			// casting to int8_t* to ensure that pointer arithmetic is performed in terms of bytes
			memcpy((int8_t*)data->node_attrs.data + (i * data->node_attrs.dim[1] + j) * dtype_size, numeric_one(data->node_attrs.dtype), dtype_size);
		}
	}

	// neighborhood graph
	data->neighbors.num_nodes = (int)data->positions.dim[0];
	data->neighbors.neighbor_map = ct_malloc(data->neighbors.num_nodes * sizeof(int*));
	// set number of neighbors initially to zero
	data->neighbors.num_neighbors = ct_calloc(data->neighbors.num_nodes, sizeof(int));
	for (int i = 0; i < data->neighbors.num_nodes; i++)
	{
		int* neighs_i = ct_malloc(data->neighbors.num_nodes * sizeof(int));  // upper bound on required memory

		// search for previous nodes 'j' which have 'i' as a neighbor
		for (int j = 0; j < i; j++)
		{
			for (int k = 0; k < data->neighbors.num_neighbors[j]; k++)
			{
				if (data->neighbors.neighbor_map[j][k] == i)
				{
					neighs_i[data->neighbors.num_neighbors[i]] = j;
					data->neighbors.num_neighbors[i]++;
					break;
				}
			}
		}

		if (data->positions.dtype == CT_SINGLE_REAL)
		{
			for (int j = i + 1; j < data->neighbors.num_nodes; j++)
			{
				const float* pos = data->positions.data;
				const float vec[3] = {
					pos[3*j    ] - pos[3*i    ],
					pos[3*j + 1] - pos[3*i + 1],
					pos[3*j + 2] - pos[3*i + 2],
				};
				const float r = norm2f(CT_SINGLE_REAL, 3, vec);
				if (r <= *((float*)params->r_cut))
				{
					neighs_i[data->neighbors.num_neighbors[i]] = j;
					data->neighbors.num_neighbors[i]++;
				}
			}
		}
		else
		{
			assert(data->positions.dtype == CT_DOUBLE_REAL);
			for (int j = i + 1; j < data->neighbors.num_nodes; j++)
			{
				const double* pos = data->positions.data;
				const double vec[3] = {
					pos[3*j    ] - pos[3*i    ],
					pos[3*j + 1] - pos[3*i + 1],
					pos[3*j + 2] - pos[3*i + 2],
				};
				const double r = norm2(CT_DOUBLE_REAL, 3, vec);
				if (r <= *((double*)params->r_cut))
				{
					neighs_i[data->neighbors.num_neighbors[i]] = j;
					data->neighbors.num_neighbors[i]++;
				}
			}
		}

		data->neighbors.neighbor_map[i] = ct_malloc(data->neighbors.num_neighbors[i] * sizeof(int));
		memcpy(data->neighbors.neighbor_map[i], neighs_i, data->neighbors.num_neighbors[i] * sizeof(int));
		ct_free(neighs_i);
	}
	assert(abstract_graph_is_consistent(&data->neighbors));
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete atomic data (free memory).
///
void delete_mlip_atomic_data(struct mlip_atomic_data* data)
{
	delete_abstract_graph(&data->neighbors);
	delete_dense_tensor(&data->positions);
	delete_dense_tensor(&data->node_attrs);
}
