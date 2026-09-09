/// \file mlip_interaction.c
/// \brief Machine learning interaction potentials interaction (neighbor summation) layer, based on the MACE architecture.

#include "mlip_interaction.h"
#include "fully_connected_net.h"


//________________________________________________________________________________________________________________________
///
/// \brief Forward-pass of the first interaction layer of the MACE architecture.
///
/// The output 'atomic_base' is the 'A' tensor in the atomic cluster expansion,
/// of dimension "number of nodes" x "number of channels" * (l_max + 1)^2.
///
void mlip_interaction_first_layer(const struct mlip_embedding* embedding, const struct mlip_model* model, struct dense_tensor* atomic_base)
{
	assert(model->params.num_layers > 0);

	const int dim_l = model->params.l_max + 1;

	// number of nodes
	const int num_nodes = mlip_embedding_num_nodes(embedding);

	// number of directed edges
	const int num_edges = (int)embedding->angular.dim[0];
	assert(embedding->radial.dim[0] == num_edges);
	assert(embedding->angular.dim[1] == dim_l * dim_l);

	// number of channels
	const int num_channels = model->params.num_channels;

	assert(embedding->node.dtype == embedding->radial.dtype);
	assert(embedding->node.dtype == embedding->angular.dtype);

	// initial linear transformation of the node features
	struct dense_tensor h;
	dense_tensor_dot(numeric_one(embedding->node.dtype), &embedding->node, TENSOR_AXIS_RANGE_TRAILING, &model->layers[0].node_linear, TENSOR_AXIS_RANGE_TRAILING, 1, &h);

	struct dense_tensor r;
	fully_connected_net_forward(&model->layers[0].radial_mlp, &embedding->radial, &r);
	assert(r.ndim == 2);
	// first dimension equals number of edges
	assert(r.dim[0] == num_edges);

	// compute the neighbor sum of the intermediate 'phi' functions:
	// phi_{ij,klm} = R_{kl}(|r_{ij}|) Y_{lm}(\hat{r}_{ij}) h_{jk},
	// where i, j are neighboring node indices (atoms), k the channel index, and (l, m) spherical harmonics indices
	struct dense_tensor phi_sum;
	const ct_long dim_phi_sum[2] = { num_nodes, num_channels * embedding->angular.dim[1] };
	allocate_zero_dense_tensor(embedding->node.dtype, 2, dim_phi_sum, &phi_sum);
	// use 'int' type for index calculations
	const int sph_dim = (int)embedding->angular.dim[1];
	if (phi_sum.dtype == CT_SINGLE_REAL)
	{
		const float* hdata = h.data;
		const float* rdata = r.data;
		const float* ydata = embedding->angular.data;
		float* phi_data = phi_sum.data;

		for (int e = 0; e < num_edges; e++)
		{
			const int i = embedding->edge_indices[2 * e];
			const int j = embedding->edge_indices[2 * e + 1];
			assert(0 <= i && i < num_nodes);
			assert(0 <= j && j < num_nodes);

			int lm_offset = 0;
			for (int l = 0; l < dim_l; l++)
			{
				for (int k = 0; k < num_channels; k++)
				{
					for (int im = 0; im < 2 * l + 1; im++)
					{
						phi_data[(j * sph_dim + lm_offset) * num_channels + k * (2 * l + 1) + im] +=
							  rdata[(e * dim_l + l) * num_channels + k]  // note R_{l,k} index order, following MACE convention
							* ydata[e * sph_dim + (lm_offset + im)]
							* hdata[i * num_channels + k];
					}
				}

				lm_offset += 2 * l + 1;
				assert(lm_offset == (l + 1) * (l + 1));
			}
		}

		const float inv_avg_num_neighbors = 1.f / *((float*)model->params.avg_num_neighbors);
		rscale_dense_tensor(&inv_avg_num_neighbors, &phi_sum);
	}
	else
	{
		const double* hdata = h.data;
		const double* rdata = r.data;
		const double* ydata = embedding->angular.data;
		double* phi_data = phi_sum.data;

		for (int e = 0; e < num_edges; e++)
		{
			const int i = embedding->edge_indices[2 * e];
			const int j = embedding->edge_indices[2 * e + 1];
			assert(0 <= i && i < num_nodes);
			assert(0 <= j && j < num_nodes);

			int lm_offset = 0;
			for (int l = 0; l < dim_l; l++)
			{
				for (int k = 0; k < num_channels; k++)
				{
					for (int im = 0; im < 2 * l + 1; im++)
					{
						phi_data[(j * sph_dim + lm_offset) * num_channels + k * (2 * l + 1) + im] +=
							  rdata[(e * dim_l + l) * num_channels + k]  // note R_{l,k} index order, following MACE convention
							* ydata[e * sph_dim + (lm_offset + im)]
							* hdata[i * num_channels + k];
					}
				}

				lm_offset += 2 * l + 1;
				assert(lm_offset == (l + 1) * (l + 1));
			}
		}

		const double inv_avg_num_neighbors = 1.0 / *((double*)model->params.avg_num_neighbors);
		rscale_dense_tensor(&inv_avg_num_neighbors, &phi_sum);
	}

	assert(model->layers[0].atomic_base_linear.ndim_logical == 2);
	dense_su2_tensor_dot(&phi_sum, TENSOR_AXIS_RANGE_TRAILING, &model->layers[0].atomic_base_linear, TENSOR_AXIS_RANGE_TRAILING, 1, atomic_base);

	delete_dense_tensor(&phi_sum);
	delete_dense_tensor(&r);
	delete_dense_tensor(&h);
}
