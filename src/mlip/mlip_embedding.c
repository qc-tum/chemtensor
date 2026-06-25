/// \file mlip_embedding.c
/// \brief Machine learning interaction potentials data embedding based on the MACE architecture.

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include "mlip_embedding.h"
#include "spherical_harmonics.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the polynomial cut-off function, single-precision version.
///
static inline float polynomial_cutoff_f(const float r, const float r_cut, const int p)
{
	assert(r >= 0);
	assert(r_cut >= 0);

	if (r >= r_cut) {
		return 0;
	}

	const float x = r / r_cut;
	return 1 - 0.5f * powf(x, p) * (2 + p * (x - 1) * (p * (x - 1) + x - 3));
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate the polynomial cut-off function, double-precision version.
///
static inline double polynomial_cutoff(const double r, const double r_cut, const int p)
{
	assert(r >= 0);
	assert(r_cut >= 0);

	if (r >= r_cut) {
		return 0;
	}

	const double x = r / r_cut;
	return 1 - 0.5 * pow(x, p) * (2 + p * (x - 1) * (p * (x - 1) + x - 3));
}


//________________________________________________________________________________________________________________________
///
/// \brief Embed atomic data based on the MACE architecture.
///
void mlip_embedding_from_atomic_data(const struct mlip_atomic_data* atomic_data, const struct mlip_model* model, struct mlip_embedding* embedding)
{
	assert(atomic_data->positions.ndim == 2);
	assert(atomic_data->positions.dim[0] == atomic_data->neighbors.num_nodes);
	assert(atomic_data->positions.dim[1] == 3);
	assert(abstract_graph_is_consistent(&atomic_data->neighbors));
	assert(atomic_data->node_attrs.dim[0] == atomic_data->neighbors.num_nodes);
	assert(atomic_data->node_attrs.dtype  == atomic_data->positions.dtype);
	assert(atomic_data->node_attrs.dtype == model->params.dtype);

	// node embedding
	assert(model->species_embedding_weight.ndim == 2);
	dense_tensor_dot(&atomic_data->node_attrs, TENSOR_AXIS_RANGE_TRAILING,
	                 &model->species_embedding_weight, TENSOR_AXIS_RANGE_TRAILING, 1,
	                 &embedding->node);

	// factor of 2 since we treat edges as directed
	const int num_edges = 2 * abstract_graph_num_edges(&atomic_data->neighbors);

	// allocate radial embedding tensor
	const ct_long dim_radial[2] = { num_edges, model->params.num_bessel_func };
	allocate_dense_tensor(model->params.dtype, 2, dim_radial, &embedding->radial);

	// allocate angular embedding tensor
	assert(model->params.l_max >= 0);
	const ct_long dim_angular[2] = { num_edges, (model->params.l_max + 1) * (model->params.l_max + 1) };
	allocate_dense_tensor(model->params.dtype, 2, dim_angular, &embedding->angular);

	embedding->edge_indices = ct_malloc(num_edges * 2 * sizeof(int));

	int i_edge = 0;
	for (int i = 0; i < atomic_data->neighbors.num_nodes; i++)
	{
		for (int k = 0; k < atomic_data->neighbors.num_neighbors[i]; k++)
		{
			const int j = atomic_data->neighbors.neighbor_map[i][k];

			embedding->edge_indices[2*i_edge    ] = i;
			embedding->edge_indices[2*i_edge + 1] = j;

			if (atomic_data->positions.dtype == CT_SINGLE_REAL)
			{
				const float* pos = atomic_data->positions.data;
				float vec[3] = {
					pos[3*j    ] - pos[3*i    ],
					pos[3*j + 1] - pos[3*i + 1],
					pos[3*j + 2] - pos[3*i + 2],
				};
				const float r = norm2f(CT_SINGLE_REAL, 3, vec);
				if (r > 0)
				{
					// normalize the vector
					vec[0] /= r;
					vec[1] /= r;
					vec[2] /= r;
				}
				else
				{
					printf("Warning: in 'construct_mlip_embedding': encountered zero difference vector between atomic positions; evaluation of spherical harmonics is not well-defined");
					// ensure that input to spherical harmonics function is valid
					vec[0] = 0;
					vec[1] = 0;
					vec[2] = 1;
				}

				// 'r_cut' interpreted as float
				const float r_cut_val = *((float*)model->params.r_cut);

				// zeroth-order Bessel functions (sinc function)
				float* j0 = (float*)embedding->radial.data + i_edge * model->params.num_bessel_func;
				const float prefactor = sqrtf(2.f / r_cut_val);
				if (r > 0)
				{
					for (int n = 0; n < model->params.num_bessel_func; n++)
					{
						j0[n] = prefactor * (sinf((float)M_PI * (n + 1) * r / r_cut_val) / r) * polynomial_cutoff_f(r, r_cut_val, model->params.p_cut);
					}
				}
				else  // r == 0
				{
					for (int n = 0; n < model->params.num_bessel_func; n++)
					{
						j0[n] = prefactor * ((float)M_PI * (n + 1) / r_cut_val);
					}
				}

				// spherical harmonics
				float* yv = (float*)embedding->angular.data + i_edge * embedding->angular.dim[1];
				const float sqrt4pi = 3.5449077018110320546f;  // sqrt(4 pi)
				for (qnumber l = 0; l <= model->params.l_max; l++)
				{
					real_spherical_harmonics_f(l, vec, yv);
					// scaling factor by normalization convention
					for (int im = 0; im < 2*l + 1; im++) {
						yv[im] *= sqrt4pi;
					}
					yv += (2*l + 1);
				}
				assert(yv == (float*)embedding->angular.data + (i_edge + 1) * embedding->angular.dim[1]);
			}
			else
			{
				assert(atomic_data->positions.dtype == CT_DOUBLE_REAL);

				const double* pos = atomic_data->positions.data;
				double vec[3] = {
					pos[3*j    ] - pos[3*i    ],
					pos[3*j + 1] - pos[3*i + 1],
					pos[3*j + 2] - pos[3*i + 2],
				};
				const double r = norm2(CT_DOUBLE_REAL, 3, vec);
				if (r > 0)
				{
					// normalize the vector
					vec[0] /= r;
					vec[1] /= r;
					vec[2] /= r;
				}
				else
				{
					printf("Warning: in 'construct_mlip_embedding': encountered zero difference vector between atomic positions; evaluation of spherical harmonics is not well-defined");
					// ensure that input to spherical harmonics function is valid
					vec[0] = 0;
					vec[1] = 0;
					vec[2] = 1;
				}

				// 'r_cut' interpreted as double
				const double r_cut_val = *((double*)model->params.r_cut);

				// zeroth-order Bessel functions (sinc function)
				double* j0 = (double*)embedding->radial.data + i_edge * model->params.num_bessel_func;
				const double prefactor = sqrt(2.0 / r_cut_val);
				if (r > 0)
				{
					for (int n = 0; n < model->params.num_bessel_func; n++)
					{
						j0[n] = prefactor * (sin(M_PI * (n + 1) * r / r_cut_val) / r) * polynomial_cutoff(r, r_cut_val, model->params.p_cut);
					}
				}
				else  // r == 0
				{
					for (int n = 0; n < model->params.num_bessel_func; n++)
					{
						j0[n] = prefactor * (M_PI * (n + 1) / r_cut_val);
					}
				}

				// spherical harmonics
				double* yv = (double*)embedding->angular.data + i_edge * embedding->angular.dim[1];
				const double sqrt4pi = 3.5449077018110320546;  // sqrt(4 pi)
				for (qnumber l = 0; l <= model->params.l_max; l++)
				{
					real_spherical_harmonics(l, vec, yv);
					// scaling factor by normalization convention
					for (int im = 0; im < 2*l + 1; im++) {
						yv[im] *= sqrt4pi;
					}
					yv += (2*l + 1);
				}
				assert(yv == (double*)embedding->angular.data + (i_edge + 1) * embedding->angular.dim[1]);
			}

			i_edge++;
		}
	}
	assert(i_edge == num_edges);
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete the data embedding structure (free memory).
///
void delete_mlip_embedding(struct mlip_embedding* embedding)
{
	ct_free(embedding->edge_indices);
	embedding->edge_indices = NULL;
	delete_dense_tensor(&embedding->angular);
	delete_dense_tensor(&embedding->radial);
	delete_dense_tensor(&embedding->node);
}
