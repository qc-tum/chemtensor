/// \file dense_layer.c
/// \brief Transformation effected by a dense neural network layer.

#include <memory.h>
#include <math.h>
#include "dense_layer.h"
#include "activation_function.h"


//________________________________________________________________________________________________________________________
///
/// \brief Delete the dense layer specification (free memory).
///
void delete_dense_layer_spec(struct dense_layer_spec* spec)
{
	delete_dense_tensor(&spec->weight);
}


//________________________________________________________________________________________________________________________
///
/// \brief Forward pass of a dense neural network layer.
///
/// The input 'x' is a matrix of dimension "batch size" x "feature dimension".
///
void dense_layer_forward(const struct dense_layer_spec* spec, const struct dense_tensor* restrict x, struct dense_tensor* restrict out)
{
	const enum numeric_type dtype = spec->weight.dtype;

	assert(x->ndim == 2);
	assert(x->dtype == dtype);

	ct_numeric alpha;
	if (spec->normalize)
	{
		// scale by 1 / sqrt(num_features)
		numeric_from_double(1.0 / sqrt(x->dim[1]), dtype, alpha);
	}
	else
	{
		memcpy(alpha, numeric_one(dtype), sizeof_numeric_type(dtype));
	}

	dense_tensor_dot(alpha, x, TENSOR_AXIS_RANGE_TRAILING, &spec->weight, TENSOR_AXIS_RANGE_TRAILING, 1, out);

	// TODO: fuse linear transformation with activation function evaluation
	switch (spec->activation_func)
	{
		case ACTIVATION_IDENTITY:
		{
			// nothing to do
			break;
		}
		case ACTIVATION_RELU:
		{
			const ct_long nelem = dense_tensor_num_elements(out);

			switch (out->dtype)
			{
				case CT_SINGLE_REAL:
				{
					float* data = out->data;
					float coeff = *((float*)spec->activation_prefactor);
					for (ct_long j = 0; j < nelem; j++)
					{
						data[j] = coeff * reluf(data[j]);
					}
					break;
				}
				case CT_DOUBLE_REAL:
				{
					double* data = out->data;
					double coeff = *((double*)spec->activation_prefactor);
					for (ct_long j = 0; j < nelem; j++)
					{
						data[j] = coeff * relu(data[j]);
					}
					break;
				}
				case CT_SINGLE_COMPLEX:
				{
					scomplex* data = out->data;
					scomplex coeff = *((scomplex*)spec->activation_prefactor);
					for (ct_long j = 0; j < nelem; j++)
					{
						data[j] = coeff * creluf(data[j]);
					}
					break;
				}
				case CT_DOUBLE_COMPLEX:
				{
					dcomplex* data = out->data;
					dcomplex coeff = *((dcomplex*)spec->activation_prefactor);
					for (ct_long j = 0; j < nelem; j++)
					{
						data[j] = coeff * crelu(data[j]);
					}
					break;
				}
				default:
				{
					// unknown data type
					assert(false);
				}
			}
			break;
		}
		case ACTIVATION_SILU:
		{
			const ct_long nelem = dense_tensor_num_elements(out);

			switch (out->dtype)
			{
				case CT_SINGLE_REAL:
				{
					float* data = out->data;
					float coeff = *((float*)spec->activation_prefactor);
					for (ct_long j = 0; j < nelem; j++)
					{
						data[j] = coeff * siluf(data[j]);
					}
					break;
				}
				case CT_DOUBLE_REAL:
				{
					double* data = out->data;
					double coeff = *((double*)spec->activation_prefactor);
					for (ct_long j = 0; j < nelem; j++)
					{
						data[j] = coeff * silu(data[j]);
					}
					break;
				}
				case CT_SINGLE_COMPLEX:
				{
					scomplex* data = out->data;
					scomplex coeff = *((scomplex*)spec->activation_prefactor);
					for (ct_long j = 0; j < nelem; j++)
					{
						data[j] = coeff * csiluf(data[j]);
					}
					break;
				}
				case CT_DOUBLE_COMPLEX:
				{
					dcomplex* data = out->data;
					dcomplex coeff = *((dcomplex*)spec->activation_prefactor);
					for (ct_long j = 0; j < nelem; j++)
					{
						data[j] = coeff * csilu(data[j]);
					}
					break;
				}
				default:
				{
					// unknown data type
					assert(false);
				}
			}
			break;
		}
		default:
		{
			// unknown activation function
			assert(false);
			break;
		}
	}
}
