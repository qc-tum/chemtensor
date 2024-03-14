/// \file mpo.c
/// \brief Matrix product operator (MPO) data structure and functions.

#include <stdio.h>
#include "mpo.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Construct an MPO from an MPO graph.
///
void mpo_from_graph(const enum numeric_type dtype, const long d, const qnumber* qsite, const struct mpo_graph* graph, const struct dense_tensor* opmap, struct mpo* mpo)
{
	assert(graph->nsites >= 1);
	assert(d >= 1);
	mpo->nsites = graph->nsites;
	mpo->d = d;

	mpo->qsite = aligned_alloc(MEM_DATA_ALIGN, d * sizeof(qnumber));
	memcpy(mpo->qsite, qsite, d * sizeof(qnumber));

	mpo->a = aligned_calloc(MEM_DATA_ALIGN, graph->nsites, sizeof(struct block_sparse_tensor));

	for (int l = 0; l < graph->nsites; l++)
	{
		// accumulate entries in a dense tensor first
		const long dim_a_loc[4] = { graph->num_nodes[l], d, d, graph->num_nodes[l + 1] };
		struct dense_tensor a_loc;
		allocate_dense_tensor(dtype, 4, dim_a_loc, &a_loc);

		for (int i = 0; i < graph->num_edges[l]; i++)
		{
			const struct mpo_graph_edge* edge = &graph->edges[l][i];
			struct dense_tensor op;
			mpo_graph_edge_local_op(edge, opmap, &op);

			assert(op.ndim == 2);
			assert(op.dim[0] == d && op.dim[1] == d);
			assert(op.dtype == a_loc.dtype);
			assert(0 <= edge->nids[0] && edge->nids[0] < graph->num_nodes[l]);
			assert(0 <= edge->nids[1] && edge->nids[1] < graph->num_nodes[l + 1]);

			// add entries of local operator 'op' to 'a_loc' (supporting multiple edges between same pair of nodes)
			const long index_start[4] = { edge->nids[0], 0, 0, edge->nids[1] };
			long offset = tensor_index_to_offset(a_loc.ndim, a_loc.dim, index_start);
			switch (a_loc.dtype)
			{
				case SINGLE_REAL:
				{
					float* al_data = a_loc.data;
					const float* op_data = op.data;
					for (long j = 0; j < d*d; j++, offset += a_loc.dim[3])
					{
						al_data[offset] += op_data[j];
					}
					break;
				}
				case DOUBLE_REAL:
				{
					double* al_data = a_loc.data;
					const double* op_data = op.data;
					for (long j = 0; j < d*d; j++, offset += a_loc.dim[3])
					{
						al_data[offset] += op_data[j];
					}
					break;
				}
				case SINGLE_COMPLEX:
				{
					scomplex* al_data = a_loc.data;
					const scomplex* op_data = op.data;
					for (long j = 0; j < d*d; j++, offset += a_loc.dim[3])
					{
						al_data[offset] += op_data[j];
					}
					break;
				}
				case DOUBLE_COMPLEX:
				{
					dcomplex* al_data = a_loc.data;
					const dcomplex* op_data = op.data;
					for (long j = 0; j < d*d; j++, offset += a_loc.dim[3])
					{
						al_data[offset] += op_data[j];
					}
					break;
				}
				default:
				{
					// unknown data type
					assert(false);
				}
			}

			delete_dense_tensor(&op);
		}

		// note: entries not adhering to the quantum number sparsity pattern are ignored
		qnumber* qbonds[2];
		for (int i = 0; i < 2; i++)
		{
			qbonds[i] = aligned_alloc(MEM_DATA_ALIGN, graph->num_nodes[l + i] * sizeof(qnumber));
			for (int j = 0; j < graph->num_nodes[l + i]; j++)
			{
				qbonds[i][j] = graph->nodes[l + i][j].qnum;
			}
		}
		const enum tensor_axis_direction axis_dir[4] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN, TENSOR_AXIS_IN };
		const qnumber* qnums[4] = { qbonds[0], qsite, qsite, qbonds[1] };
		dense_to_block_sparse_tensor(&a_loc, axis_dir, qnums, &mpo->a[l]);
		for (int i = 0; i < 2; i++)
		{
			aligned_free(qbonds[i]);
		}

		#ifdef DEBUG
		struct dense_tensor a_loc_conv;
		block_sparse_to_dense_tensor(&mpo->a[l], &a_loc_conv);
		if (!dense_tensor_allclose(&a_loc_conv, &a_loc, 0.)) {
			fprintf(stderr, "Warning: ignoring non-zero tensor entries due to the quantum number sparsity pattern in 'mpo_from_graph', site %i\n", l);
		}
		delete_dense_tensor(&a_loc_conv);
		#endif

		delete_dense_tensor(&a_loc);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete a matrix product operator (free memory).
///
void delete_mpo(struct mpo* mpo)
{
	for (int i = 0; i < mpo->nsites; i++)
	{
		delete_block_sparse_tensor(&mpo->a[i]);
	}
	aligned_free(mpo->a);
	mpo->a = NULL;
	mpo->nsites = 0;

	aligned_free(mpo->qsite);
	mpo->qsite = NULL;
	mpo->d = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Internal consistency check of the MPO data structure.
///
bool mpo_is_consistent(const struct mpo* mpo)
{
	if (mpo->nsites <= 0) {
		return false;
	}
	if (mpo->d <= 0) {
		return false;
	}

	// quantum numbers for physical legs of individual tensors must agree with 'qsite'
	for (int i = 0; i < mpo->nsites; i++)
	{
		if (mpo->a[i].ndim != 4) {
			return false;
		}
		if (mpo->a[i].dim_logical[1] != mpo->d || mpo->a[i].dim_logical[2] != mpo->d) {
			return false;
		}
		if (!qnumber_all_equal(mpo->d, mpo->a[i].qnums_logical[1], mpo->qsite) ||
		    !qnumber_all_equal(mpo->d, mpo->a[i].qnums_logical[2], mpo->qsite)) {
			return false;
		}
	}

	// virtual bond quantum numbers must match
	for (int i = 0; i < mpo->nsites - 1; i++)
	{
		if (mpo->a[i].dim_logical[3] != mpo->a[i + 1].dim_logical[0]) {
			return false;
		}
		if (!qnumber_all_equal(mpo->a[i].dim_logical[3], mpo->a[i].qnums_logical[3], mpo->a[i + 1].qnums_logical[0])) {
			return false;
		}
	}

	// axis directions
	for (int i = 0; i < mpo->nsites; i++)
	{
		if (mpo->a[i].axis_dir[0] != TENSOR_AXIS_OUT ||
		    mpo->a[i].axis_dir[1] != TENSOR_AXIS_OUT ||
		    mpo->a[i].axis_dir[2] != TENSOR_AXIS_IN  ||
		    mpo->a[i].axis_dir[3] != TENSOR_AXIS_IN) {
			return false;
		}
	}

	return true;
}


//________________________________________________________________________________________________________________________
///
/// \brief Merge two neighboring MPO tensors.
///
void mpo_merge_tensor_pair(const struct block_sparse_tensor* restrict a0, const struct block_sparse_tensor* restrict a1, struct block_sparse_tensor* restrict a)
{
	assert(a0->ndim == 4);
	assert(a1->ndim == 4);

	// combine a0 and a1 by contracting the shared bond
	struct block_sparse_tensor a0_a1_dot;
	block_sparse_tensor_dot(a0, TENSOR_AXIS_RANGE_TRAILING, a1, TENSOR_AXIS_RANGE_LEADING, 1, &a0_a1_dot);
	assert(a0_a1_dot.ndim == 6);

	// group physical input and output dimensions
	const int perm[6] = { 0, 1, 3, 2, 4, 5 };
	struct block_sparse_tensor a0_a1_dot_perm;
	transpose_block_sparse_tensor(perm, &a0_a1_dot, &a0_a1_dot_perm);
	delete_block_sparse_tensor(&a0_a1_dot);

	// combine original physical dimensions of a0 and a1 into one dimension
	struct block_sparse_tensor tmp;
	flatten_block_sparse_tensor_axes(&a0_a1_dot_perm, 1, TENSOR_AXIS_OUT, &tmp);
	delete_block_sparse_tensor(&a0_a1_dot_perm);
	flatten_block_sparse_tensor_axes(&tmp, 2, TENSOR_AXIS_IN, a);
	delete_block_sparse_tensor(&tmp);

	assert(a->ndim == 4);
}


//________________________________________________________________________________________________________________________
///
/// \brief Merge all tensors of an MPO to obtain the matrix representation on the full Hilbert space.
/// The (dummy) virtual bonds are retained in the output tensor.
///
void mpo_to_matrix(const struct mpo* mpo, struct block_sparse_tensor* mat)
{
	assert(mpo->nsites > 0);

	if (mpo->nsites == 1)
	{
		copy_block_sparse_tensor(&mpo->a[0], mat);
	}
	else if (mpo->nsites == 2)
	{
		mpo_merge_tensor_pair(&mpo->a[0], &mpo->a[1], mat);
	}
	else
	{
		struct block_sparse_tensor t[2];
		mpo_merge_tensor_pair(&mpo->a[0], &mpo->a[1], &t[0]);
		for (int i = 2; i < mpo->nsites; i++)
		{
			mpo_merge_tensor_pair(&t[i % 2], &mpo->a[i], i < mpo->nsites - 1 ? &t[(i + 1) % 2] : mat);
			delete_block_sparse_tensor(&t[i % 2]);
		}
	}

	assert(mat->ndim == 4);
}
