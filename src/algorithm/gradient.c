/// \file gradient.c
/// \brief Gradient computation for operators.

#include <memory.h>
#include <assert.h>
#include "gradient.h"
#include "operation.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Compute the value and gradient of `<chi | op | psi>` with respect to the internal MPO coefficients.
///
void operator_average_coefficient_gradient(const struct mpo_graph* graph, const struct dense_tensor* opmap, const void* coeffmap, const long num_coeffs, const struct mps* psi, const struct mps* chi, void* avr, void* dcoeff)
{
	assert(chi->d == psi->d);
	assert(qnumber_all_equal(psi->d, chi->qsite, psi->qsite));

	const enum numeric_type dtype = opmap[0].dtype;
	assert(dtype == psi->a[0].dtype);
	assert(dtype == chi->a[0].dtype);

	const int nsites = graph->nsites;
	assert(chi->nsites == nsites);
	assert(psi->nsites == nsites);
	assert(nsites >= 1);

	// construct MPO corresponding to MPO graph
	struct mpo mpo;
	mpo_from_graph(dtype, psi->d, psi->qsite, graph, opmap, coeffmap, &mpo);

	// right operator blocks
	struct block_sparse_tensor* rblocks = aligned_alloc(MEM_DATA_ALIGN, nsites * sizeof(struct block_sparse_tensor));
	compute_right_operator_blocks(psi, chi, &mpo, rblocks);
	struct block_sparse_tensor lblock;
	create_dummy_operator_block_left(&psi->a[0], &chi->a[0], &mpo.a[0], &lblock);

	// compute expectation value
	{
		struct block_sparse_tensor r;
		contraction_operator_step_right(&psi->a[0], &chi->a[0], &mpo.a[0], &rblocks[0], &r);

		// flatten left virtual bonds
		struct block_sparse_tensor t;
		flatten_block_sparse_tensor_axes(&r, 0, TENSOR_AXIS_OUT, &t);
		delete_block_sparse_tensor(&r);
		flatten_block_sparse_tensor_axes(&t, 0, TENSOR_AXIS_OUT, &r);
		delete_block_sparse_tensor(&t);

		// 'r' should now be a 1 x 1 tensor
		assert(r.ndim == 2);
		assert(r.dim_logical[0] == 1 && r.dim_logical[1] == 1);
		assert(r.blocks[0] != NULL);
		assert(r.blocks[0]->dtype == dtype);
		memcpy(avr, r.blocks[0]->data, sizeof_numeric_type(dtype));

		delete_block_sparse_tensor(&r);
	}

	// set gradients initially to zero
	memset(dcoeff, 0, num_coeffs * sizeof_numeric_type(dtype));

	for (int l = 0; l < nsites; l++)
	{
		struct block_sparse_tensor dw;
		compute_local_hamiltonian_environment(&psi->a[l], &chi->a[l], &lblock, &rblocks[l], &dw);
		assert(dw.ndim == 4);
		// physical dimensions
		assert(dw.dim_logical[1] == psi->d);
		assert(dw.dim_logical[2] == psi->d);
		assert(qnumber_all_equal(psi->d, dw.qnums_logical[1], psi->qsite));
		assert(qnumber_all_equal(psi->d, dw.qnums_logical[2], psi->qsite));
		for (int i = 0; i < 4; i++)
		{
			assert(dw.dim_logical[i] == mpo.a[l].dim_logical[i]);
			assert(qnumber_all_equal(dw.dim_logical[i], dw.qnums_logical[i], mpo.a[l].qnums_logical[i]));
			assert(dw.axis_dir[i] == -mpo.a[l].axis_dir[i]);
		}

		// entry accessor
		struct block_sparse_tensor_entry_accessor acc;
		create_block_sparse_tensor_entry_accessor(&dw, &acc);

		for (int i = 0; i < graph->num_edges[l]; i++)
		{
			const struct mpo_graph_edge* edge = &graph->edges[l][i];
			assert(graph->verts[l    ][edge->vids[0]].qnum == dw.qnums_logical[0][edge->vids[0]]);
			assert(graph->verts[l + 1][edge->vids[1]].qnum == dw.qnums_logical[3][edge->vids[1]]);

			for (int j = 0; j < edge->nopics; j++)
			{
				const int cid = edge->opics[j].cid;
				assert(cid < num_coeffs);

				const struct dense_tensor* op = &opmap[edge->opics[j].oid];
				assert(op->dtype == dtype);
				assert(op->ndim == 2);
				assert(op->dim[0] == psi->d);
				assert(op->dim[1] == psi->d);

				// explicitly contract physical axes
				switch (dtype)
				{
					case SINGLE_REAL:
					{
						float* dc = dcoeff;
						const float* opdata = op->data;
						for (long x = 0; x < op->dim[0]; x++)
						{
							for (long y = 0; y < op->dim[1]; y++)
							{
								if (opdata[x*op->dim[1] + y] == 0) {
									continue;
								}
								long index_dw[4] = { edge->vids[0], x, y, edge->vids[1] };
								float* pentry = block_sparse_tensor_get_entry(&acc, index_dw);
								if (pentry != NULL) {
									// accumulate gradient
									dc[cid] += opdata[x*op->dim[1] + y] * (*pentry);
								}
							}
						}
						break;
					}
					case DOUBLE_REAL:
					{
						double* dc = dcoeff;
						const double* opdata = op->data;
						for (long x = 0; x < op->dim[0]; x++)
						{
							for (long y = 0; y < op->dim[1]; y++)
							{
								if (opdata[x*op->dim[1] + y] == 0) {
									continue;
								}
								long index_dw[4] = { edge->vids[0], x, y, edge->vids[1] };
								double* pentry = block_sparse_tensor_get_entry(&acc, index_dw);
								if (pentry != NULL) {
									// accumulate gradient
									dc[cid] += opdata[x*op->dim[1] + y] * (*pentry);
								}
							}
						}
						break;
					}
					case SINGLE_COMPLEX:
					{
						scomplex* dc = dcoeff;
						const scomplex* opdata = op->data;
						for (long x = 0; x < op->dim[0]; x++)
						{
							for (long y = 0; y < op->dim[1]; y++)
							{
								if (opdata[x*op->dim[1] + y] == 0) {
									continue;
								}
								long index_dw[4] = { edge->vids[0], x, y, edge->vids[1] };
								scomplex* pentry = block_sparse_tensor_get_entry(&acc, index_dw);
								if (pentry != NULL) {
									// accumulate gradient
									dc[cid] += opdata[x*op->dim[1] + y] * (*pentry);
								}
							}
						}
						break;
					}
					case DOUBLE_COMPLEX:
					{
						dcomplex* dc = dcoeff;
						const dcomplex* opdata = op->data;
						for (long x = 0; x < op->dim[0]; x++)
						{
							for (long y = 0; y < op->dim[1]; y++)
							{
								if (opdata[x*op->dim[1] + y] == 0) {
									continue;
								}
								long index_dw[4] = { edge->vids[0], x, y, edge->vids[1] };
								dcomplex* pentry = block_sparse_tensor_get_entry(&acc, index_dw);
								if (pentry != NULL) {
									// accumulate gradient
									dc[cid] += opdata[x*op->dim[1] + y] * (*pentry);
								}
							}
						}
						break;
					}
					default:
					{
						// unknown data type
						assert(false);
					}
				}
			}
		}

		delete_block_sparse_tensor_entry_accessor(&acc);
		delete_block_sparse_tensor(&dw);
		delete_block_sparse_tensor(&rblocks[l]);

		// left block for next step
		struct block_sparse_tensor lblock_next;
		contraction_operator_step_left(&psi->a[l], &chi->a[l], &mpo.a[l], &lblock, &lblock_next);
		delete_block_sparse_tensor(&lblock);
		move_block_sparse_tensor_data(&lblock_next, &lblock);
	}

	aligned_free(rblocks);
	delete_block_sparse_tensor(&lblock);
	delete_mpo(&mpo);
}
