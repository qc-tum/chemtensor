/// \file su2_util.c
/// \brief Utility functions for SU(2) symmetric tensors.

#include <stdbool.h>
#include <assert.h>
#include "su2_util.h"
#include "su2_recoupling.h"
#include "aligned_memory.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Charge sector and corresponding "degeneracy" tensor of an SU(2) symmetric tensor (temporary data structure).
///
struct su2_tensor_sector
{
	struct su2_irreducible_list list;  //!< irreducible 'j' quantum number configuration
	struct dense_tensor* degensor;     //!< corresponding dense "degeneracy" tensor
};


//________________________________________________________________________________________________________________________
///
/// \brief Comparison function for sorting.
///
static int compare_su2_tensor_sectors(const void* a, const void* b)
{
	const struct su2_tensor_sector* x = a;
	const struct su2_tensor_sector* y = b;

	return compare_su2_irreducible_lists(&x->list, &y->list);
}


//________________________________________________________________________________________________________________________
///
/// \brief The two variants of a yoga subtree.
///
enum yoga_tree_variant
{
	YOGA_TREE_VARIANT_L = 0,  //!< variant with inner axis pointing down-left
	YOGA_TREE_VARIANT_R = 1,  //!< variant with inner axis pointing down-right
};


//________________________________________________________________________________________________________________________
///
/// \brief Convert a yoga to a simple subtree by a corresponding linear transformation of the dense "degeneracy" tensors.
///
void su2_convert_yoga_to_simple_subtree(const struct su2_tensor_data* restrict data_yoga, struct su2_graph* graph, const int eid, struct su2_tensor_data* restrict data_simple)
{
	assert(su2_graph_is_yoga_edge(graph, eid));
	assert(data_yoga->charge_sectors.nsec > 0);
	assert(data_yoga->charge_sectors.ndim == graph->num_edges);

	const int ndim = data_yoga->charge_sectors.ndim;

	const struct su2_graph_edge* edge = &graph->edges[eid];

	int i_ax_1, i_ax_2, i_ax_3, i_ax_4;
	enum yoga_tree_variant variant;
	if (eid == graph->nodes[edge->nid[0]].eid_child[0])
	{
		variant = YOGA_TREE_VARIANT_L;
		i_ax_1 = graph->nodes[edge->nid[1]].eid_child[0];
		i_ax_2 = graph->nodes[edge->nid[0]].eid_parent;
		i_ax_3 = graph->nodes[edge->nid[1]].eid_parent;
		i_ax_4 = graph->nodes[edge->nid[0]].eid_child[1];
	}
	else
	{
		variant = YOGA_TREE_VARIANT_R;
		i_ax_1 = graph->nodes[edge->nid[0]].eid_parent;
		i_ax_2 = graph->nodes[edge->nid[1]].eid_child[1];
		i_ax_3 = graph->nodes[edge->nid[0]].eid_child[0];
		i_ax_4 = graph->nodes[edge->nid[1]].eid_parent;
	}
	assert(0 <= i_ax_1 && i_ax_1 < ndim && i_ax_1 != eid);
	assert(0 <= i_ax_2 && i_ax_2 < ndim && i_ax_2 != eid);
	assert(0 <= i_ax_3 && i_ax_3 < ndim && i_ax_3 != eid);
	assert(0 <= i_ax_4 && i_ax_4 < ndim && i_ax_4 != eid);

	long capacity = data_yoga->charge_sectors.nsec + 16;
	struct su2_tensor_sector* sectors_simple = ct_malloc(capacity * sizeof(struct su2_tensor_sector));
	long nsec_simple = 0;

	qnumber* jlist_alt = ct_malloc(ndim * sizeof(qnumber));

	bool* marked = ct_calloc(data_yoga->charge_sectors.nsec, sizeof(bool));
	for (long c = 0; c < data_yoga->charge_sectors.nsec; c++)
	{
		if (marked[c]) {
			continue;
		}
		marked[c] = true;

		// current 'j' quantum numbers
		const qnumber* jlist = &data_yoga->charge_sectors.jlists[c * ndim];

		const qnumber j1 = jlist[i_ax_1];
		const qnumber j2 = jlist[i_ax_2];
		const qnumber j3 = jlist[i_ax_3];
		const qnumber j4 = jlist[i_ax_4];
		const qnumber jx_curr = jlist[eid];

		memcpy(jlist_alt, jlist, ndim * sizeof(qnumber));

		// enumerate all allowed 'jx' quantum numbers and mark the 'j' lists in the current 'jx' series
		const qnumber jx_min = qmax(abs(j1 - j3), abs(j2 - j4));
		const qnumber jx_max = qmin(j1 + j3, j2 + j4);
		assert((jx_max - jx_min) % 2 == 0);
		assert(jx_min <= jx_curr && jx_curr <= jx_max);
		assert((j1 + j3 + jx_curr) % 2 == 0);

		const int n = (jx_max - jx_min) / 2 + 1;
		long* idx = ct_malloc(n * sizeof(long));

		for (int j = 0; j < n; j++)
		{
			const qnumber jx = jx_min + 2*j;
			if (jx == jx_curr) {
				idx[j] = c;
				continue;
			}
			jlist_alt[eid] = jx;
			idx[j] = charge_sector_index(&data_yoga->charge_sectors, jlist_alt);
			if (idx[j] != -1) {
				marked[idx[j]] = true;
			}
		}

		// enumerate all allowed 'jy' quantum numbers
		const qnumber jy_min = qmax(abs(j1 - j2), abs(j3 - j4));
		#ifndef NDEBUG
		const qnumber jy_max = qmin(j1 + j2, j3 + j4);
		assert((jy_max - jy_min) % 2 == 0);
		assert(jx_max - jx_min == jy_max - jy_min);
		#endif
		for (int i = 0; i < n; i++)
		{
			const qnumber jy = jy_min + 2*i;

			if (nsec_simple == capacity)
			{
				// re-allocate memory
				struct su2_tensor_sector* old_sectors = sectors_simple;
				capacity += 16;
				sectors_simple = ct_malloc(capacity * sizeof(struct su2_tensor_sector));
				memcpy(sectors_simple, old_sectors, nsec_simple * sizeof(struct su2_tensor_sector));
				ct_free(old_sectors);
			}

			allocate_su2_irreducible_list(ndim, &sectors_simple[nsec_simple].list);
			memcpy(sectors_simple[nsec_simple].list.jlist, jlist, ndim * sizeof(qnumber));
			sectors_simple[nsec_simple].list.jlist[eid] = jy;

			sectors_simple[nsec_simple].degensor = NULL;
			for (int j = 0; j < n; j++)
			{
				if (idx[j] == -1) {
					continue;
				}

				const qnumber jx = jx_min + 2*j;

				const double coeff = (variant == YOGA_TREE_VARIANT_L ?
					su2_recoupling_coefficient(j1, jx, j4, jy, j3, j2) :
					su2_recoupling_coefficient(j2, jx, j3, jy, j4, j1));
				if (sectors_simple[nsec_simple].degensor == NULL)
				{
					sectors_simple[nsec_simple].degensor = ct_malloc(sizeof(struct dense_tensor));
					copy_dense_tensor(data_yoga->degensors[idx[j]], sectors_simple[nsec_simple].degensor);
					// 'alpha' can either store a float or double number
					double alpha;
					numeric_from_double(coeff, numeric_real_type(sectors_simple[nsec_simple].degensor->dtype), &alpha);
					rscale_dense_tensor(&alpha, sectors_simple[nsec_simple].degensor);
				}
				else
				{
					// ensure that 'alpha' is large enough to store any numeric type
					dcomplex alpha;
					numeric_from_double(coeff, sectors_simple[nsec_simple].degensor->dtype, &alpha);
					dense_tensor_scalar_multiply_add(&alpha, data_yoga->degensors[idx[j]], sectors_simple[nsec_simple].degensor);
				}
			}
			assert(sectors_simple[nsec_simple].degensor != NULL);

			nsec_simple++;
		}

		ct_free(idx);
	}

	ct_free(jlist_alt);

	assert(nsec_simple >= data_yoga->charge_sectors.nsec);

	// sort new charge sectors and tensors lexicographically
	qsort(sectors_simple, nsec_simple, sizeof(struct su2_tensor_sector), compare_su2_tensor_sectors);

	// copy data into output arrays
	allocate_charge_sectors(nsec_simple, ndim, &data_simple->charge_sectors);
	data_simple->degensors = ct_malloc(nsec_simple * sizeof(struct dense_tensor*));
	for (long c = 0; c < nsec_simple; c++)
	{
		memcpy(&data_simple->charge_sectors.jlists[c * ndim], sectors_simple[c].list.jlist, ndim * sizeof(qnumber));
		data_simple->degensors[c] = sectors_simple[c].degensor;
	}

	// clean up
	for (long c = 0; c < nsec_simple; c++) {
		delete_su2_irreducible_list(&sectors_simple[c].list);
	}
	ct_free(sectors_simple);
	ct_free(marked);

	// update subtree topology accordingly
	su2_graph_yoga_to_simple_subtree(graph, eid);
}
