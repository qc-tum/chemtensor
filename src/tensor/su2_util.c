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

	struct su2_irrep_trie_node irrep_trie_simple = { 0 };
	#ifndef NDEBUG
	ct_long nsec_simple = 0;
	#endif

	qnumber* jlist_alt = ct_malloc(ndim * sizeof(qnumber));

	bool* marked = ct_calloc(data_yoga->charge_sectors.nsec, sizeof(bool));
	for (ct_long c = 0; c < data_yoga->charge_sectors.nsec; c++)
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
		ct_long* idx = ct_malloc(n * sizeof(ct_long));

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

			memcpy(jlist_alt, jlist, ndim * sizeof(qnumber));
			jlist_alt[eid] = jy;
			struct dense_tensor** ds = (struct dense_tensor**)su2_irrep_trie_search_insert(jlist_alt, ndim, &irrep_trie_simple);
			assert((*ds) == NULL);

			for (int j = 0; j < n; j++)
			{
				if (idx[j] == -1) {
					continue;
				}

				const qnumber jx = jx_min + 2*j;

				const double coeff = (variant == YOGA_TREE_VARIANT_L ?
					su2_recoupling_coefficient(j1, jx, j4, jy, j3, j2) :
					su2_recoupling_coefficient(j2, jx, j3, jy, j4, j1));
				if ((*ds) == NULL)
				{
					(*ds) = ct_malloc(sizeof(struct dense_tensor));
					copy_dense_tensor(data_yoga->degensors[idx[j]], (*ds));
					// 'alpha' can either store a float or double number
					double alpha;
					numeric_from_double(coeff, numeric_real_type((*ds)->dtype), &alpha);
					rscale_dense_tensor(&alpha, (*ds));
				}
				else
				{
					// ensure that 'alpha' is large enough to store any numeric type
					dcomplex alpha;
					numeric_from_double(coeff, (*ds)->dtype, &alpha);
					dense_tensor_scalar_multiply_add(&alpha, data_yoga->degensors[idx[j]], (*ds));
				}
			}
			assert((*ds) != NULL);

			#ifndef NDEBUG
			nsec_simple++;
			#endif
		}

		ct_free(idx);
	}

	ct_free(jlist_alt);
	ct_free(marked);

	#ifndef NDEBUG
	assert(nsec_simple >= data_yoga->charge_sectors.nsec);
	#endif

	// enumerate new charge sectors and tensors lexicographically
	data_simple->degensors = (struct dense_tensor**)su2_irrep_trie_enumerate_configurations(ndim, &irrep_trie_simple, &data_simple->charge_sectors);
	#ifndef NDEBUG
	assert(data_simple->charge_sectors.nsec == nsec_simple);
	#endif

	delete_su2_irrep_trie(ndim, &irrep_trie_simple);

	// update subtree topology accordingly
	su2_graph_yoga_to_simple_subtree(graph, eid);
}
