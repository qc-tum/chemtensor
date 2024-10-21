#include <math.h>
#include <complex.h>
#include "ttns.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


char* test_ttns_vdot()
{
	// local physical dimension and quantum numbers
	const long d = 3;
	const qnumber qsite[3] = { 1, 2, -1 };

	// number of physical and branching lattice sites
	const int nsites_physical  = 5;
	const int nsites_branching = 3;
	const int nsites = nsites_physical + nsites_branching;

	// tree topology:
	//
	//  1           4
	//    \       /
	//      \   /
	//        2
	//        |
	//        |
	//        6
	//        |
	//        |
	//  7 --- 5 --- 3
	//        |
	//        |
	//        0
	//
	int neigh0[] = { 5 };
	int neigh1[] = { 2 };
	int neigh2[] = { 1, 4, 6 };
	int neigh3[] = { 5 };
	int neigh4[] = { 2 };
	int neigh5[] = { 0, 3, 6, 7 };
	int neigh6[] = { 2, 5 };
	int neigh7[] = { 5 };
	int* neighbor_map[8] = {
		neigh0, neigh1, neigh2, neigh3, neigh4, neigh5, neigh6, neigh7,
	};
	int num_neighbors[8] = {
		ARRLEN(neigh0), ARRLEN(neigh1), ARRLEN(neigh2), ARRLEN(neigh3), ARRLEN(neigh4), ARRLEN(neigh5), ARRLEN(neigh6), ARRLEN(neigh7),
	};
	struct abstract_graph topology = {
		.neighbor_map  = neighbor_map,
		.num_neighbors = num_neighbors,
		.num_nodes     = nsites,
	};
	assert(abstract_graph_is_connected_tree(&topology));

	struct rng_state rng_state;
	seed_rng_state(45, &rng_state);

	const qnumber qnum_sector = 3;

	struct ttns chi, psi;
	construct_random_ttns(CT_SINGLE_COMPLEX, nsites_physical, &topology, d, qsite, qnum_sector, 13, &rng_state, &chi);
	construct_random_ttns(CT_SINGLE_COMPLEX, nsites_physical, &topology, d, qsite, qnum_sector, 17, &rng_state, &psi);

	if (!ttns_is_consistent(&chi) || !ttns_is_consistent(&psi)) {
		return "internal TTNS consistency check failed";
	}
	if (ttns_quantum_number_sector(&chi) != qnum_sector || ttns_quantum_number_sector(&psi) != qnum_sector) {
		return "TTNS quantum number sector differs from expected value";
	}

	// scale tensor entries such that norm of logical state is around 1
	for (int i = 0; i < nsites; i++)
	{
		const float alpha = 4.5;
		rscale_block_sparse_tensor(&alpha, &chi.a[i]);
		rscale_block_sparse_tensor(&alpha, &psi.a[i]);
	}

	// compute inner product
	scomplex s;
	ttns_vdot(&chi, &psi, &s);

	// compute norm of 'psi'
	double nrm_psi = ttns_norm(&psi);

	// reference calculation

	struct block_sparse_tensor chi_vec, psi_vec;
	ttns_to_statevector(&chi, &chi_vec);
	ttns_to_statevector(&psi, &psi_vec);
	// second dimension is auxiliary dimension
	if (chi_vec.ndim != 2 || psi_vec.ndim != 2) {
		return "vector representation of a TTNS must be a tensor of degree two (physical and auxiliary axis)";
	}
	if (chi_vec.dim_logical[1] != 1 || psi_vec.dim_logical[1] != 1) {
		return "auxiliary dimension in vector representation of a TTNS must be 1";
	}
	if (chi_vec.qnums_logical[1][0] != qnum_sector || psi_vec.qnums_logical[1][0] != qnum_sector) {
		return "auxiliary dimension in vector representation of a TTNS does not contain expected quantum number sector";
	}

	scomplex s_ref;
	{
		struct block_sparse_tensor chi_vec_conj;
		copy_block_sparse_tensor(&chi_vec, &chi_vec_conj);
		conjugate_block_sparse_tensor(&chi_vec_conj);
		block_sparse_tensor_reverse_axis_directions(&chi_vec_conj);

		struct block_sparse_tensor t_vdot_ref;
		block_sparse_tensor_dot(&chi_vec_conj, TENSOR_AXIS_RANGE_TRAILING, &psi_vec, TENSOR_AXIS_RANGE_TRAILING, 2, &t_vdot_ref);
		assert(t_vdot_ref.ndim == 0);
		assert(t_vdot_ref.blocks[0] != NULL);
		assert(t_vdot_ref.blocks[0]->data != NULL);

		s_ref = *((scomplex*)t_vdot_ref.blocks[0]->data);

		delete_block_sparse_tensor(&t_vdot_ref);
		delete_block_sparse_tensor(&chi_vec_conj);
	}

	double nrm_psi_ref = block_sparse_tensor_norm2(&psi_vec);

	// compare
	if (cabsf(s - s_ref) / cabsf(s_ref) > 5e-6) {
		return "inner product of two TTNS does not match reference value";
	}
	if (fabs(nrm_psi - nrm_psi_ref) / nrm_psi > 5e-6) {
		return "norm of TTNS does not match reference value";
	}

	delete_block_sparse_tensor(&psi_vec);
	delete_block_sparse_tensor(&chi_vec);

	delete_ttns(&psi);
	delete_ttns(&chi);

	return 0;
}
