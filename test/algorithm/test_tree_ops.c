#include <math.h>
#include <complex.h>
#include "tree_ops.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


char* test_ttno_inner_product()
{
	// local physical dimension and quantum numbers
	const long d = 2;
	const qnumber qsite[2] = { -1, 2 };

	// number of physical and branching lattice sites
	const int nsites_physical  = 7;
	const int nsites_branching = 3;
	const int nsites = nsites_physical + nsites_branching;

	// tree topology:
	//
	//                    6
	//                   ╱
	//                  ╱
	//           2     9
	//            ╲   ╱
	//             ╲ ╱
	//              1
	//              │
	//              │
	//  7 ─── 4 ─── 8 ─── 3
	//             ╱ ╲
	//            ╱   ╲
	//           0     5
	//
	int neigh0[] = { 8 };
	int neigh1[] = { 2, 8, 9 };
	int neigh2[] = { 1 };
	int neigh3[] = { 8 };
	int neigh4[] = { 7, 8 };
	int neigh5[] = { 8 };
	int neigh6[] = { 9 };
	int neigh7[] = { 4 };
	int neigh8[] = { 0, 1, 3, 4, 5 };
	int neigh9[] = { 1, 6 };
	int* neighbor_map[10] = {
		neigh0, neigh1, neigh2, neigh3, neigh4, neigh5, neigh6, neigh7, neigh8, neigh9,
	};
	int num_neighbors[10] = {
		ARRLEN(neigh0), ARRLEN(neigh1), ARRLEN(neigh2), ARRLEN(neigh3), ARRLEN(neigh4), ARRLEN(neigh5), ARRLEN(neigh6), ARRLEN(neigh7), ARRLEN(neigh8), ARRLEN(neigh9),
	};
	struct abstract_graph topology = {
		.neighbor_map  = neighbor_map,
		.num_neighbors = num_neighbors,
		.num_nodes     = nsites,
	};
	assert(abstract_graph_is_connected_tree(&topology));

	struct rng_state rng_state;
	seed_rng_state(47, &rng_state);

	const qnumber qnum_sector = 2;

	struct ttns chi, psi;
	construct_random_ttns(CT_DOUBLE_COMPLEX, nsites_physical, &topology, d, qsite, qnum_sector, 13, &rng_state, &chi);
	construct_random_ttns(CT_DOUBLE_COMPLEX, nsites_physical, &topology, d, qsite, qnum_sector, 17, &rng_state, &psi);

	if (!ttns_is_consistent(&chi) || !ttns_is_consistent(&psi)) {
		return "internal TTNS consistency check failed";
	}
	if (ttns_quantum_number_sector(&chi) != qnum_sector || ttns_quantum_number_sector(&psi) != qnum_sector) {
		return "TTNS quantum number sector differs from expected value";
	}

	struct ttno op;
	construct_random_ttno(CT_DOUBLE_COMPLEX, nsites_physical, &topology, d, qsite, 22, &rng_state, &op);
	if (!ttno_is_consistent(&op)) {
		return "internal TTNO consistency check failed";
	}

	// scale tensor entries such that norm of logical states and operator is around 1
	for (int i = 0; i < nsites; i++)
	{
		const double alpha = 2.5;
		rscale_block_sparse_tensor(&alpha, &chi.a[i]);
		rscale_block_sparse_tensor(&alpha, &psi.a[i]);
		rscale_block_sparse_tensor(&alpha, &op.a[i]);
	}

	// compute expectation value
	dcomplex s;
	ttno_inner_product(&chi, &op, &psi, &s);

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

	struct block_sparse_tensor op_mat;
	ttno_to_matrix(&op, &op_mat);

	dcomplex s_ref;
	{
		struct block_sparse_tensor op_psi_ref;
		block_sparse_tensor_dot(&op_mat, TENSOR_AXIS_RANGE_TRAILING, &psi_vec, TENSOR_AXIS_RANGE_LEADING, 1, &op_psi_ref);

		struct block_sparse_tensor chi_vec_conj;
		copy_block_sparse_tensor(&chi_vec, &chi_vec_conj);
		conjugate_block_sparse_tensor(&chi_vec_conj);
		block_sparse_tensor_reverse_axis_directions(&chi_vec_conj);

		struct block_sparse_tensor t_vdot_ref;
		block_sparse_tensor_dot(&chi_vec_conj, TENSOR_AXIS_RANGE_TRAILING, &op_psi_ref, TENSOR_AXIS_RANGE_TRAILING, 2, &t_vdot_ref);
		assert(t_vdot_ref.ndim == 0);
		assert(t_vdot_ref.blocks[0] != NULL);
		assert(t_vdot_ref.blocks[0]->data != NULL);

		s_ref = *((dcomplex*)t_vdot_ref.blocks[0]->data);

		delete_block_sparse_tensor(&t_vdot_ref);
		delete_block_sparse_tensor(&chi_vec_conj);
		delete_block_sparse_tensor(&op_psi_ref);
	}

	// compare
	if (cabs(s - s_ref) / cabs(s_ref) > 5e-12) {
		return "operator inner product for tree topology does not match reference value";
	}

	delete_block_sparse_tensor(&op_mat);
	delete_block_sparse_tensor(&psi_vec);
	delete_block_sparse_tensor(&chi_vec);

	delete_ttno(&op);
	delete_ttns(&psi);
	delete_ttns(&chi);

	return 0;
}