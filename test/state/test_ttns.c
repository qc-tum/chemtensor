#include <math.h>
#include <complex.h>
#include "ttns.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


char* test_ttns_vdot()
{
	// number of physical and branching lattice sites
	const int nsites_physical  = 5;
	const int nsites_branching = 3;
	const int nsites = nsites_physical + nsites_branching;

	// tree topology:
	//
	//     1     4
	//      ╲   ╱
	//       ╲ ╱
	//        2
	//        │
	//        │
	//        6
	//        │
	//        │
	//  7 ─── 5 ─── 3
	//        │
	//        │
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

	// local physical dimensions and quantum numbers
	const long d[8] = { 3, 1, 7, 2, 4, 1, 1, 1 };
	const qnumber qsite0[3] = { 1, 2, -1 };
	const qnumber qsite1[1] = { -3 };
	const qnumber qsite2[7] = { 2, 1, 4, 0, 3, 1, 0 };
	const qnumber qsite3[2] = { 1, 0 };
	const qnumber qsite4[4] = { 1, 2, 0, 1 };
	const qnumber qzero[1]  = { 0 };
	const qnumber* qsite[8] = {
		qsite0, qsite1, qsite2, qsite3, qsite4, qzero, qzero, qzero,
	};
	
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


char* test_ttns_compress()
{
	hid_t file = H5Fopen("../test/state/data/test_ttns_compress.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_mps_compress failed";
	}

	const hid_t hdf5_dcomplex_id = construct_hdf5_double_complex_dtype(false);

	// number of physical and branching lattice sites
	const int nsites_physical  = 8;
	const int nsites_branching = 1;
	const int nsites = nsites_physical + nsites_branching;

	// tree topology:
	//
	//           6
	//           │
	//           │
	//     2     8 ─── 1
	//      ╲   ╱
	//       ╲ ╱
	//        3
	//        │
	//        │
	//  5 ─── 7 ─── 0
	//        │
	//        │
	//        4
	//
	int neigh0[] = { 7 };
	int neigh1[] = { 8 };
	int neigh2[] = { 3 };
	int neigh3[] = { 2, 7, 8 };
	int neigh4[] = { 7 };
	int neigh5[] = { 7 };
	int neigh6[] = { 8 };
	int neigh7[] = { 0, 3, 4, 5 };
	int neigh8[] = { 1, 3, 6 };
	int* neighbor_map[9] = {
		neigh0, neigh1, neigh2, neigh3, neigh4, neigh5, neigh6, neigh7, neigh8,
	};
	int num_neighbors[9] = {
		ARRLEN(neigh0), ARRLEN(neigh1), ARRLEN(neigh2), ARRLEN(neigh3), ARRLEN(neigh4), ARRLEN(neigh5), ARRLEN(neigh6), ARRLEN(neigh7), ARRLEN(neigh8),
	};
	struct abstract_graph topology = {
		.neighbor_map  = neighbor_map,
		.num_neighbors = num_neighbors,
		.num_nodes     = nsites,
	};
	assert(abstract_graph_is_connected_tree(&topology));

	// local physical dimensions and quantum numbers
	long* d = ct_malloc(nsites * sizeof(long));
	qnumber** qsite = ct_malloc(nsites * sizeof(qnumber*));
	for (int l = 0; l < nsites; l++)
	{
		char varname[1024];
		sprintf(varname, "qsite%i", l);

		hsize_t d_loc[1];
		if (get_hdf5_attribute_dims(file, varname, d_loc) < 0) {
			return "obtaining local physical dimension failed";
		}
		d[l] = d_loc[0];

		qsite[l] = ct_malloc(d[l] * sizeof(qnumber));
		if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qsite[l]) < 0) {
			return "reading physical quantum numbers from disk failed";
		}
	}

	// overall quantum number sector
	qnumber qnum_sector;
	if (read_hdf5_attribute(file, "qnum_sector", H5T_NATIVE_INT, &qnum_sector) < 0) {
		return "reading quantum number sector from disk failed";
	}

	// virtual bond dimensions and quantum numbers
	long* dim_bonds  = ct_calloc(nsites * nsites, sizeof(long));
	qnumber** qbonds = ct_calloc(nsites * nsites, sizeof(qnumber*));
	for (int l = 0; l < nsites; l++)
	{
		for (int i = 0; i < topology.num_neighbors[l]; i++)
		{
			int k = topology.neighbor_map[l][i];
			assert(k != l);
			if (k > l) {
				continue;
			}

			const int ib = k*nsites + l;

			char varname[1024];
			sprintf(varname, "qbond%i%i", k, l);

			hsize_t dim_bond[1];
			if (get_hdf5_attribute_dims(file, varname, dim_bond) < 0) {
				return "obtaining virtual bond dimension failed";
			}
			dim_bonds[ib] = dim_bond[0];

			qbonds[ib] = ct_malloc(dim_bonds[ib] * sizeof(qnumber));
			if (read_hdf5_attribute(file, varname, H5T_NATIVE_INT, qbonds[ib]) < 0) {
				return "reading virtual bond quantum numbers from disk failed";
			}
		}
	}

	for (int j = 0; j < 2; j++)  // determines compression tolerance
	{
		struct ttns ttns;
		allocate_ttns(CT_DOUBLE_COMPLEX, nsites_physical, &topology, d, (const qnumber**)qsite, qnum_sector, dim_bonds, (const qnumber**)qbonds, &ttns);

		// read TTNS tensors from disk
		for (int l = 0; l < nsites; l++)
		{
			struct dense_tensor a_dns;
			allocate_dense_tensor(ttns.a[l].dtype, ttns.a[l].ndim, ttns.a[l].dim_logical, &a_dns);
			char varname[1024];
			sprintf(varname, "a%i", l);
			if (read_hdf5_dataset(file, varname, hdf5_dcomplex_id, a_dns.data) < 0) {
				return "reading tensor entries from disk failed";
			}

			dense_to_block_sparse_tensor_entries(&a_dns, &ttns.a[l]);

			delete_dense_tensor(&a_dns);
		}

		if (!ttns_is_consistent(&ttns)) {
			return "internal TTNS consistency check failed";
		}

		if (ttns_norm(&ttns) < 1e-2) {
			return "norm of TTNS is too small";
		}

		// convert original TTNS to a state vector
		struct block_sparse_tensor vec_ref;
		ttns_to_statevector(&ttns, &vec_ref);

		// perform compression
		const int i_root = 3;
		const double tol_compress = (j == 0 ? 0. : 1e-7);
		const long max_vdim = 1024;
		int ret = ttns_compress(i_root, tol_compress, true, max_vdim, &ttns);
		if (ret < 0) {
			return "'ttns_compress' failed internally";
		}

		if (!ttns_is_consistent(&ttns)) {
			return "internal TTNS consistency check failed";
		}

		// convert compressed TTNS to a state vector
		struct block_sparse_tensor vec;
		ttns_to_statevector(&ttns, &vec);

		// compare with original state vector
		if (!block_sparse_tensor_allclose(&vec, &vec_ref, tol_compress == 0. ? 1e-13 : 1e-3)) {
			return "vector representation of TTNS after compression is not close to original state vector";
		}

		delete_block_sparse_tensor(&vec_ref);
		delete_block_sparse_tensor(&vec);
		delete_ttns(&ttns);
	}

	// clean up
	for (int l = 0; l < nsites * nsites; l++)
	{
		if (qbonds[l] != NULL) {
			assert(dim_bonds[l] > 0);
			ct_free(qbonds[l]);
		}
		else {
			assert(dim_bonds[l] == 0);
		}
	}
	ct_free(qbonds);
	ct_free(dim_bonds);
	for (int l = 0; l < nsites; l++) {
		ct_free(qsite[l]);
	}
	ct_free(qsite);
	ct_free(d);

	H5Tclose(hdf5_dcomplex_id);
	H5Fclose(file);

	return 0;
}
