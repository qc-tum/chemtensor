#include <math.h>
#include "su2_mps.h"
#include "mps.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


char* test_su2_mps_orthonormalize_qr()
{
	// number of lattice sites
	const int nsites = 5;

	// 'j' quantum numbers at each site
	qnumber jlist[] = { 0, 2 };
	const struct su2_irreducible_list site_irreps = { .jlist = jlist, .num = ARRLEN(jlist) };

	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                             j:  0  1  2
	const ct_long site_dim_degen[] = { 3, 0, 2 };

	struct rng_state rng_state;
	seed_rng_state(60, &rng_state);

	for (int m = 0; m < 2; m++)  // left- or right-orthonormalization
	{
		for (int is = 0; is < 2; is++)  // overall quantum number sector
		{
			const qnumber irrep_sector = 2 * is;
			const qnumber max_bond_irrep = 5;
			const ct_long max_bond_dim_degen = 13;

			struct su2_mps mps;
			construct_random_su2_mps(CT_DOUBLE_COMPLEX, nsites, &site_irreps, site_dim_degen, irrep_sector, max_bond_irrep, max_bond_dim_degen, &rng_state, &mps);
			// rescale tensors such that overall state norm is around 1
			for (int l = 0; l < nsites; l++)
			{
				const double alpha = 14;
				rscale_su2_tensor(&alpha, &mps.a[l]);
			}
			if (!su2_mps_is_consistent(&mps)) {
				return "internal SU(2) MPS consistency check failed";
			}

			// convert original MPS to state vector
			struct su2_tensor vec_ref;
			su2_mps_to_statevector(&mps, &vec_ref);
			if (vec_ref.ndim_logical != nsites + 2) {
				return "expecting the SU(2) statevector representation of an SU(2) MPS to have logical degree 'nsites + 2'";
			}

			double norm = su2_mps_orthonormalize_qr(&mps, m == 0 ? SU2_MPS_ORTHONORMAL_LEFT : SU2_MPS_ORTHONORMAL_RIGHT);

			if (!su2_mps_is_consistent(&mps)) {
				return "internal SU(2) MPS consistency check failed";
			}

			// convert normalized MPS to state vector
			struct su2_tensor vec;
			su2_mps_to_statevector(&mps, &vec);

			// must be normalized
			struct dense_tensor vec_dns;
			su2_to_dense_tensor(&vec, &vec_dns);
			// for right-orthonormalization, the trailing dummy bond is renormalized as part of the isometric MPS tensor at the last site
			if (fabs(dense_tensor_norm2(&vec_dns) - (m == 1 ? 1 : sqrt(irrep_sector + 1))) > 1e-13) {
				return "vector representation of SU(2) MPS after orthonormalization does not have norm 1 (after compensating for irrep sector)";
			}
			delete_dense_tensor(&vec_dns);

			// scaled vector representation must agree with original vector
			rscale_su2_tensor(&norm, &vec);
			if (!su2_tensor_allclose(&vec, &vec_ref, 1e-13)) {
				return "vector representation of SU(2) MPS after orthonormalization does not match reference";
			}

			delete_su2_tensor(&vec);
			delete_su2_tensor(&vec_ref);

			for (int l = 0; l < nsites; l++)
			{
				// mps.a[l] must be an isometry

				struct su2_tensor a_mat;
				if (m == 0)
				{
					su2_tensor_fuse_axes_add_auxiliary(&mps.a[l], 0, 1, &a_mat);
				}
				else
				{
					struct su2_tensor tmp;
					copy_su2_tensor(&mps.a[l], &tmp);
					su2_tensor_reverse_axis_simple(&tmp, 1);
					su2_tensor_fuse_axes_add_auxiliary(&tmp, 1, 2, &a_mat);
					delete_su2_tensor(&tmp);
				}
				if (!su2_tensor_is_isometry(&a_mat, 1e-13, m == 1)) {
					return "SU(2) MPS tensor is not isometric";
				}
				delete_su2_tensor(&a_mat);

				// also check dense representation
				struct dense_tensor a_dns;
				su2_to_dense_tensor(&mps.a[l], &a_dns);
				assert(a_dns.ndim == 3);
				const ct_long dim_mat[2] = { a_dns.dim[0] * (m == 0 ? a_dns.dim[1] : 1), (m == 0 ? 1 : a_dns.dim[1]) * a_dns.dim[2] };
				reshape_dense_tensor(2, dim_mat, &a_dns);
				if (!dense_tensor_is_isometry(&a_dns, 1e-13, m == 1)) {
					return "dense representation of SU(2) MPS tensor is not isometric";
				}
				delete_dense_tensor(&a_dns);
			}

			delete_su2_mps(&mps);
		}
	}

	return 0;
}


char* test_su2_mps_split_tensor_svd()
{
	struct rng_state rng_state;
	seed_rng_state(61, &rng_state);

	// 'j' quantum numbers at each site
	qnumber site_j0list[] = { 1, 3 };
	qnumber site_j1list[] = { 0, 2 };
	const struct su2_irreducible_list site_irreps[2] = {
		{ .jlist = site_j0list, .num = ARRLEN(site_j0list) },
		{ .jlist = site_j1list, .num = ARRLEN(site_j1list) },
	};
	// corresponding degeneracy dimensions, indexed by 'j' quantum numbers
	//                              j:  0  1  2  3
	const ct_long site_dim_degen0[] = { 0, 4, 0, 7 };
	const ct_long site_dim_degen1[] = { 3, 0, 5    };
	const ct_long* site_dim_degen[] = {
		site_dim_degen0,
		site_dim_degen1,
	};

	// 'j' quantum numbers of the virtual bonds
	// outer (logical and auxiliary) 'j' quantum numbers
	qnumber bond_j0list[] = { 0, 2 };
	qnumber bond_j1list[] = { 1, 3, 5 };
	qnumber bond_j2list[] = { 1, 3 };
	const struct su2_irreducible_list bond_irreps[3] = {
		{ .jlist = bond_j0list, .num = ARRLEN(bond_j0list) },
		{ .jlist = bond_j1list, .num = ARRLEN(bond_j1list) },
		{ .jlist = bond_j2list, .num = ARRLEN(bond_j2list) },
	};
	// corresponding degeneracy dimensions, indexed by 'j' quantum numbers
	//                              j:  0  1  2  3  4  5
	const ct_long bond_dim_degen0[] = { 3, 0, 6          };
	const ct_long bond_dim_degen1[] = { 0, 8, 0, 2, 0, 9 };
	const ct_long bond_dim_degen2[] = { 0, 3, 0, 5       };
	const ct_long* bond_dim_degen[3] = {
		bond_dim_degen0,
		bond_dim_degen1,
		bond_dim_degen2,
	};

	// effectively unrestricted
	const ct_long max_vdim = 1352;

	struct su2_tensor a_init[2];
	for (int l = 0; l < 2; l++)
	{
		// construct the fuse and split tree
		//
		//                       2  right virtual bond
		//                       │
		//                       │   fuse
		//                       ╱╲  split
		//                      ╱  ╲
		//  left virtual bond  0    1  physical axis
		//
		struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2f = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j2s = { .i_ax = 2, .c = { &j0,  &j1  } };
		struct su2_fuse_split_tree tree = { .tree_fuse = &j2f, .tree_split = &j2s, .ndim = 3 };
		assert(su2_fuse_split_tree_is_consistent(&tree));

		// outer (logical and auxiliary) 'j' quantum numbers
		const struct su2_irreducible_list outer_irreps[3] = { bond_irreps[l], site_irreps[l], bond_irreps[l + 1] };
		// degeneracy dimensions, indexed by 'j' quantum numbers
		const ct_long* dim_degen[3] = { bond_dim_degen[l], site_dim_degen[l], bond_dim_degen[l + 1] };

		allocate_su2_tensor(CT_DOUBLE_COMPLEX, 3, 0, &tree, outer_irreps, dim_degen, &a_init[l]);
		su2_tensor_delete_charge_sector_by_index(&a_init[l], l + 1);
		assert(su2_tensor_is_consistent(&a_init[l]));

		assert(a_init[l].charge_sectors.nsec > 0);

		dcomplex alpha;
		numeric_from_double(0.1, a_init[l].dtype, &alpha);
		su2_tensor_fill_random_normal(&alpha, numeric_zero(a_init[l].dtype), &rng_state, &a_init[l]);
	}

	// merge MPS tensors
	struct su2_tensor a_pair;
	su2_mps_merge_tensor_pair(&a_init[0], &a_init[1], &a_pair);
	su2_tensor_delete_charge_sector_by_index(&a_pair, 4);
	if (!su2_tensor_is_consistent(&a_pair)) {
		return "internal consistency check for SU(2) tensor failed";
	}
	delete_su2_tensor(&a_init[0]);
	delete_su2_tensor(&a_init[1]);
	// convert to a dense tensor, as reference
	struct dense_tensor a_pair_dns;
	su2_to_dense_tensor(&a_pair, &a_pair_dns);

	// singular value truncation tolerance
	for (int t = 0; t < 2; t++)
	{
		const double tol = (t == 0 ? 0 : 0.1);

		// singular value distribution mode
		for (int d = 0; d < 2; d++)
		{
			const enum su2_singular_value_distr svd_distr = (d == 0 ? SU2_SVD_DISTR_LEFT : SU2_SVD_DISTR_RIGHT);

			struct su2_tensor a0, a1;
			struct trunc_info info;
			if (su2_mps_split_tensor_svd(&a_pair, site_irreps, site_dim_degen, tol, max_vdim, svd_distr, &a0, &a1, &info) < 0) {
				return "'su2_mps_split_tensor_svd' failed internally";
			}

			if (!su2_tensor_is_consistent(&a0)) {
				return "internal consistency check for SU(2) tensor failed";
			}
			if (!su2_tensor_is_consistent(&a1)) {
				return "internal consistency check for SU(2) tensor failed";
			}

			if (svd_distr == SU2_SVD_DISTR_RIGHT)
			{
				// 'a0' must be an isometry (after fusing the left virtual and physical axis)
				struct su2_tensor tmp;
				su2_tensor_fuse_axes_add_auxiliary(&a0, 0, 1, &tmp);
				if (!su2_tensor_is_isometry(&tmp, 1e-13, false)) {
					return "left tensor from SU(2) symmetric MPS splitting is not an isometry";
				}
				delete_su2_tensor(&tmp);
			}
			else
			{
				// 'a1' must be an isometry (after fusing the physical and right virtual axis)
				struct su2_tensor a1_rev;
				copy_su2_tensor(&a1, &a1_rev);
				su2_tensor_reverse_axis_simple(&a1_rev, 1);
				struct su2_tensor tmp;
				su2_tensor_fuse_axes_add_auxiliary(&a1_rev, 1, 2, &tmp);
				delete_su2_tensor(&a1_rev);
				if (!su2_tensor_is_isometry(&tmp, 1e-13, true)) {
					return "right tensor from SU(2) symmetric MPS splitting is not an isometry";
				}
				delete_su2_tensor(&tmp);
			}

			// merge tensors again
			struct su2_tensor a_mrg;
			su2_mps_merge_tensor_pair(&a0, &a1, &a_mrg);

			// merged tensor must agree with the original tensor;
			// comparing dense tensors since number of charge sectors can change
			if (!dense_su2_tensor_allclose(&a_pair_dns, &a_mrg, tol == 0 ? 1e-13 : tol)) {
				return "merged SU(2) symmetric MPS tensor after splitting does not match original pair tensor";
			}

			delete_su2_tensor(&a_mrg);
			delete_su2_tensor(&a0);
			delete_su2_tensor(&a1);
		}
	}

	delete_dense_tensor(&a_pair_dns);
	delete_su2_tensor(&a_pair);

	return 0;
}


char* test_su2_mps_to_statevector()
{
	// number of lattice sites
	const int nsites = 5;

	// 'j' quantum numbers at each site
	qnumber jlist[] = { 1, 3 };
	const struct su2_irreducible_list site_irreps = { .jlist = jlist, .num = ARRLEN(jlist) };

	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                             j:  0  1  2  3
	const ct_long site_dim_degen[] = { 0, 5, 0, 2 };

	struct rng_state rng_state;
	seed_rng_state(62, &rng_state);

	const qnumber irrep_sector = 1;
	const qnumber max_bond_irrep = 5;
	const ct_long max_bond_dim_degen = 11;

	struct su2_mps mps;
	construct_random_su2_mps(CT_SINGLE_COMPLEX, nsites, &site_irreps, site_dim_degen, irrep_sector, max_bond_irrep, max_bond_dim_degen, &rng_state, &mps);
	// rescale tensors such that overall state norm is around 1
	for (int l = 0; l < nsites; l++)
	{
		const float alpha = 14;
		rscale_su2_tensor(&alpha, &mps.a[l]);
	}
	if (!su2_mps_is_consistent(&mps)) {
		return "internal SU(2) MPS consistency check failed";
	}

	// convert to state vector
	struct su2_tensor vec;
	su2_mps_to_statevector(&mps, &vec);
	if (vec.ndim_logical != nsites + 2) {
		return "expecting the SU(2) statevector representation of an SU(2) MPS to have logical degree 'nsites + 2'";
	}
	struct dense_tensor vec_dns;
	su2_to_dense_tensor(&vec, &vec_dns);
	// include leading and trailing virtual bond dimensions
	const ct_long dim[3] = { 1, dense_tensor_num_elements(&vec_dns) / vec_dns.dim[vec_dns.ndim - 1], vec_dns.dim[vec_dns.ndim - 1] };
	reshape_dense_tensor(ARRLEN(dim), dim, &vec_dns);

	// logical physical dimension
	ct_long d = 0;
	for (int k = 0; k < site_irreps.num; k++)
	{
		const qnumber j = site_irreps.jlist[k];
		assert(site_dim_degen[j] > 0);
		d += site_dim_degen[j] * (j + 1);
	}
	// set all (additive) quantum numbers in reference tensors to zero
	qnumber* qsite = ct_calloc(d, sizeof(qnumber));

	// reference matrix product state
	struct mps mps_ref;
	allocate_empty_mps(nsites, d, qsite, &mps_ref);
	for (int l = 0; l < nsites; l++)
	{
		struct dense_tensor a_loc;
		su2_to_dense_tensor(&mps.a[l], &a_loc);
		if (a_loc.ndim != 3) {
			return "each local SU(2) MPS tensor must have degree 3";
		}
		if (a_loc.dim[1] != d) {
			return "physical dimension of local SU(2) MPS tensor does not agree with reference";
		}

		const enum tensor_axis_direction axis_dir[3] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN };

		qnumber* qbonds_left  = ct_calloc(a_loc.dim[0], sizeof(qnumber));
		qnumber* qbonds_right = ct_calloc(a_loc.dim[2], sizeof(qnumber));
		const qnumber* qnums[3] = { qbonds_left, qsite, qbonds_right };

		dense_to_block_sparse_tensor(&a_loc, axis_dir, qnums, &mps_ref.a[l]);

		ct_free(qbonds_right);
		ct_free(qbonds_left);

		delete_dense_tensor(&a_loc);
	}
	assert(mps_is_consistent(&mps_ref));

	// reference statevector
	struct block_sparse_tensor vec_ref;
	mps_to_statevector(&mps_ref, &vec_ref);
	if (block_sparse_tensor_norm2(&vec_ref) == 0) {
		return "expecting a non-zero reference statevector for the random SU(2) MPS";
	}

	// compare
	if (!dense_block_sparse_tensor_allclose(&vec_dns, &vec_ref, 1e-5)) {
		return "state vector obtained from SU(2) MPS does not match reference";
	}

	delete_block_sparse_tensor(&vec_ref);
	delete_mps(&mps_ref);
	ct_free(qsite);
	delete_dense_tensor(&vec_dns);
	delete_su2_tensor(&vec);
	delete_su2_mps(&mps);

	return 0;
}
