#include <math.h>
#include <complex.h>
#include "su2_chain_ops.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


char* test_su2_contraction_operator_step_left()
{
	// physical site
	// 'j' quantum numbers
	qnumber site_jlist[] = { 1, 3 };
	const struct su2_irreducible_list site_irreps = { .jlist = site_jlist, .num = ARRLEN(site_jlist) };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                             j:  0  1  2  3
	const ct_long site_dim_degen[] = { 0, 3, 0, 4 };

	// virtual bond between 'a' and 'l'
	// 'j' quantum numbers
	qnumber bond_al_jlist[] = { 0, 4 };
	const struct su2_irreducible_list bond_al_irreps = { .jlist = bond_al_jlist, .num = ARRLEN(bond_al_jlist) };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                                j:  0  1  2  3  4
	const ct_long bond_al_dim_degen[] = { 7, 0, 0, 0, 3 };

	// virtual bond between 'b' and 'l'
	// 'j' quantum numbers
	qnumber bond_bl_jlist[] = { 0, 2 };
	const struct su2_irreducible_list bond_bl_irreps = { .jlist = bond_bl_jlist, .num = ARRLEN(bond_bl_jlist) };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                                j:  0  1  2
	const ct_long bond_bl_dim_degen[] = { 5, 0, 8 };

	// virtual bond between 'w' and 'l'
	// 'j' quantum numbers
	qnumber bond_wl_jlist[] = { 1, 3 };
	const struct su2_irreducible_list bond_wl_irreps = { .jlist = bond_wl_jlist, .num = ARRLEN(bond_wl_jlist) };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                                j:  0  1  2  3
	const ct_long bond_wl_dim_degen[] = { 0, 4, 0, 5 };

	const enum numeric_type dtype = CT_DOUBLE_COMPLEX;

	struct rng_state rng_state;
	seed_rng_state(72, &rng_state);

	struct su2_tensor a;
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

		// right virtual bond
		// 'j' quantum numbers
		qnumber right_bond_jlist[] = { 3 };
		const struct su2_irreducible_list right_bond_irreps = { .jlist = right_bond_jlist, .num = ARRLEN(right_bond_jlist) };
		// degeneracy dimensions, indexed by 'j' quantum numbers
		//                                   j:  0  1  2  3
		const ct_long right_bond_dim_degen[] = { 0, 0, 0, 4 };

		// outer (logical and auxiliary) 'j' quantum numbers
		const struct su2_irreducible_list outer_irreps[3] = { bond_al_irreps, site_irreps, right_bond_irreps };
		// degeneracy dimensions, indexed by 'j' quantum numbers
		const ct_long* dim_degen[3] = { bond_al_dim_degen, site_dim_degen, right_bond_dim_degen };

		allocate_su2_tensor(dtype, 3, 0, &tree, outer_irreps, dim_degen, &a);
		assert(su2_tensor_is_consistent(&a));
		assert(a.charge_sectors.nsec > 0);

		// fill degeneracy tensors with random entries
		const dcomplex scale = 0.1;
		su2_tensor_fill_random_normal(&scale, numeric_zero(dtype), &rng_state, &a);
	}

	struct su2_tensor b;
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

		// right virtual bond
		// 'j' quantum numbers
		qnumber right_bond_jlist[] = { 1 };
		const struct su2_irreducible_list right_bond_irreps = { .jlist = right_bond_jlist, .num = ARRLEN(right_bond_jlist) };
		// degeneracy dimensions, indexed by 'j' quantum numbers
		//                                   j:  0  1
		const ct_long right_bond_dim_degen[] = { 0, 7 };

		// outer (logical and auxiliary) 'j' quantum numbers
		const struct su2_irreducible_list outer_irreps[3] = { bond_bl_irreps, site_irreps, right_bond_irreps };
		// degeneracy dimensions, indexed by 'j' quantum numbers
		const ct_long* dim_degen[3] = { bond_bl_dim_degen, site_dim_degen, right_bond_dim_degen };

		allocate_su2_tensor(dtype, 3, 0, &tree, outer_irreps, dim_degen, &b);
		assert(su2_tensor_is_consistent(&b));
		assert(b.charge_sectors.nsec > 0);

		// fill degeneracy tensors with random entries
		const dcomplex scale = 0.1;
		su2_tensor_fill_random_normal(&scale, numeric_zero(dtype), &rng_state, &b);
	}

	struct su2_tensor w;
	{
		// construct the fuse and split tree
		//
		//  physical input axis  2    3  right virtual bond
		//                        ╲  ╱
		//                         ╲╱   fuse
		//                         │
		//                         │4
		//                         │
		//                         ╱╲   split
		//                        ╱  ╲
		//    left virtual bond  0    1  physical output axis
		//
		struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
		struct su2_tree_node j4f = { .i_ax = 4, .c = { &j2,  &j3  } };
		struct su2_tree_node j4s = { .i_ax = 4, .c = { &j0,  &j1  } };
		struct su2_fuse_split_tree tree = { .tree_fuse = &j4f, .tree_split = &j4s, .ndim = 5 };
		assert(su2_fuse_split_tree_is_consistent(&tree));

		// right virtual bond
		// 'j' quantum numbers
		qnumber right_bond_jlist[] = { 1, 5 };
		const struct su2_irreducible_list right_bond_irreps = { .jlist = right_bond_jlist, .num = ARRLEN(right_bond_jlist) };
		// degeneracy dimensions, indexed by 'j' quantum numbers
		//                                   j:  0  1  2  3  4  5
		const ct_long right_bond_dim_degen[] = { 0, 4, 0, 0, 0, 3 };

		// outer (logical and auxiliary) 'j' quantum numbers
		const struct su2_irreducible_list outer_irreps[4] = { bond_wl_irreps, site_irreps, site_irreps, right_bond_irreps };
		// degeneracy dimensions, indexed by 'j' quantum numbers
		const ct_long* dim_degen[4] = { bond_wl_dim_degen, site_dim_degen, site_dim_degen, right_bond_dim_degen };

		allocate_su2_tensor(dtype, 4, 0, &tree, outer_irreps, dim_degen, &w);
		assert(su2_tensor_is_consistent(&w));
		assert(w.charge_sectors.nsec > 0);

		// fill degeneracy tensors with random entries
		const dcomplex scale = 0.1;
		su2_tensor_fill_random_normal(&scale, numeric_zero(dtype), &rng_state, &w);
	}

	struct su2_tensor l;
	{
		// construct the fuse and split tree
		//
		//  virtual bond with 'a'  1    2  virtual bond with 'w'
		//                          ╲  ╱
		//                           ╲╱   fuse
		//                           │
		//                           │4
		//                           │
		//                           ╱╲   split
		//                          ╱  ╲
		//      left virtual bond  0    3  virtual bond with adjoint of 'b'
		//
		struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
		struct su2_tree_node j4f = { .i_ax = 4, .c = { &j1,  &j2  } };
		struct su2_tree_node j4s = { .i_ax = 4, .c = { &j0,  &j3  } };
		struct su2_fuse_split_tree tree = { .tree_fuse = &j4f, .tree_split = &j4s, .ndim = 5 };
		assert(su2_fuse_split_tree_is_consistent(&tree));

		// left virtual bond
		// 'j' quantum numbers
		qnumber left_bond_jlist[] = { 1, 3 };
		const struct su2_irreducible_list left_bond_irreps = { .jlist = left_bond_jlist, .num = ARRLEN(left_bond_jlist) };
		// degeneracy dimensions, indexed by 'j' quantum numbers
		//                                  j:  0  1  2  3
		const ct_long left_bond_dim_degen[] = { 0, 2, 0, 3 };

		// outer (logical and auxiliary) 'j' quantum numbers
		const struct su2_irreducible_list outer_irreps[4] = { left_bond_irreps, bond_al_irreps, bond_wl_irreps, bond_bl_irreps };
		// degeneracy dimensions, indexed by 'j' quantum numbers
		const ct_long* dim_degen[4] = { left_bond_dim_degen, bond_al_dim_degen, bond_wl_dim_degen, bond_bl_dim_degen };

		allocate_su2_tensor(dtype, 4, 0, &tree, outer_irreps, dim_degen, &l);
		assert(su2_tensor_is_consistent(&l));
		assert(l.charge_sectors.nsec > 0);

		// fill degeneracy tensors with random entries
		const dcomplex scale = 0.1;
		su2_tensor_fill_random_normal(&scale, numeric_zero(dtype), &rng_state, &l);
	}

	struct su2_tensor l_next;
	su2_contraction_operator_step_left(&a, &b, &w, &l, &l_next);

	if (!su2_tensor_is_consistent(&l_next)) {
		return "internal consistency check for SU(2) tensor failed";
	}
	if (l_next.charge_sectors.nsec == 0) {
		return "expecting at least one charge sector in SU(2) tensor";
	}
	if (l_next.ndim_logical != 4 || l_next.ndim_auxiliary != 0) {
		return "SU(2) symmetric tensor resulting from block contraction from left to right has incorrect degree";
	}
	if (l_next.tree.tree_split->c[0]->i_ax != 0 ||
	     l_next.tree.tree_fuse->c[0]->i_ax != 1 ||
	     l_next.tree.tree_fuse->c[1]->i_ax != 2 ||
	    l_next.tree.tree_split->c[1]->i_ax != 3) {
		return "incorrect fusion-splitting tree of SU(2) symmetric tensor resulting from block contraction from left to right";
	}

	// reference calculation
	struct dense_tensor l_next_ref;
	{
		struct dense_tensor l_dns;
		su2_to_dense_tensor(&l, &l_dns);
		assert(dense_tensor_norm2(&l_dns) > 0);

		// multiply with conjugated 'b' tensor
		struct dense_tensor b_dns;
		su2_to_dense_tensor(&b, &b_dns);
		conjugate_dense_tensor(&b_dns);
		assert(dense_tensor_norm2(&b_dns) > 0);
		struct dense_tensor s;
		dense_tensor_dot(&l_dns, TENSOR_AXIS_RANGE_TRAILING, &b_dns, TENSOR_AXIS_RANGE_LEADING, 1, &s);
		delete_dense_tensor(&b_dns);
		delete_dense_tensor(&l_dns);

		// multiply with 'w' tensor
		// re-order last three dimensions
		const int perm0[5] = { 0, 1, 4, 2, 3 };
		struct dense_tensor t;
		dense_tensor_transpose(perm0, &s, &t);
		struct dense_tensor w_dns;
		su2_to_dense_tensor(&w, &w_dns);
		assert(dense_tensor_norm2(&w_dns) > 0);
		delete_dense_tensor(&s);
		dense_tensor_dot(&t, TENSOR_AXIS_RANGE_TRAILING, &w_dns, TENSOR_AXIS_RANGE_LEADING, 2, &s);
		delete_dense_tensor(&t);
		delete_dense_tensor(&w_dns);

		// multiply with 'a' tensor
		// undo re-ordering
		const int perm1[5] = { 0, 1, 3, 4, 2 };
		dense_tensor_transpose(perm1, &s, &t);
		delete_dense_tensor(&s);
		struct dense_tensor a_dns;
		su2_to_dense_tensor(&a, &a_dns);
		assert(dense_tensor_norm2(&a_dns) > 0);
		// group leading two axes of 'a'
		const ct_long dim_a_mat[2] = { a_dns.dim[0] * a_dns.dim[1], a_dns.dim[2] };
		reshape_dense_tensor(2, dim_a_mat, &a_dns);
		// group axes 1 and 2 of 't'
		const ct_long dim_t[4] = { t.dim[0], t.dim[1] * t.dim[2], t.dim[3], t.dim[4] };
		reshape_dense_tensor(4, dim_t, &t);
		dense_tensor_multiply_axis(&t, 1, &a_dns, TENSOR_AXIS_RANGE_LEADING, &l_next_ref);
		delete_dense_tensor(&t);
		delete_dense_tensor(&a_dns);

		assert(dense_tensor_norm2(&l_next_ref) > 0);
	}

	// compare
	if (!dense_su2_tensor_allclose(&l_next_ref, &l_next, 1e-13)) {
		return "SU(2) symmetric tensor resulting from block contraction from left to right does not match reference tensor";
	}

	delete_dense_tensor(&l_next_ref);
	delete_su2_tensor(&l_next);
	delete_su2_tensor(&l);
	delete_su2_tensor(&w);
	delete_su2_tensor(&b);
	delete_su2_tensor(&a);

	return 0;
}


char* test_su2_mpo_inner_product()
{
	// number of lattice sites
	const int nsites = 5;

	// 'j' quantum numbers at each site
	qnumber jlist[] = { 1 };
	const struct su2_irreducible_list site_irreps = { .jlist = jlist, .num = ARRLEN(jlist) };

	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                             j:  0  1
	const ct_long site_dim_degen[] = { 0, 2 };

	struct rng_state rng_state;
	seed_rng_state(73, &rng_state);

	struct su2_mps psi;
	{
		const qnumber irrep_sector = 1;
		const qnumber max_bond_irrep = 5;
		const ct_long max_bond_dim_degen = 4;

		construct_random_su2_mps(CT_SINGLE_COMPLEX, nsites, &site_irreps, site_dim_degen, irrep_sector, max_bond_irrep, max_bond_dim_degen, &rng_state, &psi);
		// rescale tensors such that overall state norm is around 1
		for (int l = 0; l < nsites; l++)
		{
			const float alpha = 6;
			rscale_su2_tensor(&alpha, &psi.a[l]);
		}
		if (!su2_mps_is_consistent(&psi)) {
			return "internal SU(2) MPS consistency check failed";
		}
	}

	struct su2_mps chi;
	{
		const qnumber irrep_sector = 1;
		const qnumber max_bond_irrep = 3;
		const ct_long max_bond_dim_degen = 7;

		construct_random_su2_mps(CT_SINGLE_COMPLEX, nsites, &site_irreps, site_dim_degen, irrep_sector, max_bond_irrep, max_bond_dim_degen, &rng_state, &chi);
		// rescale tensors such that overall state norm is around 1
		for (int l = 0; l < nsites; l++)
		{
			const float alpha = 6;
			rscale_su2_tensor(&alpha, &chi.a[l]);
		}
		if (!su2_mps_is_consistent(&chi)) {
			return "internal SU(2) MPS consistency check failed";
		}
	}

	struct su2_mpo op;
	{
		const qnumber max_bond_irrep = 5;
		const ct_long max_bond_dim_degen = 3;

		construct_random_su2_mpo(CT_SINGLE_COMPLEX, nsites, &site_irreps, site_dim_degen, max_bond_irrep, max_bond_dim_degen, &rng_state, &op);
		// rescale tensors such that overall norm is around 1
		for (int l = 0; l < nsites; l++)
		{
			const float alpha = 10;
			rscale_su2_tensor(&alpha, &op.a[l]);
		}
		if (!su2_mpo_is_consistent(&op)) {
			return "internal SU(2) MPO consistency check failed";
		}
	}

	// compute operator inner product <chi | op | psi>
	scomplex s;
	su2_mpo_inner_product(&chi, &op, &psi, &s);

	// reference calculation

	// logical physical dimension
	ct_long d = 0;
	for (int k = 0; k < site_irreps.num; k++)
	{
		const qnumber j = site_irreps.jlist[k];
		assert(site_dim_degen[j] > 0);
		d += site_dim_degen[j] * (j + 1);
	}

	struct dense_tensor psi_vec;
	{
		struct su2_tensor psi_su2_vec;
		su2_mps_to_statevector(&psi, &psi_su2_vec);
		su2_to_dense_tensor(&psi_su2_vec, &psi_vec);
		delete_su2_tensor(&psi_su2_vec);
		// keep trailing virtual bond as separate dimension
		const ct_long dim[2] = { integer_product(psi_vec.dim, psi_vec.ndim - 1), psi_vec.dim[psi_vec.ndim - 1] };
		reshape_dense_tensor(2, dim, &psi_vec);
	}
	struct dense_tensor chi_vec;
	{
		struct su2_tensor chi_su2_vec;
		su2_mps_to_statevector(&chi, &chi_su2_vec);
		su2_to_dense_tensor(&chi_su2_vec, &chi_vec);
		delete_su2_tensor(&chi_su2_vec);
		// keep trailing virtual bond as separate dimension
		const ct_long dim[2] = { integer_product(chi_vec.dim, chi_vec.ndim - 1), chi_vec.dim[chi_vec.ndim - 1] };
		reshape_dense_tensor(2, dim, &chi_vec);
	}
	struct dense_tensor op_mat;
	{
		struct su2_tensor op_tensor;
		su2_mpo_to_tensor(&op, &op_tensor);
		su2_to_dense_tensor(&op_tensor, &op_mat);
		delete_su2_tensor(&op_tensor);
		// reshape into a matrix
		const ct_long hspace_dim = ipow(d, nsites);
		const ct_long dim[2] = { hspace_dim, hspace_dim };
		reshape_dense_tensor(2, dim, &op_mat);
	}

	// reference <chi | op | psi>
	scomplex s_ref;
	{
		struct dense_tensor t;
		dense_tensor_dot(&op_mat, TENSOR_AXIS_RANGE_TRAILING, &psi_vec, TENSOR_AXIS_RANGE_LEADING, 1, &t);
		assert(t.ndim == 2);  // combined physical axis and trailing virtual bond axis of 'psi'
		conjugate_dense_tensor(&chi_vec);
		struct dense_tensor r;
		dense_tensor_dot(&chi_vec, TENSOR_AXIS_RANGE_LEADING, &t, TENSOR_AXIS_RANGE_LEADING, 1, &r);
		delete_dense_tensor(&t);
		assert(r.dtype == CT_SINGLE_COMPLEX);
		assert(r.ndim == 2);  // trailing virtual bond axes of 'chi' and 'psi'
		// trace out trailing virtual bonds
		dense_tensor_trace(&r, &s_ref);
		delete_dense_tensor(&r);
	}

	// compare
	if (cabsf(s - s_ref) / cabsf(s_ref) > 5e-6) {
		return "SU(2) operator inner product does not match reference value";
	}

	delete_dense_tensor(&op_mat);
	delete_dense_tensor(&chi_vec);
	delete_dense_tensor(&psi_vec);
	delete_su2_mpo(&op);
	delete_su2_mps(&chi);
	delete_su2_mps(&psi);

	return 0;
}


char* test_su2_apply_local_hamiltonian()
{
	// physical site
	// 'j' quantum numbers
	qnumber site_jlist[] = { 0, 2 };
	const struct su2_irreducible_list site_irreps = { .jlist = site_jlist, .num = ARRLEN(site_jlist) };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                             j:  0  1  2
	const ct_long site_dim_degen[] = { 5, 0, 3 };

	// virtual bond between 'a' and 'l'
	// 'j' quantum numbers
	qnumber bond_al_jlist[] = { 1, 3 };
	const struct su2_irreducible_list bond_al_irreps = { .jlist = bond_al_jlist, .num = ARRLEN(bond_al_jlist) };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                                j:  0  1  2  3
	const ct_long bond_al_dim_degen[] = { 0, 7, 0, 4 };

	// virtual bond between 'a' and 'r'
	// 'j' quantum numbers
	qnumber bond_ar_jlist[] = { 3 };
	const struct su2_irreducible_list bond_ar_irreps = { .jlist = bond_ar_jlist, .num = ARRLEN(bond_ar_jlist) };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                                j:  0  1  2  3
	const ct_long bond_ar_dim_degen[] = { 0, 0, 0, 2 };

	// virtual bond between 'w' and 'l'
	// 'j' quantum numbers
	qnumber bond_wl_jlist[] = { 0, 4 };
	const struct su2_irreducible_list bond_wl_irreps = { .jlist = bond_wl_jlist, .num = ARRLEN(bond_wl_jlist) };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                                j:  0  1  2  3  4
	const ct_long bond_wl_dim_degen[] = { 2, 0, 0, 0, 1 };

	// virtual bond between 'w' and 'r'
	// 'j' quantum numbers
	qnumber bond_wr_jlist[] = { 0, 2 };
	const struct su2_irreducible_list bond_wr_irreps = { .jlist = bond_wr_jlist, .num = ARRLEN(bond_wr_jlist) };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                                j:  0  1  2
	const ct_long bond_wr_dim_degen[] = { 5, 0, 6 };

	// virtual bond between 'b' and 'l'
	// 'j' quantum numbers
	qnumber bond_bl_jlist[] = { 1, 3 };
	const struct su2_irreducible_list bond_bl_irreps = { .jlist = bond_bl_jlist, .num = ARRLEN(bond_bl_jlist) };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                                j:  0  1  2  3
	const ct_long bond_bl_dim_degen[] = { 0, 5, 0, 4 };

	// virtual bond between 'b' and 'r'
	// 'j' quantum numbers
	qnumber bond_br_jlist[] = { 1 };
	const struct su2_irreducible_list bond_br_irreps = { .jlist = bond_br_jlist, .num = ARRLEN(bond_br_jlist) };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                                j:  0  1  2  3  4  5
	const ct_long bond_br_dim_degen[] = { 0, 3 };

	// outer (cyclic) virtual bond between 'l' and 'r'
	// 'j' quantum numbers
	qnumber outer_bond_jlist[] = { 0, 2 };
	const struct su2_irreducible_list outer_bond_irreps = { .jlist = outer_bond_jlist, .num = ARRLEN(outer_bond_jlist) };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                                   j:  0  1  2
	const ct_long outer_bond_dim_degen[] = { 7, 0, 4 };

	const enum numeric_type dtype = CT_DOUBLE_COMPLEX;

	struct rng_state rng_state;
	seed_rng_state(74, &rng_state);

	struct su2_tensor a;
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
		const struct su2_irreducible_list outer_irreps[3] = { bond_al_irreps, site_irreps, bond_ar_irreps };
		// degeneracy dimensions, indexed by 'j' quantum numbers
		const ct_long* dim_degen[3] = { bond_al_dim_degen, site_dim_degen, bond_ar_dim_degen };

		allocate_su2_tensor(dtype, 3, 0, &tree, outer_irreps, dim_degen, &a);
		assert(su2_tensor_is_consistent(&a));
		assert(a.charge_sectors.nsec > 0);

		// fill degeneracy tensors with random entries
		const dcomplex scale = 0.1;
		su2_tensor_fill_random_normal(&scale, numeric_zero(dtype), &rng_state, &a);
	}

	struct su2_tensor w;
	{
		// construct the fuse and split tree
		//
		//  physical input axis  2    3  right virtual bond
		//                        ╲  ╱
		//                         ╲╱   fuse
		//                         │
		//                         │4
		//                         │
		//                         ╱╲   split
		//                        ╱  ╲
		//    left virtual bond  0    1  physical output axis
		//
		struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
		struct su2_tree_node j4f = { .i_ax = 4, .c = { &j2,  &j3  } };
		struct su2_tree_node j4s = { .i_ax = 4, .c = { &j0,  &j1  } };
		struct su2_fuse_split_tree tree = { .tree_fuse = &j4f, .tree_split = &j4s, .ndim = 5 };
		assert(su2_fuse_split_tree_is_consistent(&tree));

		// outer (logical and auxiliary) 'j' quantum numbers
		const struct su2_irreducible_list outer_irreps[4] = { bond_wl_irreps, site_irreps, site_irreps, bond_wr_irreps };
		// degeneracy dimensions, indexed by 'j' quantum numbers
		const ct_long* dim_degen[4] = { bond_wl_dim_degen, site_dim_degen, site_dim_degen, bond_wr_dim_degen };

		allocate_su2_tensor(dtype, 4, 0, &tree, outer_irreps, dim_degen, &w);
		assert(su2_tensor_is_consistent(&w));
		assert(w.charge_sectors.nsec > 0);

		// fill degeneracy tensors with random entries
		const dcomplex scale = 0.1;
		su2_tensor_fill_random_normal(&scale, numeric_zero(dtype), &rng_state, &w);
	}

	struct su2_tensor l;
	{
		// construct the fuse and split tree
		//
		//  virtual bond with 'a'  1    2  virtual bond with 'w'
		//                          ╲  ╱
		//                           ╲╱   fuse
		//                           │
		//                           │4
		//                           │
		//                           ╱╲   split
		//                          ╱  ╲
		//      left virtual bond  0    3  virtual bond with adjoint of 'b'
		//
		struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
		struct su2_tree_node j4f = { .i_ax = 4, .c = { &j1,  &j2  } };
		struct su2_tree_node j4s = { .i_ax = 4, .c = { &j0,  &j3  } };
		struct su2_fuse_split_tree tree = { .tree_fuse = &j4f, .tree_split = &j4s, .ndim = 5 };
		assert(su2_fuse_split_tree_is_consistent(&tree));

		// outer (logical and auxiliary) 'j' quantum numbers
		const struct su2_irreducible_list outer_irreps[4] = { outer_bond_irreps, bond_al_irreps, bond_wl_irreps, bond_bl_irreps };
		// degeneracy dimensions, indexed by 'j' quantum numbers
		const ct_long* dim_degen[4] = { outer_bond_dim_degen, bond_al_dim_degen, bond_wl_dim_degen, bond_bl_dim_degen };

		allocate_su2_tensor(dtype, 4, 0, &tree, outer_irreps, dim_degen, &l);
		assert(su2_tensor_is_consistent(&l));
		assert(l.charge_sectors.nsec > 0);

		// fill degeneracy tensors with random entries
		const dcomplex scale = 0.1;
		su2_tensor_fill_random_normal(&scale, numeric_zero(dtype), &rng_state, &l);
	}

	struct su2_tensor r;
	{
		// construct the fuse and split tree
		//
		//  virtual bond with adjoint of 'b'  2    3  right virtual bond
		//                                     ╲  ╱
		//                                      ╲╱   fuse
		//                                      │
		//                                      │4
		//                                      │
		//                                      ╱╲   split
		//                                     ╱  ╲
		//             virtual bond with 'a'  0    1  virtual bond with 'w'
		//
		struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
		struct su2_tree_node j4f = { .i_ax = 4, .c = { &j2,  &j3  } };
		struct su2_tree_node j4s = { .i_ax = 4, .c = { &j0,  &j1  } };
		struct su2_fuse_split_tree tree = { .tree_fuse = &j4f, .tree_split = &j4s, .ndim = 5 };
		assert(su2_fuse_split_tree_is_consistent(&tree));

		// outer (logical and auxiliary) 'j' quantum numbers
		const struct su2_irreducible_list outer_irreps[4] = { bond_ar_irreps, bond_wr_irreps, bond_br_irreps, outer_bond_irreps };
		// degeneracy dimensions, indexed by 'j' quantum numbers
		const ct_long* dim_degen[4] = { bond_ar_dim_degen, bond_wr_dim_degen, bond_br_dim_degen, outer_bond_dim_degen };

		allocate_su2_tensor(dtype, 4, 0, &tree, outer_irreps, dim_degen, &r);
		assert(su2_tensor_is_consistent(&r));
		assert(r.charge_sectors.nsec > 0);

		// fill degeneracy tensors with random entries
		const dcomplex scale = 0.1;
		su2_tensor_fill_random_normal(&scale, numeric_zero(dtype), &rng_state, &r);
	}

	struct su2_tensor b;
	su2_apply_local_hamiltonian(&a, &w, &l, &r, &b);

	if (!su2_tensor_is_consistent(&b)) {
		return "internal consistency check for SU(2) tensor failed";
	}
	if (b.ndim_logical != 3 || b.ndim_auxiliary != 0) {
		return "SU(2) tensor resulting from applying a local Hamiltonian operator should have three logical axes and no auxiliary axis";
	}
	if (b.tree.tree_split->c[0]->i_ax != 0 || b.tree.tree_split->c[1]->i_ax != 1 || b.tree.tree_fuse->i_ax != 2) {
		return "SU(2) tensor resulting from applying a local Hamiltonian operator has incorrect tree topology";
	}

	// reference calculation
	struct dense_tensor b_ref;
	{
		struct dense_tensor r_dns;
		su2_to_dense_tensor(&r, &r_dns);
		assert(dense_tensor_norm2(&r_dns) > 0);

		// multiply with 'a' tensor
		struct dense_tensor s;
		struct dense_tensor a_dns;
		su2_to_dense_tensor(&a, &a_dns);
		assert(dense_tensor_norm2(&a_dns) > 0);
		dense_tensor_dot(&a_dns, TENSOR_AXIS_RANGE_TRAILING, &r_dns, TENSOR_AXIS_RANGE_LEADING, 1, &s);
		delete_dense_tensor(&a_dns);
		delete_dense_tensor(&r_dns);

		// multiply with 'w' tensor
		// re-order first three dimensions
		const int perm0[5] = { 1, 2, 0, 3, 4 };
		struct dense_tensor t;
		dense_tensor_transpose(perm0, &s, &t);
		delete_dense_tensor(&s);
		struct dense_tensor w_dns;
		su2_to_dense_tensor(&w, &w_dns);
		assert(dense_tensor_norm2(&w_dns) > 0);
		dense_tensor_dot(&w_dns, TENSOR_AXIS_RANGE_TRAILING, &t, TENSOR_AXIS_RANGE_LEADING, 2, &s);
		delete_dense_tensor(&t);
		delete_dense_tensor(&w_dns);
		// undo re-ordering
		const int perm1[5] = { 2, 0, 1, 3, 4 };
		dense_tensor_transpose(perm1, &s, &t);
		delete_dense_tensor(&s);

		// multiply with 'l' tensor
		struct dense_tensor l_dns;
		su2_to_dense_tensor(&l, &l_dns);
		assert(dense_tensor_norm2(&l_dns) > 0);
		// re-order last three dimensions
		const int perm2[4] = { 0, 3, 1, 2 };
		struct dense_tensor k;
		dense_tensor_transpose(perm2, &l_dns, &k);
		delete_dense_tensor(&l_dns);
		dense_tensor_dot(&k, TENSOR_AXIS_RANGE_TRAILING, &t, TENSOR_AXIS_RANGE_LEADING, 2, &s);
		delete_dense_tensor(&k);
		delete_dense_tensor(&t);

		// trace out outer virtual bonds (assumed to be low-dimensional)
		dense_tensor_cyclic_partial_trace(&s, 1, &b_ref);
		delete_dense_tensor(&s);
	}

	// compare
	if (!dense_su2_tensor_allclose(&b_ref, &b, 1e-13)) {
		return "SU(2) symmetric tensor resulting from applying a local Hamiltonian operator does not match reference tensor";
	}

	delete_dense_tensor(&b_ref);
	delete_su2_tensor(&b);
	delete_su2_tensor(&r);
	delete_su2_tensor(&l);
	delete_su2_tensor(&w);
	delete_su2_tensor(&a);

	return 0;
}
