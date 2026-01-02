#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>
#include "su2_tensor.h"
#include "aligned_memory.h"
#include "rng.h"
#include "hdf5_util.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


char* test_su2_tensor_swap_tree_axes()
{
	for (int ndim_auxiliary = 0; ndim_auxiliary < 2; ndim_auxiliary++)
	{
		// construct an SU(2) tensor 't'
		struct su2_tensor t;
		{
			const int ndim = 7;

			// construct the fuse and split tree
			//
			//    1    0
			//     ╲  ╱        fuse
			//      ╲╱
			//      │
			//      │6
			//      │
			//      ╱╲
			//     ╱  ╲5
			//    ╱   ╱╲       split
			//   ╱   ╱  ╲
			//  2   4    3
			//
			struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
			struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
			struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
			struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
			struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
			struct su2_tree_node j5  = { .i_ax = 5, .c = { &j4,  &j3  } };
			struct su2_tree_node j6f = { .i_ax = 6, .c = { &j1,  &j0  } };
			struct su2_tree_node j6s = { .i_ax = 6, .c = { &j2,  &j5  } };

			struct su2_fuse_split_tree tree = { .tree_fuse = &j6f, .tree_split = &j6s, .ndim = ndim };

			if (!su2_fuse_split_tree_is_consistent(&tree)) {
				return "internal consistency check for the fuse and split tree failed";
			}

			// outer (logical and auxiliary) 'j' quantum numbers
			qnumber j0list[] = { 2 };
			qnumber j1list[] = { 1, 3 };
			qnumber j2list[] = { 2 };
			qnumber j3list[] = { 0, 2 };
			qnumber j4list[] = { 1, 3 };  // auxiliary if ndim_auxiliary == 1
			const struct su2_irreducible_list outer_irreps[5] = {
				{ .jlist = j0list, .num = ARRLEN(j0list) },
				{ .jlist = j1list, .num = ARRLEN(j1list) },
				{ .jlist = j2list, .num = ARRLEN(j2list) },
				{ .jlist = j3list, .num = ARRLEN(j3list) },
				{ .jlist = j4list, .num = ARRLEN(j4list) },
			};

			// degeneracy dimensions, indexed by 'j' quantum numbers
			//                         j:  0  1  2  3
			const ct_long dim_degen0[] = { 0, 0, 3    };
			const ct_long dim_degen1[] = { 0, 7, 0, 2 };
			const ct_long dim_degen2[] = { 0, 0, 5    };
			const ct_long dim_degen3[] = { 9, 0, 4    };
			const ct_long dim_degen4[] = { 0, 6, 0, 3 };  // only used if ndim_auxiliary == 0
			const ct_long* dim_degen[] = {
				dim_degen0,
				dim_degen1,
				dim_degen2,
				dim_degen3,
				dim_degen4,
			};

			const int ndim_logical = 5 - ndim_auxiliary;

			allocate_su2_tensor(CT_SINGLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &t);

			// delete some charge sectors
			su2_tensor_delete_charge_sector_by_index(&t, 2);
			su2_tensor_delete_charge_sector_by_index(&t, 5);

			if (!su2_tensor_is_consistent(&t)) {
				return "internal consistency check for SU(2) tensor failed";
			}
			if (t.charge_sectors.nsec == 0) {
				return "expecting at least one charge sector in SU(2) tensor";
			}

			// fill degeneracy tensors with random entries
			struct rng_state rng_state;
			seed_rng_state(35, &rng_state);
			su2_tensor_fill_random_normal(numeric_one(t.dtype), numeric_zero(t.dtype), &rng_state, &t);
		}

		// dense tensor representation before swap
		struct dense_tensor t_dns_ref;
		su2_to_dense_tensor(&t, &t_dns_ref);

		// swap 3 <-> 4 in tree
		su2_tensor_swap_tree_axes(&t, 3, 4);
		if (!su2_tensor_is_consistent(&t)) {
			return "internal consistency check for SU(2) tensor failed";
		}

		const struct su2_tree_node* pnode = t.tree.tree_split->c[1];
		if (pnode->c[0]->i_ax != 3 || pnode->c[1]->i_ax != 4) {
			return "incorrect axes indices in SU(2) tree after axes swap";
		}

		// dense tensor representation after swap
		struct dense_tensor t_dns;
		su2_to_dense_tensor(&t, &t_dns);

		// compare
		if (!dense_tensor_allclose(&t_dns, &t_dns_ref, 1e-6)) {
			return "dense tensor representations of SU(2) symmetric tensor before and after axes swap do not match";
		}

		delete_dense_tensor(&t_dns);
		delete_dense_tensor(&t_dns_ref);
		delete_su2_tensor(&t);
	}

	return 0;
}


char* test_su2_tensor_add_auxiliary_axis()
{
	for (int m = 0; m < 2; m++)  // whether to insert the auxiliary axis on the left or right
	{
		// construct an SU(2) tensor 't'
		struct su2_tensor t;
		{
			// construct the fuse and split tree
			//
			//  2    1
			//   ╲  ╱     fuse
			//    ╲╱
			//    │
			//    │4
			//    │
			//    ╱╲
			//   ╱  ╲     split
			//  0    3
			//
			struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
			struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
			struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
			struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
			struct su2_tree_node j4f = { .i_ax = 4, .c = { &j2,  &j1  } };
			struct su2_tree_node j4s = { .i_ax = 4, .c = { &j0,  &j3  } };

			struct su2_fuse_split_tree tree = { .tree_fuse = &j4f, .tree_split = &j4s, .ndim = 5 };
			if (!su2_fuse_split_tree_is_consistent(&tree)) {
				return "internal consistency check for the fuse and split tree failed";
			}

			// outer (logical and auxiliary) 'j' quantum numbers
			qnumber j0list[] = { 0, 4 };
			qnumber j1list[] = { 1, 3 };
			qnumber j2list[] = { 0, 2 };
			qnumber j3list[] = { 1 };  // auxiliary
			const struct su2_irreducible_list outer_irreps[4] = {
				{ .jlist = j0list, .num = ARRLEN(j0list) },
				{ .jlist = j1list, .num = ARRLEN(j1list) },
				{ .jlist = j2list, .num = ARRLEN(j2list) },
				{ .jlist = j3list, .num = ARRLEN(j3list) },
			};

			// degeneracy dimensions, indexed by 'j' quantum numbers
			//                         j:  0  1  2  3  4
			const ct_long dim_degen0[] = { 5, 0, 0, 0, 2 };
			const ct_long dim_degen1[] = { 0, 6, 0, 2    };
			const ct_long dim_degen2[] = { 4, 0, 7       };
			const ct_long* dim_degen[] = {
				dim_degen0,
				dim_degen1,
				dim_degen2,
			};

			const int ndim_logical   = 3;
			const int ndim_auxiliary = 1;

			allocate_su2_tensor(CT_SINGLE_REAL, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &t);

			// delete some charge sectors
			su2_tensor_delete_charge_sector_by_index(&t, 2);
			su2_tensor_delete_charge_sector_by_index(&t, 5);

			if (!su2_tensor_is_consistent(&t)) {
				return "internal consistency check for SU(2) tensor failed";
			}
			if (t.charge_sectors.nsec == 0) {
				return "expecting at least one charge sector in SU(2) tensor";
			}

			// fill degeneracy tensors with random entries
			struct rng_state rng_state;
			seed_rng_state(36, &rng_state);
			su2_tensor_fill_random_normal(numeric_one(t.dtype), numeric_zero(t.dtype), &rng_state, &t);
		}

		struct dense_tensor t_dns_ref;
		su2_to_dense_tensor(&t, &t_dns_ref);

		const int i_ax_insert = 1;
		su2_tensor_add_auxiliary_axis(&t, i_ax_insert, m == 0 ? false : true);

		if (!su2_tensor_is_consistent(&t)) {
			return "internal consistency check for SU(2) tensor failed";
		}

		if (t.ndim_auxiliary != 2) {
			return "incorrect number of auxiliary axes after adding an auxiliary axis";;
		}

		if (t.tree.tree_fuse->c[1]->c[m]->i_ax != i_ax_insert) {
			return "incorrect tree axis index after adding an auxiliary axis";
		}

		// adding a "dummy" auxiliary axis must not change logical tensor

		struct dense_tensor t_dns;
		su2_to_dense_tensor(&t, &t_dns);

		// compare
		if (!dense_tensor_allclose(&t_dns, &t_dns_ref, 0.)) {
			return "dense tensor representation of SU(2) tensor after adding an auxiliary axis does not agree with reference";
		}

		delete_dense_tensor(&t_dns);
		delete_dense_tensor(&t_dns_ref);
		delete_su2_tensor(&t);
	}

	return 0;
}


char* test_su2_tensor_transpose()
{
	// construct an SU(2) tensor 't'
	struct su2_tensor t;
	{
		const int ndim = 7;

		// construct the fuse and split tree
		//
		//  3   1    4
		//   ╲   ╲  ╱
		//    ╲   ╲╱       fuse
		//     ╲  ╱6
		//      ╲╱
		//      │
		//      │5
		//      │
		//      ╱╲
		//     ╱  ╲        split
		//    2    0
		//
		struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
		struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
		struct su2_tree_node j6  = { .i_ax = 6, .c = { &j1,  &j4  } };
		struct su2_tree_node j5f = { .i_ax = 5, .c = { &j3,  &j6  } };
		struct su2_tree_node j5s = { .i_ax = 5, .c = { &j2,  &j0  } };

		struct su2_fuse_split_tree tree = { .tree_fuse = &j5f, .tree_split = &j5s, .ndim = ndim };

		if (!su2_fuse_split_tree_is_consistent(&tree)) {
			return "internal consistency check for the fuse and split tree failed";
		}

		// outer (logical and auxiliary) 'j' quantum numbers
		qnumber j0list[] = { 2 };
		qnumber j1list[] = { 0, 2 };
		qnumber j2list[] = { 1, 5 };
		qnumber j3list[] = { 1, 3 };  // auxiliary
		qnumber j4list[] = { 0 };     // auxiliary
		const struct su2_irreducible_list outer_irreps[5] = {
			{ .jlist = j0list, .num = ARRLEN(j0list) },
			{ .jlist = j1list, .num = ARRLEN(j1list) },
			{ .jlist = j2list, .num = ARRLEN(j2list) },
			{ .jlist = j3list, .num = ARRLEN(j3list) },
			{ .jlist = j4list, .num = ARRLEN(j4list) },
		};

		// degeneracy dimensions, indexed by 'j' quantum numbers
		//                         j:  0  1  2  3  4  5
		const ct_long dim_degen0[] = { 0, 0, 7          };
		const ct_long dim_degen1[] = { 3, 0, 5          };
		const ct_long dim_degen2[] = { 0, 4, 0, 0, 0, 2 };
		const ct_long* dim_degen[] = {
			dim_degen0,
			dim_degen1,
			dim_degen2,
		};

		const int ndim_logical   = 3;
		const int ndim_auxiliary = 2;

		allocate_su2_tensor(CT_SINGLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &t);

		// delete some charge sectors
		qnumber jlist_del1[7];
		qnumber jlist_del2[7];
		memcpy(jlist_del1, &t.charge_sectors.jlists[2*t.charge_sectors.ndim], t.charge_sectors.ndim*sizeof(qnumber));
		memcpy(jlist_del2, &t.charge_sectors.jlists[7*t.charge_sectors.ndim], t.charge_sectors.ndim*sizeof(qnumber));
		su2_tensor_delete_charge_sector(&t, jlist_del1);
		su2_tensor_delete_charge_sector(&t, jlist_del2);

		if (!su2_tensor_is_consistent(&t)) {
			return "internal consistency check for SU(2) tensor failed";
		}
		if (t.charge_sectors.nsec == 0) {
			return "expecting at least one charge sector in SU(2) tensor";
		}

		// fill degeneracy tensors with random entries
		struct rng_state rng_state;
		seed_rng_state(37, &rng_state);
		su2_tensor_fill_random_normal(numeric_one(t.dtype), numeric_zero(t.dtype), &rng_state, &t);
	}

	const int perm[7] = { 2, 1, 0, 4, 3, 6, 5 };
	struct su2_tensor t_tp;
	su2_tensor_transpose(perm, &t, &t_tp);

	if (!su2_tensor_is_consistent(&t_tp)) {
		return "internal consistency check for SU(2) tensor failed";
	}

	// reference calculation
	struct dense_tensor t_tp_ref;
	{
		struct dense_tensor t_dns;
		su2_to_dense_tensor(&t, &t_dns);
		dense_tensor_transpose(perm, &t_dns, &t_tp_ref);
		delete_dense_tensor(&t_dns);
	}

	// compare
	if (!dense_su2_tensor_allclose(&t_tp_ref, &t_tp, 0.)) {
		return "transposed SU(2) tensor does not agree with reference";
	}

	delete_dense_tensor(&t_tp_ref);
	delete_su2_tensor(&t_tp);
	delete_su2_tensor(&t);

	return 0;
}


char* test_su2_tensor_fmove()
{
	const int ndim = 7;

	// construct the fuse and split tree
	//
	//  2    4   0
	//   ╲  ╱   ╱
	//    ╲╱   ╱         fuse
	//    5╲  ╱
	//      ╲╱
	//      │
	//      │6
	//      │
	//      ╱╲
	//     ╱  ╲          split
	//    1    3
	//
	struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
	struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
	struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
	struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
	struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
	struct su2_tree_node j5  = { .i_ax = 5, .c = { &j2,  &j4  } };
	struct su2_tree_node j6f = { .i_ax = 6, .c = { &j5,  &j0  } };
	struct su2_tree_node j6s = { .i_ax = 6, .c = { &j1,  &j3  } };

	struct su2_fuse_split_tree tree = { .tree_fuse = &j6f, .tree_split = &j6s, .ndim = ndim };

	if (!su2_fuse_split_tree_is_consistent(&tree)) {
		return "internal consistency check for the fuse and split tree failed";
	}

	// outer (logical and auxiliary) 'j' quantum numbers
	qnumber j0list[] = { 0, 2 };
	qnumber j1list[] = { 3, 5 };
	qnumber j2list[] = { 1, 3 };
	qnumber j3list[] = { 2, 4 };
	qnumber j4list[] = { 0, 2 };
	const struct su2_irreducible_list outer_irreps[5] = {
		{ .jlist = j0list, .num = ARRLEN(j0list) },
		{ .jlist = j1list, .num = ARRLEN(j1list) },
		{ .jlist = j2list, .num = ARRLEN(j2list) },
		{ .jlist = j3list, .num = ARRLEN(j3list) },
		{ .jlist = j4list, .num = ARRLEN(j4list) },
	};

	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                         j:  0  1  2  3  4  5
	const ct_long dim_degen0[] = { 6, 0, 3          };
	const ct_long dim_degen1[] = { 0, 0, 0, 7, 0, 4 };
	const ct_long dim_degen2[] = { 0, 8, 0, 3       };
	const ct_long dim_degen3[] = { 0, 0, 9, 0, 2    };
	const ct_long dim_degen4[] = { 5, 0, 3          };
	const ct_long* dim_degen[] = {
		dim_degen0,
		dim_degen1,
		dim_degen2,
		dim_degen3,
		dim_degen4,
	};

	const int ndim_logical   = 5;
	const int ndim_auxiliary = 0;

	struct su2_tensor t;
	allocate_su2_tensor(CT_DOUBLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &t);

	if (!su2_tensor_is_consistent(&t)) {
		return "internal consistency check for SU(2) tensor failed";
	}
	if (t.charge_sectors.nsec == 0) {
		return "expecting at least one charge sector in SU(2) tensor";
	}

	// fill degeneracy tensors with random entries
	struct rng_state rng_state;
	seed_rng_state(38, &rng_state);
	su2_tensor_fill_random_normal(numeric_one(t.dtype), numeric_zero(t.dtype), &rng_state, &t);

	// convert to full dense tensor, as reference
	struct dense_tensor t_dns;
	su2_to_dense_tensor(&t, &t_dns);

	if (t_dns.ndim != ndim_logical) {
		return "degree of logical dense tensor does not match expected value";
	}

	// perform F-move
	const int i_ax = 5;
	struct su2_tensor r;
	su2_tensor_fmove(&t, i_ax, &r);

	if (!su2_tensor_is_consistent(&r)) {
		return "internal consistency check for SU(2) tensor after F-move failed";
	}

	if (su2_fuse_split_tree_equal(&t.tree, &r.tree)) {
		return "fuse and split trees after F-move must be different";
	}

	// compare with original tensor
	if (!dense_su2_tensor_allclose(&t_dns, &r, 1e-13)) {
		return "F-move applied to SU(2) tensor must leave logical tensor invariant";
	}

	// perform another F-move, undoing original F-move
	struct su2_tensor t2;
	su2_tensor_fmove(&r, i_ax, &t2);

	if (!su2_tensor_is_consistent(&t2)) {
		return "internal consistency check for SU(2) tensor after F-move failed";
	}

	if (!su2_fuse_split_tree_equal(&t.tree, &t2.tree)) {
		return "fuse and split trees after F-move and reversal must be identical";
	}

	// compare with original tensor
	if (!dense_su2_tensor_allclose(&t_dns, &t2, 1e-13)) {
		return "F-move applied to SU(2) tensor must leave logical tensor invariant";
	}

	// clean up
	delete_dense_tensor(&t_dns);
	delete_su2_tensor(&t2);
	delete_su2_tensor(&r);
	delete_su2_tensor(&t);

	return 0;
}


char* test_su2_tensor_reverse_axis_simple()
{
	const int i_ax_rev_list[] = { 0, 2, 3, 4 };
	for (int m = 0; m < (int)ARRLEN(i_ax_rev_list); m++)
	{
		const int i_ax_rev = i_ax_rev_list[m];

		// construct an SU(2) tensor 't'
		struct su2_tensor t;
		{
			const int ndim = 7;

			// construct the fuse and split tree
			//
			//  3    1   4
			//   ╲  ╱   ╱
			//    ╲╱   ╱       fuse
			//    6╲  ╱
			//      ╲╱
			//      │
			//      │5
			//      │
			//      ╱╲
			//     ╱  ╲        split
			//    0    2
			//
			struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
			struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
			struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
			struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
			struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
			struct su2_tree_node j6  = { .i_ax = 6, .c = { &j3,  &j1  } };
			struct su2_tree_node j5f = { .i_ax = 5, .c = { &j6,  &j4  } };
			struct su2_tree_node j5s = { .i_ax = 5, .c = { &j0,  &j2  } };

			struct su2_fuse_split_tree tree = { .tree_fuse = &j5f, .tree_split = &j5s, .ndim = ndim };

			if (!su2_fuse_split_tree_is_consistent(&tree)) {
				return "internal consistency check for the fuse and split tree failed";
			}

			// outer (logical and auxiliary) 'j' quantum numbers
			qnumber j0list[] = { 2 };
			qnumber j1list[] = { 0, 2 };
			qnumber j2list[] = { 1, 3 };
			qnumber j3list[] = { 1, 5 };
			qnumber j4list[] = { 0 };  // auxiliary
			const struct su2_irreducible_list outer_irreps[5] = {
				{ .jlist = j0list, .num = ARRLEN(j0list) },
				{ .jlist = j1list, .num = ARRLEN(j1list) },
				{ .jlist = j2list, .num = ARRLEN(j2list) },
				{ .jlist = j3list, .num = ARRLEN(j3list) },
				{ .jlist = j4list, .num = ARRLEN(j4list) },
			};

			// degeneracy dimensions, indexed by 'j' quantum numbers
			//                         j:  0  1  2  3  4  5
			const ct_long dim_degen0[] = { 0, 0, 3          };
			const ct_long dim_degen1[] = { 3, 0, 4          };
			const ct_long dim_degen2[] = { 0, 5, 0, 2       };
			const ct_long dim_degen3[] = { 0, 7, 0, 0, 0, 8 };
			const ct_long* dim_degen[] = {
				dim_degen0,
				dim_degen1,
				dim_degen2,
				dim_degen3,
			};

			const int ndim_logical   = 4;
			const int ndim_auxiliary = 1;

			allocate_su2_tensor(CT_SINGLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &t);

			// delete some charge sectors
			qnumber jlist_del1[7];
			qnumber jlist_del2[7];
			memcpy(jlist_del1, &t.charge_sectors.jlists[2*t.charge_sectors.ndim], t.charge_sectors.ndim*sizeof(qnumber));
			memcpy(jlist_del2, &t.charge_sectors.jlists[5*t.charge_sectors.ndim], t.charge_sectors.ndim*sizeof(qnumber));
			su2_tensor_delete_charge_sector(&t, jlist_del1);
			su2_tensor_delete_charge_sector(&t, jlist_del2);

			if (!su2_tensor_is_consistent(&t)) {
				return "internal consistency check for SU(2) tensor failed";
			}
			if (t.charge_sectors.nsec == 0) {
				return "expecting at least one charge sector in SU(2) tensor";
			}

			// fill degeneracy tensors with random entries
			struct rng_state rng_state;
			seed_rng_state(39, &rng_state);
			su2_tensor_fill_random_normal(numeric_one(t.dtype), numeric_zero(t.dtype), &rng_state, &t);
		}

		struct su2_tensor t_orig;
		copy_su2_tensor(&t, &t_orig);

		// convert to a full dense tensor
		struct dense_tensor t_orig_dns;
		su2_to_dense_tensor(&t_orig, &t_orig_dns);

		if (i_ax_rev == 3)
		{
			// require an F-move since axis 3 is not a direct child of fusion tree root
			struct su2_tensor tmp;
			su2_tensor_fmove(&t, 6, &tmp);
			delete_su2_tensor(&t);
			// copy internal data
			t = tmp;
		}

		// reverse axis
		su2_tensor_reverse_axis_simple(&t, i_ax_rev);

		if (!su2_tensor_is_consistent(&t)) {
			return "internal consistency check for SU(2) tensor failed";
		}

		// expected fuse and split tree after axis reversal
		if (i_ax_rev == 0)
		{
			const int ndim = 7;

			// construct the fuse and split tree (with axis 0 reversed compared to original tree)
			//
			//  0   3    1   4
			//   ╲   ╲  ╱   ╱
			//    ╲   ╲╱   ╱       fuse
			//     ╲  6╲  ╱
			//      ╲   ╲╱
			//       ╲  ╱5
			//        ╲╱
			//        │
			//        │            split
			//        2
			//
			struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
			struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
			struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
			struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
			struct su2_tree_node j6  = { .i_ax = 6, .c = { &j3,  &j1  } };
			struct su2_tree_node j5  = { .i_ax = 5, .c = { &j6,  &j4  } };
			struct su2_tree_node j2f = { .i_ax = 2, .c = { &j0,  &j5  } };
			struct su2_tree_node j2s = { .i_ax = 2, .c = { NULL, NULL } };

			struct su2_fuse_split_tree tree_ref = { .tree_fuse = &j2f, .tree_split = &j2s, .ndim = ndim };

			if (!su2_fuse_split_tree_is_consistent(&tree_ref)) {
				return "internal consistency check for the fuse and split tree failed";
			}

			// compare
			if (!su2_fuse_split_tree_equal(&t.tree, &tree_ref)) {
				return "fuse and split tree after axis reversal does not match reference";
			}
		}
		else if (i_ax_rev == 2)
		{
			const int ndim = 7;

			// construct the fuse and split tree (with axis 2 reversed compared to original tree)
			//
			//  3    1   4   2
			//   ╲  ╱   ╱   ╱
			//    ╲╱   ╱   ╱       fuse
			//    6╲  ╱   ╱
			//      ╲╱   ╱
			//      5╲  ╱
			//        ╲╱
			//        │
			//        │            split
			//        0
			//
			struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
			struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
			struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
			struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
			struct su2_tree_node j6  = { .i_ax = 6, .c = { &j3,  &j1  } };
			struct su2_tree_node j5  = { .i_ax = 5, .c = { &j6,  &j4  } };
			struct su2_tree_node j0f = { .i_ax = 0, .c = { &j5,  &j2  } };
			struct su2_tree_node j0s = { .i_ax = 0, .c = { NULL, NULL } };

			struct su2_fuse_split_tree tree_ref = { .tree_fuse = &j0f, .tree_split = &j0s, .ndim = ndim };

			if (!su2_fuse_split_tree_is_consistent(&tree_ref)) {
				return "internal consistency check for the fuse and split tree failed";
			}

			// compare
			if (!su2_fuse_split_tree_equal(&t.tree, &tree_ref)) {
				return "fuse and split tree after axis reversal does not match reference";
			}
		}
		else if (i_ax_rev == 3)
		{
			const int ndim = 7;

			// construct the fuse and split tree (with axis 3 reversed and an F-move compared to original tree)
			//
			//    1    4
			//     ╲  ╱        fuse
			//      ╲╱
			//      │
			//      │6
			//      │
			//      ╱╲
			//     ╱  ╲5
			//    ╱   ╱╲       split
			//   ╱   ╱  ╲
			//  3   0    2
			//
			struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
			struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
			struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
			struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
			struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
			struct su2_tree_node j5  = { .i_ax = 5, .c = { &j0,  &j2  } };
			struct su2_tree_node j6f = { .i_ax = 6, .c = { &j1,  &j4  } };
			struct su2_tree_node j6s = { .i_ax = 6, .c = { &j3,  &j5  } };

			struct su2_fuse_split_tree tree_ref = { .tree_fuse = &j6f, .tree_split = &j6s, .ndim = ndim };

			if (!su2_fuse_split_tree_is_consistent(&tree_ref)) {
				return "internal consistency check for the fuse and split tree failed";
			}

			// compare
			if (!su2_fuse_split_tree_equal(&t.tree, &tree_ref)) {
				return "fuse and split tree after axis reversal does not match reference";
			}
		}
		else if (i_ax_rev == 4)
		{
			const int ndim = 7;

			// construct the fuse and split tree (with axis 4 reversed compared to original tree)
			//
			//    3    1
			//     ╲  ╱        fuse
			//      ╲╱
			//      │
			//     6│
			//      │
			//      ╱╲
			//    5╱  ╲
			//    ╱╲   ╲       split
			//   ╱  ╲   ╲
			//  0    2   4
			//
			struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
			struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
			struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
			struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
			struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
			struct su2_tree_node j5  = { .i_ax = 5, .c = { &j0,  &j2  } };
			struct su2_tree_node j6f = { .i_ax = 6, .c = { &j3,  &j1  } };
			struct su2_tree_node j6s = { .i_ax = 6, .c = { &j5,  &j4  } };

			struct su2_fuse_split_tree tree_ref = { .tree_fuse = &j6f, .tree_split = &j6s, .ndim = ndim };

			if (!su2_fuse_split_tree_is_consistent(&tree_ref)) {
				return "internal consistency check for the fuse and split tree failed";
			}

			// compare
			if (!su2_fuse_split_tree_equal(&t.tree, &tree_ref)) {
				return "fuse and split tree after axis reversal does not match reference";
			}
		}
		else
		{
			assert(false);
		}

		// construct explicit reversal operation as dense tensor
		struct dense_tensor omega_dns;
		if ((i_ax_rev == 0) || (i_ax_rev == 2))
		{
			struct su2_tensor u_cup;

			const int ndim = 3;

			// construct the fuse and split tree
			//
			//    0    1
			//     ╲  ╱
			//      ╲╱
			//      │
			//      │
			//      2
			//
			struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
			struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
			struct su2_tree_node j2f = { .i_ax = 2, .c = { &j0,  &j1  } };
			struct su2_tree_node j2s = { .i_ax = 2, .c = { NULL, NULL } };

			struct su2_fuse_split_tree tree = { .tree_fuse = &j2f, .tree_split = &j2s, .ndim = ndim };

			if (!su2_fuse_split_tree_is_consistent(&tree)) {
				return "internal consistency check for the fuse and split tree failed";
			}

			if (i_ax_rev == 0)
			{
				// outer (logical and auxiliary) 'j' quantum numbers
				qnumber j0list[] = { 2 };
				qnumber j1list[] = { 2 };
				qnumber j2list[] = { 0 };  // auxiliary
				const struct su2_irreducible_list outer_irreps[3] = {
					{ .jlist = j0list, .num = ARRLEN(j0list) },
					{ .jlist = j1list, .num = ARRLEN(j1list) },
					{ .jlist = j2list, .num = ARRLEN(j2list) },
				};

				// degeneracy dimensions, indexed by 'j' quantum numbers
				//                         j:  0  1  2
				const ct_long dim_degen0[] = { 0, 0, 3 };
				const ct_long dim_degen1[] = { 0, 0, 3 };
				const ct_long* dim_degen[] = {
					dim_degen0,
					dim_degen1,
				};

				const int ndim_logical   = 2;
				const int ndim_auxiliary = 1;

				allocate_su2_tensor(CT_SINGLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &u_cup);
			}
			else if (i_ax_rev == 2)
			{
				// outer (logical and auxiliary) 'j' quantum numbers
				qnumber j0list[] = { 1, 3 };
				qnumber j1list[] = { 1, 3 };
				qnumber j2list[] = { 0 };  // auxiliary
				const struct su2_irreducible_list outer_irreps[3] = {
					{ .jlist = j0list, .num = ARRLEN(j0list) },
					{ .jlist = j1list, .num = ARRLEN(j1list) },
					{ .jlist = j2list, .num = ARRLEN(j2list) },
				};

				// degeneracy dimensions, indexed by 'j' quantum numbers
				//                         j:  0  1  2  3
				const ct_long dim_degen0[] = { 0, 5, 0, 2 };
				const ct_long dim_degen1[] = { 0, 5, 0, 2 };
				const ct_long* dim_degen[] = {
					dim_degen0,
					dim_degen1,
				};

				const int ndim_logical   = 2;
				const int ndim_auxiliary = 1;

				allocate_su2_tensor(CT_SINGLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &u_cup);
			}
			else
			{
				assert(false);
			}

			if (!su2_tensor_is_consistent(&u_cup)) {
				return "internal consistency check for SU(2) tensor failed";
			}
			if (u_cup.charge_sectors.nsec == 0) {
				return "expecting at least one charge sector in SU(2) tensor";
			}

			// set "degeneracy" tensors to scaled identities
			for (ct_long c = 0; c < u_cup.charge_sectors.nsec; c++)
			{
				struct dense_tensor* d = u_cup.degensors[c];
				assert(d != NULL);
				assert(d->dtype == u_cup.dtype);
				assert(d->ndim  == u_cup.ndim_logical);

				// quantum number at axis 0
				const qnumber j = u_cup.charge_sectors.jlists[c*u_cup.charge_sectors.ndim];
				const float alpha = (float)sqrt(j + 1);
				dense_tensor_set_identity(d);
				rscale_dense_tensor(&alpha, d);
			}

			su2_to_dense_tensor(&u_cup, &omega_dns);

			delete_su2_tensor(&u_cup);
		}
		else if ((i_ax_rev == 3) || (i_ax_rev == 4))
		{
			struct su2_tensor u_cap;

			const int ndim = 3;

			// construct the fuse and split tree (note ordering of axes 0 and 1)
			//
			//      2
			//      │
			//      │
			//      ╱╲
			//     ╱  ╲
			//    1    0
			//
			struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
			struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
			struct su2_tree_node j2f = { .i_ax = 2, .c = { NULL, NULL } };
			struct su2_tree_node j2s = { .i_ax = 2, .c = { &j1,  &j0  } };

			struct su2_fuse_split_tree tree = { .tree_fuse = &j2f, .tree_split = &j2s, .ndim = ndim };

			if (!su2_fuse_split_tree_is_consistent(&tree)) {
				return "internal consistency check for the fuse and split tree failed";
			}

			if (i_ax_rev == 3)
			{
				// outer (logical and auxiliary) 'j' quantum numbers
				qnumber j0list[] = { 1, 5 };
				qnumber j1list[] = { 1, 5 };
				qnumber j2list[] = { 0 };  // auxiliary
				const struct su2_irreducible_list outer_irreps[3] = {
					{ .jlist = j0list, .num = ARRLEN(j0list) },
					{ .jlist = j1list, .num = ARRLEN(j1list) },
					{ .jlist = j2list, .num = ARRLEN(j2list) },
				};

				// degeneracy dimensions, indexed by 'j' quantum numbers
				//                         j:  0  1  2  3  4  5
				const ct_long dim_degen0[] = { 0, 7, 0, 0, 0, 8 };
				const ct_long dim_degen1[] = { 0, 7, 0, 0, 0, 8 };
				const ct_long* dim_degen[] = {
					dim_degen0,
					dim_degen1,
				};

				const int ndim_logical   = 2;
				const int ndim_auxiliary = 1;

				allocate_su2_tensor(CT_SINGLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &u_cap);
			}
			else if (i_ax_rev == 4)
			{
				// axis 4 is an auxiliary axis in original SU(2) tensor

				// outer (logical and auxiliary) 'j' quantum numbers
				qnumber j0list[] = { 0 };
				qnumber j1list[] = { 0 };
				qnumber j2list[] = { 0 };  // auxiliary
				const struct su2_irreducible_list outer_irreps[3] = {
					{ .jlist = j0list, .num = ARRLEN(j0list) },
					{ .jlist = j1list, .num = ARRLEN(j1list) },
					{ .jlist = j2list, .num = ARRLEN(j2list) },
				};

				// degeneracy dimensions, indexed by 'j' quantum numbers
				//                         j:  0
				const ct_long dim_degen0[] = { 1 };
				const ct_long dim_degen1[] = { 1 };
				const ct_long* dim_degen[] = {
					dim_degen0,
					dim_degen1,
				};

				const int ndim_logical   = 2;
				const int ndim_auxiliary = 1;

				allocate_su2_tensor(CT_SINGLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &u_cap);
			}
			else
			{
				assert(false);
			}

			if (!su2_tensor_is_consistent(&u_cap)) {
				return "internal consistency check for SU(2) tensor failed";
			}
			if (u_cap.charge_sectors.nsec == 0) {
				return "expecting at least one charge sector in SU(2) tensor";
			}

			// set "degeneracy" tensors to scaled identities
			for (ct_long c = 0; c < u_cap.charge_sectors.nsec; c++)
			{
				struct dense_tensor* d = u_cap.degensors[c];
				assert(d != NULL);
				assert(d->dtype == u_cap.dtype);
				assert(d->ndim  == u_cap.ndim_logical);

				// quantum number at axis 0
				const qnumber j = u_cap.charge_sectors.jlists[c*u_cap.charge_sectors.ndim];
				const float alpha = (float)sqrt(j + 1);
				dense_tensor_set_identity(d);
				rscale_dense_tensor(&alpha, d);
			}

			su2_to_dense_tensor(&u_cap, &omega_dns);

			delete_su2_tensor(&u_cap);
		}
		else
		{
			assert(false);
		}

		// all entries should be 0, 1 or -1
		{
			const ct_long nelem = dense_tensor_num_elements(&omega_dns);
			const scomplex* data = omega_dns.data;
			const float tol = 1e-5;
			for (ct_long i = 0; i < nelem; i++) {
				if ((cabsf(data[i]) > tol) && (cabsf(data[i] - 1) > tol) && (cabsf(data[i] + 1) > tol)) {
					return "'cup' tensor has unexpected entry not equal to 0, 1 or -1";
				}
			}
		}

		// contract with dense "omega" tensor to obtain reference tensor
		struct dense_tensor t_ref_dns;
		if (i_ax_rev < 4)
		{
			dense_tensor_multiply_axis(&t_orig_dns, i_ax_rev, &omega_dns, (i_ax_rev == 0 || i_ax_rev == 3) ? TENSOR_AXIS_RANGE_TRAILING : TENSOR_AXIS_RANGE_LEADING, &t_ref_dns);
		}
		else if (i_ax_rev == 4)
		{
			// axis 4 is an auxiliary axis, and reversal should leave logical tensor invariant
			copy_dense_tensor(&t_orig_dns, &t_ref_dns);
		}
		else
		{
			assert(false);
		}

		// compare
		if (!dense_su2_tensor_allclose(&t_ref_dns, &t, 1e-5)) {
			return "tensor resulting from reversing an axis of an SU(2) tensor does not match reference";
		}

		// undo reversal by reversing axis again
		su2_tensor_reverse_axis_simple(&t, i_ax_rev);

		if (i_ax_rev == 3)
		{
			// require an F-move since axis 3 is not a direct child of fusion tree root
			struct su2_tensor tmp;
			su2_tensor_fmove(&t, 6, &tmp);
			delete_su2_tensor(&t);
			// copy internal data
			t = tmp;
		}

		if (!su2_tensor_is_consistent(&t)) {
			return "internal consistency check for SU(2) tensor failed";
		}

		if (!su2_tensor_allclose(&t, &t_orig, 1e-5)) {
			return "SU(2) tensor resulting from reversing an axis twice does not agree with original tensor";
		}

		// clean up
		delete_dense_tensor(&t_ref_dns);
		delete_dense_tensor(&omega_dns);
		delete_dense_tensor(&t_orig_dns);
		delete_su2_tensor(&t_orig);
		delete_su2_tensor(&t);
	}

	return 0;
}


char* test_su2_tensor_fuse_axes()
{
	// construct an SU(2) tensor 't'
	struct su2_tensor t;
	{
		const int ndim = 7;

		// construct the fuse and split tree
		//
		//  0   3    1
		//   ╲   ╲  ╱
		//    ╲   ╲╱       fuse
		//     ╲  ╱5
		//      ╲╱
		//      │
		//      │6
		//      │
		//      ╱╲
		//     ╱  ╲        split
		//    2    4
		//
		struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
		struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
		struct su2_tree_node j5  = { .i_ax = 5, .c = { &j3,  &j1  } };
		struct su2_tree_node j6f = { .i_ax = 6, .c = { &j0,  &j5  } };
		struct su2_tree_node j6s = { .i_ax = 6, .c = { &j2,  &j4  } };

		struct su2_fuse_split_tree tree = { .tree_fuse = &j6f, .tree_split = &j6s, .ndim = ndim };

		if (!su2_fuse_split_tree_is_consistent(&tree)) {
			return "internal consistency check for the fuse and split tree failed";
		}

		// outer (logical and auxiliary) 'j' quantum numbers
		qnumber j0list[] = { 2 };
		qnumber j1list[] = { 0, 2 };
		qnumber j2list[] = { 1, 3 };
		qnumber j3list[] = { 1, 5 };
		qnumber j4list[] = { 0 };  // auxiliary
		const struct su2_irreducible_list outer_irreps[5] = {
			{ .jlist = j0list, .num = ARRLEN(j0list) },
			{ .jlist = j1list, .num = ARRLEN(j1list) },
			{ .jlist = j2list, .num = ARRLEN(j2list) },
			{ .jlist = j3list, .num = ARRLEN(j3list) },
			{ .jlist = j4list, .num = ARRLEN(j4list) },
		};

		// degeneracy dimensions, indexed by 'j' quantum numbers
		//                         j:  0  1  2  3  4  5
		const ct_long dim_degen0[] = { 0, 0, 3          };
		const ct_long dim_degen1[] = { 3, 0, 4          };
		const ct_long dim_degen2[] = { 0, 5, 0, 2       };
		const ct_long dim_degen3[] = { 0, 7, 0, 0, 0, 8 };
		const ct_long* dim_degen[] = {
			dim_degen0,
			dim_degen1,
			dim_degen2,
			dim_degen3,
		};

		const int ndim_logical   = 4;
		const int ndim_auxiliary = 1;

		allocate_su2_tensor(CT_SINGLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &t);

		// delete some charge sectors
		qnumber jlist_del1[7];
		qnumber jlist_del2[7];
		memcpy(jlist_del1, &t.charge_sectors.jlists[2*t.charge_sectors.ndim], t.charge_sectors.ndim*sizeof(qnumber));
		memcpy(jlist_del2, &t.charge_sectors.jlists[7*t.charge_sectors.ndim], t.charge_sectors.ndim*sizeof(qnumber));
		su2_tensor_delete_charge_sector(&t, jlist_del1);
		su2_tensor_delete_charge_sector(&t, jlist_del2);

		if (!su2_tensor_is_consistent(&t)) {
			return "internal consistency check for SU(2) tensor failed";
		}
		if (t.charge_sectors.nsec == 0) {
			return "expecting at least one charge sector in SU(2) tensor";
		}

		// fill degeneracy tensors with random entries
		struct rng_state rng_state;
		seed_rng_state(40, &rng_state);
		su2_tensor_fill_random_normal(numeric_one(t.dtype), numeric_zero(t.dtype), &rng_state, &t);
	}

	// fuse axes
	const int i_ax_0 = 1;
	const int i_ax_1 = 3;
	struct su2_tensor r;
	su2_tensor_fuse_axes(&t, i_ax_0, i_ax_1, &r);

	if (!su2_tensor_is_consistent(&r)) {
		return "internal consistency check for SU(2) tensor failed";
	}

	// construct explicit fusion operation as dense tensor
	struct dense_tensor u_fuse_dns;
	{
		struct su2_tensor u_fuse;

		const int ndim = 3;

		// construct the fuse and split tree
		//
		//      0
		//      │
		//      │
		//      ╱╲
		//     ╱  ╲
		//    2    1
		//
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j0f = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j0s = { .i_ax = 0, .c = { &j2,  &j1  } };

		struct su2_fuse_split_tree tree = { .tree_fuse = &j0f, .tree_split = &j0s, .ndim = ndim };

		if (!su2_fuse_split_tree_is_consistent(&tree)) {
			return "internal consistency check for the fuse and split tree failed";
		}

		// outer (logical and auxiliary) 'j' quantum numbers
		qnumber j0list[] = { 1, 3, 5, 7 };
		qnumber j1list[] = { 0, 2 };
		qnumber j2list[] = { 1, 5 };
		const struct su2_irreducible_list outer_irreps[3] = {
			{ .jlist = j0list, .num = ARRLEN(j0list) },
			{ .jlist = j1list, .num = ARRLEN(j1list) },
			{ .jlist = j2list, .num = ARRLEN(j2list) },
		};

		// degeneracy dimensions, indexed by 'j' quantum numbers
		//                          j:  0   1   2   3   4   5   6   7
		const ct_long dim_degen0[] = {  0, 49,  0, 60,  0, 56,  0, 32 };  // 49 = 3*7 + 4*7, 60 = 4*7 + 4*8, 56 = 3*8 + 4*8
		const ct_long dim_degen1[] = {  3,  0,  4                     };
		const ct_long dim_degen2[] = {  0,  7,  0,  0,  0,  8         };
		const ct_long* dim_degen[] = {
			dim_degen0,
			dim_degen1,
			dim_degen2,
		};

		const int ndim_logical   = 3;
		const int ndim_auxiliary = 0;

		allocate_su2_tensor(CT_SINGLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &u_fuse);

		if (!su2_tensor_is_consistent(&u_fuse)) {
			return "internal consistency check for SU(2) tensor failed";
		}
		if (u_fuse.charge_sectors.nsec == 0) {
			return "expecting at least one charge sector in SU(2) tensor";
		}

		for (ct_long c = 0; c < u_fuse.charge_sectors.nsec; c++)
		{
			// current 'j' quantum numbers
			const qnumber* jlist = &u_fuse.charge_sectors.jlists[c * u_fuse.charge_sectors.ndim];

			// corresponding "degeneracy" tensor
			struct dense_tensor* d = u_fuse.degensors[c];
			assert(d != NULL);
			assert(d->dtype == u_fuse.dtype);
			assert(d->ndim  == u_fuse.ndim_logical);

			// set "degeneracy" tensor to (padded) identity
			struct dense_tensor identity;
			const ct_long dim_id[2] = { d->dim[1]*d->dim[2], d->dim[1]*d->dim[2] };
			allocate_dense_tensor(d->dtype, 2, dim_id, &identity);
			dense_tensor_set_identity(&identity);
			ct_long pb;
			if (jlist[0] == 1 && jlist[1] == 2 && jlist[2] == 1) {
				pb = 3*7;
			}
			else if (jlist[0] == 3 && jlist[1] == 2 && jlist[2] == 5) {
				pb = 4*7;
			}
			else if (jlist[0] == 5 && jlist[1] == 2 && jlist[2] == 5) {
				pb = 3*8;
			}
			else {
				pb = 0;
			}
			ct_long pa;
			if (jlist[0] == 1 && jlist[1] == 0 && jlist[2] == 1) {
				pa = 4*7;
			}
			else if (jlist[0] == 3 && jlist[1] == 2 && jlist[2] == 1) {
				pa = 4*8;
			}
			else if (jlist[0] == 5 && jlist[1] == 0 && jlist[2] == 5) {
				pa = 4*8;
			}
			else {
				pa = 0;
			}
			const ct_long pad_before[2] = { pb, 0 };
			const ct_long pad_after[2]  = { pa, 0 };
			struct dense_tensor identity_pad;
			dense_tensor_pad_zeros(&identity, pad_before, pad_after, &identity_pad);
			delete_dense_tensor(&identity);
			reshape_dense_tensor(d->ndim, d->dim, &identity_pad);
			delete_dense_tensor(d);
			*u_fuse.degensors[c] = identity_pad;  // copy internal data pointers
		}

		su2_to_dense_tensor(&u_fuse, &u_fuse_dns);

		delete_su2_tensor(&u_fuse);
	}

	struct dense_tensor t_dns;
	su2_to_dense_tensor(&t, &t_dns);

	// contract with dense fusion tensor to obtain reference tensor
	struct dense_tensor r_dns_ref;
	{
		// move to-be fused axes in 't' to the end
		const int perm_t[4] = { 0, 2, 1, 3 };
		struct dense_tensor t_tmp;
		dense_tensor_transpose(perm_t, &t_dns, &t_tmp);

		// perform contraction
		struct dense_tensor r_dns_perm;
		dense_tensor_dot(&t_tmp, TENSOR_AXIS_RANGE_TRAILING, &u_fuse_dns, TENSOR_AXIS_RANGE_TRAILING, 2, &r_dns_perm);
		delete_dense_tensor(&t_tmp);

		// transpose the fused axis to become axis 1
		const int perm_r[3] = { 0, 2, 1 };
		dense_tensor_transpose(perm_r, &r_dns_perm, &r_dns_ref);
		delete_dense_tensor(&r_dns_perm);
	}

	// compare
	if (!dense_su2_tensor_allclose(&r_dns_ref, &r, 1e-5)) {
		return "tensor resulting from fusing two axes of an SU(2) tensor does not match reference";
	}

	// split axis again
	struct su2_tensor t2;
	{
		const struct su2_irreducible_list outer_irreps[2] = {
			t.outer_irreps[i_ax_0],
			t.outer_irreps[i_ax_1],
		};
		const ct_long* dim_degen[2] = {
			t.dim_degen[i_ax_0],
			t.dim_degen[i_ax_1],
		};
		su2_tensor_split_axis(&r, i_ax_0, i_ax_1, false, outer_irreps, dim_degen, &t2);

		if (!su2_tensor_is_consistent(&t2)) {
			return "internal consistency check for SU(2) tensor failed";
		}
	}

	// expected fuse and split tree after axis splitting
	{
		const int ndim = 7;

		// construct the fuse and split tree (with internal axes 5 <-> 6 flipped compared to original tree)
		//
		//  0   3    1
		//   ╲   ╲  ╱
		//    ╲   ╲╱       fuse
		//     ╲  ╱6
		//      ╲╱
		//      │
		//      │5
		//      │
		//      ╱╲
		//     ╱  ╲        split
		//    2    4
		//
		struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
		struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
		struct su2_tree_node j6  = { .i_ax = 6, .c = { &j3,  &j1  } };
		struct su2_tree_node j5f = { .i_ax = 5, .c = { &j0,  &j6  } };
		struct su2_tree_node j5s = { .i_ax = 5, .c = { &j2,  &j4  } };

		struct su2_fuse_split_tree tree_ref = { .tree_fuse = &j5f, .tree_split = &j5s, .ndim = ndim };

		if (!su2_fuse_split_tree_is_consistent(&tree_ref)) {
			return "internal consistency check for the fuse and split tree failed";
		}

		// compare
		if (!su2_fuse_split_tree_equal(&t2.tree, &tree_ref)) {
			return "fuse and split tree after axis fusion and splitting does not match reference";
		}
	}

	// compare
	if (!dense_su2_tensor_allclose(&t_dns, &t2, 1e-6)) {
		return "tensor after axis fusion and splitting should agree with original tensor";
	}

	delete_dense_tensor(&r_dns_ref);
	delete_dense_tensor(&u_fuse_dns);
	delete_dense_tensor(&t_dns);
	delete_su2_tensor(&t2);
	delete_su2_tensor(&r);
	delete_su2_tensor(&t);

	return 0;
}


char* test_su2_tensor_fuse_axes_add_auxiliary()
{
	struct rng_state rng_state;
	seed_rng_state(41, &rng_state);

	for (int m = 0; m < 2; m++)  // whether to insert the new auxiliary axis on the left or right
	{
		// construct an SU(2) tensor 't'
		struct su2_tensor t;
		{
			const int ndim = 9;

			// construct the fuse and split tree
			//
			//  5    0   3
			//   ╲  ╱   ╱
			//    ╲╱   ╱       fuse
			//    8╲  ╱
			//      ╲╱
			//      │
			//      │6
			//      │
			//      ╱╲
			//     ╱  ╲7
			//    ╱   ╱╲       split
			//   ╱   ╱  ╲
			//  2   4    1
			//
			struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
			struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
			struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
			struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
			struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
			struct su2_tree_node j5  = { .i_ax = 5, .c = { NULL, NULL } };
			struct su2_tree_node j7  = { .i_ax = 7, .c = { &j4,  &j1  } };
			struct su2_tree_node j8  = { .i_ax = 8, .c = { &j5,  &j0  } };
			struct su2_tree_node j6f = { .i_ax = 6, .c = { &j8,  &j3  } };
			struct su2_tree_node j6s = { .i_ax = 6, .c = { &j2,  &j7  } };

			struct su2_fuse_split_tree tree = { .tree_fuse = &j6f, .tree_split = &j6s, .ndim = ndim };

			if (!su2_fuse_split_tree_is_consistent(&tree)) {
				return "internal consistency check for the fuse and split tree failed";
			}

			// outer (logical and auxiliary) 'j' quantum numbers
			qnumber j0list[] = { 2, 4 };
			qnumber j1list[] = { 1, 3 };
			qnumber j2list[] = { 0, 2 };
			qnumber j3list[] = { 1, 3 };
			qnumber j4list[] = { 0, 2 };
			qnumber j5list[] = { 0 };  // auxiliary
			const struct su2_irreducible_list outer_irreps[6] = {
				{ .jlist = j0list, .num = ARRLEN(j0list) },
				{ .jlist = j1list, .num = ARRLEN(j1list) },
				{ .jlist = j2list, .num = ARRLEN(j2list) },
				{ .jlist = j3list, .num = ARRLEN(j3list) },
				{ .jlist = j4list, .num = ARRLEN(j4list) },
				{ .jlist = j5list, .num = ARRLEN(j5list) },
			};

			// degeneracy dimensions, indexed by 'j' quantum numbers
			//                         j:  0  1  2  3  4
			const ct_long dim_degen0[] = { 0, 0, 3, 0, 2 };
			const ct_long dim_degen1[] = { 0, 6, 0, 1    };
			const ct_long dim_degen2[] = { 5, 0, 4       };
			const ct_long dim_degen3[] = { 0, 4, 0, 3    };
			const ct_long dim_degen4[] = { 7, 0, 2       };
			const ct_long* dim_degen[] = {
				dim_degen0,
				dim_degen1,
				dim_degen2,
				dim_degen3,
				dim_degen4,
			};

			const int ndim_logical   = 5;
			const int ndim_auxiliary = 1;

			allocate_su2_tensor(CT_DOUBLE_REAL, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &t);

			// delete some charge sectors
			su2_tensor_delete_charge_sector_by_index(&t, 4);
			su2_tensor_delete_charge_sector_by_index(&t, 4);
			su2_tensor_delete_charge_sector_by_index(&t, 7);

			if (!su2_tensor_is_consistent(&t)) {
				return "internal consistency check for SU(2) tensor failed";
			}
			if (t.charge_sectors.nsec == 0) {
				return "expecting at least one charge sector in SU(2) tensor";
			}

			// fill degeneracy tensors with random entries
			su2_tensor_fill_random_normal(numeric_one(t.dtype), numeric_zero(t.dtype), &rng_state, &t);
		}

		// fuse axes and add a dummy auxiliary axis
		const int i_ax_0 = 1;
		const int i_ax_1 = 4;
		struct su2_tensor r;
		su2_tensor_fuse_axes_add_auxiliary(&t, i_ax_0, i_ax_1, m == 0, &r);

		if (!su2_tensor_is_consistent(&r)) {
			return "internal consistency check for SU(2) tensor failed";
		}
		if (r.ndim_logical != t.ndim_logical - 1) {
			return "SU(2) tensor resulting from fusing two axes and adding an auxiliary axis has incorrect number of logical dimensions";
		}
		if (r.ndim_auxiliary != t.ndim_auxiliary + 1) {
			return "SU(2) tensor resulting from fusing two axes and adding an auxiliary axis has incorrect number of auxiliary dimensions";
		}
		if (!su2_fuse_split_tree_equal_topology(&t.tree, &r.tree)) {
			return "fusion-splitting tree of SU(2) tensor resulting from fusing two axes and adding an auxiliary axis must have same topology as the original tree";
		}
		if (r.outer_irreps[r.ndim_logical].num != 1 || r.outer_irreps[r.ndim_logical].jlist[0] != 0) {
			return "new auxiliary axis after fusing two axes must have quantum number zero";
		}
		if (r.tree.tree_split->c[1]->c[m]->i_ax != i_ax_0) {
			return "first logical axis after fusing two axes not found in fusion-splitting tree";
		}
		if (r.tree.tree_split->c[1]->c[1 - m]->i_ax != r.ndim_logical) {
			return "new auxiliary axis after fusing two axes not found in fusion-splitting tree";
		}

		// dense tensor represents logical tensor
		struct dense_tensor r_dns;
		su2_to_dense_tensor(&r, &r_dns);

		// fusing axes without adding an auxiliary axis should result in the same logical tensor
		struct dense_tensor r_ref;
		{
			struct su2_tensor s;
			su2_tensor_fuse_axes(&t, i_ax_0, i_ax_1, &s);
			su2_to_dense_tensor(&s, &r_ref);
			delete_su2_tensor(&s);
		}

		// compare
		if (!dense_tensor_allclose(&r_dns, &r_ref, 1e-13)) {
			return "fusing two axes of an SU(2) symmetric tensor with and without adding an auxiliary axis should result in the same logical tensor";
		}

		delete_dense_tensor(&r_ref);
		delete_dense_tensor(&r_dns);
		delete_su2_tensor(&r);
		delete_su2_tensor(&t);
	}

	return 0;
}


char* test_su2_tensor_split_axis()
{
	// construct an SU(2) tensor 't'
	struct su2_tensor t;
	{
		const int ndim = 5;

		// construct the fuse and split tree
		//
		//  0    3   1
		//   ╲  ╱   ╱
		//    ╲╱   ╱       fuse
		//    4╲  ╱
		//      ╲╱
		//      │
		//      │
		//      2
		//
		struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
		struct su2_tree_node j4  = { .i_ax = 4, .c = { &j0,  &j3  } };
		struct su2_tree_node j2f = { .i_ax = 2, .c = { &j4,  &j1  } };
		struct su2_tree_node j2s = { .i_ax = 2, .c = { NULL, NULL } };

		struct su2_fuse_split_tree tree = { .tree_fuse = &j2f, .tree_split = &j2s, .ndim = ndim };

		if (!su2_fuse_split_tree_is_consistent(&tree)) {
			return "internal consistency check for the fuse and split tree failed";
		}

		// outer (logical and auxiliary) 'j' quantum numbers
		qnumber j0list[] = { 2 };
		qnumber j1list[] = { 0, 2 };
		qnumber j2list[] = { 1, 3, 5 };
		qnumber j3list[] = { 1, 3 };
		const struct su2_irreducible_list outer_irreps[4] = {
			{ .jlist = j0list, .num = ARRLEN(j0list) },
			{ .jlist = j1list, .num = ARRLEN(j1list) },
			{ .jlist = j2list, .num = ARRLEN(j2list) },
			{ .jlist = j3list, .num = ARRLEN(j3list) },
		};

		// degeneracy dimensions, indexed by 'j' quantum numbers
		//                          j:  0   1   2   3   4   5
		const ct_long dim_degen0[] = {  0,  0,  5             };
		const ct_long dim_degen1[] = {  7,  0,  3             };
		const ct_long dim_degen2[] = {  0, 57,  0, 42,  0,  8 };  // 57 = 5*7 + 2*7 + 2*4, 42 = 5*4 + 2*7 + 2*4, 8 = 2*4
		const ct_long dim_degen3[] = {  0,  2,  0,  8         };
		const ct_long* dim_degen[] = {
			dim_degen0,
			dim_degen1,
			dim_degen2,
			dim_degen3,
		};

		const int ndim_logical   = 4;
		const int ndim_auxiliary = 0;

		allocate_su2_tensor(CT_DOUBLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &t);

		// delete some charge sectors
		qnumber jlist_del1[5];
		qnumber jlist_del2[5];
		memcpy(jlist_del1, &t.charge_sectors.jlists[0*t.charge_sectors.ndim], t.charge_sectors.ndim*sizeof(qnumber));
		memcpy(jlist_del2, &t.charge_sectors.jlists[3*t.charge_sectors.ndim], t.charge_sectors.ndim*sizeof(qnumber));
		su2_tensor_delete_charge_sector(&t, jlist_del1);
		su2_tensor_delete_charge_sector(&t, jlist_del2);
		su2_tensor_delete_charge_sector_by_index(&t, 6);

		if (!su2_tensor_is_consistent(&t)) {
			return "internal consistency check for SU(2) tensor failed";
		}
		if (t.charge_sectors.nsec == 0) {
			return "expecting at least one charge sector in SU(2) tensor";
		}

		// fill degeneracy tensors with random entries
		struct rng_state rng_state;
		seed_rng_state(42, &rng_state);
		su2_tensor_fill_random_normal(numeric_one(t.dtype), numeric_zero(t.dtype), &rng_state, &t);
	}

	// split axis
	const int i_ax_split = 2;
	const int i_ax_add   = 3;
	struct su2_tensor r;
	{
		qnumber j0list[] = { 0, 2 };
		qnumber j1list[] = { 1, 3 };
		const struct su2_irreducible_list outer_irreps[2] = {
			{ .jlist = j0list, .num = ARRLEN(j0list) },
			{ .jlist = j1list, .num = ARRLEN(j1list) },
		};
		// degeneracy dimensions, indexed by 'j' quantum numbers
		//                         j:  0  1  2  3
		const ct_long dim_degen0[] = { 5, 0, 2    };
		const ct_long dim_degen1[] = { 0, 7, 0, 4 };
		const ct_long* dim_degen[] = {
			dim_degen0,
			dim_degen1,
		};
		su2_tensor_split_axis(&t, i_ax_split, i_ax_add, true, outer_irreps, dim_degen, &r);

		if (!su2_tensor_is_consistent(&r)) {
			return "internal consistency check for SU(2) tensor failed";
		}
	}

	// expected fuse and split tree after axis splitting
	{
		const int ndim = 7;

		// construct the fuse and split tree
		//
		//  0    4   1
		//   ╲  ╱   ╱
		//    ╲╱   ╱       fuse
		//    5╲  ╱
		//      ╲╱
		//      │
		//      │6
		//      │
		//      ╱╲
		//     ╱  ╲        split
		//    2    3
		//
		struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
		struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
		struct su2_tree_node j5  = { .i_ax = 5, .c = { &j0,  &j4  } };
		struct su2_tree_node j6f = { .i_ax = 6, .c = { &j5,  &j1  } };
		struct su2_tree_node j6s = { .i_ax = 6, .c = { &j2,  &j3  } };

		struct su2_fuse_split_tree tree_ref = { .tree_fuse = &j6f, .tree_split = &j6s, .ndim = ndim };

		if (!su2_fuse_split_tree_is_consistent(&tree_ref)) {
			return "internal consistency check for the fuse and split tree failed";
		}

		// compare
		if (!su2_fuse_split_tree_equal(&r.tree, &tree_ref)) {
			return "fuse and split tree after axis splitting does not match reference";
		}
	}

	// construct explicit splitting operation as dense tensor
	struct dense_tensor u_split_dns;
	{
		struct su2_tensor u_split;

		const int ndim = 3;

		// construct the fuse and split tree
		//
		//      0
		//      │
		//      │
		//      ╱╲
		//     ╱  ╲
		//    1    2
		//
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j0f = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j0s = { .i_ax = 0, .c = { &j1,  &j2  } };

		struct su2_fuse_split_tree tree = { .tree_fuse = &j0f, .tree_split = &j0s, .ndim = ndim };

		if (!su2_fuse_split_tree_is_consistent(&tree)) {
			return "internal consistency check for the fuse and split tree failed";
		}

		// outer (logical and auxiliary) 'j' quantum numbers
		const struct su2_irreducible_list outer_irreps[3] = {
			t.outer_irreps[i_ax_split],
			r.outer_irreps[i_ax_split],
			r.outer_irreps[i_ax_add],
		};
		// degeneracy dimensions
		const ct_long* dim_degen[3] = {
			t.dim_degen[i_ax_split],
			r.dim_degen[i_ax_split],
			r.dim_degen[i_ax_add],
		};

		const int ndim_logical   = 3;
		const int ndim_auxiliary = 0;

		allocate_su2_tensor(CT_DOUBLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &u_split);

		if (!su2_tensor_is_consistent(&u_split)) {
			return "internal consistency check for SU(2) tensor failed";
		}
		if (u_split.charge_sectors.nsec == 0) {
			return "expecting at least one charge sector in SU(2) tensor";
		}

		for (ct_long c = 0; c < u_split.charge_sectors.nsec; c++)
		{
			// current 'j' quantum numbers
			const qnumber* jlist = &u_split.charge_sectors.jlists[c * u_split.charge_sectors.ndim];

			// corresponding "degeneracy" tensor
			struct dense_tensor* d = u_split.degensors[c];
			assert(d != NULL);
			assert(d->dtype == u_split.dtype);
			assert(d->ndim  == u_split.ndim_logical);

			// set "degeneracy" tensor to (padded) identity
			struct dense_tensor identity;
			const ct_long dim_id[2] = { d->dim[1]*d->dim[2], d->dim[1]*d->dim[2] };
			allocate_dense_tensor(d->dtype, 2, dim_id, &identity);
			dense_tensor_set_identity(&identity);
			ct_long pb = 0;
			ct_long pa = 0;
			for (int k = 0; k < u_split.outer_irreps[1].num; k++)
			{
				const qnumber j0 = u_split.outer_irreps[1].jlist[k];

				for (int l = 0; l < u_split.outer_irreps[2].num; l++)
				{
					const qnumber j1 = u_split.outer_irreps[2].jlist[l];

					if (abs(j0 - j1) <= jlist[0] && jlist[0] <= j0 + j1 && (jlist[0] + j0 + j1) % 2 == 0)
					{
						if (j0 < jlist[1] || (j0 == jlist[1] && j1 < jlist[2]))
						{
							pb += u_split.dim_degen[1][j0] * u_split.dim_degen[2][j1];
						}
						else if (j0 > jlist[1] || (j0 == jlist[1] && j1 > jlist[2]))
						{
							pa += u_split.dim_degen[1][j0] * u_split.dim_degen[2][j1];
						}
					}
				}
			}
			const ct_long pad_before[2] = { pb, 0 };
			const ct_long pad_after[2]  = { pa, 0 };
			struct dense_tensor identity_pad;
			dense_tensor_pad_zeros(&identity, pad_before, pad_after, &identity_pad);
			delete_dense_tensor(&identity);
			reshape_dense_tensor(d->ndim, d->dim, &identity_pad);
			delete_dense_tensor(d);
			*u_split.degensors[c] = identity_pad;  // copy internal data pointers
		}

		su2_to_dense_tensor(&u_split, &u_split_dns);

		delete_su2_tensor(&u_split);
	}

	struct dense_tensor t_dns;
	su2_to_dense_tensor(&t, &t_dns);

	// contract with dense splitting tensor to obtain reference tensor
	struct dense_tensor r_dns_ref;
	dense_tensor_multiply_axis(&t_dns, i_ax_split, &u_split_dns, TENSOR_AXIS_RANGE_LEADING, &r_dns_ref);

	// compare
	if (!dense_su2_tensor_allclose(&r_dns_ref, &r, 1e-13)) {
		return "tensor resulting from splitting an axis of an SU(2) tensor does not match reference";
	}

	// fuse axes again
	struct su2_tensor t2;
	su2_tensor_fuse_axes(&r, i_ax_split, i_ax_add, &t2);

	if (!su2_tensor_is_consistent(&t2)) {
		return "internal consistency check for SU(2) tensor failed";
	}

	// compare
	if (!dense_su2_tensor_allclose(&t_dns, &t2, 1e-13)) {
		return "tensor after axis splitting and fusion should agree with original tensor";
	}

	delete_dense_tensor(&u_split_dns);
	delete_dense_tensor(&r_dns_ref);
	delete_dense_tensor(&t_dns);
	delete_su2_tensor(&t2);
	delete_su2_tensor(&r);
	delete_su2_tensor(&t);

	return 0;
}


char* test_su2_tensor_contract_simple()
{
	struct rng_state rng_state;
	seed_rng_state(43, &rng_state);

	// construct the 's' tensor
	struct su2_tensor s;
	{
		const int ndim = 9;

		// construct the fuse and split tree
		//
		//  5   0    2
		//   ╲   ╲  ╱
		//    ╲   ╲╱         fuse
		//     ╲  ╱6
		//      ╲╱
		//      │
		//      │7
		//      │
		//      ╱╲
		//    8╱  ╲          split
		//    ╱╲   ╲
		//   ╱  ╲   ╲
		//  3    1   4
		//
		struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
		struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
		struct su2_tree_node j5  = { .i_ax = 5, .c = { NULL, NULL } };
		struct su2_tree_node j6  = { .i_ax = 6, .c = { &j0,  &j2  } };
		struct su2_tree_node j8  = { .i_ax = 8, .c = { &j3,  &j1  } };
		struct su2_tree_node j7f = { .i_ax = 7, .c = { &j5,  &j6  } };
		struct su2_tree_node j7s = { .i_ax = 7, .c = { &j8,  &j4  } };

		struct su2_fuse_split_tree tree = { .tree_fuse = &j7f, .tree_split = &j7s, .ndim = ndim };

		if (!su2_fuse_split_tree_is_consistent(&tree)) {
			return "internal consistency check for the fuse and split tree failed";
		}

		// outer (logical and auxiliary) 'j' quantum numbers
		qnumber j0list[] = { 1, 3 };
		qnumber j1list[] = { 3 };
		qnumber j2list[] = { 0, 2 };
		qnumber j3list[] = { 2 };
		qnumber j4list[] = { 0 };  // auxiliary
		qnumber j5list[] = { 0 };  // auxiliary
		const struct su2_irreducible_list outer_irreps[6] = {
			{ .jlist = j0list, .num = ARRLEN(j0list) },
			{ .jlist = j1list, .num = ARRLEN(j1list) },
			{ .jlist = j2list, .num = ARRLEN(j2list) },
			{ .jlist = j3list, .num = ARRLEN(j3list) },
			{ .jlist = j4list, .num = ARRLEN(j4list) },
			{ .jlist = j5list, .num = ARRLEN(j5list) },
		};

		// degeneracy dimensions, indexed by 'j' quantum numbers
		//                         j:  0  1  2  3
		const ct_long dim_degen0[] = { 0, 4, 0, 2 };
		const ct_long dim_degen1[] = { 0, 0, 0, 3 };
		const ct_long dim_degen2[] = { 5, 0, 2    };
		const ct_long dim_degen3[] = { 0, 0, 7    };
		const ct_long* dim_degen[] = {
			dim_degen0,
			dim_degen1,
			dim_degen2,
			dim_degen3,
		};

		const int ndim_logical   = 4;
		const int ndim_auxiliary = 2;

		allocate_su2_tensor(CT_DOUBLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &s);

		if (!su2_tensor_is_consistent(&s)) {
			return "internal consistency check for SU(2) tensor failed";
		}
		if (s.charge_sectors.nsec == 0) {
			return "expecting at least one charge sector in SU(2) tensor";
		}

		// fill degeneracy tensors with random entries
		su2_tensor_fill_random_normal(numeric_one(s.dtype), numeric_zero(s.dtype), &rng_state, &s);
	}

	// construct the 't' tensor
	struct su2_tensor t;
	{
		const int ndim = 9;

		// construct the fuse and split tree
		//
		//  3    2   5   0
		//   ╲  ╱   ╱   ╱
		//    ╲╱   ╱   ╱
		//    6╲  ╱   ╱        fuse
		//      ╲╱   ╱
		//      8╲  ╱
		//        ╲╱
		//        │
		//        │7
		//        │
		//        ╱╲
		//       ╱  ╲          split
		//      1    4
		//
		struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
		struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
		struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
		struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
		struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
		struct su2_tree_node j5  = { .i_ax = 5, .c = { NULL, NULL } };
		struct su2_tree_node j6  = { .i_ax = 6, .c = { &j3,  &j2  } };
		struct su2_tree_node j8  = { .i_ax = 8, .c = { &j6,  &j5  } };
		struct su2_tree_node j7f = { .i_ax = 7, .c = { &j8,  &j0  } };
		struct su2_tree_node j7s = { .i_ax = 7, .c = { &j1,  &j4  } };

		struct su2_fuse_split_tree tree = { .tree_fuse = &j7f, .tree_split = &j7s, .ndim = ndim };

		if (!su2_fuse_split_tree_is_consistent(&tree)) {
			return "internal consistency check for the fuse and split tree failed";
		}

		// outer (logical and auxiliary) 'j' quantum numbers
		qnumber j0list[] = { 0, 2 };
		qnumber j1list[] = { 3, 5 };
		qnumber j2list[] = { 3 };
		qnumber j3list[] = { 2 };
		qnumber j4list[] = { 4 };
		qnumber j5list[] = { 0 };  // auxiliary
		const struct su2_irreducible_list outer_irreps[6] = {
			{ .jlist = j0list, .num = ARRLEN(j0list) },
			{ .jlist = j1list, .num = ARRLEN(j1list) },
			{ .jlist = j2list, .num = ARRLEN(j2list) },
			{ .jlist = j3list, .num = ARRLEN(j3list) },
			{ .jlist = j4list, .num = ARRLEN(j4list) },
			{ .jlist = j5list, .num = ARRLEN(j5list) },
		};

		// degeneracy dimensions, indexed by 'j' quantum numbers
		//                         j:  0  1  2  3  4  5
		const ct_long dim_degen0[] = { 5, 0, 2          };
		const ct_long dim_degen1[] = { 0, 0, 0, 2, 0, 4 };
		const ct_long dim_degen2[] = { 0, 0, 0, 3       };
		const ct_long dim_degen3[] = { 0, 0, 7          };
		const ct_long dim_degen4[] = { 0, 0, 0, 0, 3    };
		const ct_long* dim_degen[] = {
			dim_degen0,
			dim_degen1,
			dim_degen2,
			dim_degen3,
			dim_degen4,
		};

		const int ndim_logical   = 5;
		const int ndim_auxiliary = 1;

		allocate_su2_tensor(CT_DOUBLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &t);

		if (!su2_tensor_is_consistent(&t)) {
			return "internal consistency check for SU(2) tensor failed";
		}
		if (t.charge_sectors.nsec == 0) {
			return "expecting at least one charge sector in SU(2) tensor";
		}

		// fill degeneracy tensors with random entries
		su2_tensor_fill_random_normal(numeric_one(t.dtype), numeric_zero(t.dtype), &rng_state, &t);
	}

	for (int flip = 0; flip < 2; flip++)
	{
		if (flip == 1) {
			su2_tensor_flip_trees(&s);
			su2_tensor_flip_trees(&t);
		}

		for (int ordering = 0; ordering < 2; ordering++)
		{
			// perform contraction
			const int i_ax_s[3] = { 3, 1, 4 };
			const int i_ax_t[3] = { 3, 2, 5 };
			struct su2_tensor r;
			if (ordering == 0) {
				su2_tensor_contract_simple(&s, i_ax_s, &t, i_ax_t, 3, &r);
			}
			else {
				su2_tensor_contract_simple(&t, i_ax_t, &s, i_ax_s, 3, &r);
			}

			if (!su2_tensor_is_consistent(&r)) {
				return "internal consistency check for SU(2) tensor failed";
			}
			if (r.ndim_logical != 5) {
				return "contracted SU(2) tensor does not have expected number of logical dimensions";
			}
			if (r.ndim_auxiliary != 1) {
				return "contracted SU(2) tensor does not have expected number of auxiliary dimensions";
			}

			// reference SU(2) tree of contracted tensor
			if (ordering == 0)
			{
				// construct the fuse and split tree
				//
				//  5   0    1   2
				//   ╲   ╲  ╱   ╱
				//    ╲   ╲╱   ╱
				//     ╲  ╱6  ╱        fuse
				//      ╲╱   ╱
				//      7╲  ╱
				//        ╲╱
				//        │
				//        │8
				//        │
				//        ╱╲
				//       ╱  ╲          split
				//      3    4
				//
				struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
				struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
				struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
				struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
				struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
				struct su2_tree_node j5  = { .i_ax = 5, .c = { NULL, NULL } };
				struct su2_tree_node j6  = { .i_ax = 6, .c = { &j0,  &j1  } };
				struct su2_tree_node j7  = { .i_ax = 7, .c = { &j5,  &j6  } };
				struct su2_tree_node j8f = { .i_ax = 8, .c = { &j7,  &j2  } };
				struct su2_tree_node j8s = { .i_ax = 8, .c = { &j3,  &j4  } };

				struct su2_fuse_split_tree tree_ref = { .tree_fuse = &j8f, .tree_split = &j8s, .ndim = 9 };
				if (!su2_fuse_split_tree_is_consistent(&tree_ref)) {
					return "internal consistency check for the fuse and split tree failed";
				}

				if (flip == 1) {
					su2_fuse_split_tree_flip(&tree_ref);
				}

				if (!su2_fuse_split_tree_equal(&r.tree, &tree_ref)) {
					return "fuse and split tree of contracted SU(2) tensor does not match reference";
				}
			}
			else  // ordering == 1
			{
				// construct the fuse and split tree
				//
				//  5   3    4   0
				//   ╲   ╲  ╱   ╱
				//    ╲   ╲╱   ╱
				//     ╲  ╱8  ╱        fuse
				//      ╲╱   ╱
				//      7╲  ╱
				//        ╲╱
				//        │
				//        │6
				//        │
				//        ╱╲
				//       ╱  ╲          split
				//      1    2
				//
				struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
				struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
				struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
				struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
				struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
				struct su2_tree_node j5  = { .i_ax = 5, .c = { NULL, NULL } };
				struct su2_tree_node j8  = { .i_ax = 8, .c = { &j3,  &j4  } };
				struct su2_tree_node j7  = { .i_ax = 7, .c = { &j5,  &j8  } };
				struct su2_tree_node j6f = { .i_ax = 6, .c = { &j7,  &j0  } };
				struct su2_tree_node j6s = { .i_ax = 6, .c = { &j1,  &j2  } };

				struct su2_fuse_split_tree tree_ref = { .tree_fuse = &j6f, .tree_split = &j6s, .ndim = 9 };
				if (!su2_fuse_split_tree_is_consistent(&tree_ref)) {
					return "internal consistency check for the fuse and split tree failed";
				}

				if (flip == 1) {
					su2_fuse_split_tree_flip(&tree_ref);
				}

				if (!su2_fuse_split_tree_equal(&r.tree, &tree_ref)) {
					return "fuse and split tree of contracted SU(2) tensor does not match reference";
				}
			}

			// convert to full dense tensors
			struct dense_tensor s_dns;
			struct dense_tensor t_dns;
			su2_to_dense_tensor(&s, &s_dns);
			su2_to_dense_tensor(&t, &t_dns);

			if (dense_tensor_is_zero(&s_dns, 0.) || dense_tensor_is_zero(&t_dns, 0.)) {
				return "to-be contracted SU(2) tensors should not be zero";
			}

			struct dense_tensor s_dns_perm;
			struct dense_tensor t_dns_perm;
			const int perm_s[4] = { 0, 2, 3, 1 };
			const int perm_t[5] = { 3, 2, 0, 1, 4 };
			dense_tensor_transpose(perm_s, &s_dns, &s_dns_perm);
			dense_tensor_transpose(perm_t, &t_dns, &t_dns_perm);

			// contract dense tensors, as reference
			struct dense_tensor r_dns_ref;
			if (ordering == 0) {
				dense_tensor_dot(&s_dns_perm, TENSOR_AXIS_RANGE_TRAILING, &t_dns_perm, TENSOR_AXIS_RANGE_LEADING, 2, &r_dns_ref);
			}
			else {
				dense_tensor_dot(&t_dns_perm, TENSOR_AXIS_RANGE_LEADING, &s_dns_perm, TENSOR_AXIS_RANGE_TRAILING, 2, &r_dns_ref);
			}

			// compare
			if (!dense_su2_tensor_allclose(&r_dns_ref, &r, 1e-13)) {
				return "tensor resulting from contraction of two SU(2) tensors does not match reference";
			}

			delete_dense_tensor(&r_dns_ref);
			delete_dense_tensor(&t_dns_perm);
			delete_dense_tensor(&s_dns_perm);
			delete_dense_tensor(&t_dns);
			delete_dense_tensor(&s_dns);
			delete_su2_tensor(&r);
		}
	}

	delete_su2_tensor(&t);
	delete_su2_tensor(&s);

	return 0;
}


char* test_su2_tensor_contract_yoga()
{
	struct rng_state rng_state;
	seed_rng_state(44, &rng_state);

	// yoga tree variant
	for (int variant = 0; variant < 2; variant++)
	{
		// construct an SU(2) tensor 's'
		struct su2_tensor s;
		{
			const int ndim = 5;

			// construct the fuse and split tree (with flipped 1 <-> 3 for variant 1)
			//
			//  3    1
			//   ╲  ╱     fuse
			//    ╲╱
			//    │
			//    │4
			//    │
			//    ╱╲
			//   ╱  ╲     split
			//  2    0
			//
			struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
			struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
			struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
			struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
			struct su2_tree_node j4f = { .i_ax = 4, .c = { &j3,  &j1  } };
			struct su2_tree_node j4s = { .i_ax = 4, .c = { &j2,  &j0  } };
			if (variant == 1)
			{
				j4f.c[0] = &j1;
				j4f.c[1] = &j3;
			}

			struct su2_fuse_split_tree tree = { .tree_fuse = &j4f, .tree_split = &j4s, .ndim = ndim };

			if (!su2_fuse_split_tree_is_consistent(&tree)) {
				return "internal consistency check for the fuse and split tree failed";
			}

			// outer (logical and auxiliary) 'j' quantum numbers
			qnumber j0list[] = { 1, 5 };
			qnumber j1list[] = { 1, 3 };
			qnumber j2list[] = { 1, 3 };
			qnumber j3list[] = { 1 };
			const struct su2_irreducible_list outer_irreps[4] = {
				{ .jlist = j0list, .num = ARRLEN(j0list) },
				{ .jlist = j1list, .num = ARRLEN(j1list) },
				{ .jlist = j2list, .num = ARRLEN(j2list) },
				{ .jlist = j3list, .num = ARRLEN(j3list) },
			};

			// degeneracy dimensions, indexed by 'j' quantum numbers
			//                         j:  0  1  2  3  4  5
			const ct_long dim_degen0[] = { 0, 4, 0, 0, 0, 2 };
			const ct_long dim_degen1[] = { 0, 5, 0, 2       };
			const ct_long dim_degen2[] = { 0, 3, 0, 3       };
			const ct_long dim_degen3[] = { 0, 7             };
			const ct_long* dim_degen[] = {
				dim_degen0,
				dim_degen1,
				dim_degen2,
				dim_degen3,
			};

			const int ndim_logical   = 4;
			const int ndim_auxiliary = 0;

			allocate_su2_tensor(CT_SINGLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &s);

			// delete some charge sectors, to probe yoga transformation with missing sectors
			const qnumber jlist_del1[5] = { 1, 1, 1, 1, 2 };
			const qnumber jlist_del2[5] = { 5, 1, 3, 1, 2 };
			const qnumber jlist_del3[5] = { 5, 3, 1, 1, 4 };
			su2_tensor_delete_charge_sector(&s, jlist_del1);
			su2_tensor_delete_charge_sector(&s, jlist_del2);
			su2_tensor_delete_charge_sector(&s, jlist_del3);

			if (!su2_tensor_is_consistent(&s)) {
				return "internal consistency check for SU(2) tensor failed";
			}
			if (s.charge_sectors.nsec == 0) {
				return "expecting at least one charge sector in SU(2) tensor";
			}

			// fill degeneracy tensors with random entries
			su2_tensor_fill_random_normal(numeric_one(s.dtype), numeric_zero(s.dtype), &rng_state, &s);
		}

		// construct an SU(2) tensor 't'
		struct su2_tensor t;
		{
			const int ndim = 7;

			// construct the fuse and split tree (with flipped 2 <-> 3 for variant 1)
			//
			//  0   4    1
			//   ╲   ╲  ╱
			//    ╲   ╲╱       fuse
			//     ╲  ╱6
			//      ╲╱
			//      │
			//      │5
			//      │
			//      ╱╲
			//     ╱  ╲        split
			//    2    3
			//
			struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
			struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
			struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
			struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
			struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
			struct su2_tree_node j6  = { .i_ax = 6, .c = { &j4,  &j1  } };
			struct su2_tree_node j5f = { .i_ax = 5, .c = { &j0,  &j6  } };
			struct su2_tree_node j5s = { .i_ax = 5, .c = { &j2,  &j3  } };
			if (variant == 1)
			{
				j5s.c[0] = &j3;
				j5s.c[1] = &j2;
			}

			struct su2_fuse_split_tree tree = { .tree_fuse = &j5f, .tree_split = &j5s, .ndim = ndim };

			if (!su2_fuse_split_tree_is_consistent(&tree)) {
				return "internal consistency check for the fuse and split tree failed";
			}

			// outer (logical and auxiliary) 'j' quantum numbers
			qnumber j0list[] = { 3 };
			qnumber j1list[] = { 0, 2 };
			qnumber j2list[] = { 1, 3 };
			qnumber j3list[] = { 2 };
			qnumber j4list[] = { 0 };  // auxiliary
			const struct su2_irreducible_list outer_irreps[5] = {
				{ .jlist = j0list, .num = ARRLEN(j0list) },
				{ .jlist = j1list, .num = ARRLEN(j1list) },
				{ .jlist = j2list, .num = ARRLEN(j2list) },
				{ .jlist = j3list, .num = ARRLEN(j3list) },
				{ .jlist = j4list, .num = ARRLEN(j4list) },
			};

			// degeneracy dimensions, indexed by 'j' quantum numbers
			//                         j:  0  1  2  3
			const ct_long dim_degen0[] = { 0, 0, 0, 3 };
			const ct_long dim_degen1[] = { 3, 0, 4    };
			const ct_long dim_degen2[] = { 0, 5, 0, 2 };
			const ct_long dim_degen3[] = { 0, 0, 7    };
			const ct_long* dim_degen[] = {
				dim_degen0,
				dim_degen1,
				dim_degen2,
				dim_degen3,
			};

			const int ndim_logical   = 4;
			const int ndim_auxiliary = 1;

			allocate_su2_tensor(CT_SINGLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &t);

			// delete some charge sectors, to probe yoga transformation with missing sectors
			const qnumber jlist_del1[7] = { 3, 0, 3, 2, 0, 3, 0 };
			const qnumber jlist_del2[7] = { 3, 0, 1, 2, 0, 3, 0 };
			su2_tensor_delete_charge_sector(&t, jlist_del1);
			su2_tensor_delete_charge_sector(&t, jlist_del2);

			if (!su2_tensor_is_consistent(&t)) {
				return "internal consistency check for SU(2) tensor failed";
			}
			if (t.charge_sectors.nsec == 0) {
				return "expecting at least one charge sector in SU(2) tensor";
			}

			// fill degeneracy tensors with random entries
			su2_tensor_fill_random_normal(numeric_one(t.dtype), numeric_zero(t.dtype), &rng_state, &t);
		}

		// perform contraction
		const int i_ax_s = 1;
		const int i_ax_t = 2;
		struct su2_tensor r;
		su2_tensor_contract_yoga(&s, i_ax_s, &t, i_ax_t, &r);

		if (!su2_tensor_is_consistent(&r)) {
			return "internal consistency check for SU(2) tensor failed";
		}
		if (r.ndim_logical != 6) {
			return "contracted SU(2) tensor does not have expected number of logical dimensions";
		}
		if (r.ndim_auxiliary != 1) {
			return "contracted SU(2) tensor does not have expected number of auxiliary dimensions";
		}

		// convert to full dense tensors
		struct dense_tensor s_dns;
		struct dense_tensor t_dns;
		su2_to_dense_tensor(&s, &s_dns);
		su2_to_dense_tensor(&t, &t_dns);

		if (dense_tensor_is_zero(&s_dns, 0.) || dense_tensor_is_zero(&t_dns, 0.)) {
			return "to-be contracted SU(2) tensors should not be zero";
		}

		struct dense_tensor s_dns_perm;
		struct dense_tensor t_dns_perm;
		const int perm_s[4] = { 0, 2, 3, 1 };
		const int perm_t[4] = { 2, 0, 1, 3 };
		dense_tensor_transpose(perm_s, &s_dns, &s_dns_perm);
		dense_tensor_transpose(perm_t, &t_dns, &t_dns_perm);

		// contract dense tensors, as reference
		struct dense_tensor r_dns_ref;
		dense_tensor_dot(&s_dns_perm, TENSOR_AXIS_RANGE_TRAILING, &t_dns_perm, TENSOR_AXIS_RANGE_LEADING, 1, &r_dns_ref);

		// compare
		if (!dense_su2_tensor_allclose(&r_dns_ref, &r, 1e-5)) {
			return "tensor resulting from contraction of two SU(2) tensors does not match reference";
		}

		delete_dense_tensor(&r_dns_ref);
		delete_dense_tensor(&t_dns_perm);
		delete_dense_tensor(&s_dns_perm);
		delete_dense_tensor(&t_dns);
		delete_dense_tensor(&s_dns);

		delete_su2_tensor(&r);
		delete_su2_tensor(&t);
		delete_su2_tensor(&s);
	}

	return 0;
}


char* test_su2_to_dense_tensor()
{
	hid_t file = H5Fopen("../test/tensor/data/test_su2_to_dense_tensor.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_su2_to_dense_tensor failed";
	}

	const hid_t hdf5_dcomplex_id = construct_hdf5_double_complex_dtype(false);

	const int ndim = 7;

	// construct the fuse and split tree
	//
	//  2    4   0
	//   ╲  ╱   ╱
	//    ╲╱   ╱         fuse
	//    5╲  ╱
	//      ╲╱
	//      │
	//      │6
	//      │
	//      ╱╲
	//     ╱  ╲          split
	//    1    3
	//
	struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
	struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
	struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
	struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
	struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
	struct su2_tree_node j5  = { .i_ax = 5, .c = { &j2,  &j4  } };
	struct su2_tree_node j6f = { .i_ax = 6, .c = { &j5,  &j0  } };
	struct su2_tree_node j6s = { .i_ax = 6, .c = { &j1,  &j3  } };

	struct su2_fuse_split_tree tree = { .tree_fuse = &j6f, .tree_split = &j6s, .ndim = ndim };

	if (!su2_fuse_split_tree_is_consistent(&tree)) {
		return "internal consistency check for the fuse and split tree failed";
	}

	// outer (logical and auxiliary) 'j' quantum numbers
	qnumber j0list[] = { 0, 2, 4 };
	qnumber j1list[] = { 3, 5 };
	qnumber j2list[] = { 1, 5 };
	qnumber j3list[] = { 2, 4 };
	qnumber j4list[] = { 0 };  // auxiliary
	const struct su2_irreducible_list outer_irreps[5] = {
		{ .jlist = j0list, .num = ARRLEN(j0list) },
		{ .jlist = j1list, .num = ARRLEN(j1list) },
		{ .jlist = j2list, .num = ARRLEN(j2list) },
		{ .jlist = j3list, .num = ARRLEN(j3list) },
		{ .jlist = j4list, .num = ARRLEN(j4list) },
	};

	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                         j:  0  1  2  3  4  5
	const ct_long dim_degen0[] = { 6, 0, 3, 0, 5    };
	const ct_long dim_degen1[] = { 0, 0, 0, 7, 0, 4 };
	const ct_long dim_degen2[] = { 0, 8, 0, 0, 0, 1 };
	const ct_long dim_degen3[] = { 0, 0, 9, 0, 2    };
	const ct_long* dim_degen[] = {
		dim_degen0,
		dim_degen1,
		dim_degen2,
		dim_degen3,
	};

	const int ndim_logical   = 4;
	const int ndim_auxiliary = 1;

	struct su2_tensor t;
	allocate_su2_tensor(CT_DOUBLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_irreps, dim_degen, &t);

	if (!su2_tensor_is_consistent(&t)) {
		return "internal consistency check for SU(2) tensor failed";
	}
	if (t.charge_sectors.nsec == 0) {
		return "expecting at least one charge sector in SU(2) tensor";
	}

	// fill degeneracy tensors with random entries
	struct rng_state rng_state;
	seed_rng_state(45, &rng_state);
	su2_tensor_fill_random_normal(numeric_one(t.dtype), numeric_zero(t.dtype), &rng_state, &t);

	// convert to full dense tensor
	struct dense_tensor t_dns;
	su2_to_dense_tensor(&t, &t_dns);

	if (t_dns.ndim != ndim_logical) {
		return "degree of logical dense tensor does not match expected value";
	}

	// apply simultaneous rotations
	struct dense_tensor r;
	for (int i = 0; i < ndim_logical; i++)
	{
		const ct_long d = su2_tensor_dim_logical_axis(&t, i);
		const ct_long dim[2] = { d, d };
		struct dense_tensor w;
		allocate_dense_tensor(CT_DOUBLE_COMPLEX, 2, dim, &w);

		char varname[1024];
		sprintf(varname, "w%i", i);
		if (read_hdf5_dataset(file, varname, hdf5_dcomplex_id, w.data) < 0) {
			return "reading tensor entries from disk failed";
		}

		if (su2_tensor_logical_axis_direction(&t, i) == TENSOR_AXIS_IN)
		{
			// logically applying conjugate-transpose rotation:
			// conjugate entries, logical transposition corresponds to same multiplication axes
			conjugate_dense_tensor(&w);
		}

		if (i == 0)
		{
			dense_tensor_multiply_axis(&t_dns, i, &w, TENSOR_AXIS_RANGE_TRAILING, &r);
		}
		else
		{
			struct dense_tensor s = r;  // copy internal data pointers
			dense_tensor_multiply_axis(&s, i, &w, TENSOR_AXIS_RANGE_TRAILING, &r);
			delete_dense_tensor(&s);
		}

		delete_dense_tensor(&w);
	}

	// rotations must leave tensor invariant
	if (!dense_tensor_allclose(&r, &t_dns, 1e-13)) {
		return "SU(2) tensor not invariant under symmetry rotations";
	}

	// clean up
	delete_dense_tensor(&r);
	delete_dense_tensor(&t_dns);
	delete_su2_tensor(&t);

	H5Tclose(hdf5_dcomplex_id);
	H5Fclose(file);

	return 0;
}


char* test_su2_tensor_qr()
{
	struct rng_state rng_state;
	seed_rng_state(46, &rng_state);

	// decomposition modes
	for (int mode = 0; mode < QR_NUM_MODES; mode++)
	{
		// whether the leading axis is in the fusion or splitting tree
		for (int lfs = 0; lfs < 2; lfs++)
		{
			// whether the auxiliary axis is grouped together with the first or second axis
			for (int iaux = 0; iaux < 2; iaux++)
			{
				// construct an SU(2) tensor 'a'
				struct su2_tensor a;
				{
					struct su2_fuse_split_tree tree;
					if (iaux == 0)
					{
						// construct the fuse and split tree
						//
						//      0
						//      │
						//      │
						//      ╱╲
						//     ╱  ╲
						//    1    2
						//
						struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
						struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
						struct su2_tree_node j0f = { .i_ax = 0, .c = { NULL, NULL } };
						struct su2_tree_node j0s = { .i_ax = 0, .c = { &j1,  &j2  } };

						struct su2_fuse_split_tree t = { .tree_fuse = &j0f, .tree_split = &j0s, .ndim = 3 };

						copy_su2_fuse_split_tree(&t, &tree);
					}
					else
					{
						// construct the fuse and split tree
						//
						//    0    2
						//     ╲  ╱
						//      ╲╱
						//      │
						//      │
						//      1
						//
						struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
						struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
						struct su2_tree_node j1f = { .i_ax = 1, .c = { &j0,  &j2  } };
						struct su2_tree_node j1s = { .i_ax = 1, .c = { NULL, NULL } };

						struct su2_fuse_split_tree t = { .tree_fuse = &j1f, .tree_split = &j1s, .ndim = 3 };

						copy_su2_fuse_split_tree(&t, &tree);
					}

					if (lfs == 1) {
						su2_fuse_split_tree_flip(&tree);
					}

					if (!su2_fuse_split_tree_is_consistent(&tree)) {
						return "internal consistency check for the fuse and split tree failed";
					}

					// outer (logical and auxiliary) 'j' quantum numbers
					qnumber j0list[] = { 0, 1, 3, 7 };  // note: mixed half-integer and integer quantum numbers
					qnumber j1list[] = { 0, 3, 5, 7 };  // note: mixed half-integer and integer quantum numbers
					qnumber j2list[] = { 0 };  // auxiliary
					const struct su2_irreducible_list outer_irreps[3] = {
						{ .jlist = j0list, .num = ARRLEN(j0list) },
						{ .jlist = j1list, .num = ARRLEN(j1list) },
						{ .jlist = j2list, .num = ARRLEN(j2list) },
					};

					// degeneracy dimensions, indexed by 'j' quantum numbers
					//                         j:  0  1  2  3  4  5  6  7
					const ct_long dim_degen0[] = { 2, 5, 0, 8, 0, 0, 0, 6 };
					const ct_long dim_degen1[] = { 9, 0, 0, 7, 0, 3, 0, 4 };
					const ct_long* dim_degen[] = {
						dim_degen0,
						dim_degen1,
					};

					allocate_su2_tensor(CT_DOUBLE_COMPLEX, 2, 1, &tree, outer_irreps, dim_degen, &a);
					delete_su2_fuse_split_tree(&tree);

					// delete a charge sector
					const qnumber jlist_del[3] = { 7, 7, 0 };
					su2_tensor_delete_charge_sector(&a, jlist_del);

					// fill degeneracy tensors with random entries
					su2_tensor_fill_random_normal(numeric_one(a.dtype), numeric_zero(a.dtype), &rng_state, &a);

					if (!su2_tensor_is_consistent(&a)) {
						return "internal consistency check for SU(2) tensor failed";
					}
					if (a.charge_sectors.nsec != 2) {
						return "expecting two charge sectors in SU(2) tensor";
					}
				}

				// perform QR decomposition
				struct su2_tensor q, r;
				if (su2_tensor_qr(&a, mode, &q, &r) < 0) {
					return "SU(2) symmetric QR decomposition failed internally";
				}

				if (!su2_tensor_is_consistent(&q)) {
					return "internal consistency check for SU(2) tensor failed";
				}
				if (!su2_tensor_is_consistent(&r)) {
					return "internal consistency check for SU(2) tensor failed";
				}
				if (q.ndim_logical != 2) {
					return "expecting two logical dimensions for Q tensor from SU(2) symmetric QR decomposition";
				}
				if (r.ndim_logical != 2) {
					return "expecting two logical dimensions for R tensor from SU(2) symmetric QR decomposition";
				}
				if (q.ndim_auxiliary != a.ndim_auxiliary) {
					return "expecting same number of auxiliary dimensions in Q tensor from SU(2) symmetric QR decomposition as in original A tensor";
				}
				if (su2_tensor_dim_logical_axis(&q, 0) != su2_tensor_dim_logical_axis(&a, 0)) {
					return "leading dimension of Q from SU(2) symmetric QR decomposition must be equal to the leading dimension of A";
				}
				if (su2_tensor_dim_logical_axis(&r, 1) != su2_tensor_dim_logical_axis(&a, 1)) {
					return "trailing dimension of R from SU(2) symmetric QR decomposition must be equal to the trailing dimension of A";
				}
				if (mode == QR_COMPLETE) {
					// 'q' must be a logical square matrix
					if (su2_tensor_dim_logical_axis(&q, 0) != su2_tensor_dim_logical_axis(&q, 1)) {
						return "Q from SU(2) symmetric QR decomposition must be a logical square matrix for the \"complete\" mode";
					}
				}
				// 'q' must be an isometry
				if (!su2_tensor_is_isometry(&q, 1e-13, false)) {
					return "Q tensor from SU(2) symmetric QR decomposition is not an isometry";
				}

				// degeneracy tensors in 'r' are upper triangular, but not neccessarily the overall logical matrix

				// logical matrix product 'q r' must be equal to 'a'
				struct su2_tensor qr;
				if (q.tree.tree_fuse->i_ax == 1)
				{
					const int i_ax_q[1] = { 1 };
					const int i_ax_r[1] = { 0 };
					su2_tensor_contract_simple(&q, i_ax_q, &r, i_ax_r, 1, &qr);
				}
				else
				{
					// include auxiliary axis for contraction
					const int i_ax_q[2] = { 1, 2 };
					const int i_ax_r[2] = { 0, 2 };
					// add a dummy auxiliary axis to 'r' to ensure that contracted tensor has three axes
					su2_tensor_add_auxiliary_axis(&r, 1, false);
					su2_tensor_contract_simple(&q, i_ax_q, &r, i_ax_r, 2, &qr);
				}
				// comparing dense tensor representations since number or locations of auxiliary axes might be different
				struct dense_tensor qr_dns;
				su2_to_dense_tensor(&qr, &qr_dns);
				if (!dense_su2_tensor_allclose(&qr_dns, &a, 1e-13)) {
					return "logical matrix product Q R from SU(2) symmetric QR decomposition is not equal to original A tensor";
				}
				delete_dense_tensor(&qr_dns);
				delete_su2_tensor(&qr);

				delete_su2_tensor(&r);
				delete_su2_tensor(&q);
				delete_su2_tensor(&a);
			}
		}
	}

	return 0;
}
