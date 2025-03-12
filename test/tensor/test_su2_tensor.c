#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>
#include "su2_tensor.h"
#include "aligned_memory.h"
#include "rng.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


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
	//                       j:  0  1  2  3  4  5
	const long dim_degen0[3] = { 6, 0, 3          };
	const long dim_degen1[6] = { 0, 0, 0, 7, 0, 4 };
	const long dim_degen2[4] = { 0, 8, 0, 3       };
	const long dim_degen3[5] = { 0, 0, 9, 0, 2    };
	const long dim_degen4[3] = { 5, 0, 3          };
	const long* dim_degen[5] = {
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
	seed_rng_state(39, &rng_state);
	for (long c = 0; c < t.charge_sectors.nsec; c++)
	{
		// corresponding "degeneracy" tensor
		struct dense_tensor* d = t.degensors[c];
		assert(d != NULL);
		assert(d->dtype == t.dtype);
		assert(d->ndim  == t.ndim_logical);

		dense_tensor_fill_random_normal(numeric_one(d->dtype), numeric_zero(d->dtype), &rng_state, d);
	}

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

	// convert to full dense tensor
	struct dense_tensor r_dns;
	su2_to_dense_tensor(&r, &r_dns);

	// compare with original tensor
	if (!dense_tensor_allclose(&r_dns, &t_dns, 1e-13)) {
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

	// convert to full dense tensor
	struct dense_tensor t2_dns;
	su2_to_dense_tensor(&t2, &t2_dns);

	// compare with original tensor
	if (!dense_tensor_allclose(&t2_dns, &t_dns, 1e-13)) {
		return "F-move applied to SU(2) tensor must leave logical tensor invariant";
	}

	// clean up
	delete_dense_tensor(&t2_dns);
	delete_dense_tensor(&r_dns);
	delete_dense_tensor(&t_dns);
	delete_su2_tensor(&t2);
	delete_su2_tensor(&r);
	delete_su2_tensor(&t);

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
		//                       j:  0  1  2  3  4  5
		const long dim_degen0[3] = { 0, 0, 3          };
		const long dim_degen1[3] = { 3, 0, 4          };
		const long dim_degen2[4] = { 0, 5, 0, 2       };
		const long dim_degen3[6] = { 0, 7, 0, 0, 0, 8 };
		const long* dim_degen[4] = {
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
		for (long c = 0; c < t.charge_sectors.nsec; c++)
		{
			// corresponding "degeneracy" tensor
			struct dense_tensor* d = t.degensors[c];
			assert(d != NULL);
			assert(d->dtype == t.dtype);
			assert(d->ndim  == t.ndim_logical);

			dense_tensor_fill_random_normal(numeric_one(d->dtype), numeric_zero(d->dtype), &rng_state, d);
		}
	}

	// fuse axes
	const int i_ax_0 = 1;
	const int i_ax_1 = 3;
	struct su2_tensor r;
	su2_tensor_fuse_axes(&t, i_ax_0, i_ax_1, &r);

	if (!su2_tensor_is_consistent(&r)) {
		return "internal consistency check for SU(2) tensor failed";
	}

	// convert to full dense tensors
	struct dense_tensor t_dns;
	struct dense_tensor r_dns;
	su2_to_dense_tensor(&t, &t_dns);
	su2_to_dense_tensor(&r, &r_dns);

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
		//                        j:  0   1   2   3   4   5   6   7
		const long dim_degen0[8] = {  0, 49,  0, 60,  0, 56,  0, 32 };  // 49 = 3*7 + 4*7, 60 = 4*7 + 4*8, 56 = 3*8 + 4*8
		const long dim_degen1[3] = {  3,  0,  4                     };
		const long dim_degen2[6] = {  0,  7,  0,  0,  0,  8         };
		const long* dim_degen[3] = {
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

		for (long c = 0; c < u_fuse.charge_sectors.nsec; c++)
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
			const long dim_id[2] = { d->dim[1]*d->dim[2], d->dim[1]*d->dim[2] };
			allocate_dense_tensor(d->dtype, 2, dim_id, &identity);
			dense_tensor_set_identity(&identity);
			long pb;
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
			long pa;
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
			const long pad_before[2] = { pb, 0 };
			const long pad_after[2]  = { pa, 0 };
			struct dense_tensor identity_pad;
			dense_tensor_pad_zeros(&identity, pad_before, pad_after, &identity_pad);
			delete_dense_tensor(&identity);
			reshape_dense_tensor(d->ndim, d->dim, &identity_pad);
			delete_dense_tensor(d);
			move_dense_tensor_data(&identity_pad, u_fuse.degensors[c]);
		}

		su2_to_dense_tensor(&u_fuse, &u_fuse_dns);

		delete_su2_tensor(&u_fuse);
	}

	// contract with dense fusion tensor to obtain reference tensor
	struct dense_tensor r_dns_ref;
	{
		// move to-be fused axes in 't' to the end
		const int perm_t[4] = { 0, 2, 1, 3 };
		struct dense_tensor t_tmp;
		transpose_dense_tensor(perm_t, &t_dns, &t_tmp);

		// perform contraction
		struct dense_tensor r_dns_perm;
		dense_tensor_dot(&t_tmp, TENSOR_AXIS_RANGE_TRAILING, &u_fuse_dns, TENSOR_AXIS_RANGE_TRAILING, 2, &r_dns_perm);
		delete_dense_tensor(&t_tmp);

		// transpose the fused axis to become axis 1
		const int perm_r[3] = { 0, 2, 1 };
		transpose_dense_tensor(perm_r, &r_dns_perm, &r_dns_ref);
		delete_dense_tensor(&r_dns_perm);
	}

	// compare
	if (!dense_tensor_allclose(&r_dns, &r_dns_ref, 1e-5)) {
		return "tensor resulting from fusing two axes of an SU(2) tensor does not match reference";
	}

	// split axis again
	struct su2_tensor t2;
	{
		const struct su2_irreducible_list outer_irreps[2] = {
			t.outer_irreps[i_ax_0],
			t.outer_irreps[i_ax_1],
		};
		const long* dim_degen[2] = {
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

	// convert to a full dense tensor
	struct dense_tensor t2_dns;
	su2_to_dense_tensor(&t2, &t2_dns);

	// compare
	if (!dense_tensor_allclose(&t2_dns, &t_dns, 1e-6)) {
		return "tensor after axis fusion and splitting should agree with original tensor";
	}

	delete_dense_tensor(&t2_dns);
	delete_dense_tensor(&r_dns_ref);
	delete_dense_tensor(&u_fuse_dns);
	delete_dense_tensor(&r_dns);
	delete_dense_tensor(&t_dns);
	delete_su2_tensor(&t2);
	delete_su2_tensor(&r);
	delete_su2_tensor(&t);

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
		//                        j:  0   1   2   3   4   5
		const long dim_degen0[3] = {  0,  0,  5             };
		const long dim_degen1[3] = {  7,  0,  3             };
		const long dim_degen2[6] = {  0, 57,  0, 42,  0,  8 };  // 57 = 5*7 + 2*7 + 2*4, 42 = 5*4 + 2*7 + 2*4, 8 = 2*4
		const long dim_degen3[4] = {  0,  2,  0,  8         };
		const long* dim_degen[4] = {
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
		qnumber jlist_del3[5];
		memcpy(jlist_del1, &t.charge_sectors.jlists[0*t.charge_sectors.ndim], t.charge_sectors.ndim*sizeof(qnumber));
		memcpy(jlist_del2, &t.charge_sectors.jlists[3*t.charge_sectors.ndim], t.charge_sectors.ndim*sizeof(qnumber));
		memcpy(jlist_del3, &t.charge_sectors.jlists[8*t.charge_sectors.ndim], t.charge_sectors.ndim*sizeof(qnumber));
		su2_tensor_delete_charge_sector(&t, jlist_del1);
		su2_tensor_delete_charge_sector(&t, jlist_del2);
		su2_tensor_delete_charge_sector(&t, jlist_del3);

		if (!su2_tensor_is_consistent(&t)) {
			return "internal consistency check for SU(2) tensor failed";
		}
		if (t.charge_sectors.nsec == 0) {
			return "expecting at least one charge sector in SU(2) tensor";
		}

		// fill degeneracy tensors with random entries
		struct rng_state rng_state;
		seed_rng_state(41, &rng_state);
		for (long c = 0; c < t.charge_sectors.nsec; c++)
		{
			// corresponding "degeneracy" tensor
			struct dense_tensor* d = t.degensors[c];
			assert(d != NULL);
			assert(d->dtype == t.dtype);
			assert(d->ndim  == t.ndim_logical);

			dense_tensor_fill_random_normal(numeric_one(d->dtype), numeric_zero(d->dtype), &rng_state, d);
		}
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
		//                       j:  0  1  2  3
		const long dim_degen0[3] = { 5, 0, 2    };
		const long dim_degen1[4] = { 0, 7, 0, 4 };
		const long* dim_degen[2] = {
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

	// convert to full dense tensors
	struct dense_tensor t_dns;
	struct dense_tensor r_dns;
	su2_to_dense_tensor(&t, &t_dns);
	su2_to_dense_tensor(&r, &r_dns);

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
		const long* dim_degen[3] = {
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

		for (long c = 0; c < u_split.charge_sectors.nsec; c++)
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
			const long dim_id[2] = { d->dim[1]*d->dim[2], d->dim[1]*d->dim[2] };
			allocate_dense_tensor(d->dtype, 2, dim_id, &identity);
			dense_tensor_set_identity(&identity);
			long pb = 0;
			long pa = 0;
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
			const long pad_before[2] = { pb, 0 };
			const long pad_after[2]  = { pa, 0 };
			struct dense_tensor identity_pad;
			dense_tensor_pad_zeros(&identity, pad_before, pad_after, &identity_pad);
			delete_dense_tensor(&identity);
			reshape_dense_tensor(d->ndim, d->dim, &identity_pad);
			delete_dense_tensor(d);
			move_dense_tensor_data(&identity_pad, u_split.degensors[c]);
		}

		su2_to_dense_tensor(&u_split, &u_split_dns);

		delete_su2_tensor(&u_split);
	}

	// contract with dense splitting tensor to obtain reference tensor
	struct dense_tensor r_dns_ref;
	dense_tensor_multiply_axis(&t_dns, i_ax_split, &u_split_dns, TENSOR_AXIS_RANGE_LEADING, &r_dns_ref);

	// compare
	if (!dense_tensor_allclose(&r_dns, &r_dns_ref, 1e-13)) {
		return "tensor resulting from splitting an axis of an SU(2) tensor does not match reference";
	}

	// fuse axes again
	struct su2_tensor t2;
	su2_tensor_fuse_axes(&r, i_ax_split, i_ax_add, &t2);

	if (!su2_tensor_is_consistent(&t2)) {
		return "internal consistency check for SU(2) tensor failed";
	}

	// convert to a full dense tensor
	struct dense_tensor t2_dns;
	su2_to_dense_tensor(&t2, &t2_dns);

	// compare
	if (!dense_tensor_allclose(&t2_dns, &t_dns, 1e-13)) {
		return "tensor after axis splitting and fusion should agree with original tensor";
	}

	delete_dense_tensor(&t2_dns);
	delete_dense_tensor(&u_split_dns);
	delete_dense_tensor(&r_dns_ref);
	delete_dense_tensor(&r_dns);
	delete_dense_tensor(&t_dns);
	delete_su2_tensor(&t2);
	delete_su2_tensor(&r);
	delete_su2_tensor(&t);

	return 0;
}


char* test_su2_tensor_contract_simple()
{
	struct rng_state rng_state;
	seed_rng_state(42, &rng_state);

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
		//                       j:  0  1  2  3
		const long dim_degen0[4] = { 0, 4, 0, 2 };
		const long dim_degen1[4] = { 0, 0, 0, 3 };
		const long dim_degen2[3] = { 5, 0, 2    };
		const long dim_degen3[3] = { 0, 0, 7    };
		const long* dim_degen[4] = {
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
		for (long c = 0; c < s.charge_sectors.nsec; c++)
		{
			// corresponding "degeneracy" tensor
			struct dense_tensor* d = s.degensors[c];
			assert(d != NULL);
			assert(d->dtype == s.dtype);
			assert(d->ndim  == s.ndim_logical);

			dense_tensor_fill_random_normal(numeric_one(d->dtype), numeric_zero(d->dtype), &rng_state, d);
		}
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
		//                       j:  0  1  2  3  4  5
		const long dim_degen0[3] = { 5, 0, 2          };
		const long dim_degen1[6] = { 0, 0, 0, 2, 0, 4 };
		const long dim_degen2[4] = { 0, 0, 0, 3       };
		const long dim_degen3[3] = { 0, 0, 7          };
		const long dim_degen4[5] = { 0, 0, 0, 0, 3    };
		const long* dim_degen[5] = {
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
		for (long c = 0; c < t.charge_sectors.nsec; c++)
		{
			// corresponding "degeneracy" tensor
			struct dense_tensor* d = t.degensors[c];
			assert(d != NULL);
			assert(d->dtype == t.dtype);
			assert(d->ndim  == t.ndim_logical);

			dense_tensor_fill_random_normal(numeric_one(d->dtype), numeric_zero(d->dtype), &rng_state, d);
		}
	}

	for (int flip = 0; flip < 2; flip++)
	{
		if (flip == 1) {
			su2_tensor_flip_trees(&s);
			su2_tensor_flip_trees(&t);
		}

		// perform contraction
		const int i_ax_s[3] = { 3, 1, 4 };
		const int i_ax_t[3] = { 3, 2, 5 };
		struct su2_tensor r;
		if (flip == 0) {
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
		if (flip == 0)
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

			if (!su2_fuse_split_tree_equal(&r.tree, &tree_ref)) {
				return "fuse and split tree of contracted SU(2) tensor does not match reference";
			}
		}
		else
		{
			// construct the fuse and split tree
			//
			//      1    2
			//       ╲  ╱          fuse
			//        ╲╱
			//        │
			//        │6
			//        │
			//        ╱╲
			//      7╱  ╲
			//      ╱╲   ╲
			//     ╱  ╲8  ╲        split
			//    ╱   ╱╲   ╲
			//   ╱   ╱  ╲   ╲
			//  5   3    4   0
			//
			struct su2_tree_node j0  = { .i_ax = 0, .c = { NULL, NULL } };
			struct su2_tree_node j1  = { .i_ax = 1, .c = { NULL, NULL } };
			struct su2_tree_node j2  = { .i_ax = 2, .c = { NULL, NULL } };
			struct su2_tree_node j3  = { .i_ax = 3, .c = { NULL, NULL } };
			struct su2_tree_node j4  = { .i_ax = 4, .c = { NULL, NULL } };
			struct su2_tree_node j5  = { .i_ax = 5, .c = { NULL, NULL } };
			struct su2_tree_node j8  = { .i_ax = 8, .c = { &j3,  &j4  } };
			struct su2_tree_node j7  = { .i_ax = 7, .c = { &j5,  &j8  } };
			struct su2_tree_node j6f = { .i_ax = 6, .c = { &j1,  &j2  } };
			struct su2_tree_node j6s = { .i_ax = 6, .c = { &j7,  &j0  } };

			struct su2_fuse_split_tree tree_ref = { .tree_fuse = &j6f, .tree_split = &j6s, .ndim = 9 };
			if (!su2_fuse_split_tree_is_consistent(&tree_ref)) {
				return "internal consistency check for the fuse and split tree failed";
			}

			if (!su2_fuse_split_tree_equal(&r.tree, &tree_ref)) {
				return "fuse and split tree of contracted SU(2) tensor does not match reference";
			}
		}

		// convert to full dense tensors
		struct dense_tensor s_dns;
		struct dense_tensor t_dns;
		struct dense_tensor r_dns;
		su2_to_dense_tensor(&s, &s_dns);
		su2_to_dense_tensor(&t, &t_dns);
		su2_to_dense_tensor(&r, &r_dns);

		if (dense_tensor_is_zero(&s_dns, 0.) || dense_tensor_is_zero(&t_dns, 0.)) {
			return "to-be contracted SU(2) tensors should not be zero";
		}

		struct dense_tensor s_dns_perm;
		struct dense_tensor t_dns_perm;
		const int perm_s[4] = { 0, 2, 3, 1 };
		const int perm_t[5] = { 3, 2, 0, 1, 4 };
		transpose_dense_tensor(perm_s, &s_dns, &s_dns_perm);
		transpose_dense_tensor(perm_t, &t_dns, &t_dns_perm);

		// contract dense tensors, as reference
		struct dense_tensor r_dns_ref;
		if (flip == 0) {
			dense_tensor_dot(&s_dns_perm, TENSOR_AXIS_RANGE_TRAILING, &t_dns_perm, TENSOR_AXIS_RANGE_LEADING, 2, &r_dns_ref);
		}
		else {
			dense_tensor_dot(&t_dns_perm, TENSOR_AXIS_RANGE_LEADING, &s_dns_perm, TENSOR_AXIS_RANGE_TRAILING, 2, &r_dns_ref);
		}

		// compare
		if (!dense_tensor_allclose(&r_dns, &r_dns_ref, 1e-13)) {
			return "tensor resulting from contraction of two SU(2) tensors does not match reference";
		}

		delete_dense_tensor(&r_dns_ref);
		delete_dense_tensor(&t_dns_perm);
		delete_dense_tensor(&s_dns_perm);
		delete_dense_tensor(&r_dns);
		delete_dense_tensor(&t_dns);
		delete_dense_tensor(&s_dns);
		delete_su2_tensor(&r);
	}

	delete_su2_tensor(&t);
	delete_su2_tensor(&s);

	return 0;
}


char* test_su2_tensor_contract_yoga()
{
	struct rng_state rng_state;
	seed_rng_state(43, &rng_state);

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
			//                       j:  0  1  2  3  4  5
			const long dim_degen0[6] = { 0, 4, 0, 0, 0, 2 };
			const long dim_degen1[4] = { 0, 5, 0, 2       };
			const long dim_degen2[4] = { 0, 3, 0, 3       };
			const long dim_degen3[2] = { 0, 7             };
			const long* dim_degen[4] = {
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
			for (long c = 0; c < s.charge_sectors.nsec; c++)
			{
				// corresponding "degeneracy" tensor
				struct dense_tensor* d = s.degensors[c];
				assert(d != NULL);
				assert(d->dtype == s.dtype);
				assert(d->ndim  == s.ndim_logical);

				dense_tensor_fill_random_normal(numeric_one(d->dtype), numeric_zero(d->dtype), &rng_state, d);
			}
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
			//                       j:  0  1  2  3
			const long dim_degen0[4] = { 0, 0, 0, 3 };
			const long dim_degen1[3] = { 3, 0, 4    };
			const long dim_degen2[4] = { 0, 5, 0, 2 };
			const long dim_degen3[3] = { 0, 0, 7    };
			const long* dim_degen[4] = {
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
			for (long c = 0; c < t.charge_sectors.nsec; c++)
			{
				// corresponding "degeneracy" tensor
				struct dense_tensor* d = t.degensors[c];
				assert(d != NULL);
				assert(d->dtype == t.dtype);
				assert(d->ndim  == t.ndim_logical);

				dense_tensor_fill_random_normal(numeric_one(d->dtype), numeric_zero(d->dtype), &rng_state, d);
			}
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
		struct dense_tensor r_dns;
		su2_to_dense_tensor(&s, &s_dns);
		su2_to_dense_tensor(&t, &t_dns);
		su2_to_dense_tensor(&r, &r_dns);

		if (dense_tensor_is_zero(&s_dns, 0.) || dense_tensor_is_zero(&t_dns, 0.)) {
			return "to-be contracted SU(2) tensors should not be zero";
		}

		struct dense_tensor s_dns_perm;
		struct dense_tensor t_dns_perm;
		const int perm_s[4] = { 0, 2, 3, 1 };
		const int perm_t[4] = { 2, 0, 1, 3 };
		transpose_dense_tensor(perm_s, &s_dns, &s_dns_perm);
		transpose_dense_tensor(perm_t, &t_dns, &t_dns_perm);

		// contract dense tensors, as reference
		struct dense_tensor r_dns_ref;
		dense_tensor_dot(&s_dns_perm, TENSOR_AXIS_RANGE_TRAILING, &t_dns_perm, TENSOR_AXIS_RANGE_LEADING, 1, &r_dns_ref);

		// compare
		if (!dense_tensor_allclose(&r_dns, &r_dns_ref, 1e-5)) {
			return "tensor resulting from contraction of two SU(2) tensors does not match reference";
		}

		delete_dense_tensor(&r_dns_ref);
		delete_dense_tensor(&t_dns_perm);
		delete_dense_tensor(&s_dns_perm);
		delete_dense_tensor(&r_dns);
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
	//                       j:  0  1  2  3  4  5
	const long dim_degen0[5] = { 6, 0, 3, 0, 5    };
	const long dim_degen1[6] = { 0, 0, 0, 7, 0, 4 };
	const long dim_degen2[6] = { 0, 8, 0, 0, 0, 1 };
	const long dim_degen3[5] = { 0, 0, 9, 0, 2    };
	const long* dim_degen[4] = {
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
	seed_rng_state(44, &rng_state);
	for (long c = 0; c < t.charge_sectors.nsec; c++)
	{
		// corresponding "degeneracy" tensor
		struct dense_tensor* d = t.degensors[c];
		assert(d != NULL);
		assert(d->dtype == t.dtype);
		assert(d->ndim  == t.ndim_logical);

		dense_tensor_fill_random_normal(numeric_one(d->dtype), numeric_zero(d->dtype), &rng_state, d);
	}

	// convert to full dense tensor
	struct dense_tensor t_dns;
	su2_to_dense_tensor(&t, &t_dns);

	if (t_dns.ndim != ndim_logical) {
		return "degree of logical dense tensor does not match expected value";
	}

	// apply simultaneous rotations
	struct dense_tensor s, r;
	for (int i = 0; i < ndim_logical; i++)
	{
		const long d = su2_tensor_dim_logical_axis(&t, i);
		const long dim[2] = { d, d };
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
			move_dense_tensor_data(&r, &s);
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
