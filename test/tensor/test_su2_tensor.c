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
	//   \  /   /
	//    \/   /         fuse
	//    5\  /
	//      \/
	//      |
	//      |6
	//      |
	//      /\
	//     /  \          split
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
	const struct su2_irreducible_list outer_jlists[5] = {
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
	allocate_su2_tensor(DOUBLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_jlists, dim_degen, &t);

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


char* test_su2_to_dense_tensor()
{
	hid_t file = H5Fopen("../test/tensor/data/test_su2_to_dense_tensor.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_su2_to_dense_tensor failed";
	}

	const int ndim = 7;

	// construct the fuse and split tree
	//
	//  2    4   0
	//   \  /   /
	//    \/   /         fuse
	//    5\  /
	//      \/
	//      |
	//      |6
	//      |
	//      /\
	//     /  \          split
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
	const struct su2_irreducible_list outer_jlists[5] = {
		{ .jlist = j0list, .num = sizeof(j0list) / sizeof(qnumber) },
		{ .jlist = j1list, .num = sizeof(j1list) / sizeof(qnumber) },
		{ .jlist = j2list, .num = sizeof(j2list) / sizeof(qnumber) },
		{ .jlist = j3list, .num = sizeof(j3list) / sizeof(qnumber) },
		{ .jlist = j4list, .num = sizeof(j4list) / sizeof(qnumber) },
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
	allocate_su2_tensor(DOUBLE_COMPLEX, ndim_logical, ndim_auxiliary, &tree, outer_jlists, dim_degen, &t);

	if (!su2_tensor_is_consistent(&t)) {
		return "internal consistency check for SU(2) tensor failed";
	}
	if (t.charge_sectors.nsec == 0) {
		return "expecting at least one charge sector in SU(2) tensor";
	}

	// fill degeneracy tensors with random entries
	struct rng_state rng_state;
	seed_rng_state(42, &rng_state);
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
		allocate_dense_tensor(DOUBLE_COMPLEX, 2, dim, &w);

		char varname[1024];
		sprintf(varname, "w%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, w.data) < 0) {
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

	H5Fclose(file);

	return 0;
}
