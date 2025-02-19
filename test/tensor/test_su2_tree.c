#include "su2_tree.h"
#include "aligned_memory.h"
#include "util.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


char* test_su2_tree_enumerate_charge_sectors()
{
	hid_t file = H5Fopen("../test/tensor/data/test_su2_tree_enumerate_charge_sectors.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_su2_tree_enumerate_charge_sectors failed";
	}

	const int ndim = 9;

	// construct the tree
	//
	//        │
	//        │7
	//        ╱╲
	//       ╱  ╲5
	//      ╱   ╱╲
	//     ╱  8╱  ╲
	//    ╱   ╱╲   ╲
	//   ╱   ╱  ╲   ╲
	//  2   0    4   1
	//
	struct su2_tree_node j2 = { .i_ax = 2, .c = { NULL, NULL } };
	struct su2_tree_node j0 = { .i_ax = 0, .c = { NULL, NULL } };
	struct su2_tree_node j4 = { .i_ax = 4, .c = { NULL, NULL } };
	struct su2_tree_node j1 = { .i_ax = 1, .c = { NULL, NULL } };
	struct su2_tree_node j8 = { .i_ax = 8, .c = { &j0,  &j4  } };
	struct su2_tree_node j5 = { .i_ax = 5, .c = { &j8,  &j1  } };
	struct su2_tree_node j7 = { .i_ax = 7, .c = { &j2,  &j5  } };

	if (su2_tree_num_nodes(&j7) != 7) {
		return "number of nodes in SU(2) symmetry tree does not match expected value";
	}

	qnumber j0list[] = { 1, 3 };
	qnumber j1list[] = { 0, 2, 6 };
	qnumber j2list[] = { 3, 7 };
	qnumber j4list[] = { 0, 4, 10 };
	const struct su2_irreducible_list leaf_ranges[5] = {
		{ .jlist = j0list, .num = ARRLEN(j0list) },
		{ .jlist = j1list, .num = ARRLEN(j1list) },
		{ .jlist = j2list, .num = ARRLEN(j2list) },
		{ .jlist = NULL,   .num = 0              },  // not used
		{ .jlist = j4list, .num = ARRLEN(j4list) },
	};

	struct charge_sectors sectors;
	su2_tree_enumerate_charge_sectors(&j7, ndim, leaf_ranges, &sectors);

	// read reference tensor from disk
	hsize_t dims_ref[2];
	if (get_hdf5_dataset_dims(file, "charge_sectors", dims_ref) < 0) {
		return "obtaining charge sector dimensions failed";
	}
	qnumber* sectors_ref = ct_malloc(dims_ref[0] * dims_ref[1] * sizeof(qnumber));
	// read values from disk
	if (read_hdf5_dataset(file, "charge_sectors", H5T_NATIVE_INT, sectors_ref) < 0) {
		return "reading charge sectors from disk failed";
	}

	// compare
	if ((long)dims_ref[0] != sectors.nsec) {
		return "number of charge sectors does not agree with reference";
	}
	if ((int)dims_ref[1] != sectors.ndim) {
		return "number of quantum numbers does not agree with reference";
	}
	for (int i = 0; i < sectors.nsec; i++) {
		for (int j = 0; j < sectors.ndim; j++) {
			if (sectors.jlists[i * sectors.ndim + j] != sectors_ref[i * sectors.ndim + j]) {
				return "charge sector quantum number does not agree with reference";
			}
		}
	}

	ct_free(sectors_ref);
	delete_charge_sectors(&sectors);

	H5Fclose(file);

	return 0;
}


char* test_su2_fuse_split_tree_enumerate_charge_sectors()
{
	hid_t file = H5Fopen("../test/tensor/data/test_su2_fuse_split_tree_enumerate_charge_sectors.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_su2_fuse_split_tree_enumerate_charge_sectors failed";
	}

	const int ndim = 11;

	// construct the fuse and split tree
	//
	//    4    3   0
	//     ╲  ╱   ╱
	//      ╲╱   ╱         fuse
	//      7╲  ╱
	//        ╲╱
	//        │
	//        │9
	//        │
	//        ╱╲
	//       ╱  ╲
	//      ╱    ╲         split
	//    8╱      ╲10
	//    ╱╲      ╱╲
	//   ╱  ╲    ╱  ╲
	//  5    1  6    2
	//
	struct su2_tree_node j0  = { .i_ax =  0, .c = { NULL, NULL } };
	struct su2_tree_node j1  = { .i_ax =  1, .c = { NULL, NULL } };
	struct su2_tree_node j2  = { .i_ax =  2, .c = { NULL, NULL } };
	struct su2_tree_node j3  = { .i_ax =  3, .c = { NULL, NULL } };
	struct su2_tree_node j4  = { .i_ax =  4, .c = { NULL, NULL } };
	struct su2_tree_node j5  = { .i_ax =  5, .c = { NULL, NULL } };
	struct su2_tree_node j6  = { .i_ax =  6, .c = { NULL, NULL } };
	struct su2_tree_node j7  = { .i_ax =  7, .c = { &j4,  &j3  } };
	struct su2_tree_node j8  = { .i_ax =  8, .c = { &j5,  &j1  } };
	struct su2_tree_node j10 = { .i_ax = 10, .c = { &j6,  &j2  } };
	struct su2_tree_node j9f = { .i_ax =  9, .c = { &j7,  &j0  } };
	struct su2_tree_node j9s = { .i_ax =  9, .c = { &j8,  &j10 } };

	struct su2_fuse_split_tree tree = { .tree_fuse = &j9f, .tree_split = &j9s, .ndim = ndim };

	if (!su2_fuse_split_tree_is_consistent(&tree)) {
		return "internal consistency check for the fuse and split tree failed";
	}

	qnumber j0list[] = { 0, 4, 6 };
	qnumber j1list[] = { 3, 9 };
	qnumber j2list[] = { 5, 7 };
	qnumber j3list[] = { 3 };
	qnumber j4list[] = { 2, 4 };
	qnumber j5list[] = { 0, 2, 6 };
	qnumber j6list[] = { 1, 5 };
	const struct su2_irreducible_list leaf_ranges[7] = {
		{ .jlist = j0list, .num = ARRLEN(j0list) },
		{ .jlist = j1list, .num = ARRLEN(j1list) },
		{ .jlist = j2list, .num = ARRLEN(j2list) },
		{ .jlist = j3list, .num = ARRLEN(j3list) },
		{ .jlist = j4list, .num = ARRLEN(j4list) },
		{ .jlist = j5list, .num = ARRLEN(j5list) },
		{ .jlist = j6list, .num = ARRLEN(j6list) },
	};

	struct charge_sectors sectors;
	su2_fuse_split_tree_enumerate_charge_sectors(&tree, leaf_ranges, &sectors);

	// read reference tensor from disk
	hsize_t dims_ref[2];
	if (get_hdf5_dataset_dims(file, "charge_sectors", dims_ref) < 0) {
		return "obtaining charge sector dimensions failed";
	}
	qnumber* sectors_ref = ct_malloc(dims_ref[0] * dims_ref[1] * sizeof(qnumber));
	// read values from disk
	if (read_hdf5_dataset(file, "charge_sectors", H5T_NATIVE_INT, sectors_ref) < 0) {
		return "reading charge sectors from disk failed";
	}

	// compare
	if ((long)dims_ref[0] != sectors.nsec) {
		return "number of charge sectors does not agree with reference";
	}
	if ((int)dims_ref[1] != sectors.ndim) {
		return "number of quantum numbers does not agree with reference";
	}
	for (int i = 0; i < sectors.nsec; i++) {
		for (int j = 0; j < sectors.ndim; j++) {
			if (sectors.jlists[i * sectors.ndim + j] != sectors_ref[i * sectors.ndim + j]) {
				return "charge sector quantum number does not agree with reference";
			}
		}
	}

	ct_free(sectors_ref);
	delete_charge_sectors(&sectors);

	H5Fclose(file);

	return 0;
}
