#include "su2_tree.h"
#include "aligned_memory.h"
#include "util.h"


char* test_su2_tree_enumerate_charge_sectors()
{
	hid_t file = H5Fopen("../test/data/test_su2_tree_enumerate_charge_sectors.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_su2_tree_enumerate_charge_sectors failed";
	}

	const int ndim = 9;

	// construct the tree
	//
	//        |
	//        |7
	//        /\
	//       /  \5
	//      /   /\
	//     /  8/  \
	//    /   /\   \
	//   /   /  \   \
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
		{ .jlist = j0list, .num = sizeof(j0list) / sizeof(qnumber) },
		{ .jlist = j1list, .num = sizeof(j1list) / sizeof(qnumber) },
		{ .jlist = j2list, .num = sizeof(j2list) / sizeof(qnumber) },
		{ .jlist = NULL,   .num = 0                                },  // not used
		{ .jlist = j4list, .num = sizeof(j4list) / sizeof(qnumber) },
	};

	struct charge_sectors sectors;
	su2_tree_enumerate_charge_sectors(&j7, ndim, leaf_ranges, &sectors);

	// read reference tensor from disk
	hsize_t dims_ref[2];
	if (get_hdf5_dataset_dims(file, "charge_sectors", dims_ref) < 0) {
		return "obtaining charge sector dimensions failed";
	}
	qnumber* sectors_ref = aligned_alloc(MEM_DATA_ALIGN, dims_ref[0] * dims_ref[1] * sizeof(qnumber));
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

	aligned_free(sectors_ref);
	delete_charge_sectors(&sectors);

	H5Fclose(file);

	return 0;
}
