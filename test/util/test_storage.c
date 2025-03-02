#include "mps.h"
#include "storage.h"

char* test_save_mps_hdf5() {
	const long nsites = 7; // number of lattice sites
	const long d = 4;      // local physical dimension

	const qnumber qsite[4] = {0, 2, 0, -1};

	struct rng_state rng_state;
	seed_rng_state(41, &rng_state);

	struct mps msource;
	const long max_vdim = 16;
	const qnumber qnum_sector = 1;
	construct_random_mps(CT_DOUBLE_REAL, nsites, d, qsite, qnum_sector, max_vdim, &rng_state, &msource);

	const char* filename = "test_save_mps_hdf5.hdf5";

	int status;

	status = save_mps_hdf5(&msource, filename);
	if (status < 0) {
		return "storing mps as hdf5 file failed.";
	}

	struct mps mloaded;
	status = load_mps_hdf5(filename, &mloaded);
	if (status < 0) {
		return "loading mps from hdf5 file failed.";
	}

	if (msource.nsites != mloaded.nsites) {
		return "'nsites' of loaded does not match source.";
	}

	if (msource.d != mloaded.d) {
		return "local physical dimension 'd' of loaded does not match source.";
	}

	if (!qnumber_all_equal(msource.d, mloaded.qsite, msource.qsite)) {
		return "local physical quantum numbers of loaded does not match source.";
	}

	for (int i = 0; i < nsites; i++) {
		if (!block_sparse_tensor_allclose(&mloaded.a[i], &msource.a[i], 0.)) {
			return "MPS tensor of the loaded one does not match original MPS tensor";
		}
	}

	delete_mps(&msource);
	delete_mps(&mloaded);

	return 0;
}