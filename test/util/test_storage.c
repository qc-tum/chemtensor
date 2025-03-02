#include "mps.h"
#include "storage.h"

char* test_save_mps_hdf5() {
	const long nsites = 7; // number of lattice sites
	const long d = 4;      // local physical dimension

	const qnumber qsite[4] = {0, 2, 0, -1};

	struct rng_state rng_state;
	seed_rng_state(41, &rng_state);

	struct mps src;
	const long max_vdim = 16;
	const qnumber qnum_sector = 1;
	construct_random_mps(CT_SINGLE_COMPLEX, nsites, d, qsite, qnum_sector, max_vdim, &rng_state, &src);

	const char* filename = "test_save_mps_hdf5.hdf5";

	int status;

	status = save_mps_hdf5(&src, filename);
	if (status < 0) {
		return "storing mps as hdf5 file failed.";
	}

	struct mps loaded;
	status = load_mps_hdf5(filename, &loaded);
	if (status < 0) {
		return "loading mps from hdf5 file failed.";
	}

	if (!mps_equals(&src, &loaded)) {
		return "source and loaded MPSs don't match.";
	}

	delete_mps(&src);
	delete_mps(&loaded);

	return 0;
}