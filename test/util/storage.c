#include "mps.h"
#include "storage.h"

char* test_save_mps_hdf5()
{
	const long nsites = 7; // number of lattice sites
	const long d = 4; // local physical dimension

	const qnumber qsite[4] = { 0, 2, 0, -1 };

	struct rng_state rng_state;
	seed_rng_state(41, &rng_state);

	struct mps m;
	const long max_vdim = 16;
	const qnumber qnum_sector = 1;
	construct_random_mps(CT_DOUBLE_COMPLEX, nsites, d, qsite, qnum_sector, max_vdim, &rng_state, &m);

    printf("\n");
    for(size_t i = 0; i < m.d; i++) {
        printf("%d\n", m.qsite[i]);
    }

    int status = save_mps_hdf5(&m, "test_save_mps_hdf5.hdf5");
    return 0;
}