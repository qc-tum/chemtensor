#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "hamiltonian.h"
#include "dmrg.h"
#include "aligned_memory.h"
#include "rng.h"
#include "timing.h"


int main()
{
	// number of spin-endowed lattice sites (spatial orbitals)
	const int nsites = 9;
	printf("number of spin-endowed lattice sites (spatial orbitals): %i -> Hilbert space dimension: %li\n", nsites, ipow(4, nsites));

	// construct MPO representation of Hamiltonian
	struct mpo hamiltonian;
	{
		// Hamiltonian coefficients
		hid_t file = H5Fopen("../perf/perf_dmrg_coeffs.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
		if (file < 0) {
			fprintf(stderr, "'H5Fopen' failed");
			return -1;
		}
		struct dense_tensor tkin;
		const long dim_tkin[2] = { nsites, nsites };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim_tkin, &tkin);
		if (read_hdf5_dataset(file, "tkin", H5T_NATIVE_DOUBLE, tkin.data) < 0) {
			fprintf(stderr, "reading kinetic hopping coefficients from disk failed");
			return -1;
		}
		struct dense_tensor vint;
		const long dim_vint[4] = { nsites, nsites, nsites, nsites };
		allocate_dense_tensor(CT_DOUBLE_REAL, 4, dim_vint, &vint);
		if (read_hdf5_dataset(file, "vint", H5T_NATIVE_DOUBLE, vint.data) < 0) {
			fprintf(stderr, "reading interaction potential coefficients from disk failed");
			return -1;
		}
		H5Fclose(file);

		struct mpo_assembly assembly;
		construct_spin_molecular_hamiltonian_mpo_assembly(&tkin, &vint, false, &assembly);
		mpo_from_assembly(&assembly, &hamiltonian);
		delete_mpo_assembly(&assembly);

		delete_dense_tensor(&vint);
		delete_dense_tensor(&tkin);
	}
	if (!mpo_is_consistent(&hamiltonian)) {
		fprintf(stderr, "internal consistency check for molecular Hamiltonian MPO failed");
		return -1;
	}

	// overall quantum number sector of quantum state (particle number and spin)
	const qnumber q_pnum = nsites;
	const qnumber q_spin = 1;
	printf("psi restricted to overall quantum number sector with particle number %i and spin %i (integer convention)\n", q_pnum, q_spin);
	const qnumber qnum_sector = encode_quantum_number_pair(q_pnum, q_spin);

	// maximum allowed MPS bond dimension
	const long max_vdim = 512;

	// initial state vector as MPS
	struct mps psi;
	{
		struct rng_state rng_state;
		seed_rng_state(42, &rng_state);

		construct_random_mps(hamiltonian.a[0].dtype, hamiltonian.nsites, hamiltonian.d, hamiltonian.qsite, qnum_sector, max_vdim, &rng_state, &psi);
		if (!mps_is_consistent(&psi)) {
			fprintf(stderr, "internal MPS consistency check failed");
			return -1;
		}
	}

	#ifdef _OPENMP
	printf("maximum number of OpenMP threads: %d\n", omp_get_max_threads());
	#else
	printf("OpenMP not available\n");
	#endif

	// get the tick resolution
	const double ticks_per_sec = (double)get_tick_resolution();

	// run two-site DMRG
	const int num_sweeps      = 2;
	const int maxiter_lanczos = 25;
	double tol_split          = 1e-8;
	double* en_sweeps = ct_malloc(num_sweeps * sizeof(double));
	double* entropy   = ct_malloc((nsites - 1) * sizeof(double));
	printf("Running two-site DMRG (using num_sweeps = %i, maxiter_lanczos = %i, tol_split = %g)... ", num_sweeps, maxiter_lanczos, tol_split);
	const uint64_t tick_start = get_time_ticks();
	int ret = dmrg_twosite(&hamiltonian, num_sweeps, maxiter_lanczos, tol_split, max_vdim, &psi, en_sweeps, entropy);
	const uint64_t tick_end = get_time_ticks();
	if (ret < 0) {
		fprintf(stderr, "'dmrg_twosite' failed internally");
		return -2;
	}
	printf("Done.\n");

	printf("wall clock time for DMRG: %g seconds\n", (tick_end - tick_start) / ticks_per_sec);

	printf("||psi||: %g\n", mps_norm(&psi));

	printf("psi bond dimensions: ");
	for (int l = 0; l < nsites + 1; l++) {
		printf("%li ", mps_bond_dim(&psi, l));
	}
	printf("\n");
	printf("splitting entropies for each bond: ");
	for (int l = 0; l < nsites - 1; l++) {
		printf("%g ", entropy[l]);
	}
	printf("\n");

	printf("energy after each DMRG sweep:\n");
	for (int i = 0; i < num_sweeps; i++) {
		printf("%i: %.15g\n", i + 1, en_sweeps[i]);
	}

	ct_free(entropy);
	ct_free(en_sweeps);
	delete_mps(&psi);
	delete_mpo(&hamiltonian);

	return 0;
}
