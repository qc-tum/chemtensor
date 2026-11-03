#ifdef _OPENMP
#include <omp.h>
#endif
#include "hamiltonian.h"
#include "dmrg.h"
#include "aligned_memory.h"
#include "rng.h"
#include "hdf5_util.h"
#include "logging.h"
#include "logo.h"


int main()
{
	// log file
	fd_log = fopen("perf_dmrg.log", "a");
	if (fd_log == NULL) {
		logprint("Warning: cannot create or open log file\n");
	}
	// set verbosity level
	ct_verbosity = CT_VERBOSITY_TIMING;

	logprint(ct_logo);
	logprint("Git version: %s\n\n", CT_GIT_VERSION);

	// number of spin-endowed lattice sites (spatial orbitals)
	const int nsites = 9;
	logprint("number of spin-endowed lattice sites (spatial orbitals): %i -> Hilbert space dimension: %li\n", nsites, ipow(4, nsites));

	// construct MPO representation of Hamiltonian
	struct mpo hamiltonian;
	{
		// Hamiltonian coefficients
		hid_t file = H5Fopen("../perf/perf_dmrg_coeffs.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
		if (file < 0) {
			logerror("'H5Fopen' failed");
			return -1;
		}
		struct dense_tensor tkin;
		const ct_long dim_tkin[2] = { nsites, nsites };
		allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim_tkin, &tkin);
		if (read_hdf5_dataset(file, "tkin", H5T_NATIVE_DOUBLE, tkin.data) < 0) {
			logerror("reading kinetic hopping coefficients from disk failed");
			return -1;
		}
		struct dense_tensor vint;
		const ct_long dim_vint[4] = { nsites, nsites, nsites, nsites };
		allocate_dense_tensor(CT_DOUBLE_REAL, 4, dim_vint, &vint);
		if (read_hdf5_dataset(file, "vint", H5T_NATIVE_DOUBLE, vint.data) < 0) {
			logerror("reading interaction potential coefficients from disk failed");
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
		logerror("internal consistency check for molecular Hamiltonian MPO failed");
		return -1;
	}

	// overall quantum number sector of quantum state (particle number and spin)
	const qnumber q_pnum = nsites;
	const qnumber q_spin = 1;
	logprint("psi restricted to overall quantum number sector with particle number %i and spin %i (integer convention)\n", q_pnum, q_spin);
	const qnumber qnum_sector = encode_quantum_number_pair(q_pnum, q_spin);

	// maximum allowed MPS bond dimension
	const ct_long max_vdim = 512;

	// initial state vector as MPS
	struct mps psi;
	{
		struct rng_state rng_state;
		seed_rng_state(42, &rng_state);

		construct_random_mps(hamiltonian.a[0].dtype, hamiltonian.nsites, hamiltonian.d, hamiltonian.qsite, qnum_sector, max_vdim, &rng_state, &psi);
		if (!mps_is_consistent(&psi)) {
			logerror("internal MPS consistency check failed");
			return -1;
		}
	}

	#ifdef _OPENMP
	logprint("maximum number of OpenMP threads: %d\n", omp_get_max_threads());
	#else
	logprint("OpenMP not available\n");
	#endif

	// run two-site DMRG
	struct dmrg_twosite_options options;
	dmrg_twosite_default_options(&options);
	options.max_num_sweeps = 2;
	options.max_vdim = max_vdim;
	struct dmrg_twosite_info info;
	int ret = dmrg_twosite(&hamiltonian, &options, &psi, &info);
	if (ret < 0) {
		logerror("'dmrg_twosite' failed internally");
		return -2;
	}

	logprint("||psi||: %g\n", mps_norm(&psi));

	delete_dmrg_twosite_info(&info);
	delete_mps(&psi);
	delete_mpo(&hamiltonian);

	if (fd_log != NULL)
	{
		fclose(fd_log);
		fd_log = NULL;
	}

	return 0;
}
