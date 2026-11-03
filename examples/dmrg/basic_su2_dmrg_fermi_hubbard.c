// SU(2) symmetric Fermi-Hubbard model Hamiltonian and DMRG
//
// This demonstration uses the C interface
// to construct the Fermi-Hubbard Hamiltonian as an SU(2) symmetric matrix product operator
// and to run DMRG for SU(2) symmetric tensors.

#ifdef _OPENMP
#include <omp.h>
#endif
#include "lapack_ct.h"
#include "su2_hamiltonian.h"
#include "su2_dmrg.h"
#include "hamiltonian.h"
#include "aligned_memory.h"
#include "rng.h"
#include "util.h"
#include "logging.h"
#include "logo.h"


int main()
{
	// log file
	fd_log = fopen("basic_su2_dmrg_fermi_hubbard.log", "a");
	if (fd_log == NULL) {
		logprint("Warning: cannot create or open log file\n");
	}
	// set verbosity level
	ct_verbosity = CT_VERBOSITY_BASIC;

	logprint(ct_logo);
	logprint("Git version: %s\n\n", CT_GIT_VERSION);

	// number of spin-endowed lattice sites (local dimension is 4)
	const int nsites = 6;
	logprint("number of spin-endowed lattice sites: %i -> Hilbert space dimension: %li\n", nsites, ipow(4, nsites));

	// Hamiltonian parameters
	const double t  = 1.0;
	const double u  = 4.0;
	const double mu = 1.5;
	logprint("Fermi-Hubbard Hamiltonian parameters: t = %g, u = %g, mu = %g\n", t, u, mu);

	// overall spin quantum number sector of quantum state
	const qnumber irrep_sector = 1;
	logprint("psi restricted to overall spin quantum number sector %i (integer convention)\n", irrep_sector);

	// maximum allowed logical MPS bond dimension
	const ct_long max_vdim = 64;

	// construct MPO representation of Hamiltonian
	struct su2_mpo hamiltonian;
	construct_fermi_hubbard_1d_su2_mpo(nsites, t, u, mu, &hamiltonian);
	if (!su2_mpo_is_consistent(&hamiltonian)) {
		logerror("internal consistency check for SU(2) symmetric Fermi-Hubbard Hamiltonian MPO failed");
		return -1;
	}

	// initial state vector as MPS
	struct su2_mps psi;
	{
		struct rng_state rng_state;
		seed_rng_state(42, &rng_state);

		const qnumber max_bond_irrep = 5;
		const ct_long max_bond_dim_degen = 12;

		construct_random_su2_mps(hamiltonian.a[0].dtype, hamiltonian.nsites,
			&hamiltonian.a[0].outer_irreps[1], hamiltonian.a[0].dim_degen[1],
			irrep_sector, max_bond_irrep, max_bond_dim_degen, &rng_state, &psi);
		if (!su2_mps_is_consistent(&psi)) {
			logerror("internal SU(2) symmetric MPS consistency check failed");
			return -1;
		}
	}

	#ifdef _OPENMP
	logprint("maximum number of OpenMP threads: %d\n", omp_get_max_threads());
	#else
	logprint("OpenMP not available\n");
	#endif

	// run two-site DMRG
	struct su2_dmrg_twosite_options options;
	su2_dmrg_twosite_default_options(&options);
	options.max_num_sweeps  = 4;
	options.maxiter_lanczos = 5;
	options.tol_split       = 1e-8;
	options.max_vdim        = max_vdim;
	struct su2_dmrg_twosite_info info;
	if (su2_dmrg_twosite(&hamiltonian, &options, &psi, &info) < 0) {
		logerror("'su2_dmrg_twosite' failed internally");
		return -2;
	}

	// matrix representation of Hamiltonian
	struct dense_tensor hmat;
	{
		// use a standard (instead of SU(2) symmetric) MPO since conversion to matrix form would take too long
		struct mpo_assembly assembly;
		construct_fermi_hubbard_1d_mpo_assembly(nsites, t, u, mu, &assembly);
		struct mpo hamiltonian_mpo;
		mpo_from_assembly(&assembly, &hamiltonian_mpo);
		delete_mpo_assembly(&assembly);

		struct block_sparse_tensor hsparse;
		mpo_to_matrix(&hamiltonian_mpo, &hsparse);
		delete_mpo(&hamiltonian_mpo);

		// convert to dense tensor
		block_sparse_to_dense_tensor(&hsparse, &hmat);
		delete_block_sparse_tensor(&hsparse);

		// dimension and data type consistency checks
		assert(hmat.ndim == 4);
		// dummy virtual bond dimensions are retained
		assert(hmat.dim[0] == 1 && hmat.dim[3] == 1);
		assert(hmat.dim[1] == ipow(4, nsites));
		assert(hmat.dim[1] == hmat.dim[2]);
		assert(hmat.dtype == CT_DOUBLE_REAL);

		const ct_long dim_mat[2] = { hmat.dim[1], hmat.dim[2] };
		reshape_dense_tensor(2, dim_mat, &hmat);
	}

	// reference eigenvalues (based on exact diagonalization of matrix representation)
	struct dense_tensor w_ref;
	if (dense_tensor_eigvalsh(&hmat, &w_ref) < 0) {
		logerror("computing reference eigenvalues failed");
		return -3;
	}
	const double en_ref = ((double*)w_ref.data)[0];
	logprint("reference ground state energy: %.15g\n", en_ref);

	logprint("energy after each DMRG sweep:\n");
	for (int i = 0; i < info.num_sweeps; i++) {
		logprint("%i: %.15g, diff to reference: %g\n", i + 1, info.en_sweeps[i], info.en_sweeps[i] - en_ref);
	}

	delete_dense_tensor(&w_ref);
	delete_dense_tensor(&hmat);
	delete_su2_dmrg_twosite_info(&info);
	delete_su2_mps(&psi);
	delete_su2_mpo(&hamiltonian);

	if (fd_log != NULL)
	{
		fclose(fd_log);
		fd_log = NULL;
	}

	return 0;
}
