// Fermi-Hubbard model Hamiltonian and DMRG
//
// This demonstration uses the C interface
// to construct the Fermi-Hubbard Hamiltonian as a matrix product operator
// and to run DMRG.

#ifdef _OPENMP
#include <omp.h>
#endif
#include "lapack_ct.h"
#include "hamiltonian.h"
#include "dmrg.h"
#include "aligned_memory.h"
#include "rng.h"
#include "util.h"
#include "logging.h"
#include "logo.h"


int main()
{
	// log file
	fd_log = fopen("basic_dmrg_fermi_hubbard.log", "a");
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

	// overall quantum number sector of quantum state (particle number and spin)
	const qnumber q_pnum = 7;
	const qnumber q_spin = 1;
	logprint("psi restricted to overall quantum number sector with particle number %i and spin %i (integer convention)\n", q_pnum, q_spin);
	const qnumber qnum_sector = encode_quantum_number_pair(q_pnum, q_spin);

	// maximum allowed MPS bond dimension
	const ct_long max_vdim = 32;

	// construct MPO representation of Hamiltonian
	struct mpo hamiltonian;
	{
		struct mpo_assembly assembly;
		construct_fermi_hubbard_1d_mpo_assembly(nsites, t, u, mu, &assembly);
		mpo_from_assembly(&assembly, &hamiltonian);
		delete_mpo_assembly(&assembly);
	}
	if (!mpo_is_consistent(&hamiltonian)) {
		logerror("internal consistency check for Fermi-Hubbard Hamiltonian MPO failed");
		return -1;
	}

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
	options.max_num_sweeps  = 4;
	options.maxiter_lanczos = 5;
	options.tol_split       = 1e-8;
	options.max_vdim        = max_vdim;
	struct dmrg_twosite_info info;
	if (dmrg_twosite(&hamiltonian, &options, &psi, &info) < 0) {
		logerror("'dmrg_twosite' failed internally");
		return -2;
	}

	logprint("||psi||: %g\n", mps_norm(&psi));

	logprint("psi bond dimensions: ");
	for (int l = 0; l < nsites + 1; l++) {
		logprint("%li ", mps_bond_dim(&psi, l));
	}
	logprint("\n");
	logprint("splitting entropies for each bond: ");
	for (int l = 0; l < nsites - 1; l++) {
		logprint("%g ", info.entropy[l]);
	}
	logprint("\n");

	// matrix representation of Hamiltonian
	struct dense_tensor hmat;
	{
		struct block_sparse_tensor hsparse;
		mpo_to_matrix(&hamiltonian, &hsparse);

		// convert to dense tensor
		block_sparse_to_dense_tensor(&hsparse, &hmat);
		delete_block_sparse_tensor(&hsparse);

		// dimension and data type consistency checks
		assert(hmat.ndim == 4);
		// dummy virtual bond dimensions are retained
		assert(hmat.dim[0] == 1 && hmat.dim[3] == 1);
		assert(hmat.dim[1] == ipow(hamiltonian.d, hamiltonian.nsites));
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
