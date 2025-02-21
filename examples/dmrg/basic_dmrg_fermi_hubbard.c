#include <lapacke.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "hamiltonian.h"
#include "dmrg.h"
#include "aligned_memory.h"
#include "rng.h"
#include "util.h"


int main()
{
	// number of spin-endowed lattice sites (local dimension is 4)
	const int nsites = 6;
	printf("number of spin-endowed lattice sites: %i -> Hilbert space dimension: %li\n", nsites, ipow(4, nsites));

	// Hamiltonian parameters
	const double t  = 1.0;
	const double u  = 4.0;
	const double mu = 1.5;
	printf("Fermi-Hubbard Hamiltonian parameters: t = %g, u = %g, mu = %g\n", t, u, mu);

	// overall quantum number sector of quantum state (particle number and spin)
	const qnumber q_pnum = 7;
	const qnumber q_spin = 1;
	printf("psi restricted to overall quantum number sector with particle number %i and spin %i (integer convention)\n", q_pnum, q_spin);
	const qnumber qnum_sector = encode_quantum_number_pair(q_pnum, q_spin);

	// maximum allowed MPS bond dimension
	const long max_vdim = 32;

	// construct MPO representation of Hamiltonian
	struct mpo hamiltonian;
	{
		struct mpo_assembly assembly;
		construct_fermi_hubbard_1d_mpo_assembly(nsites, t, u, mu, &assembly);
		mpo_from_assembly(&assembly, &hamiltonian);
		delete_mpo_assembly(&assembly);
	}
	if (!mpo_is_consistent(&hamiltonian)) {
		fprintf(stderr, "internal consistency check for Fermi-Hubbard Hamiltonian MPO failed");
		return -1;
	}

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

	// run two-site DMRG
	const int num_sweeps      = 4;
	const int maxiter_lanczos = 25;
	double tol_split          = 1e-8;
	double* en_sweeps = ct_malloc(num_sweeps * sizeof(double));
	double* entropy   = ct_malloc((nsites - 1) * sizeof(double));
	printf("Running two-site DMRG (using num_sweeps = %i, maxiter_lanczos = %i, tol_split = %g)... ", num_sweeps, maxiter_lanczos, tol_split);
	if (dmrg_twosite(&hamiltonian, num_sweeps, maxiter_lanczos, tol_split, max_vdim, &psi, en_sweeps, entropy) < 0) {
		fprintf(stderr, "'dmrg_twosite' failed internally");
		return -2;
	}
	printf("Done.\n");

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
	}

	// reference eigenvalues (based on exact diagonalization of matrix representation)
	double* w_ref = ct_malloc(hmat.dim[1] * sizeof(double));
	int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', hmat.dim[1], hmat.data, hmat.dim[2], w_ref);
	if (info != 0) {
		fprintf(stderr, "LAPACK function 'dsyev' failed, return value: %i\n", info);
		return info;
	}
	printf("reference ground state energy: %g\n", w_ref[0]);

	printf("energy after each DMRG sweep:\n");
	for (int i = 0; i < num_sweeps; i++) {
		printf("%i: %g, diff to reference: %g\n", i + 1, en_sweeps[i], en_sweeps[i] - w_ref[0]);
	}

	ct_free(w_ref);
	delete_dense_tensor(&hmat);
	ct_free(entropy);
	ct_free(en_sweeps);
	delete_mps(&psi);
	delete_mpo(&hamiltonian);

	return 0;
}
