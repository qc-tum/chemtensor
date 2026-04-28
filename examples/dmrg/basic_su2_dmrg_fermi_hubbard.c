// SU(2) symmetric Fermi-Hubbard model Hamiltonian and DMRG
//
// This demonstration uses the C interface
// to construct the Fermi-Hubbard Hamiltonian as an SU(2) symmetric matrix product operator
// and to run DMRG for SU(2) symmetric tensors.

#include <stdio.h>
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

	// overall spin quantum number sector of quantum state
	const qnumber irrep_sector = 1;
	printf("psi restricted to overall spin quantum number sector %i (integer convention)\n", irrep_sector);

	// maximum allowed logical MPS bond dimension
	const ct_long max_vdim = 64;

	// construct MPO representation of Hamiltonian
	struct su2_mpo hamiltonian;
	construct_fermi_hubbard_1d_su2_mpo(nsites, t, u, mu, &hamiltonian);
	if (!su2_mpo_is_consistent(&hamiltonian)) {
		fprintf(stderr, "internal consistency check for SU(2) symmetric Fermi-Hubbard Hamiltonian MPO failed");
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
			fprintf(stderr, "internal SU(2) symmetric MPS consistency check failed");
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
	const int maxiter_lanczos = 5;
	double tol_split          = 1e-8;
	double* en_sweeps = ct_malloc(num_sweeps * sizeof(double));
	double* entropy   = ct_malloc((nsites - 1) * sizeof(double));
	printf("Running two-site DMRG for SU(2) symmetric tensors (using num_sweeps = %i, maxiter_lanczos = %i, tol_split = %g)... ", num_sweeps, maxiter_lanczos, tol_split);
	if (su2_dmrg_twosite(&hamiltonian, num_sweeps, maxiter_lanczos, tol_split, max_vdim, &psi, en_sweeps, entropy) < 0) {
		fprintf(stderr, "'su2_dmrg_twosite' failed internally");
		return -2;
	}
	printf("Done.\n");

	printf("psi logical bond dimensions: ");
	for (int l = 0; l < nsites + 1; l++) {
		printf("%li ", su2_mps_bond_dim(&psi, l));
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
		fprintf(stderr, "computing reference eigenvalues failed");
		return -3;
	}
	const double en_ref = ((double*)w_ref.data)[0];
	printf("reference ground state energy: %.15g\n", en_ref);

	printf("energy after each DMRG sweep:\n");
	for (int i = 0; i < num_sweeps; i++) {
		printf("%i: %.15g, diff to reference: %g\n", i + 1, en_sweeps[i], en_sweeps[i] - en_ref);
	}

	delete_dense_tensor(&w_ref);
	delete_dense_tensor(&hmat);
	ct_free(entropy);
	ct_free(en_sweeps);
	delete_su2_mps(&psi);
	delete_su2_mpo(&hamiltonian);

	return 0;
}
