#include <math.h>
#include <complex.h>
#include "su2_dmrg.h"
#include "su2_hamiltonian.h"
#include "su2_chain_ops.h"
#include "hamiltonian.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


char* test_su2_dmrg_singlesite()
{
	// number of lattice sites
	const int nsites = 7;

	// physical quantum numbers (multiplied by 2)
	qnumber site_jlist[1] = { 1 };
	const struct su2_irreducible_list site_irreps = { .jlist = site_jlist, .num = ARRLEN(site_jlist) };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                             j:  0  1
	const ct_long site_dim_degen[] = { 0, 1 };

	struct rng_state rng_state;
	seed_rng_state(81, &rng_state);

	// MPO representation of Hamiltonian
	struct su2_mpo hamiltonian;
	const double J = 1.3;
	construct_heisenberg_1d_su2_mpo(nsites, J, &hamiltonian);

	// initial state vector as an SU(2) symmetric MPS
	struct su2_mps psi;
	{
		const qnumber irrep_sector = 1;
		const qnumber max_bond_irrep = 5;
		const ct_long max_bond_dim_degen = 27;

		construct_random_su2_mps(CT_DOUBLE_REAL, nsites, &site_irreps, site_dim_degen, irrep_sector, max_bond_irrep, max_bond_dim_degen, &rng_state, &psi);
		// rescale tensors such that overall state norm is around 1
		for (int l = 0; l < nsites; l++)
		{
			const double alpha = 14;
			rscale_su2_tensor(&alpha, &psi.a[l]);
		}
		if (!su2_mps_is_consistent(&psi)) {
			return "internal SU(2) MPS consistency check failed";
		}
	}

	// run DMRG

	const int num_sweeps = 3;
	const int maxiter_lanczos = 5;

	double* en_sweeps = ct_malloc(num_sweeps * sizeof(double));

	if (su2_dmrg_singlesite(&hamiltonian, num_sweeps, maxiter_lanczos, &psi, en_sweeps) < 0) {
		return "'su2_dmrg_singlesite' failed internally";
	}

	if (!su2_mps_is_consistent(&psi)) {
		return "internal SU(2) MPS consistency check failed";
	}

	// expectation value <psi | H | psi>
	double avr_ref;
	su2_mpo_inner_product(&psi, &hamiltonian, &psi, &avr_ref);
	if (fabs(en_sweeps[num_sweeps - 1] - avr_ref) / fabs(avr_ref) > 1e-13) {
		return "final energy of single-site DMRG sweeps for SU(2) symmetric tensors does not match reference expectation value";
	}

	// reference calculation

	// reference matrix
	struct dense_tensor hamiltonian_mat;
	{
		// use a standard (instead of SU(2) symmetric) MPO since conversion to matrix form would take too long
		struct mpo_assembly assembly;
		construct_heisenberg_xxz_1d_mpo_assembly(nsites, J, 1.0, 0.0, &assembly);
		struct mpo hamiltonian_mpo;
		mpo_from_assembly(&assembly, &hamiltonian_mpo);
		delete_mpo_assembly(&assembly);

		struct block_sparse_tensor hamiltonian_tensor;
		mpo_to_matrix(&hamiltonian_mpo, &hamiltonian_tensor);
		delete_mpo(&hamiltonian_mpo);

		block_sparse_to_dense_tensor(&hamiltonian_tensor, &hamiltonian_mat);
		delete_block_sparse_tensor(&hamiltonian_tensor);

		assert(hamiltonian_mat.dtype == CT_DOUBLE_REAL);
		const ct_long dim_mat[2] = { hamiltonian_mat.dim[1], hamiltonian_mat.dim[2] };
		reshape_dense_tensor(2, dim_mat, &hamiltonian_mat);
		assert(hamiltonian_mat.ndim == 2);
	}

	struct dense_tensor lambda_ref;
	if (dense_tensor_eigvalsh(&hamiltonian_mat, &lambda_ref) < 0) {
		return "'dense_tensor_eigvalsh' failed internally";
	}
	const double en_ref = ((double*)lambda_ref.data)[0];
	delete_dense_tensor(&lambda_ref);

	delete_dense_tensor(&hamiltonian_mat);

	// compare
	if (fabs(en_sweeps[num_sweeps - 1] - en_ref) / fabs(en_ref) > 1e-13) {
		return "final energy of single-site DMRG sweeps for SU(2) symmetric tensors does not match reference ground state energy";
	}

	ct_free(en_sweeps);
	delete_su2_mps(&psi);
	delete_su2_mpo(&hamiltonian);

	return 0;
}
