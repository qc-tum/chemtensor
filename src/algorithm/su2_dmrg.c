/// \file su2_dmrg.c
/// \brief DMRG algorithm for SU(2) symmetric tensors.

#include <stdio.h>
#include "su2_dmrg.h"
#include "su2_chain_ops.h"
#include "krylov.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Container of SU(2) symmetric Hamiltonian data for applying a site-local Hamiltonian operator.
///
struct su2_local_hamiltonian_data
{
	const struct su2_tensor* w;  //!< local Hamiltonian operator
	const struct su2_tensor* l;  //!< left tensor network block
	const struct su2_tensor* r;  //!< right tensor network block
	struct su2_tensor* a;        //!< local input MPS tensor (entries will be filled dynamically)
};


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper function for applying a site-local Hamiltonian operator, required for Lanczos iteration.
///
static void su2_apply_local_hamiltonian_wrapper_d(const ct_long n, const void* restrict data, const double* restrict v, double* restrict ret)
{
	// suppress unused parameter warning
	#ifdef NDEBUG
	(void)n;
	#endif

	struct su2_local_hamiltonian_data* hdata = (struct su2_local_hamiltonian_data*)data;

	// interpret input vector as MPS tensor entries
	assert(hdata->a->dtype == CT_DOUBLE_REAL);
	assert(n == su2_tensor_num_elements_degensors(hdata->a));
	su2_tensor_deserialize_renormalized_entries(hdata->a, v);

	struct su2_tensor ha;
	su2_apply_local_hamiltonian(hdata->a, hdata->w, hdata->l, hdata->r, &ha);

	assert(ha.dtype == CT_DOUBLE_REAL);
	assert(n == su2_tensor_num_elements_degensors(&ha));
	assert(charge_sectors_equal(&ha.charge_sectors, &hdata->a->charge_sectors));
	su2_tensor_serialize_renormalized_entries(&ha, ret);

	delete_su2_tensor(&ha);
}


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper function for applying a site-local Hamiltonian operator, required for Lanczos iteration.
///
static void su2_apply_local_hamiltonian_wrapper_z(const ct_long n, const void* restrict data, const dcomplex* restrict v, dcomplex* restrict ret)
{
	// suppress unused parameter warning
	#ifdef NDEBUG
	(void)n;
	#endif

	struct su2_local_hamiltonian_data* hdata = (struct su2_local_hamiltonian_data*)data;

	// interpret input vector as MPS tensor entries
	assert(hdata->a->dtype == CT_DOUBLE_COMPLEX);
	assert(n == su2_tensor_num_elements_degensors(hdata->a));
	su2_tensor_deserialize_renormalized_entries(hdata->a, v);

	struct su2_tensor ha;
	su2_apply_local_hamiltonian(hdata->a, hdata->w, hdata->l, hdata->r, &ha);

	assert(ha.dtype == CT_DOUBLE_COMPLEX);
	assert(n == su2_tensor_num_elements_degensors(&ha));
	assert(charge_sectors_equal(&ha.charge_sectors, &hdata->a->charge_sectors));
	su2_tensor_serialize_renormalized_entries(&ha, ret);

	delete_su2_tensor(&ha);
}


//________________________________________________________________________________________________________________________
///
/// \brief Minimize site-local energy by a Lanczos iteration; memory will be allocated for result 'a_opt'.
///
static int su2_minimize_local_energy(const struct su2_tensor* restrict w, const struct su2_tensor* restrict l, const struct su2_tensor* restrict r,
	const struct su2_tensor* restrict a_start, const int maxiter, double* restrict en_min, struct su2_tensor* restrict a_opt)
{
	assert(w->dtype == l->dtype);
	assert(w->dtype == r->dtype);
	assert(w->dtype == a_start->dtype);

	const ct_long n = su2_tensor_num_elements_degensors(a_start);
	void* vstart = ct_malloc(n * sizeof_numeric_type(a_start->dtype));
	su2_tensor_serialize_renormalized_entries(a_start, vstart);

	// using 'a_opt' as temporary tensor for iterations
	allocate_su2_tensor_like(a_start, a_opt);

	struct su2_local_hamiltonian_data hdata = { .w = w, .l = l, .r = r, .a = a_opt };

	void* u_opt = ct_malloc(n * sizeof_numeric_type(a_start->dtype));

	switch (a_start->dtype)
	{
		case CT_SINGLE_REAL:
		{
			// not implemented yet
			assert(false);
			break;
		}
		case CT_DOUBLE_REAL:
		{
			int ret = eigensystem_krylov_symmetric(n, su2_apply_local_hamiltonian_wrapper_d, &hdata, vstart, maxiter, 1, en_min, u_opt);
			if (ret < 0) {
				return ret;
			}
			break;
		}
		case CT_SINGLE_COMPLEX:
		{
			// not implemented yet
			assert(false);
			break;
		}
		case CT_DOUBLE_COMPLEX:
		{
			int ret = eigensystem_krylov_hermitian(n, su2_apply_local_hamiltonian_wrapper_z, &hdata, vstart, maxiter, 1, en_min, u_opt);
			if (ret < 0) {
				return ret;
			}
			break;
		}
		default:
		{
			// unknown data type
			assert(false);
		}
	}

	su2_tensor_deserialize_renormalized_entries(a_opt, u_opt);

	ct_free(u_opt);
	ct_free(vstart);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Run the single-site DMRG algorithm for SU(2) symmetric tensors:
/// Approximate the ground state as MPS via left and right sweeps and local single-site optimizations.
/// The input 'psi' is used as the starting state and is updated in-place during the optimization. Its virtual bond dimensions cannot increase.
///
int su2_dmrg_singlesite(const struct su2_mpo* hamiltonian, const int num_sweeps, const int maxiter_lanczos, struct su2_mps* psi, double* en_sweeps)
{
	// number of lattice sites
	const int nsites = hamiltonian->nsites;
	assert(nsites == psi->nsites);
	assert(nsites >= 1);

	// currently only double precision supported
	assert(numeric_real_type(hamiltonian->a[0].dtype) == CT_DOUBLE_REAL);

	// right-normalize input matrix product state
	double nrm = su2_mps_orthonormalize_qr(psi, SU2_MPS_ORTHONORMAL_RIGHT);
	if (nrm == 0) {
		printf("Warning: in 'su2_dmrg_singlesite': initial MPS has norm zero (possibly due to mismatching quantum numbers)\n");
	}

	// left and right operator blocks
	struct su2_tensor* lblocks = ct_malloc(nsites * sizeof(struct su2_tensor));
	struct su2_tensor* rblocks = ct_malloc(nsites * sizeof(struct su2_tensor));
	su2_compute_right_operator_blocks(psi, psi, hamiltonian, rblocks);
	su2_create_dummy_operator_block_left(hamiltonian->a[0].dtype, &lblocks[0]);
	for (int i = 1; i < nsites; i++) {
		copy_su2_tensor(&lblocks[0], &lblocks[i]);
	}

	// TODO: number of sweeps should be determined by tolerance and some convergence measure
	for (int n = 0; n < num_sweeps; n++)
	{
		double en;

		// sweep from left to right
		for (int i = 0; i < nsites - 1; i++)
		{
			struct su2_tensor a_opt;
			int ret = su2_minimize_local_energy(&hamiltonian->a[i], &lblocks[i], &rblocks[i], &psi->a[i], maxiter_lanczos, &en, &a_opt);
			if (ret < 0) {
				return ret;
			}
			assert(a_opt.ndim_logical == 3 && a_opt.ndim_auxiliary == 0);
			delete_su2_tensor(&psi->a[i]);
			psi->a[i] = a_opt;  // copy internal data pointers

			// left-orthonormalize current psi->a[i]
			su2_mps_local_orthonormalize_qr(&psi->a[i], &psi->a[i + 1]);

			// update the left blocks
			delete_su2_tensor(&lblocks[i + 1]);
			su2_contraction_operator_step_left(&psi->a[i], &psi->a[i], &hamiltonian->a[i], &lblocks[i], &lblocks[i + 1]);
		}

		// sweep from right to left
		for (int i = nsites - 1; i > 0; i--)
		{
			struct su2_tensor a_opt;
			int ret = su2_minimize_local_energy(&hamiltonian->a[i], &lblocks[i], &rblocks[i], &psi->a[i], maxiter_lanczos, &en, &a_opt);
			if (ret < 0) {
				return ret;
			}
			assert(a_opt.ndim_logical == 3 && a_opt.ndim_auxiliary == 0);
			delete_su2_tensor(&psi->a[i]);
			psi->a[i] = a_opt;  // copy internal data pointers

			// right-orthonormalize current psi->a[i]
			su2_mps_local_orthonormalize_rq(&psi->a[i], &psi->a[i - 1]);

			// update the right blocks
			delete_su2_tensor(&rblocks[i - 1]);
			su2_contraction_operator_step_right(&psi->a[i], &psi->a[i], &hamiltonian->a[i], &rblocks[i], &rblocks[i - 1]);
		}

		// right-normalize leftmost tensor to ensure that 'psi' is normalized
		{
			// dummy tensor at site "-1"
			struct su2_tensor a_head;
			su2_mps_create_dummy_head_tensor(&psi->a[0], &a_head);

			su2_mps_local_orthonormalize_rq(&psi->a[0], &a_head);

			delete_su2_tensor(&a_head);
		}

		// record energy after each sweep
		en_sweeps[n] = en;
	}

	// clean up
	for (int i = 0; i < nsites; i++)
	{
		delete_su2_tensor(&rblocks[i]);
		delete_su2_tensor(&lblocks[i]);
	}
	ct_free(rblocks);
	ct_free(lblocks);

	return 0;
}
