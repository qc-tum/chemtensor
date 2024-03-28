/// \file dmrg.c
/// \brief DMRG algorithm.

#include "dmrg.h"
#include "operation.h"
#include "krylov.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Container of Hamiltonian data for applying site-local Hamiltonian operator.
///
struct local_hamiltonian_data
{
	const struct block_sparse_tensor* w;  //!< local Hamiltonian operator
	const struct block_sparse_tensor* l;  //!< left tensor network block
	const struct block_sparse_tensor* r;  //!< right tensor network block
	struct block_sparse_tensor* a;        //!< local input MPS tensor (entries will be filled dynamically)
};


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper function for applying a site-local Hamiltonian operator, required for Lanczos iteration.
///
static void apply_local_hamiltonian_wrapper_d(const long n, const void* restrict data, const double* restrict v, double* restrict ret)
{
	struct local_hamiltonian_data* hdata = (struct local_hamiltonian_data*)data;

	// interpret input vector as MPS tensor entries
	assert(hdata->a->dtype == DOUBLE_REAL);
	assert(n == block_sparse_tensor_num_elements_blocks(hdata->a));
	block_sparse_tensor_deserialize_entries(hdata->a, v);

	struct block_sparse_tensor ha;
	apply_local_hamiltonian(hdata->a, hdata->w, hdata->l, hdata->r, &ha);

	assert(ha.dtype == DOUBLE_REAL);
	assert(n == block_sparse_tensor_num_elements_blocks(&ha));
	block_sparse_tensor_serialize_entries(&ha, ret);

	delete_block_sparse_tensor(&ha);
}


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper function for applying a site-local Hamiltonian operator, required for Lanczos iteration.
///
static void apply_local_hamiltonian_wrapper_z(const long n, const void* restrict data, const dcomplex* restrict v, dcomplex* restrict ret)
{
	struct local_hamiltonian_data* hdata = (struct local_hamiltonian_data*)data;

	// interpret input vector as MPS tensor entries
	assert(hdata->a->dtype == DOUBLE_COMPLEX);
	assert(n == block_sparse_tensor_num_elements_blocks(hdata->a));
	block_sparse_tensor_deserialize_entries(hdata->a, v);

	struct block_sparse_tensor ha;
	apply_local_hamiltonian(hdata->a, hdata->w, hdata->l, hdata->r, &ha);

	assert(ha.dtype == DOUBLE_COMPLEX);
	assert(n == block_sparse_tensor_num_elements_blocks(&ha));
	block_sparse_tensor_serialize_entries(&ha, ret);

	delete_block_sparse_tensor(&ha);
}


//________________________________________________________________________________________________________________________
///
/// \brief Minimize site-local energy by a Lanczos iteration; memory will be allocated for result 'a_opt'.
///
static int minimize_local_energy(const struct block_sparse_tensor* restrict w, const struct block_sparse_tensor* restrict l, const struct block_sparse_tensor* restrict r,
	const struct block_sparse_tensor* restrict a_start, const int maxiter, double* restrict en_min, struct block_sparse_tensor* restrict a_opt)
{
	assert(w->dtype == l->dtype);
	assert(w->dtype == r->dtype);
	assert(w->dtype == a_start->dtype);

	const int n = block_sparse_tensor_num_elements_blocks(a_start);
	void* vstart = aligned_alloc(MEM_DATA_ALIGN, n * sizeof_numeric_type(a_start->dtype));
	block_sparse_tensor_serialize_entries(a_start, vstart);

	// using 'a_opt' as temporary tensor for iterations
	allocate_block_sparse_tensor_like(a_start, a_opt);

	struct local_hamiltonian_data hdata = { .w = w, .l = l, .r = r, .a = a_opt };

	void* u_opt = aligned_alloc(MEM_DATA_ALIGN, n * sizeof_numeric_type(a_start->dtype));

	switch (a_start->dtype)
	{
		case SINGLE_REAL:
		{
			// not implemented yet
			assert(false);
			break;
		}
		case DOUBLE_REAL:
		{
			int ret = eigensystem_krylov_symmetric(n, apply_local_hamiltonian_wrapper_d, &hdata, vstart, maxiter, 1, en_min, u_opt);
			if (ret < 0) {
				return ret;
			}
			break;
		}
		case SINGLE_COMPLEX:
		{
			// not implemented yet
			assert(false);
			break;
		}
		case DOUBLE_COMPLEX:
		{
			int ret = eigensystem_krylov_hermitian(n, apply_local_hamiltonian_wrapper_z, &hdata, vstart, maxiter, 1, en_min, u_opt);
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

	block_sparse_tensor_deserialize_entries(a_opt, u_opt);

	aligned_free(u_opt);
	aligned_free(vstart);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Run the single-site DMRG algorithm: Approximate the ground state as MPS via left and right sweeps and local single-site optimizations.
/// The input 'psi' is used as starting state and is updated in-place during the optimization. Its virtual bond dimensions cannot increase.
///
int dmrg_singlesite(const struct mpo* hamiltonian, const int num_sweeps, const int maxiter_lanczos, struct mps* psi, double* en_sweeps)
{
	// number of lattice sites
	const int nsites = hamiltonian->nsites;
	assert(nsites == psi->nsites);
	assert(nsites >= 1);

	// currently only double precision supported
	assert(numeric_real_type(hamiltonian->a[0].dtype) == DOUBLE_REAL);

	// right-normalize input matrix product state
	mps_orthonormalize_qr(psi, MPS_ORTHONORMAL_RIGHT);

	// left and right operator blocks
	struct block_sparse_tensor* lblocks = aligned_alloc(MEM_DATA_ALIGN, nsites * sizeof(struct block_sparse_tensor));
	struct block_sparse_tensor* rblocks = aligned_alloc(MEM_DATA_ALIGN, nsites * sizeof(struct block_sparse_tensor));
	compute_right_operator_blocks(psi, psi, hamiltonian, rblocks);
	create_dummy_operator_block_left(&psi->a[0], &psi->a[0], &hamiltonian->a[0], &lblocks[0]);
	for (int i = 1; i < nsites; i++) {
		copy_block_sparse_tensor(&lblocks[0], &lblocks[i]);
	}

	// TODO: number of sweeps should be determined by tolerance and some convergence measure
	for (int n = 0; n < num_sweeps; n++)
	{
		double en;

		// sweep from left to right
		for (int i = 0; i < nsites - 1; i++)
		{
			struct block_sparse_tensor a_opt;
			int ret = minimize_local_energy(&hamiltonian->a[i], &lblocks[i], &rblocks[i], &psi->a[i], maxiter_lanczos, &en, &a_opt);
			if (ret < 0) {
				return ret;
			}
			assert(a_opt.ndim == 3);
			delete_block_sparse_tensor(&psi->a[i]);
			move_block_sparse_tensor_data(&a_opt, &psi->a[i]);

			// left-orthonormalize current psi->a[i]
			mps_local_orthonormalize_qr(&psi->a[i], &psi->a[i + 1]);

			// update the left blocks
			delete_block_sparse_tensor(&lblocks[i + 1]);
			contraction_operator_step_left(&psi->a[i], &psi->a[i], &hamiltonian->a[i], &lblocks[i], &lblocks[i + 1]);
		}

		// sweep from right to left
		for (int i = nsites - 1; i > 0; i--)
		{
			struct block_sparse_tensor a_opt;
			int ret = minimize_local_energy(&hamiltonian->a[i], &lblocks[i], &rblocks[i], &psi->a[i], maxiter_lanczos, &en, &a_opt);
			if (ret < 0) {
				return ret;
			}
			assert(a_opt.ndim == 3);
			delete_block_sparse_tensor(&psi->a[i]);
			move_block_sparse_tensor_data(&a_opt, &psi->a[i]);

			// right-orthonormalize current psi->a[i]
			mps_local_orthonormalize_rq(&psi->a[i], &psi->a[i - 1]);

			// update the right blocks
			delete_block_sparse_tensor(&rblocks[i - 1]);
			contraction_operator_step_right(&psi->a[i], &psi->a[i], &hamiltonian->a[i], &rblocks[i], &rblocks[i - 1]);
		}

		// right-normalize leftmost tensor to ensure that 'psi' is normalized
		{
			// dummy tensor at site "-1"
			struct block_sparse_tensor t;
			const long dim[3] = { psi->a[0].dim_logical[0], 1, psi->a[0].dim_logical[0] };
			const enum tensor_axis_direction axis_dir[3] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN };
			qnumber qzero[1] = { 0 };
			const qnumber* qnums[3] = { psi->a[0].qnums_logical[0], qzero, psi->a[0].qnums_logical[0] };
			allocate_block_sparse_tensor(psi->a[0].dtype, 3, dim, axis_dir, qnums, &t);

			mps_local_orthonormalize_rq(&psi->a[0], &t);

			delete_block_sparse_tensor(&t);
		}

		// record energy after each sweep
		en_sweeps[n] = en;
	}

	// clean up
	for (int i = 0; i < nsites; i++)
	{
		delete_block_sparse_tensor(&rblocks[i]);
		delete_block_sparse_tensor(&lblocks[i]);
	}
	aligned_free(rblocks);
	aligned_free(lblocks);

	return 0;
}
