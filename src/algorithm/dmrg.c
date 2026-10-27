/// \file dmrg.c
/// \brief DMRG algorithm.

#include <math.h>
#include <stdio.h>
#include "dmrg.h"
#include "chain_ops.h"
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
static void apply_local_hamiltonian_wrapper_d(const ct_long n, const void* restrict data, const double* restrict v, double* restrict ret)
{
	// suppress unused parameter warning
	#ifdef NDEBUG
	(void)n;
	#endif

	struct local_hamiltonian_data* hdata = (struct local_hamiltonian_data*)data;

	// interpret input vector as MPS tensor entries
	assert(hdata->a->dtype == CT_DOUBLE_REAL);
	assert(hdata->a->nelem == n);
	block_sparse_tensor_redirect_data_pointer(hdata->a, (void*)v);

	struct block_sparse_tensor ha;
	apply_local_hamiltonian(hdata->a, hdata->w, hdata->l, hdata->r, &ha);

	assert(ha.dtype == CT_DOUBLE_REAL);
	assert(ha.nelem == n);
	memcpy(ret, ha.data, ha.nelem * sizeof(double));

	delete_block_sparse_tensor(&ha);
}


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper function for applying a site-local Hamiltonian operator, required for Lanczos iteration.
///
static void apply_local_hamiltonian_wrapper_z(const ct_long n, const void* restrict data, const dcomplex* restrict v, dcomplex* restrict ret)
{
	// suppress unused parameter warning
	#ifdef NDEBUG
	(void)n;
	#endif

	struct local_hamiltonian_data* hdata = (struct local_hamiltonian_data*)data;

	// interpret input vector as MPS tensor entries
	assert(hdata->a->dtype == CT_DOUBLE_COMPLEX);
	assert(hdata->a->nelem == n);
	block_sparse_tensor_redirect_data_pointer(hdata->a, (void*)v);

	struct block_sparse_tensor ha;
	apply_local_hamiltonian(hdata->a, hdata->w, hdata->l, hdata->r, &ha);

	assert(ha.dtype == CT_DOUBLE_COMPLEX);
	assert(ha.nelem == n);
	memcpy(ret, ha.data, ha.nelem * sizeof(dcomplex));

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

	struct block_sparse_tensor a_iter = *a_start;  // copy internal data pointers
	struct local_hamiltonian_data hdata = { .w = w, .l = l, .r = r, .a = &a_iter };

	allocate_block_sparse_tensor_like(a_start, a_opt);

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
			int ret = eigensystem_krylov_symmetric(a_start->nelem, apply_local_hamiltonian_wrapper_d, &hdata, a_start->data, maxiter, 1, en_min, a_opt->data);
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
			int ret = eigensystem_krylov_hermitian(a_start->nelem, apply_local_hamiltonian_wrapper_z, &hdata, a_start->data, maxiter, 1, en_min, a_opt->data);
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

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Set default single-site DMRG options.
///
void dmrg_singlesite_default_options(struct dmrg_singlesite_options* options)
{
	options->delta_en_tol    = 1e-8;
	options->max_num_sweeps  = 6;
	options->maxiter_lanczos = 20;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete the single-site DMRG information (free memory).
///
void delete_dmrg_singlesite_info(struct dmrg_singlesite_info* info)
{
	ct_free(info->en_sweeps);
	info->num_sweeps = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Run the single-site DMRG algorithm: Approximate the ground state as MPS via left and right sweeps and local single-site optimizations.
/// The input 'psi' is used as the starting state and is updated in-place during the optimization. Its virtual bond dimensions cannot increase.
///
int dmrg_singlesite(const struct mpo* hamiltonian, const struct dmrg_singlesite_options* options, struct mps* psi, struct dmrg_singlesite_info* info)
{
	// number of lattice sites
	const int nsites = hamiltonian->nsites;
	assert(nsites == psi->nsites);
	assert(nsites >= 1);

	// validity check of options
	assert(options->delta_en_tol >= 0);
	assert(options->max_num_sweeps > 0);
	assert(options->maxiter_lanczos > 0);

	// currently only double precision supported
	assert(numeric_real_type(hamiltonian->a[0].dtype) == CT_DOUBLE_REAL);

	// right-normalize input matrix product state
	double nrm = mps_orthonormalize_qr(psi, MPS_ORTHONORMAL_RIGHT);
	if (nrm == 0) {
		printf("Warning: in 'dmrg_singlesite': initial MPS has norm zero (possibly due to mismatching quantum numbers)\n");
	}

	// left and right operator blocks
	struct block_sparse_tensor* lblocks = ct_malloc(nsites * sizeof(struct block_sparse_tensor));
	struct block_sparse_tensor* rblocks = ct_malloc(nsites * sizeof(struct block_sparse_tensor));
	compute_right_operator_blocks(psi, psi, hamiltonian, rblocks);
	create_dummy_operator_block_left(&psi->a[0], &psi->a[0], &hamiltonian->a[0], &lblocks[0]);
	for (int i = 1; i < nsites; i++) {
		copy_block_sparse_tensor(&lblocks[0], &lblocks[i]);
	}

	info->en_sweeps = ct_calloc(options->max_num_sweeps, sizeof(double));

	for (int n = 0; n < options->max_num_sweeps; n++)
	{
		double en;

		// sweep from left to right
		for (int i = 0; i < nsites - 1; i++)
		{
			struct block_sparse_tensor a_opt;
			int ret = minimize_local_energy(&hamiltonian->a[i], &lblocks[i], &rblocks[i], &psi->a[i], options->maxiter_lanczos, &en, &a_opt);
			if (ret < 0) {
				return ret;
			}
			assert(a_opt.ndim == 3);
			delete_block_sparse_tensor(&psi->a[i]);
			psi->a[i] = a_opt;  // copy internal data pointers

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
			int ret = minimize_local_energy(&hamiltonian->a[i], &lblocks[i], &rblocks[i], &psi->a[i], options->maxiter_lanczos, &en, &a_opt);
			if (ret < 0) {
				return ret;
			}
			assert(a_opt.ndim == 3);
			delete_block_sparse_tensor(&psi->a[i]);
			psi->a[i] = a_opt;  // copy internal data pointers

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
			const ct_long dim[3] = { psi->a[0].dim_logical[0], 1, psi->a[0].dim_logical[0] };
			const enum tensor_axis_direction axis_dir[3] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN };
			qnumber qzero[1] = { 0 };
			const qnumber* qnums[3] = { psi->a[0].axis_qnums_logical[0], qzero, psi->a[0].axis_qnums_logical[0] };
			allocate_block_sparse_tensor(psi->a[0].dtype, 3, dim, axis_dir, qnums, &t);

			mps_local_orthonormalize_rq(&psi->a[0], &t);

			delete_block_sparse_tensor(&t);
		}

		// record number of sweeps and energy after each sweep
		info->num_sweeps = n + 1;
		info->en_sweeps[n] = en;

		// stopping criterion based on relative energy difference
		if (n > 0 && fabs(info->en_sweeps[n] - info->en_sweeps[n - 1]) / fmax(fabs(info->en_sweeps[n]), 1) < options->delta_en_tol) {
			break;
		}
	}

	// clean up
	for (int i = 0; i < nsites; i++)
	{
		delete_block_sparse_tensor(&rblocks[i]);
		delete_block_sparse_tensor(&lblocks[i]);
	}
	ct_free(rblocks);
	ct_free(lblocks);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Set default two-site DMRG options.
///
void dmrg_twosite_default_options(struct dmrg_twosite_options* options)
{
	options->delta_en_tol    = 1e-8;
	options->max_num_sweeps  = 6;
	options->maxiter_lanczos = 20;
	options->tol_split       = 1e-10;
	options->max_vdim        = 256;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete the two-site DMRG information (free memory).
///
void delete_dmrg_twosite_info(struct dmrg_twosite_info* info)
{
	ct_free(info->entropy);
	ct_free(info->en_sweeps);
	info->num_sweeps = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Run the two-site DMRG algorithm: Approximate the ground state as MPS via left and right sweeps and local two-site optimizations.
/// The input 'psi' is used as the starting state and is updated in-place during the optimization.
///
int dmrg_twosite(const struct mpo* hamiltonian, const struct dmrg_twosite_options* options, struct mps* psi, struct dmrg_twosite_info* info)
{
	// number of lattice sites
	const int nsites = hamiltonian->nsites;
	assert(nsites == psi->nsites);
	assert(nsites >= 2);

	// validity check of options
	assert(options->delta_en_tol >= 0);
	assert(options->max_num_sweeps > 0);
	assert(options->maxiter_lanczos > 0);
	assert(options->tol_split >= 0);
	assert(options->max_vdim > 0);

	// currently only double precision supported
	assert(numeric_real_type(hamiltonian->a[0].dtype) == CT_DOUBLE_REAL);

	// right-normalize input matrix product state
	double nrm = mps_orthonormalize_qr(psi, MPS_ORTHONORMAL_RIGHT);
	if (nrm == 0) {
		printf("Warning: in 'dmrg_twosite': initial MPS has norm zero (possibly due to mismatching quantum numbers)\n");
	}

	// left and right operator blocks
	struct block_sparse_tensor* lblocks = ct_malloc(nsites * sizeof(struct block_sparse_tensor));
	struct block_sparse_tensor* rblocks = ct_malloc(nsites * sizeof(struct block_sparse_tensor));
	compute_right_operator_blocks(psi, psi, hamiltonian, rblocks);
	create_dummy_operator_block_left(&psi->a[0], &psi->a[0], &hamiltonian->a[0], &lblocks[0]);
	for (int i = 1; i < nsites; i++) {
		copy_block_sparse_tensor(&lblocks[0], &lblocks[i]);
	}

	// precompute merged neighboring Hamiltonian MPO tensors
	struct block_sparse_tensor* h2 = ct_malloc((nsites - 1) * sizeof(struct block_sparse_tensor));
	for (int i = 0; i < nsites - 1; i++) {
		mpo_merge_tensor_pair(&hamiltonian->a[i], &hamiltonian->a[i + 1], &h2[i]);
	}

	info->en_sweeps = ct_calloc(options->max_num_sweeps, sizeof(double));
	info->entropy   = ct_malloc((nsites - 1) * sizeof(double));

	for (int n = 0; n < options->max_num_sweeps; n++)
	{
		double en;

		// sweep from left to right
		for (int i = 0; i < nsites - 2; i++)
		{
			// merge neighboring MPS tensors
			struct block_sparse_tensor a_cur;
			mps_merge_tensor_pair(&psi->a[i], &psi->a[i + 1], &a_cur);
			delete_block_sparse_tensor(&psi->a[i]);
			delete_block_sparse_tensor(&psi->a[i + 1]);

			// minimize local two-site energy using merged tensor as starting point
			struct block_sparse_tensor a_opt;
			int ret = minimize_local_energy(&h2[i], &lblocks[i], &rblocks[i + 1], &a_cur, options->maxiter_lanczos, &en, &a_opt);
			delete_block_sparse_tensor(&a_cur);
			if (ret < 0) {
				return ret;
			}

			// split optimized two-site MPS tensor into two tensors
			const ct_long d_pair[2] = { psi->d, psi->d };
			const qnumber* qsite_pair[2] = { psi->qsite, psi->qsite };
			struct trunc_info trunc_info;
			ret = mps_split_tensor_svd(&a_opt, d_pair, qsite_pair, options->tol_split, options->max_vdim, false, SVD_DISTR_RIGHT, &psi->a[i], &psi->a[i + 1], &trunc_info);
			if (ret < 0) {
				return ret;
			}
			delete_block_sparse_tensor(&a_opt);

			// update the left blocks
			delete_block_sparse_tensor(&lblocks[i + 1]);
			contraction_operator_step_left(&psi->a[i], &psi->a[i], &hamiltonian->a[i], &lblocks[i], &lblocks[i + 1]);
		}

		// sweep from right to left
		for (int i = nsites - 2; i >= 0; i--)
		{
			// merge neighboring MPS tensors
			struct block_sparse_tensor a_cur;
			mps_merge_tensor_pair(&psi->a[i], &psi->a[i + 1], &a_cur);
			delete_block_sparse_tensor(&psi->a[i]);
			delete_block_sparse_tensor(&psi->a[i + 1]);

			// minimize local two-site energy using merged tensor as starting point
			struct block_sparse_tensor a_opt;
			int ret = minimize_local_energy(&h2[i], &lblocks[i], &rblocks[i + 1], &a_cur, options->maxiter_lanczos, &en, &a_opt);
			delete_block_sparse_tensor(&a_cur);
			if (ret < 0) {
				return ret;
			}

			// split optimized two-site MPS tensor into two tensors
			const ct_long d_pair[2] = { psi->d, psi->d };
			const qnumber* qsite_pair[2] = { psi->qsite, psi->qsite };
			struct trunc_info trunc_info;
			ret = mps_split_tensor_svd(&a_opt, d_pair, qsite_pair, options->tol_split, options->max_vdim, false, SVD_DISTR_LEFT, &psi->a[i], &psi->a[i + 1], &trunc_info);
			if (ret < 0) {
				return ret;
			}
			delete_block_sparse_tensor(&a_opt);
			// record entropy
			info->entropy[i] = trunc_info.entropy;

			// update the right blocks
			delete_block_sparse_tensor(&rblocks[i]);
			contraction_operator_step_right(&psi->a[i + 1], &psi->a[i + 1], &hamiltonian->a[i + 1], &rblocks[i + 1], &rblocks[i]);
		}

		// right-normalize leftmost tensor to ensure that 'psi' is normalized
		{
			// dummy tensor at site "-1"
			struct block_sparse_tensor t;
			const ct_long dim[3] = { psi->a[0].dim_logical[0], 1, psi->a[0].dim_logical[0] };
			const enum tensor_axis_direction axis_dir[3] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN };
			qnumber qzero[1] = { 0 };
			const qnumber* qnums[3] = { psi->a[0].axis_qnums_logical[0], qzero, psi->a[0].axis_qnums_logical[0] };
			allocate_block_sparse_tensor(psi->a[0].dtype, 3, dim, axis_dir, qnums, &t);

			mps_local_orthonormalize_rq(&psi->a[0], &t);

			delete_block_sparse_tensor(&t);
		}

		// record number of sweeps and energy after each sweep
		info->num_sweeps = n + 1;
		info->en_sweeps[n] = en;

		// stopping criterion based on relative energy difference
		if (n > 0 && fabs(info->en_sweeps[n] - info->en_sweeps[n - 1]) / fmax(fabs(info->en_sweeps[n]), 1) < options->delta_en_tol) {
			break;
		}
	}

	// clean up
	for (int i = 0; i < nsites - 1; i++)
	{
		delete_block_sparse_tensor(&h2[i]);
	}
	ct_free(h2);
	for (int i = 0; i < nsites; i++)
	{
		delete_block_sparse_tensor(&rblocks[i]);
		delete_block_sparse_tensor(&lblocks[i]);
	}
	ct_free(rblocks);
	ct_free(lblocks);

	return 0;
}
