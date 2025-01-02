/// \file thc.c
/// \brief Tensor hypercontraction (THC) representation of a molecular Hamiltonian and corresponding utility functions (see <a href="https://arxiv.org/abs/2409.12708">arXiv:2409.12708</a>).

#include <assert.h>
#include "thc.h"
#include "hamiltonian.h"
#include "chain_ops.h"
#include "aligned_memory.h"


//________________________________________________________________________________________________________________________
///
/// \brief Construct the tensor hypercontraction representation of a molecular Hamiltonian.
///
int construct_thc_spin_molecular_hamiltonian(const struct dense_tensor* restrict tkin, const struct dense_tensor* restrict thc_kernel,
	const struct dense_tensor* restrict thc_transform, struct thc_spin_molecular_hamiltonian* hamiltonian)
{
	// consistency checks
	assert(tkin->dtype == CT_DOUBLE_REAL);
	// require real data types due to use of elementary THC MPOs
	assert(thc_kernel->dtype    == CT_DOUBLE_REAL);
	assert(thc_transform->dtype == CT_DOUBLE_REAL);
	// symmetry
	assert(dense_tensor_is_self_adjoint(tkin, 1e-14));
	assert(dense_tensor_is_self_adjoint(thc_kernel, 1e-14));
	// dimensions
	assert(thc_transform->ndim == 2);
	assert(thc_transform->dim[1] == thc_kernel->dim[0]);
	assert(thc_transform->dim[0] == tkin->dim[0]);

	const int nsites   = (int)tkin->dim[0];
	const int thc_rank = (int)thc_kernel->dim[0];

	copy_dense_tensor(tkin,          &hamiltonian->tkin);
	copy_dense_tensor(thc_kernel,    &hamiltonian->thc_kernel);
	copy_dense_tensor(thc_transform, &hamiltonian->thc_transform);

	// generate the internal elementary MPO terms

	const int perm[2] = { 1, 0 };

	// MPOs for the kinetic term
	// diagonalize kinetic coefficient matrix
	int ret = dense_tensor_eigh(tkin, &hamiltonian->u_kin, &hamiltonian->en_kin);
	if (ret < 0) {
		return ret;
	}
	hamiltonian->mpo_kin = ct_malloc(2*nsites * sizeof(struct mpo));
	struct dense_tensor u_kin_t;
	transpose_dense_tensor(perm, &hamiltonian->u_kin, &u_kin_t);
	assert(u_kin_t.dtype == CT_DOUBLE_REAL);
	assert(u_kin_t.dim[0] == nsites && u_kin_t.dim[1] == nsites);
	for (int i = 0; i < nsites; i++)
	{
		for (int sigma = 0; sigma < 2; sigma++)
		{
			const double* coeff = &((double*)u_kin_t.data)[i*nsites];
			struct mpo_assembly assembly;
			construct_quadratic_spin_fermionic_mpo_assembly(nsites, coeff, coeff, sigma, &assembly);
			mpo_from_assembly(&assembly, &hamiltonian->mpo_kin[2*i + sigma]);
			delete_mpo_assembly(&assembly);
		}
	}
	delete_dense_tensor(&u_kin_t);

	// elementary MPOs for the interaction (Coulomb) term in THC representation
	hamiltonian->mpo_thc = ct_malloc(2*thc_rank * sizeof(struct mpo));
	struct dense_tensor thc_transform_t;
	transpose_dense_tensor(perm, &hamiltonian->thc_transform, &thc_transform_t);
	assert(thc_transform_t.dtype == CT_DOUBLE_REAL);
	assert(thc_transform_t.dim[0] == thc_rank && thc_transform_t.dim[1] == nsites);
	for (int mu = 0; mu < thc_rank; mu++)
	{
		for (int sigma = 0; sigma < 2; sigma++)
		{
			const double* coeff = &((double*)thc_transform_t.data)[mu*nsites];
			struct mpo_assembly assembly;
			construct_quadratic_spin_fermionic_mpo_assembly(nsites, coeff, coeff, sigma, &assembly);
			mpo_from_assembly(&assembly, &hamiltonian->mpo_thc[2*mu + sigma]);
			delete_mpo_assembly(&assembly);
		}
	}
	delete_dense_tensor(&thc_transform_t);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Delete the tensor hypercontraction representation of a molecular Hamiltonian (free memory).
///
void delete_thc_spin_molecular_hamiltonian(struct thc_spin_molecular_hamiltonian* hamiltonian)
{
	const int nsites   = (int)hamiltonian->tkin.dim[0];
	const int thc_rank = (int)hamiltonian->thc_kernel.dim[0];

	for (int ms = 0; ms < 2*thc_rank; ms++) {
		delete_mpo(&hamiltonian->mpo_thc[ms]);
	}
	ct_free(hamiltonian->mpo_thc);
	hamiltonian->mpo_thc = NULL;

	for (int is = 0; is < 2*nsites; is++) {
		delete_mpo(&hamiltonian->mpo_kin[is]);
	}
	ct_free(hamiltonian->mpo_kin);
	hamiltonian->mpo_kin = NULL;

	delete_dense_tensor(&hamiltonian->u_kin);
	delete_dense_tensor(&hamiltonian->en_kin);

	delete_dense_tensor(&hamiltonian->thc_transform);
	delete_dense_tensor(&hamiltonian->thc_kernel);
	delete_dense_tensor(&hamiltonian->tkin);
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply a molecular Hamiltonian in tensor hypercontraction representation to a state in MPS form.
///
int apply_thc_spin_molecular_hamiltonian(const struct thc_spin_molecular_hamiltonian* hamiltonian,
	const struct mps* restrict psi, const double tol, const long max_vdim, struct mps* restrict h_psi)
{
	const int nsites   = (int)hamiltonian->tkin.dim[0];
	const int thc_rank = (int)hamiltonian->thc_kernel.dim[0];
	assert(psi->nsites == nsites);

	struct trunc_info* info = ct_malloc(nsites * sizeof(struct trunc_info));

	// kinetic term
	assert(hamiltonian->en_kin.dtype == CT_DOUBLE_REAL);
	const double* en_kin_data = hamiltonian->en_kin.data;
	for (int i = 0; i < nsites; i++)
	{
		for (int sigma = 0; sigma < 2; sigma++)
		{
			struct mps kin_psi;
			apply_mpo(&hamiltonian->mpo_kin[2*i + sigma], psi, &kin_psi);
			double trunc_scale;
			int ret = mps_compress_rescale(tol, max_vdim, MPS_ORTHONORMAL_LEFT, &kin_psi, &trunc_scale, info);
			if (ret < 0) {
				return ret;
			}
			assert(kin_psi.a[kin_psi.nsites - 1].dtype == CT_DOUBLE_REAL);
			scale_block_sparse_tensor(&en_kin_data[i], &kin_psi.a[kin_psi.nsites - 1]);

			if (i > 0 || sigma > 0)
			{
				// accumulate
				struct mps tmp;
				mps_add(h_psi, &kin_psi, &tmp);
				delete_mps(&kin_psi);
				delete_mps(h_psi);
				double trunc_scale;
				int ret = mps_compress_rescale(tol, max_vdim, MPS_ORTHONORMAL_LEFT, &tmp, &trunc_scale, info);
				if (ret < 0) {
					return ret;
				}
				move_mps_data(&tmp, h_psi);
			}
			else
			{
				move_mps_data(&kin_psi, h_psi);
			}
		}
	}

	// interaction term
	assert(hamiltonian->thc_kernel.dtype == CT_DOUBLE_REAL);
	const double* thc_kernel_data = hamiltonian->thc_kernel.data;
	for (int nu = 0; nu < thc_rank; nu++)
	{
		for (int tau = 0; tau < 2; tau++)
		{
			struct mps thc1_psi;
			apply_mpo(&hamiltonian->mpo_thc[2*nu + tau], psi, &thc1_psi);
			double trunc_scale;
			int ret = mps_compress_rescale(tol, max_vdim, MPS_ORTHONORMAL_LEFT, &thc1_psi, &trunc_scale, info);
			if (ret < 0) {
				return ret;
			}
			for (int mu = 0; mu < thc_rank; mu++)
			{
				struct mps thc1_k_psi;
				copy_mps(&thc1_psi, &thc1_k_psi);
				const double alpha = 0.5 * thc_kernel_data[mu*thc_rank + nu];
				scale_block_sparse_tensor(&alpha, &thc1_k_psi.a[thc1_k_psi.nsites - 1]);

				for (int sigma = 0; sigma < 2; sigma++)
				{
					struct mps thc2_psi;
					apply_mpo(&hamiltonian->mpo_thc[2*mu + sigma], &thc1_k_psi, &thc2_psi);
					double trunc_scale;
					int ret = mps_compress_rescale(tol, max_vdim, MPS_ORTHONORMAL_LEFT, &thc2_psi, &trunc_scale, info);
					if (ret < 0) {
						return ret;
					}

					// accumulate
					struct mps tmp;
					mps_add(h_psi, &thc2_psi, &tmp);
					delete_mps(&thc2_psi);
					delete_mps(h_psi);
					ret = mps_compress_rescale(tol, max_vdim, MPS_ORTHONORMAL_LEFT, &tmp, &trunc_scale, info);
					if (ret < 0) {
						return ret;
					}
					move_mps_data(&tmp, h_psi);
				}

				delete_mps(&thc1_k_psi);
			}

			delete_mps(&thc1_psi);
		}
	}

	ct_free(info);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Generate the matrix representation of the tensor hypercontraction molecular Hamiltonian on the full Hilbert space.
///
int thc_spin_molecular_hamiltonian_to_matrix(const struct thc_spin_molecular_hamiltonian* hamiltonian, struct block_sparse_tensor* mat)
{
	const int nsites   = (int)hamiltonian->tkin.dim[0];
	const int thc_rank = (int)hamiltonian->thc_kernel.dim[0];

	// kinetic term
	const double* en_kin_data = hamiltonian->en_kin.data;
	for (int i = 0; i < nsites; i++)
	{
		for (int sigma = 0; sigma < 2; sigma++)
		{
			if (i > 0 || sigma > 0)
			{
				struct block_sparse_tensor mat_loc;
				mpo_to_matrix(&hamiltonian->mpo_kin[2*i + sigma], &mat_loc);
				block_sparse_tensor_scalar_multiply_add(&en_kin_data[i], &mat_loc, mat);
				delete_block_sparse_tensor(&mat_loc);
			}
			else
			{
				// first term
				assert(i == 0 && sigma == 0);
				mpo_to_matrix(&hamiltonian->mpo_kin[0], mat);
				scale_block_sparse_tensor(&en_kin_data[0], mat);
			}
		}
	}

	// convert individual THC MPOs to sparse matrices
	struct block_sparse_tensor* mat_thc = ct_malloc(2*thc_rank * sizeof(struct block_sparse_tensor));
	for (int ms = 0; ms < 2*thc_rank; ms++) {
		mpo_to_matrix(&hamiltonian->mpo_thc[ms], &mat_thc[ms]);
	}

	// diagonalize the THC kernel
	struct dense_tensor u_kernel;
	struct dense_tensor lambda_kernel;
	int ret = dense_tensor_eigh(&hamiltonian->thc_kernel, &u_kernel, &lambda_kernel);
	if (ret < 0) {
		return ret;
	}
	assert(u_kernel.dtype      == CT_DOUBLE_REAL);
	assert(lambda_kernel.dtype == CT_DOUBLE_REAL);

	// add interaction terms
	const double* u_kernel_data = u_kernel.data;
	const double* lambda_kernel_data = lambda_kernel.data;
	for (int nu = 0; nu < thc_rank; nu++)
	{
		struct block_sparse_tensor g;
		for (int mu = 0; mu < thc_rank; mu++)
		{
			for (int sigma = 0; sigma < 2; sigma++)
			{
				if (mu > 0 || sigma > 0)
				{
					block_sparse_tensor_scalar_multiply_add(&u_kernel_data[mu*thc_rank + nu], &mat_thc[2*mu + sigma], &g);
				}
				else
				{
					// first term
					assert(mu == 0 && sigma == 0);
					copy_block_sparse_tensor(&mat_thc[0], &g);
					scale_block_sparse_tensor(&u_kernel_data[nu], &g);
				}
			}
		}
		// compute g^2
		// remove dummy virtual bond axes for multiplication
		assert(g.ndim == 4);
		assert(g.dim_logical[0] == 1 && g.dim_logical[3] == 1);
		assert(g.qnums_logical[0][0] == 0 && g.qnums_logical[3][0] == 0);
		assert(g.axis_dir[1] == TENSOR_AXIS_OUT);
		assert(g.axis_dir[2] == TENSOR_AXIS_IN);
		struct block_sparse_tensor gl, gr;
		flatten_block_sparse_tensor_axes(&g, 2, TENSOR_AXIS_IN,  &gl);
		flatten_block_sparse_tensor_axes(&g, 0, TENSOR_AXIS_OUT, &gr);
		delete_block_sparse_tensor(&g);
		// multiply tensors
		struct block_sparse_tensor g2;
		block_sparse_tensor_dot(&gl, TENSOR_AXIS_RANGE_TRAILING, &gr, TENSOR_AXIS_RANGE_LEADING, 1, &g2);
		delete_block_sparse_tensor(&gr);
		delete_block_sparse_tensor(&gl);
		// accumulate
		const double alpha = 0.5 * lambda_kernel_data[nu];
		block_sparse_tensor_scalar_multiply_add(&alpha, &g2, mat);
		delete_block_sparse_tensor(&g2);
	}

	delete_dense_tensor(&lambda_kernel);
	delete_dense_tensor(&u_kernel);

	for (int ms = 0; ms < 2*thc_rank; ms++) {
		delete_block_sparse_tensor(&mat_thc[ms]);
	}
	ct_free(mat_thc);

	return 0;
}
