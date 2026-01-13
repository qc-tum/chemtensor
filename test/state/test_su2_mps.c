#include <math.h>
#include "su2_mps.h"
#include "mps.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


char* test_su2_mps_orthonormalize_qr()
{
	// number of lattice sites
	const int nsites = 5;

	// 'j' quantum numbers at each site
	qnumber jlist[] = { 0, 2 };
	const struct su2_irreducible_list site_irreps = { .jlist = jlist, .num = ARRLEN(jlist) };

	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                             j:  0  1  2
	const ct_long site_dim_degen[] = { 3, 0, 2 };

	struct rng_state rng_state;
	seed_rng_state(61, &rng_state);

	for (int m = 0; m < 2; m++)  // left- or right-orthonormalization
	{
		for (int is = 0; is < 2; is++)  // overall quantum number sector
		{
			const qnumber irrep_sector = 2 * is;
			const qnumber max_bond_irrep = 5;
			const ct_long max_bond_dim_degen = 13;

			struct su2_mps mps;
			construct_random_su2_mps(CT_DOUBLE_COMPLEX, nsites, &site_irreps, site_dim_degen, irrep_sector, max_bond_irrep, max_bond_dim_degen, &rng_state, &mps);
			// rescale tensors such that overall state norm is around 1
			for (int l = 0; l < nsites; l++)
			{
				const double alpha = 14;
				rscale_su2_tensor(&alpha, &mps.a[l]);
			}
			if (!su2_mps_is_consistent(&mps)) {
				return "internal SU(2) MPS consistency check failed";
			}

			// convert original MPS to state vector
			struct su2_tensor vec_ref;
			su2_mps_to_statevector(&mps, &vec_ref);
			if (vec_ref.ndim_logical != nsites + 2) {
				return "expecting the SU(2) statevector representation of an SU(2) MPS to have logical degree 'nsites + 2'";
			}

			double norm = su2_mps_orthonormalize_qr(&mps, m == 0 ? SU2_MPS_ORTHONORMAL_LEFT : SU2_MPS_ORTHONORMAL_RIGHT);

			if (!su2_mps_is_consistent(&mps)) {
				return "internal SU(2) MPS consistency check failed";
			}

			// convert normalized MPS to state vector
			struct su2_tensor vec;
			su2_mps_to_statevector(&mps, &vec);

			// must be normalized
			struct dense_tensor vec_dns;
			su2_to_dense_tensor(&vec, &vec_dns);
			// for right-orthonormalization, the trailing dummy bond is renormalized as part of the isometric MPS tensor at the last site
			if (fabs(dense_tensor_norm2(&vec_dns) - (m == 1 ? 1 : sqrt(irrep_sector + 1))) > 1e-13) {
				return "vector representation of SU(2) MPS after orthonormalization does not have norm 1 (after compensating for irrep sector)";
			}
			delete_dense_tensor(&vec_dns);

			// scaled vector representation must agree with original vector
			rscale_su2_tensor(&norm, &vec);
			if (!su2_tensor_allclose(&vec, &vec_ref, 1e-13)) {
				return "vector representation of SU(2) MPS after orthonormalization does not match reference";
			}

			delete_su2_tensor(&vec);
			delete_su2_tensor(&vec_ref);

			for (int l = 0; l < nsites; l++)
			{
				// mps.a[l] must be an isometry

				struct su2_tensor a_mat;
				if (m == 0)
				{
					su2_tensor_fuse_axes_add_auxiliary(&mps.a[l], 0, 1, &a_mat);
				}
				else
				{
					struct su2_tensor tmp;
					copy_su2_tensor(&mps.a[l], &tmp);
					su2_tensor_reverse_axis_simple(&tmp, 1);
					su2_tensor_fuse_axes_add_auxiliary(&tmp, 1, 2, &a_mat);
					delete_su2_tensor(&tmp);
				}
				if (!su2_tensor_is_isometry(&a_mat, 1e-13, m == 1)) {
					return "SU(2) MPS tensor is not isometric";
				}
				delete_su2_tensor(&a_mat);

				// also check dense representation
				struct dense_tensor a_dns;
				su2_to_dense_tensor(&mps.a[l], &a_dns);
				assert(a_dns.ndim == 3);
				const ct_long dim_mat[2] = { a_dns.dim[0] * (m == 0 ? a_dns.dim[1] : 1), (m == 0 ? 1 : a_dns.dim[1]) * a_dns.dim[2] };
				reshape_dense_tensor(2, dim_mat, &a_dns);
				if (!dense_tensor_is_isometry(&a_dns, 1e-13, m == 1)) {
					return "dense representation of SU(2) MPS tensor is not isometric";
				}
				delete_dense_tensor(&a_dns);
			}

			delete_su2_mps(&mps);
		}
	}

	return 0;
}


char* test_su2_mps_to_statevector()
{
	// number of lattice sites
	const int nsites = 5;

	// 'j' quantum numbers at each site
	qnumber jlist[] = { 1, 3 };
	const struct su2_irreducible_list site_irreps = { .jlist = jlist, .num = ARRLEN(jlist) };

	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                             j:  0  1  2  3
	const ct_long site_dim_degen[] = { 0, 5, 0, 2 };

	struct rng_state rng_state;
	seed_rng_state(62, &rng_state);

	const qnumber irrep_sector = 1;
	const qnumber max_bond_irrep = 5;
	const ct_long max_bond_dim_degen = 11;

	struct su2_mps mps;
	construct_random_su2_mps(CT_SINGLE_COMPLEX, nsites, &site_irreps, site_dim_degen, irrep_sector, max_bond_irrep, max_bond_dim_degen, &rng_state, &mps);
	// rescale tensors such that overall state norm is around 1
	for (int l = 0; l < nsites; l++)
	{
		const float alpha = 14;
		rscale_su2_tensor(&alpha, &mps.a[l]);
	}
	if (!su2_mps_is_consistent(&mps)) {
		return "internal SU(2) MPS consistency check failed";
	}

	// convert to state vector
	struct su2_tensor vec;
	su2_mps_to_statevector(&mps, &vec);
	if (vec.ndim_logical != nsites + 2) {
		return "expecting the SU(2) statevector representation of an SU(2) MPS to have logical degree 'nsites + 2'";
	}
	struct dense_tensor vec_dns;
	su2_to_dense_tensor(&vec, &vec_dns);
	// include leading and trailing virtual bond dimensions
	const ct_long dim[3] = { 1, dense_tensor_num_elements(&vec_dns) / vec_dns.dim[vec_dns.ndim - 1], vec_dns.dim[vec_dns.ndim - 1] };
	reshape_dense_tensor(ARRLEN(dim), dim, &vec_dns);

	// logical physical dimension
	ct_long d = 0;
	for (int k = 0; k < site_irreps.num; k++)
	{
		const qnumber j = site_irreps.jlist[k];
		assert(site_dim_degen[j] > 0);
		d += site_dim_degen[j] * (j + 1);
	}
	// set all (additive) quantum numbers in reference tensors to zero
	qnumber* qsite = ct_calloc(d, sizeof(qnumber));

	// reference matrix product state
	struct mps mps_ref;
	allocate_empty_mps(nsites, d, qsite, &mps_ref);
	for (int l = 0; l < nsites; l++)
	{
		struct dense_tensor a_loc;
		su2_to_dense_tensor(&mps.a[l], &a_loc);
		if (a_loc.ndim != 3) {
			return "each local SU(2) MPS tensor must have degree 3";
		}
		if (a_loc.dim[1] != d) {
			return "physical dimension of local SU(2) MPS tensor does not agree with reference";
		}

		const enum tensor_axis_direction axis_dir[3] = { TENSOR_AXIS_OUT, TENSOR_AXIS_OUT, TENSOR_AXIS_IN };

		qnumber* qbonds_left  = ct_calloc(a_loc.dim[0], sizeof(qnumber));
		qnumber* qbonds_right = ct_calloc(a_loc.dim[2], sizeof(qnumber));
		const qnumber* qnums[3] = { qbonds_left, qsite, qbonds_right };

		dense_to_block_sparse_tensor(&a_loc, axis_dir, qnums, &mps_ref.a[l]);

		ct_free(qbonds_right);
		ct_free(qbonds_left);

		delete_dense_tensor(&a_loc);
	}
	assert(mps_is_consistent(&mps_ref));

	// reference statevector
	struct block_sparse_tensor vec_ref;
	mps_to_statevector(&mps_ref, &vec_ref);
	if (block_sparse_tensor_norm2(&vec_ref) == 0) {
		return "expecting a non-zero reference statevector for the random SU(2) MPS";
	}

	// compare
	if (!dense_block_sparse_tensor_allclose(&vec_dns, &vec_ref, 1e-5)) {
		return "state vector obtained from SU(2) MPS does not match reference";
	}

	delete_block_sparse_tensor(&vec_ref);
	delete_mps(&mps_ref);
	ct_free(qsite);
	delete_dense_tensor(&vec_dns);
	delete_su2_tensor(&vec);
	delete_su2_mps(&mps);

	return 0;
}
