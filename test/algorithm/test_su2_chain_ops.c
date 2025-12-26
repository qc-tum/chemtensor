#include <math.h>
#include <complex.h>
#include "su2_chain_ops.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


char* test_su2_mpo_inner_product()
{
	// number of lattice sites
	const int nsites = 5;

	// 'j' quantum numbers at each site
	qnumber jlist[] = { 1 };
	const struct su2_irreducible_list site_irreps = { .jlist = jlist, .num = ARRLEN(jlist) };

	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                             j:  0  1
	const ct_long site_dim_degen[] = { 0, 2 };

	struct rng_state rng_state;
	seed_rng_state(73, &rng_state);

	struct su2_mps psi;
	{
		const qnumber irrep_sector = 1;
		const qnumber max_bond_irrep = 5;
		const ct_long max_bond_dim_degen = 4;

		construct_random_su2_mps(CT_SINGLE_COMPLEX, nsites, &site_irreps, site_dim_degen, irrep_sector, max_bond_irrep, max_bond_dim_degen, &rng_state, &psi);
		// rescale tensors such that overall state norm is around 1
		for (int l = 0; l < nsites; l++)
		{
			const float alpha = 6;
			rscale_su2_tensor(&alpha, &psi.a[l]);
		}
		if (!su2_mps_is_consistent(&psi)) {
			return "internal SU(2) MPS consistency check failed";
		}
	}

	struct su2_mps chi;
	{
		const qnumber irrep_sector = 1;
		const qnumber max_bond_irrep = 3;
		const ct_long max_bond_dim_degen = 7;

		construct_random_su2_mps(CT_SINGLE_COMPLEX, nsites, &site_irreps, site_dim_degen, irrep_sector, max_bond_irrep, max_bond_dim_degen, &rng_state, &chi);
		// rescale tensors such that overall state norm is around 1
		for (int l = 0; l < nsites; l++)
		{
			const float alpha = 6;
			rscale_su2_tensor(&alpha, &chi.a[l]);
		}
		if (!su2_mps_is_consistent(&chi)) {
			return "internal SU(2) MPS consistency check failed";
		}
	}

	struct su2_mpo op;
	{
		const qnumber max_bond_irrep = 5;
		const ct_long max_bond_dim_degen = 3;

		construct_random_su2_mpo(CT_SINGLE_COMPLEX, nsites, &site_irreps, site_dim_degen, max_bond_irrep, max_bond_dim_degen, &rng_state, &op);
		// rescale tensors such that overall norm is around 1
		for (int l = 0; l < nsites; l++)
		{
			const float alpha = 10;
			rscale_su2_tensor(&alpha, &op.a[l]);
		}
		if (!su2_mpo_is_consistent(&op)) {
			return "internal SU(2) MPO consistency check failed";
		}
	}

	// compute operator inner product <chi | op | psi>
	struct su2_tensor s;
	su2_mpo_inner_product(&chi, &op, &psi, &s);

	if (!su2_tensor_is_consistent(&s)) {
		return "internal consistency check for SU(2) tensor failed";
	}
	if (s.ndim_logical != 2 || s.ndim_auxiliary != 1) {
		return "SU(2) tensor computed by MPO inner product should have two logical axes and one auxiliary axis";
	}

	// extract numerical value
	if (s.charge_sectors.ndim != 3) {
		return "unexpected charge sector dimension of SU(2) tensor resulting from MPO inner product";
	}
	if (s.charge_sectors.jlists[0] != 0 || s.charge_sectors.jlists[1] != 0 || s.charge_sectors.jlists[2] != 0) {
		return "expecting zero quantum numbers in first charge sector of SU(2) tensor resulting from MPO inner product";
	}
	if (s.degensors[0]->ndim != 2) {
		return "expecting degeneracy tensor of degree 2 in SU(2) tensor resulting from MPO inner product";
	}
	if (s.degensors[0]->dim[0] != 1 || s.degensors[0]->dim[1] != 1) {
		return "degeneracy tensor of zero charge sector in SU(2) tensor resulting from MPO inner product should have dimension 1 x 1";
	}
	const scomplex s_val = ((scomplex*)s.degensors[0]->data)[0];
	delete_su2_tensor(&s);

	// reference calculation

	// logical physical dimension
	ct_long d = 0;
	for (int k = 0; k < site_irreps.num; k++)
	{
		const qnumber j = site_irreps.jlist[k];
		assert(site_dim_degen[j] > 0);
		d += site_dim_degen[j] * (j + 1);
	}

	struct dense_tensor psi_vec;
	{
		struct su2_tensor psi_su2_vec;
		su2_mps_to_statevector(&psi, &psi_su2_vec);
		su2_to_dense_tensor(&psi_su2_vec, &psi_vec);
		delete_su2_tensor(&psi_su2_vec);
		// keep trailing virtual bond as separate dimension
		const ct_long dim[2] = { integer_product(psi_vec.dim, psi_vec.ndim - 1), psi_vec.dim[psi_vec.ndim - 1] };
		reshape_dense_tensor(2, dim, &psi_vec);
	}
	struct dense_tensor chi_vec;
	{
		struct su2_tensor chi_su2_vec;
		su2_mps_to_statevector(&chi, &chi_su2_vec);
		su2_to_dense_tensor(&chi_su2_vec, &chi_vec);
		delete_su2_tensor(&chi_su2_vec);
		// keep trailing virtual bond as separate dimension
		const ct_long dim[2] = { integer_product(chi_vec.dim, chi_vec.ndim - 1), chi_vec.dim[chi_vec.ndim - 1] };
		reshape_dense_tensor(2, dim, &chi_vec);
	}
	struct dense_tensor op_mat;
	{
		struct su2_tensor op_tensor;
		su2_mpo_to_tensor(&op, &op_tensor);
		su2_to_dense_tensor(&op_tensor, &op_mat);
		delete_su2_tensor(&op_tensor);
		// reshape into a matrix
		const ct_long hspace_dim = ipow(d, nsites);
		const ct_long dim[2] = { hspace_dim, hspace_dim };
		reshape_dense_tensor(2, dim, &op_mat);
	}

	// reference <chi | op | psi>
	scomplex s_ref;
	{
		struct dense_tensor t;
		dense_tensor_dot(&op_mat, TENSOR_AXIS_RANGE_TRAILING, &psi_vec, TENSOR_AXIS_RANGE_LEADING, 1, &t);
		assert(t.ndim == 2);  // combined physical axis and trailing virtual bond axis of 'psi'
		conjugate_dense_tensor(&chi_vec);
		struct dense_tensor r;
		dense_tensor_dot(&chi_vec, TENSOR_AXIS_RANGE_LEADING, &t, TENSOR_AXIS_RANGE_LEADING, 1, &r);
		delete_dense_tensor(&t);
		assert(r.dtype == CT_SINGLE_COMPLEX);
		assert(r.ndim == 2);  // trailing virtual bond axes of 'chi' and 'psi'
		// trace out trailing virtual bonds
		dense_tensor_trace(&r, &s_ref);
		delete_dense_tensor(&r);
		// compensate for Clebsch-Gordan factor from fused axes in initial "dummy" right operator block
		s_ref /= (-sqrtf(2.f));
	}

	// compare
	if (cabsf(s_val - s_ref) / cabsf(s_ref) > 5e-6) {
		return "SU(2) operator inner product does not match reference value";
	}

	delete_dense_tensor(&op_mat);
	delete_dense_tensor(&chi_vec);
	delete_dense_tensor(&psi_vec);
	delete_su2_mpo(&op);
	delete_su2_mps(&chi);
	delete_su2_mps(&psi);

	return 0;
}
