/// \file su2_hamiltonian.c
/// \brief Construction of common SU(2) symmetric quantum Hamiltonians.

#include <assert.h>
#include "su2_hamiltonian.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


//________________________________________________________________________________________________________________________
///
/// \brief Construct the SU(2) symmetric MPO representation of the XXX Heisenberg Hamiltonian 'sum J (X X + Y Y + Z Z)' on a one-dimensional lattice.
///
void construct_heisenberg_1d_su2_mpo(const int nsites, const double J, struct su2_mpo* mpo)
{
	// structural tensors (Clebsch-Gordan coefficients) generate the
	// 1/2 (S_up ⊗ S_down + S_down ⊗ S_up) + S_z ⊗ S_z term,
	// up to a prefactor of -4/3
	//
	// general layout of each MPO tensor in the bulk (with shown bond irreps multiplied by 2):
	// j_bond 0    0    2
	//   0 (  1    0  -3/4 J )
	//   0 (  0    1    0    )
	//   2 (  0    1    0    )
	//
	// generates the sequence:
	//      op   j    op   j    op   j    op   j    op   j    op
	// ...  I    0    I    0 -3J/4 S 2    S    0    I    0    I  ...

	assert(nsites >= 2);

	// physical quantum numbers (multiplied by 2)
	qnumber site_jlist[1] = { 1 };
	const struct su2_irreducible_list site_irreps = { .jlist = site_jlist, .num = ARRLEN(site_jlist) };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                             j:  0  1
	const ct_long site_dim_degen[] = { 0, 1 };

	// bond quantum numbers (multiplied by 2)
	qnumber bond_jlist[2] = { 0, 2 };
	struct su2_irreducible_list bond_irreps_bulk     = { .jlist = bond_jlist, .num = 2 };
	struct su2_irreducible_list bond_irreps_boundary = { .jlist = bond_jlist, .num = 1 };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                                      j:  0  1  2
	const ct_long bond_dim_degen_bulk[]     = { 2, 0, 1 };
	const ct_long bond_dim_degen_boundary[] = { 1       };

	struct su2_irreducible_list* bond_irreps = ct_malloc((nsites + 1) * sizeof(struct su2_irreducible_list));
	const ct_long** bond_dim_degen = ct_malloc((nsites + 1) * sizeof(ct_long*));
	// i = 0
	{
		bond_irreps[0] = bond_irreps_boundary;
		bond_dim_degen[0] = bond_dim_degen_boundary;
	}
	for (int i = 1; i < nsites; i++)
	{
		bond_irreps[i] = bond_irreps_bulk;
		bond_dim_degen[i] = bond_dim_degen_bulk;
	}
	// i = nsites
	{
		bond_irreps[nsites] = bond_irreps_boundary;
		bond_dim_degen[nsites] = bond_dim_degen_boundary;
	}

	allocate_su2_mpo(CT_DOUBLE_REAL, nsites, &site_irreps, site_dim_degen, bond_irreps, bond_dim_degen, mpo);

	ct_free(bond_dim_degen);
	ct_free(bond_irreps);

	// degeneracy tensors
	// i = 0
	{
		assert(mpo->a[0].ndim_logical == 4);
		assert(mpo->a[0].charge_sectors.ndim == 5);
		assert(mpo->a[0].charge_sectors.nsec == 2);

		// degeneracy tensor acting on bond irrep 0
		{
			#ifndef NDEBUG
			const qnumber* jlists = &mpo->a[0].charge_sectors.jlists[0];
			assert(
				jlists[0] == 0 &&
				jlists[1] == 1 &&
				jlists[2] == 1 &&
				jlists[3] == 0 &&
				jlists[4] == 1);
			#endif
			struct dense_tensor* dt = mpo->a[0].degensors[0];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 1 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 2);
			assert(dt->dtype == CT_DOUBLE_REAL);
			double* data = dt->data;
			data[0] = 1;
			data[1] = 0;
		}
		// transition from left bond irrep 0 to right bond irrep 2, implementing left part of S dot S
		{
			#ifndef NDEBUG
			const qnumber* jlists = &mpo->a[0].charge_sectors.jlists[5];
			assert(
				jlists[0] == 0 &&
				jlists[1] == 1 &&
				jlists[2] == 1 &&
				jlists[3] == 2 &&
				jlists[4] == 1);
			#endif
			struct dense_tensor* dt = mpo->a[0].degensors[1];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 1 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 1);
			assert(dt->dtype == CT_DOUBLE_REAL);
			double* data = dt->data;
			data[0] = -0.75 * J;
		}
	}
	for (int i = 1; i < nsites - 1; i++)
	{
		assert(mpo->a[i].ndim_logical == 4);
		assert(mpo->a[i].charge_sectors.ndim == 5);
		assert(mpo->a[i].charge_sectors.nsec == 5);

		// set degeneracy tensor acting on bond irrep 0 to identity map
		{
			#ifndef NDEBUG
			const qnumber* jlists = &mpo->a[i].charge_sectors.jlists[0];
			assert(
				jlists[0] == 0 &&
				jlists[1] == 1 &&
				jlists[2] == 1 &&
				jlists[3] == 0 &&
				jlists[4] == 1);
			#endif
			struct dense_tensor* dt = mpo->a[i].degensors[0];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 2 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 2);
			assert(dt->dtype == CT_DOUBLE_REAL);
			double* data = dt->data;
			data[0] = 1;
			data[3] = 1;
		}
		// transition from left bond irrep 0 to right bond irrep 2, implementing left part of S dot S
		{
			#ifndef NDEBUG
			const qnumber* jlists = &mpo->a[i].charge_sectors.jlists[5];
			assert(
				jlists[0] == 0 &&
				jlists[1] == 1 &&
				jlists[2] == 1 &&
				jlists[3] == 2 &&
				jlists[4] == 1);
			#endif
			struct dense_tensor* dt = mpo->a[i].degensors[1];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 2 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 1);
			assert(dt->dtype == CT_DOUBLE_REAL);
			double* data = dt->data;
			data[0] = -0.75 * J;
			data[1] = 0;
		}
		// transition from left bond irrep 2 to right bond irrep 0, implementing right part of S dot S
		{
			#ifndef NDEBUG
			const qnumber* jlists = &mpo->a[i].charge_sectors.jlists[10];
			assert(
				jlists[0] == 2 &&
				jlists[1] == 1 &&
				jlists[2] == 1 &&
				jlists[3] == 0 &&
				jlists[4] == 1);
			#endif
			struct dense_tensor* dt = mpo->a[i].degensors[2];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 1 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 2);
			double* data = dt->data;
			data[0] = 0;
			data[1] = 1;
		}
		// remaining degeneracy tensors are not required
		{
			#ifndef NDEBUG
			const qnumber* jlists = &mpo->a[i].charge_sectors.jlists[15];
			assert(
				jlists[0] == 2 &&
				jlists[1] == 1 &&
				jlists[2] == 1 &&
				jlists[3] == 2 &&
				jlists[4] == 1);
			#endif
			su2_tensor_delete_charge_sector_by_index(&mpo->a[i], 3);
		}
		{
			#ifndef NDEBUG
			const qnumber* jlists = &mpo->a[i].charge_sectors.jlists[15];
			assert(
				jlists[0] == 2 &&
				jlists[1] == 1 &&
				jlists[2] == 1 &&
				jlists[3] == 2 &&
				jlists[4] == 3);
			#endif
			su2_tensor_delete_charge_sector_by_index(&mpo->a[i], 3);
		}

		assert(mpo->a[i].charge_sectors.nsec == 3);
	}
	// i = nsites - 1
	{
		assert(mpo->a[nsites - 1].ndim_logical == 4);
		assert(mpo->a[nsites - 1].charge_sectors.ndim == 5);
		assert(mpo->a[nsites - 1].charge_sectors.nsec == 2);

		// degeneracy tensor acting on bond irrep 0
		{
			#ifndef NDEBUG
			const qnumber* jlists = &mpo->a[nsites - 1].charge_sectors.jlists[0];
			assert(
				jlists[0] == 0 &&
				jlists[1] == 1 &&
				jlists[2] == 1 &&
				jlists[3] == 0 &&
				jlists[4] == 1);
			#endif
			struct dense_tensor* dt = mpo->a[nsites - 1].degensors[0];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 2 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 1);
			assert(dt->dtype == CT_DOUBLE_REAL);
			double* data = dt->data;
			data[0] = 0;
			data[1] = 1;
		}
		// transition from left bond irrep 2 to right bond irrep 0, implementing right part of S dot S
		{
			#ifndef NDEBUG
			const qnumber* jlists = &mpo->a[nsites - 1].charge_sectors.jlists[5];
			assert(
				jlists[0] == 2 &&
				jlists[1] == 1 &&
				jlists[2] == 1 &&
				jlists[3] == 0 &&
				jlists[4] == 1);
			#endif
			struct dense_tensor* dt = mpo->a[nsites - 1].degensors[1];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 1 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 1);
			double* data = dt->data;
			data[0] = 1;
		}
	}
}
