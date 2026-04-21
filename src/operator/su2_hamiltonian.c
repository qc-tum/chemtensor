/// \file su2_hamiltonian.c
/// \brief Construction of common SU(2) symmetric quantum Hamiltonians.

#include <assert.h>
#include "su2_hamiltonian.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


//________________________________________________________________________________________________________________________
///
/// \brief Set a single entry of a degree-four tensor with entries of type double (helper function).
///
static inline void set_four_tensor_data_entry(
	struct dense_tensor* t,
	const ct_long i0,
	const ct_long i1,
	const ct_long i2,
	const ct_long i3,
	const double val)
{
	assert(t->ndim == 4);
	assert(0 <= i0 && i0 < t->dim[0]);
	assert(0 <= i1 && i1 < t->dim[1]);
	assert(0 <= i2 && i2 < t->dim[2]);
	assert(0 <= i3 && i3 < t->dim[3]);
	assert(t->dtype == CT_DOUBLE_REAL);
	double* data = t->data;
	data[((i0*t->dim[1] + i1)*t->dim[2] + i2)*t->dim[3] + i3] = val;
}


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
		struct su2_tensor* a_loc = &mpo->a[0];

		assert(a_loc->ndim_logical == 4);
		assert(a_loc->charge_sectors.ndim == 5);
		assert(a_loc->charge_sectors.nsec == 2);

		// degeneracy tensor acting on bond irrep 0, implementing left-connected identity string
		{
			const ct_long c = 0;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 0 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 0 &&  // right virtual bond
				jlists[4] == 1);   // inner tree axis
			#endif
			//           j=1
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=0 ─<─0───┤   │
			//        │   ├───3─<─ j=0
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=1

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 1 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 2);
			set_four_tensor_data_entry(dt, 0, 0, 0, 0, 1.0);
		}
		// transition from left bond irrep 0 to right bond irrep 2, implementing the left part of S dot S
		{
			const ct_long c = 1;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 0 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 2 &&  // right virtual bond
				jlists[4] == 1);   // inner tree axis
			#endif
			//           j=1
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=0 ─<─0───┤   │
			//        │   ├───3─<─ j=2
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=1

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 1 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 1);
			set_four_tensor_data_entry(dt, 0, 0, 0, 0, -0.75 * J);
		}
	}
	for (int i = 1; i < nsites - 1; i++)
	{
		struct su2_tensor* a_loc = &mpo->a[i];

		assert(a_loc->ndim_logical == 4);
		assert(a_loc->charge_sectors.ndim == 5);
		assert(a_loc->charge_sectors.nsec == 5);

		// set degeneracy tensor acting on bond irrep 0 to identity map
		{
			const ct_long c = 0;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 0 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 0 &&  // right virtual bond
				jlists[4] == 1);   // inner tree axis
			#endif
			//           j=1
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=0 ─<─0───┤   │
			//        │   ├───3─<─ j=0
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=1

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 2 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 2);
			set_four_tensor_data_entry(dt, 0, 0, 0, 0, 1.0);
			set_four_tensor_data_entry(dt, 1, 0, 0, 1, 1.0);
		}
		// transition from left bond irrep 0 to right bond irrep 2, implementing the left part of S dot S
		{
			const ct_long c = 1;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 0 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 2 &&  // right virtual bond
				jlists[4] == 1);   // inner tree axis
			#endif
			//           j=1
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=0 ─<─0───┤   │
			//        │   ├───3─<─ j=2
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=1

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 2 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 1);
			set_four_tensor_data_entry(dt, 0, 0, 0, 0, -0.75 * J);
		}
		// transition from left bond irrep 2 to right bond irrep 0, implementing the right part of S dot S
		{
			const ct_long c = 2;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 2 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 0 &&  // right virtual bond
				jlists[4] == 1);   // inner tree axis
			#endif
			//           j=1
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=2 ─<─0───┤   │
			//        │   ├───3─<─ j=0
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=1

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 1 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 2);
			set_four_tensor_data_entry(dt, 0, 0, 0, 1, 1.0);
		}
		// remaining degeneracy tensors are not required
		{
			const ct_long c = 3;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 2 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 2 &&  // right virtual bond
				jlists[4] == 1);   // inner tree axis
			#endif
			su2_tensor_delete_charge_sector_by_index(a_loc, c);
		}
		{
			const ct_long c = 3;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 2 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 2 &&  // right virtual bond
				jlists[4] == 3);   // inner tree axis
			#endif
			su2_tensor_delete_charge_sector_by_index(a_loc, c);
		}

		assert(a_loc->charge_sectors.nsec == 3);
	}
	// i = nsites - 1
	{
		struct su2_tensor* a_loc = &mpo->a[nsites - 1];

		assert(a_loc->ndim_logical == 4);
		assert(a_loc->charge_sectors.ndim == 5);
		assert(a_loc->charge_sectors.nsec == 2);

		// degeneracy tensor acting on bond irrep 0, implementing right-connected identity string
		{
			const ct_long c = 0;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 0 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 0 &&  // right virtual bond
				jlists[4] == 1);   // inner tree axis
			#endif
			//           j=1
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=0 ─<─0───┤   │
			//        │   ├───3─<─ j=0
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=1

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 2 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 1);
			set_four_tensor_data_entry(dt, 1, 0, 0, 0, 1.0);
		}
		// transition from left bond irrep 2 to right bond irrep 0, implementing the right part of S dot S
		{
			const ct_long c = 1;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 2 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 0 &&  // right virtual bond
				jlists[4] == 1);   // inner tree axis
			#endif
			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 1 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 1);
			set_four_tensor_data_entry(dt, 0, 0, 0, 0, 1.0);
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct the SU(2) symmetric MPO representation of the Fermi-Hubbard Hamiltonian with nearest-neighbor hopping on a one-dimensional lattice.
///
void construct_fermi_hubbard_1d_su2_mpo(const int nsites, const double t, const double u, const double mu, struct su2_mpo* mpo)
{
	// SU(2) tree structure relevant for kinetic hopping terms,
	// with the diagonal segment the virtual bond,
	// requiring separate channels for left and right hopping
	//
	// site i  i+1
	//      │   │
	//      │╲  ^
	//      │ ╲ │
	//      ^  ╲│
	//      │   │

	// virtual bond channels
	//            constant           channel index    j_bond  description
	const ct_long BOND_IDX_TERMINAL          = 0;  //    0    left and right terminal bonds
	const ct_long BOND_IDX_IDSTRING_LEFT     = 0;  //    0    left-connected identity string
	const ct_long BOND_IDX_IDSTRING_RIGHT    = 1;  //    0    right-connected identity string
	const ct_long BOND_IDX_HOP_LEFT_TO_RIGHT = 0;  //    1    first "hopping" channel: from left to right
	const ct_long BOND_IDX_HOP_RIGHT_TO_LEFT = 1;  //    1    second "hopping" channel: from right to left

	assert(nsites >= 2);

	// physical basis states are ordered as (|00>, |11>, |10>, |01>)
	// physical quantum numbers (multiplied by 2)
	qnumber site_jlist[2] = { 0, 1 };
	const struct su2_irreducible_list site_irreps = { .jlist = site_jlist, .num = ARRLEN(site_jlist) };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                             j:  0  1
	const ct_long site_dim_degen[] = { 2, 1 };

	// bond quantum numbers (multiplied by 2)
	qnumber bond_jlist[2] = { 0, 1 };
	struct su2_irreducible_list bond_irreps_bulk     = { .jlist = bond_jlist, .num = 2 };
	struct su2_irreducible_list bond_irreps_boundary = { .jlist = bond_jlist, .num = 1 };
	// degeneracy dimensions, indexed by 'j' quantum numbers
	//                                      j:  0  1
	const ct_long bond_dim_degen_bulk[]     = { 2, 2 };  // left and right-connected identity strings for j = 0, and
	                                                     // hopping from left to right and right to left for j = 1
	const ct_long bond_dim_degen_boundary[] = { 1    };

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

	const double sqrt2 = 1.4142135623730950488;

	// degeneracy tensors
	// i = 0
	{
		struct su2_tensor* a_loc = &mpo->a[0];

		assert(a_loc->ndim_logical == 4);
		assert(a_loc->charge_sectors.ndim == 5);
		assert(a_loc->charge_sectors.nsec == 4);

		// degeneracy tensor acting on bond irrep 0, implementing left-connected identity string and
		// parts of the interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn) terms
		{
			const ct_long c = 0;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 0 &&  // left virtual bond
				jlists[1] == 0 &&  // physical output axis
				jlists[2] == 0 &&  // physical input axis
				jlists[3] == 0 &&  // right virtual bond
				jlists[4] == 0);   // inner tree axis
			#endif
			//           j=0
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=0 ─<─0───┤   │
			//        │   ├───3─<─ j=0
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=0

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 1 &&
				dt->dim[1] == 2 &&
				dt->dim[2] == 2 &&
				dt->dim[3] == 2);
			set_four_tensor_data_entry(dt, BOND_IDX_TERMINAL, 0, 0, BOND_IDX_IDSTRING_LEFT, 1.0);
			set_four_tensor_data_entry(dt, BOND_IDX_TERMINAL, 1, 1, BOND_IDX_IDSTRING_LEFT, 1.0);
			// parts of the interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn) terms
			set_four_tensor_data_entry(dt, BOND_IDX_TERMINAL, 0, 0, BOND_IDX_IDSTRING_RIGHT, 0.25 * u);
			set_four_tensor_data_entry(dt, BOND_IDX_TERMINAL, 1, 1, BOND_IDX_IDSTRING_RIGHT, 0.25 * u - 2 * mu);
		}
		// left tensor sending or receiving a fermion to or from the right, for odd input parity
		{
			const ct_long c = 1;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 0 &&  // left virtual bond
				jlists[1] == 0 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 1 &&  // right virtual bond
				jlists[4] == 0);   // inner tree axis
			#endif
			//           j=0
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=0 ─<─0───┤   │
			//        │   ├───3─<─ j=1
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=1

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 1 &&
				dt->dim[1] == 2 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 2);
			// first "hopping" channel (from left to right), input states |01> and |10>, output state |00>
			set_four_tensor_data_entry(dt, BOND_IDX_TERMINAL, 0, 0, BOND_IDX_HOP_LEFT_TO_RIGHT, -sqrt2 * t);
			// second "hopping" channel (from right to left), input states |01> and |10>, output state |11>
			// Jordan-Wigner sign factor accounted for by Clebsch-Gordan coefficient
			set_four_tensor_data_entry(dt, BOND_IDX_TERMINAL, 1, 0, BOND_IDX_HOP_RIGHT_TO_LEFT, -sqrt2 * t);
		}
		// left tensor sending or receiving a fermion to or from the right, for even input parity
		{
			const ct_long c = 2;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 0 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 0 &&  // physical input axis
				jlists[3] == 1 &&  // right virtual bond
				jlists[4] == 1);   // inner tree axis
			#endif
			//           j=1
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=0 ─<─0───┤   │
			//        │   ├───3─<─ j=1
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=0

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 1 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 2 &&
				dt->dim[3] == 2);
			// first "hopping" channel (from left to right), input state |11>, output states |01> and |10>
			// negative sign factor to account for anti-commutation relations
			set_four_tensor_data_entry(dt, BOND_IDX_TERMINAL, 0, 1, BOND_IDX_HOP_LEFT_TO_RIGHT,  t);
			// second "hopping" channel (from right to left), input state |00>, output states |01> and |10>
			set_four_tensor_data_entry(dt, BOND_IDX_TERMINAL, 0, 0, BOND_IDX_HOP_RIGHT_TO_LEFT, -t);
		}
		// degeneracy tensor acting on bond irrep 0, implementing left-connected identity string and
		// parts of the interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn) terms
		{
			const ct_long c = 3;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 0 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 0 &&  // right virtual bond
				jlists[4] == 1);   // inner tree axis
			#endif
			//           j=1
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=0 ─<─0───┤   │
			//        │   ├───3─<─ j=0
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=1

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 1 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 2);
			set_four_tensor_data_entry(dt, BOND_IDX_TERMINAL, 0, 0, BOND_IDX_IDSTRING_LEFT, 1.0);
			// parts of the interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn) terms
			set_four_tensor_data_entry(dt, BOND_IDX_TERMINAL, 0, 0, BOND_IDX_IDSTRING_RIGHT, -0.25 * u - mu);
		}
	}
	for (int i = 1; i < nsites - 1; i++)
	{
		struct su2_tensor* a_loc = &mpo->a[i];

		assert(a_loc->ndim_logical == 4);
		assert(a_loc->charge_sectors.ndim == 5);
		assert(a_loc->charge_sectors.nsec == 9);

		// degeneracy tensor acting on bond irrep 0, implementing identity map and
		// parts of the interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn) terms
		{
			const ct_long c = 0;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 0 &&  // left virtual bond
				jlists[1] == 0 &&  // physical output axis
				jlists[2] == 0 &&  // physical input axis
				jlists[3] == 0 &&  // right virtual bond
				jlists[4] == 0);   // inner tree axis
			#endif
			//           j=0
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=0 ─<─0───┤   │
			//        │   ├───3─<─ j=0
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=0

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 2 &&
				dt->dim[1] == 2 &&
				dt->dim[2] == 2 &&
				dt->dim[3] == 2);
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_LEFT,  0, 0, BOND_IDX_IDSTRING_LEFT,  1.0);
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_LEFT,  1, 1, BOND_IDX_IDSTRING_LEFT,  1.0);
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_RIGHT, 0, 0, BOND_IDX_IDSTRING_RIGHT, 1.0);
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_RIGHT, 1, 1, BOND_IDX_IDSTRING_RIGHT, 1.0);
			// parts of the interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn) terms
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_LEFT, 0, 0, BOND_IDX_IDSTRING_RIGHT, 0.25 * u);
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_LEFT, 1, 1, BOND_IDX_IDSTRING_RIGHT, 0.25 * u - 2 * mu);
		}
		// left tensor sending or receiving a fermion to or from the right, for odd input parity
		{
			const ct_long c = 1;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 0 &&  // left virtual bond
				jlists[1] == 0 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 1 &&  // right virtual bond
				jlists[4] == 0);   // inner tree axis
			#endif
			//           j=0
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=0 ─<─0───┤   │
			//        │   ├───3─<─ j=1
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=1

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 2 &&
				dt->dim[1] == 2 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 2);
			// first "hopping" channel (from left to right), input states |01> and |10>, output state |00>
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_LEFT, 0, 0, BOND_IDX_HOP_LEFT_TO_RIGHT, -sqrt2 * t);
			// second "hopping" channel (from right to left), input states |01> and |10>, output state |11>
			// Jordan-Wigner sign factor accounted for by Clebsch-Gordan coefficient
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_LEFT, 1, 0, BOND_IDX_HOP_RIGHT_TO_LEFT, -sqrt2 * t);
		}
		// left tensor sending or receiving a fermion to or from the right, for even input parity
		{
			const ct_long c = 2;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 0 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 0 &&  // physical input axis
				jlists[3] == 1 &&  // right virtual bond
				jlists[4] == 1);   // inner tree axis
			#endif
			//           j=1
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=0 ─<─0───┤   │
			//        │   ├───3─<─ j=1
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=0

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 2 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 2 &&
				dt->dim[3] == 2);
			// first "hopping" channel (from left to right), input state |11>, output states |01> and |10>
			// negative sign factor to account for anti-commutation relations
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_LEFT, 0, 1, BOND_IDX_HOP_LEFT_TO_RIGHT,  t);
			// second "hopping" channel (from right to left), input state |00>, output states |01> and |10>
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_LEFT, 0, 0, BOND_IDX_HOP_RIGHT_TO_LEFT, -t);
		}
		// degeneracy tensor acting on bond irrep 0, implementing identity map and
		// parts of the interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn) terms
		{
			const ct_long c = 3;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 0 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 0 &&  // right virtual bond
				jlists[4] == 1);   // inner tree axis
			#endif
			//           j=1
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=0 ─<─0───┤   │
			//        │   ├───3─<─ j=0
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=1

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 2 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 2);
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_LEFT,  0, 0, BOND_IDX_IDSTRING_LEFT,  1.0);
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_RIGHT, 0, 0, BOND_IDX_IDSTRING_RIGHT, 1.0);
			// parts of the interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn) terms
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_LEFT, 0, 0, BOND_IDX_IDSTRING_RIGHT, -0.25 * u - mu);
		}
		// degeneracy tensor not required
		{
			const ct_long c = 4;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 1 &&  // left virtual bond
				jlists[1] == 0 &&  // physical output axis
				jlists[2] == 0 &&  // physical input axis
				jlists[3] == 1 &&  // right virtual bond
				jlists[4] == 1);   // inner tree axis
			#endif
			su2_tensor_delete_charge_sector_by_index(a_loc, c);
		}
		// right tensor sending or receiving a fermion to or from the left, for odd input parity
		{
			const ct_long c = 4;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 1 &&  // left virtual bond
				jlists[1] == 0 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 0 &&  // right virtual bond
				jlists[4] == 1);   // inner tree axis
			#endif
			//           j=0
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=1 ─<─0───┤   │
			//        │   ├───3─<─ j=0
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=1

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 2 &&
				dt->dim[1] == 2 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 2);
			// first "hopping" channel (from left to right), input states |01> and |10>, output state |11>
			set_four_tensor_data_entry(dt, BOND_IDX_HOP_LEFT_TO_RIGHT, 1, 0, BOND_IDX_IDSTRING_RIGHT, 1.0);
			// second "hopping" channel (from right to left), input states |01> and |10>, output state |00>
			set_four_tensor_data_entry(dt, BOND_IDX_HOP_RIGHT_TO_LEFT, 0, 0, BOND_IDX_IDSTRING_RIGHT, 1.0);
		}
		// right tensor sending or receiving a fermion to or from the left, for even input parity
		{
			const ct_long c = 5;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 1 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 0 &&  // physical input axis
				jlists[3] == 0 &&  // right virtual bond
				jlists[4] == 0);   // inner tree axis
			#endif
			//           j=1
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=1 ─<─0───┤   │
			//        │   ├───3─<─ j=0
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=0

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 2 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 2 &&
				dt->dim[3] == 2);
			// first "hopping" channel (from left to right), input state |00>, output states |01> and |10>
			// negative sign factor to compensate for sign from Clebsch-Gordan coefficients, receiving an "anti-particle"
			set_four_tensor_data_entry(dt, BOND_IDX_HOP_LEFT_TO_RIGHT, 0, 0, BOND_IDX_IDSTRING_RIGHT, -sqrt2);
			// second "hopping" channel (from right to left), input state |11>, output states |01> and |10>
			// Jordan-Wigner sign factor accounted for by Clebsch-Gordan coefficient
			set_four_tensor_data_entry(dt, BOND_IDX_HOP_RIGHT_TO_LEFT, 0, 1, BOND_IDX_IDSTRING_RIGHT, sqrt2);
		}
		// degeneracy tensor not required
		{
			const ct_long c = 6;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 1 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 1 &&  // right virtual bond
				jlists[4] == 0);   // inner tree axis
			#endif
			su2_tensor_delete_charge_sector_by_index(a_loc, c);
		}
		// degeneracy tensor not required
		{
			const ct_long c = 6;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 1 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 1 &&  // right virtual bond
				jlists[4] == 2);   // inner tree axis
			#endif
			su2_tensor_delete_charge_sector_by_index(a_loc, c);
		}

		assert(a_loc->charge_sectors.nsec == 6);
	}
	// i = nsites - 1
	{
		struct su2_tensor* a_loc = &mpo->a[nsites - 1];

		assert(a_loc->ndim_logical == 4);
		assert(a_loc->charge_sectors.ndim == 5);
		assert(a_loc->charge_sectors.nsec == 4);

		// degeneracy tensor acting on bond irrep 0, implementing right-connected identity string and
		// parts of the interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn) terms
		{
			const ct_long c = 0;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 0 &&  // left virtual bond
				jlists[1] == 0 &&  // physical output axis
				jlists[2] == 0 &&  // physical input axis
				jlists[3] == 0 &&  // right virtual bond
				jlists[4] == 0);   // inner tree axis
			#endif
			//           j=0
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=0 ─<─0───┤   │
			//        │   ├───3─<─ j=0
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=0

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 2 &&
				dt->dim[1] == 2 &&
				dt->dim[2] == 2 &&
				dt->dim[3] == 1);
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_RIGHT, 0, 0, BOND_IDX_TERMINAL, 1.0);
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_RIGHT, 1, 1, BOND_IDX_TERMINAL, 1.0);
			// parts of the interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn) terms
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_LEFT, 0, 0, BOND_IDX_TERMINAL, 0.25 * u);
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_LEFT, 1, 1, BOND_IDX_TERMINAL, 0.25 * u - 2 * mu);
		}
		// degeneracy tensor acting on bond irrep 0, implementing right-connected identity string and
		// parts of the interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn) terms
		{
			const ct_long c = 1;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 0 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 0 &&  // right virtual bond
				jlists[4] == 1);   // inner tree axis
			#endif
			//           j=1
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=0 ─<─0───┤   │
			//        │   ├───3─<─ j=0
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=1

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 2 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 1);
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_RIGHT, 0, 0, BOND_IDX_TERMINAL, 1.0);
			// parts of the interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn) terms
			set_four_tensor_data_entry(dt, BOND_IDX_IDSTRING_LEFT, 0, 0, BOND_IDX_TERMINAL, -0.25 * u - mu);
		}
		// right tensor sending or receiving a fermion to or from the left, for odd input parity
		{
			const ct_long c = 2;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 1 &&  // left virtual bond
				jlists[1] == 0 &&  // physical output axis
				jlists[2] == 1 &&  // physical input axis
				jlists[3] == 0 &&  // right virtual bond
				jlists[4] == 1);   // inner tree axis
			#endif
			//           j=0
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=1 ─<─0───┤   │
			//        │   ├───3─<─ j=0
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=1

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 2 &&
				dt->dim[1] == 2 &&
				dt->dim[2] == 1 &&
				dt->dim[3] == 1);
			// first "hopping" channel (from left to right), input states |01> and |10>, output state |11>
			set_four_tensor_data_entry(dt, BOND_IDX_HOP_LEFT_TO_RIGHT, 1, 0, BOND_IDX_TERMINAL, 1.0);
			// second "hopping" channel (from right to left), input states |01> and |10>, output state |00>
			set_four_tensor_data_entry(dt, BOND_IDX_HOP_RIGHT_TO_LEFT, 0, 0, BOND_IDX_TERMINAL, 1.0);
		}
		// right tensor sending or receiving a fermion to or from the left, for even input parity
		{
			const ct_long c = 3;

			#ifndef NDEBUG
			const qnumber* jlists = &a_loc->charge_sectors.jlists[c * a_loc->charge_sectors.ndim];
			assert(
				jlists[0] == 1 &&  // left virtual bond
				jlists[1] == 1 &&  // physical output axis
				jlists[2] == 0 &&  // physical input axis
				jlists[3] == 0 &&  // right virtual bond
				jlists[4] == 0);   // inner tree axis
			#endif
			//           j=1
			//            ^
			//            │
			//        ╭───1───╮
			//        │   │   │
			// j=1 ─<─0───┤   │
			//        │   ├───3─<─ j=0
			//        │   │   │
			//        ╰───2───╯
			//            │
			//            ^
			//           j=0

			struct dense_tensor* dt = a_loc->degensors[c];
			assert(dt->ndim == 4);
			assert(
				dt->dim[0] == 2 &&
				dt->dim[1] == 1 &&
				dt->dim[2] == 2 &&
				dt->dim[3] == 1);
			// first "hopping" channel (from left to right), input state |00>, output states |01> and |10>
			// negative sign factor to compensate for sign from Clebsch-Gordan coefficients, receiving an "anti-particle"
			set_four_tensor_data_entry(dt, BOND_IDX_HOP_LEFT_TO_RIGHT, 0, 0, BOND_IDX_TERMINAL, -sqrt2);
			// second "hopping" channel (from right to left), input state |11>, output states |01> and |10>
			// Jordan-Wigner sign factor accounted for by Clebsch-Gordan coefficient
			set_four_tensor_data_entry(dt, BOND_IDX_HOP_RIGHT_TO_LEFT, 0, 1, BOND_IDX_TERMINAL, sqrt2);
		}
	}
}
