/// \file su2_hamiltonian.c
/// \brief Construction of common SU(2) symmetric quantum Hamiltonians.

#include <assert.h>
#include "su2_hamiltonian.h"
#include "aligned_memory.h"


#define ARRLEN(a) (sizeof(a) / sizeof(a[0]))


//________________________________________________________________________________________________________________________
///
/// \brief Single double entry of a degree-four tensor (auxiliary data structure).
///
struct four_tensor_data_entry
{
	ct_long i[4];  //!< indices
	double val;    //!< value
};


//________________________________________________________________________________________________________________________
///
/// \brief Set a single double entry of a degree-four tensor (helper function).
///
static inline void set_four_tensor_data_entry(const struct four_tensor_data_entry* entry, struct dense_tensor* t)
{
	assert(t->ndim == 4);
	assert(0 <= entry->i[0] && entry->i[0] < t->dim[0]);
	assert(0 <= entry->i[1] && entry->i[1] < t->dim[1]);
	assert(0 <= entry->i[2] && entry->i[2] < t->dim[2]);
	assert(0 <= entry->i[3] && entry->i[3] < t->dim[3]);
	assert(t->dtype == CT_DOUBLE_REAL);

	double* data = t->data;
	data[((entry->i[0]*t->dim[1] + entry->i[1])*t->dim[2] + entry->i[2])*t->dim[3] + entry->i[3]] = entry->val;
}


//________________________________________________________________________________________________________________________
///
/// \brief Collection of SU(2) degeneracy tensor entries within a specified charge sector of a degree-four SU(2) tensor.
///
struct su2_four_tensor_sector_data
{
	qnumber jlist[5];                        //!< charge sector specified by 'j' quantum numbers times 2
	struct four_tensor_data_entry* entries;  //!< degeneracy tensor entries
	int nentries;                            //!< number of entries
};


//________________________________________________________________________________________________________________________
///
/// \brief Set selected degeneracy tensor entries of an (already allocated) SU(2) tensor of logical degree four.
/// The charge sectors not appearing in the input list are deleted from the tensor.
///
static void su2_four_tensor_set_entries(const struct su2_four_tensor_sector_data* data, const int num, struct su2_tensor* t)
{
	assert(t->dtype == CT_DOUBLE_REAL);
	assert(t->ndim_logical == 4);
	assert(t->charge_sectors.ndim == 5);
	assert(num > 0);

	// indicator of retained sectors
	bool* retained = ct_calloc(t->charge_sectors.nsec, sizeof(bool));

	for (int i = 0; i < num; i++)
	{
		const ct_long c = charge_sector_index(&t->charge_sectors, data[i].jlist);
		assert(c != -1);

		struct dense_tensor* dt = t->degensors[c];
		assert(dt != NULL);
		for (int j = 0; j < data[i].nentries; j++)
		{
			set_four_tensor_data_entry(&data[i].entries[j], dt);
		}

		// mark as retained
		retained[c] = true;
	}

	// delete the non-retained charge sectors
	ct_long* idx_delete = ct_malloc(t->charge_sectors.nsec * sizeof(ct_long));  // upper bound on required memory
	ct_long num_delete = 0;
	for (ct_long c = 0; c < t->charge_sectors.nsec; c++)
	{
		if (!retained[c])
		{
			idx_delete[num_delete] = c;
			num_delete++;
		}
	}
	// at least one sector must have been filled with entries
	assert(num_delete < t->charge_sectors.nsec);
	su2_tensor_delete_charge_sectors_by_indices(t, idx_delete, num_delete);
	ct_free(idx_delete);

	ct_free(retained);
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

	// virtual bond channels
	//            constant        channel index    j_bond  description
	const ct_long BOND_IDX_TERMINAL       = 0;  //    0    left and right terminal bonds
	const ct_long BOND_IDX_IDSTRING_LEFT  = 0;  //    0    left-connected identity string
	const ct_long BOND_IDX_IDSTRING_RIGHT = 1;  //    0    right-connected identity string
	const ct_long BOND_IDX_INTERACTION    = 0;  //    2    virtual bond for 1/2 (S_up ⊗ S_down + S_down ⊗ S_up) + S_z ⊗ S_z term

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
	const ct_long bond_dim_degen_bulk[]     = { 2, 0, 1 };  // left and right-connected identity strings for j = 0
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

	// set degeneracy tensor acting on bond irrep 0 to identity map,
	// implementing the left- and right-connected identity strings
	//
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
	//
	struct four_tensor_data_entry entries_01101_bulk[] = {
		{ .i = { BOND_IDX_IDSTRING_LEFT,  0, 0, BOND_IDX_IDSTRING_LEFT  }, .val = 1.0 },
		{ .i = { BOND_IDX_IDSTRING_RIGHT, 0, 0, BOND_IDX_IDSTRING_RIGHT }, .val = 1.0 },
	};
	struct four_tensor_data_entry entries_01101_boundary_left[] = {
		entries_01101_bulk[0],
	};
	struct four_tensor_data_entry entries_01101_boundary_right[] = {
		entries_01101_bulk[1],
	};
	entries_01101_boundary_left [0].i[0] = BOND_IDX_TERMINAL;
	entries_01101_boundary_right[0].i[3] = BOND_IDX_TERMINAL;

	// transition from left bond irrep 0 to right bond irrep 2, implementing the left part of S dot S
	//
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
	//
	struct four_tensor_data_entry entries_01121_bulk[] = {
		{ .i = { BOND_IDX_IDSTRING_LEFT, 0, 0, BOND_IDX_INTERACTION }, .val = -0.75 * J },
	};
	struct four_tensor_data_entry entries_01121_boundary_left[] = {
		entries_01121_bulk[0],
	};
	entries_01121_boundary_left[0].i[0] = BOND_IDX_TERMINAL;

	// transition from left bond irrep 2 to right bond irrep 0, implementing the right part of S dot S
	//
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
	//
	struct four_tensor_data_entry entries_21101_bulk[] = {
		{ .i = { BOND_IDX_INTERACTION, 0, 0, BOND_IDX_IDSTRING_RIGHT }, .val = 1.0 },
	};
	struct four_tensor_data_entry entries_21101_boundary_right[] = {
		entries_21101_bulk[0],
	};
	entries_21101_boundary_right[0].i[3] = BOND_IDX_TERMINAL;

	struct su2_four_tensor_sector_data data_bulk[] = {
		{ .jlist = { 0, 1, 1, 0, 1 }, .entries = entries_01101_bulk, .nentries = ARRLEN(entries_01101_bulk) },
		{ .jlist = { 0, 1, 1, 2, 1 }, .entries = entries_01121_bulk, .nentries = ARRLEN(entries_01121_bulk) },
		{ .jlist = { 2, 1, 1, 0, 1 }, .entries = entries_21101_bulk, .nentries = ARRLEN(entries_21101_bulk) },
	};
	struct su2_four_tensor_sector_data data_boundary_left[] = {
		{ .jlist = { 0, 1, 1, 0, 1 }, .entries = entries_01101_boundary_left, .nentries = ARRLEN(entries_01101_boundary_left) },
		{ .jlist = { 0, 1, 1, 2, 1 }, .entries = entries_01121_boundary_left, .nentries = ARRLEN(entries_01121_boundary_left) },
	};
	struct su2_four_tensor_sector_data data_boundary_right[] = {
		{ .jlist = { 0, 1, 1, 0, 1 }, .entries = entries_01101_boundary_right, .nentries = ARRLEN(entries_01101_boundary_right) },
		{ .jlist = { 2, 1, 1, 0, 1 }, .entries = entries_21101_boundary_right, .nentries = ARRLEN(entries_21101_boundary_right) },
	};

	// set entries of MPO tensors
	// i = 0
	{
		assert(mpo->a[0].ndim_logical == 4);
		assert(mpo->a[0].charge_sectors.ndim == 5);
		assert(mpo->a[0].charge_sectors.nsec == 2);

		su2_four_tensor_set_entries(data_boundary_left, ARRLEN(data_boundary_left), &mpo->a[0]);

		assert(mpo->a[0].charge_sectors.nsec == 2);
	}
	for (int i = 1; i < nsites - 1; i++)
	{
		assert(mpo->a[i].ndim_logical == 4);
		assert(mpo->a[i].charge_sectors.ndim == 5);
		assert(mpo->a[i].charge_sectors.nsec == 5);

		su2_four_tensor_set_entries(data_bulk, ARRLEN(data_bulk), &mpo->a[i]);

		assert(mpo->a[i].charge_sectors.nsec == 3);
	}
	// i = nsites - 1
	{
		assert(mpo->a[nsites - 1].ndim_logical == 4);
		assert(mpo->a[nsites - 1].charge_sectors.ndim == 5);
		assert(mpo->a[nsites - 1].charge_sectors.nsec == 2);

		su2_four_tensor_set_entries(data_boundary_right, ARRLEN(data_boundary_right), &mpo->a[nsites - 1]);

		assert(mpo->a[nsites - 1].charge_sectors.nsec == 2);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Construct the SU(2) symmetric MPO representation of the Fermi-Hubbard Hamiltonian with nearest-neighbor hopping on a one-dimensional lattice.
///
void construct_fermi_hubbard_1d_su2_mpo(const int nsites, const double t, const double u, const double mu, struct su2_mpo* mpo)
{
	// ordering of physical basis states per site:
	//   j   degeneracy   states
	//   0       2        |00>, |11>
	//   1       1        |01>, |10>
	//
	// 'j' quantum numbers are multiplied by 2

	// SU(2) tree structure relevant for kinetic hopping terms,
	// with the diagonal segment the inner virtual bond,
	// and the outer (trivial) virtual bonds not shown,
	// requiring separate channels for left and right hopping:
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

	// degeneracy tensor acting on site quantum number 0 (|00> and |11> states) and bond irrep 0,
	// implementing the left- and right-connected identity strings and
	// parts of the interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn) terms
	//
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
	//
	struct four_tensor_data_entry entries_00000_bulk[] = {
		{ .i = { BOND_IDX_IDSTRING_LEFT,  0, 0, BOND_IDX_IDSTRING_LEFT  }, .val = 1.0 },
		{ .i = { BOND_IDX_IDSTRING_LEFT,  1, 1, BOND_IDX_IDSTRING_LEFT  }, .val = 1.0 },
		{ .i = { BOND_IDX_IDSTRING_RIGHT, 0, 0, BOND_IDX_IDSTRING_RIGHT }, .val = 1.0 },
		{ .i = { BOND_IDX_IDSTRING_RIGHT, 1, 1, BOND_IDX_IDSTRING_RIGHT }, .val = 1.0 },
		// parts of the interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn) terms
		{ .i = { BOND_IDX_IDSTRING_LEFT,  0, 0, BOND_IDX_IDSTRING_RIGHT }, .val = 0.25 * u },
		{ .i = { BOND_IDX_IDSTRING_LEFT,  1, 1, BOND_IDX_IDSTRING_RIGHT }, .val = 0.25 * u - 2 * mu },
	};
	struct four_tensor_data_entry entries_00000_boundary_left[] = {
		entries_00000_bulk[0],
		entries_00000_bulk[1],
		entries_00000_bulk[4],
		entries_00000_bulk[5],
	};
	struct four_tensor_data_entry entries_00000_boundary_right[] = {
		entries_00000_bulk[2],
		entries_00000_bulk[3],
		entries_00000_bulk[4],
		entries_00000_bulk[5],
	};
	for (int j = 0; j < 4; j++)
	{
		entries_00000_boundary_left [j].i[0] = BOND_IDX_TERMINAL;
		entries_00000_boundary_right[j].i[3] = BOND_IDX_TERMINAL;
	}

	// degeneracy tensor acting on site quantum number 1 (|01> and |10> states) and bond irrep 0,
	// implementing the left- and right-connected identity strings and
	// parts of the interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn) terms
	//
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
	//
	struct four_tensor_data_entry entries_01101_bulk[] = {
		{ .i = { BOND_IDX_IDSTRING_LEFT,  0, 0, BOND_IDX_IDSTRING_LEFT  }, .val = 1.0 },
		{ .i = { BOND_IDX_IDSTRING_RIGHT, 0, 0, BOND_IDX_IDSTRING_RIGHT }, .val = 1.0 },
		// parts of the interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn) terms
		{ .i = { BOND_IDX_IDSTRING_LEFT,  0, 0, BOND_IDX_IDSTRING_RIGHT }, .val = -0.25 * u - mu },
	};
	struct four_tensor_data_entry entries_01101_boundary_left[] = {
		entries_01101_bulk[0],
		entries_01101_bulk[2],
	};
	struct four_tensor_data_entry entries_01101_boundary_right[] = {
		entries_01101_bulk[1],
		entries_01101_bulk[2],
	};
	for (int j = 0; j < 2; j++)
	{
		entries_01101_boundary_left [j].i[0] = BOND_IDX_TERMINAL;
		entries_01101_boundary_right[j].i[3] = BOND_IDX_TERMINAL;
	}

	// left tensor sending or receiving a fermion to or from the right, for odd input parity
	//
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
	//
	struct four_tensor_data_entry entries_00110_bulk[] = {
		// first "hopping" channel (from left to right), input states |01> and |10>, output state |00>
		{ .i = { BOND_IDX_IDSTRING_LEFT, 0, 0, BOND_IDX_HOP_LEFT_TO_RIGHT }, .val = -sqrt2 * t },
		// second "hopping" channel (from right to left), input states |01> and |10>, output state |11>
		// Jordan-Wigner sign factor accounted for by Clebsch-Gordan coefficient
		{ .i = { BOND_IDX_IDSTRING_LEFT, 1, 0, BOND_IDX_HOP_RIGHT_TO_LEFT }, .val = -sqrt2 * t },
	};
	struct four_tensor_data_entry entries_00110_boundary_left[] = {
		entries_00110_bulk[0],
		entries_00110_bulk[1],
	};
	for (int j = 0; j < 2; j++) {
		entries_00110_boundary_left[j].i[0] = BOND_IDX_TERMINAL;
	}

	// left tensor sending or receiving a fermion to or from the right, for even input parity
	//
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
	//
	struct four_tensor_data_entry entries_01011_bulk[] = {
		// first "hopping" channel (from left to right), input state |11>, output states |01> and |10>
		// negative sign factor to account for anti-commutation relations
		{ .i = { BOND_IDX_IDSTRING_LEFT, 0, 1, BOND_IDX_HOP_LEFT_TO_RIGHT }, .val =  t },
		// second "hopping" channel (from right to left), input state |00>, output states |01> and |10>
		{ .i = { BOND_IDX_IDSTRING_LEFT, 0, 0, BOND_IDX_HOP_RIGHT_TO_LEFT }, .val = -t },
	};
	struct four_tensor_data_entry entries_01011_boundary_left[] = {
		entries_01011_bulk[0],
		entries_01011_bulk[1],
	};
	for (int j = 0; j < 2; j++) {
		entries_01011_boundary_left[j].i[0] = BOND_IDX_TERMINAL;
	}

	// right tensor sending or receiving a fermion to or from the left, for odd input parity
	//
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
	//
	struct four_tensor_data_entry entries_10101_bulk[] = {
		// first "hopping" channel (from left to right), input states |01> and |10>, output state |11>
		{ .i = { BOND_IDX_HOP_LEFT_TO_RIGHT, 1, 0, BOND_IDX_IDSTRING_RIGHT }, .val = 1.0 },
		// second "hopping" channel (from right to left), input states |01> and |10>, output state |00>
		{ .i = { BOND_IDX_HOP_RIGHT_TO_LEFT, 0, 0, BOND_IDX_IDSTRING_RIGHT }, .val = 1.0 },
	};
	struct four_tensor_data_entry entries_10101_boundary_right[] = {
		entries_10101_bulk[0],
		entries_10101_bulk[1],
	};
	for (int j = 0; j < 2; j++) {
		entries_10101_boundary_right[j].i[3] = BOND_IDX_TERMINAL;
	}

	// right tensor sending or receiving a fermion to or from the left, for even input parity
	//
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
	//
	struct four_tensor_data_entry entries_11000_bulk[] = {
		// first "hopping" channel (from left to right), input state |00>, output states |01> and |10>
		// negative sign factor to compensate for sign from Clebsch-Gordan coefficients, receiving an "anti-particle"
		{ .i = { BOND_IDX_HOP_LEFT_TO_RIGHT, 0, 0, BOND_IDX_IDSTRING_RIGHT }, .val = -sqrt2 },
		// second "hopping" channel (from right to left), input state |11>, output states |01> and |10>
		// Jordan-Wigner sign factor accounted for by Clebsch-Gordan coefficient
		{ .i = { BOND_IDX_HOP_RIGHT_TO_LEFT, 0, 1, BOND_IDX_IDSTRING_RIGHT }, .val =  sqrt2 },
	};
	struct four_tensor_data_entry entries_11000_boundary_right[] = {
		entries_11000_bulk[0],
		entries_11000_bulk[1],
	};
	for (int j = 0; j < 2; j++) {
		entries_11000_boundary_right[j].i[3] = BOND_IDX_TERMINAL;
	}

	struct su2_four_tensor_sector_data data_bulk[] = {
		{ .jlist = { 0, 0, 0, 0, 0 }, .entries = entries_00000_bulk, .nentries = ARRLEN(entries_00000_bulk) },
		{ .jlist = { 0, 1, 1, 0, 1 }, .entries = entries_01101_bulk, .nentries = ARRLEN(entries_01101_bulk) },
		{ .jlist = { 0, 0, 1, 1, 0 }, .entries = entries_00110_bulk, .nentries = ARRLEN(entries_00110_bulk) },
		{ .jlist = { 0, 1, 0, 1, 1 }, .entries = entries_01011_bulk, .nentries = ARRLEN(entries_01011_bulk) },
		{ .jlist = { 1, 0, 1, 0, 1 }, .entries = entries_10101_bulk, .nentries = ARRLEN(entries_10101_bulk) },
		{ .jlist = { 1, 1, 0, 0, 0 }, .entries = entries_11000_bulk, .nentries = ARRLEN(entries_11000_bulk) },
	};
	struct su2_four_tensor_sector_data data_boundary_left[] = {
		{ .jlist = { 0, 0, 0, 0, 0 }, .entries = entries_00000_boundary_left, .nentries = ARRLEN(entries_00000_boundary_left) },
		{ .jlist = { 0, 1, 1, 0, 1 }, .entries = entries_01101_boundary_left, .nentries = ARRLEN(entries_01101_boundary_left) },
		{ .jlist = { 0, 0, 1, 1, 0 }, .entries = entries_00110_boundary_left, .nentries = ARRLEN(entries_00110_boundary_left) },
		{ .jlist = { 0, 1, 0, 1, 1 }, .entries = entries_01011_boundary_left, .nentries = ARRLEN(entries_01011_boundary_left) },
	};
	struct su2_four_tensor_sector_data data_boundary_right[] = {
		{ .jlist = { 0, 0, 0, 0, 0 }, .entries = entries_00000_boundary_right, .nentries = ARRLEN(entries_00000_boundary_right) },
		{ .jlist = { 0, 1, 1, 0, 1 }, .entries = entries_01101_boundary_right, .nentries = ARRLEN(entries_01101_boundary_right) },
		{ .jlist = { 1, 0, 1, 0, 1 }, .entries = entries_10101_boundary_right, .nentries = ARRLEN(entries_10101_boundary_right) },
		{ .jlist = { 1, 1, 0, 0, 0 }, .entries = entries_11000_boundary_right, .nentries = ARRLEN(entries_11000_boundary_right) },
	};

	// set entries of MPO tensors
	// i = 0
	{
		assert(mpo->a[0].ndim_logical == 4);
		assert(mpo->a[0].charge_sectors.ndim == 5);
		assert(mpo->a[0].charge_sectors.nsec == 4);

		su2_four_tensor_set_entries(data_boundary_left, ARRLEN(data_boundary_left), &mpo->a[0]);

		assert(mpo->a[0].charge_sectors.nsec == 4);
	}
	for (int i = 1; i < nsites - 1; i++)
	{
		assert(mpo->a[i].ndim_logical == 4);
		assert(mpo->a[i].charge_sectors.ndim == 5);
		assert(mpo->a[i].charge_sectors.nsec == 9);

		su2_four_tensor_set_entries(data_bulk, ARRLEN(data_bulk), &mpo->a[i]);

		assert(mpo->a[i].charge_sectors.nsec == 6);
	}
	// i = nsites - 1
	{
		assert(mpo->a[nsites - 1].ndim_logical == 4);
		assert(mpo->a[nsites - 1].charge_sectors.ndim == 5);
		assert(mpo->a[nsites - 1].charge_sectors.nsec == 4);

		su2_four_tensor_set_entries(data_boundary_right, ARRLEN(data_boundary_right), &mpo->a[nsites - 1]);

		assert(mpo->a[nsites - 1].charge_sectors.nsec == 4);
	}
}
