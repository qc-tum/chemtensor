/// \file ttns.h
/// \brief Tree tensor network state (TTNS) data structure.

#pragma once

#include <assert.h>
#include "block_sparse_tensor.h"
#include "abstract_graph.h"


//________________________________________________________________________________________________________________________
///
/// \brief Tree tensor network state (TTNS) data structure.
///
/// Sites are enumerated as physical sites first, then branching sites.
/// The tensor at the last site contains an additional (dummy) auxiliary axis for reaching non-zero quantum number sectors.
/// The TTNS supports non-uniform local physical dimensions.
/// The branching sites have a (dummy) physical axis of dimension 1 and quantum number 0.
///
struct ttns
{
	struct block_sparse_tensor* a;   //!< tensors associated with sites, with interleaved physical and virtual bond dimensions (ordered by site indices)
	struct abstract_graph topology;  //!< logical tree topology; nodes correspond to physical and branching sites
	int nsites_physical;             //!< number of physical sites
	int nsites_branching;            //!< number of branching sites
};


//________________________________________________________________________________________________________________________
//

// allocation and construction

void allocate_ttns(const enum numeric_type dtype, const int nsites_physical, const struct abstract_graph* topology, const ct_long* d, const qnumber** qsite, const qnumber qnum_sector, const ct_long* dim_bonds, const qnumber** qbonds, struct ttns* ttns);

void delete_ttns(struct ttns* ttns);

void copy_ttns(const struct ttns* restrict src, struct ttns* restrict dst);

void construct_random_ttns(const enum numeric_type dtype, const int nsites_physical, const struct abstract_graph* topology, const ct_long* d, const qnumber** qsite, const qnumber qnum_sector, const ct_long max_vdim, struct rng_state* rng_state, struct ttns* ttns);

bool ttns_is_consistent(const struct ttns* ttns);


//________________________________________________________________________________________________________________________
//


ct_long ttns_local_dimension(const struct ttns* ttns, const int i_site);

ct_long ttns_maximum_bond_dimension(const struct ttns* ttns);

int ttns_tensor_bond_axis_index(const struct abstract_graph* topology, const int i_site, const int i_neigh);


//________________________________________________________________________________________________________________________
///
/// \brief Quantum number sector of a TTNS.
///
static inline qnumber ttns_quantum_number_sector(const struct ttns* ttns)
{
	const int l = ttns->nsites_physical + ttns->nsites_branching - 1;

	assert(l >= 0);
	assert(ttns->a[l].ndim >= 2);
	assert(ttns->a[l].dim_logical[ttns->a[l].ndim - 1] == 1);

	return ttns->a[l].qnums_logical[ttns->a[l].ndim - 1][0];
}


//________________________________________________________________________________________________________________________
//

// inner product and norm

void ttns_vdot(const struct ttns* chi, const struct ttns* psi, void* ret);

void local_ttns_inner_product(const struct block_sparse_tensor* restrict chi, const struct block_sparse_tensor* restrict psi,
	const struct abstract_graph* topology, const int i_site, const int i_parent, struct block_sparse_tensor* restrict inner_bonds);

double ttns_norm(const struct ttns* psi);


//________________________________________________________________________________________________________________________
///
/// \brief TTNS tensor axis type.
///
enum ttns_tensor_axis_type
{
	TTNS_TENSOR_AXIS_PHYSICAL  = 0,  //!< physical axis
	TTNS_TENSOR_AXIS_VIRTUAL   = 1,  //!< virtual bond axis
	TTNS_TENSOR_AXIS_AUXILIARY = 2,  //!< auxiliary (dummy) axis, for reaching non-zero quantum number sectors
};


//________________________________________________________________________________________________________________________
///
/// \brief TTNS tensor axis description.
///
struct ttns_tensor_axis_desc
{
	enum ttns_tensor_axis_type type;  //!< tensor axis type
	int index;                        //!< local site index (for a physical axis), or neighbor site index (for a virtual bond)
};

void ttns_tensor_get_axis_desc(const struct abstract_graph* topology, const int i_site, struct ttns_tensor_axis_desc* desc);


//________________________________________________________________________________________________________________________
//

// orthonormalization

double ttns_orthonormalize_qr(const int i_root, struct ttns* ttns);


//________________________________________________________________________________________________________________________
//

// compression

int ttns_compress(const int i_root, const double tol, const bool relative_thresh, const ct_long max_vdim, struct ttns* ttns);


//________________________________________________________________________________________________________________________
//

// conversion to a statevector (intended for testing)

void ttns_to_statevector(const struct ttns* ttns, struct block_sparse_tensor* vec);
