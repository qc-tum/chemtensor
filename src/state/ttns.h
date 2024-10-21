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
/// The tensor at site 0 contains an additional (dummy) auxiliary second axis for reaching non-zero quantum number sectors.
///
struct ttns
{
	struct block_sparse_tensor* a;   //!< tensors associated with sites, with interleaved physical and virtual bond dimensions (ordered by site indices)
	struct abstract_graph topology;  //!< logical tree topology; nodes correspond to physical and branching sites
	qnumber* qsite;                  //!< physical quantum numbers at each physical site
	long d;                          //!< local physical dimension of each physical site
	int nsites_physical;             //!< number of physical sites
	int nsites_branching;            //!< number of branching sites
};


//________________________________________________________________________________________________________________________
//

// allocation and construction

void allocate_ttns(const enum numeric_type dtype, const int nsites_physical, const struct abstract_graph* topology, const long d, const qnumber* qsite, const qnumber qnum_sector, const long* dim_bonds, const qnumber** qbonds, struct ttns* ttns);

void delete_ttns(struct ttns* ttns);

void construct_random_ttns(const enum numeric_type dtype, const int nsites_physical, const struct abstract_graph* topology, const long d, const qnumber* qsite, const qnumber qnum_sector, const long max_vdim, struct rng_state* rng_state, struct ttns* ttns);

bool ttns_is_consistent(const struct ttns* ttns);


//________________________________________________________________________________________________________________________
///
/// \brief Quantum number sector of a TTNS.
///
static inline qnumber ttns_quantum_number_sector(const struct ttns* ttns)
{
	assert(ttns->nsites_physical >= 1);
	assert(ttns->a[0].ndim >= 2);
	assert(ttns->a[0].dim_logical[0] == ttns->d && ttns->a[0].dim_logical[1] == 1);

	return ttns->a[0].qnums_logical[1][0];
}


//________________________________________________________________________________________________________________________
//

// inner product and norm

void ttns_vdot(const struct ttns* chi, const struct ttns* psi, void* ret);

double ttns_norm(const struct ttns* psi);


//________________________________________________________________________________________________________________________
///
/// \brief TTNS tensor axis type.
///
enum ttns_tensor_axis_type
{
	TTNS_TENSOR_AXIS_PHYSICAL  = 0,  //!< physical axis
	TTNS_TENSOR_AXIS_AUXILIARY = 1,  //!< auxiliary (dummy) axis, for reaching non-zero quantum number sectors
	TTNS_TENSOR_AXIS_VIRTUAL   = 2,  //!< virtual bond axis
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

void ttns_tensor_get_axis_desc(const struct ttns* ttns, const int i_site, struct ttns_tensor_axis_desc* desc);


//________________________________________________________________________________________________________________________
//

// conversion to a statevector (intended for testing)

void ttns_to_statevector(const struct ttns* ttns, struct block_sparse_tensor* vec);
