/// \file ttno.h
/// \brief Tree tensor network operator (TTNO) data structure and functions.

#pragma once

#include "block_sparse_tensor.h"
#include "ttno_graph.h"


//________________________________________________________________________________________________________________________
///
/// \brief Tree tensor network operator (TTNO) assembly: structural description which can be used to construct a TTNO.
///
struct ttno_assembly
{
	struct ttno_graph graph;         //!< TTNO graph data structure
	struct dense_tensor* opmap;      //!< local operator map (look-up table)
	void* coeffmap;                  //!< coefficient map (look-up table)
	qnumber* qsite;                  //!< physical quantum numbers at each site
	long d;                          //!< local physical dimension of each site
	enum numeric_type dtype;         //!< data type of local operators and coefficients
	int num_local_ops;               //!< number of local operators (length of 'opmap' array)
	int num_coeffs;                  //!< number of coefficients (length of 'coeffmap' array)
};


void delete_ttno_assembly(struct ttno_assembly* assembly);


//________________________________________________________________________________________________________________________
///
/// \brief Tree tensor network operator (TTNO) data structure.
///
struct ttno
{
	struct block_sparse_tensor* a;   //!< tensors associated with sites, with interleaved physical and virtual bond dimensions (ordered by site indices)
	struct abstract_graph topology;  //!< logical tree topology; nodes correspond to physical and branching sites
	qnumber* qsite;                  //!< physical quantum numbers at each site
	long d;                          //!< local physical dimension of each site
	int nsites_physical;             //!< number of physical sites
	int nsites_branching;            //!< number of branching sites
};


//________________________________________________________________________________________________________________________
//

// allocation and construction

void allocate_ttno(const enum numeric_type dtype, const int nsites_physical, const struct abstract_graph* topology, const long d, const qnumber* qsite, const long* dim_bonds, const qnumber** qbonds, struct ttno* ttno);

void ttno_from_assembly(const struct ttno_assembly* assembly, struct ttno* ttno);

void construct_random_ttno(const enum numeric_type dtype, const int nsites_physical, const struct abstract_graph* topology, const long d, const qnumber* qsite, const long max_vdim, struct rng_state* rng_state, struct ttno* ttno);

void delete_ttno(struct ttno* ttno);

bool ttno_is_consistent(const struct ttno* ttno);


//________________________________________________________________________________________________________________________
///
/// \brief TTNO tensor axis type.
///
enum ttno_tensor_axis_type
{
	TTNO_TENSOR_AXIS_PHYS_OUT = 0,  //!< physical output axis
	TTNO_TENSOR_AXIS_PHYS_IN  = 1,  //!< physical input axis
	TTNO_TENSOR_AXIS_VIRTUAL  = 2,  //!< virtual bond axis
};


//________________________________________________________________________________________________________________________
///
/// \brief TTNO tensor axis description.
///
struct ttno_tensor_axis_desc
{
	enum ttno_tensor_axis_type type;  //!< tensor axis type
	int index;                        //!< local site index (for a physical axis), or neighbor site index (for a virtual bond)
};

void ttno_tensor_get_axis_desc(const struct ttno* ttno, const int i_site, struct ttno_tensor_axis_desc* desc);


//________________________________________________________________________________________________________________________
//

// conversion to a matrix (intended for testing)

void ttno_to_matrix(const struct ttno* ttno, struct block_sparse_tensor* mat);
